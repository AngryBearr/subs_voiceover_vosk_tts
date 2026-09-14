"""Standalone, fail-closed SH08 eight-pair matrix coordinator.

The module intentionally owns only the experiment protocol.  Network transport
is imported lazily by the command line runner; preparation and reporting are
offline operations.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Awaitable, Callable, Mapping, Sequence

from filelock import FileLock

from utils.length_check import check_length
from utils.shortening_plan import PlannerConfig, context_items, plan_subtitles
from utils.shortening_prompts import (
    ARBITER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    build_generation_prompt,
    build_review_prompt,
    generation_schema,
    parse_generation_partial,
    parse_review_partial,
    review_schema,
    PROMPT_VERSION,
)
from utils.shortening_resume import atomic_json, canonical_hash, _read_json
from utils.vosk_text import normalize_vosk_text
from utils.sh08_matrix_models import MODEL_SPECS, MAX_TOKENS, OPENCODE_REQUIRED_VERSION


VERSION = "sh08-matrix-v1"
RANK_SALT = "sh08-matrix-v1:20260914:"
MAX_CASES = 60
PAYLOAD_LIMIT = 120_000
DIRECT_CAP = Decimal("5")
SUBSCRIPTION_CAP_UNITS = 300
TOTAL_CAP_UNITS = 600
PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("luna", "luna"),
    ("luna", "deepseek"),
    ("luna", "sol"),
    ("deepseek", "luna"),
    ("deepseek", "deepseek"),
    ("deepseek", "sol"),
    ("gemini", "sol"),
    ("muse", "sol"),
)

# Keep the historical public name as an alias, while keeping the immutable
# model contract in the pure models module.
DEFAULT_SPECS = MODEL_SPECS

Backend = Callable[[str, str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class MatrixError(ValueError):
    """A matrix input or durable state cannot be trusted."""


class BudgetLimited(MatrixError):
    """A request cannot be reserved without exceeding a hard cap."""


class UnknownRequest(MatrixError):
    """A previously started request has no durable accounting."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_path(output_dir: Path) -> Path:
    return Path(output_dir).resolve().parent / f".{Path(output_dir).name}.run.lock"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return _sha_bytes(path.read_bytes())
    except OSError as exc:
        raise MatrixError(f"cannot hash {path}: {exc}") from exc


def _text(item: Mapping[str, Any]) -> tuple[str, bool]:
    value = item.get("text")
    if isinstance(value, str):
        return value, True
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(value), True
    return "", False


def _safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise MatrixError("source path must be a relative string")
    result = (root / value).resolve()
    if root.resolve() not in result.parents:
        raise MatrixError("source path escapes its root")
    return result


def _load_json(path: Path, label: str) -> Any:
    try:
        return _read_json(path, label)
    except Exception as exc:
        raise MatrixError(str(exc)) from exc


def _model_fingerprint(model_path: Path) -> tuple[str, str, list[dict[str, str]]]:
    """Return the calibration metadata identity without importing Vosk."""
    from utils.vosk_calibration import model_metadata
    metadata = model_metadata(model_path)
    return metadata['fingerprint'], metadata['config_sha256'], deepcopy(metadata['artifacts'])


def _pilot_exclusions(pilot: Mapping[str, Any], corpus_sources: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    raw: set[str] = set()
    parsed: set[str] = set()
    pilot_rows = pilot.get('sources', [])
    for row in pilot_rows:
        matches = [source for source in corpus_sources if source.get('sha256') == row.get('sha256')]
        if len(matches) != 1 or not isinstance(matches[0].get('parsed_sha256'), str):
            raise MatrixError('pilot source does not map uniquely to the corpus')
        raw.add(matches[0]['sha256'])
        parsed.add(matches[0]['parsed_sha256'])
    references = [source for source in corpus_sources if source.get('path') == 'Survivor.S44E11.srt']
    if len(references) != 1:
        raise MatrixError('historical reference source is missing or ambiguous')
    raw.add(references[0]['sha256'])
    parsed.add(references[0]['parsed_sha256'])
    return raw, parsed


def _qualified_context(context: Sequence[Mapping[str, Any]], source_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in context:
        if not isinstance(row, Mapping):
            continue
        cue = row.get("cue_id")
        if not isinstance(cue, str):
            continue
        item = {key: deepcopy(row[key]) for key in ("text", "start", "end") if key in row}
        item["cue_id"] = f"{source_id}:{cue}"
        result.append(item)
    return result


def _code_files() -> list[Path]:
    root = Path(__file__).resolve().parent
    names = (
        "sh08_matrix.py", "sh08_matrix_models.py", "sh08_matrix_transport.py",
        "shortening_plan.py", "shortening_prompts.py", "shortening_resume.py",
        "vosk_text.py", "vosk_calibration.py", "length_check.py", "text_metrics.py",
        "opencode_clean.py", "opencode_transport.py", "deepseek_transport.py",
        "deepseek_limits.py", "llm_types.py",
    )
    return [root / name for name in names]


def _code_hashes(*, require_transport: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _code_files():
        if not path.exists():
            raise MatrixError(f"required code file missing: {path}")
        result[path.name] = _sha_file(path)
    return result


def _profile_identity(profile: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    from utils.vosk_calibration import model_metadata, profile_identity, validate_character_profile
    identity = profile_identity(model_metadata(model_path), 8, 1.5)
    validate_character_profile(dict(profile), expected_identity=identity)
    return identity


def _prepare_manifest_unlocked(
    output_dir: Path,
    *,
    corpus_manifest_path: Path | None = None,
    pilot_manifest_path: Path | None = None,
    profile_path: Path | None = None,
    model_path: Path | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable 60-case matrix contract using local files only."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    root = Path(__file__).resolve().parents[1]
    corpus_path = corpus_manifest_path or root / "output/shortening_corpus_20260910_v2/manifest.json"
    pilot_path = pilot_manifest_path or root / "docs/sh08_pilot_manifest.json"
    profile_file = profile_path or root / "output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json"
    corpus = _load_json(Path(corpus_path), "corpus manifest")
    pilot = _load_json(Path(pilot_path), "pilot manifest")
    profile = _load_json(Path(profile_file), "calibration profile")
    if not isinstance(corpus, dict) or not isinstance(pilot, dict) or not isinstance(profile, dict):
        raise MatrixError("preparation manifests must be objects")
    approval_path = root / 'docs/sh08_live_protocol.json'
    approval = _load_json(approval_path, 'approved matrix protocol')
    consent = approval.get('user_consent') if isinstance(approval, dict) else None
    if not isinstance(consent, dict) or consent.get("no_byok_confirmed") is not True or consent.get("muse_data_approved") is not True:
        raise MatrixError("explicit no-BYOK and Muse data consent is required")
    pilot_sources = pilot.get("sources")
    if not isinstance(pilot_sources, list) or len(pilot_sources) != 3:
        raise MatrixError("pilot must declare exactly three sources")
    pilot_root = Path(str(pilot.get("corpus_root", "")))
    if not pilot_root.is_absolute():
        pilot_root = (Path(pilot_path).resolve().parent / pilot_root).resolve()
    for pilot_source in pilot_sources:
        if not isinstance(pilot_source, Mapping) or not isinstance(pilot_source.get("filename"), str) or not isinstance(pilot_source.get("sha256"), str):
            raise MatrixError("pilot source declaration is incomplete")
        pilot_file = _safe_path(pilot_root, pilot_source["filename"])
        if _sha_file(pilot_file) != pilot_source["sha256"]:
            raise MatrixError(f"pilot source hash mismatch: {pilot_source['filename']}")
        from utils.srt_to_json import parse_srt_file
        pilot_parsed = parse_srt_file(str(pilot_file))
        matches = [source for source in corpus['sources'] if source.get('sha256') == pilot_source['sha256']]
        if len(matches) != 1 or canonical_hash(pilot_parsed) != matches[0].get('parsed_sha256'):
            raise MatrixError(f"pilot parsed source hash mismatch: {pilot_source['filename']}")
    pilot_profile = pilot.get("profile")
    if isinstance(pilot_profile, dict) and isinstance(pilot_profile.get("sha256"), str) and _sha_file(Path(profile_file)) != pilot_profile["sha256"]:
        raise MatrixError("calibration profile differs from pilot manifest")
    actual_root = Path(input_root or corpus.get("input_root", ""))
    if not actual_root.is_absolute():
        actual_root = (Path(corpus_path).resolve().parent / actual_root).resolve()
    model = Path(model_path or root / str(pilot.get("model", {}).get("path", "")))
    identity = _profile_identity(profile, model)
    configured = corpus.get("planner_config")
    if not isinstance(configured, dict) or any(configured.get(k) != v for k, v in {"cps": 13.0, "max_speed": 1.5, "gap_ms": 200, "context_before": 3, "context_after": 3}.items()):
        raise MatrixError("corpus planner configuration differs from the fixed SH08 planner")
    planner = PlannerConfig(cps=13.0, max_speed=1.5, gap_ms=200, context_before=3, context_after=3, length_basis="normalized_tts_text", calibration_profile=profile, calibration_identity=identity)
    sources = corpus.get("sources")
    if not isinstance(sources, list):
        raise MatrixError("corpus sources are missing")
    excluded_raw, excluded_parsed = _pilot_exclusions(pilot, [s for s in sources if isinstance(s, dict)])
    source_snapshots: dict[str, list[dict[str, Any]]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            continue
        source_id = source["source_id"]
        source_raw = source.get("sha256", source.get("raw_sha256"))
        source_parsed = source.get("parsed_sha256", source.get("parsedsha"))
        is_s44e11 = "s44e11" in str(source.get("path", source.get("filename", ""))).casefold()
        if is_s44e11:
            json_name = source.get("json_path")
            raw_name = source.get("path") or source.get("raw_source_path")
            if not isinstance(json_name, str) or not isinstance(raw_name, str):
                raise MatrixError("S44E11 exclusion lacks source copies")
            s44_copy = _load_json(_safe_path(Path(corpus_path).resolve().parent, json_name), f"source copy {source_id}")
            s44_raw = _safe_path(actual_root, raw_name)
            if not isinstance(s44_copy, list) or canonical_hash(s44_copy) != source_parsed or _sha_file(s44_raw) != source_raw:
                raise MatrixError("S44E11 source hash mismatch")
            from utils.srt_to_json import parse_srt_file
            if canonical_hash(parse_srt_file(str(s44_raw))) != source_parsed:
                raise MatrixError("S44E11 parsed source hash mismatch")
            exclusions.append({"source_id": source_id, "reason": "excluded_S44E11"})
            continue
        why: str | None = None
        if source_raw in excluded_raw:
            why = "excluded_by_raw_sha256"
        elif source_parsed in excluded_parsed:
            why = "excluded_by_canonical_sha256"
        if why:
            exclusions.append({"source_id": source_id, "reason": why})
            continue
        json_name = source.get("json_path")
        if not isinstance(json_name, str):
            # source_inventory rows are metadata only and cannot be used.
            exclusions.append({"source_id": source_id, "reason": "missing_source_copy"})
            continue
        source_copy_path = _safe_path(Path(corpus_path).resolve().parent, json_name)
        items = _load_json(source_copy_path, f"source copy {source_id}")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise MatrixError(f"source copy {source_id} is not a list of objects")
        if not isinstance(source_parsed, str) or canonical_hash(items) != source_parsed:
            raise MatrixError(f"canonical source hash mismatch: {source_id}")
        raw_name = source.get("path") or source.get("raw_source_path")
        if not isinstance(raw_name, str):
            raise MatrixError(f"raw source path missing: {source_id}")
        raw_path = _safe_path(actual_root, raw_name)
        raw_sha = _sha_file(raw_path)
        if not isinstance(source_raw, str) or raw_sha != source_raw:
            raise MatrixError(f"raw source hash mismatch: {source_id}")
        # Parsing is deliberately local and is only performed during prepare.
        from utils.srt_to_json import parse_srt_file
        parsed_actual = parse_srt_file(str(raw_path))
        if canonical_hash(parsed_actual) != source_parsed:
            raise MatrixError(f"raw source parses differently from source copy: {source_id}")
        if source.get("split") != "test":
            exclusions.append({"source_id": source_id, "reason": "not_test_split"})
            continue
        source_snapshots[source_id] = deepcopy(items)
        source_meta[source_id] = deepcopy(source)
    if len(source_snapshots) != 7:
        raise MatrixError(f"expected seven test sources after exclusions, got {len(source_snapshots)}")

    pool: list[dict[str, Any]] = []
    pool_counts: Counter[str] = Counter()
    for source_id in sorted(source_snapshots):
        items = source_snapshots[source_id]
        plans = plan_subtitles(items, planner)
        for plan in plans:
            text, valid_text = _text(items[plan.position])
            if plan.status not in {"needs_shortening", "short_timing", "invalid"} or (not text and plan.status != "invalid"):
                continue
            case_id = f"{source_id}:{plan.cue_id}"
            context = _qualified_context(context_items(items, plans, [plan.cue_id], planner), source_id)
            row = {
                "case_id": case_id,
                "source_id": source_id,
                "cue_id": case_id,
                "old_id": plan.cue_id,
                "new_id": case_id,
                "position": plan.position,
                "source_index": items[plan.position].get("index", plan.position),
                "text": text,
                "original": text,
                "start": plan.start,
                "end": plan.end,
                "available_ms": plan.available_ms,
                "max_chars": plan.max_chars,
                "status": plan.status,
                "risk": {"severity": plan.severity, "reason": plan.reason, "short_timing": plan.short_timing_risk, "zero": plan.max_chars == 0},
                "length_basis": plan.length_basis,
                "context": context,
            }
            pool.append(row)
            pool_counts[plan.status] += 1
    pool.sort(key=lambda row: (_sha_bytes((RANK_SALT + row["case_id"]).encode("utf-8")), row["case_id"]))
    if len(pool) < MAX_CASES:
        raise MatrixError(f"eligible pool must contain at least {MAX_CASES} cases")
    selected = pool[:MAX_CASES]
    selected_ids = {row["case_id"] for row in selected}
    for row in pool:
        row["rank_sha256"] = _sha_bytes((RANK_SALT + row["case_id"]).encode("utf-8"))
    selected = [{key: deepcopy(row[key]) for key in ("case_id", "source_id", "cue_id", "old_id", "new_id", "position", "source_index", "text", "original", "start", "end", "available_ms", "max_chars", "status", "risk", "length_basis", "context", "rank_sha256")} for row in selected]
    batches = [{"batch_id": "batch-0", "case_ids": [row["case_id"] for row in selected[:6]]}]
    for number, start in enumerate(range(6, len(selected), 27), 1):
        batches.append({"batch_id": f"batch-{number}", "case_ids": [row["case_id"] for row in selected[start:start + 27]]})
    selection = {"version": VERSION, "cases": selected, "selected_case_ids": [row["case_id"] for row in selected], "pool_count": len(pool), "pool_counts": dict(pool_counts), "not_selected": [{"case_id": row["case_id"], "reason": "rank_after_first_60"} for row in pool[MAX_CASES:]], "exclusions": exclusions, "batches": batches}
    selection_sha = canonical_hash(selection)
    code_hashes = _code_hashes()
    tariffs = {key: {"input_per_million": spec["input_per_million"], "output_per_million": spec["output_per_million"], "source_url": "https://opencode.ai/docs/" if spec["subscription"] else "https://api-docs.deepseek.com/quick_start/pricing" if key == "deepseek" else "https://ai.google.dev/gemini-api/docs/pricing" if key == "gemini" else "https://openrouter.ai/meta/muse-spark-1.3-contributor"} for key, spec in DEFAULT_SPECS.items()}
    body: dict[str, Any] = {
        "schema": "sh08_matrix_manifest.v1",
        "version": VERSION,
        "created_at": _now(),
        "approved_scope": {"cases": MAX_CASES, "repetitions": 2, "pairs": [list(pair) for pair in PAIR_ORDER], "smoke_cases": 6, "smoke_repetition": 1},
        "source": {"corpus_manifest": str(Path(corpus_path).resolve()), "corpus_manifest_sha256": _sha_file(Path(corpus_path)), "input_root": str(actual_root), "sources": {key: {"raw_sha256": source_meta[key].get("sha256"), "parsed_sha256": source_meta[key].get("parsed_sha256"), "metadata": source_meta[key]} for key in sorted(source_meta)}, "source_snapshots_sha256": canonical_hash(source_snapshots)},
        "profile": {"path": str(Path(profile_file).resolve()), "sha256": _sha_file(Path(profile_file)), "identity": identity, "model": {"path": str(model.resolve()), "metadata": _model_fingerprint(model)[2]}},
        "selection": selection,
        "selection_sha256": selection_sha,
        "batches": batches,
        "code_hashes": code_hashes,
        "approval": {"path": str(approval_path), "sha256": _sha_file(approval_path)},
        "runtime": {"python": platform.python_version(), "system": platform.system(), "implementation": sys.implementation.name},
        "prompts": {"version": PROMPT_VERSION, "generator_system": GENERATOR_SYSTEM_PROMPT, "arbiter_system": ARBITER_SYSTEM_PROMPT, "generation_schema": generation_schema(), "review_schema": review_schema()},
        "full_specs": deepcopy(DEFAULT_SPECS),
        "opencode_required_version": OPENCODE_REQUIRED_VERSION,
        "limits": {"direct_api_cap_usd": str(DIRECT_CAP), "subscription_conservative_attempt_units": SUBSCRIPTION_CAP_UNITS, "total_attempt_units": TOTAL_CAP_UNITS, "max_tokens": MAX_TOKENS, "payload_bytes": PAYLOAD_LIMIT, "opencode_subscription_attempt_units": 6, "opencode_retry_policy": "permanent six charged quota units per prompt"},
        "baselines_snapshot_sha256": canonical_hash({"planner": asdict(planner), "sources": source_snapshots}),
        "user_consent": {"no_byok_confirmed": True, "muse_data_approved": True, "placeholder": False, "approved_source": deepcopy(consent.get("approved_source", "user conversation; docs/sh08_live_protocol.json")), "approved_scope": deepcopy(consent.get("approved_scope", "SH08 matrix"))},
        "tariffs": tariffs,
        "status": "prepared",
    }
    body["checksum"] = canonical_hash(body)
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "manifest.json", body)
    atomic_json(output_dir / "source_snapshots.json", source_snapshots)
    atomic_json(output_dir / "selection.json", selection)
    atomic_json(output_dir / "queue.json", {"status": "prepared", "pending": selected[0:MAX_CASES], "batches": batches})
    _save_ledger(output_dir, {"schema": "sh08_matrix_ledger.v1", "manifest_checksum": body["checksum"], "requests": {}, "created_at": _now()})
    return body


def prepare_manifest(
    output_dir: Path,
    *,
    corpus_manifest_path: Path | None = None,
    pilot_manifest_path: Path | None = None,
    profile_path: Path | None = None,
    model_path: Path | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    """Build a fresh manifest while holding the prepare lock."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    lock_path = _lock_path(output_dir)
    with FileLock(str(lock_path), timeout=0):
        return _prepare_manifest_unlocked(
            output_dir,
            corpus_manifest_path=corpus_manifest_path,
            pilot_manifest_path=pilot_manifest_path,
            profile_path=profile_path,
            model_path=model_path,
            input_root=input_root,
        )


def _specs() -> dict[str, dict[str, Any]]:
    return deepcopy(MODEL_SPECS)


def _read_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _load_json(output_dir / "manifest.json", "matrix manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("checksum"), str):
        raise MatrixError("matrix manifest is incomplete")
    copy = dict(manifest)
    checksum = copy.pop("checksum")
    if canonical_hash(copy) != checksum:
        raise MatrixError("matrix manifest checksum mismatch")
    selection = _load_json(output_dir / "selection.json", "matrix selection")
    if canonical_hash(selection) != manifest.get("selection_sha256"):
        raise MatrixError("matrix selection checksum mismatch")
    if selection != manifest.get("selection"):
        raise MatrixError("embedded selection differs from selection export")
    if canonical_hash(_load_json(output_dir / "source_snapshots.json", "source snapshots")) != manifest.get("source", {}).get("source_snapshots_sha256"):
        raise MatrixError("source snapshot checksum mismatch")
    approval = manifest.get('approval')
    if not isinstance(approval, dict) or not isinstance(approval.get('path'), str) or not isinstance(approval.get('sha256'), str):
        raise MatrixError('matrix approval reference is missing')
    if _sha_file(Path(approval['path'])) != approval['sha256']:
        raise MatrixError('matrix approval changed')
    source_info = manifest.get("source")
    if isinstance(source_info, Mapping) and isinstance(source_info.get("corpus_manifest"), str) and isinstance(source_info.get("corpus_manifest_sha256"), str):
        source_manifest_path = Path(source_info["corpus_manifest"])
        if not source_manifest_path.exists() or _sha_file(source_manifest_path) != source_info["corpus_manifest_sha256"]:
            raise MatrixError("current corpus manifest hash mismatch")
        for source_id, source in source_info['sources'].items():
            metadata = source['metadata']
            source_path = _safe_path(Path(source_info['input_root']), metadata['path'])
            if _sha_file(source_path) != source['raw_sha256']:
                raise MatrixError(f'current source hash mismatch: {source_id}')
            snapshot_path = _safe_path(source_manifest_path.parent, metadata['json_path'])
            if canonical_hash(_load_json(snapshot_path, 'current source copy')) != source['parsed_sha256']:
                raise MatrixError(f'current source copy mismatch: {source_id}')
    if manifest.get("full_specs") != MODEL_SPECS:
        raise MatrixError("manifest model specifications differ from immutable specifications")
    if manifest.get("opencode_required_version") != OPENCODE_REQUIRED_VERSION:
        raise MatrixError("manifest OpenCode version differs from immutable protocol")
    scope = manifest.get("approved_scope")
    if not isinstance(scope, Mapping) or scope.get("cases") != 60 or scope.get("repetitions") != 2 or scope.get("pairs") != [list(pair) for pair in PAIR_ORDER] or scope.get("smoke_cases") != 6 or scope.get("smoke_repetition") != 1:
        raise MatrixError("manifest scope does not match fixed SH08 protocol")
    limits = manifest.get("limits")
    if not isinstance(limits, Mapping) or limits.get("max_tokens") != MAX_TOKENS or limits.get("payload_bytes") != PAYLOAD_LIMIT or limits.get("direct_api_cap_usd") != str(DIRECT_CAP) or limits.get("subscription_conservative_attempt_units") != SUBSCRIPTION_CAP_UNITS or limits.get("total_attempt_units") != TOTAL_CAP_UNITS:
        raise MatrixError("manifest limits do not match fixed SH08 protocol")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, Mapping) or prompts.get("version") != PROMPT_VERSION or prompts.get("generator_system") != GENERATOR_SYSTEM_PROMPT or prompts.get("arbiter_system") != ARBITER_SYSTEM_PROMPT or prompts.get("generation_schema") != generation_schema() or prompts.get("review_schema") != review_schema():
        raise MatrixError("manifest prompts do not match fixed SH08 protocol")
    consent = manifest.get("user_consent")
    if not isinstance(consent, Mapping) or consent.get("no_byok_confirmed") is not True or consent.get("muse_data_approved") is not True or consent.get("placeholder") is not False or not consent.get("approved_source") or not consent.get("approved_scope"):
        raise MatrixError("manifest consent is incomplete")
    profile = manifest.get("profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("path"), str) or not isinstance(profile.get("sha256"), str):
        raise MatrixError("profile identity is missing")
    profile_path = Path(profile["path"])
    if not profile_path.exists() or _sha_file(profile_path) != profile["sha256"]:
        raise MatrixError("calibration profile hash mismatch")
    model_info = profile.get("model")
    if isinstance(model_info, Mapping) and isinstance(model_info.get("path"), str) and isinstance(profile.get("identity"), Mapping):
        _profile_identity(_load_json(profile_path, "calibration profile"), Path(model_info["path"]))
    expected_code = manifest.get("code_hashes")
    if not isinstance(expected_code, dict):
        raise MatrixError("manifest code hashes are missing")
    actual = _code_hashes(require_transport=False)
    if set(expected_code) != set(actual) or any(not isinstance(value, str) for value in expected_code.values()):
        raise MatrixError("manifest code hash key set is incomplete")
    for key, value in expected_code.items():
        if actual.get(key) != value:
            raise MatrixError(f"code hash mismatch: {key}")
    if selection.get("version") != VERSION or len(selection.get("cases", [])) != MAX_CASES:
        raise MatrixError("selection is not the frozen 60-case selection")
    if selection.get("selected_case_ids") != [case.get("case_id") for case in selection.get("cases", [])] or any(not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str) or isinstance(case.get("max_chars"), bool) or not isinstance(case.get("max_chars"), int) or case.get("max_chars") < 0 for case in selection.get("cases", [])):
        raise MatrixError("selection case identities or limits are invalid")
    batches = manifest.get("batches")
    if not isinstance(batches, list) or [len(batch.get("case_ids", [])) for batch in batches if isinstance(batch, Mapping)] != [6, 27, 27] or [batch.get("batch_id") for batch in batches if isinstance(batch, Mapping)] != ["batch-0", "batch-1", "batch-2"]:
        raise MatrixError("generation batch boundaries do not match fixed protocol")
    return manifest


def _validate_ledger(ledger: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> None:
    if manifest is not None and isinstance(manifest.get("checksum"), str) and ledger.get("manifest_checksum") != manifest.get("checksum"):
        raise MatrixError("ledger is not bound to this manifest")
    requests = ledger.get("requests")
    if not isinstance(requests, dict):
        raise MatrixError("ledger requests are invalid")
    specs = manifest.get("full_specs", MODEL_SPECS) if isinstance(manifest, Mapping) else MODEL_SPECS
    for key, row in requests.items():
        if not isinstance(key, str) or not isinstance(row, dict):
            raise MatrixError("ledger request is invalid")
        if row.get("status") not in {"pending", "unknown", "complete"}:
            raise MatrixError("ledger request status is invalid")
        descriptor = row.get("descriptor")
        if not isinstance(descriptor, dict) or canonical_hash(descriptor) != key:
            raise MatrixError("ledger descriptor key mismatch")
        model_key = descriptor.get("model_key")
        spec = specs.get(model_key) if isinstance(specs, Mapping) else None
        if not isinstance(spec, Mapping):
            raise MatrixError("ledger model key is invalid")
        if row.get("quota_units") != spec.get("quota_units") or row.get("subscription") is not spec.get("subscription"):
            raise MatrixError("ledger reservation does not match model weights")
        for field in ("reservation_usd", "cost_usd"):
            _decimal(row.get(field), f"ledger {field}")
        if row.get("status") == "complete":
            response = row.get("response")
            if not isinstance(response, dict):
                raise MatrixError("complete ledger row has no response")
            if response.get("model") != spec.get("model") or response.get("provider") != spec.get("provider"):
                raise MatrixError("complete response identity mismatch")
            if response.get("error") is not None or response.get("cost_known") is not True:
                raise MatrixError("complete response has unknown accounting")


def _load_ledger(output_dir: Path, *, fresh: bool = False, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = output_dir / "ledger.json"
    if not path.exists():
        if fresh:
            return {"schema": "sh08_matrix_ledger.v1", "manifest_checksum": manifest.get("checksum") if manifest else None, "requests": {}, "created_at": _now()}
        raise MatrixError("ledger is missing on resume")
    value = _load_json(path, "matrix ledger")
    if not isinstance(value, dict) or not isinstance(value.get("checksum"), str):
        raise MatrixError("matrix ledger envelope is incomplete")
    copy = dict(value)
    checksum = copy.pop("checksum")
    if canonical_hash(copy) != checksum:
        raise MatrixError("matrix ledger checksum mismatch")
    _validate_ledger(copy, manifest)
    return copy


def _save_ledger(output_dir: Path, ledger: dict[str, Any]) -> None:
    body = deepcopy(ledger)
    body["checksum"] = canonical_hash(body)
    atomic_json(output_dir / "ledger.json", body)


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MatrixError(f"{label} is not finite") from exc
    if not result.is_finite() or result < 0:
        raise MatrixError(f"{label} is not finite")
    return result


def _reservation(spec: Mapping[str, Any]) -> Decimal:
    input_rate = _decimal(spec.get("input_per_million", 0), "input tariff")
    output_rate = _decimal(spec.get("output_per_million", 0), "output tariff")
    return ((Decimal(1_048_576) * input_rate / Decimal(1_000_000)) + (Decimal(MAX_TOKENS) * output_rate / Decimal(1_000_000))) * Decimal("1.10")


def _usage_cost(response: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[Decimal, bool]:
    known = response.get("cost_known")
    usage = response.get("usage")
    if not isinstance(known, bool) or not known or not isinstance(usage, Mapping):
        return Decimal(0), False
    inputs = usage.get("input_tokens")
    outputs = usage.get("output_tokens")
    for usage_key in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens', 'cache_read_tokens', 'cache_write_tokens'):
        usage_value = usage.get(usage_key)
        if usage_value is not None and (isinstance(usage_value, bool) or not isinstance(usage_value, int) or usage_value < 0):
            return Decimal(0), False
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs < 0 or isinstance(outputs, bool) or not isinstance(outputs, int) or outputs < 0:
        return Decimal(0), False
    if inputs > int(spec.get("context_limit", 0)) or outputs > int(spec.get("max_tokens", 0)):
        return Decimal(0), False
    reported = response.get("cost_usd")
    if reported is not None:
        try:
            return _decimal(reported, "reported cost"), True
        except MatrixError:
            return Decimal(0), False
    return (Decimal(inputs) * _decimal(spec.get("input_per_million", 0), "input tariff") + Decimal(outputs) * _decimal(spec.get("output_per_million", 0), "output tariff")) / Decimal(1_000_000), True


def _payload_size(system: str, user: str, schema: dict[str, Any]) -> int:
    return len((system + user + json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8"))


class _Admission:
    def __init__(self, output_dir: Path, manifest: Mapping[str, Any], ledger: dict[str, Any], specs: Mapping[str, Mapping[str, Any]], *, report_only: bool = False) -> None:
        self.output_dir, self.manifest, self.ledger, self.specs = output_dir, manifest, ledger, specs
        self.report_only = report_only

    def _totals(self) -> tuple[Decimal, int, int, int]:
        known = Decimal(0)
        pending = Decimal(0)
        sub = total = 0
        for row in self.ledger["requests"].values():
            if not isinstance(row, dict):
                raise MatrixError("ledger request is invalid")
            status = row.get("status")
            if status in {"pending", "unknown"}:
                pending += _decimal(row.get("reservation_usd", 0), "pending reservation")
            if status == "complete":
                known += _decimal(row.get("cost_usd", 0), "known cost")
            if row.get("subscription"):
                units = row.get("quota_units")
                if isinstance(units, int) and units >= 0:
                    sub += units
            units = row.get("quota_units")
            if isinstance(units, int) and units >= 0:
                total += units
        return known, pending, sub, total

    def admit(self, descriptor: dict[str, Any], *, key: str) -> dict[str, Any] | None:
        existing = self.ledger["requests"].get(key)
        if isinstance(existing, dict):
            if existing.get("status") == "complete":
                return existing
            raise UnknownRequest(f"request {key} is not durably complete")
        if any(isinstance(row, dict) and row.get("status") in {"pending", "unknown"} for row in self.ledger["requests"].values()):
            raise UnknownRequest("an earlier request has unknown or pending accounting")
        if self.report_only:
            raise BudgetLimited('offline report reached an unexecuted request')
        model_key = descriptor["model_key"]
        spec = self.specs.get(model_key)
        if not isinstance(spec, Mapping):
            raise MatrixError(f"unknown model key: {model_key}")
        if descriptor.get("payload_bytes", 0) > PAYLOAD_LIMIT:
            raise MatrixError("payload exceeds 120000 UTF-8 bytes")
        if int(spec.get("max_tokens", 0)) != MAX_TOKENS or int(spec.get("context_limit", 0)) < 1_048_576:
            raise MatrixError("transport model limits do not match immutable protocol")
        reserve = _reservation(spec)
        units = int(spec.get("quota_units", 0))
        subscription = bool(spec.get("subscription"))
        known, pending, sub, total = self._totals()
        if known + pending + reserve > DIRECT_CAP:
            raise BudgetLimited("direct API cap reservation would be exceeded")
        if sub + (units if subscription else 0) > SUBSCRIPTION_CAP_UNITS or total + units > TOTAL_CAP_UNITS:
            raise BudgetLimited("quota reservation would be exceeded")
        row = {"key": key, "status": "pending", "descriptor": deepcopy(descriptor), "started_at": _now(), "reservation_usd": str(reserve), "cost_usd": "0", "quota_units": units, "subscription": subscription}
        self.ledger["requests"][key] = row
        _save_ledger(self.output_dir, self.ledger)
        return None

    def finish(self, key: str, response: Mapping[str, Any] | None, error: str | None = None) -> dict[str, Any]:
        row = self.ledger["requests"][key]
        if response is None:
            row.update({"status": "unknown", "error": error or "UnknownRequest", "finished_at": _now()})
            _save_ledger(self.output_dir, self.ledger)
            raise UnknownRequest(row["error"])
        data = {key: deepcopy(response[key]) if key in response else None for key in ("text", "raw", "calls", "usage", "cost_usd", "cost_known", "cost_kind", "error", "physical_calls_observed", "model", "provider", "elapsed_sec")}
        spec = self.specs[row["descriptor"]["model_key"]]
        cost, known = _usage_cost(response, spec)
        reserve = _decimal(row["reservation_usd"], "reservation")
        physical = response.get("physical_calls_observed")
        expected_model = spec.get("model")
        expected_provider = spec.get("provider")
        model_ok = response.get("model") == expected_model
        provider_ok = response.get("provider") == expected_provider
        calls = response.get("calls")
        tools_ok = isinstance(calls, list) and not any(isinstance(call, Mapping) and (call.get("tool_calls", 0) not in (0, None) or call.get("status") in {"error", "failed"}) for call in calls)
        physical_ok = physical == 1 or (physical is None and bool(spec.get("subscription")))
        if response.get("error") is not None or not physical_ok or not known or cost > reserve or not model_ok or not provider_ok or not tools_ok:
            row.update({"status": "unknown", "error": "unknown or unexpected accounting", "response": data, "finished_at": _now()})
            _save_ledger(self.output_dir, self.ledger)
            raise UnknownRequest(row["error"])
        row.update({"status": "complete", "response": data, "text": data["text"], "raw": data["raw"], "calls": data["calls"], "usage": data["usage"], "cost_usd": str(cost), "cost_known": True, "finished_at": _now(), "reservation_usd": "0"})
        _save_ledger(self.output_dir, self.ledger)
        return row


async def _request(
    admission: _Admission,
    backend: Backend,
    *,
    descriptor: dict[str, Any],
    system: str,
    user: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, bool]:
    if not isinstance(system, str) or not isinstance(user, str) or not isinstance(schema, dict) or not isinstance(descriptor.get("model_key"), str):
        raise MatrixError("invalid request payload")
    if descriptor.get("payload_bytes") != _payload_size(system, user, schema):
        raise MatrixError("request payload accounting mismatch")
    key = canonical_hash(descriptor)
    existing = admission.admit(descriptor, key=key)
    if existing is not None:
        response = existing.get("response")
        return (response if isinstance(response, dict) else None), None, True
    try:
        response = await backend(descriptor["model_key"], system, user, schema)
        if not isinstance(response, dict):
            raise TypeError("backend response must be an object")
        finished = admission.finish(key, response)
        return finished.get("response"), None, False
    except asyncio.CancelledError:
        try:
            admission.finish(key, None, "CancelledError")
        except UnknownRequest:
            pass
        raise
    except Exception as exc:
        if isinstance(exc, UnknownRequest):
            raise
        try:
            admission.finish(key, None, str(exc))
        except UnknownRequest as unknown:
            raise unknown from exc
        return None, type(exc).__name__, False


def _case_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selection = manifest["selection"]
    return {row["case_id"]: row for row in selection["cases"]}


def _merge_context(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in entries:
        cue_id = entry.get("cue_id")
        if not isinstance(cue_id, str):
            continue
        if cue_id not in merged:
            merged[cue_id] = deepcopy(dict(entry))
        else:
            merged[cue_id]["is_target"] = bool(merged[cue_id].get("is_target")) or bool(entry.get("is_target"))
    return list(merged.values())


def _request_descriptor(manifest: Mapping[str, Any], *, rep: int, stage: str, generator: str, arbiter: str | None, batch: str, system: str, user: str, schema: dict[str, Any], origin: str | None = None) -> dict[str, Any]:
    model_key = arbiter if stage in {"review", "recheck"} else generator
    return {"rep": rep, "stage": stage, "generator": generator, "arbiter": arbiter, "batch": batch, "system_prompt": system, "user_prompt": user, "schema": schema, "spec": deepcopy(manifest["full_specs"].get(model_key, {})), "candidate_origin": origin, "model_key": model_key, "payload_bytes": _payload_size(system, user, schema)}


def _response_text(response: Mapping[str, Any] | None) -> str | None:
    value = response.get("text") if isinstance(response, Mapping) else None
    return value if isinstance(value, str) else None


def _write_outputs(output_dir: Path, manifest: Mapping[str, Any], ledger: Mapping[str, Any], branches: Mapping[str, dict[str, Any]], *, status: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases = _case_map(manifest)
    for case_id, case in cases.items():
        matching = [row for key, row in branches.items() if row.get("case_id") == case_id]
        rows.append({"case_id": case_id, "source_id": case["source_id"], "original": case["original"], "original_index": case["source_index"], "start": case["start"], "end": case["end"], "context": case["context"], "branches": matching, "audio_status": "not_measured"})
    report = _report(manifest, ledger, rows, status=status)
    case_body = {"schema": VERSION, "manifest_checksum": manifest["checksum"], "ledger_checksum": canonical_hash(ledger), "rows": rows, "provenance": canonical_hash(rows)}
    case_body["checksum"] = canonical_hash(case_body)
    atomic_json(output_dir / "case_results.json", case_body)
    atomic_json(output_dir / "report.json", report)
    _atomic_text(output_dir / "report.md", _markdown_report(report))
    def final_accepted(branch: Mapping[str, Any]) -> bool:
        return branch.get('accepted') is True or branch.get('repair', {}).get('recheck', {}).get('accepted') is True
    pending = [{"rep": branch.get("rep"), "generator": branch.get("generator"), "arbiter": branch.get("arbiter"), "case_id": branch.get("case_id"), "reason": branch.get("reason", branch.get("advice", "unresolved"))} for branch in branches.values() if branch.get("stage") == "review" and (branch.get("status") != "complete" or not final_accepted(branch))]
    atomic_json(output_dir / "queue.json", {"status": status, "pending": pending, "pending_case_ids": sorted({item["case_id"] for item in pending}), "scope": "all 60 frozen cases; smoke processes only six"})
    return report


def _mark_pending(manifest: Mapping[str, Any], branches: dict[str, dict[str, Any]], *, smoke: bool) -> None:
    """Make budget-stopped work visible instead of silently truncating it."""
    repetitions = (1,) if smoke else (1, 2)
    cases = _case_map(manifest)
    for repetition in repetitions:
        for batch in manifest["batches"]:
            for case_id in batch["case_ids"]:
                if case_id not in cases:
                    continue
                for generator in ("luna", "deepseek", "gemini", "muse"):
                    key = f"{repetition}:generation:{generator}:{batch['batch_id']}:{case_id}"
                    branches.setdefault(key, {"case_id": case_id, "rep": repetition, "stage": "generation", "generator": generator, "batch": batch["batch_id"], "status": "pending", "reason": "run_budget_limited"})
                for generator, arbiter in PAIR_ORDER:
                    key = f"{repetition}:review:{generator}:{arbiter}:{batch['batch_id']}:{case_id}"
                    branches.setdefault(key, {"case_id": case_id, "rep": repetition, "stage": "review", "generator": generator, "arbiter": arbiter, "batch": batch["batch_id"], "status": "pending", "reason": "run_budget_limited"})


def _atomic_text(path: Path, text: str) -> None:
    """Atomically export the human-readable report without a JSON wrapper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = ["# SH08 matrix report", "", f"Status: {report.get('status')}", "", "## First pass", "", "| Pair/repetition | Accepted | Declined | Missing | Unresolved |", "|---|---:|---:|---:|---:|"]
    for key, counts in report.get("firstpass_counts", {}).items():
        lines.append(f"| {key} | {counts['accepted']} | {counts['declined']} | {counts['missing']} | {counts['unresolved']} |")
    lines += ["", "## Repair coverage", "", "| Pair/repetition | Eligible | Attempted | Rechecked | Resolved | Budget skipped | Pending |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, counts in report.get("repair_coverage", {}).items():
        lines.append(f"| {key} | {counts['eligible']} | {counts['attempted']} | {counts['rechecked']} | {counts['resolved']} | {counts['skipped_for_budget']} | {counts['pending']} |")
    costs = report.get("costs", {})
    lines += ["", "## Costs", "", f"Known sum: {costs.get('known_sum_usd')}", f"Pending reservations: {costs.get('pending_reservation_usd')}", f"Logical prompts: {costs.get('logical_prompts')}", "", str(report.get("claim_boundary", ""))]
    return "\n".join(lines) + "\n"


def _report(manifest: Mapping[str, Any], ledger: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, status: str) -> dict[str, Any]:
    requests = [value for value in ledger.get("requests", {}).values() if isinstance(value, Mapping)]
    cases = _case_map(manifest)
    eligible_ids = {case_id for case_id, case in cases.items() if case.get("max_chars", 0) > 0 and case.get("status") != "invalid"}
    zero_count = sum(case.get("max_chars") == 0 and case.get("status") != "invalid" for case in cases.values())
    invalid_count = sum(case.get("status") == "invalid" for case in cases.values())
    rows_by_case = {str(row.get("case_id")): row for row in rows}
    all_branches = [branch for row in rows for branch in row.get("branches", []) if isinstance(branch, Mapping)]
    pair_rep_counts: dict[str, dict[str, int]] = {}
    accepted_counts: dict[str, int] = {}
    firstpass_verdicts: Counter[str] = Counter()
    for rep in (1, 2):
        for generator, arbiter in PAIR_ORDER:
            pair = f"{generator}->{arbiter}"
            counts = {"accepted": 0, "declined": 0, "missing": 0, "unresolved": 0, "cannot_shorten": 0, "processed": 0, "zero": 0, "invalid": 0, "denominator": MAX_CASES}
            for case_id in cases:
                if cases[case_id].get('status') == 'invalid':
                    counts["invalid"] += 1
                elif cases[case_id].get("max_chars") == 0:
                    counts["zero"] += 1
                branch = next((value for value in all_branches if value.get("stage") == "review" and value.get("rep") == rep and value.get("generator") == generator and value.get("arbiter") == arbiter and value.get("case_id") == case_id), None)
                if branch is None:
                    if case_id in eligible_ids:
                        counts["missing"] += 1
                elif branch.get("status") != "complete":
                    if case_id in eligible_ids:
                        generation = next((value for value in all_branches if value.get("stage") == "generation" and value.get("rep") == rep and value.get("generator") == generator and value.get("case_id") == case_id), None)
                        if isinstance(generation, Mapping) and generation.get("reason") == "cannot_shorten":
                            counts["cannot_shorten"] += 1
                        else:
                            counts["unresolved"] += 1
                else:
                    counts["processed"] += 1
                    if branch.get("accepted"):
                        counts["accepted"] += 1
                    else:
                        counts["declined"] += 1
                    firstpass_verdicts[str(branch.get("verdict") or "unresolved")] += 1
            pair_rep_counts[f"rep{rep}:{pair}"] = counts
            accepted_counts[pair] = accepted_counts.get(pair, 0) + counts["accepted"]

    firstpass_by_origin: dict[tuple[int, str, str, str], dict[str, str]] = defaultdict(dict)
    for branch in all_branches:
        if branch.get('stage') == 'review' and branch.get('generator') in {'luna', 'deepseek'} and branch.get('arbiter') in {'luna', 'deepseek', 'sol'} and branch.get('status') == 'complete':
            identity = (int(branch.get('rep', 0)), str(branch.get('generator')), str(branch.get('case_id')), str(branch.get('candidate_origin_receipt_id')))
            firstpass_by_origin[identity][str(branch['arbiter'])] = str(branch.get('verdict'))
    disagreements = sum(len(set(values.values())) > 1 for values in firstpass_by_origin.values() if len(values) == 3)
    disagreement_completeness = sum(len(values) == 3 for values in firstpass_by_origin.values())
    coverage: dict[str, dict[str, int]] = {}
    finalaccepted_counts: dict[str, int] = defaultdict(int)
    for rep in (1, 2):
        for generator, arbiter in PAIR_ORDER:
            key = f"rep{rep}:{generator}->{arbiter}"
            relevant = [branch for branch in all_branches if branch.get("stage") == "review" and branch.get("rep") == rep and branch.get("generator") == generator and branch.get("arbiter") == arbiter and branch.get("case_id") in eligible_ids and branch.get("status") == "complete" and branch.get("verdict") in {"accept", "revise"} and branch.get("candidate") and (branch.get("verdict") == "revise" or not branch.get("local_fit", {}).get("fits"))]
            for branch in all_branches:
                if branch.get('stage') == 'review' and branch.get('rep') == rep and branch.get('generator') == generator and branch.get('arbiter') == arbiter and branch.get('accepted') is True:
                    finalaccepted_counts[f'{generator}->{arbiter}'] += 1
            attempted = [branch for branch in relevant if branch.get("repair_attempted")]
            rechecked = [branch for branch in attempted if isinstance(branch.get("repair"), Mapping) and isinstance(branch["repair"].get("recheck"), Mapping)]
            resolved = [branch for branch in rechecked if branch["repair"]["recheck"].get("status") == "complete"]
            for branch in resolved:
                if branch["repair"]["recheck"].get("accepted"):
                    finalaccepted_counts[f"{generator}->{arbiter}"] += 1
            skipped = [branch for branch in relevant if branch.get("repair_status") == "skipped_for_budget"]
            coverage[key] = {"eligible": len(relevant), "attempted": len(attempted), "rechecked": len(rechecked), "resolved": len(resolved), "skipped_for_budget": len(skipped), "pending": max(0, len(relevant) - len(attempted) - len(skipped)), "denominator": len(eligible_ids)}

    known_sum = Decimal(0)
    pending_reserve = Decimal(0)
    quota_units = 0
    sub_units = 0
    provider_costs: dict[str, Decimal] = defaultdict(Decimal)
    deepseek_upper = Decimal(0)
    observed_steps = 0
    physical_calls = 0
    for request in requests:
        if request.get("status") == "complete":
            known_sum += _decimal(request.get("cost_usd"), "cost")
            response = request.get("response")
            if isinstance(response, Mapping):
                provider_costs[str(response.get("provider"))] += _decimal(request.get("cost_usd"), "cost")
                if isinstance(response.get("calls"), list):
                    observed_steps += len(response["calls"])
                if isinstance(response.get("physical_calls_observed"), int):
                    physical_calls += response["physical_calls_observed"]
            if request.get("descriptor", {}).get("model_key") == "deepseek":
                deepseek_upper += _decimal(request.get('cost_usd'), 'derived DeepSeek cost')
        elif request.get("status") in {"pending", "unknown"}:
            pending_reserve += _decimal(request.get("reservation_usd"), "reservation")
        quota_units += int(request.get("quota_units", 0))
        if request.get("subscription") is True:
            sub_units += int(request.get("quota_units", 0))
    timing: dict[str, dict[str, float]] = defaultdict(lambda: {"elapsed_sec": 0.0, "requests": 0.0})
    for request in requests:
        response = request.get("response")
        descriptor = request.get("descriptor", {})
        if isinstance(response, Mapping) and isinstance(response.get("elapsed_sec"), (int, float)):
            key = f"{descriptor.get('stage', 'unknown')}:{response.get('provider')}"
            timing[key]["elapsed_sec"] += float(response["elapsed_sec"])
            timing[key]["requests"] += 1.0
    return {
        "schema": VERSION, "status": status, "scope": manifest.get("approved_scope"),
        "firstpass": {"denominator_per_pair_rep": MAX_CASES, "verdicts": dict(firstpass_verdicts), "accepted_counts": accepted_counts, "firstpass_counts": pair_rep_counts}, "finalaccepted_counts": dict(finalaccepted_counts),
        "firstpass_counts": pair_rep_counts, "eligible_denominator": len(eligible_ids), "zero_count": zero_count, "invalid_count": invalid_count,
        "initial_arb_disagreements": {"luna_deepseek_three_judges": disagreements, "completeness_count": disagreement_completeness},
        "disagreements": {"luna_deepseek_three_judges": disagreements, "completeness_count": disagreement_completeness},
        "primary_complete": all(pair_rep_counts[key]["unresolved"] == 0 and pair_rep_counts[key]["missing"] == 0 for key in pair_rep_counts),
        "repairs_complete": all(value["pending"] == 0 and value["skipped_for_budget"] == 0 and value["attempted"] == value["rechecked"] for value in coverage.values()),
        "costs": {"known_sum_usd": str(known_sum), "pending_reservation_usd": str(pending_reserve), "global_quota_units": quota_units, "subscription_quota_units": sub_units, "logical_prompts": len(requests), "sublogical_prompts": sum(1 for request in requests if request.get("subscription")), "observed_assistant_steps": observed_steps, "physical_calls_observed": physical_calls, "provider_actual_usd": {key: str(value) for key, value in provider_costs.items() if key == 'openrouter'}, "deepseek_conservative_upper_usd": str(deepseek_upper), "subscription_unpriced": True, "global_actual_qualifier": 'known_sum combines provider-reported OpenRouter cost and conservative no-cache DeepSeek token cost; subscription is unpriced'},
        "time_per_stage_provider": dict(timing), "repair_coverage": coverage, "audio_status": "not_measured", "claim_boundary": "No semantic truth or full-pilot claim.",
    }


async def run_matrix(output_dir: Path, backend: Backend, *, smoke: bool = False, _report_only: bool = False) -> dict[str, Any]:
    """Run first-pass generations/reviews and then bounded same-generator repairs."""
    output_dir = Path(output_dir)
    lock = FileLock(str(_lock_path(output_dir)), timeout=0)
    with lock:
        manifest = _read_manifest(output_dir)
        ledger = _load_ledger(output_dir, fresh=False, manifest=manifest)
        specs = _specs()
        cases = _case_map(manifest)
        selected_ids = list(manifest["selection"]["selected_case_ids"])
        requested_ids = selected_ids[:6] if smoke else selected_ids
        branches: dict[str, dict[str, Any]] = {}
        admission = _Admission(output_dir, manifest, ledger, specs, report_only=_report_only)
        generators = ("luna", "deepseek", "gemini", "muse")
        init_reps = (1,) if smoke else (1, 2)
        for rep in init_reps:
            for batch in manifest["batches"]:
                for case_id in batch["case_ids"]:
                    if case_id not in cases:
                        continue
                    for generator in generators:
                        branches[f"{rep}:generation:{generator}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": rep, "stage": "generation", "generator": generator, "batch": batch["batch_id"], "status": "pending"}
                        for pair_generator, arbiter in PAIR_ORDER:
                            if pair_generator == generator:
                                branches[f"{rep}:review:{generator}:{arbiter}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": rep, "stage": "review", "generator": generator, "arbiter": arbiter, "batch": batch["batch_id"], "status": "pending"}
        candidates: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = {}
        review_prompt_fingerprints: dict[tuple[int, str, str], str] = {}
        stopped = False
        stop_reason = "budget_limited"
        for repetition in (1,) if smoke else (1, 2):
            for batch in manifest["batches"]:
                batch_ids = [case_id for case_id in batch["case_ids"] if case_id in requested_ids]
                if not batch_ids:
                    continue
                context = [entry for case_id in batch_ids for entry in cases[case_id]["context"]]
                # Context is stable and deduplicated by qualified cue ID.
                context = _merge_context(context)
                for generator in generators:
                    targets = [{"cue_id": case_id, "original": cases[case_id]["original"], "max_chars": cases[case_id]["max_chars"], "length_basis": cases[case_id]["length_basis"]} for case_id in batch_ids if cases[case_id]["max_chars"] > 0 and cases[case_id]["status"] != "invalid"]
                    if not targets:
                        for case_id in batch_ids:
                            branch = branches[f"{repetition}:generation:{generator}:{batch['batch_id']}:{case_id}"]
                            branch.update({"status": "unresolved", "reason": "invalid_case" if cases[case_id]["status"] == "invalid" else "zero_prediction_unresolved"})
                        candidates[(repetition, batch["batch_id"], generator)] = {}
                        continue
                    prompt = build_generation_prompt(context, targets)
                    descriptor = _request_descriptor(manifest, rep=repetition, stage="generation", generator=generator, arbiter=None, batch=batch["batch_id"], system=GENERATOR_SYSTEM_PROMPT, user=prompt, schema=generation_schema())
                    try:
                        response, error, replayed = await _request(admission, backend, descriptor=descriptor, system=GENERATOR_SYSTEM_PROMPT, user=prompt, schema=generation_schema())
                    except (BudgetLimited, UnknownRequest, MatrixError) as exc:
                        stopped = True
                        stop_reason = "blocked_unknown" if isinstance(exc, (UnknownRequest, MatrixError)) else "budget_limited"
                        for case_id in batch_ids:
                            branches[f"{repetition}:generation:{generator}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": repetition, "stage": "generation", "generator": generator, "batch": batch["batch_id"], "status": "unresolved", "reason": type(exc).__name__}
                        break
                    parsed = parse_generation_partial(_response_text(response) or "", [row["cue_id"] for row in targets])
                    current: dict[str, dict[str, Any]] = {}
                    for case_id in batch_ids:
                        result = parsed.results.get(case_id)
                        if result is None:
                            current[case_id] = {"status": "unresolved", "reason": parsed.errors.get(case_id, error or "missing_generation")}
                            continue
                        if result.get("status") != "ok":
                            current[case_id] = {"status": "unresolved", "reason": "cannot_shorten", "generation": deepcopy(result)}
                            continue
                        text = result["text"]
                        fit = check_length(text, int(cases[case_id]["max_chars"]), cases[case_id]["length_basis"])
                        current[case_id] = {"status": "candidate", "text": text, "candidate_hash": _sha_bytes(text.encode()), "local_fit": fit, "origin": canonical_hash(descriptor)}
                    generation_key = canonical_hash(descriptor)
                    for case_id in batch_ids:
                        value = current[case_id]
                        branches[f"{repetition}:generation:{generator}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": repetition, "stage": "generation", "generator": generator, "batch": batch["batch_id"], "status": value["status"], "reason": value.get("reason"), "text": value.get("text"), "generation": value.get("generation"), "candidate_hash": value.get("candidate_hash"), "local_fit": value.get("local_fit"), "origin": value.get("origin"), "receipt_id": generation_key, "parse_error": parsed.errors.get(case_id)}
                    candidates[(repetition, batch["batch_id"], generator)] = current
                if stopped:
                    _mark_pending(manifest, branches, smoke=smoke)
                    return _write_outputs(output_dir, manifest, ledger, branches, status='reported_incomplete' if _report_only else stop_reason)
                # Every first-pass judge follows every generation, before repairs.
                for generator, arbiter in PAIR_ORDER:
                    current = candidates.get((repetition, batch["batch_id"], generator), {})
                    target_rows = [row for row in batch_ids if current.get(row, {}).get("status") == "candidate"]
                    for case_id in batch_ids:
                        if case_id not in target_rows:
                            reason = "invalid_case" if cases[case_id]["status"] == "invalid" else "zero_prediction_unresolved" if cases[case_id]["max_chars"] == 0 else "candidate_absent"
                            branches[f"{repetition}:review:{generator}:{arbiter}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": repetition, "stage": "review", "generator": generator, "arbiter": arbiter, "batch": batch["batch_id"], "status": "unresolved", "reason": reason}
                    if not target_rows:
                        continue
                    targets = [{"cue_id": case_id, "original": cases[case_id]["original"], "candidate": current[case_id]["text"], "max_chars": cases[case_id]["max_chars"], "actual_chars": current[case_id]["local_fit"]["length"], "length_basis": cases[case_id]["length_basis"]} for case_id in target_rows]
                    prompt = build_review_prompt(context, targets)
                    prompt_key = (repetition, batch["batch_id"], generator)
                    prior_prompt = review_prompt_fingerprints.setdefault(prompt_key, prompt)
                    if prior_prompt != prompt:
                        raise MatrixError("first-pass judge prompts differ for the same candidate")
                    descriptor = _request_descriptor(manifest, rep=repetition, stage="review", generator=generator, arbiter=arbiter, batch=batch["batch_id"], system=ARBITER_SYSTEM_PROMPT, user=prompt, schema=review_schema(), origin=canonical_hash({case_id: current[case_id]["origin"] for case_id in target_rows}))
                    review_receipt_id = canonical_hash(descriptor)
                    try:
                        response, error, replayed = await _request(admission, backend, descriptor=descriptor, system=ARBITER_SYSTEM_PROMPT, user=prompt, schema=review_schema())
                    except (BudgetLimited, UnknownRequest, MatrixError) as exc:
                        stopped = True
                        stop_reason = "blocked_unknown" if isinstance(exc, (UnknownRequest, MatrixError)) else "budget_limited"
                        for case_id in target_rows:
                            branches[f"{repetition}:review:{generator}:{arbiter}:{batch['batch_id']}:{case_id}"] = {"case_id": case_id, "rep": repetition, "stage": "review", "generator": generator, "arbiter": arbiter, "batch": batch["batch_id"], "status": "unresolved", "reason": type(exc).__name__, "candidate": current[case_id]}
                        break
                    parsed = parse_review_partial(_response_text(response) or "", target_rows)
                    for case_id in target_rows:
                        result = parsed.results.get(case_id)
                        item = current[case_id]
                        accepted = bool(result and result.get("verdict") == "accept" and item["local_fit"].get("fits"))
                        branch = {"case_id": case_id, "rep": repetition, "stage": "review", "generator": generator, "arbiter": arbiter, "batch": batch["batch_id"], "candidate": item["text"], "candidate_hash": item["candidate_hash"], "local_fit": item["local_fit"], "status": "complete" if result else "unresolved", "accepted": accepted, "verdict": result.get("verdict") if result else None, "issues": result.get("issues", []) if result else [], "advice": result.get("advice", "") if result else parsed.errors.get(case_id, error), "candidate_origin_receipt_id": item["origin"], "origin_receipt_id": item["origin"], "review_receipt_id": review_receipt_id, "parse_error": parsed.errors.get(case_id)}
                        branches[f"{repetition}:review:{generator}:{arbiter}:{batch['batch_id']}:{case_id}"] = branch
                    if stopped:
                        break
                _write_outputs(output_dir, manifest, ledger, branches, status="running")
                if stopped:
                    _mark_pending(manifest, branches, smoke=smoke)
                    return _write_outputs(output_dir, manifest, ledger, branches, status='reported_incomplete' if _report_only else stop_reason)
        if not smoke:
            # Repairs are one bounded request per pair/repetition/batch/generator.
            groups: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for branch in branches.values():
                if branch.get("stage") != "review" or branch.get("status") != "complete" or not branch.get("candidate"):
                    continue
                if branch.get("verdict") == "accept" and branch.get("local_fit", {}).get("fits"):
                    continue
                if branch.get("verdict") not in {"accept", "revise"}:
                    continue
                groups[(int(branch["rep"]), str(branch["batch"]), str(branch["generator"]), str(branch["arbiter"]))].append(branch)
            for (rep, batch_id, generator, arbiter), group in sorted(groups.items()):
                batch = next((value for value in manifest["batches"] if value["batch_id"] == batch_id), None)
                if not isinstance(batch, dict):
                    continue
                batch_ids = [case_id for case_id in batch["case_ids"] if case_id in cases]
                context = _merge_context([entry for case_id in batch_ids for entry in cases[case_id]["context"]])
                group_by_id = {branch["case_id"]: branch for branch in group}
                targets = []
                for case_id in batch_ids:
                    branch = group_by_id.get(case_id)
                    if branch is None:
                        continue
                    case = cases[case_id]
                    fit = branch.get("local_fit", {})
                    targets.append({"cue_id": case_id, "original": case["original"], "max_chars": case["max_chars"], "length_basis": case["length_basis"], "prior_candidate": branch["candidate"], "feedback": f"local length {fit.get('length')} exceeds limit {case['max_chars']}; " + " ".join(branch.get("issues", [])) + " " + branch.get("advice", "")})
                if not targets:
                    continue
                prompt = build_generation_prompt(context, targets)
                descriptor = _request_descriptor(manifest, rep=rep, stage="repair", generator=generator, arbiter=arbiter, batch=batch_id + ":repair", system=GENERATOR_SYSTEM_PROMPT, user=prompt, schema=generation_schema(), origin=canonical_hash({row["cue_id"]: group_by_id[row["cue_id"]]["candidate_hash"] for row in targets}))
                try:
                    response, error, _ = await _request(admission, backend, descriptor=descriptor, system=GENERATOR_SYSTEM_PROMPT, user=prompt, schema=generation_schema())
                except (BudgetLimited, UnknownRequest, MatrixError) as exc:
                    stopped = True
                    stop_reason = "blocked_unknown" if isinstance(exc, (UnknownRequest, MatrixError)) else "budget_limited"
                    for branch in group:
                        branch["repair_status"] = "skipped_for_budget" if isinstance(exc, BudgetLimited) else "unresolved"
                        branch["repair_reason"] = type(exc).__name__
                    if isinstance(exc, BudgetLimited):
                        for candidate in branches.values():
                            if candidate.get("stage") == "review" and candidate.get("status") == "complete" and not candidate.get("repair_attempted") and candidate.get("verdict") in {"accept", "revise"}:
                                candidate["repair_status"] = "skipped_for_budget"
                    _mark_pending(manifest, branches, smoke=False)
                    return _write_outputs(output_dir, manifest, ledger, branches, status='reported_incomplete' if _report_only else stop_reason)
                parsed = parse_generation_partial(_response_text(response) or "", [row["cue_id"] for row in targets])
                for target in targets:
                    case_id = target["cue_id"]
                    branch = group_by_id[case_id]
                    repaired = parsed.results.get(case_id)
                    branch["repair_attempted"] = True
                    if repaired is None or repaired.get("status") != "ok":
                        branch["repair_status"] = "unresolved"
                        branch["repair_reason"] = parsed.errors.get(case_id, "cannot_shorten" if repaired else error or "malformed_repair")
                        continue
                    case = cases[case_id]
                    fit = check_length(repaired["text"], int(case["max_chars"]), case["length_basis"])
                    branch["repair"] = {"stage": "repair", "status": "candidate", "text": repaired["text"], "candidate_hash": _sha_bytes(repaired["text"].encode()), "local_fit": fit, "origin": canonical_hash(descriptor), "receipt_id": canonical_hash(descriptor)}
                    if not fit.get("fits"):
                        branch["repair"]["status"] = "invalid_length"
                review_targets = []
                eligible_rechecks: list[dict[str, Any]] = []
                for branch in group:
                    repair = branch.get("repair")
                    if not isinstance(repair, dict) or repair.get("status") != "candidate":
                        continue
                    case = cases[branch["case_id"]]
                    review_targets.append({"cue_id": branch["case_id"], "original": case["original"], "candidate": repair["text"], "max_chars": case["max_chars"], "actual_chars": repair["local_fit"]["length"], "length_basis": case["length_basis"]})
                    eligible_rechecks.append(branch)
                if review_targets:
                    review_prompt = build_review_prompt(context, review_targets)
                    review_descriptor = _request_descriptor(manifest, rep=rep, stage="recheck", generator=generator, arbiter=arbiter, batch=batch_id + ":repair", system=ARBITER_SYSTEM_PROMPT, user=review_prompt, schema=review_schema(), origin=canonical_hash({branch["case_id"]: branch["repair"]["candidate_hash"] for branch in eligible_rechecks}))
                    recheck_receipt_id = canonical_hash(review_descriptor)
                    try:
                        review_response, review_error, _ = await _request(admission, backend, descriptor=review_descriptor, system=ARBITER_SYSTEM_PROMPT, user=review_prompt, schema=review_schema())
                        review_parsed = parse_review_partial(_response_text(review_response) or "", [row["cue_id"] for row in review_targets])
                        for branch in eligible_rechecks:
                            value = review_parsed.results.get(branch["case_id"])
                            branch["repair"]["recheck"] = {"status": "complete" if value else "unresolved", "accepted": bool(value and value.get("verdict") == "accept" and branch["repair"]["local_fit"].get("fits")), "verdict": value.get("verdict") if value else None, "issues": value.get("issues", []) if value else [], "advice": value.get("advice", "") if value else review_parsed.errors.get(branch["case_id"], review_error), "receipt_id": recheck_receipt_id}
                    except (BudgetLimited, UnknownRequest, MatrixError) as exc:
                        stopped = True
                        stop_reason = "blocked_unknown" if isinstance(exc, (UnknownRequest, MatrixError)) else "budget_limited"
                        for branch in eligible_rechecks:
                            branch["repair"]["recheck"] = {"status": "budget_limited" if isinstance(exc, BudgetLimited) else "unresolved", "reason": type(exc).__name__}
                        if isinstance(exc, BudgetLimited):
                            for candidate in branches.values():
                                if candidate.get("stage") == "review" and candidate.get("status") == "complete" and not candidate.get("repair_attempted") and candidate.get("verdict") in {"accept", "revise"}:
                                    candidate["repair_status"] = "skipped_for_budget"
                        _mark_pending(manifest, branches, smoke=False)
                        return _write_outputs(output_dir, manifest, ledger, branches, status='reported_incomplete' if _report_only else stop_reason)
                _write_outputs(output_dir, manifest, ledger, branches, status="running")
        return _write_outputs(output_dir, manifest, ledger, branches, status="smoke_complete" if smoke else "complete")


def report_matrix(output_dir: Path) -> dict[str, Any]:
    """Derive reports only from durable requests; never trust editable exports."""
    async def forbidden_backend(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('offline report attempted inference')
    return asyncio.run(run_matrix(Path(output_dir), forbidden_backend, _report_only=True))


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Offline-safe SH08 matrix coordinator")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--allow-muse-data", action="store_true")
    run.add_argument("--openrouter-no-byok", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_manifest(args.output_dir)
        return 0
    if args.command == "report":
        report_matrix(args.output_dir)
        return 0
    if not (args.allow_muse_data and args.openrouter_no_byok):
        return 2
    from utils.sh08_matrix_transport import MatrixBackend
    async def execute() -> dict[str, Any]:
        async with MatrixBackend(allow_muse_data=True, openrouter_no_byok=True) as backend:
            _read_manifest(args.output_dir)
            _load_ledger(args.output_dir, manifest=_read_manifest(args.output_dir))
            for model_key in MODEL_SPECS:
                result = backend.preflight(model_key)
                if not result.get('ok'):
                    raise MatrixError(f"local preflight failed for {model_key}: {result.get('error')}")
            return await run_matrix(args.output_dir, backend, smoke=args.smoke)
    try:
        result = asyncio.run(execute())
    except (BudgetLimited, UnknownRequest, MatrixError):
        return 2
    return 0 if result.get("status") in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
