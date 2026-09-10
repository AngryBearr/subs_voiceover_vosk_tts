"""Build a deterministic, source-disjoint subtitle shortening corpus.

This module only inventories and samples existing subtitle material.  It does not
generate subtitle text and does not call a model or a speech service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles
from utils.srt_to_json import parse_srt_file
from utils.vosk_text import normalize_vosk_text


DEFAULT_SEED = 20260910
PLANNER_CONFIG = PlannerConfig(cps=13.0, max_speed=1.5, gap_ms=200)

SPLIT_BASENAMES: dict[str, str] = {
    "S01E02.srt": "fit",
    "Survivor.S44E01.srt": "fit",
    "Survivor.S44E02.srt": "fit",
    "Survivor.S44E04.srt": "fit",
    "Survivor.S44E05.srt": "fit",
    "Australian Survivor S01E03_ru.srt": "calibration",
    "Survivor.S44E07.srt": "calibration",
    "Survivor.S44E08.srt": "calibration",
    "Survivor.S44E09.srt": "calibration",
    "S01E01.srt": "test",
    "Australian Survivor S01E04_ru.srt": "test",
    "Survivor.S44E03.srt": "test",
    "Survivor.S44E06.srt": "test",
    "Survivor.S44E10.srt": "test",
    "Survivor.S44E11.srt": "test",
    "Survivor.S44E12.srt": "test",
    "Survivor.S44E13.srt": "test",
}

_ALIAS_RE = re.compile(r"\(1\)(?=\.srt$)", re.IGNORECASE)
_LENGTH_BANDS = ((0, 15), (16, 35), (36, 60), (61, 100), (101, math.inf))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _text(item: dict[str, Any]) -> str:
    value = item.get("text", "")
    if isinstance(value, list):
        return " ".join(part for part in value if isinstance(part, str))
    return value if isinstance(value, str) else ""


def _timing_valid(item: dict[str, Any]) -> bool:
    start, end = item.get("start"), item.get("end")
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and start >= 0
        and end > start
    )


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _language_flag(texts: Iterable[str]) -> str:
    joined = " ".join(texts)
    letters = [char for char in joined if char.isalpha()]
    if not letters:
        return "unknown"
    ratio = sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters)
    return "cyrillic" if ratio >= 0.20 else "non_cyrillic"


def _source_stats(items: list[dict[str, Any]], plans: list[CuePlan]) -> dict[str, Any]:
    status_counts = Counter(plan.status for plan in plans)
    severity_counts = Counter(plan.severity for plan in plans)
    lengths = [len(_text(item)) for item in items]
    valid_lengths = [len(plan.text) for plan in plans if plan.status != "invalid"]
    invalid_zero = sum(
        isinstance(item.get("start"), int)
        and isinstance(item.get("end"), int)
        and item.get("start") == item.get("end")
        for item in items
    )
    invalid_negative = sum(
        isinstance(item.get("start"), int)
        and isinstance(item.get("end"), int)
        and (item.get("start") < 0 or item.get("end") < 0 or item.get("end") < item.get("start"))
        for item in items
    )
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "cue_chars": {
            "max": max(lengths, default=0),
            "p50": _percentile(valid_lengths, 0.50),
            "p90": _percentile(valid_lengths, 0.90),
        },
        "at_least_80": sum(length >= 80 for length in lengths),
        "at_least_120": sum(length >= 120 for length in lengths),
        "invalid_timing": {
            "zero_or_negative_count": invalid_zero + invalid_negative,
            "zero_duration_count": invalid_zero,
            "negative_or_reversed_count": invalid_negative,
        },
        "language_flag": _language_flag(_text(item) for item in items),
        "target_eligible_count": sum(plan.status == "needs_shortening" and bool(plan.text) for plan in plans),
        "target_skipped_count": sum(plan.status != "needs_shortening" or not plan.text for plan in plans),
        "missing_severity_strata": [
            severity for severity in ("easy", "medium", "hard") if severity_counts.get(severity, 0) == 0
        ],
    }


def _filename_key(path: Path, root: Path) -> tuple[str, str]:
    return (path.name.casefold(), path.relative_to(root).as_posix().casefold())


def _preferred_path(paths: list[Path], root: Path) -> Path:
    without_alias = [path for path in paths if not _ALIAS_RE.search(path.name)]
    return sorted(without_alias or paths, key=lambda path: _filename_key(path, root))[0]


def _source_id(preferred_raw_sha256: str) -> str:
    """Use the preferred immutable source bytes for the opaque stable ID."""
    return preferred_raw_sha256[:12]


def _plan_for_source(items: list[dict[str, Any]]) -> list[CuePlan]:
    return plan_subtitles(items, PLANNER_CONFIG)


def select_source_targets(
    items: list[dict[str, Any]],
    plans: list[CuePlan],
    count: int = 6,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Select deterministic needs-shortening cue IDs without semantic judgments."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    del items  # The plans carry all algorithmic selection data.
    eligible = [plan for plan in plans if plan.status == "needs_shortening" and plan.text]
    by_id = {plan.cue_id: plan for plan in eligible}
    selected: list[CuePlan] = []

    def take(ordered: list[CuePlan], limit: int) -> None:
        for plan in ordered[:limit]:
            if plan.cue_id not in {item.cue_id for item in selected} and len(selected) < count:
                selected.append(plan)

    take(sorted(eligible, key=lambda plan: (-len(plan.text), plan.position, plan.cue_id)), 2)
    take(sorted(eligible, key=lambda plan: (-plan.reduction_fraction, plan.position, plan.cue_id)), 2)
    remaining = [plan for plan in eligible if plan.cue_id not in {item.cue_id for item in selected}]
    rng = random.Random(seed)
    representative = [plan for plan in remaining if plan.severity in {"easy", "medium"}]
    rng.shuffle(representative)
    take(representative, count)
    remaining = [plan for plan in remaining if plan.cue_id not in {item.cue_id for item in selected}]
    rng.shuffle(remaining)
    take(remaining, count)
    if len(selected) < count:
        take(sorted(by_id.values(), key=lambda plan: (plan.position, plan.cue_id)), count)
    return [plan.cue_id for plan in sorted(selected, key=lambda plan: plan.position)[:count]]


def _context(items: list[dict[str, Any]], position: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = max(0, position - PLANNER_CONFIG.context_before)
    end = min(len(items), position + PLANNER_CONFIG.context_after + 1)
    for index in range(start, end):
        item = items[index]
        row: dict[str, Any] = {"cue_id": f"cue-{index}", "text": _text(item)}
        if isinstance(item.get("start"), int):
            row["start"] = item["start"]
        if isinstance(item.get("end"), int):
            row["end"] = item["end"]
        result.append(row)
    return result


def _length_band(length: int) -> str:
    for low, high in _LENGTH_BANDS:
        if low <= length <= high:
            if high == math.inf:
                return ">100"
            if low == 0:
                return "<=15"
            return f"{low}-{high}"
    raise AssertionError(length)


def _choose_measurement_candidates(
    candidates: list[dict[str, Any]],
    quota: int,
    seed: int,
    mandatory_ids: set[str] | None = None,
    excluded_text_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    mandatory_ids = mandatory_ids or set()
    excluded_text_hashes = excluded_text_hashes or set()
    available = [row for row in candidates if row["text_hash"] not in excluded_text_hashes]
    mandatory = [row for row in available if row["case_id"] in mandatory_ids]
    chosen_ids = {row["case_id"] for row in mandatory}
    selected = mandatory[:quota]
    rest = [row for row in available if row["case_id"] not in chosen_ids]
    rng = random.Random(seed)
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rest:
        by_band[_length_band(len(row["text"]))].append(row)
    for rows in by_band.values():
        rows.sort(key=lambda row: row["case_id"])
        rng.shuffle(rows)
    bands = list(by_band)
    rng.shuffle(bands)
    band_index = 0
    while len(selected) < quota and any(by_band.values()):
        if not bands:
            break
        band = bands[band_index % len(bands)]
        band_index += 1
        if by_band[band]:
            selected.append(by_band[band].pop())
    return selected[:quota]


def _audit_ids(cases: list[dict[str, Any]], seed: int) -> tuple[list[str], list[str], list[str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(case["source_id"], case["severity"])].append(case)
    for values in groups.values():
        values.sort(key=lambda case: case["case_id"])

    severity_order = ("easy", "medium", "hard")
    source_ids = sorted({case["source_id"] for case in cases})
    chosen: list[str] = []
    for source_index, source_id in enumerate(source_ids):
        local_rng = random.Random(f"{seed}:audit:{source_id}")
        available_severities = {
            severity: list(groups.get((source_id, severity), [])) for severity in severity_order
        }
        for values in available_severities.values():
            local_rng.shuffle(values)
        preferred_index = source_index % len(severity_order)
        selected: dict[str, Any] | None = None
        for offset in range(len(severity_order)):
            severity = severity_order[(preferred_index + offset) % len(severity_order)]
            if available_severities[severity]:
                selected = available_severities[severity].pop()
                break
        if selected is not None:
            chosen.append(selected["case_id"])

    chosen_set = set(chosen)
    remaining = [case for case in cases if case["case_id"] not in chosen_set]
    remaining_by_severity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in remaining:
        remaining_by_severity[case["severity"]].append(case)
    remaining_rng = random.Random(seed)
    for values in remaining_by_severity.values():
        values.sort(key=lambda case: case["case_id"])
        remaining_rng.shuffle(values)
    remaining_target = min(4, max(0, min(12, len(cases)) - len(chosen)))
    severity_cursor = 0
    for _ in range(remaining_target):
        severity = next(
            (
                severity_order[(severity_cursor + offset) % len(severity_order)]
                for offset in range(len(severity_order))
                if remaining_by_severity.get(severity_order[(severity_cursor + offset) % len(severity_order)])
            ),
            None,
        )
        if severity is None:
            break
        chosen.append(remaining_by_severity[severity].pop()["case_id"])
        severity_cursor = (severity_order.index(severity) + 1) % len(severity_order)

    audit_cases = [case for case in cases if case["case_id"] in set(chosen)]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in audit_cases:
        by_source[case["source_id"]].append(case)
    crosscheck_rng = random.Random(seed + 1)
    crosscheck_sources = sorted(by_source)
    crosscheck_rng.shuffle(crosscheck_sources)
    crosscheck = [
        crosscheck_sources_case["case_id"]
        for source_id in crosscheck_sources[: min(6, len(crosscheck_sources))]
        for crosscheck_sources_case in sorted(by_source[source_id], key=lambda case: case["case_id"])[
            :1
        ]
    ]
    order_rng = random.Random(seed + 2)
    order_candidates = [
        next(case for case in audit_cases if case["case_id"] == case_id)
        for case_id in crosscheck
    ]
    order_rng.shuffle(order_candidates)
    order_check: list[str] = []
    order_sources: set[str] = set()
    for case in order_candidates:
        if case["source_id"] in order_sources:
            continue
        order_check.append(case["case_id"])
        order_sources.add(case["source_id"])
        if len(order_check) == min(3, len(crosscheck)):
            break
    return chosen, crosscheck, order_check


def _discover_sources(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((path for path in input_dir.rglob("*.srt") if path.is_file()), key=lambda path: _filename_key(path, input_dir))
    if not paths:
        raise ValueError(f"No .srt files found under {input_dir}")
    records: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_bytes()
        items = parse_srt_file(str(path))
        records.append(
            {
                "path": path,
                "relative_path": path.relative_to(input_dir).as_posix(),
                "raw_sha256": _sha256_bytes(raw),
                "parsed_sha256": _canonical_hash(items),
                "items": items,
            }
        )
    return records


def _assign_split(paths: list[Path]) -> str:
    canonical_names = {_ALIAS_RE.sub("", path.name) for path in paths}
    splits = {SPLIT_BASENAMES.get(name) for name in canonical_names}
    splits.discard(None)
    unexpected = [path.name for path in paths if _ALIAS_RE.sub("", path.name) not in SPLIT_BASENAMES]
    if unexpected:
        raise ValueError(f"Unassigned source file(s): {', '.join(sorted(unexpected))}")
    if len(splits) != 1:
        raise ValueError(f"Canonical duplicate group crosses partitions: {', '.join(path.name for path in paths)}")
    return next(iter(splits))


def build_corpus(input_dir: Path, output_dir: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Build and write one immutable corpus manifest and its source JSON copies."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    records = _discover_sources(input_dir)
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_canonical[record["parsed_sha256"]].append(record)
        by_raw[record["raw_sha256"]].append(record)

    groups: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for parsed_sha, aliases in sorted(by_canonical.items()):
        paths = [record["path"] for record in aliases]
        split = _assign_split(paths)
        canonical = next(record for record in aliases if record["path"] == _preferred_path(paths, input_dir))
        source_id = _source_id(canonical["raw_sha256"])
        plans = _plan_for_source(canonical["items"])
        groups.append(
            {
                "canonical_sha256": parsed_sha,
                "raw_sha256s": sorted({record["raw_sha256"] for record in aliases}),
                "paths": sorted(record["relative_path"] for record in aliases),
                "preferred_path": canonical["relative_path"],
                "source_id": source_id,
                "split": split,
            }
        )
        sources.append(
            {
                "source_id": source_id,
                "path": canonical["relative_path"],
                "sha256": canonical["raw_sha256"],
                "parsed_sha256": parsed_sha,
                "split": split,
                "json_path": f"sources/{source_id}.json",
                "cue_count": len(canonical["items"]),
                "stats": _source_stats(canonical["items"], plans),
                "_items": canonical["items"],
                "_plans": plans,
            }
        )
    sources.sort(key=lambda source: (source["split"], source["path"].casefold(), source["source_id"]))

    output_dir.mkdir(parents=True)
    (output_dir / "sources").mkdir()
    for source in sources:
        (output_dir / source["json_path"]).write_text(
            json.dumps(source["_items"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    cases: list[dict[str, Any]] = []
    for source in sources:
        if source["split"] != "test":
            continue
        items = source["_items"]
        plans = source["_plans"]
        selected = select_source_targets(items, plans, count=6, seed=seed)
        selected_set = set(selected)
        for plan in plans:
            if plan.cue_id not in selected_set:
                continue
            position = plan.position
            item = items[position]
            source_index = item.get("index", position)
            case: dict[str, Any] = {
                "case_id": f"{source['source_id']}:{plan.cue_id}",
                "source_id": source["source_id"],
                "cue_id": plan.cue_id,
                "position": position,
                "source_index": source_index,
                "original": plan.text,
                "max_chars": plan.max_chars,
                "available_ms": plan.available_ms,
                "severity": plan.severity,
                "reduction_fraction": plan.reduction_fraction,
                "source_json_path": source["json_path"],
                "context": _context(items, position),
                "original_len": len(plan.text),
                "long_stress": len(plan.text) >= 120,
                "long_stress_80": len(plan.text) >= 80,
                "long_stress_120": len(plan.text) >= 120,
            }
            cases.append(case)
    cases.sort(key=lambda case: (case["source_id"], case["position"]))

    measurement_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_by_id = {case["case_id"]: case for case in cases}
    for source in sources:
        for position, (item, plan) in enumerate(zip(source["_items"], source["_plans"])):
            text = _text(item)
            if plan.status == "invalid" or not text:
                continue
            case_id = f"{source['source_id']}:cue-{position}"
            measurement_candidates[source["split"]].append(
                {
                    "case_id": case_id,
                    "source_id": source["source_id"],
                    "cue_id": f"cue-{position}",
                    "split": source["split"],
                    "text": text,
                    "position": position,
                    "available_ms": plan.available_ms,
                    "source_sha256": source["sha256"],
                    "text_hash": _sha256_bytes(normalize_vosk_text(text).encode("utf-8")),
                    "length_band": _length_band(len(text)),
                }
            )

    quotas = {"fit": 16, "calibration": 12, "test": 8}
    source_quota = {"fit": 16, "calibration": 12, "test": 8}
    measurement_rows: list[dict[str, Any]] = []
    reserved_hashes: set[str] = set()
    shortfalls: dict[str, Any] = {}
    removed_cross_split: dict[str, int] = {}
    for split in ("test", "calibration", "fit"):
        split_sources = [source for source in sources if source["split"] == split]
        split_selected: list[dict[str, Any]] = []
        for source in split_sources:
            candidates = [row for row in measurement_candidates[split] if row["source_id"] == source["source_id"]]
            mandatory = {case["case_id"] for case in cases if case["source_id"] == source["source_id"]}
            if split != "test":
                removed_cross_split[f"{split}:{source['source_id']}"] = sum(
                    row["text_hash"] in reserved_hashes for row in candidates
                )
            selected = _choose_measurement_candidates(
                candidates,
                source_quota[split],
                seed + len(split_selected) + len(source["source_id"]),
                mandatory_ids=mandatory if split == "test" else set(),
                excluded_text_hashes=reserved_hashes if split != "test" else set(),
            )
            split_selected.extend(selected)
            if len(selected) < source_quota[split]:
                shortfalls[f"{split}:{source['source_id']}"] = {
                    "requested": source_quota[split],
                    "actual": len(selected),
                }
        if split == "test":
            reserved_hashes.update(row["text_hash"] for row in split_selected)
        else:
            reserved_hashes.update(row["text_hash"] for row in split_selected)
        measurement_rows.extend(split_selected)

    # Report global bins and split counts from actual rows; no synthetic padding is made.
    measurement_rows.sort(key=lambda row: (row["split"], row["source_id"], row["position"]))
    measurement_cases = [
        {key: row[key] for key in ("case_id", "source_id", "cue_id", "split", "text", "position", "available_ms", "source_sha256", "text_hash", "length_band")}
        for row in measurement_rows
    ]
    measurement_summary = {
        "requested_by_split": {split: quotas[split] * sum(source["split"] == split for source in sources) for split in quotas},
        "actual_by_split": dict(Counter(row["split"] for row in measurement_rows)),
        "length_bands": dict(Counter(row["length_band"] for row in measurement_rows)),
        "cross_split_duplicate_policy": "test reserved first; calibration excludes test; fit excludes test and calibration",
        "removed_cross_split_text_rows": removed_cross_split,
        "quota_shortfalls": shortfalls,
        "case_severity_counts": dict(Counter(case["severity"] for case in cases)),
        "case_missing_severity_strata": [
            severity for severity in ("easy", "medium", "hard")
            if not any(case["severity"] == severity for case in cases)
        ],
        "case_quota_shortfalls": {
            source["source_id"]: {
                "requested": 6,
                "actual": sum(case["source_id"] == source["source_id"] for case in cases),
            }
            for source in sources
            if source["split"] == "test"
            and sum(case["source_id"] == source["source_id"] for case in cases) < 6
        },
        "case_long_stress_counts": {
            "original_len_ge_80": sum(case["original_len"] >= 80 for case in cases),
            "original_len_ge_120": sum(case["original_len"] >= 120 for case in cases),
        },
    }

    audit, crosscheck, order_check = _audit_ids(cases, seed)
    case_by_id = {case["case_id"]: case for case in cases}
    source_coverage = {
        name: {
            "count": len({case_by_id[case_id]["source_id"] for case_id in case_ids}),
            "source_ids": sorted({case_by_id[case_id]["source_id"] for case_id in case_ids}),
        }
        for name, case_ids in (
            ("audit", audit),
            ("crosscheck", crosscheck),
            ("order_check", order_check),
        )
    }
    audit_selection_meta = {
        "first_pass": "one case per test source, with seeded local RNG and severity round-robin",
        "remainder_pass": "seeded severity balance from cases not selected in the first pass",
        "source_coverage": source_coverage,
    }
    for source in sources:
        source.pop("_items", None)
        source.pop("_plans", None)
    raw_groups = [
        {"sha256": sha, "paths": sorted(record["relative_path"] for record in rows)}
        for sha, rows in sorted(by_raw.items())
        if len(rows) > 1
    ]
    canonical_groups = [
        {"sha256": sha, "paths": sorted(record["relative_path"] for record in rows)}
        for sha, rows in sorted(by_canonical.items())
        if len(rows) > 1
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input_root": str(input_dir),
        "planner_config": {
            "cps": PLANNER_CONFIG.cps,
            "max_speed": PLANNER_CONFIG.max_speed,
            "gap_ms": PLANNER_CONFIG.gap_ms,
            "context_before": PLANNER_CONFIG.context_before,
            "context_after": PLANNER_CONFIG.context_after,
        },
        "source_inventory": [
            {
                "path": record["relative_path"],
                "sha256": record["raw_sha256"],
                "parsed_sha256": record["parsed_sha256"],
            }
            for record in records
        ],
        "duplicate_groups": {"raw_sha256": raw_groups, "canonical_sha256": canonical_groups, "canonical_groups": groups},
        "sources": sources,
        "cases": cases,
        "measurement_cases": measurement_cases,
        "measurement_summary": measurement_summary,
        "audit_case_ids": audit,
        "crosscheck_case_ids": crosscheck,
        "order_check_case_ids": order_check,
        "audit_selection_meta": audit_selection_meta,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        "Shortening corpus manifest\n"
        f"sources={len(sources)} inventory_files={len(records)} cases={len(cases)} "
        f"measurement_cases={len(measurement_cases)} audit={len(audit)} "
        f"crosscheck={len(crosscheck)} order_check={len(order_check)}\n"
        f"fit={sum(row['split'] == 'fit' for row in measurement_rows)} "
        f"calibration={sum(row['split'] == 'calibration' for row in measurement_rows)} "
        f"test={sum(row['split'] == 'test' for row in measurement_rows)}\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic subtitle shortening corpus")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = build_corpus(args.input_dir, args.output_dir, seed=args.seed)
    print(json.dumps({"sources": len(manifest["sources"]), "cases": len(manifest["cases"]), "measurement_cases": len(manifest["measurement_cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
