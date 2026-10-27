"""Pipeline: parse -> units -> calibrate -> synth0 -> oracle -> schedule -> annotate -> compress
-> measure -> verify -> resolve -> export -> report.

Continuation cues are merged into speech units; everything after ``units`` works on units.

Every stage reads and writes JSON artifacts in the output directory.  Each
artifact records the hashes of its inputs; a stage is skipped when its
artifact already matches, so reruns are cheap and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from utils.srt_to_json import parse_srt_file
from voiceover.calibrate import (CALIBRATION_VERSION, Calibration, calibrate, derive_timeline_policy,
                                 engine_identity)
from voiceover.compress import (Annotation, CompressionPolicy, annotate_scenes, build_jobs, run_compression,
                                shortening_jobs, summarize)
from voiceover.export import export_final
from voiceover.engine import Engine, PreparedText, SynthResult, build_engine, result_dict
from voiceover.llm import LLMClient, LLMConfig
from voiceover.measure import MANUAL_RUNG, candidate_items, manual_items, measure_row, row_key, split_key
from voiceover.oracle import Oracle, fit_oracle
from voiceover.overrides import Overrides, load_overrides
# Re-exported: the text-target helpers moved to voiceover.placement.
from voiceover.placement import (TEXT_DEAD_ZONE_SEC, TIMING_LIMITED_MIN_SYLLABLES, solve_placements,
                                 syllables_to_remove, text_band, text_target as _text_target)
from voiceover.render import StretchFn, stretch_clip
from voiceover.resolve import resolve
from voiceover.text import join_lines, syllables
from voiceover.timeline import TimelinePolicy
from voiceover.units import MergePolicy, build_units
from voiceover.verify import VerifyPolicy, verify_rungs, VERIFY_RULES_VERSION

STAGES = ("parse", "units", "calibrate", "synth0", "oracle", "schedule", "annotate", "compress", "measure", "verify",
          "resolve", "export", "report")
OVERRIDES_FILE = "overrides.json"
log = logging.getLogger("voiceover")


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict) or "engine" not in config or "policy" not in config:
        raise ValueError("config must contain 'engine' and 'policy' objects")
    return config


# ---------------------------------------------------------------- parse

def parse_input(source: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Return cues with stable ids; accepts .srt or a JSON list of cues."""
    if source.suffix.lower() == ".srt":
        raw = parse_srt_file(str(source))
    else:
        raw = read_json(source)
        if not isinstance(raw, list):
            raise ValueError("JSON input must be a list of cues")
    cues: list[dict[str, Any]] = []
    for position, item in enumerate(raw):
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"cue at position {position} has non-integer timing")
        cue = {
            "cue_id": f"cue-{position}",
            "position": position,
            "index": item.get("index", position + 1),
            "start": start,
            "end": end,
            "text": join_lines(item.get("text")),
        }
        if item.get("voice") is not None:  # per-line voice (e.g. by speaker gender, from stt.dub)
            cue["voice"] = item["voice"]
        cues.append(cue)
    if limit is not None:
        cues = cues[:limit]
    return cues


def stage_parse(source: Path, out: Path, limit: int | None) -> dict[str, Any]:
    path = out / "cues.json"
    cues = parse_input(source, limit)
    artifact = {"schema": "voiceover-cues-v1", "source": str(source.resolve()), "source_sha256": _file_sha(source),
                "limit": limit, "cues": cues}
    if path.exists() and read_json(path).get("cues") != cues:
        raise ValueError(f"{out} already holds a different source; use a new output directory")
    write_json(path, artifact)
    return artifact


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- units

def stage_units(cues_artifact: dict[str, Any], policy: MergePolicy, out: Path) -> dict[str, Any]:
    """Merge continuation cues into speech units."""
    path = out / "units.json"
    inputs = {"cues_sha256": json_sha256(cues_artifact["cues"]), "policy": policy.__dict__}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("units: artifact matches, skipping")
            return existing
    units = build_units(cues_artifact["cues"], policy)
    artifact = {"schema": "voiceover-units-v1", "inputs": inputs, "units": [unit.to_dict() for unit in units]}
    write_json(path, artifact)
    return artifact


# ---------------------------------------------------------------- calibrate

PROBE_RATES = (0.8, 1.1, 1.4)


def stage_calibrate(engine_config: dict[str, Any], policy_config: dict[str, Any], cache_dir: Path, workers: int,
                    out: Path) -> Calibration | None:
    """Find the engine rate that gives the base tempo in syllables per second.

    Returns ``None`` for legacy policies written in engine rate units.
    """
    if "base_syl_per_sec" not in policy_config:
        return None
    target = float(policy_config["base_syl_per_sec"])
    path = out / "calibration.json"
    inputs = {"engine": engine_identity(engine_config), "target_syl_per_sec": target,
              "calibration_version": CALIBRATION_VERSION, "probe_rates": list(PROBE_RATES)}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("calibrate: artifact matches, skipping")
            return Calibration.from_dict(existing["calibration"])
    result = calibrate(engine_config, target, cache_dir, workers, probe_rates=PROBE_RATES)
    write_json(path, {"schema": "voiceover-calibration-v1", "inputs": inputs, "calibration": result.to_dict()})
    log.info("calibrate: voice %s target %.2f syl/s -> rate %.3f, measured %.2f syl/s (%d probes + %d refinement%s)",
             engine_config.get("voice"), target, result.rate, result.measured_syl_per_sec, len(PROBE_RATES),
             len(result.samples) - len(PROBE_RATES), "" if len(result.samples) - len(PROBE_RATES) == 1 else "s")
    if not result.within_tolerance:
        log.warning("calibrate: measured speed is outside the tolerance of the target")
    return result


# ---------------------------------------------------------------- synth0

def _duration_row(unit: dict[str, Any], text: PreparedText, result: SynthResult) -> dict[str, Any]:
    row = {"unit_id": unit["unit_id"], "cue_ids": list(unit["cue_ids"]), "position": unit["position"],
           "index": unit["index"], "start": unit["start"], "end": unit["end"], "merged": unit["merged"],
           "scene_id": unit["scene_id"], "display_text": text.display_text,
           "tts_text": text.tts_text, "normalization": text.normalization,
           "normalization_error": text.error, "notes": list(text.notes),
           **{k: v for k, v in result_dict(result).items() if k != "cue_id"}}
    if unit.get("voice") is not None:
        row["voice"] = unit["voice"]
    return row


def _engine_items(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The engine's key stays ``cue_id``; here it carries the unit id.
    return [{"cue_id": unit["unit_id"], "index": unit["index"], "text": unit["text"], "voice": unit.get("voice")}
            for unit in units]


def stage_synth0(units_artifact: dict[str, Any], engine: Engine, out: Path, workers: int,
                 cache_dir: Path) -> dict[str, Any]:
    """Prepare and measure every unit once; distinct engine texts are cached across runs."""
    path = out / "durations0.json"
    inputs = {"units_sha256": json_sha256(units_artifact["units"]), "engine_identity_sha256": engine.identity_sha(),
              "synth_identity_sha256": engine.synth_identity_sha()}
    units = units_artifact["units"]
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            failed = {row["unit_id"] for row in existing["rows"] if row.get("error")}
            if not failed:
                log.info("synth0: artifact matches, skipping")
                return existing
            log.info("synth0: artifact matches; retrying %d unit(s) that failed earlier", len(failed))
            retry = [unit for unit in units if unit["unit_id"] in failed]
            prepared = engine.prepare(_engine_items(retry))
            results = {row.cue_id: row for row in engine.synthesize_many(prepared, cache_dir, 1)}
            texts = {row.cue_id: row for row in prepared}
            by_id = {unit["unit_id"]: unit for unit in units}
            existing["rows"] = [_duration_row(by_id[row["unit_id"]], texts[row["unit_id"]], results[row["unit_id"]])
                                if row["unit_id"] in results else row for row in existing["rows"]]
            write_json(path, existing)
            return existing
        log.info("synth0: inputs changed (text or engine); rebuilding, cached texts are reused")
    started = time.time()
    log.info("synth0: preparing %d units (expansion + stress)", len(units))
    prepared = engine.prepare(_engine_items(units))
    prepare_sec = time.time() - started
    log.info("synth0: synthesizing with %d worker(s)", workers)
    started = time.time()
    results = engine.synthesize_many(prepared, cache_dir, workers)
    synth_sec = time.time() - started
    by_result = {row.cue_id: row for row in results}
    rows = [_duration_row(unit, text, by_result[unit["unit_id"]]) for unit, text in zip(units, prepared)]
    artifact = {"schema": "voiceover-durations-v3", "inputs": inputs, "engine_identity": engine.identity(),
                "timing": {"prepare_sec": prepare_sec, "synth_sec": synth_sec, "workers": workers}, "rows": rows}
    write_json(path, artifact)
    return artifact


# ---------------------------------------------------------------- oracle

def stage_oracle(durations: dict[str, Any], out: Path) -> Oracle:
    path = out / "oracle.json"
    inputs = {"durations_sha256": json_sha256(durations["rows"])}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            return Oracle.from_dict(existing["oracle"])
    samples = [(row["tts_text"], float(row["speech_sec"])) for row in durations["rows"] if not row.get("error")]
    oracle = fit_oracle(samples, durations["inputs"]["engine_identity_sha256"])
    write_json(path, {"schema": "voiceover-oracle-v1", "inputs": inputs, "oracle": oracle.to_dict()})
    return oracle


# ---------------------------------------------------------------- schedule

def stage_schedule(durations: dict[str, Any], oracle: Oracle, policy: TimelinePolicy, cues_artifact: dict[str, Any],
                   out: Path) -> dict[str, Any]:
    path = out / "schedule0.json"
    inputs = {"durations_sha256": json_sha256(durations["rows"]), "policy": policy.__dict__,
              "oracle_sha256": json_sha256(oracle.to_dict()),
              "cues_sha256": json_sha256(cues_artifact["cues"])}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            return existing
    cues_by_id = {cue["cue_id"]: cue for cue in cues_artifact["cues"]}
    enriched, skipped, solve_sec = solve_placements(durations["rows"], policy, oracle, cues_by_id)
    artifact = {"schema": "voiceover-schedule-v2", "inputs": inputs, "solve_sec": solve_sec,
                "skipped_units": skipped, "placements": enriched}
    write_json(path, artifact)
    return artifact


# ---------------------------------------------------------------- annotate / compress

def _llm_identity(client: LLMClient) -> dict[str, Any]:
    """The config fields that change what the model answers."""
    config = client.config
    return {"provider": config.provider, "model": config.model, "temperature": config.temperature,
            "max_tokens": config.max_tokens, "reasoning": config.reasoning}


def _stats_delta(client: LLMClient, before: dict[str, int]) -> dict[str, int]:
    return {key: value - before[key] for key, value in client.stats().items()}


def stage_annotate(units_artifact: dict[str, Any], durations: dict[str, Any], schedule_artifact: dict[str, Any],
                   policy: CompressionPolicy, client: LLMClient, out: Path) -> dict[str, Any]:
    """One cached model call per scene that holds a unit to shorten."""
    path = out / "annotations.json"
    jobs, scenes = build_jobs(units_artifact["units"], schedule_artifact["placements"], durations["rows"], policy)
    marked = shortening_jobs(jobs)
    marked_scenes = sorted({job.scene_id for job in marked})
    inputs = {"units_sha256": json_sha256([[j.unit_id, j.text, j.text_band] for j in marked]),
              "scenes_sha256": json_sha256({s: [[u.unit_id, u.text] for u in scenes[s]] for s in marked_scenes}),
              "llm": _llm_identity(client), "prompt_version": policy.prompt_version}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("annotate: artifact matches, skipping")
            return existing
    before = client.stats()
    annotations = annotate_scenes(client, scenes, jobs, policy)
    delta = _stats_delta(client, before)
    log.info("annotate: %d scenes, %d cached", len(marked_scenes), delta["cache_hits"])
    artifact = {"schema": "voiceover-annotations-v1", "inputs": inputs, "llm_stats": delta,
                "llm_models": sorted(client.models_seen),
                "annotations": {uid: annotation.to_dict() for uid, annotation in annotations.items()}}
    write_json(path, artifact)
    return artifact


def stage_compress(units_artifact: dict[str, Any], durations: dict[str, Any], schedule_artifact: dict[str, Any],
                   oracle: Oracle, annotations_artifact: dict[str, Any], policy: CompressionPolicy,
                   client: LLMClient, out: Path) -> dict[str, Any]:
    """Ladder compression per unit; the oracle and local checks pick the rung."""
    path = out / "candidates.json"
    inputs = {"annotations_sha256": json_sha256(annotations_artifact["annotations"]),
              "durations_sha256": json_sha256(durations["rows"]),
              "schedule_sha256": json_sha256(schedule_artifact["placements"]),
              "oracle_sha256": json_sha256(oracle.to_dict()), "llm": _llm_identity(client),
              "prompt_version": policy.prompt_version, "policy": policy.to_dict()}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("compress: artifact matches, skipping")
            return existing
    jobs, scenes = build_jobs(units_artifact["units"], schedule_artifact["placements"], durations["rows"], policy)
    annotations = {uid: Annotation.from_dict(uid, data) for uid, data in annotations_artifact["annotations"].items()}
    before = client.stats()
    entries = run_compression(client, jobs, scenes, oracle, policy, annotations)
    delta = _stats_delta(client, before)
    ledger = [entry.to_dict() for entry in entries]
    summary = summarize(ledger)
    log.info("compress: %d units, resolved %d, unresolved %d, rejected %d, timing_limited %d; "
             "llm calls %d, cache hits %d", summary["units"], summary["resolved"], summary["unresolved"],
             summary["rejected"], summary["timing_limited"], delta["calls_made"], delta["cache_hits"])
    artifact = {"schema": "voiceover-candidates-v1", "inputs": inputs, "llm_stats": delta,
                "llm_models": sorted(client.models_seen), "summary": summary, "ledger": ledger}
    write_json(path, artifact)
    return artifact


# ---------------------------------------------------------------- measure / verify / resolve / export

def overrides_sha256(overrides: Overrides) -> str | None:
    """Content hash of the overrides that change results; ``None`` when there are none."""
    return None if overrides.is_empty else json_sha256(overrides.canonical())


def stage_measure(candidates: dict[str, Any], durations: dict[str, Any], engine: Engine, out: Path, workers: int,
                  cache_dir: Path, overrides: Overrides | None = None) -> dict[str, Any]:
    """Synthesise every candidate rung that passed the hard checks, with the same engine as synth0.

    Manual override texts are measured too, one row per text with ``rung == "manual"`` (engine key ``unit#manual``).
    """
    overrides = overrides or Overrides.empty()
    path = out / "durations1.json"
    inputs = {"candidates_sha256": json_sha256(candidates["ledger"]),
              "synth_identity_sha256": engine.synth_identity_sha(), "overrides_sha256": overrides_sha256(overrides)}
    indices = {row["unit_id"]: int(row["index"]) for row in durations["rows"]}
    voices = {row["unit_id"]: row.get("voice") for row in durations["rows"]}
    items = candidate_items(candidates["ledger"], indices) + manual_items(overrides.texts, indices)
    for item in items:
        item["voice"] = voices.get(split_key(item["cue_id"])[0])
    texts = {item["cue_id"]: item["text"] for item in items}

    def rows_for(selected: list[dict[str, Any]], workers: int) -> dict[str, dict[str, Any]]:
        if not selected:
            return {}
        prepared = engine.prepare(selected)
        by_key = {row.cue_id: row for row in prepared}
        results = engine.synthesize_many(prepared, cache_dir, workers)
        rows: dict[str, dict[str, Any]] = {}
        for result in results:
            unit_id, rung = split_key(result.cue_id)
            rows[result.cue_id] = measure_row(unit_id, rung, texts[result.cue_id], by_key[result.cue_id], result)
        return rows

    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            failed = {row_key(row["unit_id"], row["rung"]) for row in existing["rows"] if row.get("error")}
            if not failed:
                log.info("measure: artifact matches, skipping")
                return existing
            log.info("measure: artifact matches; retrying %d clip(s) that failed earlier", len(failed))
            retried = rows_for([item for item in items if item["cue_id"] in failed], 1)
            existing["rows"] = [retried.get(row_key(row["unit_id"], row["rung"]), row) for row in existing["rows"]]
            write_json(path, existing)
            return existing
        log.info("measure: candidates or engine changed; rebuilding, cached texts are reused")
    started = time.time()
    rows = rows_for(items, workers)
    synth_sec = time.time() - started
    log.info("measure: %d candidate clips synthesised", len(rows))
    artifact = {"schema": "voiceover-durations1-v1", "inputs": inputs,
                "timing": {"synth_sec": synth_sec, "workers": workers},
                "rows": [rows[item["cue_id"]] for item in items if item["cue_id"] in rows]}
    write_json(path, artifact)
    return artifact


def stage_verify(candidates: dict[str, Any], durations1: dict[str, Any], policy: VerifyPolicy, client: LLMClient,
                 out: Path) -> dict[str, Any]:
    """Entailment check of every measured rung (one cached model call each)."""
    path = out / "verified.json"
    # Manual texts are never verified, so overrides must not invalidate this artifact.
    rows1 = [row for row in durations1["rows"] if row["rung"] != MANUAL_RUNG]
    inputs = {"durations1_sha256": json_sha256(rows1),
              "candidates_sha256": json_sha256(candidates["ledger"]), "llm": _llm_identity(client),
              "prompt_version": policy.prompt_version, "enabled": policy.enabled,
              "rules_version": VERIFY_RULES_VERSION}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("verify: artifact matches, skipping")
            return existing
    measured = {(row["unit_id"], row["rung"]) for row in rows1 if not row.get("error")}
    before = client.stats()
    verdicts = verify_rungs(client, policy, candidates["ledger"], measured)
    delta = _stats_delta(client, before)
    log.info("verify: %d rungs, rejected %d; llm calls %d, cache hits %d", len(verdicts),
             sum(1 for v in verdicts if not v["ok"]), delta["calls_made"], delta["cache_hits"])
    artifact = {"schema": "voiceover-verified-v1", "inputs": inputs, "llm_stats": delta,
                "llm_models": sorted(client.models_seen), "verdicts": verdicts}
    write_json(path, artifact)
    return artifact


def stage_resolve(units_artifact: dict[str, Any], cues_artifact: dict[str, Any], durations: dict[str, Any],
                  durations1: dict[str, Any], verified: dict[str, Any], candidates: dict[str, Any],
                  schedule_artifact: dict[str, Any], policy: TimelinePolicy, oracle: Oracle,
                  compression: CompressionPolicy, out: Path, overrides: Overrides | None = None) -> dict[str, Any]:
    """Pick rungs by measurement and re-solve the timeline (at most two solves)."""
    overrides = overrides or Overrides.empty()
    path = out / "schedule1.json"
    inputs = {"durations0_sha256": json_sha256(durations["rows"]), "durations1_sha256": json_sha256(durations1["rows"]),
              "verified_sha256": json_sha256(verified["verdicts"]),
              "candidates_sha256": json_sha256(candidates["ledger"]),
              "schedule0_sha256": json_sha256(schedule_artifact["placements"]), "policy": policy.__dict__,
              "oracle_sha256": json_sha256(oracle.to_dict()), "cues_sha256": json_sha256(cues_artifact["cues"]),
              "fit_slack_sec": compression.fit_slack_sec, "overrides_sha256": overrides_sha256(overrides)}
    if path.exists():
        existing = read_json(path)
        if existing.get("inputs") == inputs:
            log.info("resolve: artifact matches, skipping")
            return existing
    resolution = resolve(units_artifact["units"], durations["rows"], durations1["rows"], verified["verdicts"],
                         candidates["ledger"], schedule_artifact["placements"], policy, oracle,
                         cues_artifact["cues"], compression.fit_slack_sec, overrides)
    artifact = {"schema": "voiceover-schedule1-v1", "inputs": inputs, "passes": resolution.passes,
                "choices": resolution.choices, "placements": resolution.placements}
    for item in resolution.passes:
        log.info("resolve: pass %d changed %d units, still needing text %d", item["pass"], item["changed_units"],
                 item["needs_text"])
    write_json(path, artifact)
    return artifact


def stage_export(cues_artifact: dict[str, Any], units_artifact: dict[str, Any], durations: dict[str, Any],
                 durations1: dict[str, Any], schedule1: dict[str, Any], verified: dict[str, Any],
                 candidates: dict[str, Any], policy: TimelinePolicy, oracle: Oracle, calibration: Calibration | None,
                 config: dict[str, Any], engine: Engine, llm_info: dict[str, Any], out: Path,
                 stretch: StretchFn = stretch_clip, overrides: Overrides | None = None) -> dict[str, Any]:
    """Render the final track and write subtitles, ledger, human queue and manifest (always re-runs)."""
    overrides = overrides or Overrides.empty()
    manifest_extra = {
        "engine_identity": engine.identity(),
        "calibration": calibration.to_dict() if calibration is not None else None,
        "oracle": oracle.to_dict(), "policy": policy.__dict__,
        "prompt_versions": {"compression": CompressionPolicy.from_dict(config.get("compression", {})).prompt_version,
                            "verify": VerifyPolicy.from_dict(config.get("verify", {})).prompt_version},
        "llm": {"compression": {**_llm_identity_dict(candidates), "models": candidates["llm_models"],
                                "stats": candidates["llm_stats"]},
                "verify": {**llm_info, "models": verified["llm_models"], "stats": verified["llm_stats"]}},
        "stage_inputs": {"durations0": json_sha256(durations["rows"]), "durations1": json_sha256(durations1["rows"]),
                         "candidates": json_sha256(candidates["ledger"]), "verified": json_sha256(verified["verdicts"]),
                         "schedule1": json_sha256(schedule1["placements"]),
                         "cues": json_sha256(cues_artifact["cues"]), "units": json_sha256(units_artifact["units"]),
                         "overrides": overrides_sha256(overrides)}}
    export_cfg = config.get("export", {})
    sample_rate = int(export_cfg.get("sample_rate", 22050))
    loudness = export_cfg.get("loudness_db")
    manifest_extra["loudness"] = {"loudness_db": loudness, "max_gain_db": float(export_cfg.get("max_gain_db", 20.0))}
    summary = export_final(out / "export", cues=cues_artifact["cues"], units=units_artifact["units"],
                           durations0=durations, durations1=durations1, schedule1=schedule1, verified=verified,
                           policy=policy, sample_rate=sample_rate, manifest_extra=manifest_extra, stretch=stretch,
                           accepted=overrides.accepted, loudness_db=None if loudness is None else float(loudness),
                           max_gain_db=float(export_cfg.get("max_gain_db", 20.0)))
    log.info("export: %s", summary["counts"])
    return summary


def _llm_identity_dict(candidates: dict[str, Any]) -> dict[str, Any]:
    return dict(candidates["inputs"]["llm"])


# ---------------------------------------------------------------- report

def _bucket(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge:g}"
    return f"> {edges[-1]:g}"


def stage_report(cues: dict[str, Any], units_artifact: dict[str, Any], durations: dict[str, Any], oracle: Oracle, schedule_artifact: dict[str, Any],
                 policy: TimelinePolicy, out: Path, candidates: dict[str, Any] | None = None,
                 calibration: Calibration | None = None, resolution: dict[str, Any] | None = None,
                 export_summary: dict[str, Any] | None = None, overrides: Overrides | None = None) -> dict[str, Any]:
    rows = durations["rows"]
    placements = schedule_artifact["placements"]
    ok_rows = [row for row in rows if not row.get("error") and row["speech_sec"] > 0]
    speech_total = sum(row["speech_sec"] for row in ok_rows)
    status_counts: dict[str, int] = {}
    for placement in placements:
        status_counts[placement["status"]] = status_counts.get(placement["status"], 0) + 1
    needs_text = [p for p in placements if p["text_band"] != "none"]
    deficit_hist: dict[str, int] = {}
    for p in needs_text:
        key = _bucket(p["removed_sec"], [0.1, 0.3, 0.6, 1.0, 2.0])
        deficit_hist[key] = deficit_hist.get(key, 0) + 1
    tempo_values = [p["tempo_factor"] for p in placements]
    normalization_counts: dict[str, int] = {}
    for row in rows:
        normalization_counts[row["normalization"]] = normalization_counts.get(row["normalization"], 0) + 1
    naive_over = sum(1 for row in ok_rows if row["speech_sec"] * 1000 > (row["end"] - row["start"]))
    band_counts: dict[str, int] = {}
    for p in placements:
        band_counts[p["text_band"]] = band_counts.get(p["text_band"], 0) + 1
    units = units_artifact["units"]
    merged = [unit for unit in units if unit["merged"]]
    merge_reasons: dict[str, int] = {}
    for unit in merged:
        for reason in unit["reasons"]:
            merge_reasons[reason] = merge_reasons.get(reason, 0) + 1
    # The first member's drift is just the unit's placement shift (bounded by the policy);
    # only later members can drift beyond the sync tolerance.
    drifts = sorted(abs(m["drift_ms"]) for p in placements for m in p.get("members", [])[1:])
    summary = {
        "cue_count": len(cues["cues"]),
        "unit_count": len(units),
        "merged_units": len(merged),
        "merged_cues": sum(len(unit["cue_ids"]) - 1 for unit in merged),
        "merge_reasons": merge_reasons,
        "scene_count": len({unit["scene_id"] for unit in units}),
        "drift_p50_ms": int(statistics.median(drifts)) if drifts else 0,
        "drift_p90_ms": drifts[int(0.9 * (len(drifts) - 1))] if drifts else 0,
        "drift_max_ms": drifts[-1] if drifts else 0,
        "drift_over_1000ms": sum(1 for d in drifts if d > 1000),
        "measured_units": len(ok_rows),
        "measured_cues": len(ok_rows),
        "synthesis_errors": sum(1 for row in rows if row.get("error")),
        "normalization": normalization_counts,
        "speech_total_sec": round(speech_total, 1),
        "median_syllables_per_sec": round(statistics.median(
            syllables(row["tts_text"]) / row["speech_sec"] for row in ok_rows if syllables(row["tts_text"]) > 0), 2),
        "oracle": oracle.to_dict(),
        "naive_overflow_cues": naive_over,
        "status_counts": status_counts,
        "needs_text_cues": len(needs_text),
        "text_bands": band_counts,
        "removed_total_sec": round(sum(p["removed_sec"] for p in needs_text), 2),
        "deficit_histogram": deficit_hist,
        "tempo_mean": round(statistics.fmean(tempo_values), 3) if tempo_values else None,
        "tempo_p90": round(sorted(tempo_values)[int(0.9 * (len(tempo_values) - 1))], 3) if tempo_values else None,
        "collisions": sum(1 for p in placements if p["collision_ms"] > 0),
        "timing": {**durations.get("timing", {}), "solve_sec": schedule_artifact.get("solve_sec")},
        "policy": policy.__dict__,
        "calibration": calibration.to_dict() if calibration is not None else None,
    }
    if candidates is not None:
        summary["compression"] = {**candidates["summary"], "llm_models": candidates["llm_models"],
                                  "llm_stats": candidates["llm_stats"]}
    if resolution is not None and candidates is not None:
        summary["resolution"] = _resolution_summary(resolution, candidates, export_summary)
        if overrides is not None and not overrides.is_empty:
            summary["resolution"]["overrides"] = {"manual_texts": len(overrides.texts),
                                                  "accepted": len(overrides.accepted)}
    if export_summary is not None:
        summary["export"] = export_summary
    write_json(out / "summary.json", summary)
    text = _render_report(summary, needs_text, rows, policy, calibration)
    if candidates is not None:
        text += _render_compression(summary["compression"], candidates["ledger"])
    if "resolution" in summary:
        text += _render_phase4(summary["resolution"], summary.get("export"))
    (out / "report.md").write_text(text, encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any], needs_text: list[dict[str, Any]], rows: list[dict[str, Any]],
                   policy: TimelinePolicy, calibration: Calibration | None = None) -> str:
    by_id = {row["unit_id"]: row for row in rows}
    oracle = summary["oracle"]
    lines = [
        "# Фаза 2: речевые единицы, первый синтез, оракул, расписание",
        "",
        *_calibration_lines(calibration),
        f"Реплик: {summary['cue_count']}, единиц измерено: {summary['measured_units']}, ошибок синтеза: {summary['synthesis_errors']}.",
        f"Нормализация: {summary['normalization']}.",
        f"Суммарная речь при базовой скорости: {summary['speech_total_sec']} с; "
        f"медианный темп: {summary['median_syllables_per_sec']} слогов/с.",
        "",
        "## Речевые единицы",
        "",
        f"Единиц: {summary['unit_count']}, из них склеенных: {summary['merged_units']} "
        f"(поглощено реплик: {summary['merged_cues']}), сцен: {summary['scene_count']}.",
        f"Причины склейки: {summary['merge_reasons']}.",
        f"Сдвиг начала реплик внутри склеенных единиц (мс): медиана {summary['drift_p50_ms']}, "
        f"90-й процентиль {summary['drift_p90_ms']}, максимум {summary['drift_max_ms']}, "
        f"больше 1000 мс: {summary['drift_over_1000ms']}.",
        "",
        "## Оракул длительности (подогнан на этом прогоне)",
        "",
        f"секунды = {oracle['coefficients']['intercept']:.3f} + {oracle['coefficients']['syllables']:.4f} × слоги "
        f"+ {oracle['coefficients']['pauses']:.3f} × паузы",
        f"Средняя ошибка {oracle['mae_sec']:.3f} с, запас (90-й процентиль недооценки) {oracle['margin_sec']:.3f} с, "
        f"выборка {oracle['sample_count']} реплик.",
        "",
        "## Расписание",
        "",
        f"Политика: старт от −{policy.lead_ms} до +{policy.lag_ms} мс, пауза {policy.gap_ms} мс, "
        f"комфортный темп {policy.tempo_comfort_total}, потолок {policy.tempo_ceiling_total} (база {policy.base_rate})"
        f"{_syl_limits(calibration, policy)}.",
        f"Наивно не влезает в своё окно: {summary['naive_overflow_cues']} реплик.",
        f"После решателя: {summary['status_counts']}.",
        f"Нужно сокращать текст: {summary['needs_text_cues']} реплик, всего вырезать {summary['removed_total_sec']} с.",
        f"Дефицит по корзинам (с): {summary['deficit_histogram']}.",
        f"Классы текстовой работы: {summary['text_bands']} "
        "(easy ≤20 %, medium ≤40 %, hard ≤60 %, timing_limited — текстом не решается).",
        f"Темп: средний {summary['tempo_mean']}, 90-й процентиль {summary['tempo_p90']}; коллизий: {summary['collisions']}.",
        "",
        "## Крупнейшие дефициты",
        "",
        "| единица | реплики | окно, с | речь, с | темп | вырезать, с | слоги сейчас → цель | класс | текст |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for p in sorted(needs_text, key=lambda p: -p["removed_sec"])[:25]:
        row = by_id[p["unit_id"]]
        lines.append(f"| {p['unit_id']} | {'+'.join(p['cue_ids'])} | {(row['end'] - row['start']) / 1000:.2f} | {row['speech_sec']:.2f} | "
                     f"{p['total_speed']:.2f} | {p['removed_sec']:.2f} | {p['current_syllables']} → {p['target_syllables']} | "
                     f"{p['text_band']} | {row['display_text'].replace('|', '/')} |")
    failed = [row for row in rows if row["normalization"] != "ok" or row.get("error")]
    if failed:
        lines += ["", "## Реплики с проблемами подготовки", ""]
        for row in failed[:40]:
            lines.append(f"- {row['unit_id']}: {row['normalization']} {row.get('normalization_error') or row.get('error') or ''}"
                         f" — {row['display_text']}")
    return "\n".join(lines) + "\n"


def _resolution_summary(resolution: dict[str, Any], candidates: dict[str, Any],
                        export_summary: dict[str, Any] | None) -> dict[str, Any]:
    choices = resolution["choices"]
    placements = resolution["placements"]
    changed = [c for c in choices if c["chosen_rung"] is not None]
    estimates = {e["unit_id"]: {r["rung"]: r["estimated_sec"] for r in e["rungs"]} for e in candidates["ledger"]}
    paired = [(estimates.get(c["unit_id"], {}).get(c["chosen_rung"]), c["chosen_speech_sec"]) for c in changed]
    paired = [(est, measured) for est, measured in paired if est is not None]
    counts = export_summary["counts"] if export_summary else {}
    return {
        "passes": resolution["passes"],
        "status_counts": {name: sum(1 for p in placements if p["status"] == name)
                          for name in sorted({p["status"] for p in placements})},
        "changed": len(changed),
        "resolved": sum(1 for c in choices if c["status"] == "resolved"),
        "unresolved": sum(1 for c in choices if c["status"] == "unresolved"),
        "manual": sum(1 for c in choices if c["status"] == "manual"),
        "reviewed": sum(1 for c in choices if c.get("reviewed")),
        "rejected": sum(1 for c in choices if c["status"] == "rejected"),
        "timing_limited": sum(1 for c in choices if c["status"] == "timing_limited"),
        "forced": counts.get("forced"),
        "lossy": sum(1 for c in choices if c["lossy"]),
        "collisions": sum(1 for p in placements if p["collision_ms"] > 0),
        "measured_sec_saved": round(sum(c["original_speech_sec"] - c["chosen_speech_sec"] for c in changed), 2),
        "estimated_sec_total": round(sum(est for est, _ in paired), 2),
        "measured_sec_total": round(sum(measured for _, measured in paired), 2),
    }


def _render_phase4(resolution: dict[str, Any], export: dict[str, Any] | None) -> str:
    lines = ["", "## Фаза 4: измеренный пересчёт и экспорт", "",
             "| проход | заменено единиц | осталось с дефицитом | статусы |", "|---|---:|---:|---|"]
    for item in resolution["passes"]:
        lines.append(f"| {item['pass']} | {item['changed_units']} | {item['needs_text']} | {item['status_counts']} |")
    lines += ["",
              f"Итоговые статусы расписания: {resolution['status_counts']}; коллизий: {resolution['collisions']}.",
              f"Единиц с заменой текста: {resolution['changed']} (решено {resolution['resolved']}, "
              f"не решено {resolution['unresolved']}, отклонено {resolution['rejected']}, "
              f"ограничено таймингом {resolution['timing_limited']}); с потерей смысла: {resolution['lossy']}.",
              f"Оценка оракула для выбранных ступеней: {resolution['estimated_sec_total']} с, "
              f"измерение: {resolution['measured_sec_total']} с; реально сэкономлено: "
              f"{resolution['measured_sec_saved']} с (при базовой скорости).",
              f"Форсированный темп (остаток дефицита не решён текстом): {resolution['forced']}."]
    applied = resolution.get("overrides")
    if applied is not None:
        lines.append(f"Ручные правки применены: ручных текстов {applied['manual_texts']}, "
                     f"принятых единиц {applied['accepted']}.")
    if export is not None:
        lines += [f"Очередь на ручную проверку: {export['human_queue']} ед.",
                  f"Длительность дорожки: {export['track']['duration_sec']} с.",
                  f"Файлы экспорта ({export['directory']}): {', '.join(export['files'])}."]
    return "\n".join(lines) + "\n"


def _calibration_lines(calibration: Calibration | None) -> list[str]:
    if calibration is None:
        return ["Калибровка темпа: не использовалась (темп задан в единицах движка).", ""]
    return [f"Калибровка темпа: голос {calibration.engine.get('voice')}, цель {calibration.target_syl_per_sec:g} слог/с, "
            f"найденная скорость движка {calibration.rate:g}, измерено {calibration.measured_syl_per_sec:.2f} слог/с.", ""]


def _syl_limits(calibration: Calibration | None, policy: TimelinePolicy) -> str:
    if calibration is None:
        return ""
    target = calibration.target_syl_per_sec
    return (f"; в слогах/с: база {target:g}, комфорт {target * policy.comfort_factor:.2f}, "
            f"потолок {target * policy.ceiling_factor:.2f}")


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _render_compression(compression: dict[str, Any], ledger: list[dict[str, Any]]) -> str:
    stats = compression["llm_stats"]
    models = ", ".join(compression["llm_models"]) or "—"
    lines = [
        "", "## Сокращение текста", "",
        f"Единиц обработано: {compression['units']}; отправлено в модель: {compression['sent_to_model']}; "
        f"решено: {compression['resolved']}, не решено: {compression['unresolved']}, "
        f"отклонено: {compression['rejected']}, ограничено таймингом: {compression['timing_limited']}.",
        f"Выбранные ступени: {compression['rung_histogram']}; с потерей смысла (ступень 5): {compression['lossy']}.",
        f"Убрано слогов: {compression['syllables_removed']}; оценка выигрыша по оракулу: "
        f"{compression['estimated_sec_saved']} с (при базовой скорости).",
        f"Модель: {models}. Вызовов в этом запуске: {stats['calls_made']}, из кэша: {stats['cache_hits']}, "
        f"токенов: {stats['tokens_in']} на входе, {stats['tokens_out']} на выходе.",
        "", "| единица | реплики | ступень | статус | было → стало (слоги) | оригинал | результат |",
        "|---|---|---|---|---|---|---|"]
    for e in ledger:
        rung = "—" if e["chosen_rung"] is None else str(e["chosen_rung"]) + (" (lossy)" if e["lossy"] else "")
        lines.append(f"| {e['unit_id']} | {'+'.join(e['cue_ids'])} | {rung} | {e['status']} | "
                     f"{e['original_syllables']} → {e['chosen_syllables']} | {_cell(e['original'])} | "
                     f"{_cell(e['chosen_text'])} |")
    queue = [e for e in ledger if e["status"] != "resolved"]
    lines += ["", "### Очередь на ручную проверку", ""]
    if not queue:
        lines.append("Пусто.")
        return "\n".join(lines) + "\n"
    lines += ["| единица | реплики | статус | цель, с | лучшая оценка, с | дефицит, с | лучший вариант |",
              "|---|---|---|---:|---:|---:|---|"]
    for e in queue:
        best = e["chosen_estimated_sec"] if e["chosen_estimated_sec"] is not None else e["speech_sec"]
        text = e["chosen_text"] if e["chosen_rung"] is not None else e["original"]
        lines.append(f"| {e['unit_id']} | {'+'.join(e['cue_ids'])} | {e['status']} | {e['target_base_sec']:.2f} | "
                     f"{best:.2f} | {max(0.0, best - e['target_base_sec']):.2f} | {_cell(text)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI

def run(source: Path, out: Path, config: dict[str, Any], *, until: str = "report", workers: int | None = None,
        limit: int | None = None, engine_factory: Callable[[dict[str, Any]], Engine] = build_engine,
        llm_factory: Callable[[LLMConfig], LLMClient] = LLMClient,
        stretch: StretchFn = stretch_clip, overrides_path: Path | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    if overrides_path is None and (out / OVERRIDES_FILE).is_file():
        overrides_path = out / OVERRIDES_FILE
    overrides = load_overrides(overrides_path)
    if not overrides.is_empty:
        log.info("overrides: %d manual texts, %d accepted units from %s", len(overrides.texts),
                 len(overrides.accepted), overrides_path)
    cues = stage_parse(source, out, limit)
    log.info("parse: %d cues", len(cues["cues"]))
    if until == "parse":
        return {"stage": "parse", "cues": len(cues["cues"])}
    units = stage_units(cues, MergePolicy.from_dict(config.get("units", {})), out)
    log.info("units: %d units", len(units["units"]))
    overrides.check_units(unit["unit_id"] for unit in units["units"])
    if until == "units":
        return {"stage": "units", "units": len(units["units"])}
    synthesis = config.get("synthesis", {})
    cache_dir = Path(str(synthesis.get("cache_dir", "output/tts_cache")))
    worker_count = workers or int(synthesis.get("workers", 1))
    calibration = stage_calibrate(config["engine"], config["policy"], cache_dir, worker_count, out)
    if calibration is None:
        policy = TimelinePolicy.from_dict(config["policy"])
        engine = engine_factory(config["engine"])
    else:
        policy = derive_timeline_policy(config["policy"], calibration)
        engine = engine_factory({**config["engine"], "base_rate": calibration.rate})
    if until == "calibrate":
        return {"stage": "calibrate", "rate": calibration.rate if calibration else None}
    durations = stage_synth0(units, engine, out, worker_count, cache_dir)
    if until == "synth0":
        return {"stage": "synth0", "rows": len(durations["rows"])}
    oracle = stage_oracle(durations, out)
    log.info("oracle: %s", oracle.to_dict())
    if until == "oracle":
        return {"stage": "oracle", **oracle.to_dict()}
    schedule_artifact = stage_schedule(durations, oracle, policy, cues, out)
    if until == "schedule":
        return {"stage": "schedule", "placements": len(schedule_artifact["placements"])}
    compression = CompressionPolicy.from_dict(config.get("compression", {}))
    client = llm_factory(LLMConfig.from_dict(config.get("llm", {})))
    annotations = stage_annotate(units, durations, schedule_artifact, compression, client, out)
    if until == "annotate":
        return {"stage": "annotate", "annotations": len(annotations["annotations"])}
    candidates = stage_compress(units, durations, schedule_artifact, oracle, annotations, compression, client, out)
    if until == "compress":
        return {"stage": "compress", **candidates["summary"]}
    durations1 = stage_measure(candidates, durations, engine, out, worker_count, cache_dir, overrides)
    if until == "measure":
        return {"stage": "measure", "rows": len(durations1["rows"]),
                "errors": sum(1 for row in durations1["rows"] if row.get("error"))}
    verify_config = config.get("verify", {})
    verify_policy = VerifyPolicy.from_dict(verify_config)
    verify_client = llm_factory(LLMConfig.from_dict({**config.get("llm", {}), **verify_config.get("llm", {})}))
    verified = stage_verify(candidates, durations1, verify_policy, verify_client, out)
    if until == "verify":
        return {"stage": "verify", "verdicts": len(verified["verdicts"]),
                "rejected": sum(1 for v in verified["verdicts"] if not v["ok"])}
    resolution = stage_resolve(units, cues, durations, durations1, verified, candidates, schedule_artifact, policy,
                               oracle, compression, out, overrides)
    if until == "resolve":
        return {"stage": "resolve", "passes": resolution["passes"]}
    export_summary = stage_export(cues, units, durations, durations1, resolution, verified, candidates, policy, oracle,
                                  calibration, config, engine, _llm_identity(verify_client), out, stretch, overrides)
    if until == "export":
        return {"stage": "export", **export_summary}
    return stage_report(cues, units, durations, oracle, schedule_artifact, policy, out, candidates, calibration,
                        resolution, export_summary, overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voiceover v3: measure, fit oracle, schedule, shorten text, report")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run stages up to --until")
    run_parser.add_argument("source", type=Path, help="Input .srt or JSON cue list")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, default=Path("voiceover_config.json"))
    run_parser.add_argument("--until", choices=STAGES, default="report")
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--limit", type=int, default=None, help="Only the first N cues (smoke runs)")
    run_parser.add_argument("--overrides", type=Path, default=None,
                            help="Manual overrides JSON (default: <out>/overrides.json when it exists)")
    run_parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.source, args.out, load_config(args.config), until=args.until, workers=args.workers,
                 limit=args.limit, overrides_path=args.overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
