"""End-to-end SRT -> analyzed JSON -> normalized JSON -> TTS -> post-processing pipeline.

Run (from repo root):
    uv run -m pipeline input.srt --output-dir output_run
    uv run -m pipeline input.srt --prepare-text --output-dir text_run

The pipeline is intentionally composed from the existing project modules:
    - utils.srt_to_json
    - utils.analyze_text (wrapped by utils.text_analysis)
    - utils.text_normalizer (wrapped by utils.text_normalization)
    - synthesize.synthesize_cli / synthesize.synthesize_batch
    - utils.reduce_silence
    - utils.audio_analysis
    - utils.audio_resample

Config defaults live in pipeline_config.json or pipeline_config_vosk.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.srt_to_json import SRTParseError, parse_srt_file
from utils.text_analysis import analyze_subtitles_items
from utils.pipeline_text import apply_reviewed_text, prepare_text, resolve_speed_policy


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    base_dir = Path(__file__).parent

    defaults_path = base_dir / "pipeline_config.json"
    if not defaults_path.exists():
        defaults_path = base_dir / "pipeline_config_vosk.json"

    if not defaults_path.exists():
        raise FileNotFoundError(
            "No pipeline defaults found. Expected pipeline_config.json or pipeline_config_vosk.json in repo root."
        )

    defaults = _load_json(defaults_path)
    if not isinstance(defaults, dict):
        raise ValueError(f"{defaults_path.name} must be a JSON object")

    if config_path and config_path.exists():
        user = _load_json(config_path)
        if not isinstance(user, dict):
            raise ValueError("Config file must be a JSON object")
        _deep_update(defaults, user)
        # Old configs must not inherit a newly enabled audio policy or mixing.
        for key in ("silero_pauses", "silero_edges", "mix"):
            if key not in user:
                defaults[key] = {"enabled": False}

    return defaults


def validate_complete_mix(items: List[Dict[str, Any]], source: List[Dict[str, Any]], segments: Path,
                          report_path: Path, *, file_prefix: str = "",
                          pauses_path: Path | None = None, synthesis_rate: float = 1.25,
                          max_total_speed: float = 1.5, allow_speedup: bool = True,
                          prefer_ffmpeg: bool = True, edges_path: Path | None = None,
                          edge_items: List[Dict[str, Any]] | None = None) -> None:
    """Recheck full source coverage, current processed WAV hashes and PCM fits."""
    import hashlib
    import soundfile as sf
    from utils.shortening_plan import PlannerConfig, plan_subtitles
    from utils.pipeline_text import SpeedPolicy, json_sha256

    report = _load_json(report_path)
    context = json_sha256({"items": items, "synthesis_rate": synthesis_rate,
                          "max_total_speed": max_total_speed, "file_prefix": file_prefix,
                          "ext": ".wav", "allow_speedup": allow_speedup, "prefer_ffmpeg": prefer_ffmpeg})
    if report.get("context_sha256") != context:
        raise ValueError("Full-track export blocked: stale source/text/postprocess timing context")
    rows = report.get("segments", [])
    source_coordinates = [(row.get("index"), row.get("start"), row.get("end")) for row in source]
    if (not source_coordinates or len({row[0] for row in source_coordinates}) != len(source_coordinates)
            or [(row.get("index"), row.get("start"), row.get("end")) for row in items] != source_coordinates
            or [(row.get("index"), row.get("start"), row.get("end")) for row in rows] != source_coordinates
            or report.get("state") != "complete" or report.get("failed_count") != 0):
        raise ValueError("Full-track export blocked: incomplete source/current measured timing")
    caps = {row.get("max_total_speed") for row in rows}
    bases = {row.get("synthesis_rate") for row in rows}
    if len(caps) != 1 or len(bases) != 1:
        raise ValueError("Full-track export blocked: inconsistent speed policy")
    policy = SpeedPolicy(synthesis_rate, max_total_speed)
    if caps != {policy.max_total_speed} or bases != {policy.base_speech_rate}:
        raise ValueError("Full-track export blocked: stale speed policy")
    plans = plan_subtitles(items, PlannerConfig(max_speed=policy.max_total_speed))
    for position, (item, row, plan) in enumerate(zip(items, rows, plans)):
        path = segments / f"{file_prefix}{item['index']}.wav"
        if not path.is_file():
            raise ValueError(f"Full-track export blocked: missing expected WAV {path}")
        with path.open("rb") as stream:
            current_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        info = sf.info(path)
        if (row.get("status") not in {"fits", "accelerated_fits"} or plan.status == "invalid"
                or row.get("available_ms") != plan.available_ms
                or report.get("wav_sha256", {}).get(str(item["index"])) != current_hash
                or Path(row["audio_path"]).resolve() != path.resolve()
                or info.frames <= 0 or info.frames * 1000 > plan.available_ms * info.samplerate
                or row.get("duration_after_sec") != info.frames / info.samplerate
                or row.get("effective_total_speed", float("inf")) > policy.max_total_speed):
            raise ValueError(f"Full-track export blocked: stale/unfit current WAV {path}")
        if position + 1 < len(items) and (item["start"] * info.samplerate + info.frames * 1000
                                         > items[position + 1]["start"] * info.samplerate):
            raise ValueError(f"Full-track export blocked: timeline overlap after {item['index']}")
    if pauses_path is not None:
        from utils.silero_pauses import digest, runtime_identity
        receipts = _load_json(pauses_path)
        if len(receipts) != len(items):
            raise ValueError("Full-track export blocked: incomplete pause processing")
        identity = runtime_identity()
        for item, receipt in zip(items, receipts):
            if (receipt.get("status") != "complete" or receipt.get("identity") != identity
                    or Path(receipt["source_path"]).name != f"{item['index']}.pretempo.wav"
                    or Path(receipt["output_path"]).name != f"{item['index']}.pauses.wav"
                    or digest(Path(receipt["source_path"])) != receipt["source_sha256"]
                    or digest(Path(receipt["output_path"])) != receipt["output_sha256"]):
                raise ValueError("Full-track export blocked: stale pause processing provenance")
    if edges_path is not None:
        from utils.analyze_text import join_text_lines
        from utils.silero_pauses import validate_edge_receipt
        if pauses_path is None or edge_items is None or [(x.get("index"), x.get("start"), x.get("end"))
                for x in edge_items] != source_coordinates:
            raise ValueError("Full-track export blocked: missing edge source/processing binding")
        edges = _load_json(edges_path)
        if len(edges) != len(items):
            raise ValueError("Full-track export blocked: incomplete edge processing")
        edge_plans = plan_subtitles(edge_items, PlannerConfig(max_speed=policy.max_total_speed))
        for item, plan, internal, edge in zip(edge_items, edge_plans, _load_json(pauses_path), edges):
            if Path(edge["output_path"]).name != f"{item['index']}.edges.wav":
                raise ValueError("Full-track export blocked: unexpected edge WAV")
            validate_edge_receipt(edge, internal, available_ms=plan.available_ms,
                                  chosen_text=join_text_lines(item.get("text", "")))


def run_pipeline(
    srt_path: Path,
    *,
    output_dir: Path,
    config: Dict[str, Any],
    prepare_text_only: bool = False,
    reviewed_text_path: Path | None = None,
) -> Path:
    """Return the offline handoff or the final segments folder, depending on stage."""

    policy = resolve_speed_policy(config)
    a_cfg = config.get("analysis", {})
    cps = float(a_cfg.get("avg_chars_per_sec", 13.0))
    if "max_mismatch_ratio" in a_cfg and a_cfg["max_mismatch_ratio"] != policy.max_total_speed:
        logging.warning("analysis.max_mismatch_ratio=%s overridden by the one total cap %s",
                        a_cfg["max_mismatch_ratio"], policy.max_total_speed)
    if prepare_text_only and reviewed_text_path is not None:
        raise ValueError("--prepare-text and --reviewed-text are separate stages")
    if not prepare_text_only and (config.get("silero_pauses", {}).get("enabled", False)
                                 or config.get("silero_edges", {}).get("enabled", False)):
        from utils.silero_pauses import validate_profile
        validate_profile(config)
    subs: List[Dict[str, Any]] = parse_srt_file(str(srt_path))
    prepared = prepare_text(subs, policy, cps)
    chosen = subs
    review_report: List[Dict[str, Any]] | None = None
    if reviewed_text_path is not None:
        chosen, review_report = apply_reviewed_text(subs, _load_json(reviewed_text_path), policy, cps)

    json_dir = output_dir / "json"
    segments_dir = output_dir / "segments"
    if not prepare_text_only and segments_dir.exists() and any(segments_dir.iterdir()):
        raise ValueError("Segments output is not empty; use a new output directory to avoid stale audio/rate metadata")
    if not prepare_text_only and str(config.get("segments", {}).get("output_naming", "index")) != "index":
        raise ValueError("Bounded pipeline requires segments.output_naming=index")
    if not prepare_text_only and str(config.get("segments", {}).get("ext", ".wav")) != ".wav":
        raise ValueError("Bounded Vosk pipeline requires segments.ext=.wav")
    if config.get("mix", {}).get("enabled", False) and config.get("segments", {}).get("file_prefix", ""):
        raise ValueError("Complete-track mixer requires empty segments.file_prefix")
    indices = [item["index"] for item in subs]
    if len(set(indices)) != len(indices):
        raise ValueError("Bounded pipeline requires unique SRT indices")

    # 1) SRT -> JSON
    subs_json = json_dir / "subs.json"
    if subs_json.exists() and _load_json(subs_json) != subs:
        raise ValueError("Output already contains another source; use a new output directory")
    _save_json(subs_json, subs)
    if prepare_text_only:
        handoff_path = json_dir / "native_text.json"
        if handoff_path.exists() and _load_json(handoff_path) != prepared:
            raise ValueError("Existing handoff differs; use a new output directory")
        _save_json(handoff_path, prepared)
        logging.info("Prepared %d original cues offline; no model/normalizer/TTS calls: %s", len(subs), handoff_path)
        return handoff_path
    segments_dir.mkdir(parents=True, exist_ok=True)
    if review_report is not None:
        _save_json(json_dir / "reviewed_text_report.json", review_report)
    # Keep source subs.json untouched by the in-place analyzer.
    chosen = copy.deepcopy(chosen)
    edge_items = copy.deepcopy(chosen)  # Bind words before RUNorm/RUAccent/model stress.
    _save_json(json_dir / "subs_chosen.json", chosen)
    logging.info("Vosk bounded policy: fixed synthesis %.3fx, total cap %.3fx; analyzer cap derives from it",
                 policy.base_speech_rate, policy.max_total_speed)
    if bool(config.get("tts", {}).get("use_analysis_speech_rate", False)):
        logging.warning("Analysis-driven TTS rate disabled by bounded policy; acceleration uses measured audio only")

    # 2) Analyze
    analysis_result = analyze_subtitles_items(
        chosen,
        avg_chars_per_sec=cps,
        max_mismatch_ratio=policy.max_total_speed,
    )
    analyzed_items = analysis_result["items"]
    analyzed_json = json_dir / "subs_analyzed.json"
    _save_json(analyzed_json, analyzed_items)

    critical_items = analysis_result.get("critical_items") or []
    if critical_items:
        logging.warning("Critical subtitles: %d", len(critical_items))
        if bool(a_cfg.get("fail_on_critical", False)):
            raise SystemExit(1)

    # 3) Normalize RUNorm+RUAccent
    # Lazy import: RUAccent pulls transformers/torch which is slow to import.
    from utils.text_normalization import normalize_subtitles_items

    n_cfg = config.get("normalization", {})
    normalized_items = normalize_subtitles_items(
        analyzed_items,
        norm_device=str(n_cfg.get("norm_device", "cpu")),
        accent_device=str(n_cfg.get("accent_device", "cpu")),
        batch_size=int(n_cfg.get("batch_size", 10)),
    )
    normalized_json = json_dir / "subs_normalized.json"
    _save_json(normalized_json, normalized_items)

    # 4) Synthesize
    # Lazy import: Vosk TTS/onnxruntime can be heavy.
    from synthesize.synthesize_batch import synthesize_json_lines

    tts_cfg = config.get("tts", {})
    seg_cfg = config.get("segments", {})

    model_path = str(tts_cfg.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("tts.model_path must be set (path to Vosk TTS model dir)")

    output_naming = str(seg_cfg.get("output_naming", "index"))
    file_prefix = str(seg_cfg.get("file_prefix", ""))

    synthesize_json_lines(
        json_path=str(normalized_json),
        model_path=model_path,
        device=str(tts_cfg.get("device", "cpu")),
        output_folder=str(segments_dir),
        file_prefix=file_prefix,
        voice=int(tts_cfg.get("voice", 0)),
        voice_male=tts_cfg.get("voice_male"),
        voice_female=tts_cfg.get("voice_female"),
        gender_key=str(tts_cfg.get("gender_key", "gender")),
        default_speech_rate=policy.base_speech_rate,
        use_analysis_speech_rate=False,
        base_speech_rate=policy.base_speech_rate,
        max_extra_pct=0.0,
        output_naming=output_naming,
        text_key="text",
        output_sample_rate=tts_cfg.get("output_sample_rate"),
    )

    # 5) Reduce silence
    from utils.reduce_silence import reduce_silence_in_dir

    rs_cfg = config.get("reduce_silence", {})
    if bool(rs_cfg.get("enabled", True)):
        reduce_silence_in_dir(
            segments_dir,
            in_place=True,
            pattern=f"{file_prefix}*{str(seg_cfg.get('ext', '.wav'))}",
            silence_dbfs=float(rs_cfg.get("silence_dbfs", -35.0)),
            threshold_sec=float(rs_cfg.get("threshold_sec", 0.5)),
            target_sec=float(rs_cfg.get("target_sec", 0.2)),
        )

    # 6) Resample before measuring final timing (and binding the report to WAV hashes).
    from utils.audio_resample import resample_wav_dir

    r_cfg = config.get("resample", {})
    if bool(r_cfg.get("enabled", True)):
        resample_wav_dir(
            segments_dir,
            target_sr=int(r_cfg.get("target_sr", 44100)),
            pattern=f"{file_prefix}*{str(seg_cfg.get('ext', '.wav'))}",
            in_place=True,
        )

    # 7) Fixed internal pause policy, before any timing decision or tempo.
    pauses_path: Path | None = None
    edges_path: Path | None = None
    if config.get("silero_pauses", {}).get("enabled", False):
        from utils.silero_pauses import process_segments
        pauses_path = json_dir / "silero_pauses.json"
        edges: List[Dict[str, Any]] | None = [] if config.get("silero_edges", {}).get("enabled", False) else None
        _save_json(pauses_path, process_segments(normalized_items, segments_dir,
            output_dir / "pause_stage", file_prefix=file_prefix, budget_items=edge_items,
            max_total_speed=policy.max_total_speed, edge_receipts=edges))
        if edges is not None:
            edges_path = json_dir / "silero_edges.json"
            _save_json(edges_path, edges)

    # 8) Actual timing, with only the required remaining pitch-preserving tempo.
    from utils.audio_analysis import check_and_speedup_segments

    aa_cfg = config.get("audio_analysis", {})
    results = check_and_speedup_segments(
        normalized_items, segments_dir, file_prefix=file_prefix,
        ext=str(seg_cfg.get("ext", ".wav")), in_place=True,
        synthesis_rate=policy.base_speech_rate, max_total_speed=policy.max_total_speed,
        prefer_ffmpeg=bool(aa_cfg.get("prefer_ffmpeg", True)),
        allow_speedup=bool(aa_cfg.get("enabled", True)),
        report_path=json_dir / "audio_timing.json",
    )
    failed = sum(result.status not in {"fits", "accelerated_fits"} for result in results)
    logging.info("Audio stage complete: %d/%d cues unresolved/invalid/missing (not truncated)", failed, len(results))

    mix_cfg = config.get("mix", {})
    if mix_cfg.get("enabled", False):
        if failed:
            logging.warning("Full-track export blocked: %d unresolved/invalid/missing cues; segments retained", failed)
        else:
            from utils.audio_mixer import mix_segments
            validate_complete_mix(normalized_items, subs, segments_dir, json_dir / "audio_timing.json",
                                  pauses_path=pauses_path, synthesis_rate=policy.base_speech_rate,
                                  max_total_speed=policy.max_total_speed,
                                  allow_speedup=bool(aa_cfg.get("enabled", True)),
                                  prefer_ffmpeg=bool(aa_cfg.get("prefer_ffmpeg", True)),
                                  edges_path=edges_path, edge_items=edge_items)
            mix_dir = output_dir / "mix"
            mix_dir.mkdir(parents=True, exist_ok=True)
            mixed = mix_segments(subs_json, segments_dir, mix_dir / "voiceover",
                                 sample_rate=int(mix_cfg.get("sample_rate", 44100)),
                                 channels=int(mix_cfg.get("channels", 1)), missing_policy="error",
                                 transcode_to=None)
            logging.info("Verified complete timeline track: %s", mixed)

    return segments_dir


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full SRT -> Vosk TTS pipeline")
    p.add_argument("srt", help="Input .srt file")
    p.add_argument("--output-dir", default="pipeline_out", help="Output folder")
    p.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Pipeline config JSON (merged over defaults)",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--prepare-text", action="store_true", help="Write offline native text handoff and stop before normalization/TTS")
    p.add_argument("--reviewed-text", type=Path, help="Explicitly use a source/policy-bound native reviewed JSON")
    p.add_argument("--base-speech-rate", type=float, help="Fixed Vosk synthesis rate (default 1.25)")
    p.add_argument("--max-total-speed", type=float, help="Single synthesis x tempo cap, also used by the analyzer (1..1.5)")
    p.add_argument("--mix", action="store_true", help="Assemble complete verified track at original subtitle starts")
    p.add_argument("--no-silero-pauses", action="store_true", help="Disable both reviewed Vosk internal and edge pause stages")
    p.add_argument("--no-silero-edges", action="store_true", help="Keep internal cuts but disable conditional edge cuts")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    srt_path = Path(args.srt)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    cfg = load_pipeline_config(Path(args.config))
    if args.mix:
        cfg.setdefault("mix", {})["enabled"] = True
    if args.no_silero_pauses:
        cfg.setdefault("silero_pauses", {})["enabled"] = False
        cfg.setdefault("silero_edges", {})["enabled"] = False
    if args.no_silero_edges:
        cfg.setdefault("silero_edges", {})["enabled"] = False
    if args.base_speech_rate is not None:
        cfg.setdefault("tts", {})["base_speech_rate"] = args.base_speech_rate
    if args.max_total_speed is not None:
        cfg.setdefault("tts", {})["max_total_speed"] = args.max_total_speed
    try:
        segments_dir = run_pipeline(srt_path, output_dir=Path(args.output_dir), config=cfg,
                                    prepare_text_only=args.prepare_text, reviewed_text_path=args.reviewed_text)
    except SRTParseError as e:
        logging.error("%s", str(e))
        raise SystemExit(1)

    if not args.prepare_text:
        report_path = Path(args.output_dir) / "json" / "audio_timing.json"
        report = _load_json(report_path)
        print(f"Measured audio: unresolved/invalid/missing={report['failed_count']}/{len(report['segments'])}; report={report_path}")
    print(str(segments_dir))


if __name__ == "__main__":  # pragma: no cover
    main()
