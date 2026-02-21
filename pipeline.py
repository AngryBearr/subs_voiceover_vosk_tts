"""End-to-end SRT -> analyzed JSON -> normalized JSON -> TTS -> post-processing pipeline.

Run (from repo root):
    uv run -m pipeline input.srt --output-dir output_run

The pipeline is intentionally composed from the existing project modules:
    - utils.srt_to_json
    - utils.analyze_text (wrapped by utils.text_analysis)
    - utils.text_normalizer (wrapped by utils.text_normalization)
    - synthesize.synthesize_cli / synthesize.synthesize_batch
    - utils.reduce_silence
    - utils.audio_analysis
    - utils.audio_resample

Config defaults live in pipeline_config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.srt_to_json import SRTParseError, parse_srt_file
from utils.text_analysis import analyze_subtitles_items


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

    return defaults


def run_pipeline(
    srt_path: Path,
    *,
    output_dir: Path,
    config: Dict[str, Any],
) -> Path:
    """Run pipeline and return path to the final segments folder."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # 1) SRT -> JSON
    subs: List[Dict[str, Any]] = parse_srt_file(str(srt_path))
    subs_json = json_dir / "subs.json"
    _save_json(subs_json, subs)

    # 2) Analyze
    a_cfg = config.get("analysis", {})
    analysis_result = analyze_subtitles_items(
        subs,
        avg_chars_per_sec=float(a_cfg.get("avg_chars_per_sec", 13.0)),
        max_mismatch_ratio=float(a_cfg.get("max_mismatch_ratio", 1.3)),
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
        default_speech_rate=float(tts_cfg.get("base_speech_rate", 1.25)),
        use_analysis_speech_rate=bool(tts_cfg.get("use_analysis_speech_rate", True)),
        base_speech_rate=float(tts_cfg.get("base_speech_rate", 1.25)),
        max_extra_pct=float(tts_cfg.get("max_extra_pct", 0.2)),
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

    # 6) Post audio analysis + speedup
    from utils.audio_analysis import check_and_speedup_segments

    aa_cfg = config.get("audio_analysis", {})
    if bool(aa_cfg.get("enabled", True)):
        check_and_speedup_segments(
            normalized_items,
            segments_dir,
            file_prefix=file_prefix,
            ext=str(seg_cfg.get("ext", ".wav")),
            overage_threshold_pct=float(aa_cfg.get("overage_threshold_pct", 0.1)),
            speedup_factor=float(aa_cfg.get("speedup_factor", 1.1)),
            in_place=True,
            prefer_ffmpeg=bool(aa_cfg.get("prefer_ffmpeg", True)),
        )

    # 7) Resample to 44.1kHz
    from utils.audio_resample import resample_wav_dir

    r_cfg = config.get("resample", {})
    if bool(r_cfg.get("enabled", True)):
        resample_wav_dir(
            segments_dir,
            target_sr=int(r_cfg.get("target_sr", 44100)),
            pattern=f"{file_prefix}*{str(seg_cfg.get('ext', '.wav'))}",
            in_place=True,
        )

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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    srt_path = Path(args.srt)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    cfg = load_pipeline_config(Path(args.config))
    try:
        segments_dir = run_pipeline(srt_path, output_dir=Path(args.output_dir), config=cfg)
    except SRTParseError as e:
        logging.error("%s", str(e))
        raise SystemExit(1)

    print(str(segments_dir))


if __name__ == "__main__":  # pragma: no cover
    main()
