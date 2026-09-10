"""Optional, diagnostic audio timing verification for shortening trials."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from utils.vosk_text import normalize_vosk_text

MAX_SPEECH_RATE = 1.5
RAW_PREPROCESSING = "raw_no_ruaccent"
PUNCTUATION_PREPROCESSING = "vosk_punctuation_v1_no_ruaccent"
PREPROCESSING = PUNCTUATION_PREPROCESSING
TEXT_MODES = ("raw", "punctuation")
SynthesisCallable = Callable[..., Any]


_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_API_KEY_PATTERN = re.compile(r"(?i)(?<![\w-])sk[-_][A-Za-z0-9_-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically so an interrupted check remains readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "cue"


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _budget(record: dict[str, Any]) -> tuple[float, float, float, float]:
    budget = record.get("budget")
    budget = budget if isinstance(budget, dict) else {}
    available_ms = _number(budget.get("available_ms"))
    start = _number(budget.get("start"))
    end = _number(budget.get("end"))
    source_ms = max(0.0, end - start)
    pause_ms = max(0.0, available_ms - source_ms)
    return available_ms, source_ms, pause_ms, available_ms


def _source_report(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("source_report")
    if not isinstance(source, dict):
        source = report.get("source_manifest")
    if not isinstance(source, dict):
        return {"absolute_path": None, "sha256": None}
    return {
        "absolute_path": source.get("absolute_path") or source.get("path"),
        "sha256": source.get("sha256"),
    }


def _source_input(report: dict[str, Any]) -> tuple[str | None, str | None]:
    source = report.get("source_manifest")
    if not isinstance(source, dict):
        return None, None
    source_path = source.get("absolute_path") or source.get("path")
    absolute_path: str | None = None
    if isinstance(source_path, (str, Path)):
        absolute_path = str(Path(source_path).expanduser().resolve())
    source_sha256 = source.get("sha256")
    return absolute_path, source_sha256 if isinstance(source_sha256, str) else None


def _safe_error_message(value: object) -> str:
    message = str(value)
    message = _BEARER_PATTERN.sub("Bearer [REDACTED]", message)
    message = _API_KEY_PATTERN.sub("[REDACTED]", message)
    return message[:500]


def _error_payload(exc: Exception) -> dict[str, str | None]:
    cause = exc.__cause__ or exc.__context__
    return {
        "type": type(exc).__name__,
        "message": _safe_error_message(exc),
        "cause_type": type(cause).__name__ if cause is not None else None,
        "cause_message": _safe_error_message(cause) if cause is not None else None,
    }


def _audio_file_info(path: Path, elapsed_cpu_sec: float, *, reused: bool = False) -> dict[str, Any]:
    # soundfile is intentionally imported only while verification executes.
    import soundfile as sf

    info = sf.info(str(path))
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    if sample_rate <= 0 or frames < 0:
        raise ValueError("invalid WAV metadata")
    duration = frames / sample_rate
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid WAV duration")
    return {
        "path": str(path),
        "duration_sec": duration,
        "frames": frames,
        "samplerate": sample_rate,
        "elapsed_cpu_sec": elapsed_cpu_sec,
        "reused": reused,
    }


def _invoke_synthesis(
    synthesis_fn: SynthesisCallable,
    text: str,
    path: Path,
    *,
    voice: int,
    speech_rate: float,
    model_path: Path,
    synth_instance: Any,
) -> Any:
    """Call the real or injected synthesizer without forcing test doubles' shape."""
    supplied: dict[str, Any] = {
        "text": text,
        "voice": voice,
        "speech_rate": speech_rate,
        "model_path": str(model_path),
        "output_path": str(path),
        "synth": synth_instance,
    }
    try:
        signature = inspect.signature(synthesis_fn)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        return synthesis_fn(**supplied)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return synthesis_fn(**supplied)
    accepted = {
        name: value for name, value in supplied.items() if name in signature.parameters
    }
    return synthesis_fn(**accepted)


def verify_audio(
    report: dict[str, Any],
    output_dir: Path,
    *,
    model_path: Path,
    voice: int = 8,
    speech_rate: float = 1.5,
    limit: int = 6,
    synthesis_fn: SynthesisCallable | None = None,
    text_mode: str = "punctuation",
) -> dict[str, Any]:
    """Synthesize selected trial records and measure their actual WAV durations.

    Only accepted records and source-position-stable ``short_timing`` records are
    considered.  This function never changes the supplied trial report.
    """
    if not isinstance(report, dict):
        raise TypeError("report must be an object")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist or is not a directory: {model_path}")
    if isinstance(speech_rate, bool):
        raise ValueError("speech_rate must be finite and > 0")
    try:
        requested_rate = float(speech_rate)
    except (TypeError, ValueError):
        raise ValueError("speech_rate must be finite and > 0") from None
    if not math.isfinite(requested_rate) or requested_rate <= 0:
        raise ValueError("speech_rate must be finite and > 0")
    if text_mode not in TEXT_MODES:
        raise ValueError("text_mode must be 'raw' or 'punctuation'")
    effective_rate = min(requested_rate, MAX_SPEECH_RATE)
    preprocessing = (
        PREPROCESSING if text_mode == "punctuation" else RAW_PREPROCESSING
    )

    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("report.records must be a list")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("status") in {"accepted", "short_timing"}
    ]
    selected.sort(key=lambda record: (_number(record.get("position"), 0), str(record.get("cue_id", ""))))
    selected = selected[:limit]

    output_dir.mkdir(parents=True)
    wav_dir = output_dir / "wav"
    wav_dir.mkdir()
    source = _source_report(report)
    source_input_path, source_input_sha256 = _source_input(report)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source_report": source,
        "source_input_path": source_input_path,
        "source_input_sha256": source_input_sha256,
        "model_path": str(model_path),
        "voice": int(voice),
        "speech_rate": effective_rate,
        "requested_speech_rate": requested_rate,
        "speech_rate_cap": MAX_SPEECH_RATE,
        "preprocessing": preprocessing,
        "synthetic_fallback": False,
        "speed_postprocess": False,
        "silence_trim": False,
        "semantic_quality_judgment": False,
        "records": [],
        "summary": {
            "tested_accepted": sum(record.get("status") == "accepted" for record in selected),
            "actual_fits": 0,
            "short_final_overlap_count": 0,
            "long_final_overflow_count": 0,
            "fail": 0,
            "short_overlap": 0,
            "physical_file_count": 0,
        },
    }
    checkpoint = output_dir / "audio_report.json"
    _write_json(checkpoint, metadata)

    synth_instance: Any = None
    if synthesis_fn is None:
        # These imports are deliberately execution-only: importing this module
        # must not load Vosk or attempt a model download.
        from synthesize.synthesize import create_synth, synthesize_text

        synth_instance = create_synth(str(model_path), device="cpu")
        synthesis_fn = synthesize_text

    cache: dict[str, dict[str, Any]] = {}

    def synthesize_one(tts_text: str, role: str, record: dict[str, Any]) -> dict[str, Any]:
        if tts_text in cache:
            cached_path = output_dir / cache[tts_text]["path"]
            if not cached_path.exists():
                raise FileNotFoundError(f"cached WAV is missing: {cached_path}")
            reused = dict(cache[tts_text])
            reused["reused"] = True
            reused["reused_from"] = cache[tts_text]["path"]
            reused["elapsed_cpu_sec"] = 0.0
            return reused
        position = _safe_name(record.get("position", "unknown"))
        cue_id = _safe_name(record.get("cue_id", position))
        path = wav_dir / f"{position}_{cue_id}_{role}.wav"
        started_cpu = time.process_time()
        result = _invoke_synthesis(
            synthesis_fn,
            tts_text,
            path,
            voice=int(voice),
            speech_rate=effective_rate,
            model_path=model_path,
            synth_instance=synth_instance,
        )
        elapsed_cpu = max(0.0, time.process_time() - started_cpu)
        returned = Path(result) if isinstance(result, (str, Path)) else path
        # The requested output path is the reproducible path for the evidence.
        # A synthesizer that fails to create it must not produce an audio claim.
        if not path.exists() and returned != path and returned.exists():
            raise FileNotFoundError("synthesizer did not write requested WAV path")
        if not path.exists():
            raise FileNotFoundError(f"generated WAV is missing: {path}")
        info = _audio_file_info(path, elapsed_cpu)
        info["path"] = str(path.relative_to(output_dir))
        cache[tts_text] = dict(info)
        return info

    def text_fields(display_text: str) -> dict[str, Any]:
        tts_text = display_text if text_mode == "raw" else normalize_vosk_text(display_text)
        return {
            "display_text": display_text,
            "tts_text": tts_text,
            "was_normalized": tts_text != display_text,
        }

    def attach_text_fields(audio: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(audio)
        enriched.update(fields)
        return enriched

    for record in selected:
        cue_id = record.get("cue_id")
        status = record.get("status")
        result: dict[str, Any] = {
            "cue_id": cue_id,
            "position": record.get("position"),
            "status": status,
            "preprocessing": preprocessing,
            "original": None,
            "final": None,
        }
        record_added = False
        try:
            original = record.get("original")
            if not isinstance(original, str) or not original.strip():
                raise ValueError("original text is missing")
            original_fields = text_fields(original)
            result.update(
                {
                    "original_text": original,
                    "original_display_text": original_fields["display_text"],
                    "original_tts_text": original_fields["tts_text"],
                    "original_was_normalized": original_fields["was_normalized"],
                }
            )
            available_ms, source_ms, pause_ms, effective_ms = _budget(record)
            original_audio = synthesize_one(
                original_fields["tts_text"], "original", record
            )
            result["original"] = attach_text_fields(original_audio, original_fields)
            metadata["records"].append(result)
            record_added = True
            _write_json(checkpoint, metadata)
            final_audio: dict[str, Any] | None = None
            final_text: str | None = None
            final_fields: dict[str, Any] | None = None
            if status == "accepted":
                candidate = record.get("final_text")
                if not isinstance(candidate, str):
                    candidate = record.get("candidate")
                if not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError("accepted final text is missing")
                final_text = candidate
                final_fields = text_fields(candidate)
                result.update(
                    {
                        "final_text": candidate,
                        "final_display_text": final_fields["display_text"],
                        "final_tts_text": final_fields["tts_text"],
                        "final_was_normalized": final_fields["was_normalized"],
                    }
                )
                _write_json(checkpoint, metadata)
                final_audio = synthesize_one(
                    final_fields["tts_text"], "final", record
                )
                result["final"] = attach_text_fields(final_audio, final_fields)

            original_overflow = max(0.0, original_audio["duration_sec"] - effective_ms / 1000.0)
            final_overflow = (
                max(0.0, final_audio["duration_sec"] - effective_ms / 1000.0)
                if final_audio is not None
                else None
            )
            original_fits = original_overflow <= 0.0
            final_fits = final_overflow is not None and final_overflow <= 0.0
            result.update(
                {
                    "original_text": original,
                    "final_text": final_text,
                    "available_ms": available_ms,
                    "source_duration_ms": source_ms,
                    "source_duration_sec": source_ms / 1000.0,
                    "pause_ms": pause_ms,
                    "effective_window_ms": effective_ms,
                    "effective_window_sec": effective_ms / 1000.0,
                    "original_fits": original_fits,
                    "final_fits": final_fits if status == "accepted" else None,
                    "original_overflow_sec": original_overflow,
                    "overflow_sec": final_overflow if status == "accepted" else original_overflow,
                    "fits": final_fits if status == "accepted" else original_fits,
                }
            )
            if status == "accepted":
                assert final_audio is not None
                assert final_text is not None
                final_words_count = len(final_fields["display_text"].split()) if final_fields is not None else 0
                result["final_words_count"] = final_words_count
                result["final_chars_count"] = len(final_fields["display_text"]) if final_fields is not None else 0
                if final_fits:
                    result["timing_action"] = "fits"
                elif final_words_count <= 2:
                    result["timing_action"] = "short_phrase_overlap_pending"
                    metadata["summary"]["short_final_overlap_count"] += 1
                else:
                    result["timing_action"] = "needs_timing_revision"
                    metadata["summary"]["long_final_overflow_count"] += 1
                if final_fits:
                    metadata["summary"]["actual_fits"] += 1
            elif status == "short_timing":
                result["timing_action"] = (
                    "fits" if original_fits else "short_phrase_overlap_pending"
                )
                if not original_fits:
                    metadata["summary"]["short_overlap"] += 1
        except Exception as exc:
            metadata["summary"]["fail"] += 1
            result["error"] = _error_payload(exc)
            result["error_type"] = type(exc).__name__
            result["timing_action"] = "synthesis_error"
        if not record_added:
            metadata["records"].append(result)
        metadata["summary"]["physical_file_count"] = len(cache)
        _write_json(checkpoint, metadata)

    metadata["summary"]["physical_file_count"] = len(cache)
    _write_json(checkpoint, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify shortening trial audio durations")
    parser.add_argument("trial_report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/vosk-model-tts-ru-0.10-multi"))
    parser.add_argument("--voice", type=int, default=8)
    parser.add_argument("--speech-rate", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--text-mode", choices=list(TEXT_MODES), default="punctuation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    with args.trial_report.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("trial report must contain a JSON object")
    report_for_verify = dict(report)
    report_for_verify["source_report"] = {
        "absolute_path": str(args.trial_report.resolve()),
        "sha256": _sha256(args.trial_report),
    }
    result = verify_audio(
        report_for_verify,
        args.output_dir,
        model_path=args.model_path,
        voice=args.voice,
        speech_rate=args.speech_rate,
        limit=args.limit,
        text_mode=args.text_mode,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
