"""
Reusable text-to-speech function for Vosk TTS.

Exposes synthesize_text(...) that loads a model and writes a .wav.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import from the installed/checked-out package
from vosk_tts import Model, Synth  # type: ignore
import wave


class SynthesisError(Exception):
    """Raised for synthesis-related failures with helpful context."""


def compute_speech_rate_from_analysis(
    analysis: dict | None,
    *,
    base_rate: float = 1.25,
    max_extra_pct: float = 0.20,
    round_ndigits: int = 2,
) -> float:
    """Compute a conservative speech_rate using subtitle text analysis.

    This follows the pipeline rule:
        - base_rate is the default starting point.
        - mismatch_ratio / extended_mismatch_ratio are treated as required
          acceleration coefficients to fit the text into the available window.
        - we accelerate only if needed (ratio > base_rate).
        - we cap acceleration to avoid overly fast speech.
    """

    def _as_pos_float(v: object) -> float | None:
        try:
            f = float(v)  # type: ignore[arg-type]
        except Exception:
            return None
        if f <= 0 or f != f:  # NaN
            return None
        return f

    if base_rate <= 0:
        raise ValueError("base_rate must be > 0")

    ratio: float | None = None
    if isinstance(analysis, dict):
        mr = _as_pos_float(analysis.get("mismatch_ratio"))
        emr = _as_pos_float(analysis.get("extended_mismatch_ratio"))
        candidates = [x for x in (mr, emr) if x is not None]
        if candidates:
            ratio = min(candidates)

    target = base_rate if ratio is None else max(base_rate, ratio)

    if max_extra_pct is None:
        capped = target
    else:
        max_rate = base_rate * (1.0 + float(max_extra_pct))
        capped = min(target, max_rate)

    if round_ndigits is not None:
        capped = round(float(capped), int(round_ndigits))

    # Defensive: never return <=0
    return max(0.01, float(capped))


@dataclass
class SynthesizeResult:
    output_path: Path


def _ensure_output_path(
    output_path: Optional[str],
    filename_prefix: Optional[str],
    default_dir: Path,
) -> Path:
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
        return out

    default_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = (filename_prefix or "tts").strip().replace(" ", "_")
    fname = f"{prefix}_{stamp}.wav" if prefix else f"tts_{stamp}.wav"
    return default_dir / fname


class _ForceCPUProviders:
    """Context manager to force onnxruntime to use CPU provider only.

    This temporarily monkey-patches onnxruntime.get_available_providers
    so that Model(...) constructed inside the context uses CPU.
    """

    def __enter__(self):
        import onnxruntime as ort  # imported here to avoid global side-effects

        self._ort = ort
        self._prev = getattr(ort, "get_available_providers")

        def _only_cpu():  # noqa: N802
            return ["CPUExecutionProvider"]

        ort.get_available_providers = _only_cpu  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_prev"):
            self._ort.get_available_providers = self._prev  # type: ignore[assignment]
        return False


def create_synth(
    model_path: str,
    device: str = "cpu",
) -> Synth:
    """Create and return a Synth instance with proper provider handling.

    This centralizes device/provider selection so it can be reused in batch flows.
    """
    model_dir = Path(model_path)
    device = (device or "cpu").lower()

    if device in ("cpu",):
        with _ForceCPUProviders():
            model = Model(model_path=str(model_dir))
    elif device in ("cuda", "gpu"):
        try:
            import onnxruntime as ort
            ort.preload_dlls()
            providers = ort.get_available_providers()  # type: ignore[attr-defined]
            if "CUDAExecutionProvider" not in providers:
                logging.warning(
                    "CUDA provider not available in onnxruntime installation; proceeding on CPU."
                )
        except Exception:
            # If ORT isn't importable here for any reason, we'll let Model handle it.
            pass
        model = Model(model_path=str(model_dir))
    else:
        logging.warning(f"Unknown device '{device}', defaulting to CPU.")
        with _ForceCPUProviders():
            model = Model(model_path=str(model_dir))

    return Synth(model)


def synthesize_text(
    text: str,
    voice: Optional[int] = 0,
    speech_rate: float = 1.0,
    model_path: str = "",
    device: str = "cpu",
    output_path: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    output_sample_rate: Optional[int] = None,
    synth: Optional[Synth] = None,
) -> Path:
    """
    Synthesize text to a .wav file using Vosk TTS.

    Parameters:
        text: Text to synthesize (non-empty).
        voice: Speaker id for multispeaker models (default 0).
        speech_rate: Speaking speed multiplier (> 0).
        model_path: Path to the model directory containing model.onnx, dictionary, config.json, etc.
        device: "cpu" or "cuda" (advisory: GPU use depends on onnxruntime build and providers).
        output_path: Full output .wav path. If omitted, one is generated under ./out.
        filename_prefix: Prefix used when output_path is not provided.

    Returns:
        Path to the written .wav file.

    Raises:
        ValueError: On invalid parameters.
        FileNotFoundError: If model_path is invalid or missing required files.
        SynthesisError: On model load or inference errors.
        OSError: On filesystem-related errors.
    """
    # Basic validations
    if text is None or not str(text).strip():
        raise ValueError("Text must be a non-empty string.")
    if speech_rate is None or speech_rate <= 0:
        raise ValueError("speech_rate must be > 0.")
    if synth is None and not model_path:
        raise ValueError("model_path is required when synth is not provided.")

    if synth is None:
        model_dir = Path(model_path)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"Model path does not exist or is not a directory: {model_path}")

        # Pre-check for common required files to give clearer error messages
        required_files = [
            model_dir / "model.onnx",
            model_dir / "dictionary",
            model_dir / "config.json",
        ]
        missing = [str(p.name) for p in required_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Model directory is missing required files: {', '.join(missing)}"
            )

    device = (device or "cpu").lower()
    NATIVE_SR = 22050  # Model native output rate
    if output_sample_rate is not None and output_sample_rate <= 0:
        raise ValueError("output_sample_rate must be a positive integer if provided.")

    # Prepare output path
    out_path = _ensure_output_path(output_path, filename_prefix, default_dir=Path.cwd() / "out")

    try:
        if synth is None:
            synth = create_synth(str(model_dir), device=device)

        speaker_id = 0 if voice is None else int(voice)

        # If no resampling requested (or native rate requested), use fast path
        if output_sample_rate is None or output_sample_rate == NATIVE_SR:
            synth.synth(
                text=text,
                oname=str(out_path),
                speaker_id=speaker_id,
                speech_rate=float(speech_rate),
            )
            logging.info(f"Synthesis complete: {out_path}")
            return out_path

        # Resampling requested: try to import helper and resample
        target_sr = int(output_sample_rate)
        try:
            try:
                # Prefer absolute package-style import
                from utils.audio_resample import (
                    resample_int16_scipy,
                    ResampleUnavailableError,
                )
            except Exception:
                # Fallback: direct sibling import if run in different contexts
                from audio_resample import (
                    resample_int16_scipy,  # type: ignore
                    ResampleUnavailableError,  # type: ignore
                )

            audio_i16 = synth.synth_audio(
                text=text,
                speaker_id=speaker_id,
                speech_rate=float(speech_rate),
            )
            audio_i16 = resample_int16_scipy(audio_i16, from_sr=NATIVE_SR, to_sr=target_sr)

            with wave.open(str(out_path), "w") as f:
                f.setnchannels(1)
                f.setsampwidth(2)  # int16
                f.setframerate(target_sr)
                f.writeframes(audio_i16.tobytes())

            logging.info(f"Synthesis complete (resampled to {target_sr} Hz): {out_path}")
            return out_path
        except Exception as re:
            # If SciPy is unavailable or resampling failed, log and continue with native rate
            msg = str(re)
            logging.error(
                "Resampling requested but failed (%s). Falling back to native %d Hz output.",
                msg,
                NATIVE_SR,
            )
            synth.synth(
                text=text,
                oname=str(out_path),
                speaker_id=speaker_id,
                speech_rate=float(speech_rate),
            )
            logging.info(f"Synthesis complete (native {NATIVE_SR} Hz fallback): {out_path}")
            return out_path

    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "onnx" in lower or "runtime" in lower:
            raise SynthesisError(
                "ONNX inference failed. Check model integrity and onnxruntime installation: " + msg
            ) from e
        if "no such file" in lower or "not found" in lower:
            raise SynthesisError(
                f"Model files missing or invalid at {model_path}: {msg}"
            ) from e
        if "speaker" in lower or "sid" in lower:
            raise SynthesisError(
                f"Invalid voice/speaker id ({voice}). Try a value within the model's supported range."
            ) from e
        raise SynthesisError(f"Synthesis failed: {msg}") from e
