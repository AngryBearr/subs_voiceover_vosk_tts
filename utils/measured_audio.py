"""Measured Vosk audio for one cue: normalize, synthesize, trim pauses, bounded tempo, fit verdict."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from utils.analyze_text import join_text_lines
from utils.pipeline_text import resolve_speed_policy


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class AudioEngine:
    """Lazy, process-local model reuse; source-bound normalization for one cue."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.normalizer: Any = None
        self.accentizer: Any = None
        self.synth: Any = None

    def normalize(self, item: dict[str, Any]) -> str:
        from utils.text_normalizer import _load_runorm, _load_ruaccent, runorm_pass, ruaccent_pass
        settings = self.config.get("normalization", {})
        if self.normalizer is None:
            self.normalizer = _load_runorm(str(settings.get("norm_device", "cpu")))
        if self.accentizer is None:
            self.accentizer = _load_ruaccent(str(settings.get("accent_device", "cpu")))
        return join_text_lines(ruaccent_pass(runorm_pass([item], self.normalizer), self.accentizer,
                                            batch_size=1)[0]["text"])

    def synthesize(self, text: str, path: Path, inputs: list[str]) -> None:
        from synthesize.synthesize import create_synth, synthesize_text
        settings = self.config["tts"]
        if self.synth is None:
            self.synth = create_synth(settings["model_path"], str(settings.get("device", "cpu")))
        synthesize_text(text, voice=int(settings.get("voice", 8)),
                        speech_rate=resolve_speed_policy(self.config).base_speech_rate,
                        synth=self.synth, output_path=str(path), output_sample_rate=None,
                        tts_input_texts=inputs)


def wav_info(path: Path) -> dict[str, Any]:
    import soundfile as sf
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Empty/invalid WAV")
    return {"path": str(path), "frames": info.frames, "sample_rate": info.samplerate,
            "duration_sec": info.frames / info.samplerate, "sha256": file_sha256(path)}


def measure_audio(config: dict[str, Any], raw: Path, pretempo: Path, final: Path,
                   allowed: float, attempt: dict[str, Any], pause_processor: Any = None,
                   chosen_text: str | None = None, available_ms: int | None = None) -> None:
    from utils.audio_analysis import render_bounded_tempo
    from utils.audio_resample import resample_wav_file
    from utils.reduce_silence import reduce_silence_file
    if config.get("silero_edges", {}).get("enabled", False):
        from utils.silero_pauses import validate_profile
        validate_profile(config)
        if available_ms is None or type(available_ms) is not int or allowed != available_ms / 1000:
            raise ValueError("Edge stage requires the exact canonical available_ms window")
    if pretempo.exists() or final.exists() or attempt.get("raw"):
        raise ValueError("Audio processing already attempted; never repeat stages on saved audio")
    attempt["raw"] = wav_info(raw)
    shutil.copyfile(raw, pretempo)
    settings = config.get("reduce_silence", {})
    if settings.get("enabled", True):
        reduce_silence_file(pretempo, pretempo, silence_dbfs=float(settings.get("silence_dbfs", -35)),
                            threshold_sec=float(settings.get("threshold_sec", 0.5)),
                            target_sec=float(settings.get("target_sec", 0.2)))
    settings = config.get("resample", {})
    if settings.get("enabled", True):
        resample_wav_file(pretempo, pretempo, target_sr=int(settings.get("target_sr", 44100)))
    attempt["pretempo"] = wav_info(pretempo)
    measured_source = pretempo
    if config.get("silero_pauses", {}).get("enabled", False):
        from utils.silero_pauses import PauseProcessor
        processor = pause_processor if pause_processor is not None else PauseProcessor()
        measured_source = pretempo.with_name("pauses.wav")
        attempt["silero_pauses"] = processor.process(pretempo, measured_source)
        attempt["pauses"] = wav_info(measured_source)
        if config.get("silero_edges", {}).get("enabled", False):
            from utils.silero_pauses import process_edges
            edge_source = pretempo.with_name("edges.wav")
            assert available_ms is not None
            attempt["silero_edges"] = process_edges(measured_source, edge_source, attempt["silero_pauses"],
                                                    available_ms=available_ms, chosen_text=chosen_text)
            attempt["edges"] = wav_info(edge_source)
            measured_source = edge_source
    before = wav_info(measured_source)["duration_sec"]
    policy = resolve_speed_policy(config)
    remaining = policy.max_total_speed / policy.base_speech_rate
    factor = 1.0
    attempt["required_factor"] = max(1.0, before / allowed)
    attempt["render_attempts"] = []
    if before > allowed and before / allowed <= remaining and config.get("audio_analysis", {}).get("enabled", True):
        factor, _ = render_bounded_tempo(measured_source, final, allowed_duration_sec=allowed,
                                        max_factor=remaining, attempts=attempt["render_attempts"])
    else:
        shutil.copyfile(measured_source, final)
    attempt["final"] = wav_info(final)
    attempt["effective_factor"] = factor
    attempt["effective_total_speed"] = factor * policy.base_speech_rate
    attempt["status"] = ("accelerated_fits" if factor > 1 else "fits") if attempt["final"]["duration_sec"] <= allowed else "unresolved"


