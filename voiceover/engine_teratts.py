"""TeraTTSv2 (TeraSpace, ONNX flow-matching TTS) behind the ``Engine`` protocol.

Text preparation is the Vosk one (rule-based expansion of numbers/abbreviations/Latin words,
RUAccent stress marks with the project dictionary, ASCII punctuation): TeraTTS keeps words that
already carry ``+`` and runs its own bundled accentizer only on the rest.

TeraTTS predicts only the total length of an utterance and its diffusion model fills that time.
When the prediction is too long the whole phrase sounds drawn out, too short — rushed. With
``target_syl_per_sec`` set, a clip whose speech rate is off by more than ``rate_tolerance`` is
synthesised again with the length that gives the target rate (times ``base_rate``), so all
lines share one natural tempo and the timeline solver only fine-tunes it. ``base_rate`` maps
to ``duration_scale = 1 / base_rate``, so calibration works as for Vosk.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voiceover.engine import (PreparedText, SynthResult, VoskEngine, _sha256_file, cache_key, postprocess_wav)
from voiceover.text import syllables

log = logging.getLogger(__name__)

SAMPLE_RATE = 44_100
_MODELS: dict[tuple[str, str, str], Any] = {}


def snapshot(repo: str) -> Path:
    """Local snapshot of the model repository (downloaded once into the HF cache)."""
    if Path(repo).is_dir():
        return Path(repo)
    cached = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
                                  / "snapshots" / "*")))
    if cached:
        return Path(cached[-1])
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


def load_model(path: Path, device: str, diffusion_model: str) -> Any:
    """The TeraTTS model, loaded once per process."""
    key = (str(path), device, diffusion_model)
    if key not in _MODELS:
        import onnxruntime as ort
        from transformers import AutoModel
        if device == "cuda":
            ort.preload_dlls()
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        _MODELS[key] = AutoModel.from_pretrained(str(path), trust_remote_code=True, provider=provider,
                                                 diffusion_model=diffusion_model)
    return _MODELS[key]


def rate_correction(n_syllables: int, speech_sec: float, target_syl_per_sec: float, tolerance: float,
                    min_syllables: int = 4) -> float | None:
    """Factor for the utterance length that brings its rate to the target, or None if it is close enough."""
    if target_syl_per_sec <= 0 or n_syllables < min_syllables or speech_sec <= 0:
        return None
    ratio = speech_sec / (n_syllables / target_syl_per_sec)  # > 1: slower than the target
    if 1 / (1 + tolerance) <= ratio <= 1 + tolerance:
        return None
    return float(np.clip(1.0 / ratio, 0.5, 2.0))


class TeraTTSEngine(VoskEngine):
    """TeraTTSv2 voices (ru_f1, ru_f2, ru_m1, ru_m5) with the Vosk text preparation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.repo = str(config.get("model", "TeraSpace/TeraTTSv2"))
        self.path = snapshot(self.repo)
        self.voice = str(config.get("voice", "ru_m5"))  # type: ignore[assignment]
        self.base_rate = float(config.get("base_rate", 1.0))
        self.device = str(config.get("device", "cuda"))
        self.diffusion_model = str(config.get("diffusion_model", "distilled"))
        self.target_syl_per_sec = float(config.get("target_syl_per_sec", 5.5))
        self.rate_tolerance = float(config.get("rate_tolerance", 0.15))
        self.normalization = dict(config.get("normalization", {}))
        self.trim = dict(config.get("trim", {}))
        self._identity: dict[str, Any] | None = None
        self._accentizer: Any = None
        self._check_voice(self.voice)

    def _check_voice(self, voice: str) -> None:
        if not (self.path / "styles" / voice).is_dir():
            choices = ", ".join(sorted(p.name for p in (self.path / "styles").iterdir() if p.is_dir()))
            raise ValueError(f"TeraTTS voice not found: {voice} (available: {choices})")

    def synth_identity(self) -> dict[str, Any]:
        sampler = "sampler_distilled_cfg3_8step.onnx" if self.diffusion_model == "distilled" \
            else "sampler_teacher_8step.onnx"
        return {
            "engine": "teratts",
            "model": self.repo,
            "snapshot": self.path.name,
            "sampler_sha256": _sha256_file(self.path / "models" / sampler),
            "voice": self.voice,
            "base_rate": self.base_rate,
            "diffusion_model": self.diffusion_model,
            "target_syl_per_sec": self.target_syl_per_sec,
            "rate_tolerance": self.rate_tolerance,
            "trim": dict(self.trim),
        }

    def _voice(self, row: PreparedText) -> str:  # type: ignore[override]
        return self.voice if row.voice is None else str(row.voice)

    def _synth_one(self, model: Any, text: str, voice: str, wav: Path) -> dict[str, Any]:
        scale = 1.0 / self.base_rate
        audio = model.generate_speech(f"<ru>{text}</ru>", voice=voice, duration_scale=scale)
        sf.write(str(wav), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
        raw_sec, speech_sec, edge_ms, reduced = postprocess_wav(wav, self.trim)
        factor = rate_correction(syllables(text), speech_sec, self.target_syl_per_sec * self.base_rate,
                                 self.rate_tolerance)
        if factor is not None:
            scale *= factor
            audio = model.generate_speech(f"<ru>{text}</ru>", voice=voice, duration_scale=scale)
            sf.write(str(wav), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
            raw_sec, speech_sec, edge_ms, reduced = postprocess_wav(wav, self.trim)
        return {"path": str(wav), "sha256": _sha256_file(wav), "raw_sec": raw_sec, "speech_sec": speech_sec,
                "edge_trimmed_ms": edge_ms, "reduced_pauses": reduced, "error": None,
                "duration_scale": round(scale, 4)}

    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int = 1) -> list[SynthResult]:
        """Synthesise every distinct (text, voice) once on one model; ``workers`` is ignored (GPU)."""
        cache_dir = cache_dir / self.synth_identity_sha()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict[str, Any]] = {}
        todo: dict[str, tuple[str, str]] = {}
        for row in prepared:
            voice = self._voice(row)
            key = cache_key(row.tts_text, None if voice == self.voice else voice)
            if key in entries or key in todo:
                continue
            meta, wav = cache_dir / f"{key}.json", cache_dir / f"{key}.wav"
            if meta.exists() and wav.exists():
                entry = json.loads(meta.read_text(encoding="utf-8"))
                if entry.get("tts_text") == row.tts_text and not entry.get("error"):
                    entries[key] = entry
                    continue
            self._check_voice(voice)
            todo[key] = (row.tts_text, voice)
        log.info("synthesis: %d distinct texts, %d cached, %d to synthesize", len(entries) + len(todo), len(entries),
                 len(todo))
        model = load_model(self.path, self.device, self.diffusion_model) if todo else None
        for key, (text, voice) in todo.items():
            wav = cache_dir / f"{key}.wav"
            try:
                entry = {**self._synth_one(model, text, voice, wav), "tts_text": text}
                (cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - recorded per cue, never hidden
                entry = {"path": str(wav), "sha256": "", "raw_sec": 0.0, "speech_sec": 0.0, "edge_trimmed_ms": 0,
                         "reduced_pauses": 0, "error": f"{type(exc).__name__}: {exc}"[:500], "tts_text": text}
            entries[key] = entry
        results = []
        for row in prepared:
            voice = self._voice(row)
            entry = entries[cache_key(row.tts_text, None if voice == self.voice else voice)]
            fields = {name: entry.get(name) for name in SynthResult.__dataclass_fields__ if name != "cue_id"}
            results.append(SynthResult(cue_id=row.cue_id, **fields))
        return results
