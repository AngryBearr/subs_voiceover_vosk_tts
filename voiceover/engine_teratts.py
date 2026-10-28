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

The diffusion sampler sometimes mumbles or drops a word or draws one out. With ``verify``
every clip is transcribed by GigaAM right away (``voiceover.speech_check``); a clip with a word
not heard, a garbled word or a drawn-out word is synthesised again with other seeds and the
best take is kept; ``verify.retry_lost_words`` (default) also retries while a short word ("и",
"в") is not heard.

A voice can also be an equal blend of built-in voices, ``ru_f1+ru_f2`` (the user's choice for the
female voice, 28.10: between ru_f1's cartoonish 272 Hz and ru_f2's 204 Hz). Its two style vectors
are the means of the parts', saved once under ``models/teratts_styles/<name>/`` and handed to
TeraTTS as a path; a fixed blend is as stable as a built-in voice.
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
STYLES_DIR = Path(__file__).resolve().parent.parent / "models" / "teratts_styles"
_STYLE_FILES = ("style_ttl.npy", "style_dp.npy")
LOST_WORD_SCORE = 0.15  # per word of the text not heard, with verify.retry_lost_words


def style_dir(release: Path, voice: str, styles_dir: Path = STYLES_DIR) -> Path:
    """The folder with a voice's style vectors: built-in, or an equal blend ("ru_f1+ru_f2") made once."""
    parts = voice.split("+")
    if len(parts) == 1:
        return release / "styles" / voice
    target = styles_dir / voice
    if not all((target / name).exists() for name in _STYLE_FILES):
        target.mkdir(parents=True, exist_ok=True)
        for name in _STYLE_FILES:
            vectors = [np.load(release / "styles" / part / name).astype(np.float32) for part in parts]
            np.save(target / name, np.mean(vectors, axis=0).astype(np.float32))
    return target


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


class _ScaledNoise:
    """numpy stand-in for TeraTTS's sampler module: initial noise scaled by a temperature.

    TeraTTS draws the diffusion's starting latent from N(0, 1) with no temperature option;
    0.667 (as in teratts-android) cuts dropped words noticeably. Everything else is numpy.
    """

    def __init__(self, temperature: float) -> None:
        self._temperature = temperature
        self.random = self  # the sampler calls np.random.default_rng(seed).standard_normal(shape)

    def default_rng(self, seed: Any = None) -> Any:
        rng = np.random.default_rng(seed)
        temperature = self._temperature

        class _Rng:
            def standard_normal(self, *args: Any, **kwargs: Any) -> np.ndarray:
                return rng.standard_normal(*args, **kwargs) * temperature

            def __getattr__(self, name: str) -> Any:
                return getattr(rng, name)

        return _Rng()

    def __getattr__(self, name: str) -> Any:
        return getattr(np.random if name in vars(np.random) and name != "random" else np, name)


def set_temperature(model: Any, temperature: float) -> None:
    """Scale the starting noise of the loaded model's sampler (1.0 = the model's own behaviour)."""
    import sys

    for name, module in list(sys.modules.items()):  # the remote-code module ...transformers_modules...teratts
        if name.rsplit(".", 1)[-1] == "teratts" and "_generate_latent" in vars(module) and "np" in vars(module):
            real = module.np if not isinstance(module.np, _ScaledNoise) else np
            module.np = real if temperature == 1.0 else _ScaledNoise(temperature)


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
        # another take also while a word of the text is not heard, short ones included (the speech check
        # skips words under 3 letters): S29E03, 640 lines: words not heard 93 -> 64, CER 0.52% -> 0.37%
        self.verify = {"enabled": True, "attempts": 3, "retry_lost_words": True, **dict(config.get("verify", {}))}
        self.temperature = float(config.get("temperature", 0.667))
        self._recognizer: Any = None
        self._identity: dict[str, Any] | None = None
        self._accentizer: Any = None
        self._check_voice(self.voice)

    def _check_voice(self, voice: str) -> None:
        missing = [part for part in voice.split("+") if not (self.path / "styles" / part).is_dir()]
        if missing:
            choices = ", ".join(sorted(p.name for p in (self.path / "styles").iterdir() if p.is_dir()))
            raise ValueError(f"TeraTTS voice not found: {', '.join(missing)} (available: {choices}, "
                             f"or a blend like ru_f1+ru_f2)")

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
            "temperature": self.temperature,
            "target_syl_per_sec": self.target_syl_per_sec,
            "rate_tolerance": self.rate_tolerance,
            "trim": dict(self.trim),
            **({"verify": dict(self.verify)} if self.verify.get("enabled") else {}),
        }

    def _voice(self, row: PreparedText) -> str:  # type: ignore[override]
        return self.voice if row.voice is None else str(row.voice)

    def _take(self, model: Any, text: str, voice: str, wav: Path, scale: float, seed: int
              ) -> tuple[float, float, int, int]:
        style = voice if "+" not in voice else str(style_dir(self.path, voice))  # TeraTTS joins it to styles/
        audio = model.generate_speech(f"<ru>{text}</ru>", voice=style, duration_scale=scale, seed=seed)
        sf.write(str(wav), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
        return postprocess_wav(wav, self.trim)

    def _check(self, text: str, wav: Path) -> dict[str, Any]:
        from voiceover.speech_check import Recognizer, check_audio

        if self._recognizer is None:
            self._recognizer = Recognizer(self.device)
        audio, sr = sf.read(str(wav), dtype="float32")
        return check_audio(self._recognizer, text, audio, sr)

    def _score(self, text: str, check: dict[str, Any]) -> tuple[float, int]:
        """(take score, words of the text not heard); lost short words count with ``retry_lost_words``."""
        from voiceover.speech_check import clip_score, missing_words, plain_words

        lost = missing_words(plain_words(text), plain_words(check["heard"])) if self.verify.get("retry_lost_words") else 0
        return clip_score(check) + LOST_WORD_SCORE * lost, lost

    def _synth_one(self, model: Any, text: str, voice: str, wav: Path) -> dict[str, Any]:

        seed = 1234
        scale = 1.0 / self.base_rate
        raw_sec, speech_sec, edge_ms, reduced = self._take(model, text, voice, wav, scale, seed)
        factor = rate_correction(syllables(text), speech_sec, self.target_syl_per_sec * self.base_rate,
                                 self.rate_tolerance)
        if factor is not None:
            scale *= factor
            raw_sec, speech_sec, edge_ms, reduced = self._take(model, text, voice, wav, scale, seed)
        entry: dict[str, Any] = {"duration_scale": round(scale, 4), "seed": seed}
        if self.verify.get("enabled"):
            check = self._check(text, wav)
            score, lost = self._score(text, check)
            best = (score, seed, check, (raw_sec, speech_sec, edge_ms, reduced), lost)
            retry = wav.with_suffix(".retry.wav")
            for attempt in range(1, int(self.verify["attempts"])):
                if not ({"pronunciation", "drawn_out"} & set(best[2]["flags"]) or best[4]):
                    break
                stats = self._take(model, text, voice, retry, scale, seed + attempt)
                check = self._check(text, retry)
                score, lost = self._score(text, check)
                if score < best[0]:
                    retry.replace(wav)
                    best = (score, seed + attempt, check, stats, lost)
            retry.unlink(missing_ok=True)
            raw_sec, speech_sec, edge_ms, reduced = best[3]
            entry.update(seed=best[1], check={k: best[2][k] for k in ("cer", "stretch", "flags", "heard")})
        return {"path": str(wav), "sha256": _sha256_file(wav), "raw_sec": raw_sec, "speech_sec": speech_sec,
                "edge_trimmed_ms": edge_ms, "reduced_pauses": reduced, "error": None, **entry}

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
        if model is not None:
            set_temperature(model, self.temperature)
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
