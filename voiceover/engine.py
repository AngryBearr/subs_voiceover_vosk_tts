"""TTS engine adapters.

An engine prepares text for itself, synthesises it at its base rate and
returns measured speech seconds after the production post-processing chain
(edge trim and long-pause reduction).  Everything downstream sees only the
``Engine`` protocol and the engine identity hash.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from utils.reduce_silence import reduce_silence_audio
from utils.vosk_text import normalize_vosk_text
from voiceover.text import join_lines

# Latin letters that are visually identical to Cyrillic ones and sometimes leak
# into Russian subtitles; inside a Cyrillic word they break the TTS dictionary.
_HOMOGLYPHS = str.maketrans({"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
                             "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
                             "O": "О", "P": "Р", "T": "Т", "X": "Х"})
_MIXED_WORD = re.compile(r"\b(?=[^\W\d_]*[А-Яа-яЁё])(?=[^\W\d_]*[A-Za-z])[^\W\d_]+\b")


def fix_homoglyphs(text: str) -> str:
    """Replace Latin look-alike letters inside words that also contain Cyrillic."""
    return _MIXED_WORD.sub(lambda match: match.group(0).translate(_HOMOGLYPHS), text)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedText:
    cue_id: str
    display_text: str
    tts_text: str
    normalization: str  # ok | failed | accent_failed | disabled
    error: str | None = None
    notes: tuple[str, ...] = ()  # spans the rule-based expander was unsure about


@dataclass(frozen=True)
class SynthResult:
    cue_id: str
    path: str
    sha256: str
    raw_sec: float
    speech_sec: float
    edge_trimmed_ms: int
    reduced_pauses: int
    error: str | None = None


class Engine(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def identity_sha(self) -> str: ...
    def synth_identity_sha(self) -> str: ...
    def prepare(self, items: list[dict[str, Any]]) -> list[PreparedText]: ...
    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int) -> list[SynthResult]: ...


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(tts_text: str) -> str:
    """Cache key of one synthesis: the exact engine text (engine identity is the directory)."""
    return _sha256_text(tts_text)[:24]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dbfs_to_amp(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def trim_edges(audio: np.ndarray, sr: int, *, dbfs: float, pad_ms: int) -> tuple[np.ndarray, int]:
    """Cut leading/trailing near-silence, keeping ``pad_ms`` on each side."""
    data = audio if audio.ndim == 2 else audio[:, None]
    if data.shape[0] == 0:
        return data, 0
    loud = np.max(np.abs(data), axis=1) >= _dbfs_to_amp(dbfs)
    indices = np.flatnonzero(loud)
    if indices.size == 0:
        return data, 0
    pad = int(sr * pad_ms / 1000)
    first = max(0, int(indices[0]) - pad)
    last = min(data.shape[0], int(indices[-1]) + 1 + pad)
    removed_ms = int(round((data.shape[0] - (last - first)) * 1000 / sr))
    return data[first:last], removed_ms


def postprocess_wav(path: Path, trim: dict[str, Any]) -> tuple[float, float, int, int]:
    """Apply the production post-chain in place; return raw/final seconds and stats."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    raw_sec = data.shape[0] / sr
    edge_ms = 0
    if trim.get("edge_enabled", True):
        data, edge_ms = trim_edges(data, sr, dbfs=float(trim.get("edge_dbfs", -40.0)),
                                   pad_ms=int(trim.get("edge_pad_ms", 30)))
    reduced = 0
    if trim.get("pauses_enabled", True):
        data, reduced = reduce_silence_audio(data, sr, silence_dbfs=float(trim.get("silence_dbfs", -35.0)),
                                             threshold_sec=float(trim.get("threshold_sec", 0.5)),
                                             target_sec=float(trim.get("target_sec", 0.2)))
    sf.write(str(path), data, sr)
    return raw_sec, data.shape[0] / sr, edge_ms, reduced


_WORKER_SYNTH: Any = None


def _worker_init(model_path: str, device: str, threads: int | None = None) -> None:
    global _WORKER_SYNTH
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, str(threads))
    from synthesize.synthesize import create_synth
    _WORKER_SYNTH = create_synth(model_path, device)


def _worker_synth(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["path"])
    try:
        assert _WORKER_SYNTH is not None
        _WORKER_SYNTH.synth(text=job["tts_text"], oname=str(path), speaker_id=int(job["voice"]),
                            speech_rate=float(job["base_rate"]))
        raw_sec, speech_sec, edge_ms, reduced = postprocess_wav(path, job["trim"])
        return {"cue_id": job["cue_id"], "path": str(path), "sha256": _sha256_file(path), "raw_sec": raw_sec,
                "speech_sec": speech_sec, "edge_trimmed_ms": edge_ms, "reduced_pauses": reduced, "error": None}
    except Exception as exc:  # noqa: BLE001 - the error is recorded per cue, never hidden
        return {"cue_id": job["cue_id"], "path": str(path), "sha256": "", "raw_sec": 0.0, "speech_sec": 0.0,
                "edge_trimmed_ms": 0, "reduced_pauses": 0, "error": f"{type(exc).__name__}: {exc}"[:500]}


class VoskEngine:
    """Vosk TTS behind the ``Engine`` protocol."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_path = Path(str(config["model_path"]))
        if not (self.model_path / "model.onnx").exists():
            raise FileNotFoundError(f"Vosk model not found: {self.model_path}")
        self.voice = int(config.get("voice", 8))
        self.base_rate = float(config.get("base_rate", 1.25))
        self.device = str(config.get("device", "cpu"))
        self.normalization = dict(config.get("normalization", {}))
        self.trim = dict(config.get("trim", {}))
        self._identity: dict[str, Any] | None = None
        self._accentizer: Any = None

    def synth_identity(self) -> dict[str, Any]:
        """Everything that changes the audio of a given engine text."""
        return {
            "engine": "vosk",
            "package_version": importlib.metadata.version("vosk-tts"),
            "model_dir": str(self.model_path),
            "model_onnx_sha256": _sha256_file(self.model_path / "model.onnx"),
            "model_config_sha256": _sha256_file(self.model_path / "config.json"),
            "voice": self.voice,
            "base_rate": self.base_rate,
            "trim": dict(self.trim),
        }

    def identity(self) -> dict[str, Any]:
        """Synthesis identity plus the text preparation chain."""
        if self._identity is None:
            enabled = bool(self.normalization.get("enabled", True))
            self._identity = {
                **self.synth_identity(),
                "normalization": {
                    "enabled": enabled,
                    "expander": f"rutextnorm/{importlib.metadata.version('rutextnorm')}" if enabled else None,
                    "accent": f"ruaccent/{importlib.metadata.version('ruaccent')}" if enabled else None,
                    "punctuation_map": "vosk_text.normalize_vosk_text",
                    "prepare_version": 2,  # bump when the preparation code changes engine texts
                },
            }
        return self._identity

    def identity_sha(self) -> str:
        return _sha256_text(json.dumps(self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def synth_identity_sha(self) -> str:
        return _sha256_text(json.dumps(self.synth_identity(), ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")))

    def _load_accentizer(self) -> None:
        from utils.text_normalizer import _load_ruaccent
        if self._accentizer is None:
            self._accentizer = _load_ruaccent(str(self.normalization.get("accent_device", "cpu")))

    def prepare(self, items: list[dict[str, Any]]) -> list[PreparedText]:
        """Expand numbers/abbreviations by rules, then add stress; fail closed per cue."""
        results: list[PreparedText] = []
        if not self.normalization.get("enabled", True):
            for item in items:
                display = fix_homoglyphs(join_lines(item.get("text")))
                results.append(PreparedText(str(item["cue_id"]), display, normalize_vosk_text(display), "disabled"))
            return results
        from utils.text_normalizer import ruaccent_pass
        from voiceover.normalize import expand, space_after_ellipsis
        self._load_accentizer()
        batch_size = int(self.normalization.get("batch_size", 10))
        expanded: list[dict[str, Any] | None] = []
        errors: list[str | None] = []
        notes: list[tuple[str, ...]] = []
        for item in items:
            display = fix_homoglyphs(join_lines(item.get("text")))
            try:
                result = expand(space_after_ellipsis(display))
                expanded.append({"index": item.get("index"), "_normalized_text": result.text})
                errors.append(None)
                notes.append(result.uncertain)
            except Exception as exc:  # noqa: BLE001 - recorded per cue
                expanded.append(None)
                errors.append(f"expand: {exc}"[:300])
                notes.append(())
        accented: list[str | None] = [None] * len(items)
        pending = [index for index, row in enumerate(expanded) if row is not None]
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset + batch_size]
            rows = [expanded[index] for index in chunk]
            try:
                out = ruaccent_pass(rows, self._accentizer, batch_size=len(rows))  # type: ignore[arg-type]
                for index, row in zip(chunk, out):
                    accented[index] = join_lines(row["text"])
            except Exception:  # noqa: BLE001 - retry one by one to isolate the failing cue
                for index in chunk:
                    try:
                        out = ruaccent_pass([expanded[index]], self._accentizer, batch_size=1)  # type: ignore[list-item]
                        accented[index] = join_lines(out[0]["text"])
                    except Exception as exc:  # noqa: BLE001
                        errors[index] = f"RUAccent: {exc}"[:300]
        for index, item in enumerate(items):
            display = fix_homoglyphs(join_lines(item.get("text")))
            cue_id = str(item["cue_id"])
            if expanded[index] is None:
                results.append(PreparedText(cue_id, display, normalize_vosk_text(display), "failed", errors[index]))
            elif accented[index] is None:
                text = expanded[index]["_normalized_text"]  # type: ignore[index]
                results.append(PreparedText(cue_id, display, normalize_vosk_text(text), "accent_failed", errors[index],
                                            notes[index]))
            else:
                results.append(PreparedText(cue_id, display, normalize_vosk_text(accented[index] or ""), "ok", None,
                                            notes[index]))
        return results

    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int = 1) -> list[SynthResult]:
        """Synthesize every distinct engine text once; ``cache_dir`` is keyed by text.

        A cached entry is a ``<key>.wav`` plus ``<key>.json`` with the measured
        values.  Failed syntheses are not cached, so they are retried next time.
        """
        cache_dir = cache_dir / self.synth_identity_sha()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached: dict[str, dict[str, Any]] = {}
        jobs: dict[str, dict[str, Any]] = {}
        for row in prepared:
            key = cache_key(row.tts_text)
            if key in cached or key in jobs:
                continue
            meta, wav = cache_dir / f"{key}.json", cache_dir / f"{key}.wav"
            if meta.exists() and wav.exists():
                entry = json.loads(meta.read_text(encoding="utf-8"))
                if entry.get("tts_text") == row.tts_text and not entry.get("error"):
                    cached[key] = entry
                    continue
            jobs[key] = {"cue_id": key, "tts_text": row.tts_text, "path": str(wav),
                         "voice": self.voice, "base_rate": self.base_rate, "trim": self.trim}
        log.info("synthesis: %d distinct texts, %d cached, %d to synthesize", len(cached) + len(jobs),
                 len(cached), len(jobs))
        job_list = list(jobs.values())
        if not job_list:
            raw: list[dict[str, Any]] = []
        elif workers <= 1:
            _worker_init(str(self.model_path), self.device)
            raw = [_worker_synth(job) for job in job_list]
        else:
            threads = max(1, (os.cpu_count() or workers) // workers)
            context = multiprocessing.get_context("spawn")
            with context.Pool(min(workers, len(job_list)), initializer=_worker_init,
                              initargs=(str(self.model_path), self.device, threads)) as pool:
                raw = pool.map(_worker_synth, job_list, chunksize=4)
        for job, result in zip(job_list, raw):
            key = job["cue_id"]
            entry = {**result, "tts_text": job["tts_text"]}
            entry.pop("cue_id", None)
            if not entry.get("error"):
                (cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
            cached[key] = entry
        results: list[SynthResult] = []
        for row in prepared:
            entry = cached[cache_key(row.tts_text)]
            fields = {name: entry.get(name) for name in SynthResult.__dataclass_fields__ if name != "cue_id"}
            results.append(SynthResult(cue_id=row.cue_id, **fields))
        return results


def build_engine(config: dict[str, Any]) -> Engine:
    name = str(config.get("name", "vosk"))
    if name == "vosk":
        return VoskEngine(config)
    raise ValueError(f"unknown engine: {name}")


def result_dict(result: SynthResult) -> dict[str, Any]:
    return asdict(result)
