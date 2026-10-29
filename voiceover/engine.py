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
from dataclasses import asdict, dataclass, replace
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
    voice: int | str | None = None  # per-cue voice; None: the engine's default voice


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


def cache_key(tts_text: str, voice: int | str | None = None) -> str:
    """Cache key of one synthesis: the exact engine text (engine identity, with the default
    voice, is the directory); another voice than the default one is part of the key."""
    return _sha256_text(tts_text if voice is None else f"voice={voice}\n{tts_text}")[:24]


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
CPU_THREADS_ENV = "VOICEOVER_CPU_THREADS"  # set by stt.run when Vosk shares the CPU with the GPU engines


def cpu_threads() -> int:
    """CPU threads Vosk may use: all cores, or ``VOICEOVER_CPU_THREADS`` (the rest feeds the GPU engines)."""
    cores = os.cpu_count() or 4
    try:
        return max(1, min(cores, int(os.environ.get(CPU_THREADS_ENV, cores))))
    except ValueError:
        return cores


def _worker_init(model_path: str, device: str, threads: int | None = None) -> None:
    global _WORKER_SYNTH
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, str(threads))
    from synthesize.synthesize import create_synth
    # ONNX Runtime ignores the variables above: its sessions get the cap only when the CPU is shared
    _WORKER_SYNTH = create_synth(model_path, device, threads if os.environ.get(CPU_THREADS_ENV) else None)


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


TERATTS = "teratts:"  # a line's voice spoken by TeraTTS inside another engine's track (host, people without a clip)


class VoskEngine:
    """Vosk TTS behind the ``Engine`` protocol.

    A line whose voice is ``teratts:<voice>`` (the host's voice of ``voiceover.host_voice``) is spoken
    by TeraTTS with the ``teratts`` settings, in its own cache; the track mixes both (clips are
    resampled when placed).
    """

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
        self.listen_stress = bool(config.get("listen_stress", True))
        self.teratts_config = dict(config.get("teratts", {}))  # for teratts:<voice> lines
        self._teratts: Any = None
        self._identity: dict[str, Any] | None = None
        self._accentizer: Any = None

    def _is_teratts(self, row: PreparedText) -> bool:
        return str(self.voice if row.voice is None else row.voice).startswith(TERATTS)

    def _teratts_engine(self) -> Any:
        if getattr(self, "_teratts", None) is None:
            from voiceover.engine_teratts import TeraTTSEngine

            self._teratts = TeraTTSEngine({"voice": "ru_m5", "device": "cuda", "trim": dict(self.trim),
                                           "listen_stress": getattr(self, "listen_stress", True),
                                           "normalization": self.normalization,  # the stress words to hear
                                           **self.teratts_config})
        return self._teratts

    def _with_teratts(self, prepared: list[PreparedText], cache_dir: Path,
                      own: Any) -> list[SynthResult]:
        """``teratts:<voice>`` lines by TeraTTS, the rest by ``own`` (rows -> results), in the rows' order."""
        spoken = [row for row in prepared if self._is_teratts(row)]
        if not spoken:
            return own(prepared)
        tera = self._teratts_engine()
        by_tera = tera.synthesize_many([replace(row, voice=str(self.voice if row.voice is None else row.voice)
                                                [len(TERATTS):]) for row in spoken], cache_dir)
        rest = own([row for row in prepared if not self._is_teratts(row)])
        done = {r.cue_id: r for r in [*by_tera, *rest]}
        return [done[row.cue_id] for row in prepared]

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
                    **({"accent_per_cue": True} if self.normalization.get("accent_per_cue", True) else {}),
                    **({"fill_stress": True} if self.normalization.get("fill_stress", True) else {}),
                    **({"keep_input_yo_homographs": True} if self.normalization.get("keep_input_yo_homographs")
                       else {}),
                    **({"stress_dict_sha256": _sha256_text(json.dumps(self._stress_dict(), ensure_ascii=False,
                                                                       sort_keys=True))}
                       if self._stress_dict() else {}),
                },
            }
        return self._identity

    def identity_sha(self) -> str:
        return _sha256_text(json.dumps(self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def synth_identity_sha(self) -> str:
        return _sha256_text(json.dumps(self.synth_identity(), ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")))

    def _stress_dict(self) -> dict[str, dict[str, Any]]:
        """Stress corrections from ``voiceover.stress_check`` (``normalization.stress_dict``), read once."""
        if getattr(self, "_stress", None) is None:
            path = self.normalization.get("stress_dict")
            fixes: dict[str, dict[str, Any]] = {}
            if path and Path(str(path)).is_file():
                data = json.loads(Path(str(path)).read_text(encoding="utf-8"))
                fixes = {key: dict(data.get(key, {})) for key in ("dictionary", "sentences")}
            self._stress = fixes if any(fixes.values()) else {}
        return self._stress

    def _listen_here(self, text: str) -> list[str]:
        """Words of an engine text whose stress the stress check looked at (``normalization.stress_dict``):
        each take of the line is heard for them (``voiceover.stress_listen``); none with ``listen_stress`` off."""
        if getattr(self, "_listen", None) is None:
            from voiceover.stress_listen import load_listen_words

            self._listen = load_listen_words(self.normalization.get("stress_dict")) \
                if getattr(self, "listen_stress", True) else set()
        if not self._listen:
            return []
        from voiceover.stress_listen import words_in

        return words_in(text, self._listen)

    def _load_accentizer(self) -> None:
        from utils.text_normalizer import _load_ruaccent
        if self._accentizer is None:
            self._accentizer = _load_ruaccent(str(self.normalization.get("accent_device", "cpu")))
            fixes = self._stress_dict()
            if fixes:
                self._accentizer.accents.update(fixes["dictionary"])
                log.info("stress dictionary: %d guessed words, homograph fixes in %d sentences",
                         len(fixes["dictionary"]), len(fixes["sentences"]))

    def prepare(self, items: list[dict[str, Any]]) -> list[PreparedText]:
        """Expand numbers/abbreviations by rules, then add stress; fail closed per cue."""
        results: list[PreparedText] = []
        if not self.normalization.get("enabled", True):
            for item in items:
                display = fix_homoglyphs(join_lines(item.get("text")))
                results.append(PreparedText(str(item["cue_id"]), display, normalize_vosk_text(display), "disabled",
                                            voice=item.get("voice")))
            return results
        from utils.text_normalizer import ruaccent_pass
        from voiceover.normalize import expand, space_after_ellipsis
        self._load_accentizer()
        batch_size = int(self.normalization.get("batch_size", 10))
        per_cue = bool(self.normalization.get("accent_per_cue", True))  # the joined batch drops stress marks
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
                out = ruaccent_pass(rows, self._accentizer, batch_size=len(rows), per_cue=per_cue)  # type: ignore[arg-type]
                for index, row in zip(chunk, out):
                    accented[index] = join_lines(row["text"])
            except Exception:  # noqa: BLE001 - retry one by one to isolate the failing cue
                for index in chunk:
                    try:
                        out = ruaccent_pass([expanded[index]], self._accentizer, batch_size=1,  # type: ignore[list-item]
                                            per_cue=per_cue)
                        accented[index] = join_lines(out[0]["text"])
                    except Exception as exc:  # noqa: BLE001
                        errors[index] = f"RUAccent: {exc}"[:300]
        for index, item in enumerate(items):
            display = fix_homoglyphs(join_lines(item.get("text")))
            cue_id = str(item["cue_id"])
            if expanded[index] is None:
                results.append(PreparedText(cue_id, display, normalize_vosk_text(display), "failed", errors[index],
                                            voice=item.get("voice")))
            elif accented[index] is None:
                text = expanded[index]["_normalized_text"]  # type: ignore[index]
                results.append(PreparedText(cue_id, display, normalize_vosk_text(text), "accent_failed", errors[index],
                                            notes[index], voice=item.get("voice")))
            else:
                text = accented[index] or ""
                if self.normalization.get("keep_input_yo_homographs"):
                    from voiceover.normalize import keep_yo_spelling
                    text = keep_yo_spelling(expanded[index]["_normalized_text"], text,  # type: ignore[index]
                                            self._accentizer.yo_homographs, self._accentizer.accents)
                if self.normalization.get("fill_stress", True):
                    from utils.text_normalizer import fill_dictionary_stress
                    text = fill_dictionary_stress(text, self._accentizer.accents, self._accentizer.omographs)
                text = normalize_vosk_text(text)
                sentence_fixes = self._stress_dict().get("sentences")
                if sentence_fixes:
                    from voiceover.stress_check import apply_sentence_fixes
                    text = apply_sentence_fixes(text, sentence_fixes, text.replace("+", ""))
                results.append(PreparedText(cue_id, display, text, "ok", None, notes[index], voice=item.get("voice")))
        return results

    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int = 1) -> list[SynthResult]:
        return self._with_teratts(prepared, cache_dir, lambda rows: self._synthesize_vosk(rows, cache_dir, workers))

    def _synthesize_vosk(self, prepared: list[PreparedText], cache_dir: Path, workers: int = 1) -> list[SynthResult]:
        """Synthesize every distinct engine text once; ``cache_dir`` is keyed by text.

        A cached entry is a ``<key>.wav`` plus ``<key>.json`` with the measured
        values, written as each line is done (an interrupted run resumes where it stopped).
        Failed syntheses are not cached, so they are retried next time.
        """
        cache_dir = cache_dir / self.synth_identity_sha()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached: dict[str, dict[str, Any]] = {}
        jobs: dict[str, dict[str, Any]] = {}
        for row in prepared:
            voice = self._voice(row)
            key = cache_key(row.tts_text, None if voice == self.voice else voice)
            if key in cached or key in jobs:
                continue
            meta, wav = cache_dir / f"{key}.json", cache_dir / f"{key}.wav"
            if meta.exists() and wav.exists():
                entry = json.loads(meta.read_text(encoding="utf-8"))
                if entry.get("tts_text") == row.tts_text and not entry.get("error"):
                    cached[key] = entry
                    continue
            jobs[key] = {"cue_id": key, "tts_text": row.tts_text, "path": str(wav),
                         "voice": voice, "base_rate": self.base_rate, "trim": self.trim}
        log.info("synthesis: %d distinct texts, %d cached, %d to synthesize", len(cached) + len(jobs),
                 len(cached), len(jobs))
        job_list = list(jobs.values())

        def store(job: dict[str, Any], result: dict[str, Any]) -> None:
            key = job["cue_id"]
            entry = {**result, "tts_text": job["tts_text"]}
            entry.pop("cue_id", None)
            if not entry.get("error"):
                (cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
            cached[key] = entry

        cores = cpu_threads()
        if job_list and workers <= 1:
            _worker_init(str(self.model_path), self.device, cores if cores < (os.cpu_count() or cores) else None)
            for job in job_list:
                store(job, _worker_synth(job))
        elif job_list:
            threads = max(1, cores // workers)
            context = multiprocessing.get_context("spawn")
            with context.Pool(min(workers, len(job_list)), initializer=_worker_init,
                              initargs=(str(self.model_path), self.device, threads)) as pool:
                for job, result in zip(job_list, pool.imap(_worker_synth, job_list, chunksize=4)):
                    store(job, result)
        self._hear_stress(prepared, cached, cache_dir)
        results: list[SynthResult] = []
        for row in prepared:
            voice = self._voice(row)
            entry = cached[cache_key(row.tts_text, None if voice == self.voice else voice)]
            fields = {name: entry.get(name) for name in SynthResult.__dataclass_fields__ if name != "cue_id"}
            results.append(SynthResult(cue_id=row.cue_id, **fields))
        return results

    def _voice(self, row: PreparedText) -> int:
        return self.voice if row.voice is None else int(row.voice)

    def _hear_stress(self, prepared: list[PreparedText], entries: dict[str, dict[str, Any]], cache_dir: Path) -> None:
        """Lines with words whose stress was checked: each take heard once by TeraAnaly; one said with
        another stress is synthesised again (Vosk's sampling noise differs on every run) and the take
        with the fewest such words is kept (``stress_attempts`` takes at most)."""
        from voiceover.stress_listen import StressListener, hear, listened, record, wrong_words

        todo: dict[str, tuple[PreparedText, list[str]]] = {}
        for row in prepared:
            key = cache_key(row.tts_text, None if self._voice(row) == self.voice else self._voice(row))
            entry = entries.get(key)
            listen = self._listen_here(row.tts_text) if entry and not entry.get("error") else []
            if listen and key not in todo and not listened(entry, listen):  # type: ignore[arg-type]
                todo[key] = (row, listen)
        if not todo:
            return
        listener = StressListener(str(self.config.get("listen_device", "cuda")))
        if _WORKER_SYNTH is None:
            _worker_init(str(self.model_path), self.device)
        retaken = still = 0
        for key, (row, listen) in todo.items():
            entry = entries[key]
            wav = Path(entry["path"])
            audio, rate = sf.read(str(wav), dtype="float32")
            heard = hear(listener, audio, rate, row.tts_text, set(listen))
            retry = wav.with_suffix(".retry.wav")
            for _ in range(1, int(self.config.get("stress_attempts", 3))):
                if not wrong_words(heard):
                    break
                result = _worker_synth({"cue_id": key, "tts_text": row.tts_text, "path": str(retry),
                                        "voice": self._voice(row), "base_rate": self.base_rate, "trim": self.trim})
                if result["error"]:
                    break
                audio, rate = sf.read(str(retry), dtype="float32")
                again = hear(listener, audio, rate, row.tts_text, set(listen))
                if wrong_words(again) < wrong_words(heard):
                    retry.replace(wav)
                    entry.update({k: result[k] for k in ("sha256", "raw_sec", "speech_sec", "edge_trimmed_ms",
                                                          "reduced_pauses")})
                    heard = again
                    retaken += 1
            retry.unlink(missing_ok=True)
            still += bool(wrong_words(heard))
            entry["stress"] = record(listen, heard)
            (cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("stress heard in %d lines: %d taken again, %d still with another stress", len(todo), retaken, still)


def build_engine(config: dict[str, Any]) -> Engine:
    name = str(config.get("name", "vosk"))
    if name == "vosk":
        return VoskEngine(config)
    if name == "teratts":
        from voiceover.engine_teratts import TeraTTSEngine
        return TeraTTSEngine(config)
    if name == "f5":
        from voiceover.engine_f5 import F5Engine
        return F5Engine(config)
    raise ValueError(f"unknown engine: {name}")


def result_dict(result: SynthResult) -> dict[str, Any]:
    return asdict(result)
