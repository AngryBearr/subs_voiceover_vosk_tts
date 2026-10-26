"""ESpeech F5-TTS (ESpeech-TTS-1_RL-V2) behind the ``Engine`` protocol: lines in cloned voices.

A voice is a reference clip (WAV + transcript in ``<name>.txt`` next to it, as written by
``stt.voice_bank --export``); the dubbing script gives every line its speaker's clip as
``<path>#<content hash>`` (``voice_id``), so a new clip under the same name changes the script and
the voice-over is redone. Text
preparation is the Vosk one (rule-based expansion, RUAccent stress with ``+``, which ESpeech
reads). Each line is synthesised at a natural length (``target_syl_per_sec``), heard by GigaAM
and retried with other seeds when a word is missing or garbled (``verify.attempts``).

F5 needs its own packages (``f5_env``, ``bash scripts/setup_envs.sh f5``); the synthesis runs
there as ``voiceover.f5_worker``, once per voice-over run, with one model load.

Speakers without a reference get a fallback voice per gender: a Vosk sentence that F5 clones
(``fallback_references``).
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from voiceover.engine import PreparedText, SynthResult, VoskEngine, _sha256_file, cache_key, postprocess_wav
from voiceover.text import syllables

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_TEXT = "Сегодня на острове снова жарко, и всех ждёт непростое испытание. Посмотрим, кто справится лучше."


def voice_id(path: str | Path) -> str:
    """A reference clip as a voice: its path plus a hash of the clip and its transcript."""
    import hashlib

    path = Path(path)
    text = path.with_suffix(".txt")
    digest = hashlib.sha256((_sha256_file(path) + (text.read_text(encoding="utf-8") if text.exists() else ""))
                            .encode("utf-8")).hexdigest()
    return f"{path}#{digest[:12]}"


def clip_path(voice: str) -> str:
    return voice.split("#", 1)[0]


def fallback_references(folder: Path, model_path: Path, voices: dict[str, int]) -> dict[str, str]:
    """``{gender: wav}``: a Vosk voice per gender saying one sentence, for F5 to clone."""
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for gender, voice in voices.items():
        wav = folder / f"fallback_{gender}_vosk{voice}.wav"
        if not wav.exists():
            subprocess.run([str(REPO_ROOT / "subs_env/bin/python"), "-m", "synthesize.synthesize_cli",
                            "--text", FALLBACK_TEXT, "--model", str(model_path), "--voice", str(voice),
                            "--output", str(wav), "--output-sample-rate", "24000"],
                           cwd=REPO_ROOT, check=True, capture_output=True)
            audio, rate = sf.read(str(wav), dtype="float32")
            sf.write(str(wav), np.concatenate([audio, np.zeros(rate, dtype=np.float32)]), rate)  # F5 wants 1 s of silence
            wav.with_suffix(".txt").write_text(FALLBACK_TEXT.replace("ё", "е") + "\n", encoding="utf-8")
        out[gender] = str(wav)
    return out


class F5Engine(VoskEngine):
    """Cloned voices: ``voice`` (and per-line voices) are reference WAV paths."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.repo = str(config.get("model", "ESpeech/ESpeech-TTS-1_RL-V2"))
        self.nfe = int(config.get("nfe", 32))
        self.voice = str(config.get("voice", ""))  # type: ignore[assignment]
        self.base_rate = float(config.get("base_rate", 1.0))  # calibration probes it like Vosk's speech rate
        self.device = str(config.get("device", "cuda"))
        self.target_syl_per_sec = float(config.get("target_syl_per_sec", 5.0))
        verify = {"enabled": True, "attempts": 3, **dict(config.get("verify", {}))}
        self.attempts = int(verify["attempts"]) if verify["enabled"] else 1
        self.python = Path(str(config.get("python", REPO_ROOT / "f5_env" / "bin" / "python")))
        self.normalization = dict(config.get("normalization", {}))
        self.trim = dict(config.get("trim", {}))
        self._identity: dict[str, Any] | None = None
        self._accentizer: Any = None
        if not self.python.exists():
            raise FileNotFoundError(f"{self.python} missing: create f5_env with `bash scripts/setup_envs.sh f5`")
        if not Path(clip_path(self.voice)).is_file():
            raise FileNotFoundError(f"F5 default voice (reference WAV) not found: {self.voice!r}")

    def synth_identity(self) -> dict[str, Any]:
        return {"engine": "f5", "model": self.repo, "nfe": self.nfe, "target_syl_per_sec": self.target_syl_per_sec,
                "base_rate": self.base_rate, "attempts": self.attempts, "trim": dict(self.trim),
                "worker": 2,  # bump when voiceover.f5_worker changes the audio (2: leaked reference cut off)
                "voice": self._voice_key(self.voice)}

    @staticmethod
    def _voice_key(voice: str) -> str:
        """The clip's path and content: another clip under the same name is another voice."""
        return voice_id(clip_path(voice))

    def _voice(self, row: PreparedText) -> str:  # type: ignore[override]
        return self.voice if row.voice is None else str(row.voice)

    def duration(self, text: str) -> float:
        """Natural length of a line at the target speech rate times ``base_rate`` (seconds)."""
        return max(0.6, syllables(text) / (self.target_syl_per_sec * self.base_rate) + 0.25)

    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int = 1) -> list[SynthResult]:
        cache_dir = cache_dir / self.synth_identity_sha()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        keys: dict[str, str] = {}
        entries: dict[str, dict[str, Any]] = {}
        todo: dict[str, dict[str, Any]] = {}
        voice_keys = {v: self._voice_key(v) for v in {self._voice(row) for row in prepared}}
        for row in prepared:
            voice = self._voice(row)
            key = cache_key(row.tts_text, voice_keys[voice])
            keys[row.cue_id] = key
            if key in entries or key in todo:
                continue
            meta = cache_dir / f"{key}.json"
            if meta.exists() and (cache_dir / f"{key}.wav").exists():
                entry = json.loads(meta.read_text(encoding="utf-8"))
                if entry.get("tts_text") == row.tts_text and not entry.get("error"):
                    entries[key] = entry
                    continue
            ref = clip_path(voice)
            todo[key] = {"key": key, "text": row.tts_text, "ref_wav": ref,
                         "ref_text": Path(ref).with_suffix(".txt").read_text(encoding="utf-8").strip(),
                         "duration": round(self.duration(row.tts_text), 3), "out_wav": str(cache_dir / f"{key}.wav")}
        log.info("synthesis: %d distinct texts, %d cached, %d to synthesize (F5, %d voices)",
                 len(entries) + len(todo), len(entries), len(todo), len(voice_keys))
        if todo:
            self._run_worker(list(todo.values()), cache_dir)
        for key, item in todo.items():
            wav = Path(item["out_wav"])
            result = item.get("result") or {"error": "no result from the F5 worker"}
            if result.get("error") or not wav.exists():
                entry = {"path": str(wav), "sha256": "", "raw_sec": 0.0, "speech_sec": 0.0, "edge_trimmed_ms": 0,
                         "reduced_pauses": 0, "error": result.get("error"), "tts_text": item["text"]}
            else:
                raw_sec, speech_sec, edge_ms, reduced = postprocess_wav(wav, self.trim)
                entry = {"path": str(wav), "sha256": _sha256_file(wav), "raw_sec": raw_sec, "speech_sec": speech_sec,
                         "edge_trimmed_ms": edge_ms, "reduced_pauses": reduced, "error": None, "tts_text": item["text"],
                         "voice": item["ref_wav"], **{k: result[k] for k in ("seed", "takes", "check") if k in result}}
                (cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
            entries[key] = entry
        results = []
        for row in prepared:
            entry = entries[keys[row.cue_id]]
            fields = {name: entry.get(name) for name in SynthResult.__dataclass_fields__ if name != "cue_id"}
            results.append(SynthResult(cue_id=row.cue_id, **fields))
        return results

    def _run_worker(self, items: list[dict[str, Any]], cache_dir: Path) -> None:
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            job, out = Path(tmp) / "job.json", Path(tmp) / "results.json"
            job.write_text(json.dumps({"model": self.repo, "nfe": self.nfe, "attempts": self.attempts,
                                       "device": self.device, "items": items}, ensure_ascii=False), encoding="utf-8")
            done = subprocess.run([str(self.python), "-m", "voiceover.f5_worker", str(job), str(out)], cwd=REPO_ROOT,
                                  stderr=subprocess.PIPE, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"F5 worker failed (exit {done.returncode}): {done.stderr[-2000:]}")
            results = {r["key"]: r for r in json.loads(out.read_text(encoding="utf-8"))}
        for item in items:
            item["result"] = results.get(item["key"])
