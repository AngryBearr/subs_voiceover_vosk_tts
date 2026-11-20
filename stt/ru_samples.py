"""Russian reference samples of the season's people, for ESpeech to clone instead of the English clips.

ESpeech clones a person from ``<Stem>.wav`` (+ ``.txt``) of ``<Season>.references/`` (``stt.voice_bank --export``),
an English clip. A blind listening test (19.11) preferred a cloning from a Russian sample of the same person: X-Voice
(``voiceover.xvoice_worker`` in ``xvoice_env``, cross-lingual cloning) says the Russian translation of what the person
said in the clip, in the person's voice. This module makes those samples:

    subs_env/bin/python -m stt.ru_samples <season>.references [--takes 6] [--only Name ...] [--force]
                                        [--season-dir PATH]

Per person: the Russian text is taken from the episode's ``subs.ru.json`` (rows over the clip's span, trimmed to about
40 syllables, extended with the person's neighbouring rows when it is too short); X-Voice makes ``--takes`` takes in one
run; each take is trimmed of silence, heard by GigaAM (character errors against the text; a leaked end of the English
reference before the line is cut off when that checks better) and compared with the English clip by the speaker
embedder of the pipeline. The pick is the take within the character-error and length gates with the best similarity
(else the fewest errors). ``<Stem>.ru.wav`` (finished like voice_bank's clips) and ``<Stem>.ru.txt`` are written beside
the English clip and ``ru_samples.json`` lists them; ``references.json`` is not touched (``stt.run`` watches its time).
A person without a usable take or Russian text is remembered in the index (``failed``) and skipped on later runs
until ``--force`` / ``--only`` or a changed English clip (``clip``: episode, span, hash of the wav).
The engine takes them with the setting ``voiceover.engines.f5.reference_language: "ru"``.
X-Voice weights are CC-BY-NC: non-commercial use only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from stt.runner import REPO_ROOT, load_json
from utils.atomic_io import write_text_atomic
from voiceover.text import syllables

log = logging.getLogger("stt.ru_samples")

MAX_SYLLABLES = 40  # a longer text is cut to whole sentences / rows below this
MIN_SYLLABLES = 8  # a shorter text is extended with the person's neighbouring rows
EXTEND_SYLLABLES = 20  # ... up to at least this many
MAX_CER = 0.05
MIN_SPEECH_S, MAX_SPEECH_S = 3.5, 11.0
RATE = 24_000
SIMILARITY_MODEL = "resnet293-ft"
INDEX = "ru_samples.json"
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def _clean(rows: list[dict[str, Any]]) -> str:
    texts = (" ".join(str(t) for t in r.get("text") or []).replace("+", "") for r in rows)
    return " ".join(" ".join(t.split()) for t in texts).strip()


def _spans(ref: dict[str, Any]) -> list[tuple[float, float]]:
    """The clip's source spans in seconds: its pieces when stitched, else ``[start, end]``."""
    pieces = ref.get("pieces") or [[ref["start"], ref["end"]]]
    return [(float(a), float(b)) for a, b in pieces]


def _overlapping(rows: list[dict[str, Any]], spans: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Rows (times in ms) that overlap any span (seconds), in the order of the rows."""
    return [r for r in rows if any(r["start"] < b * 1000 and r["end"] > a * 1000 for a, b in spans)]


def cut_sentences(text: str, limit: int = MAX_SYLLABLES) -> str:
    """The text's first whole sentences up to ``limit`` syllables (the first sentence stays even when it is longer)."""
    kept: list[str] = []
    total = 0
    for sentence in _SENTENCE.split(text):
        n = syllables(sentence)
        if kept and total + n > limit:
            break
        kept.append(sentence)
        total += n
    return " ".join(kept)


def russian_text(rows: list[dict[str, Any]], ref: dict[str, Any]) -> str:
    """Russian text of a reference clip: the translated rows over its span, without stress marks.

    ``rows`` are ``subs.ru.json`` rows (``start`` / ``end`` in ms, ``text`` a list of strings, ``speaker``), the clip's
    ``start`` / ``end`` / ``pieces`` are seconds. Only the clip person's rows count (``ref["speaker"]``, when rows carry
    it). Over ``MAX_SYLLABLES`` the text is cut to whole sentences; under ``MIN_SYLLABLES`` it is extended with the
    person's nearest other rows (by time) until ``EXTEND_SYLLABLES``. Empty when the episode has no such rows."""
    spans = _spans(ref)
    speaker = ref.get("speaker")
    own = [r for r in rows if r.get("speaker") == speaker] if speaker and any(r.get("speaker") == speaker for r in rows) \
        else rows
    chosen = _overlapping(own, spans)
    if syllables(_clean(chosen)) < MIN_SYLLABLES and own:
        middle = 1000 * sum(a + b for a, b in spans) / (2 * len(spans))
        for row in sorted((r for r in own if r not in chosen), key=lambda r: abs((r["start"] + r["end"]) / 2 - middle)):
            if syllables(_clean(chosen)) >= EXTEND_SYLLABLES:
                break
            chosen.append(row)
        chosen.sort(key=lambda r: r["start"])
    return cut_sentences(_clean(chosen))


def select_take(takes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The best take: within the error and length gates with the best similarity, else the fewest errors.

    A take is ``{"cer", "speech_s", "similarity", ...}``; None for no takes."""
    if not takes:
        return None
    ok = [t for t in takes if t["cer"] <= MAX_CER and MIN_SPEECH_S <= t["speech_s"] <= MAX_SPEECH_S]
    if ok:
        return max(ok, key=lambda t: t["similarity"])
    return min(takes, key=lambda t: (t["cer"], -t["similarity"]))


def finish(wav: np.ndarray, rate: int = RATE) -> np.ndarray:
    """A reference clip's last touches (as voice_bank): 50 ms fade in, 100 ms fade out, 1 s of silence at the end."""
    wav = np.array(wav, dtype=np.float32)
    fade = int(0.05 * rate)
    wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:len(wav)]
    wav[-2 * fade:] *= np.linspace(1, 0, 2 * fade, dtype=np.float32)[-len(wav):]
    return np.concatenate([wav, np.zeros(rate, dtype=np.float32)])


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[а-яё]+", text.lower().replace("ё", "е")))


def cer(reference: str, heard: str) -> float:
    """Character error rate of the heard text against the reference (letters only, ё = е)."""
    a, b = _norm(reference), _norm(heard)
    row = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return float(row[len(b)] / max(len(a), 1))


def _windows(audio: np.ndarray, size: int = 16_000, hop: int = 8_000) -> np.ndarray:
    if len(audio) < size:
        audio = np.pad(audio, (0, size - len(audio)))
    return np.stack([audio[i:i + size] for i in range(0, len(audio) - size + 1, hop)])


def _read_mono(path: Path, rate: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    wav = wav.mean(axis=1) if wav.ndim > 1 else wav
    return librosa.resample(wav, orig_sr=sr, target_sr=rate) if sr != rate else wav


def run_worker(items: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]]:
    """Make the takes in ``xvoice_env`` (``voiceover.xvoice_worker``, one model load); return its per-item results."""
    python = REPO_ROOT / "xvoice_env" / "bin" / "python"
    repo = (REPO_ROOT / "models" / "X-Voice").resolve()
    if not python.exists() or not repo.is_dir():
        raise SystemExit("xvoice_env or models/X-Voice missing: create them with `bash scripts/setup_envs.sh xvoice`")
    job, out, log_path = tmp / "job.json", tmp / "results.json", tmp / "worker.log"
    job.write_text(json.dumps({"xvoice_repo": str(repo), "items": items}, ensure_ascii=False), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as worker_log:  # a full pipe would stall the worker
        done = subprocess.run([str(python), "-I", str(REPO_ROOT / "voiceover" / "xvoice_worker.py"), str(job), str(out)],
                              cwd=repo, stderr=worker_log, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"X-Voice worker failed (exit {done.returncode}): "
                           f"{log_path.read_text(encoding='utf-8', errors='replace')[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def score_take(path: Path, text: str, recognizer: Any, embed: Any, reference_print: np.ndarray) -> dict[str, Any] | None:
    """Trim, hear and compare one take; ``{"cer", "speech_s", "similarity", "audio" (24 kHz speech), "cut_s"}``."""
    import librosa

    from voiceover.speech_check import leading_extra

    wav = _read_mono(path, RATE)
    if not np.isfinite(wav).all():  # a broken take (NaN from the model) is dropped, not trimmed
        return None
    speech, _ = librosa.effects.trim(wav, top_db=40)
    if len(speech) < RATE // 2:
        return None
    heard, words = recognizer(speech, RATE)
    error, cut = cer(text, heard), 0.0
    start = leading_extra(text, words)
    if start is not None and start > 0.15:  # the end of the English reference spoken before the line
        trial = speech[int(max(0.0, start - 0.05) * RATE):]
        trial_error = cer(text, recognizer(trial, RATE)[0])
        if trial_error < error:
            speech, error, cut = trial, trial_error, max(0.0, start - 0.05)
    vector = embed.batch(_windows(librosa.resample(speech, orig_sr=RATE, target_sr=16_000))).mean(axis=0)
    return {"cer": round(error, 4), "speech_s": round(len(speech) / RATE, 2), "cut_s": round(cut, 2),
            "similarity": round(float(vector @ reference_print / np.linalg.norm(vector)), 4), "audio": speech}


def clip_id(wav: Path, ref: dict[str, Any]) -> dict[str, Any]:
    """Identity of an English clip: its episode and span plus a hash of the wav (voice_bank may replace it in place)."""
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()[:12] if wav.is_file() else ""
    return {"episode": ref.get("episode"), "start": ref.get("start"), "end": ref.get("end"), "sha": digest}


def make_samples(references: Path, takes: int = 6, only: list[str] | None = None, force: bool = False,
                 device: str = "cuda", season_dir: Path | None = None) -> dict[str, Any]:
    """Make the Russian samples of the references folder; return (and write) the ``ru_samples.json`` content.

    ``season_dir`` holds the episode folders with ``subs.ru.json`` (default: the parent of ``references``); the
    outputs always go next to ``references``."""
    import librosa
    import soundfile as sf

    season = (season_dir or references.resolve().parent)
    refs = load_json(references / "references.json")
    index_path = references / INDEX
    index: dict[str, Any] = load_json(index_path) if index_path.exists() else {}
    wanted = {n.lower() for n in only or []}
    changed = False  # the index is rewritten only then: the dub stage watches its time

    def save() -> None:
        write_text_atomic(index_path, json.dumps(index, ensure_ascii=False, indent=1) + "\n")

    def fail(stem: str, clip: dict[str, Any], reason: str) -> None:
        nonlocal changed
        index[stem] = {"failed": reason, "clip": clip, "tried_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        changed = True
        log.warning("%s: %s, remembered (--force or --only tries again)", stem, reason)

    todo: list[tuple[dict[str, Any], str, str, dict[str, Any]]] = []  # (reference, stem, text, clip id)
    for ref in refs:
        stem = Path(ref["wav"]).stem
        if wanted and str(ref.get("name", "")).lower() not in wanted and stem.lower() not in wanted:
            continue
        clip = clip_id(references / ref["wav"], ref)
        entry = index.get(stem) or {}
        explicit = force or bool(wanted)
        sample = references / f"{stem}.ru.wav"
        if not explicit:
            if entry.get("failed") and entry.get("clip") == clip:
                log.info("%s: failed before (%s), skipped", stem, entry["failed"])
                continue
            if sample.exists() and not entry.get("failed") and (not entry.get("clip") or entry["clip"] == clip):
                log.info("%s: the Russian sample exists, kept", stem)  # an entry without a clip id is taken as current
                continue
        if sample.exists() and not explicit:
            log.info("%s: the English clip changed, the sample is made again", stem)
        if not (season / str(ref.get("episode", ""))).is_dir():
            fail(stem, clip, "clip from another season")
            continue
        subs = season / str(ref.get("episode", "")) / "subs.ru.json"
        if not subs.exists():
            log.warning("%s: no %s, skipped", stem, subs)
            continue
        text = russian_text(load_json(subs), ref)
        if not text:
            fail(stem, clip, f"no Russian rows over the clip {ref['start']:.1f}-{ref['end']:.1f} s")
            continue
        todo.append((ref, stem, text, clip))
    if changed:
        save()
    if not todo:
        log.info("nothing to make")
        return index
    started = time.monotonic()
    made = 0
    with tempfile.TemporaryDirectory(dir=references) as tmp_name:
        tmp = Path(tmp_name)
        items = [{"ref_wav": str((references / ref["wav"]).resolve()), "text": text,
                  "out_wav": str(tmp / f"{stem}_{seed}.wav"), "seed": seed}
                 for ref, stem, text, _ in todo for seed in range(takes)]
        for result in run_worker(items, tmp):
            if result.get("error"):
                log.warning("%s: %s", Path(result["out_wav"]).name, result["error"])
        from stt.refine_speakers import make_embedder
        from voiceover.speech_check import Recognizer

        recognizer, embed = Recognizer(device), make_embedder(SIMILARITY_MODEL, device)
        for ref, stem, text, clip in todo:
            english, _ = librosa.effects.trim(_read_mono(references / ref["wav"], 16_000), top_db=40)
            reference_print = embed.batch(_windows(english)).mean(axis=0)
            reference_print = reference_print / np.linalg.norm(reference_print)
            scored: list[dict[str, Any]] = []
            for seed in range(takes):
                path = tmp / f"{stem}_{seed}.wav"
                if path.exists() and (one := score_take(path, text, recognizer, embed, reference_print)):
                    scored.append({"take": seed, **one})
            best = select_take(scored)
            if best is None:
                for old in (references / f"{stem}.ru.wav", references / f"{stem}.ru.txt"):
                    old.unlink(missing_ok=True)  # a sample of another clip would clone the wrong person
                fail(stem, clip, "no usable take")
                save()
                continue
            sf.write(str(references / f"{stem}.ru.wav"), finish(best.pop("audio")), RATE, subtype="PCM_16")
            write_text_atomic(references / f"{stem}.ru.txt", text + "\n")
            index[stem] = {"text": text, "take": best["take"], "cer": best["cer"], "similarity": best["similarity"],
                           "speech_s": best["speech_s"], "episode": ref.get("episode"), "clip": clip,
                           "made_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            made += 1
            log.info("%s: take %d, CER %.1f%%, similarity %.3f, %.1f s: %s", stem, best["take"], 100 * best["cer"],
                     best["similarity"], best["speech_s"], text)
            # after every person: a crash later in the run must not leave samples without their entry
            save()
    log.info("%d samples made in %d s", made, round(time.monotonic() - started))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("references", type=Path, help="the season's <Season>.references folder")
    parser.add_argument("--takes", type=int, default=6, help="takes per person (seeds 0..n-1)")
    parser.add_argument("--only", nargs="+", help="make only these people (name or file stem); remakes existing ones")
    parser.add_argument("--force", action="store_true", help="make samples that exist again")
    parser.add_argument("--season-dir", type=Path, help="folder of the episode folders with subs.ru.json "
                        "(default: the parent of the references folder)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    make_samples(args.references, args.takes, args.only, args.force, args.device, args.season_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
