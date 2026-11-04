"""A voice-cloning reference clip made by hand from time ranges of an episode.

For a person the automatic export finds no clean clip for (``missing.json`` says why), someone watches
the episode, finds where the person speaks alone (an interview) and gives the ranges, e.g. ``12:30-12:41``.
Each range is cut from the episode video, cleaned from music and other sound with the same vocal
separator the automatic clips use, trimmed of silence at both ends; the pieces are joined with short
pauses into 6-10 s of speech, ended like an automatic clip (fades, 1 s of silence). Whisper writes the
transcript from the finished clip, as in the automatic export.

    uv run -m stt.voice_bank "S.voices.json" --manual-clip --person "Saiounia Hughley" --media ep1.mkv \\
        --range 12:30-12:41 --range 1:02:05-1:02:14 [--preview clip.wav]
    uv run -m stt.voice_bank "S.voices.json" --remove-manual --person "Saiounia Hughley"

Saved clips are ``<refs>/<name>.manual.wav`` + ``.txt`` and an entry in ``<refs>/manual.json`` (the entry
also goes into ``references.json``). No export ever replaces or drops a manual clip; removing it brings
back the person's automatic clip if there was one, else the next ``--missing`` export looks for one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stt import voice_bank
from stt.voice_bank import (CLONE_RATE, MIN_SAMPLE_SCORE, SegmentEmbedder, VoiceBank, _read_media, _safe_name, _unit,
                            load_manual, same_person, write_index)
from utils.atomic_io import fsync_file, write_text_atomic

log = logging.getLogger(__name__)

MAX_RANGE_S = 60.0  # one range may not be longer (cutting and cleaning a long one is slow and pointless)
MAX_CLIP_S = 10.0  # speech (with pauses) in a clip; F5 clips references over 12 s, 1 s of silence is added
MIN_CLIP_S = 3.0  # less speech than this gets a warning (6-10 s is good)
PAUSE_S = 0.3  # silence between joined pieces
EDGE_PAD_S = 0.1  # silence kept around a trimmed piece
SILENCE_DB = -30.0  # frames this far under a piece's loud frames count as silence
MIN_PAUSE_S = 0.25  # the shortest silence a long piece may be cut at
MUSIC_WARN_DB = 6.0  # accompaniment this much louder than the voice (measured before cleaning) gets a warning;
# interviews over music read -2..+4 dB and clean up fine

_TIME = r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?"
_RANGE = re.compile(rf"^({_TIME})\s*[-–—]\s*({_TIME})$")


def parse_time(text: str) -> float:
    """``mm:ss``, ``h:mm:ss`` or seconds (``75``, ``75.5``) -> seconds."""
    *head, seconds = text.split(":")
    if head and (float(seconds) >= 60 or any(int(x) >= 60 for x in head[1:])):
        raise ValueError(f"В времени «{text}» минут или секунд больше 59.")
    value = 0
    for x in head:
        value = value * 60 + int(x)
    return value * 60 + float(seconds)


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """Time ranges, one per line or separated by commas/semicolons: ``12:30-12:41``, ``1:02:05 - 1:02:14``,
    ``750-761.5`` (the dash may be ``-``, an en dash or an em dash). Errors are ``ValueError`` with a
    Russian message for the user."""
    ranges: list[tuple[float, float]] = []
    for raw in re.split(r"[\n,;]+", text):
        item = raw.strip()
        if not item:
            continue
        found = _RANGE.match(item)
        if not found:
            raise ValueError(f"Не понял отрезок «{item}». Нужен вид 12:30-12:41, 1:02:05-1:02:14 или 750-761.5 "
                             "(от и до, минуты:секунды или секунды).")
        start, end = parse_time(found.group(1)), parse_time(found.group(2))
        if end <= start:
            raise ValueError(f"В отрезке «{item}» конец не позже начала.")
        if end - start > MAX_RANGE_S:
            raise ValueError(f"Отрезок «{item}» длиннее {MAX_RANGE_S:.0f} секунд ({end - start:.0f} с): "
                             "достаточно 6-10 секунд речи, возьмите кусок покороче.")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("Не указан ни один отрезок. Пример: 12:30-12:41")
    return ranges


def format_time(seconds: float) -> str:
    whole = int(seconds)
    h, rest = divmod(whole, 3600)
    m, s = divmod(rest, 60)
    tail = f".{int(round((seconds - whole) * 10)) % 10}" if seconds != whole else ""
    return f"{h}:{m:02d}:{s:02d}{tail}" if h else f"{m}:{s:02d}{tail}"


class ClipModels:
    """The heavy models a clip needs, loaded on first use; keep one instance to make several clips."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._separator: Any = None
        self._embedder: Any = None
        self._transcriber: Any = None

    @property
    def separator(self) -> Any:
        if self._separator is None:
            from stt.separate import VocalSeparator

            self._separator = VocalSeparator(self.device)
        return self._separator

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = SegmentEmbedder(self.device)
        return self._embedder

    @property
    def transcriber(self) -> Any:
        if self._transcriber is None:
            self._transcriber = voice_bank.load_transcriber(self.device)
        return self._transcriber


@dataclass
class ManualClip:
    wav: Path
    duration: float  # seconds of speech with the pauses between pieces (the 1 s of silence at the end excluded)
    text: str
    pieces: list[dict[str, Any]] = field(default_factory=list)  # per range: start, end, seconds (used), music_db
    warnings: list[str] = field(default_factory=list)  # Russian, for the user
    similarity: float | None = None  # to the person's prints in the bank, if it has any
    music_db: float | None = None  # the loudest accompaniment of the pieces, relative to the voice


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.zeros(0)
    return np.sqrt((x[:n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(1))


def _silent_frames(x: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """(mask of silent 20 ms frames, frame length); all frames silent when the piece holds no voice."""
    frame = int(0.02 * rate)
    rms = _frame_rms(x, frame)
    if rms.size == 0:
        return np.ones(0, dtype=bool), frame
    loud = float(np.percentile(rms, 95))
    if loud < 1e-4:
        return np.ones(rms.size, dtype=bool), frame
    return rms < loud * 10 ** (SILENCE_DB / 20), frame


def trim_silence(x: np.ndarray, rate: int) -> np.ndarray | None:
    """The piece without silence at both ends (``EDGE_PAD_S`` kept); None when it holds no voice."""
    silent, frame = _silent_frames(x, rate)
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return None
    pad = int(EDGE_PAD_S * rate)
    return x[max(0, int(voiced[0]) * frame - pad):min(len(x), (int(voiced[-1]) + 1) * frame + pad)]


def _cut_at_pause(x: np.ndarray, rate: int, limit_s: float) -> np.ndarray | None:
    """The part of a piece up to its last pause (at least ``MIN_PAUSE_S`` of silence) that ends within
    ``limit_s``; None when there is no pause that early."""
    silent, frame = _silent_frames(x, rate)
    best: int | None = None
    run = 0
    for i, quiet in enumerate(silent):
        run = run + 1 if quiet else 0
        if run * frame / rate >= MIN_PAUSE_S:
            begin = i - run + 1
            cut = begin * frame + int(EDGE_PAD_S * rate)
            if cut / rate <= limit_s and begin > 0:
                best = cut
    return x[:best] if best else None


def _join(pieces: list[np.ndarray], rate: int) -> np.ndarray:
    pause = np.zeros(int(PAUSE_S * rate), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, piece in enumerate(pieces):
        if i:
            out.append(pause)
        out.append(piece)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def make_manual_clip(media: Path, ranges: list[tuple[float, float]], out_wav: Path, *, device: str = "cuda",
                     bank: VoiceBank | None = None, person: str | None = None, text: str | None = None,
                     models: ClipModels | None = None) -> ManualClip:
    """Cut, clean and join the ranges of ``media`` into ``out_wav`` (24 kHz mono PCM16 reference clip).

    Heavy: loads the vocal separator and Whisper (about 20-30 s on a GPU the first time, a few seconds
    per range later with ``models`` kept); the embedder only when ``bank`` and ``person`` give prints to
    compare with. ``text`` replaces the Whisper transcript. Never writes into the season: only ``out_wav``.
    """
    import soundfile as sf
    import soxr

    if not ranges:
        raise ValueError("Не указан ни один отрезок.")
    if not Path(media).exists():
        raise FileNotFoundError(f"Нет файла с видео: {media}")
    models = models or ClipModels(device)
    separate = models.separator
    warnings: list[str] = []
    used: list[np.ndarray] = []
    pieces: list[dict[str, Any]] = []
    total = 0.0
    for start, end in ranges:
        label = f"{format_time(start)}-{format_time(end)}"
        stereo = _read_media(str(media), start, end - start, separate.rate, 2)
        if stereo.shape[1] < separate.rate // 2:
            warnings.append(f"Отрезок {label}: в видео там почти нет звука (видео короче?).")
            continue
        vocals, music_db = separate(stereo)
        mono = soxr.resample(vocals.mean(0), separate.rate, CLONE_RATE).astype(np.float32)
        piece = trim_silence(mono, CLONE_RATE)
        if piece is None:
            warnings.append(f"Отрезок {label}: после очистки в нём нет голоса, пропущен.")
            continue
        if music_db > MUSIC_WARN_DB:
            warnings.append(f"Отрезок {label}: рядом громко звучит музыка или шум ({music_db:+.0f} дБ к голосу), "
                            "голос может получиться с примесью.")
        gap = PAUSE_S if used else 0.0
        seconds = len(piece) / CLONE_RATE
        if total + gap + seconds > MAX_CLIP_S:
            room = MAX_CLIP_S - total - gap
            cut = _cut_at_pause(piece, CLONE_RATE, room) if room >= 1.0 else None
            if cut is None and not used:  # one long piece without a pause: cut it hard
                cut = piece[:int(MAX_CLIP_S * CLONE_RATE)]
                warnings.append(f"Отрезок {label} длиннее {MAX_CLIP_S:.0f} с и без пауз: обрезан на {MAX_CLIP_S:.0f} с.")
            if cut is not None:
                used.append(cut)
                total += gap + len(cut) / CLONE_RATE
                pieces.append({"start": start, "end": end, "seconds": round(len(cut) / CLONE_RATE, 2),
                               "music_db": music_db, "cut": True})
            warnings.append(f"Достаточно {MAX_CLIP_S:.0f} секунд речи: отрезок {label} "
                            f"{'вошёл не целиком' if cut is not None else 'не вошёл'}, следующие не вошли.")
            break
        used.append(piece)
        total += gap + seconds
        pieces.append({"start": start, "end": end, "seconds": round(seconds, 2), "music_db": music_db})
    if not used:
        raise ValueError("Ни в одном из отрезков не нашлось голоса. " + " ".join(warnings))
    speech = _join(used, CLONE_RATE)
    peak = float(np.abs(speech).max())
    if peak > 0.99:
        speech = speech * (0.99 / peak)
    voice = voice_bank._finish_voice(speech)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), voice, CLONE_RATE, subtype="PCM_16")
    duration = len(speech) / CLONE_RATE
    if duration < MIN_CLIP_S:
        warnings.append(f"Речи всего {duration:.1f} с: клонирование лучше работает на 6-10 секундах, "
                        "добавьте ещё отрезок.")
    similarity: float | None = None
    mine = [_unit(p["embedding"]) for p in bank.prints if person and same_person(p["name"], person)] if bank else []
    if mine:
        vector = models.embedder(soxr.resample(speech, CLONE_RATE, 16_000).astype(np.float32), 16_000)
        similarity = round(max(float(vector @ q) for q in mine), 3)
        if similarity < MIN_SAMPLE_SCORE:
            warnings.append(f"Голос слабо похож на этого человека в других эпизодах ({similarity:.2f} при норме "
                            f"от {MIN_SAMPLE_SCORE:.2f}): проверьте, что в этих отрезках говорит именно он и никто "
                            "больше.")
    if text is None:
        text = voice_bank.transcribe_clip(models.transcriber, voice)
        if not text:
            warnings.append("Whisper не расслышал в клипе слов: проверьте отрезки или впишите текст вручную.")
    clip = ManualClip(out_wav, round(duration, 2), text.strip(), pieces, warnings, similarity,
                      max((p["music_db"] for p in pieces), default=None))
    log.info("manual clip %s: %.1f s from %d range(s), similarity %s, %d warning(s)", out_wav.name, duration,
             len(pieces), similarity, len(warnings))
    return clip


def _known_gender(bank: VoiceBank | None, name: str) -> str | None:
    """Gender from the bank's prints, else from the season cast file next to the bank."""
    if bank is None:
        return None
    person = next((p for p in bank.people() if same_person(p["name"], name)), None)
    if person and person.get("gender") in ("male", "female"):
        return str(person["gender"])
    cast = bank.path.with_name(bank.path.name.replace(".voices.json", ".cast.json"))
    if cast.exists():
        for member in json.loads(cast.read_text(encoding="utf-8")).get("cast", []):
            names = [member.get("name", ""), *member.get("aliases", [])]
            if any(n and same_person(n, name) for n in names) and member.get("gender") in ("male", "female"):
                return str(member["gender"])
    return None


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        fsync_file(tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_manual_clip(refs_dir: Path, name: str, clip: ManualClip | Path, *, text: str | None = None,
                     gender: str | None = None, label_ru: str | None = None, episode: str = "",
                     ranges: list[tuple[float, float]] | None = None, bank: VoiceBank | None = None
                     ) -> dict[str, Any]:
    """Put a hand-made clip into the season's clip folder as ``name``'s voice; returns its index entry.

    Writes ``<name>.manual.wav``/``.txt``, the entry in ``manual.json`` and the merged ``references.json``
    (last, as stt.run watches it). A person who is not in the bank needs ``gender`` (else ``ValueError``);
    ``label_ru`` falls back to the bank, then to the name. Exports never touch the entry afterwards.
    """
    import soundfile as sf

    wav = clip.wav if isinstance(clip, ManualClip) else Path(clip)
    text = (text if text is not None else clip.text if isinstance(clip, ManualClip) else "") or ""
    if not text.strip():
        raise ValueError("Нужен текст клипа: впишите, что в нём говорят.")
    person = next((p for p in bank.people() if same_person(p["name"], name)), None) if bank else None
    refs_dir.mkdir(parents=True, exist_ok=True)
    existing = next((r for r in voice_bank.load_json_list(refs_dir / "references.json")
                     if same_person(r["name"], name)), None)
    gender = gender or (person or {}).get("gender") or _known_gender(bank, name) or (existing or {}).get("gender")
    if gender not in ("male", "female"):
        raise ValueError("Не знаю пол этого человека: укажите male или female (мужской или женский).")
    full = (person or {}).get("name") or name
    manual = load_manual(refs_dir)
    previous = next((m for m in manual if same_person(m["name"], name)), None)
    replaced = previous["replaced"] if previous else (existing if existing and existing.get("source") != "manual"
                                                      else None)
    stem = f"{_safe_name(full)}.manual"
    ranges = [(float(a), float(b)) for a, b in ranges or []]
    speech = clip.duration if isinstance(clip, ManualClip) else max(0.0, sf.info(str(wav)).duration - 1.0)
    score = clip.similarity if isinstance(clip, ManualClip) and clip.similarity is not None else 0.0
    music = clip.music_db if isinstance(clip, ManualClip) and clip.music_db is not None else 0.0
    entry = {"name": full, "label_ru": label_ru or (person or {}).get("label_ru") or (existing or {}).get("label_ru")
             or full, "gender": gender, "wav": f"{stem}.wav", "text": text.strip(), "source": "manual",
             "episode": episode, "ranges": [list(r) for r in ranges], "duration": round(speech, 2),
             # the fields the automatic entries have, so that readers of the index need no special case
             "start": ranges[0][0] if ranges else 0.0, "end": ranges[-1][1] if ranges else 0.0,
             "score": score, "min_window": score, "music_db": music, "replaced": replaced}
    _copy_atomic(wav, refs_dir / f"{stem}.wav")
    write_text_atomic(refs_dir / f"{stem}.txt", entry["text"] + "\n")
    manual = [m for m in manual if not same_person(m["name"], name)] + [entry]
    write_text_atomic(refs_dir / "manual.json", json.dumps(manual, ensure_ascii=False, indent=1) + "\n")
    missing = voice_bank.load_missing(refs_dir)
    if any(same_person(k, name) for k in missing):  # the reasons no longer apply
        write_text_atomic(refs_dir / "missing.json", json.dumps(
            {k: v for k, v in missing.items() if not same_person(k, name)}, ensure_ascii=False, indent=1) + "\n")
    write_index(refs_dir, voice_bank.load_json_list(refs_dir / "references.json"))
    return {k: v for k, v in entry.items() if k != "replaced"}


def remove_manual_clip(refs_dir: Path, name: str) -> bool:
    """Delete a person's hand-made clip; the automatic clip it replaced comes back when its file is still
    there, else the next ``--missing`` export looks for one. False when the person had no manual clip."""
    manual = load_manual(refs_dir)
    entry = next((m for m in manual if same_person(m["name"], name)), None)
    if entry is None:
        return False
    write_text_atomic(refs_dir / "manual.json", json.dumps([m for m in manual if m is not entry],
                                                           ensure_ascii=False, indent=1) + "\n")
    stem = Path(entry["wav"]).stem
    (refs_dir / f"{stem}.wav").unlink(missing_ok=True)
    (refs_dir / f"{stem}.txt").unlink(missing_ok=True)
    auto = voice_bank.load_json_list(refs_dir / "references.json")
    auto = [r for r in auto if not same_person(r["name"], name)]
    old = entry.get("replaced")
    if old and (refs_dir / old["wav"]).exists():
        auto.append(old)
    write_index(refs_dir, auto)
    return True


def episode_for_media(bank_path: Path, media: Path) -> str:
    """The bank's episode folder name whose ``source.json`` names this video, else the video's file name."""
    for source in sorted(bank_path.parent.glob("*/source.json")):
        try:
            found = json.loads(source.read_text(encoding="utf-8")).get("media")
        except (OSError, ValueError):
            continue
        if found and Path(found).resolve() == Path(media).resolve():
            return source.parent.name
    return Path(media).stem


def clip_json(clip: ManualClip) -> dict[str, Any]:
    """The machine-readable form of a cut clip (``--json``): what the UI shows and what ``save_manual_clip`` needs."""
    return {"wav": str(clip.wav), "duration": clip.duration, "text": clip.text, "warnings": clip.warnings,
            "similarity": clip.similarity, "music_db": clip.music_db,
            "pieces": [[p["start"], p["end"]] for p in clip.pieces]}


def run_cli(args: Any, bank: VoiceBank) -> int:
    """``--manual-clip`` / ``--remove-manual`` of ``stt.voice_bank`` (``args`` is its parsed namespace).

    With ``--json`` the result of ``--manual-clip --preview`` is one JSON line on stdout (``clip_json``), or
    ``{"error": "<Russian message>"}`` with exit code 2."""
    import tempfile

    as_json = bool(getattr(args, "json", False))

    def fail(message: object) -> int:
        print(json.dumps({"error": str(message)}, ensure_ascii=False) if as_json else message)
        return 2

    refs = args.refs or voice_bank.references_dir(bank.path)
    if not args.person:
        return fail("Нужно имя: --person NAME")
    if args.remove_manual:
        if remove_manual_clip(refs, args.person):
            print(f"manual clip of {args.person} removed from {refs}; references.json updated")
        else:
            print(f"{args.person} has no manual clip in {refs}")
        return 0
    if not args.media or not args.range:
        return fail("Нужны --media VIDEO и хотя бы один --range 12:30-12:41")
    try:
        ranges = parse_ranges("\n".join(args.range))
    except ValueError as exc:
        return fail(exc)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            clip = make_manual_clip(args.media, ranges, args.preview or Path(tmp) / "clip.wav", device=args.device,
                                    bank=bank, person=args.person, text=args.text)
        except (ValueError, FileNotFoundError) as exc:
            return fail(exc)
        if as_json:
            print(json.dumps(clip_json(clip), ensure_ascii=False))
            return 0
        _print_clip(clip)
        if args.preview:
            print(f"preview only: {args.preview} (nothing written in {refs})")
            return 0
        try:
            entry = save_manual_clip(refs, args.person, clip, text=args.text, gender=args.gender, bank=bank,
                                     episode=episode_for_media(bank.path, args.media),
                                     ranges=[(p["start"], p["end"]) for p in clip.pieces])
        except ValueError as exc:
            return fail(exc)
    print(f"saved: {refs / entry['wav']}, {(refs / entry['wav']).with_suffix('.txt')}")
    print(f"index: {refs / 'manual.json'}, {refs / 'references.json'}")
    return 0


def _print_clip(clip: ManualClip) -> None:
    print(f"clip: {clip.duration:.1f} s of speech from {len(clip.pieces)} range(s), similarity "
          f"{'-' if clip.similarity is None else format(clip.similarity, '.2f')}")
    print(f"text: {clip.text}")
    for warning in clip.warnings:
        print(f"warning: {warning}")
