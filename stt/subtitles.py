"""Existing subtitles of an episode as a source of who speaks: "NAME:" labels (SDH) and ">>" turn marks (broadcast captions).

Subtitle times are off by about 0.3 s and one cue may hold several voices, so every piece of text is pinned to the
recognized words around it; a piece the audio does not confirm is dropped.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import subprocess
import tempfile
import time
import zipfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

log = logging.getLogger("stt.subtitles")

# Labels naming no single person.
GENERIC_LABELS = {
    "MAN", "WOMAN", "MEN", "WOMEN", "ALL", "BOTH", "SURVIVORS", "HEROES", "VILLAINS", "TOGETHER", "CROWD", "GROUP",
    "NARRATOR", "VOICE", "PRODUCER", "OTHERS", "EVERYONE", "PEOPLE", "TEAM", "TRIBE", "ANNOUNCER", "SPEAKER", "BOY",
    "GIRL", "PLAYERS", "CONTESTANTS", "MAN ON TV", "WOMAN ON TV", "CAMERAMAN", "CHILD", "KIDS", "HOST", "REPORTER",
    "SINGER", "SINGERS", "NOTE", "NOTES", "SIGN", "CAPTION", "NEXT", "NEXT TIME", "PREVIOUSLY", "THANKS",
    "SOMEONE", "ANYONE", "NOBODY", "CHILDREN", "KID", "LADY", "LADIES", "GUY", "GUYS", "CHYRON", "TITLE", "TEXT",
    "SUBTITLE", "SUBTITLES", "TRANSLATION", "TRANSLATOR", "INTERPRETER", "PLAYER", "WINNER", "ANSWER", "QUESTION",
    "WARNING", "REMINDER", "TIP", "EXAMPLE", "FACT", "RULES", "SCORE", "TIME", "TODAY", "TONIGHT", "TOMORROW",
    "SOMETHING", "LOOK", "LISTEN", "WELL", "DESTINATION", "LOCATION", "CHALLENGE", "REWARD", "IMMUNITY",
}
MIN_SOFT_LABEL_USES = 3  # a mixed or lower case name is a label only when the file uses it at least this often
HOST_LABELS = {"PROBST", "JEFF PROBST", "JEFF", "JONATHAN LAPAGLIA", "LAPAGLIA", "HOST JEFF"}
HOST = "_HOST"

_LABEL = re.compile(r"^\s*(?:(?:[-–—]|>>+)\s*)?([A-Za-z][A-Za-z .'’-]{1,23}):\s*(.*)$")
_MAX_CAPS_LABEL = 21
_MAX_SOFT_LABEL_WORDS = 3
_DASH = re.compile(r"^\s*[-–—]\s*(.*)$")
_TURN = re.compile(r"^\s*>>+\s*(.*)$")
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]|<[^>]+>|\{[^}]*\}")
_TOKEN = re.compile(r"[a-z0-9']+")
_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")
_NOT_PERSON = re.compile(r"\b(MAN|WOMAN|VOICE|ANNOUNCER)\b|\d")


@dataclass
class Piece:
    """One voice's text inside a cue. ``person`` is "" when the subtitles do not name it."""

    person: str
    text: str
    cue_start: float
    cue_end: float
    tokens: list[str] = field(default_factory=list)
    opens_turn: bool = False  # opened by a dash, ">>" or a label: another voice than the piece before
    turn_mark: bool = False  # opened by ">>": broadcast captions mark every change of voice this way


@dataclass
class PinnedPiece:
    """A piece matched against recognized words: ``first`` and ``last`` index the word list."""

    piece: Piece
    first: int
    last: int
    start: float
    end: float
    match: float


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("’", "'"))


def clean_person(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw.replace("’", "'")).strip(" .-").upper()
    return HOST if name in HOST_LABELS else name


def names_person(person: str) -> bool:
    """Whether a cleaned label names one person (not "MAN", "ALL", "WOMAN 2", ...)."""
    return bool(person) and person not in GENERIC_LABELS and not _NOT_PERSON.search(person)


def parse_cues(path: Path) -> list[tuple[float, float, list[str]]]:
    """(start, end, lines) of every cue in seconds; tolerant: the files found next to videos are not all clean SRT."""
    return parse_cue_text(path.read_text(encoding="utf-8", errors="replace"))


def parse_cue_text(text: str) -> list[tuple[float, float, list[str]]]:
    """``parse_cues`` for text that is already in memory."""
    cues: list[tuple[float, float, list[str]]] = []
    start: float | None = None
    end = 0.0
    lines: list[str] = []
    for raw in text.splitlines():
        if "-->" in raw:
            if start is not None:
                cues.append((start, end, lines))
            stamps = _TIME.findall(raw)
            if len(stamps) < 2:
                start = None
                continue
            start, end = (int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000 for h, m, s, ms in stamps[:2])
            lines = []
        elif start is not None and raw.strip() and not raw.strip().isdigit():
            lines.append(raw.strip())
    if start is not None:
        cues.append((start, end, lines))
    return cues


def _is_label(match: re.Match[str]) -> bool:
    """All-caps captions: "I THINK: YES" is text, not a label; the text after a real label is in mixed case."""
    rest = match.group(2)
    return not rest or len(tokens(rest)) <= 1 or rest != rest.upper()


def label_of(text: str) -> tuple[str, str, bool] | None:
    """(name, rest of the line, soft) when the line opens with "NAME:" (after an optional dash or ">>"), else None.
    An ALL-CAPS name is a label as it always was; a name in any other case ("Jeff", "jeff", "Jeff Probst") is *soft*:
    it must name a person (not "Note") and the file must use it often enough (``soft_label_names``)."""
    match = _LABEL.match(text)
    if not match:
        return None
    name, rest = match.group(1), match.group(2)
    if name == name.upper():
        return (name, rest, False) if len(name) <= _MAX_CAPS_LABEL and _is_label(match) else None
    if len(name.split()) > _MAX_SOFT_LABEL_WORDS or not names_person(clean_person(name)):
        return None
    return name, rest, True


def soft_label_names(cues: Sequence[tuple[float, float, list[str]]], min_uses: int = MIN_SOFT_LABEL_USES) -> set[str]:
    """Cleaned mixed/lower case names that the file uses as a label at least ``min_uses`` times: "Jeff:" at the start of
    many lines is a speaker, "Their destination:" once is a sentence."""
    counts: Counter[str] = Counter()
    for _, _, lines in cues:
        for line in lines:
            found = label_of(_NOISE.sub(" ", line).strip())
            if found and found[2]:
                counts[clean_person(found[0])] += 1
    return {name for name, n in counts.items() if n >= min_uses}


def pieces_of(cue: tuple[float, float, list[str]], soft_names: set[str] | None = None) -> list[Piece]:
    """Split one cue by voices: a label, a dash or ">>" opens a piece, other lines continue the current one.
    ``soft_names`` (``soft_label_names`` of the file): the mixed/lower case names that count as labels; None: all."""
    start, end, lines = cue
    out: list[Piece] = []
    for line in lines:
        text = _NOISE.sub(" ", line).strip()
        label = label_of(text)
        if label and label[2] and soft_names is not None and clean_person(label[0]) not in soft_names:
            label = None
        turn = _TURN.match(text) or _DASH.match(text)
        mark = text.startswith(">>")
        if label:
            out.append(Piece(clean_person(label[0]), label[1], start, end, opens_turn=True, turn_mark=mark))
        elif turn:
            out.append(Piece("", turn.group(1), start, end, opens_turn=True, turn_mark=mark))
        elif out:
            out[-1].text += " " + text
        else:
            out.append(Piece("", text, start, end))
    for piece in out:
        piece.tokens = tokens(piece.text)
    return out


def file_pieces(cues: Sequence[tuple[float, float, list[str]]]) -> list[Piece]:
    """Pieces of a whole file: soft (mixed/lower case) labels only where the file uses the name often enough."""
    accepted = soft_label_names(cues)
    return join_continued([p for cue in cues for p in pieces_of(cue, accepted)])


def join_continued(pieces: list[Piece], max_gap: float = 1.0) -> list[Piece]:
    """A labelled sentence running on into the next cue stays with its person (the label is written only once)."""
    out: list[Piece] = []
    for piece in pieces:
        last = out[-1] if out else None
        if (last is not None and last.person and not piece.person and not piece.opens_turn and piece.tokens
                and not re.search(r"[.?!…:]\W*$", last.text.strip()) and piece.cue_start - last.cue_end <= max_gap):
            last.text = f"{last.text} {piece.text}".strip()
            last.cue_end = piece.cue_end
            last.tokens += piece.tokens
            continue
        out.append(piece)
    return out


def read_pieces(path: Path) -> list[Piece]:
    return file_pieces(parse_cues(path))


def subtitle_kind(pieces: Sequence[Piece]) -> str:
    """"labels", "turns" or "plain": what the file can tell about who speaks."""
    labelled = sum(names_person(p.person) for p in pieces)
    turns = sum(p.turn_mark or (p.opens_turn and not p.person) for p in pieces)
    if labelled >= 20:
        return "labels"
    return "turns" if turns >= 20 else "plain"


class WordIndex:
    """Recognized words as tokens, for quick lookups by time."""

    def __init__(self, words: Sequence[Any]) -> None:
        self.index = [i for i, w in enumerate(words) if tokens(w.text) and w.end > w.start and getattr(w, "source", None) != "ocr"]
        self.toks = ["".join(tokens(words[i].text)) for i in self.index]
        self.starts = [float(words[i].start) for i in self.index]
        self.ends = [float(words[i].end) for i in self.index]  # ascending in practice: Whisper words do not cross

    def between(self, lo: float, hi: float) -> tuple[int, int]:
        """Positions [a, b) of the words touching the interval."""
        a = bisect_right(self.ends, lo)
        b = a
        while b < len(self.starts) and self.starts[b] < hi:
            b += 1
        return a, b


def _shifts(pieces: Sequence[Piece], index: WordIndex, wide_s: float) -> list[tuple[float, float]]:
    """(cue start, shift cue -> words) of the long lines that the words clearly hold nearby."""
    found: list[tuple[float, float]] = []
    for piece in pieces:
        if len(piece.tokens) < 5:
            continue
        a, b = index.between(piece.cue_start - wide_s, piece.cue_end + wide_s)
        near = index.toks[a:b]
        block = SequenceMatcher(None, piece.tokens, near, autojunk=False).find_longest_match(0, len(piece.tokens), 0, len(near))
        if block.size >= 5:
            found.append((piece.cue_start, index.starts[a + block.b] - block.a * 0.3 - piece.cue_start))
    return found


def local_offsets(pieces: Sequence[Piece], index: WordIndex, wide_s: float = 120.0, span: int = 15) -> list[list[float]]:
    """Shifts to try per piece: the median over the nearest long lines, then the extremes of those lines (a piece next to
    a cut). Another cut of the episode (no ads, a recap removed) moves the subtitles by a different amount per act, so
    one number for the file is not enough. Within a second the subtitles count as in sync: shift 0."""
    found = _shifts(pieces, index, wide_s)
    if not found:
        return [[0.0] for _ in pieces]
    times = [t for t, _ in found]
    shifts = [s for _, s in found]
    out: list[list[float]] = []
    for piece in pieces:
        k = bisect_right(times, piece.cue_start)
        near = sorted(shifts[max(k - span, 0):k + span])
        middle = near[len(near) // 2]
        tries = [middle if abs(middle) > 1.0 else 0.0]
        if len(piece.tokens) >= 3:  # a short piece would match by chance elsewhere
            tries += [s for s in (near[0], near[-1]) if abs(s - tries[0]) > 1.0]
        out.append(tries)
    return out


def pin_piece(piece: Piece, index: WordIndex, offset: float = 0.0, margin: float = 1.5, min_share: float = 0.6) -> PinnedPiece | None:
    """The run of words of a piece near its cue; None when the audio does not confirm the text."""
    if not piece.tokens:
        return None
    a, b = index.between(piece.cue_start + offset - margin, piece.cue_end + offset + margin)
    near = index.toks[a:b]
    if not near:
        return None
    blocks = [x for x in SequenceMatcher(None, piece.tokens, near, autojunk=False).get_matching_blocks() if x.size]
    if not blocks:
        # a short piece may differ in spelling ("okay" / "ok"): accept one close word
        joined = "".join(piece.tokens)
        best, k = max((SequenceMatcher(None, joined, t).ratio(), j) for j, t in enumerate(near))
        if len(piece.tokens) > 2 or best < 0.75:
            return None
        where = a + k
        return PinnedPiece(piece, index.index[where], index.index[where], index.starts[where], index.ends[where], round(best, 3))
    share = sum(x.size for x in blocks) / len(piece.tokens)
    if share < (1.0 if len(piece.tokens) <= 2 else min_share):
        return None
    first, last = a + blocks[0].b, a + blocks[-1].b + blocks[-1].size - 1
    if last - first + 1 > 2 * len(piece.tokens) + 3:  # scattered matches: not this line
        return None
    return PinnedPiece(piece, index.index[first], index.index[last], index.starts[first], index.ends[last], round(share, 3))


def pin_pieces(pieces: Sequence[Piece], words: Sequence[Any]) -> tuple[list[PinnedPiece], float]:
    """Every piece that the words confirm, and the median time shift found."""
    index = WordIndex(words)
    tries = local_offsets(pieces, index)
    pinned: list[PinnedPiece] = []
    for piece, shifts in zip(pieces, tries):
        hit = next((p for p in (pin_piece(piece, index, s) for s in shifts) if p is not None), None)
        if hit is not None:
            pinned.append(hit)
    middle = sorted(s[0] for s in tries)
    return pinned, middle[len(middle) // 2] if middle else 0.0


def labelled(pinned: Sequence[PinnedPiece]) -> list[PinnedPiece]:
    return [p for p in pinned if names_person(p.piece.person)]


def turn_starts(pinned: Sequence[PinnedPiece]) -> list[int]:
    """Word indices where the subtitles open another voice."""
    return sorted({p.first for p in pinned if p.piece.opens_turn})


def turn_marks(pinned: Sequence[PinnedPiece]) -> list[int]:
    """Word indices where a ">>" mark opens another voice."""
    return sorted({p.first for p in pinned if p.piece.turn_mark})


# ---- finding subtitles of an episode: inside the video, next to it, in a local archive, online

SOURCE_SRT = "subs.source.en.srt"
BASE_INDEX = "index.json"
ARCHIVE_IMDB = "tt0239195"  # the hunt archive (``subtitles.archive``) holds Survivor US only
SOURCE_JSON = "subs.source.json"
_KIND_RANK = {"labels": 2, "turns": 1, "plain": 0}
_EPISODE_CODES = (re.compile(r"(?<![a-z0-9])s(\d{1,2})[\s._-]?e(\d{1,3})(?!\d)", re.I),
                  re.compile(r"(?<![a-z0-9])(\d{1,2})x(\d{2,3})(?!\d)", re.I))
_TEXT_CODECS = {"subrip", "srt", "mov_text", "ass", "ssa", "webvtt"}
_SIDECAR_SUFFIXES = {".srt", ".vtt"}
_SUBS_FOLDERS = {"subs", "sub", "subtitles"}
_OUR_PRODUCTS = re.compile(r"(?<![a-z])ru(?![a-z])|translated|whisperx|_edit|_synced|^subs\.")
_STREAM = re.compile(r"^\s*Stream #(\d+:\d+)(?:\[[^\]]*\])?(?:\((\w+)\))?[^:]*:\s*Subtitle:\s*(\w+)(.*)$")
_TITLE = re.compile(r"^\s+title\s*:\s*(.*)$", re.I)
_HEARING = re.compile(r"hearing|\bsdh\b|\bcc\b|closed.?caption|\bhi\b", re.I)
_VTT_SHORT = re.compile(r"(?m)^(\d\d):(\d\d)[.,](\d{3})(\s*-->\s*)(\d\d):(\d\d)[.,](\d{3})")
MAX_TEXT_BYTES = 5_000_000
GESTDOWN_SHOWS = {"tt0239195": "5rxCcEsDlBbYRBuNnzkGXf"}  # IMDb id -> Gestdown (Addic7ed proxy) show id
STREMIO_URL = "https://opensubtitles-v3.strem.io/subtitles/series/{imdb}:{season}:{episode}.json"
GESTDOWN_URL = "https://api.gestdown.info"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
MIN_REQUEST_INTERVAL_S = 1.1
_last_request: dict[str, float] = {}


def episode_code(media: Path | str) -> tuple[int, int] | None:
    """(season, episode) from names like "Show.S51E01.Title.mkv", "S40_E02" or "10x05"; None when there is none."""
    name = Path(media).stem
    for pattern in _EPISODE_CODES:
        found = pattern.search(name)
        if found:
            return int(found.group(1)), int(found.group(2))
    return None


def decode_subtitle(raw: bytes) -> str:
    """Text of a subtitle file in the encodings such files come in."""
    encodings = ["utf-8-sig", "cp1252"]
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def to_srt_text(text: str) -> str:
    """WebVTT with short times (mm:ss.mmm) as SRT-like text that ``parse_cue_text`` reads; SRT stays as it is."""
    if not text.lstrip("﻿ \n").startswith("WEBVTT"):
        return text
    return _VTT_SHORT.sub(lambda m: f"00:{m.group(1)}:{m.group(2)},{m.group(3)}{m.group(4)}00:{m.group(5)}:{m.group(6)},{m.group(7)}", text)


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower()[:20000])
    common = {"the", "you", "and", "to", "i", "a", "is", "it", "that", "of", "we", "in"}
    # 0.2 rejected the real Amazon SDH track of S51E01 (0.18: names and shouts); other languages stay far under 0.1
    return bool(words) and sum(w in common for w in words) / len(words) > 0.1


def mostly_cyrillic(text: str) -> bool:
    letters = [c for c in text[:20000] if c.isalpha()]
    return bool(letters) and sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters) > 0.3


def describe_text(text: str) -> dict[str, Any]:
    """What a subtitle text can tell about who speaks: kind, named lines, turn marks, cues."""
    text = to_srt_text(text)
    cues = parse_cue_text(text)
    pieces = file_pieces(cues)
    return {"kind": subtitle_kind(pieces), "labels": sum(names_person(p.person) for p in pieces),
            "turns": sum(p.opens_turn and not p.person for p in pieces), "cues": len(cues),
            "names": len({p.person for p in pieces if names_person(p.person)}),
            "turn_marks": sum(p.turn_mark for p in pieces)}


def _rank(info: dict[str, Any]) -> tuple[int, int, int]:
    """Better first: the kind, then the number of named lines (labels) or marks (turns), then the cues."""
    count = info["labels"] if info["kind"] == "labels" else info["turns"] if info["kind"] == "turns" else 0
    return _KIND_RANK[info["kind"]], count, info["cues"]


@dataclass
class Candidate:
    origin: str
    where: str
    text: str
    info: dict[str, Any]


def _candidate(origin: str, where: str, text: str) -> Candidate | None:
    """A usable English subtitle text, else None (not subtitles, Russian, other language, ours)."""
    text = to_srt_text(text)
    if len(text) < 200 or "-->" not in text or mostly_cyrillic(text) or not looks_english(text):
        return None
    info = describe_text(text)
    return Candidate(origin, where, text, info) if info["cues"] >= 10 else None


def _best(found: Sequence[Candidate | None]) -> Candidate | None:
    return max((c for c in found if c is not None), key=lambda c: _rank(c.info), default=None)


def file_candidate(origin: str, path: Path) -> Candidate | None:
    """``_candidate`` for a subtitle file on disk; None when it cannot be read or is not usable English subtitles."""
    try:
        return _candidate(origin, str(path), decode_subtitle(path.read_bytes()[:MAX_TEXT_BYTES]))
    except OSError as error:
        log.warning("Cannot read %s: %s", path, error)
        return None


# -- inside the video

def parse_subtitle_streams(report: str) -> list[dict[str, Any]]:
    """Subtitle streams from the text `ffmpeg -i` prints: index ("0:3"), language, codec, title, hearing-impaired flag."""
    streams: list[dict[str, Any]] = []
    for line in report.splitlines():
        stream = _STREAM.match(line)
        if stream:
            index, lang, codec, rest = stream.groups()
            streams.append({"index": index, "lang": (lang or "").lower(), "codec": codec.lower(), "title": "",
                            "hearing": "hearing impaired" in rest.lower()})
            continue
        title = _TITLE.match(line)
        if title and streams and not streams[-1]["title"]:
            streams[-1]["title"] = title.group(1).strip()
        elif line.strip().startswith("Stream #"):  # a stream of another kind ends the metadata of the last one
            streams.append({"index": "", "codec": "", "lang": "", "title": "", "hearing": False})
    return [s for s in streams if s["index"]]


def english_text_streams(streams: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """English streams in a text codec (image subtitles are skipped), hearing-impaired ones first."""
    keep = []
    for s in streams:
        english = s["lang"] in ("eng", "en") or (s["lang"] in ("", "und") and (not s["title"] or "english" in s["title"].lower()))
        if s["codec"] in _TEXT_CODECS and english:
            keep.append({**s, "hearing": s["hearing"] or bool(_HEARING.search(s["title"]))})
    return sorted(keep, key=lambda s: not s["hearing"])


def embedded_candidates(media: Path, max_streams: int = 4, timeout_s: float = 900.0) -> list[Candidate]:
    """English text subtitle streams of the video, extracted in one ffmpeg pass (the file is read once)."""
    from utils.ffmpeg import ffmpeg_exe

    if media.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"):
        return []
    try:
        exe = ffmpeg_exe()
        report = subprocess.run([exe, "-hide_banner", "-i", str(media)], capture_output=True, text=True, errors="replace",
                                timeout=120).stderr
        streams = english_text_streams(parse_subtitle_streams(report))[:max_streams]
        if not streams:
            return []
        with tempfile.TemporaryDirectory(prefix="subs_embedded_") as tmp:
            command = [exe, "-v", "error", "-y", "-i", str(media)]
            for k, s in enumerate(streams):
                command += ["-map", s["index"], "-c:s", "srt", str(Path(tmp) / f"{k}.srt")]
            subprocess.run(command, capture_output=True, timeout=timeout_s)
            found = []
            for k, s in enumerate(streams):
                path = Path(tmp) / f"{k}.srt"
                if path.is_file():
                    found.append(_candidate("embedded", f"{media.name}#{s['index']}", decode_subtitle(path.read_bytes())))
            return [c for c in found if c is not None]
    except (OSError, subprocess.SubprocessError, FileNotFoundError) as error:
        log.warning("Subtitle streams of %s not read: %s", media.name, error)
        return []


# -- next to the video

def is_our_product(name: str) -> bool:
    """Subtitles this project wrote (Russian, translated, edited, synced, ...) are never a source."""
    return bool(_OUR_PRODUCTS.search(name.lower()))


def _same_episode(name: str, media: Path, code: tuple[int, int] | None) -> bool:
    lower, stem = name.lower(), media.stem.lower()
    return lower.startswith(stem) or (code is not None and episode_code(name) == code)


def sidecar_paths(media: Path, limit: int = 12) -> list[Path]:
    """.srt / .vtt files of this episode next to the video and in its subs/ folder. A folder inside subs/ named after the
    episode (scene releases: Subs/<release>/2_English.srt) counts with all its files."""
    code = episode_code(media)
    folder = media.parent
    found: list[Path] = []
    try:
        places: list[tuple[Path, bool]] = [(folder, False)]  # (folder, every file in it counts)
        for sub in sorted(d for d in folder.iterdir() if d.is_dir() and d.name.lower() in _SUBS_FOLDERS):
            places.append((sub, False))
            places += [(d, True) for d in sorted(sub.iterdir()) if d.is_dir() and _same_episode(d.name, media, code)]
        for place, whole in places:
            for path in sorted(place.iterdir()):
                if path.is_file() and path.suffix.lower() in _SIDECAR_SUFFIXES and not is_our_product(path.name) \
                        and (whole or _same_episode(path.name, media, code)):
                    found.append(path)
    except OSError as error:
        log.warning("Cannot list %s: %s", folder, error)
    return found[:limit]


def sidecar_candidates(media: Path) -> list[Candidate]:
    out = []
    for path in sidecar_paths(media):
        try:
            out.append(_candidate("sidecar", str(path), decode_subtitle(path.read_bytes()[:MAX_TEXT_BYTES])))
        except OSError as error:
            log.warning("Cannot read %s: %s", path, error)
    return [c for c in out if c is not None]


# -- the season's own subtitle folder and the subtitle base

def folder_candidates(folder: Path, code: tuple[int, int], limit: int = 12) -> list[Candidate]:
    """.srt / .vtt files of one episode (matched by its SxxEyy code) in the season's own subtitle folder."""
    try:
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _SIDECAR_SUFFIXES
                       and not is_our_product(p.name) and episode_code(p.name) == code)
    except OSError as error:
        log.warning("Cannot list %s: %s", folder, error)
        return []
    return [c for p in paths[:limit] if (c := file_candidate("season folder", p)) is not None]


def show_allows(media: Path, imdb: str | dict[str, str | None] | None, show_imdb: str) -> bool:
    """Whether a file collection of the show ``show_imdb`` may serve this video. The ``imdb`` map decides when it names
    the video's show (another show or null: no); when it does not, the show is unknown and the collection is used."""
    if isinstance(imdb, dict):
        path = path_words(media)
        match = [value for part, value in imdb.items() if path_words(part) in path]
        return not match or match[0] == show_imdb
    return imdb is None or imdb == show_imdb


def base_candidates(base: Path, media: Path, imdb: str | dict[str, str | None] | None, season: int, episode: int) -> list[Candidate]:
    """The base's file for this episode (``<base>/<show>/index.json`` made by ``stt.subtitle_base``): the first show
    folder whose IMDb id suits the video. Its alternates are not used here, they are for a person to look at."""
    key = f"S{season:02d}E{episode:02d}"
    try:
        folders = sorted(d for d in base.iterdir() if (d / BASE_INDEX).is_file())
    except OSError as error:
        log.warning("Subtitle base %s not read: %s", base, error)
        return []
    for folder in folders:
        try:
            index = json.loads((folder / BASE_INDEX).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            log.warning("Subtitle base index %s not read: %s", folder / BASE_INDEX, error)
            continue
        entry = (index.get("episodes") or {}).get(key)
        if entry and show_allows(media, imdb, str(index.get("imdb") or "")):
            found = file_candidate("base", base / str(entry.get("file", "")))
            return [found] if found is not None else []
    return []


# -- local archive

def archive_candidates(archive: Path, season: int, episode: int, limit: int = 4) -> list[Candidate]:
    """Files of a local archive (``manifest.jsonl``: season, episode, path, language, n_label_lines, n_turn_marks, n_cues)."""
    manifest = archive / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("season") == season and row.get("episode") == episode and row.get("language", "en") == "en":
                rows.append(row)
    except OSError as error:
        log.warning("Subtitle archive %s not read: %s", manifest, error)
        return []
    rows.sort(key=lambda r: (r.get("n_label_lines", 0), r.get("n_turn_marks", 0), r.get("n_cues", 0)), reverse=True)
    out = []
    for row in rows[:limit]:
        path = Path(str(row.get("path", "")))
        out.append(file_candidate("archive", path if path.is_absolute() else archive / path))
    return [c for c in out if c is not None]


# -- online: simple sources without a key (the OpenSubtitles Stremio addon, Gestdown); text only, nothing is run

def _fetch(url: str, timeout_s: float = 20.0) -> bytes | None:
    """GET with a timeout and a pause between requests to one host; None on any failure."""
    import requests

    host = urlparse(url).netloc
    wait = _last_request.get(host, 0.0) + MIN_REQUEST_INTERVAL_S - time.time()
    if wait > 0:
        time.sleep(wait)
    try:
        response = requests.get(url, timeout=timeout_s, headers={"User-Agent": USER_AGENT}, stream=True)
        _last_request[host] = time.time()
        if response.status_code != 200:
            log.warning("Subtitles: %s answered %s", host, response.status_code)
            return None
        data = b""
        for chunk in response.iter_content(65536):
            data += chunk
            if len(data) > MAX_TEXT_BYTES:
                return None
        return data
    except requests.RequestException as error:
        _last_request[host] = time.time()
        log.warning("Subtitles: %s not reachable (%s)", host, type(error).__name__)
        return None


def unzip_texts(raw: bytes) -> list[bytes]:
    """.srt / .vtt members of a zip, read in memory (nothing is extracted to disk)."""
    out: list[bytes] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in archive.infolist():
                if not info.is_dir() and info.filename.lower().endswith(tuple(_SIDECAR_SUFFIXES)) and info.file_size <= MAX_TEXT_BYTES:
                    out.append(archive.read(info))
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError):
        pass
    return out


def _texts(raw: bytes) -> list[str]:
    """Subtitle texts in a download: the text itself or the members of a zip; HTML pages are not subtitles."""
    if raw[:2] == b"PK":
        return [decode_subtitle(member) for member in unzip_texts(raw)]
    if raw[:600].lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml")):
        return []
    return [decode_subtitle(raw)]


def online_candidates(imdb: str, season: int, episode: int, budget: int = 8, per_source: int = 3) -> list[Candidate]:
    """Candidates from the Stremio OpenSubtitles addon, then (when it gave no named lines) Gestdown. At most ``budget``
    requests in all, ``per_source`` downloads per source; a failed request only means fewer candidates."""
    found: list[Candidate] = []
    used = 0

    def get(url: str) -> bytes | None:
        nonlocal used
        used += 1
        return _fetch(url)

    def download(origin: str, urls: list[str]) -> None:
        for url in urls:
            if used >= budget:
                return
            raw = get(url)
            found.extend(c for text in (_texts(raw) if raw else []) if (c := _candidate(origin, url, text)) is not None)

    def listing(url: str) -> Any:
        raw = get(url)
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return None

    stremio = listing(STREMIO_URL.format(imdb=imdb, season=season, episode=episode))
    english = [x for x in (stremio or {}).get("subtitles", []) if isinstance(x, dict) and x.get("lang") == "eng" and x.get("url")]
    download("online:stremio", [x["url"] for x in english[:per_source]])
    top = _best(found)
    if (top is not None and top.info["kind"] == "labels") or imdb not in GESTDOWN_SHOWS or used >= budget - 1:
        return found
    gestdown = listing(f"{GESTDOWN_URL}/subtitles/get/{GESTDOWN_SHOWS[imdb]}/{season}/{episode}/English")
    matches = [m for m in (gestdown or {}).get("matchingSubtitles", []) if isinstance(m, dict) and m.get("downloadUri")]
    matches.sort(key=lambda m: (not m.get("hearingImpaired"), -int(m.get("downloadCount") or 0)))
    download("online:gestdown", [GESTDOWN_URL + m["downloadUri"] for m in matches[:per_source]])
    return found


# -- all together

def path_words(media: Path | str) -> str:
    """The video's path in lower case with ``.``, ``_`` and ``-`` as spaces: "The.Amazing.Race" -> "the amazing race"."""
    return re.sub(r"[._\-]+", " ", str(media).lower())


def imdb_of(media: Path, imdb: str | dict[str, str | None] | None) -> str | None:
    """The show's IMDb id: as given, or from ``{"part of the path": id}`` (the first part found; null: no online search)."""
    if not isinstance(imdb, dict):
        return imdb
    path = path_words(media)
    return next((value for part, value in imdb.items() if path_words(part) in path), None)


def find_subtitles(media: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The best existing English subtitles of the episode, written as ``out_dir/subs.source.en.srt``
    (+ ``subs.source.json``: origin, path_or_url, kind, labels, turns, cues); None when there are none.

    Order: the season's own folder (``cfg["folder"]``), inside the video, next to it, the subtitle base (``cfg["base"]``,
    made by ``stt.subtitle_base``), the local archive (``cfg["archive"]``; Survivor US only, see ``show_allows``), online
    (``cfg["online"]``, ``cfg["imdb"]``).
    The best kind wins (labels > turns > plain), then the most named lines / marks; a source that gives named lines
    ends the search. The answer, "nothing found" too, is cached in ``subs.source.json`` for the same search settings.
    """
    media, out_dir = Path(media), Path(out_dir)
    searched = {"online": bool(cfg.get("online", False)), "imdb": imdb_of(media, cfg.get("imdb", "tt0239195")), "archive": str(cfg.get("archive") or "")}
    searched.update({k: str(cfg[k]) for k in ("folder", "base") if cfg.get(k)})  # only when set: older caches stay valid
    cache, text_path = out_dir / SOURCE_JSON, out_dir / SOURCE_SRT
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("searched") == searched and (saved.get("origin") is None or text_path.is_file()):
                return None if saved.get("origin") is None else saved
        except (OSError, json.JSONDecodeError):
            pass
    code = episode_code(media)
    best: Candidate | None = None

    def consider(found: Sequence[Candidate | None]) -> bool:
        nonlocal best
        top = _best([best, *found])
        best = top
        return best is not None and best.info["kind"] == "labels"

    imdb_cfg = cfg.get("imdb", "tt0239195")
    done = bool(code and searched.get("folder") and consider(folder_candidates(Path(searched["folder"]).expanduser(), code)))
    done = done or consider(embedded_candidates(media)) or consider(sidecar_candidates(media))
    if not done and code and searched.get("base"):
        done = consider(base_candidates(Path(searched["base"]).expanduser(), media, imdb_cfg, *code))
    if not done and code and searched["archive"] and show_allows(media, imdb_cfg, ARCHIVE_IMDB):
        done = consider(archive_candidates(Path(searched["archive"]).expanduser(), *code))
    if not done and code and searched["online"] and searched["imdb"]:
        consider(online_candidates(str(searched["imdb"]), *code))
    out_dir.mkdir(parents=True, exist_ok=True)
    if best is None:
        cache.write_text(json.dumps({"origin": None, "path_or_url": None, "kind": "none", "labels": 0, "turns": 0, "cues": 0,
                                     "searched": searched}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log.info("No existing English subtitles found for %s", media.name)
        return None
    text_path.write_text(best.text, encoding="utf-8")
    result = {"origin": best.origin, "path_or_url": best.where, **{k: best.info[k] for k in ("kind", "labels", "turns", "cues")},
              "sha1": hashlib.sha1(best.text.encode("utf-8")).hexdigest()[:10], "searched": searched}
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log.info("Subtitles for %s: %s (%s), %d named lines, %d turn marks", media.name, best.origin, best.info["kind"],
             best.info["labels"], best.info["turns"])
    return result
