"""Blind listening tool: a small local web page to compare and rate audio/video variants without knowing them.

Run ``python -m ui.blind_listen <folder>`` and open http://127.0.0.1:7862. The folder is either one listening
pack or a root folder whose sub-folders are packs (every pack becomes a tab). Supported pack layouts:

1. **multitrack** - video files (``.mkv`` ...) with two or more audio tracks; every file is a group, every audio
   track a variant (a track titled "Оригинал"/"Original"/"Reference" is a reference). Browsers cannot switch
   tracks inside a file, so the tracks are split once, without re-encoding, into ``<pack>/.listen_cache/``.
2. **groups** - sub-folders of audio files; every sub-folder is a group, ``00_*`` files are references (not rated),
   the rest (``A.wav``, ``B.wav`` ...) are the blind variants.
3. **clips** - audio files right in the pack folder, rated one by one on a fixed ``1..5`` scale with defect tags.
4. **manifest** - a ``blind_pack.json`` file (schema below, written by :func:`write_pack_manifest`).

Neutral captions come from ``README.txt`` / ``list.txt`` lines that start with a group or file name.

**Blind isolation.** Nothing here reads ``key.json``: discovery, the pack/ratings endpoints and the media endpoint
only see media files, captions and the manifest's whitelisted fields. The key is read by :mod:`ui.blind_decode`,
which the server imports only inside the explicit ``POST /api/decode`` request (it needs ``"confirm": true``).

``blind_pack.json`` (schema ``blind_pack/1``)::

    {
      "schema": "blind_pack/1",
      "title": "ESpeech steps", "description": "what to listen for",
      "mode": "comparison",            # or "single": every variant rated on its own fixed scale
      "scale": {"min": 1, "max": null, # null = number of variants in the group (comparison default)
                "labels": {"1": "хуже всех"}},
      "tags": ["Обрыв", "Шум/хвост"],   # optional quick defect tags (single mode default: DEFAULT_TAGS)
      "key_file": "key.json",          # where the decoding key lives (read only on decode)
      "groups": [
        {"id": "g01", "title": "Devin, short line", "text": "«Нельзя.»",
         "sync": null,                 # true = variants are time-aligned (instant switching at the same second)
         "video": "g01/scene.mp4",     # optional muted video played in sync with the tracks
         "references": [{"id": "ref", "label": "Оригинал", "file": "g01/00_ref.wav"}],
         "variants": [{"id": "A", "file": "g01/A.wav"}, {"id": "B", "file": "g01/B.wav"}]}
      ]
    }

The matching key (schema ``blind_key/1``) is ``{"schema": "blind_key/1", "legend": {name: text},
"groups": {group_id: {"facets": {...}, "variants": {blind_id: {"name": ..., "description": ..., "facets": {...}}}}}}``.

Ratings are saved after every action to ``<state>/ratings.json`` (``<state>`` is the pack folder unless
``--state-dir`` is given), the decoded results to ``results_decoded.json`` / ``results_decoded.md`` next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from utils.atomic_io import write_text_atomic
from utils.ffmpeg import ffmpeg_exe

log = logging.getLogger("blind_listen")

AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"})
VIDEO_EXTS = frozenset({".mkv", ".mp4", ".webm", ".mov", ".m4v"})
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS
MANIFEST_NAME = "blind_pack.json"
KEY_NAME = "key.json"
RATINGS_NAME = "ratings.json"
RESULTS_JSON = "results_decoded.json"
RESULTS_MD = "results_decoded.md"
CACHE_DIR_NAME = ".listen_cache"
CAPTION_FILES = ("README.txt", "readme.txt", "README.md", "list.txt")
DEFAULT_TAGS = ("Обрыв", "Шум/хвост", "Щелчок", "Искажение", "Чисто")
SINGLE_SCALE_LABELS = {"1": "явный брак", "5": "чисто"}
REFERENCE_WORDS = ("оригинал", "original", "ориентир", "reference", "эталон")
HTML_TEMPLATE = Path(__file__).with_name("blind_listen.html")
DEFAULT_PORT = 7862
SYNC_TOLERANCE_S = 0.08

PackKind = Literal["multitrack", "groups", "clips", "manifest"]
PackMode = Literal["comparison", "single"]

KIND_TITLES_RU: dict[str, str] = {
    "multitrack": "видео с несколькими дорожками",
    "groups": "группы аудио",
    "clips": "отдельные клипы",
    "manifest": "набор по манифесту",
}

CONTENT_TYPES: dict[str, str] = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".aac": "audio/aac", ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".mov": "video/quicktime",
}


# --------------------------------------------------------------------------------------------- data model


@dataclass
class MediaItem:
    """One playable thing: a blind variant, a reference or a muted scene video."""

    id: str
    label: str
    kind: Literal["audio", "video"]
    path: Path | None = None  # None until a multitrack cache is built
    stream: int | None = None  # audio stream number inside the container (multitrack only)
    duration: float | None = None
    sync: bool = False  # time-aligned with the other synced items of its group


@dataclass
class Group:
    """A set of variants rated against each other (or, in single mode, one by one)."""

    id: str
    title: str
    caption: str = ""
    variants: list[MediaItem] = field(default_factory=list)
    references: list[MediaItem] = field(default_factory=list)
    video: MediaItem | None = None
    source: Path | None = None  # the multitrack container
    sync: bool = False
    scale_min: int = 1
    scale_max: int = 5


@dataclass
class Pack:
    """One listening pack: a folder with groups, its own ratings and its own key."""

    rel_id: str  # "" for a single pack, the sub-folder name inside a collection
    dir: Path
    title: str
    kind: PackKind
    mode: PackMode
    groups: list[Group]
    description: str = ""
    readme: str = ""
    tags: list[str] = field(default_factory=list)
    scale_labels: dict[str, str] = field(default_factory=dict)
    key_file: str = KEY_NAME

    def group(self, group_id: str) -> Group | None:
        """The group with this id, if any."""
        return next((group for group in self.groups if group.id == group_id), None)


@dataclass
class Collection:
    """What the server shows: one pack, or several packs found in the sub-folders of a root folder."""

    root: Path
    packs: list[Pack]
    is_collection: bool
    title: str
    readme: str = ""


@dataclass
class StreamInfo:
    """One stream of a media container as reported by ffprobe/ffmpeg."""

    index: int
    codec_type: str
    codec_name: str
    title: str = ""
    language: str = ""


# --------------------------------------------------------------------------------------------- helpers


def natural_key(text: str) -> list[Any]:
    """Sort key that orders "clip_2" before "clip_10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _visible(path: Path) -> bool:
    return not path.name.startswith(".")


def _media_files(folder: Path) -> list[Path]:
    """Media files directly inside ``folder`` (hidden files skipped), in natural order."""
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    files = [p for p in entries if p.is_file() and _visible(p) and p.suffix.lower() in MEDIA_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def _sub_dirs(folder: Path) -> list[Path]:
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    return sorted((p for p in entries if p.is_dir() and _visible(p)), key=lambda p: natural_key(p.name))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _readme(folder: Path) -> str:
    """The pack's own description (README.txt or README.md), never the key."""
    for name in ("README.txt", "readme.txt", "README.md"):
        text = _read_text(folder / name)
        if text.strip():
            return text.strip()
    return ""


def _caption_lines(folder: Path) -> list[str]:
    lines: list[str] = []
    for name in CAPTION_FILES:
        lines.extend(_read_text(folder / name).splitlines())
    return lines


def find_caption(lines: list[str], *names: str) -> str:
    """Neutral caption for a group/file: the rest of the first line that starts with one of ``names``.

    Understands ``name — text``, ``name/ (7 вар.) — text`` and ``name | Спикер: ... | Текст: ...`` lines.
    """
    for name in names:
        if not name:
            continue
        for raw in lines:
            line = raw.strip()
            if not line.startswith(name):
                continue
            rest = line[len(name):]
            if rest and rest[0] not in "/ \t|—–-:(":
                continue  # "01_a" must not match "01_ab"
            rest = rest.lstrip("/").strip()
            for separator in (" — ", " – ", " - "):
                if separator in rest:
                    return rest.split(separator, 1)[1].strip()
            if rest.startswith("|"):
                return rest[1:].strip()
            if "|" in rest:
                return rest.split("|", 1)[1].strip()
            return rest.strip(" :—–-")
    return ""


def first_line(text: str) -> str:
    """First non-empty line of a text (the pack's headline)."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def wav_duration(path: Path) -> float | None:
    """Duration of a WAV file from its header (None for other formats or broken files)."""
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (wave.Error, OSError, EOFError):
        return None


def _durations_match(items: list[MediaItem]) -> bool:
    durations = [item.duration for item in items]
    if len(durations) < 2 or any(d is None or d <= 0 for d in durations):
        return False
    known = [d for d in durations if d is not None]
    return max(known) - min(known) <= SYNC_TOLERANCE_S


def _mark_sync(group: Group) -> None:
    """Flag the variants (and same-length references) that can be switched at the same second."""
    if not group.sync:
        return
    lengths = [v.duration for v in group.variants if v.duration]
    master = lengths[0] if lengths else None
    for item in group.variants:
        item.sync = True
    for item in group.references:
        item.sync = master is None or (item.duration is not None and abs(item.duration - master) <= SYNC_TOLERANCE_S)


def _is_reference_stem(stem: str) -> bool:
    low = stem.lower()
    return low.startswith("00_") or low in {"ref", "reference", "original", "orig"}


def _reference_label(stem: str) -> str:
    name = stem.removeprefix("00_")
    return "Ориентир: " + name.replace("_", " ").strip()


def _media_kind(path: Path) -> Literal["audio", "video"]:
    return "video" if path.suffix.lower() in VIDEO_EXTS else "audio"


def _blind_letter(index: int) -> str:
    """A, B, ..., Z, AA, AB, ..."""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


# --------------------------------------------------------------------------------------------- probing


_PROBE_CACHE: dict[tuple[str, float, int], list[StreamInfo]] = {}
_PROBE_LOCK = threading.Lock()
_STREAM_RE = re.compile(r"^\s*Stream #\d+:(\d+)(?:\[[^\]]*\])?(?:\(([^)]*)\))?: (\w+): ([\w-]+)")
_TITLE_RE = re.compile(r"^\s+title\s*:\s*(.*)$")


def probe_streams(path: Path) -> list[StreamInfo]:
    """Streams of a media file: ffprobe when installed, otherwise parsed from ``ffmpeg -i`` (bundled ffmpeg)."""
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    with _PROBE_LOCK:
        if cache_key in _PROBE_CACHE:
            return _PROBE_CACHE[cache_key]
    streams = _probe_ffprobe(path) if shutil.which("ffprobe") else None
    if streams is None:
        streams = _probe_ffmpeg(path)
    with _PROBE_LOCK:
        _PROBE_CACHE[cache_key] = streams
    return streams


def _probe_ffprobe(path: Path) -> list[StreamInfo] | None:
    command = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(result.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams: list[StreamInfo] = []
    for raw in data.get("streams", []):
        tags = {str(k).lower(): str(v) for k, v in (raw.get("tags") or {}).items()}
        if (raw.get("disposition") or {}).get("attached_pic"):
            continue
        streams.append(StreamInfo(int(raw.get("index", len(streams))), str(raw.get("codec_type", "")),
                                  str(raw.get("codec_name", "")), tags.get("title", ""), tags.get("language", "")))
    return streams


def _probe_ffmpeg(path: Path) -> list[StreamInfo]:
    try:
        result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, check=False,
                                text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError, FileNotFoundError) as error:
        log.warning("cannot probe %s: %s", path, error)
        return []
    streams: list[StreamInfo] = []
    current: StreamInfo | None = None
    for line in result.stderr.splitlines():
        match = _STREAM_RE.match(line)
        if match:
            index, language, codec_type, codec_name = match.groups()
            current = None
            if "(attached pic)" in line:
                continue
            current = StreamInfo(int(index), codec_type.lower(), codec_name.lower(), "", language or "")
            streams.append(current)
            continue
        title = _TITLE_RE.match(line)
        if title and current is not None and not current.title:
            current.title = title.group(1).strip()
    return streams


def audio_streams(path: Path) -> list[StreamInfo]:
    """Audio streams of a container, in order."""
    return [s for s in probe_streams(path) if s.codec_type == "audio"]


# --------------------------------------------------------------------------------------------- discovery


def classify_dir(folder: Path) -> PackKind | None:
    """Layout of a pack folder, or None when it holds nothing to listen to."""
    if (folder / MANIFEST_NAME).is_file():
        return "manifest"
    files = _media_files(folder)
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
    if any(len(audio_streams(v)) >= 2 for v in videos):
        return "multitrack"
    if not files and any(_media_files(sub) for sub in _sub_dirs(folder)):
        return "groups"
    if files:
        return "clips"
    return None


def _is_standalone_pack(folder: Path) -> bool:
    """A sub-folder that is a pack of its own (not merely one group of its parent)."""
    if (folder / MANIFEST_NAME).is_file() or (folder / KEY_NAME).is_file():
        return True
    kind = classify_dir(folder)
    return kind in ("multitrack", "groups")


def discover(root: Path) -> Collection:
    """Find the packs under ``root``: the folder itself or, for a root folder, its pack sub-folders."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {root}")
    own_kind = classify_dir(root)
    subs = [sub for sub in _sub_dirs(root) if _is_standalone_pack(sub)]
    if own_kind in (None, "groups") and subs and not (root / MANIFEST_NAME).is_file():
        packs = [pack for sub in subs if (pack := load_pack(sub, sub.name)) is not None]
        if packs:
            return Collection(root, packs, True, root.name, _readme(root))
    pack = load_pack(root, "")
    if pack is None:
        raise ValueError(f"В папке {root} не нашлось ни аудио, ни видео для прослушивания")
    return Collection(root, [pack], False, pack.title, pack.readme)


def load_pack(folder: Path, rel_id: str) -> Pack | None:
    """Read one pack folder (never its key)."""
    kind = classify_dir(folder)
    if kind is None:
        return None
    if kind == "manifest":
        return _load_manifest_pack(folder, rel_id)
    readme = _readme(folder)
    lines = _caption_lines(folder)
    if kind == "multitrack":
        groups = _multitrack_groups(folder, lines)
        mode: PackMode = "comparison"
    elif kind == "groups":
        groups = _folder_groups(folder, lines)
        mode = "comparison"
    else:
        groups = _clip_groups(folder, lines)
        mode = "single"
    groups = [g for g in groups if g.variants]
    if not groups:
        return None
    pack = Pack(rel_id, folder, folder.name, kind, mode, groups, description=first_line(readme), readme=readme)
    if mode == "single":
        pack.tags = list(DEFAULT_TAGS)
        pack.scale_labels = dict(SINGLE_SCALE_LABELS)
    return pack


def _multitrack_groups(folder: Path, lines: list[str]) -> list[Group]:
    groups: list[Group] = []
    for video in (f for f in _media_files(folder) if f.suffix.lower() in VIDEO_EXTS):
        tracks = audio_streams(video)
        if len(tracks) < 2:
            continue
        group = Group(video.name, video.stem, find_caption(lines, video.name, video.stem), source=video, sync=True)
        if any(s.codec_type == "video" for s in probe_streams(video)):
            group.video = MediaItem("video", "Видео", "video", sync=True)
        used: set[str] = set()
        for number, track in enumerate(tracks):
            title = track.title.strip()
            if any(word in title.lower() for word in REFERENCE_WORDS):
                ref_id = f"ref{len(group.references) + 1}"
                group.references.append(MediaItem(ref_id, title or "Оригинал", "audio", stream=number, sync=True))
                continue
            label = re.sub(r"^(вариант|variant|track|дорожка)\s+", "", title, flags=re.IGNORECASE).strip()
            label = label or _blind_letter(len(group.variants))
            blind_id = label if label not in used else f"{label}{number}"
            used.add(blind_id)
            group.variants.append(MediaItem(blind_id, label, "audio", stream=number, sync=True))
        group.scale_max = max(len(group.variants), 2)
        groups.append(group)
    return groups


def _items_from_files(files: list[Path]) -> tuple[list[MediaItem], list[MediaItem]]:
    variants: list[MediaItem] = []
    references: list[MediaItem] = []
    for path in files:
        item = MediaItem(path.stem, path.stem, _media_kind(path), path=path, duration=wav_duration(path))
        if _is_reference_stem(path.stem):
            item.id = f"ref{len(references) + 1}"
            item.label = _reference_label(path.stem)
            references.append(item)
        else:
            variants.append(item)
    return variants, references


def _folder_groups(folder: Path, lines: list[str]) -> list[Group]:
    groups: list[Group] = []
    for sub in _sub_dirs(folder):
        files = _media_files(sub)
        if not files:
            continue
        variants, references = _items_from_files(files)
        group = Group(sub.name, sub.name, find_caption(lines + _caption_lines(sub), sub.name),
                      variants=variants, references=references)
        group.scale_max = max(len(variants), 2)
        group.sync = _durations_match(variants)
        _mark_sync(group)
        groups.append(group)
    return groups


def _clip_groups(folder: Path, lines: list[str]) -> list[Group]:
    variants: list[MediaItem] = []
    for path in _media_files(folder):
        caption = find_caption(lines, path.name, path.stem)
        label = path.stem + (f" — {caption}" if caption else "")
        variants.append(MediaItem(path.stem, label, _media_kind(path), path=path, duration=wav_duration(path)))
    return [Group("all", "Все клипы", "", variants=variants, scale_min=1, scale_max=5)]


def _safe_member(folder: Path, relative: str) -> Path | None:
    """``folder/relative`` when it exists and stays inside ``folder`` (no ``..`` escapes)."""
    if not relative:
        return None
    path = (folder / relative).resolve()
    try:
        path.relative_to(folder.resolve())
    except ValueError:
        log.warning("manifest file outside the pack ignored: %s", relative)
        return None
    if not path.is_file():
        log.warning("manifest file missing: %s", path)
        return None
    return path


def _load_manifest_pack(folder: Path, rel_id: str) -> Pack | None:
    try:
        data = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        log.warning("bad %s in %s: %s", MANIFEST_NAME, folder, error)
        return None
    mode: PackMode = "single" if data.get("mode") == "single" else "comparison"
    scale = data.get("scale") or {}
    scale_min = int(scale.get("min") or 1)
    fixed_max = scale.get("max")
    groups: list[Group] = []
    for number, raw in enumerate(data.get("groups") or []):
        group_id = str(raw.get("id") or f"g{number + 1:02d}")
        group = Group(group_id, str(raw.get("title") or group_id), str(raw.get("text") or raw.get("caption") or ""))
        for kind_name, target in (("variants", group.variants), ("references", group.references)):
            for index, entry in enumerate(raw.get(kind_name) or []):
                path = _safe_member(folder, str(entry.get("file") or ""))
                if path is None:
                    continue
                default_id = _blind_letter(index) if kind_name == "variants" else f"ref{index + 1}"
                item_id = str(entry.get("id") or default_id)
                label = str(entry.get("label") or (item_id if kind_name == "variants" else "Ориентир"))
                target.append(MediaItem(item_id, label, _media_kind(path), path=path, duration=wav_duration(path)))
        video = _safe_member(folder, str(raw.get("video") or ""))
        if video is not None:
            group.video = MediaItem("video", "Видео", "video", path=video, sync=True)
        sync_flag = raw.get("sync")
        group.sync = bool(sync_flag) if sync_flag is not None else (group.video is not None
                                                                    or _durations_match(group.variants))
        _mark_sync(group)
        if group.sync and sync_flag:
            for item in group.variants + group.references:
                item.sync = True
        group.scale_min = scale_min
        if fixed_max:
            group.scale_max = int(fixed_max)
        else:
            group.scale_max = 5 if mode == "single" else max(len(group.variants), 2)
        if group.variants:
            groups.append(group)
    if not groups:
        return None
    labels = {str(k): str(v) for k, v in (scale.get("labels") or {}).items()}
    tags = [str(t) for t in data.get("tags") or []]
    if mode == "single":
        labels = labels or dict(SINGLE_SCALE_LABELS)
        tags = tags or list(DEFAULT_TAGS)
    readme = str(data.get("description") or "") or _readme(folder)
    return Pack(rel_id, folder, str(data.get("title") or folder.name), "manifest", mode, groups,
                description=first_line(readme), readme=readme, tags=tags, scale_labels=labels,
                key_file=str(data.get("key_file") or KEY_NAME))


# --------------------------------------------------------------------------------------------- manifest writer


@dataclass
class ManifestVariant:
    """A variant for :func:`write_pack_manifest`: its real name goes only to the key."""

    name: str
    file: str | Path
    description: str = ""
    facets: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestGroup:
    """A group for :func:`write_pack_manifest`."""

    id: str
    variants: list[ManifestVariant]
    title: str = ""
    text: str = ""
    references: list[tuple[str, str | Path]] = field(default_factory=list)  # (public label, file)
    video: str | Path | None = None
    facets: dict[str, str] = field(default_factory=dict)
    sync: bool | None = None


def audio_fingerprint(path: Path) -> str:
    """Content digest of a media file: PCM frames + format for WAV (ignores header metadata), SHA-256 bytes otherwise."""
    digest = hashlib.sha256()
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                params = f"{handle.getnchannels()}|{handle.getsampwidth()}|{handle.getframerate()}|".encode("ascii")
                digest.update(params)
                while True:
                    chunk = handle.readframes(65536)
                    if not chunk:
                        break
                    digest.update(chunk)
            return "wav:" + digest.hexdigest()
        except (OSError, wave.Error):
            digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(262144)
            if not chunk:
                break
            digest.update(chunk)
    return "raw:" + digest.hexdigest()


def find_duplicate_variants(items: list[MediaItem]) -> list[list[str]]:
    """Groups of variant IDs whose audio files have identical content (empty when all variants are distinct)."""
    by_hash: dict[str, list[str]] = {}
    for item in items:
        if item.path is None or not item.path.is_file():
            continue
        by_hash.setdefault(audio_fingerprint(item.path), []).append(item.id)
    return [ids for ids in by_hash.values() if len(ids) > 1]


def _decode_pcm_f32(path: Path, sample_rate: int = 16000) -> bytes:
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True, check=True,
    )
    return result.stdout


def check_duplicates(root: Path, cache_root: Path | None = None, *, decode_audio: bool = True,
                     near_db: float = -60.0) -> dict[str, Any]:
    """Verify every comparison group in ``root`` has distinct variants (exact digest + optional decoded PCM diff)."""
    import numpy as np

    app = BlindListenApp(root, cache_root=cache_root)
    report_packs: list[dict[str, Any]] = []
    any_issues = False
    for p_index, pack in enumerate(app.collection.packs):
        if pack.mode != "comparison":
            continue
        pack_issues: list[dict[str, Any]] = []
        for g_index, group in enumerate(pack.groups):
            if group.source is not None:
                app.prepare_group(p_index, g_index)
            exact = find_duplicate_variants(group.variants)
            near: list[dict[str, Any]] = []
            if decode_audio and len(group.variants) > 1:
                waves: list[tuple[str, Any]] = []
                for item in group.variants:
                    if item.path is not None and item.path.is_file():
                        arr = np.frombuffer(_decode_pcm_f32(item.path), dtype=np.float32)
                        waves.append((item.id, arr))
                for i in range(len(waves)):
                    id_a, a = waves[i]
                    for j in range(i + 1, len(waves)):
                        id_b, b = waves[j]
                        n = min(len(a), len(b))
                        if n == 0 or abs(len(a) - len(b)) > 320:
                            continue
                        rms_ref = float(np.sqrt(np.mean(a[:n] ** 2) + 1e-12))
                        rms_diff = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2) + 1e-12))
                        rel_db = float(20.0 * np.log10(rms_diff / max(rms_ref, 1e-12)))
                        if rel_db <= near_db:
                            near.append({"pair": [id_a, id_b], "diff_rel_db": round(rel_db, 1)})
            if exact or near:
                any_issues = True
                pack_issues.append({"group": group.id, "exact": exact, "near": near})
        report_packs.append({"pack": pack.rel_id or pack.title, "groups_checked": len(pack.groups),
                             "issues": pack_issues})
    return {"ok": not any_issues, "packs": report_packs}


def write_pack_manifest(pack_dir: str | Path, title: str, groups: list[ManifestGroup], *, description: str = "",
                        mode: PackMode = "comparison", scale_max: int | None = None,
                        scale_labels: dict[str, str] | None = None, tags: list[str] | None = None,
                        seed: int | None = None, copy_files: bool = True,
                        allow_duplicates: bool = False) -> Path:
    """Build a blind pack for a future experiment: shuffle the variants, write ``blind_pack.json`` and ``key.json``.

    Every group's variants are shuffled (``seed`` makes it repeatable) and named A, B, C ...; with ``copy_files``
    the media are copied to ``<pack_dir>/<group id>/<letter><ext>`` so the file names reveal nothing. The real
    names, descriptions, facets and extra ``info`` go to ``key.json`` only. Unless ``allow_duplicates`` is True,
    identical variant files within a comparison group raise ``ValueError``. Returns the manifest path.
    """
    pack_dir = Path(pack_dir)
    if not allow_duplicates and mode == "comparison":
        for group in groups:
            by_hash: dict[str, list[str]] = {}
            for variant in group.variants:
                src = Path(variant.file)
                if not src.is_absolute():
                    src = pack_dir / src
                if src.is_file():
                    by_hash.setdefault(audio_fingerprint(src), []).append(variant.name)
            dups = [names for names in by_hash.values() if len(names) > 1]
            if dups:
                detail = "; ".join(", ".join(names) for names in dups)
                raise ValueError(f"Duplicate audio variants in group {group.id!r}: {detail}")
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest_groups: list[dict[str, Any]] = []
    key_groups: dict[str, Any] = {}

    def place(source: str | Path, group_id: str, stem: str) -> str:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = pack_dir / source_path
        if not copy_files:
            return os.path.relpath(source_path.resolve(), pack_dir.resolve())
        target = pack_dir / group_id / f"{stem}{source_path.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target.relative_to(pack_dir).as_posix()

    for group in groups:
        variants = list(group.variants)
        rng.shuffle(variants)
        entry: dict[str, Any] = {"id": group.id, "title": group.title or group.id, "text": group.text,
                                 "references": [], "variants": []}
        if group.sync is not None:
            entry["sync"] = group.sync
        if group.video is not None:
            entry["video"] = place(group.video, group.id, "00_video")
        for number, (label, file) in enumerate(group.references):
            entry["references"].append({"id": f"ref{number + 1}", "label": label,
                                        "file": place(file, group.id, f"00_ref{number + 1}")})
        key_variants: dict[str, Any] = {}
        for number, variant in enumerate(variants):
            letter = _blind_letter(number)
            entry["variants"].append({"id": letter, "file": place(variant.file, group.id, letter)})
            key_variants[letter] = {"name": variant.name, "description": variant.description,
                                    "facets": dict(variant.facets), "info": dict(variant.info),
                                    "source": str(variant.file)}
        manifest_groups.append(entry)
        key_groups[group.id] = {"facets": dict(group.facets), "variants": key_variants}

    scale: dict[str, Any] = {"min": 1, "max": scale_max}
    if scale_labels:
        scale["labels"] = scale_labels
    manifest = {"schema": "blind_pack/1", "title": title, "description": description, "mode": mode,
                "scale": scale, "tags": list(tags or []), "key_file": KEY_NAME, "groups": manifest_groups}
    legend = {v.name: v.description for g in groups for v in g.variants if v.description}
    key = {"schema": "blind_key/1", "seed": seed, "legend": legend, "groups": key_groups}
    manifest_path = pack_dir / MANIFEST_NAME
    write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(pack_dir / KEY_NAME, json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


# --------------------------------------------------------------------------------------------- ratings


def state_dir_for(pack: Pack, state_root: Path | None) -> Path:
    """Where the pack's ratings and decoded results live."""
    if state_root is None:
        return pack.dir
    return state_root / pack.rel_id if pack.rel_id else state_root


def empty_ratings(pack: Pack) -> dict[str, Any]:
    """A fresh ratings document."""
    return {"schema": "blind_listen.ratings/1", "pack": pack.rel_id or pack.title, "unique_ranks": False,
            "decoded_at": None, "updated_at": None, "groups": {}}


def load_ratings(path: Path, pack: Pack) -> dict[str, Any]:
    """Saved ratings (an empty document when there are none yet or the file is broken)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_ratings(pack)
    if not isinstance(data, dict):
        return empty_ratings(pack)
    base = empty_ratings(pack)
    base.update(data)
    if not isinstance(base.get("groups"), dict):
        base["groups"] = {}
    return base


def _clean_text(value: Any, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, str) else ""


def clean_group_state(group: Group, state: Any) -> dict[str, Any]:
    """Keep only valid fields of a group state sent by the page (scores inside the scale, known ids)."""
    state = state if isinstance(state, dict) else {}
    variant_ids = {item.id for item in group.variants}
    all_ids = variant_ids | {item.id for item in group.references}
    scores: dict[str, int] = {}
    for item_id, value in (state.get("scores") or {}).items():
        valid = isinstance(value, int) and not isinstance(value, bool) and group.scale_min <= value <= group.scale_max
        if item_id in variant_ids and valid:
            scores[item_id] = value
    notes = {k: _clean_text(v, 4000) for k, v in (state.get("notes") or {}).items()
             if k in all_ids and _clean_text(v, 4000).strip()}
    tags: dict[str, list[str]] = {}
    for item_id, values in (state.get("tags") or {}).items():
        if item_id in variant_ids and isinstance(values, list):
            kept = [str(v)[:40] for v in values if isinstance(v, str) and v.strip()][:20]
            if kept:
                tags[item_id] = kept
    listened = sorted({str(v) for v in state.get("listened") or [] if str(v) in all_ids}, key=natural_key)
    return {"scores": scores, "notes": notes, "tags": tags, "comment": _clean_text(state.get("comment"), 20000),
            "listened": listened, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def pack_progress(pack: Pack, ratings: dict[str, Any]) -> dict[str, int]:
    """How much is rated: complete groups (comparison) or rated clips (single)."""
    saved = ratings.get("groups") or {}
    if pack.mode == "single":
        total = sum(len(g.variants) for g in pack.groups)
        done = sum(1 for g in pack.groups for v in g.variants if v.id in (saved.get(g.id) or {}).get("scores", {}))
        return {"done": done, "total": total}
    done = 0
    for group in pack.groups:
        scores = (saved.get(group.id) or {}).get("scores", {})
        done += all(v.id in scores for v in group.variants)
    return {"done": done, "total": len(pack.groups)}


class RatingsStore:
    """Thread-safe ratings file of one pack, saved atomically after every change."""

    def __init__(self, pack: Pack, state_dir: Path) -> None:
        self.pack = pack
        self.path = state_dir / RATINGS_NAME
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        """Current ratings."""
        with self._lock:
            return load_ratings(self.path, self.pack)

    def _write(self, data: dict[str, Any]) -> None:
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_text_atomic(self.path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def save_group(self, group: Group, state: Any) -> dict[str, Any]:
        """Replace one group's state; returns the whole document."""
        with self._lock:
            data = load_ratings(self.path, self.pack)
            data["groups"][group.id] = clean_group_state(group, state)
            self._write(data)
            return data

    def update_settings(self, unique_ranks: bool | None = None, decoded: bool = False) -> dict[str, Any]:
        """Change pack-wide settings (strict unique ranks, decode time)."""
        with self._lock:
            data = load_ratings(self.path, self.pack)
            if unique_ranks is not None:
                data["unique_ranks"] = bool(unique_ranks)
            if decoded:
                data["decoded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._write(data)
            return data


# --------------------------------------------------------------------------------------------- multitrack cache


_BROWSER_VIDEO = frozenset({"h264", "av1", "vp9"})
_BROWSER_AUDIO = frozenset({"aac", "mp3", "alac"})


def _default_cache_root(pack: Pack) -> Path:
    candidate = pack.dir / CACHE_DIR_NAME
    if os.access(pack.dir, os.W_OK):
        return candidate
    digest = hashlib.sha1(str(pack.dir).encode("utf-8")).hexdigest()[:12]
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "blind_listen" / digest


def _group_cache_dir(cache_root: Path, group: Group) -> Path:
    assert group.source is not None
    stat = group.source.stat()
    digest = hashlib.sha1(f"{group.source.name}|{stat.st_size}|{stat.st_mtime}".encode()).hexdigest()[:10]
    return cache_root / f"{group.source.stem}-{digest}"


def build_multitrack_cache(group: Group, cache_root: Path) -> float:
    """Split a multitrack container into a muted MP4 video and one M4A per audio track (stream copy when the
    codecs play in browsers, re-encoding otherwise). Fills the items' paths; returns the seconds spent."""
    assert group.source is not None
    started = time.monotonic()
    folder = _group_cache_dir(cache_root, group)
    items = group.variants + group.references
    targets = {item.id: folder / f"a{item.stream}.m4a" for item in items}
    video_target = folder / "video.mp4"
    has_video = any(s.codec_type == "video" for s in probe_streams(group.source))
    if all(p.is_file() for p in targets.values()) and (video_target.is_file() or not has_video):
        _assign_cache_paths(group, targets, video_target if has_video else None)
        return 0.0
    folder.mkdir(parents=True, exist_ok=True)
    streams = probe_streams(group.source)
    video_codec = next((s.codec_name for s in streams if s.codec_type == "video"), "")
    audio_codecs = [s.codec_name for s in streams if s.codec_type == "audio"]
    for force_encode in (False, True):
        command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(group.source)]
        parts: list[tuple[Path, Path]] = []
        if has_video:
            part = video_target.with_name(video_target.name + ".part")
            copy = video_codec in _BROWSER_VIDEO and not force_encode
            codec = ["-c:v", "copy"] if copy else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                                     "-pix_fmt", "yuv420p"]
            command += ["-map", "0:v:0", *codec, "-an", "-sn", "-movflags", "+faststart", "-f", "mp4", str(part)]
            parts.append((part, video_target))
        for item in items:
            assert item.stream is not None
            target = targets[item.id]
            part = target.with_name(target.name + ".part")
            codec_name = audio_codecs[item.stream] if item.stream < len(audio_codecs) else ""
            copy = codec_name in _BROWSER_AUDIO and not force_encode
            codec = ["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "192k"]
            command += ["-map", f"0:a:{item.stream}", *codec, "-vn", "-sn", "-movflags", "+faststart",
                        "-f", "mp4", str(part)]
            parts.append((part, target))
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                check=False)
        if result.returncode == 0 and all(p.is_file() for p, _ in parts):
            for part, target in parts:
                os.replace(part, target)
            _assign_cache_paths(group, targets, video_target if has_video else None)
            return time.monotonic() - started
        for part, _ in parts:
            part.unlink(missing_ok=True)
        log.warning("ffmpeg split of %s failed (encode=%s): %s", group.source, force_encode, result.stderr[-500:])
    raise RuntimeError(f"Не удалось разделить дорожки файла {group.source.name}")


def _assign_cache_paths(group: Group, targets: dict[str, Path], video: Path | None) -> None:
    for item in group.variants + group.references:
        item.path = targets[item.id]
    if group.video is not None:
        group.video.path = video


# --------------------------------------------------------------------------------------------- web app


class BlindListenApp:
    """Everything the HTTP handler needs: the packs, where ratings and caches live, per-pack locks."""

    def __init__(self, root: Path, state_root: Path | None = None, cache_root: Path | None = None) -> None:
        self.collection = discover(root)
        self.state_root = state_root.resolve() if state_root else None
        self.cache_root = cache_root.resolve() if cache_root else None
        self.stores = [RatingsStore(pack, state_dir_for(pack, self.state_root)) for pack in self.collection.packs]
        self._cache_locks: dict[tuple[int, int], threading.Lock] = {}
        self._cache_guard = threading.Lock()
        self._decode_lock = threading.Lock()

    # -- lookups

    def pack(self, index: Any) -> Pack:
        """Pack by its index (raises KeyError)."""
        try:
            number = int(index)
        except (TypeError, ValueError) as error:
            raise KeyError("pack") from error
        if not 0 <= number < len(self.collection.packs):
            raise KeyError("pack")
        return self.collection.packs[number]

    def cache_root_for(self, pack: Pack) -> Path:
        """Folder for the split tracks of a multitrack pack."""
        if self.cache_root is not None:
            return self.cache_root / (pack.rel_id or pack.dir.name)
        return _default_cache_root(pack)

    def prepare_group(self, pack_index: int, group_index: int) -> float:
        """Make the group playable (split multitrack containers once)."""
        pack = self.pack(pack_index)
        group = pack.groups[group_index]
        if group.source is None:
            return 0.0
        with self._cache_guard:
            lock = self._cache_locks.setdefault((pack_index, group_index), threading.Lock())
        with lock:
            return build_multitrack_cache(group, self.cache_root_for(pack))

    # -- public (blind) views

    def packs_summary(self) -> dict[str, Any]:
        """Tabs of the page: every pack with its progress."""
        packs = []
        for index, (pack, store) in enumerate(zip(self.collection.packs, self.stores)):
            ratings = store.load()
            packs.append({"index": index, "id": pack.rel_id or pack.dir.name, "title": pack.title,
                          "description": pack.description, "kind": pack.kind,
                          "kind_title": KIND_TITLES_RU[pack.kind], "mode": pack.mode,
                          "progress": pack_progress(pack, ratings), "decoded": self.is_decoded(index)})
        return {"title": self.collection.title, "is_collection": self.collection.is_collection,
                "readme": self.collection.readme if self.collection.is_collection else "", "packs": packs}

    def pack_view(self, index: int) -> dict[str, Any]:
        """Everything the page needs for one pack: groups, media URLs and saved ratings (no key data)."""
        pack = self.pack(index)
        ratings = self.stores[index].load()
        groups = []
        for g_index, group in enumerate(pack.groups):
            def item_view(item: MediaItem, role: str, number: int, g_index: int = g_index) -> dict[str, Any]:
                return {"id": item.id, "label": item.label, "kind": item.kind, "sync": item.sync,
                        "duration": item.duration, "url": f"/media/{index}/{g_index}/{role}/{number}"}

            dups: list[list[str]] | None = None
            if pack.mode == "comparison" and group.variants and all(
                v.path is not None and v.path.is_file() for v in group.variants
            ):
                dups = find_duplicate_variants(group.variants)
            groups.append({
                "index": g_index, "id": group.id, "title": group.title, "caption": group.caption,
                "sync": group.sync, "needs_prepare": group.source is not None,
                "duplicates": dups,
                "scale": {"min": group.scale_min, "max": group.scale_max},
                "video": ({"url": f"/media/{index}/{g_index}/video/0"} if group.video is not None else None),
                "variants": [item_view(item, "v", n) for n, item in enumerate(group.variants)],
                "references": [item_view(item, "r", n) for n, item in enumerate(group.references)],
            })
        return {"index": index, "id": pack.rel_id or pack.dir.name, "title": pack.title,
                "description": pack.description, "readme": pack.readme, "kind": pack.kind,
                "kind_title": KIND_TITLES_RU[pack.kind], "mode": pack.mode, "tags": pack.tags,
                "scale_labels": pack.scale_labels, "unique_ranks": bool(ratings.get("unique_ranks")),
                "decoded": self.is_decoded(index), "progress": pack_progress(pack, ratings),
                "ratings": ratings.get("groups") or {}, "groups": groups}

    def media_path(self, pack_index: int, group_index: int, role: str, number: int) -> Path:
        """File behind a media URL (builds the multitrack cache on first use)."""
        pack = self.pack(pack_index)
        if not 0 <= group_index < len(pack.groups):
            raise KeyError("group")
        group = pack.groups[group_index]
        if group.source is not None:
            self.prepare_group(pack_index, group_index)
        if role == "video":
            item = group.video
        else:
            items = group.variants if role == "v" else group.references if role == "r" else []
            item = items[number] if 0 <= number < len(items) else None
        if item is None or item.path is None:
            raise KeyError("media")
        return item.path

    # -- decoding (the only place that touches the key)

    def results_paths(self, index: int) -> tuple[Path, Path]:
        """Decoded results files of a pack."""
        folder = state_dir_for(self.pack(index), self.state_root)
        return folder / RESULTS_JSON, folder / RESULTS_MD

    def is_decoded(self, index: int) -> bool:
        """True once the results were decoded (by the page or ``--analyze``)."""
        return self.results_paths(index)[0].is_file()

    def decode(self, index: int) -> dict[str, Any]:
        """Open the key of one pack, write and return the decoded results."""
        from ui import blind_decode  # imported here on purpose: nothing else may reach the key

        with self._decode_lock:
            pack = self.pack(index)
            store = self.stores[index]
            store.update_settings(decoded=True)
            results = blind_decode.decode_pack(pack, store.load(), state_dir_for(pack, self.state_root))
            return {"results": results, "markdown": blind_decode.render_markdown(results)}

    def saved_results(self, index: int) -> dict[str, Any] | None:
        """Results decoded earlier (read from the results file, not from the key)."""
        json_path, md_path = self.results_paths(index)
        if not json_path.is_file():
            return None
        try:
            results = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return {"results": results, "markdown": _read_text(md_path)}


class _Handler(BaseHTTPRequestHandler):
    """HTTP endpoints of the page (see :class:`BlindListenApp`)."""

    server_version = "BlindListen/1"
    app: BlindListenApp  # set on the subclass by make_server

    def log_message(self, format: str, *args: Any) -> None:  # same signature as the base class
        log.debug("%s - %s", self.address_string(), format % args)

    # -- responses

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _same_origin(self) -> bool:
        """Reject cross-site requests (another web page must not rate or decode on the user's behalf)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlsplit(origin).netloc == (self.headers.get("Host") or "")

    # -- routing

    def do_HEAD(self) -> None:  # http.server naming
        self._route_get(head_only=True)

    def do_GET(self) -> None:
        self._route_get(head_only=False)

    def _route_get(self, head_only: bool) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._send_page(head_only)
            elif url.path == "/api/packs":
                self._send_json(self.app.packs_summary())
            elif url.path == "/api/pack":
                self._send_json(self.app.pack_view(int(query.get("pack", ["0"])[0])))
            elif url.path == "/api/ratings":
                index = int(query.get("pack", ["0"])[0])
                self.app.pack(index)
                self._send_json(self.app.stores[index].load())
            elif url.path == "/api/results":
                saved = self.app.saved_results(int(query.get("pack", ["0"])[0]))
                if saved is None:
                    self._error(HTTPStatus.NOT_FOUND, "Ключи этого набора ещё не расшифрованы")
                else:
                    self._send_json(saved)
            elif url.path.startswith("/media/"):
                self._send_media(url.path, head_only)
            elif url.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._error(HTTPStatus.NOT_FOUND, "Нет такой страницы")
        except (KeyError, ValueError, IndexError):
            self._error(HTTPStatus.NOT_FOUND, "Не найдено")
        except RuntimeError as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if not self._same_origin():
            self._error(HTTPStatus.FORBIDDEN, "Запрос с чужой страницы отклонён")
            return
        url = urlsplit(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(length, 2_000_000)).decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("body")
        except (ValueError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "Неверный запрос")
            return
        try:
            if url.path == "/api/ratings":
                self._post_ratings(body)
            elif url.path == "/api/settings":
                index = int(body.get("pack", 0))
                self.app.pack(index)
                data = self.app.stores[index].update_settings(unique_ranks=bool(body.get("unique_ranks")))
                self._send_json({"ok": True, "unique_ranks": data["unique_ranks"]})
            elif url.path == "/api/prepare":
                pack_index = int(body.get("pack", 0))
                group_index = int(body.get("group", 0))
                seconds = self.app.prepare_group(pack_index, group_index)
                pack = self.app.pack(pack_index)
                group = pack.groups[group_index]
                dups = find_duplicate_variants(group.variants) if pack.mode == "comparison" else []
                self._send_json({"ok": True, "seconds": round(seconds, 2), "duplicates": dups})
            elif url.path == "/api/decode":
                if body.get("confirm") is not True:
                    self._error(HTTPStatus.BAD_REQUEST, "Расшифровка требует подтверждения")
                    return
                self._send_json(self.app.decode(int(body.get("pack", 0))))
            else:
                self._error(HTTPStatus.NOT_FOUND, "Нет такой команды")
        except (KeyError, IndexError, TypeError, ValueError):
            self._error(HTTPStatus.NOT_FOUND, "Не найдено")
        except (RuntimeError, OSError) as error:
            log.exception("request failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def _post_ratings(self, body: dict[str, Any]) -> None:
        index = int(body.get("pack", 0))
        pack = self.app.pack(index)
        group = pack.group(str(body.get("group", "")))
        if group is None:
            self._error(HTTPStatus.NOT_FOUND, "Нет такой группы")
            return
        data = self.app.stores[index].save_group(group, body.get("state"))
        self._send_json({"ok": True, "state": data["groups"][group.id], "progress": pack_progress(pack, data),
                         "saved_at": data["updated_at"]})

    # -- static page and media

    def _send_page(self, head_only: bool) -> None:
        body = HTML_TEMPLATE.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_media(self, path: str, head_only: bool) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 5:
            raise KeyError("media")
        _, pack_index, group_index, role, number = parts
        file_path = self.app.media_path(int(pack_index), int(group_index), role, int(number))
        send_file_range(self, file_path, head_only)


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """``(start, end)`` inclusive for a single ``bytes=`` range; None for no/unsupported range.

    Raises ``ValueError`` for a range that cannot be satisfied (HTTP 416).
    """
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    spec = header[len("bytes="):].strip()
    start_text, _, end_text = spec.partition("-")
    try:
        if not start_text:  # suffix: the last N bytes
            length = int(end_text)
            if length <= 0:
                raise ValueError("range")
            return max(size - length, 0), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError as error:
        raise ValueError("range") from error
    if start >= size or start > end:
        raise ValueError("range")
    return start, min(end, size - 1)


def send_file_range(handler: BaseHTTPRequestHandler, path: Path, head_only: bool = False) -> None:
    """Send a file with ``Range`` support (206 Partial Content), as browsers need for seeking."""
    size = path.stat().st_size
    content_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    try:
        byte_range = parse_range(handler.headers.get("Range"), size)
    except ValueError:
        handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return
    start, end = byte_range if byte_range else (0, size - 1)
    length = max(end - start + 1, 0)
    handler.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "no-cache")
    if byte_range:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if head_only or length == 0:
        return
    with path.open("rb") as source:
        source.seek(start)
        remaining = length
        while remaining > 0:
            chunk = source.read(min(256 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def make_server(app: BlindListenApp, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """An HTTP server bound to ``host:port`` serving ``app`` (port 0 picks a free one)."""
    handler = type("BlindListenHandler", (_Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


# --------------------------------------------------------------------------------------------- CLI


def analyze(root: Path, state_root: Path | None = None) -> list[Path]:
    """Offline decoding: ``ratings.json`` + key → ``results_decoded.json/.md`` for every pack; returns md paths."""
    from ui import blind_decode

    collection = discover(root)
    written: list[Path] = []
    combined: list[str] = []
    for pack in collection.packs:
        folder = state_dir_for(pack, state_root)
        ratings = load_ratings(folder / RATINGS_NAME, pack)
        try:
            results = blind_decode.decode_pack(pack, ratings, folder)
        except (OSError, ValueError) as error:
            log.error("%s: %s", pack.title, error)
            continue
        written.append(folder / RESULTS_MD)
        combined.append(blind_decode.render_markdown(results, heading_level=2))
    if collection.is_collection and combined:
        folder = state_root or collection.root
        text = f"# Итоги слепого прослушивания: {collection.title}\n\n" + "\n\n".join(combined)
        write_text_atomic(folder / RESULTS_MD, text.rstrip() + "\n")
        written.append(folder / RESULTS_MD)
    return written


def main(argv: list[str] | None = None) -> int:
    """Command line: start the page or decode offline."""
    parser = argparse.ArgumentParser(
        prog="python -m ui.blind_listen",
        description="Слепое прослушивание: страница для оценки вариантов и расшифровка ключей.")
    parser.add_argument("pack_dir", type=Path, help="папка набора или общая папка с наборами")
    parser.add_argument("--analyze", action="store_true",
                        help="не запускать страницу, а расшифровать ratings.json по key.json в results_decoded.*")
    parser.add_argument("--check-duplicates", action="store_true",
                        help="проверить на 100%, что в каждой группе сравнения нет одинаковых вариантов (выход 1 при дублях)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"порт страницы (по умолчанию {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию только этот компьютер)")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="где хранить ratings.json и итоги (по умолчанию в папке набора)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="куда класть разделённые дорожки видео (по умолчанию <набор>/.listen_cache)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.check_duplicates:
            report = check_duplicates(args.pack_dir, args.cache_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        if args.analyze:
            for path in analyze(args.pack_dir, args.state_dir):
                print(f"Итоги записаны: {path}")
            return 0
        app = BlindListenApp(args.pack_dir, args.state_dir, args.cache_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    server = make_server(app, args.host, args.port)
    names = ", ".join(f"{p.title} ({len(p.groups)} гр.)" for p in app.collection.packs)
    print(f"Наборы: {names}")
    print(f"Откройте в браузере: http://{args.host}:{server.server_address[1]}  (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
