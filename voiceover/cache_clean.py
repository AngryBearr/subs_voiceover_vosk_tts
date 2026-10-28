"""Clean the TTS cache: keep the takes the episode folders use, delete the rest.

Every synthesised line is a ``<cache>/<engine set>/<key>.wav`` with its ``.json`` (sets change with
the engine version and settings). An episode's voice-over names the takes it used in
``voiceover.<engine>/*.json`` (``durations0.json``, ``speech_check.json``, ...); takes no episode
names are only good for a re-run with exactly the same text and settings, which a new voice-over
does not make. Takes newer than ``--keep-days`` stay too (a run still going or cut off and resumed
later). Without ``--delete`` only reports what would go.

    subs_env/bin/python -m voiceover.cache_clean [--roots output/stt output/stt_s29_v2 ...] [--delete]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("voiceover.cache_clean")

REPO_ROOT = Path(__file__).resolve().parents[1]
TAKE = re.compile(r"tts_cache/([0-9a-f]{16})/([0-9a-f]+)\.wav")


@dataclass
class Plan:
    used: set[tuple[str, str]] = field(default_factory=set)  # (set, key) named by an episode
    keep_files: int = 0
    keep_bytes: int = 0
    drop: list[Path] = field(default_factory=list)
    drop_bytes: int = 0
    sets_emptied: list[Path] = field(default_factory=list)


def default_roots() -> list[Path]:
    """Work folders of ``stt_config.json`` (+ local) and every ``output/*`` folder with episodes."""
    roots = {p for p in (REPO_ROOT / "output").glob("*") if p.is_dir() and p.name != "tts_cache"}
    try:
        from stt.run import load_config

        work = (load_config(REPO_ROOT / "stt_config.json").get("paths") or {}).get("work")
        if work:
            path = Path(work).expanduser()
            roots.add(path if path.is_absolute() else REPO_ROOT / path)
    except Exception:  # noqa: BLE001 - the output folders are enough
        pass
    return sorted(roots)


def used_takes(roots: list[Path]) -> set[tuple[str, str]]:
    """(set, key) of every take named in the voice-over files under the roots."""
    used: set[tuple[str, str]] = set()
    for root in roots:
        for path in root.rglob("voiceover.*/*.json"):
            try:
                used |= set(TAKE.findall(path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return used


def plan(cache: Path, roots: list[Path], keep_days: float) -> Plan:
    result = Plan(used=used_takes(roots))
    fresh_after = time.time() - keep_days * 86400
    for set_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        kept = False
        for path in sorted(set_dir.iterdir()):
            if path.is_dir():  # a worker's temporary folder
                kept = kept or path.stat().st_mtime >= fresh_after
                continue
            key = path.name.split(".", 1)[0]
            size = path.stat().st_size
            if (set_dir.name, key) in result.used or path.stat().st_mtime >= fresh_after:
                result.keep_files += 1
                result.keep_bytes += size
                kept = True
            else:
                result.drop.append(path)
                result.drop_bytes += size
        if not kept:
            result.sets_emptied.append(set_dir)
    return result


def apply(result: Plan) -> None:
    for path in result.drop:
        path.unlink(missing_ok=True)
    for set_dir in result.sets_emptied:
        for leftover in sorted(set_dir.rglob("*"), reverse=True):  # stale worker folders
            leftover.rmdir() if leftover.is_dir() else leftover.unlink()
        set_dir.rmdir()


def cache_dir() -> Path:
    config = json.loads((REPO_ROOT / "voiceover_config.json").read_text(encoding="utf-8"))
    path = Path(str((config.get("synthesis") or {}).get("cache_dir", "output/tts_cache")))
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete TTS cache takes no episode folder uses")
    parser.add_argument("--cache", type=Path, default=None, help="default: synthesis.cache_dir of voiceover_config.json")
    parser.add_argument("--roots", type=Path, nargs="*", default=None,
                        help="folders with episode work folders (default: paths.work and output/*)")
    parser.add_argument("--keep-days", type=float, default=3.0, help="takes newer than this stay (default 3)")
    parser.add_argument("--delete", action="store_true", help="delete (without it: only report)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cache = args.cache or cache_dir()
    if not cache.is_dir():
        log.info("no TTS cache at %s", cache)
        return 0
    roots = args.roots if args.roots is not None else default_roots()
    result = plan(cache, roots, args.keep_days)
    gb = 2 ** 30
    log.info("TTS cache %s: %d takes named by episodes; keep %d files (%.1f GB), %s %d files (%.1f GB), "
             "%d of %d sets emptied", cache, len(result.used), result.keep_files, result.keep_bytes / gb,
             "deleting" if args.delete else "would delete", len(result.drop), result.drop_bytes / gb,
             len(result.sets_emptied), sum(p.is_dir() for p in cache.iterdir()))
    if args.delete:
        apply(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
