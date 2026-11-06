"""One subtitle base: the best English subtitle file per episode, picked from a hunt archive.

    python -m stt.subtitle_base build --archive <hunt> --out <base> [--show survivor] [--dry-run]

The archive (``manifest.jsonl``: season, episode, path, language, ...) holds many files per episode. Every candidate is
read again with the current label rules (``stt.subtitles.describe_text``: names in any case), not judged by the counts
in the manifest, and ranked per episode: labels > turns > plain, then named lines, distinct names, turn marks, cues.
The winner is written to ``<base>/<show>/S<ss>/S<ss>E<ee>.en.srt`` and ``<base>/<show>/index.json`` lists, per
episode, the file, its facts, the original path and the next two candidates (``alternates``). ``stt.run`` reads the
base through ``subtitles.base`` (``stt.subtitles.base_candidates``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from stt import subtitles as subs
from utils.atomic_io import write_text_atomic

log = logging.getLogger("stt.subtitle_base")

REPO_ROOT = Path(__file__).resolve().parent.parent
ALTERNATES = 2
KINDS = ("labels", "turns", "plain")
# the hunt's own kinds (``tools/analyze.py``) in our words, for the "hunt" line of the summary
HUNT_KINDS = {"labels": "labels", "labels_mixed": "labels", "turn_names": "turns", "turn": "turns", "plain": "plain"}


def episode_key(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:02d}"


def show_imdb(show: str, config_path: Path | None = None) -> str | None:
    """IMDb id of a show key ("survivor") from the ``subtitles.imdb`` map of the config."""
    from stt.run import load_config

    config = load_config(config_path or REPO_ROOT / "stt_config.json")
    mapping = (config.get("subtitles") or {}).get("imdb")
    return mapping.get(show.lower()) if isinstance(mapping, dict) else mapping


def rank(entry: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Better first: the kind, then named lines, distinct names, turn marks, cues."""
    return (subs._KIND_RANK[entry["kind"]], entry["named_lines"], entry["names"], entry["turn_marks"], entry["cues"])


def read_manifest(archive: Path) -> dict[tuple[int, int], list[Path]]:
    """Distinct English files per (season, episode) of ``archive/manifest.jsonl``."""
    found: dict[tuple[int, int], list[Path]] = {}
    manifest = archive / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        season, episode = row.get("season"), row.get("episode")
        if not isinstance(season, int) or not isinstance(episode, int) or row.get("language", "en") != "en":
            continue
        path = Path(str(row.get("path", "")))
        path = path if path.is_absolute() else archive / path
        if path not in found.setdefault((season, episode), []):
            found[(season, episode)].append(path)
    return found


def evaluate(path: Path) -> tuple[dict[str, Any], str] | None:
    """(facts, SRT text) of one file, read with the current label rules; None when it is not usable subtitles."""
    candidate = subs.file_candidate("archive", path)
    if candidate is None:
        return None
    info = candidate.info
    return {"kind": info["kind"], "named_lines": info["labels"], "names": info["names"],
            "turn_marks": info["turn_marks"], "cues": info["cues"], "source": str(path)}, candidate.text


def rank_episode(paths: list[Path]) -> list[tuple[dict[str, Any], str]]:
    """The usable candidates of one episode, best first."""
    evaluated = [e for p in paths if (e := evaluate(p)) is not None]
    return sorted(evaluated, key=lambda e: rank(e[0]), reverse=True)


def build(archive: Path, out: Path, show: str = "survivor", imdb: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Pick the best file per episode and write the base (nothing is written with ``dry_run``). Returns the summary:
    ``index`` (what index.json holds), ``before`` / ``after`` (episodes per kind: the hunt's own kinds of the manifest /
    read again with the current rules)."""
    by_episode = read_manifest(archive)
    kinds_before: dict[tuple[int, int], str] = {}
    for line in (archive / "manifest.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (row.get("season"), row.get("episode"))
        kind = HUNT_KINDS.get(str(row.get("kind")))
        if key in by_episode and kind:
            best = kinds_before.get(key)  # type: ignore[arg-type]
            if best is None or subs._KIND_RANK[kind] > subs._KIND_RANK[best]:
                kinds_before[key] = kind  # type: ignore[index]
    episodes: dict[str, Any] = {}
    texts: dict[str, str] = {}
    for (season, episode), paths in sorted(by_episode.items()):
        ranked = rank_episode(paths)
        if not ranked:
            log.warning("%s: no usable file among %d", episode_key(season, episode), len(paths))
            continue
        key = episode_key(season, episode)
        winner, text = ranked[0]
        file = f"{show}/S{season:02d}/{key}.en.srt"
        episodes[key] = {"file": file, **winner, "alternates": [facts for facts, _ in ranked[1:1 + ALTERNATES]]}
        texts[file] = text
    index = {"show": show, "imdb": imdb, "archive": str(archive), "episodes": episodes}
    before = Counter(kinds_before.get(k, "plain") for k in by_episode)
    after = Counter(e["kind"] for e in episodes.values())
    if not dry_run:
        for file, text in texts.items():
            write_text_atomic(out / file, text)
        write_text_atomic(out / show / subs.BASE_INDEX, json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    return {"index": index, "before": dict(before), "after": dict(after), "episodes": len(by_episode)}


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"episodes in the archive: {summary['episodes']}, in the base: {len(summary['index']['episodes'])}"]
    for label, title in (("before", "hunt kinds"), ("after", "read now")):
        lines.append(f"{title:>10}: " + ", ".join(f"{k} {summary[label].get(k, 0)}" for k in KINDS))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("build", help="pick the best file per episode of a hunt archive")
    cmd.add_argument("--archive", required=True, type=Path, help="folder with manifest.jsonl")
    cmd.add_argument("--out", required=True, type=Path, help="base folder to write")
    cmd.add_argument("--show", default="survivor", help="show key: a folder in the base and a key of subtitles.imdb")
    cmd.add_argument("--config", type=Path, default=None, help="config with the subtitles.imdb map (default stt_config.json)")
    cmd.add_argument("--dry-run", action="store_true", help="write nothing, print the counts per kind before and after")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    archive = args.archive.expanduser()
    if not (archive / "manifest.jsonl").is_file():
        parser.error(f"no manifest.jsonl in {archive}")
    summary = build(archive, args.out.expanduser(), args.show, show_imdb(args.show, args.config), args.dry_run)
    print(format_summary(summary))
    print("dry run: nothing written" if args.dry_run else f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
