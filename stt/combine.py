"""One video per episode with every voice-over as its own track, for episodes voiced before that.

    subs_env/bin/python -m stt.combine "output/stt_s20/*" [--delete-separate]

Episode folders of ``stt.run`` voiced into a video per engine (``<name>.ru.<engine>.mkv``) get the
combined ``<name>.ru.mkv`` that ``stt.run`` now writes itself (``mix.tracks: combined``), in
``paths.finished`` when set. Each engine's mix is copied out of its own video as it is (no new
mixing, no re-encoding), or taken from ``<name>.ru.<engine>.m4a`` / the mixed WAV. The video is
checked (every engine's track, the original, the subtitles); then ``--delete-separate`` removes
the per-engine videos and mixed WAVs.
"""

from __future__ import annotations

import argparse
import glob
import logging
import subprocess
import sys
from pathlib import Path

from stt.run import REPO_ROOT, Episode, load_config
from stt.runner import load_json
from utils.ffmpeg import ffmpeg_exe

log = logging.getLogger("stt.combine")


def engine_track(episode: Episode, engine: str) -> Path | None:
    """The engine's mix as AAC: kept, copied from the engine's own video, or encoded from its WAV."""
    from stt import mix

    track = episode.track(engine)
    if track.exists():
        return track
    stem = episode.media.stem
    for folder in dict.fromkeys([episode.out, episode.finished]):
        video = folder / f"{stem}.ru.{engine}.mkv"
        if video.exists():  # the voice-over is its first audio track
            subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                            "-map", "0:a:0", "-c", "copy", str(track)], check=True)
            return track
    wav = episode.out / f"{stem}.ru.{engine}.wav"
    if wav.exists():
        mix.encode_track(wav, track)
        return track
    return None


def combine(folder: Path, config: dict, delete_separate: bool = False) -> Path | None:
    """The combined video of one episode folder; None when it has no finished voice-over."""
    from stt import mix

    source = folder / "source.json"
    if not source.exists():
        return None
    media = Path(load_json(source)["media"])
    voiced = [e for e in config["voiceover"].get("engines", {}) if (folder / ".stages" / f"mix.{e}").exists()]
    if not voiced or not media.exists():
        log.warning("%s: %s", folder.name, "no finished voice-over" if not voiced else f"video missing: {media}")
        return None
    args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine=voiced[0], start=None,
                              duration=None)
    episode = Episode(media, folder.parent, args, config)
    engines = [e for e in voiced if engine_track(episode, e)]
    if not engines:
        log.warning("%s: no mix found for %s", folder.name, ", ".join(voiced))
        return None
    episode.mux()
    out = episode.finished_video()
    kinds = [kind for kind, _ in mix.streams(out)]
    complete = kinds.count("audio") == len(engines) + 1 and kinds.count("subtitle") == 1
    log.info("%s: %s (%s)%s", folder.name, out, ", ".join(engines), "" if complete else f", CHECK FAILED: {kinds}")
    if delete_separate and complete:
        stale = folder / f"{media.stem}.ru.mkv"  # an earlier combined video in the episode folder
        if stale.exists() and stale != out:
            stale.unlink()
            log.info("%s: deleted %s", folder.name, stale.name)
        for engine in voiced:
            for path in (folder / f"{media.stem}.ru.{engine}.mkv", episode.finished / f"{media.stem}.ru.{engine}.mkv",
                         folder / f"{media.stem}.ru.{engine}.wav"):
                if path.exists() and path != out:
                    path.unlink()
                    log.info("%s: deleted %s", folder.name, path.name)
    return out if complete else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One video per episode with every voice-over as a track")
    parser.add_argument("folders", nargs="+", help="Episode folders of stt.run or glob patterns")
    parser.add_argument("--config", default=str(REPO_ROOT / "stt_config.json"))
    parser.add_argument("--delete-separate", action="store_true",
                        help="Delete the per-engine videos and mixed WAVs once the combined video is checked")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    config = load_config(Path(args.config))
    folders = [Path(m) for pattern in args.folders for m in sorted(glob.glob(pattern))]
    failed = 0
    for folder in folders:
        if folder.is_dir() and (folder / "source.json").exists():
            failed += combine(folder, config, args.delete_separate) is None
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
