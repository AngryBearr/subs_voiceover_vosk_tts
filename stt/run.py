"""One command from an English video to a Russian voice-over (whole season at once if needed).

    stt_env/bin/python -m stt.run "Season 10/*.mkv" [--mode fast|accurate] [--until translate]

Per episode, in `<out-root>/<video name>/`:
    transcribe  English subtitles with speakers (stt.transcribe)       subs.en.json
    glossary    season glossary: names/terms + Russian renderings     glossary.json
    speakers    names, Russian labels, final gender (LLM)             speakers.llm.json
    translate   timing-aware EN -> RU translation (LLM)               subs.ru.json
    dub         voiced lines and voices                               dub.ru.json
    voiceover   synthesis + timeline fitting (`voiceover` package of the
                subtitle-shortening work, run from its own checkout)  voiceover/export/track.wav
    mix         voice over the ducked original (+ mkv with tracks)    <name>.ru.wav, <name>.ru.mkv

Finished stages are skipped (no repeated LLM costs); `--force STAGE` recomputes it and
everything after it. Defaults come from `stt_config.json`. Episodes in the same folder share a
season glossary (`<out-root>/<folder name>.glossary.json`).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from stt.runner import REPO_ROOT, load_json, save_json

log = logging.getLogger("stt.run")

STAGES = ["transcribe", "glossary", "speakers", "translate", "dub", "voiceover", "mix"]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dotenv(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """KEY=VALUE lines of the repo .env (parsed as data, never executed)."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2).strip().strip("'\"")
    return values


class Episode:
    def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.media = media
        self.out = out_root / media.stem
        self.args = args
        self.config = config
        default_season = out_root / f"{media.parent.name}.glossary.json"
        self.season = Path(args.season_glossary) if args.season_glossary else default_season
        self.forced_from: int | None = STAGES.index(args.force) if args.force else None

    def needed(self, stage: str, outputs: list[Path]) -> bool:
        if self.forced_from is not None and STAGES.index(stage) >= self.forced_from:
            return True
        return not all(p.exists() for p in outputs)

    def run(self) -> None:
        until = STAGES.index(self.args.until)
        steps = {
            "transcribe": (self.transcribe, [self.out / "subs.en.json"]),
            "glossary": (self.glossary, [self.out / "glossary.json"]),
            "speakers": (self.speakers, [self.out / "speakers.llm.json"]),
            "translate": (self.translate, [self.out / "subs.ru.json"]),
            "dub": (self.dub, [self.out / "dub.ru.json"]),
            "voiceover": (self.voiceover, [self.out / "voiceover" / "export" / "track.wav"]),
            "mix": (self.mix, [self.out / f"{self.media.stem}.ru.wav"]),
        }
        for position, stage in enumerate(STAGES[: until + 1]):
            func, outputs = steps[stage]
            if not self.needed(stage, outputs):
                log.info("[%s] %s: done earlier", self.media.stem, stage)
                continue
            log.info("[%s] %s ...", self.media.stem, stage)
            func()
            if self.forced_from is None or position < self.forced_from:
                self.forced_from = position + 1  # later stages depend on fresh output

    # ---- stages
    def transcribe(self) -> None:
        from stt import transcribe

        mode = self.config["modes"][self.args.mode]
        argv = [str(self.media), "--out", str(self.out), "--asr", mode["asr"],
                "--diarizer", self.config["diarizer"]]
        if mode.get("fuse"):
            argv += ["--fuse", ",".join(mode["fuse"]), "--fuse-model", self.config["llm_model"]]
            if (self.season.with_suffix(".txt")).exists():
                argv += ["--glossary", str(self.season.with_suffix(".txt"))]
        ocr = self.config.get("ocr", {})
        argv += ["--ocr-fps", str(ocr.get("fps", 2.0))] if ocr.get("enabled", True) else ["--no-ocr"]
        if self.args.start:
            argv += ["--start", str(self.args.start)]
        if self.args.duration:
            argv += ["--duration", str(self.args.duration)]
        transcribe.main(argv)

    def glossary(self) -> None:
        from stt import glossary

        glossary.main([str(self.out), "--season", str(self.season), "--model", self.config["llm_model"]])
        # Plain-text copy of the season glossary for the next episode's ASR fusion prompt.
        lines = (self.out / "glossary.txt").read_text(encoding="utf-8")
        self.season.with_suffix(".txt").write_text(lines, encoding="utf-8")

    def speakers(self) -> None:
        from stt import speakers

        speakers.main([str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model",
                       self.config["llm_model"]])

    def translate(self) -> None:
        from stt import translate

        cfg = self.config["translate"]
        argv = [str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model", self.config["llm_model"],
                "--cps", str(cfg["cps"]), "--extend", str(cfg["extend"]), "--parallel", str(cfg["parallel"])]
        if cfg.get("tts_profile"):
            argv += ["--tts-profile", str(REPO_ROOT / cfg["tts_profile"])]
        translate.main(argv)

    def dub(self) -> None:
        from stt import dub

        cfg = self.config["dub"]
        dub.main([str(self.out), "--voices", cfg["voices"], "--interjections", cfg["interjections"]])

    def voiceover(self) -> None:
        cfg = self.config["voiceover"]
        repo = Path(self.args.voiceover or cfg.get("repo") or "")
        if not repo or not (repo / "voiceover" / "pipeline.py").exists():
            raise SystemExit("voiceover: set --voiceover PATH (a checkout of the subtitle-shortening branch "
                             "with its venv and Vosk model) or voiceover.repo in stt_config.json")
        python = repo / cfg["python"]
        vo_config = load_json(repo / cfg["config"])
        vo_config["engine"]["voice"] = int(cfg["voice"])
        config_path = self.out / "voiceover.config.json"
        save_json(config_path, vo_config)
        env = {**os.environ, **dotenv(), "PATH": f"{python.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        cmd = [str(python), "-m", "voiceover.pipeline", "run", str((self.out / "dub.ru.json").resolve()),
               "--out", str((self.out / "voiceover").resolve()), "--config", str(config_path.resolve())]
        log_path = self.out / "logs" / "voiceover.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            code = subprocess.run(cmd, cwd=repo, env=env, stdout=log_file, stderr=subprocess.STDOUT).returncode
        if code != 0:
            raise SystemExit(f"voiceover failed (exit {code}), see {log_path}")

    def mix(self) -> None:
        from stt import mix

        cfg = self.config["mix"]
        out_wav = self.out / f"{self.media.stem}.ru.wav"
        argv = [str(self.media), str(self.out / "voiceover" / "export" / "track.wav"), "--out", str(out_wav),
                "--duck-db", str(cfg["duck_db"])]
        if self.args.start:
            argv += ["--start", str(self.args.start)]
        if cfg.get("mux") and not self.args.start:
            argv += ["--mux", str(self.out / f"{self.media.stem}.ru.mkv"), "--srt", str(self.out / "subs.ru.srt")]
        mix.main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="English video -> Russian voice-over, end to end")
    parser.add_argument("media", nargs="+", help="Video files or glob patterns (a whole season works)")
    parser.add_argument("--out-root", default="output/stt", help="One sub-folder per episode")
    parser.add_argument("--config", default=str(REPO_ROOT / "stt_config.json"))
    parser.add_argument("--mode", choices=["fast", "accurate"], default=None, help="Default from the config")
    parser.add_argument("--until", choices=STAGES, default="mix", help="Last stage to run")
    parser.add_argument("--force", choices=STAGES, default=None, help="Recompute this stage and the ones after it")
    parser.add_argument("--season-glossary", default=None, help="Default: <out-root>/<folder>.glossary.json")
    parser.add_argument("--voiceover", default=None, help="Checkout of the subtitle-shortening branch")
    parser.add_argument("--start", type=float, default=None, help="Process from this time (tests), seconds")
    parser.add_argument("--duration", type=float, default=None, help="Process this many seconds (tests)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    config = load_config(Path(args.config))
    args.mode = args.mode or config["mode"]
    files: list[Path] = []
    for pattern in args.media:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        files += [Path(m) for m in matches]
    missing = [f for f in files if not f.is_file()]
    if missing or not files:
        raise SystemExit(f"Media not found: {missing or args.media}")
    for media in files:
        Episode(media, Path(args.out_root), args, config).run()


if __name__ == "__main__":
    main()
