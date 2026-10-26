"""One command from an English video to a Russian voice-over (whole season at once if needed).

    subs_env/bin/python -m stt.run "Season 10/*.mkv" [--mode fast|accurate] [--until translate]

Per episode, in `<out-root>/<video name>/`:
    transcribe  English subtitles with speakers (stt.transcribe)       subs.en.json
    glossary    season glossary: names/terms + Russian renderings     glossary.json
    speakers    names, Russian labels, final gender (LLM + voice bank) speakers.llm.json
    translate   timing-aware EN -> RU translation (LLM)               subs.ru.json
    dub         voiced lines and a voice per line (speaker gender)    dub.ru.<engine>.json
    stress      stress of guessed words/homographs re-checked (LLMs)  stress.json
    voiceover   synthesis (Vosk or TeraTTS) + timeline fitting        voiceover.<engine>/export/track.wav
    check       Russian ASR check of the spoken clips                 voiceover.<engine>/speech_check.md
                + lines to check by ear (gender / speaker)            review.json, review.srt
    mix         voice over the ducked original (+ mkv with tracks)    <name>.ru.<engine>.wav / .mkv

Finished stages are skipped (no repeated LLM costs) unless an earlier stage was redone since;
`--force STAGE` recomputes it and everything after it. Defaults come from `stt_config.json`. Episodes in the same folder share a
season glossary (`<out-root>/<folder name>.glossary.json`) and voice bank
(`<out-root>/<folder name>.voices.json`, `stt.voice_bank`).
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from stt.runner import REPO_ROOT, load_json, save_json
from utils.llm_route import with_route

log = logging.getLogger("stt.run")

STAGES = ["transcribe", "glossary", "speakers", "translate", "dub", "stress", "voiceover", "check", "mix"]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``override`` on top: nested dicts merge, everything else is replaced."""
    out = dict(base)
    for key, value in override.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: Path) -> dict[str, Any]:
    """The config, with ``<name>.local.json`` next to it on top (personal settings, not in git:
    e.g. LLM requests through a ChatGPT subscription in OpenCode)."""
    config = json.loads(path.read_text(encoding="utf-8"))
    local = path.with_name(path.stem + ".local.json")
    if local.exists():
        config = deep_merge(config, json.loads(local.read_text(encoding="utf-8")))
    return config


def dotenv(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """KEY=VALUE lines of the repo .env (parsed as data, never executed)."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2).strip().strip("'\"")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Episode:
    def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.media = media
        self.out = out_root / media.stem
        self.args = args
        self.config = config
        default_season = out_root / f"{media.parent.name}.glossary.json"
        self.season = Path(args.season_glossary) if args.season_glossary else default_season
        self.voice_bank = self.season.with_name(self.season.name.replace(".glossary.json", "") + ".voices.json")
        self.forced_from: int | None = STAGES.index(args.force) if args.force else None
        vo = config.get("voiceover", {})
        self.engine = str(getattr(args, "engine", None) or vo.get("engine", "vosk"))
        self.vo_dir = self.out / f"voiceover.{self.engine}"

    def llm(self, stage: str) -> Any:
        """Model of an LLM step: llm.stages.<stage>, else llm.model (one model for everything).

        The model id gets its route (``utils.llm_route``): models matching a pattern of
        ``llm.opencode`` (e.g. "openai/*") go through the OpenCode login of a subscription, with
        ``llm.router`` (openrouter | orcarouter) as fallback unless ``llm.fallback`` is false;
        the rest go to ``llm.router``. Ids that already carry a route keep it.
        """
        llm = self.config.get("llm") or {"model": self.config.get("llm_model", "openai/gpt-6-luna")}
        stages = llm.get("stages") or {}
        model = stages[stage] if stages.get(stage) is not None else llm["model"]
        patterns = llm.get("opencode") or []
        router = llm.get("router", "openrouter")

        def route(name: str) -> str:
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                return with_route(name, ["opencode", router] if llm.get("fallback", True) else ["opencode"])
            return with_route(name, [router])

        return [route(m) for m in model] if isinstance(model, list) else route(model)

    def stamp(self, stage: str) -> Path:
        """Marks when a stage last finished (per engine for the engine's stages)."""
        engine_stage = stage in ("dub", "voiceover", "check", "mix")
        return self.out / ".stages" / (f"{stage}.{self.engine}" if engine_stage else stage)

    def needed(self, stage: str, outputs: list[Path], previous: str | None = None,
               inputs: list[Path] | None = None) -> bool:
        """Run a stage if forced, if an output is missing, if the previous stage finished after it,
        or if a manual input (``speakers.override.json``, ``corrections.json``) changed since.

        Finish marks, not file times: some stages update earlier outputs (speakers rewrites
        subs.en.json), which must not make the earlier stages look stale.
        """
        if self.forced_from is not None and STAGES.index(stage) >= self.forced_from:
            return True
        if not all(p.exists() for p in outputs):
            return True
        mark = self.stamp(stage)
        if mark.exists() and any(p.exists() and p.stat().st_mtime > mark.stat().st_mtime for p in inputs or []):
            return True
        if previous is None or not self.stamp(previous).exists():
            return False  # runs from before the marks existed
        return not mark.exists() or mark.stat().st_mtime < self.stamp(previous).stat().st_mtime

    def run(self) -> None:
        until = STAGES.index(self.args.until)
        steps = {
            "transcribe": (self.transcribe, [self.out / "subs.en.json"]),
            "glossary": (self.glossary, [self.out / "glossary.json"]),
            "speakers": (self.speakers, [self.out / "speakers.llm.json"], [self.out / "speakers.override.json"]),
            "translate": (self.translate, [self.out / "subs.ru.json"]),
            "dub": (self.dub, [self.out / f"dub.ru.{self.engine}.json"],
                    [self.out / "corrections.json"]  # manual fixes; for F5 also new or replaced voice clips
                    + ([self.references / "references.json"] if self.engine == "f5" else [])),
            "stress": (self.stress, [self.out / "stress.json"]),
            "voiceover": (self.voiceover, [self.vo_dir / "export" / "track.wav"]),
            "check": (self.check, [self.vo_dir / "speech_check.json"]),
            "mix": (self.mix, [self.out / f"{self.media.stem}.ru.{self.engine}.wav"]),
        }
        for position, stage in enumerate(STAGES[: until + 1]):
            func, outputs, *inputs = steps[stage]
            if not self.needed(stage, outputs, STAGES[position - 1] if position else None, *inputs):
                log.info("[%s] %s: done earlier", self.media.stem, stage)
                continue
            log.info("[%s] %s ...", self.media.stem, stage)
            func()
            self.stamp(stage).parent.mkdir(parents=True, exist_ok=True)
            self.stamp(stage).touch()
            if self.forced_from is None or position < self.forced_from:
                self.forced_from = position + 1  # later stages depend on fresh output

    # ---- stages
    def transcribe(self) -> None:
        from stt import transcribe

        mode = self.config["modes"][self.args.mode]
        argv = [str(self.media), "--out", str(self.out), "--asr", mode["asr"],
                "--diarizer", self.config["diarizer"]]
        if mode.get("fuse"):
            argv += ["--fuse", ",".join(mode["fuse"]), "--fuse-model", self.llm("fuse")]
            if (self.season.with_suffix(".txt")).exists():
                argv += ["--glossary", str(self.season.with_suffix(".txt"))]
        ocr = self.config.get("ocr", {})
        argv += ["--ocr-fps", str(ocr.get("fps", 2.0))] if ocr.get("enabled", True) else ["--no-ocr"]
        refine = self.config.get("speaker_refine") or {}
        if refine.get("model", "none") != "none":
            argv += ["--refine-speakers", refine["model"], "--refine-margin", str(refine.get("margin", 0.1))]
        if self.args.start:
            argv += ["--start", str(self.args.start)]
        if self.args.duration:
            argv += ["--duration", str(self.args.duration)]
        transcribe.main(argv)

    @property
    def cast(self) -> Path:
        """The season's cast from the web (stt.cast)."""
        return self.voice_bank.with_name(self.voice_bank.name.replace(".voices.json", ".cast.json"))

    def _find_cast(self) -> bool:
        """Look the season's cast up once (cheap LLM with web search); a failure only costs the evidence."""
        cfg = self.config.get("cast") or {}
        if not cfg.get("enabled", True):
            return False
        if not self.cast.exists():
            from stt import cast

            model = (self.config.get("llm") or {}).get("stages", {}).get("cast") or cast.DEFAULT_MODEL
            try:
                cast.main([str(self.out), "--cast", str(self.cast), "--model", model])
            except Exception as error:  # noqa: BLE001 - names still come from the transcript and the voices
                log.warning("[%s] cast lookup failed: %s", self.media.stem, error)
        return self.cast.exists()

    def glossary(self) -> None:
        from stt import glossary

        argv = [str(self.out), "--season", str(self.season), "--model", self.llm("glossary")]
        glossary.main(argv + (["--cast", str(self.cast)] if self._find_cast() else []))
        # Plain-text copy of the season glossary for the next episode's ASR fusion prompt.
        lines = (self.out / "glossary.txt").read_text(encoding="utf-8")
        self.season.with_suffix(".txt").write_text(lines, encoding="utf-8")

    def speakers(self) -> None:
        from stt import speakers

        argv = [str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model", self.llm("speakers")]
        if (self.config.get("voice_bank") or {}).get("enabled", True):
            argv += ["--voice-bank", str(self.voice_bank)]
        if self._find_cast():
            argv += ["--cast", str(self.cast)]
        speakers.main(argv)

    def translate(self) -> None:
        from stt import translate

        cfg = self.config["translate"]
        argv = [str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model", self.llm("translate"),
                "--cps", str(cfg["cps"]), "--extend", str(cfg["extend"]), "--parallel", str(cfg["parallel"])]
        if cfg.get("tts_profile"):
            argv += ["--tts-profile", str(REPO_ROOT / cfg["tts_profile"])]
        translate.main(argv)

    def _engine(self) -> dict[str, Any]:
        cfg = self.config["voiceover"]
        engine = dict(cfg.get("engines", {}).get(self.engine, {}))
        if "voices" not in engine:
            raise SystemExit(f"voiceover.engines.{self.engine}.voices missing in the config")
        if self.engine == "f5":  # fallback voices: a Vosk voice per gender, cloned by F5
            from voiceover.engine_f5 import fallback_references

            vo = load_json(REPO_ROOT / cfg.get("config", "voiceover_config.json"))
            vosk = cfg.get("engines", {}).get("vosk", {}).get("voices", {"female": 49, "male": 56})
            refs = fallback_references(self.references.parent / "fallback_voices",
                                       REPO_ROOT / vo["engine"]["model_path"], {g: int(v) for g, v in vosk.items()})
            from voiceover.engine_f5 import voice_id

            engine["voices"] = {g: voice_id(refs[g] if v == "auto" else v) for g, v in engine["voices"].items()}
        return engine

    @property
    def references(self) -> Path:
        """Voice-cloning clips of the season's people (stt.voice_bank --export)."""
        return self.voice_bank.with_name(self.voice_bank.name.replace(".voices.json", ".references"))

    def _speaker_voices(self) -> Path:
        """{speaker id: reference clip} of this episode's people with a clip (F5 engine)."""
        from stt.voice_bank import same_person

        if self.voice_bank.exists():  # people new in this episode get a clip; others keep theirs
            self._module("voices", "stt.voice_bank", [str(self.voice_bank), "--export", str(self.references),
                                                       "--missing"])
        index = self.references / "references.json"
        refs = load_json(index) if index.exists() else []
        mapping = {}
        for card in load_json(self.out / "speakers.json"):
            ref = next((r for r in refs if card.get("name") and same_person(r["name"], card["name"])), None)
            if ref:
                from voiceover.engine_f5 import voice_id

                mapping[card["id"]] = voice_id((self.references / ref["wav"]).resolve())
        path = self.out / "speaker_voices.f5.json"
        save_json(path, mapping)
        log.info("[%s] F5 voices: %d speakers with their own clip", self.media.stem, len(mapping))
        return path

    def dub(self) -> None:
        from stt import dub

        voices = ",".join(f"{k}={v}" for k, v in self._engine()["voices"].items())
        argv = [str(self.out), "--voices", voices, "--interjections", self.config["dub"]["interjections"],
                "--output", f"dub.ru.{self.engine}.json"]
        if self.engine == "f5":
            argv += ["--speaker-voices", str(self._speaker_voices())]
        dub.main(argv)

    def _module(self, name: str, module: str, args: list[str]) -> None:
        """Run a module of this repo in the current interpreter, logging to logs/<name>.log."""
        env = {**os.environ, **dotenv(), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        log_path = self.out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            code = subprocess.run([sys.executable, "-m", module, *args], cwd=REPO_ROOT, env=env, stdout=log_file,
                                  stderr=subprocess.STDOUT).returncode
        if code != 0:
            raise SystemExit(f"{name} failed (exit {code}), see {log_path}")

    def _voiceover_config(self) -> Path:
        """voiceover_config.json with this run's engine, voices, text preparation and timing; saved per engine."""
        cfg = self.config["voiceover"]
        vo_config = load_json(REPO_ROOT / cfg.get("config", "voiceover_config.json"))
        engine = {k: v for k, v in self._engine().items() if k != "voices"}
        if "sample_rate" in engine:  # export rate of the track (TeraTTS speaks at 44.1 kHz)
            vo_config["export"] = {**vo_config.get("export", {}), "sample_rate": int(engine.pop("sample_rate"))}
        normalization = {**vo_config["engine"].get("normalization", {}), **cfg.get("normalization", {}),
                         **engine.pop("normalization", {})}
        if (self.out / "stress.json").exists():
            normalization["stress_dict"] = str((self.out / "stress.json").resolve())
        vo_config["engine"] = {**vo_config["engine"], **engine, "name": self.engine, "normalization": normalization,
                               "voice": self._engine()["voices"]["male"]}
        for section in ("policy", "units"):  # dub-specific timing (preferred lag, base tempo, no unit merging)
            vo_config[section] = {**vo_config.get(section, {}), **cfg.get(section, {})}
        openrouter = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                      "api_key_env": "OPENROUTER_API_KEY", "reasoning": "default", "max_tokens": 8192}
        vo_config["llm"] = {**vo_config.get("llm", {}), **openrouter, "model": self.llm("voiceover")}  # shortening
        vo_config["verify"] = {**vo_config.get("verify", {}),
                               "llm": {**vo_config.get("verify", {}).get("llm", {}), **openrouter,
                                       "model": self.llm("voiceover")}}  # meaning check of shortened lines
        config_path = self.out / f"voiceover.{self.engine}.config.json"
        save_json(config_path, vo_config)
        return config_path

    def stress(self) -> None:
        cfg = self.config["voiceover"]
        if not cfg.get("stress_check", True):
            save_json(self.out / "stress.json", {"schema": "voiceover-stress-v3", "skipped": True})
            return
        models = self.llm("stress")
        screen = self.llm("stress_screen") if (self.config.get("llm") or {}).get("stages", {}).get("stress_screen") \
            else None
        args = [str((self.out / f"dub.ru.{self.engine}.json").resolve()), "--out", str((self.out / "stress.json").resolve()),
                "--config", str(self._voiceover_config().resolve()),  # the texts exactly as this engine prepares them
                "--models", ",".join(models if isinstance(models, list) else [models])]
        self._module("stress", "voiceover.stress_check", args + (["--screen", screen] if screen else ["--no-screen"]))

    def voiceover(self) -> None:
        config_path = self._voiceover_config()
        self._reset_voiceover_dir()
        self._module(f"voiceover.{self.engine}", "voiceover.pipeline", [
            "run", str((self.out / f"dub.ru.{self.engine}.json").resolve()), "--out", str(self.vo_dir.resolve()),
            "--config", str(config_path.resolve())])

    def _reset_voiceover_dir(self) -> None:
        """A new dubbing script needs a fresh voice-over folder; manual overrides are kept."""
        cues = self.vo_dir / "cues.json"
        if not cues.exists() or load_json(cues).get("source_sha256") == _sha256(self.out / f"dub.ru.{self.engine}.json"):
            return
        overrides = self.vo_dir / "overrides.json"
        kept = overrides.read_bytes() if overrides.exists() else None
        shutil.rmtree(self.vo_dir)
        if kept is not None:
            self.vo_dir.mkdir(parents=True)
            overrides.write_bytes(kept)
        log.info("[%s] voiceover: dubbing script changed, started a fresh %s", self.media.stem, self.vo_dir.name)

    def check(self) -> None:
        if not self.config["voiceover"].get("speech_check", True):
            save_json(self.vo_dir / "speech_check.json", {"schema": "voiceover-speech-check-v1", "skipped": True})
            return
        self._module(f"check.{self.engine}", "voiceover.speech_check", [str(self.vo_dir.resolve()),
                                                                         "--root", str(REPO_ROOT)])
        review = self.out / "review.json"
        if self.config.get("review", True) and (
                not review.exists() or review.stat().st_mtime < (self.out / "subs.ru.json").stat().st_mtime):
            try:  # lines to check by ear: a help for a person, never a reason to stop the episode
                self._module("review", "stt.review", [str(self.out.resolve())])
            except SystemExit as error:
                log.warning("[%s] %s", self.media.stem, error)

    def mix(self) -> None:
        from stt import mix

        cfg = self.config["mix"]
        out_wav = self.out / f"{self.media.stem}.ru.{self.engine}.wav"
        argv = [str(self.media), str(self.vo_dir / "export" / "track.wav"), "--out", str(out_wav),
                "--duck-db", str(cfg["duck_db"])]
        if cfg.get("duck") == "voices":  # lower only the original voices, keep music and effects
            argv += ["--duck-voices-only", "--rest-duck-db", str(cfg.get("rest_duck_db", 3.0))]
        if self.args.start:  # a test piece: the voice-over is on the episode timeline from 0
            argv += ["--start", str(self.args.start), "--voice-from-zero"]
        crowd = self.config.get("crowd", {})
        if crowd.get("enabled"):  # experimental: cheers/chants by several voices, under the voice-over
            from stt import crowd as crowd_layer

            crowd_layer.main([str(self.out), "--engine", self.engine, "--voices", str(crowd.get("voices", 3))])
            argv += ["--extra", f"{self.out / f'crowd.{self.engine}.wav'}:{crowd.get('gain_db', -4.0)}"]
        if cfg.get("mux"):  # a test piece gets a short re-encoded video of the piece
            argv += ["--mux", str(self.out / f"{self.media.stem}.ru.{self.engine}.mkv")]
            if cfg.get("subtitles", True):  # Russian subtitles as a track of the mkv
                argv += ["--srt", str(self.out / "subs.ru.srt")]
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
    parser.add_argument("--engine", default=None, help="TTS engine (voiceover.engines in the config: vosk | teratts)")
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
