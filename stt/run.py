"""One command from an English video to a Russian voice-over (whole season at once if needed).

    subs_env/bin/python -m stt.run "Season 10/*.mkv" [--mode fast|accurate] [--until translate]

Per episode, in `<out-root>/<video name>/`:
    transcribe  English subtitles with speakers (stt.transcribe)       subs.en.json
                (+ existing subtitles that name speakers, if found)    subs.source.en.srt, subtitle_hints.json
    glossary    season glossary: names/terms + Russian renderings     glossary.json
    speakers    names, Russian labels, final gender (LLM + voice bank) speakers.llm.json
    translate   timing-aware EN -> RU translation (LLM)               subs.ru.json
    dub         voiced lines and a voice per line (speaker gender)    dub.ru.<engine>.json
    stress      stress of guessed words/homographs re-checked (LLMs)  stress.json
    voiceover   synthesis (Vosk or TeraTTS) + timeline fitting        voiceover.<engine>/export/track.wav
    check       Russian ASR check of the spoken clips                 voiceover.<engine>/speech_check.md
                + lines to check by ear (gender / speaker)            review.json, review.srt
    mix         voice over the ducked original as an AAC track        <name>.ru.<engine>.m4a
                + the finished video                                  <name>.ru.mkv (or .ru.<engine>.mkv)

Finished stages are skipped (no repeated LLM costs) unless an earlier stage was redone since;
`--force STAGE` recomputes it and everything after it. Defaults come from `stt_config.json`. Episodes in the same folder share a
season glossary (`<season dir>/<folder name>.glossary.json`) and voice bank
(`<season dir>/<folder name>.voices.json`, `stt.voice_bank`).

Folders (`paths` in the config): `work` holds the episode folders (default `output/stt`, `--out-root`
overrides it), `season` the season's files (glossary, cast, voice bank and voice clips; default:
`work`), `finished` the finished videos in a sub-folder per season (default: each episode's folder).
`mix.tracks`: "combined" puts every voice-over finished so far into one video as its own audio
track (`mix.track_order` first, the first is the default track), "separate" writes a video per engine.

Optional `season_voices` (config, off by default; `--no-season-voices`): once every episode has its transcript and
names (`speakers`), the words of each episode are assigned again over the people named in the whole season
(`stt.season_voices`, no LLM, no GPU); the later stages then run on the changed transcript.

Several engines (`--engine f5,teratts,vosk`) run episode by episode; with `voiceover.parallel_cpu`
(default) Vosk, which synthesises on the CPU, runs in a second lane next to the GPU engines: it
starts an episode once the GPU lane has its English subtitles and translation, and the stages
both lanes share (stress check, review list, the finished video) take turns.
"""

from __future__ import annotations

import argparse
import fnmatch
import functools
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from stt import progress
from stt.runner import REPO_ROOT, load_json, save_json
from utils.llm_route import split_route, with_route

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


def apply_preset(config: dict[str, Any], name: str | None) -> dict[str, Any]:
    """The config with a named set of settings (``presets`` in the config) on top; no name: as it is."""
    if not name:
        return config
    presets = config.get("presets") or {}
    if name not in presets:
        raise SystemExit(f"Unknown preset {name!r}; the config has: {', '.join(presets) or 'none'}")
    return deep_merge(config, presets[name].get("settings") or {})


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


def finished_folder(config: dict[str, Any], media: Path) -> Path | None:
    """``paths.finished`` / ``paths.finished_folder`` with ``{season}`` = the video's folder name
    ("STT_{season}" -> "STT_Season 20"); None when finished videos stay in the episode folders."""
    paths = config.get("paths") or {}
    if not paths.get("finished"):
        return None
    return Path(paths["finished"]).expanduser() / str(paths.get("finished_folder") or "{season}").format(
        season=media.parent.name)


class Background:
    """Stage runs a lane hands to threads (the previous engine's mix) so it can go on; ``join`` waits for all of them and
    raises the first error. Every thread is named after its engine, as the stages' log lines are."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []

    def start(self, name: str, func: Callable[[], None]) -> None:
        def work() -> None:
            try:
                func()
            except BaseException as error:  # noqa: BLE001 - raised by join in the lane's thread
                log.error("%s failed: %s", name, error)
                self._errors.append(error)

        thread = threading.Thread(target=work, name=name)
        self._threads.append(thread)
        thread.start()

    def failed(self) -> bool:
        return bool(self._errors)

    def join(self) -> None:
        for thread in self._threads:
            thread.join()
        self._threads.clear()
        if self._errors:
            error, self._errors = self._errors[0], []
            raise error


# On-screen translation of an episode running in the background (``Episode.start_onscreen``), by episode folder
_ONSCREEN_JOBS: dict[Path, Future[Path | None]] = {}
_ONSCREEN_GUARD = threading.Lock()
# What the combined video of an episode was last muxed from (the tracks and their times), by episode folder
_MUXED: dict[Path, tuple[tuple[str, int], ...]] = {}


def wait_onscreen() -> None:
    """Wait for every on-screen translation started in the background (their errors are logged, never raised)."""
    with _ONSCREEN_GUARD:
        jobs = list(_ONSCREEN_JOBS.values())
    for job in jobs:
        job.result()


class Episode:
    def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any],
                 lock: threading.Lock | None = None) -> None:
        self.media = media
        self.lock = lock  # shared with the other lane's Episode of the same video (parallel engines)
        self.out = out_root / media.stem
        self.args = args
        self.config = config
        paths = config.get("paths") or {}
        season_dir = Path(paths["season"]).expanduser() if paths.get("season") else out_root
        default_season = season_dir / f"{media.parent.name}.glossary.json"
        self.season = Path(args.season_glossary) if args.season_glossary else default_season
        self.voice_bank = self.season.with_name(self.season.name.replace(".glossary.json", "") + ".voices.json")
        self.season_stress = self.season.with_name(self.season.name.replace(".glossary.json", "") + ".stress.json")
        self.forced_from: int | None = STAGES.index(args.force) if args.force else None
        vo = config.get("voiceover", {})
        self.engine = str(getattr(args, "engine", None) or vo.get("engine", "vosk")).split(",")[0].strip()
        self.vo_dir = self.out / f"voiceover.{self.engine}"
        self._subtitles: dict[str, Any] | None | bool = False  # False: not looked up yet

    def subtitle_source(self) -> dict[str, Any] | None:
        """Existing English subtitles of the episode that tell who speaks ("labels" or "turns"), else None.
        Looked up once (embedded, next to the video, archive, online; cached in ``subs.source.json``)."""
        cfg = self.config.get("subtitles") or {}
        if self._subtitles is False:
            self._subtitles = None
            if cfg.get("enabled", False) and not getattr(self.args, "no_subtitles", False):
                from stt.subtitles import find_subtitles

                try:
                    found = find_subtitles(self.media, self.out, cfg)
                except Exception as error:  # noqa: BLE001 - subtitles are a bonus, never a reason to stop
                    log.warning("[%s] subtitle search failed: %s", self.media.stem, error)
                    found = None
                self._subtitles = found if found and found.get("kind") in ("labels", "turns") else None
        return self._subtitles if isinstance(self._subtitles, dict) else None

    def subtitle_key(self) -> str:
        """What the speaker assignment of the transcript uses from existing subtitles; "none" without them."""
        found = self.subtitle_source()
        if not found:
            return "none"
        bonus = (self.config.get("subtitles") or {}).get("turn_bonus", 0.3)
        return f"{found['kind']}:{found.get('sha1', '')}:{bonus}"

    @property
    def subtitle_stamp(self) -> Path:
        return self.out / ".stages" / "transcribe.subtitles"

    def llm(self, stage: str) -> Any:
        """Model of an LLM step: llm.stages.<stage>, else llm.model (one model for everything).

        The model id gets its route (``utils.llm_route``): models matching a pattern of
        ``llm.opencode`` (e.g. "openai/*") go through the OpenCode login of a subscription, with
        ``llm.router`` (openrouter | orcarouter, or a list tried in order) as fallback unless
        ``llm.fallback`` is false; the rest go to ``llm.router``. Ids that already carry a route
        keep it.
        """
        llm = self.config.get("llm") or {"model": self.config.get("llm_model", "openai/gpt-6-luna")}
        stages = llm.get("stages") or {}
        model = stages[stage] if stages.get(stage) is not None else llm["model"]
        return [self.route(m) for m in model] if isinstance(model, list) else self.route(model)

    def route(self, name: str) -> str:
        """A model id with its route (``llm``): see ``llm()``; web search (``:online``) is OpenRouter's."""
        llm = self.config.get("llm") or {}
        if name.endswith(":online"):
            return with_route(name, ["openrouter"])
        router = llm.get("router", "openrouter")
        routers = [router] if isinstance(router, str) else list(router)
        if any(fnmatch.fnmatchcase(name, p) for p in llm.get("opencode") or []):
            return with_route(name, ["opencode", *routers] if llm.get("fallback", True) else ["opencode"])
        return with_route(name, routers)

    def stamp(self, stage: str) -> Path:
        """Marks when a stage last finished (per engine for the engine's stages)."""
        engine_stage = stage in ENGINE_STAGES
        return self.out / ".stages" / (f"{stage}.{self.engine}" if engine_stage else stage)

    def needed(self, stage: str, outputs: list[Path], previous: str | None = None,
               inputs: list[Path] | None = None) -> bool:
        """Run a stage if forced, if an output is missing, if the previous stage finished after it,
        or if a manual input (``speakers.override.json``, ``corrections.json``) changed since.

        Finish marks, not file times: some stages update earlier outputs (speakers rewrites
        subs.en.json), which must not make the earlier stages look stale.
        """
        user_forced = self.args.force is not None and STAGES.index(stage) >= STAGES.index(self.args.force)
        if stage == "stress" and not user_forced and all(p.exists() for p in outputs) and self.stamp(stage).exists() \
                and self._stress_input_unchanged():
            return False  # another engine's dubbing script with the same lines: the same stress check
        if self.forced_from is not None and STAGES.index(stage) >= self.forced_from:
            return True
        if not all(p.exists() for p in outputs):
            return True
        # Other subtitles than the last assembly used: assemble again (cached stages stay). A transcript made before this
        # rule has no stamp and stays as it is: re-assembling would rerun every later stage, LLM steps included.
        if stage == "transcribe" and self.subtitle_stamp.exists():
            used = self.subtitle_stamp.read_text(encoding="utf-8").strip()
            if self.subtitle_key() != used:
                log.info("[%s] subtitles changed (%s -> %s): speakers are assigned again", self.media.stem, used,
                         self.subtitle_key())
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
            "mix": (self.mix, [self.track(self.engine)]),
        }
        background: Background | None = getattr(self.args, "mix_background", None)
        for position, stage in enumerate(STAGES[: until + 1]):
            func, outputs, *inputs = steps[stage]
            with self.lock if self.lock and stage in SHARED_STAGES else nullcontext():
                if not self.needed(stage, outputs, STAGES[position - 1] if position else None, *inputs):
                    log.info("[%s] %s: done earlier", self.media.stem, stage)
                    continue
                if stage == "mix" and background is not None:
                    # the lane goes on with its next engine's dub / voiceover meanwhile; it joins this thread at its end
                    background.start(f"mix-{self.engine}", functools.partial(self._execute, stage, func))
                else:
                    self._execute(stage, func)
            if self.forced_from is None or position < self.forced_from:
                self.forced_from = position + 1  # later stages depend on fresh output

    def _execute(self, stage: str, func: Callable[[], None]) -> None:
        """Run one stage that is needed: progress file, finish mark, timing log."""
        engine = self.engine if stage in ENGINE_STAGES else None
        log.info("[%s] %s ...", self.media.stem, f"{stage} ({engine})" if engine else stage)
        started = time.monotonic()
        with progress.track(self.out, stage, engine):  # .progress/<stage>[.<engine>].json for the UI
            func()
            progress.clear_partial(self.out, stage, engine)  # paid answers kept for a restart are not needed now
            self.stamp(stage).parent.mkdir(parents=True, exist_ok=True)
            self.stamp(stage).touch()
        seconds = round(time.monotonic() - started)
        if engine:
            log.info("[%s] %s (%s): finished in %d s", self.media.stem, stage, engine, seconds)
        else:
            log.info("[%s] %s: finished in %d s", self.media.stem, stage, seconds)

    # ---- stages
    def transcribe(self) -> None:
        from stt import transcribe

        from stt.diarize import DIARIZERS

        mode = self.config["modes"][self.args.mode]
        diarizer = self.config["diarizer"]
        env = DIARIZERS[diarizer].env if diarizer in DIARIZERS else "subs_env"
        if not (REPO_ROOT / env).exists():  # e.g. DiariZen without its venv: the built-in diarizer instead
            log.warning("[%s] %s needs %s (bash scripts/setup_envs.sh diarizen); using pyannote-community-1",
                        self.media.stem, diarizer, env)
            diarizer = "pyannote-community-1"
        argv = [str(self.media), "--out", str(self.out), "--asr", mode["asr"], "--diarizer", diarizer]
        if mode.get("fuse"):
            argv += ["--fuse", ",".join(mode["fuse"]), "--fuse-model", self.llm("fuse")]
            if (self.season.with_suffix(".txt")).exists():
                argv += ["--glossary", str(self.season.with_suffix(".txt"))]
        diar_options = self.config.get("diarizer_options") or {}
        if not diar_options.get("median_filter", True):
            argv.append("--diar-no-median-filter")
        if diar_options.get("exclusive", False):
            argv.append("--diar-exclusive")
        if diar_options.get("ahc_threshold", 0.5) != 0.5:
            argv += ["--diar-ahc-threshold", str(diar_options["ahc_threshold"])]
        ocr = self.config.get("ocr", {})
        argv += ["--ocr-fps", str(ocr.get("fps", 2.0))] if ocr.get("enabled", True) else ["--no-ocr"]
        refine = self.config.get("speaker_refine") or {}
        if refine.get("model", "none") != "none":
            argv += ["--refine-speakers", refine["model"], "--refine-margin", str(refine.get("margin", 0.1))]
        word_speakers = self.config.get("word_speakers") or {}
        if word_speakers.get("enabled", False):
            argv += ["--word-speakers", "viterbi", "--word-speakers-model", word_speakers.get("model", "resnet293"),
                     "--word-switch-penalty", str(word_speakers.get("switch_penalty", 0.3)),
                     "--word-prior", str(word_speakers.get("prior", 0.0)),
                     "--word-change-prior", str(word_speakers.get("change_prior", 0.15)),
                     "--word-window", str(word_speakers.get("window_s", 1.0)),
                     "--word-passes", str(word_speakers.get("passes", 1))]
        subtitles = self.subtitle_source()
        if subtitles:
            argv += ["--subtitles", str(self.out / "subs.source.en.srt"), "--subtitle-turn-bonus",
                     str((self.config.get("subtitles") or {}).get("turn_bonus", 0.3))]
        if self.args.start:
            argv += ["--start", str(self.args.start)]
        if self.args.duration:
            argv += ["--duration", str(self.args.duration)]
        transcribe.main(argv)
        self.subtitle_stamp.parent.mkdir(parents=True, exist_ok=True)
        self.subtitle_stamp.write_text(self.subtitle_key() + "\n", encoding="utf-8")

    @property
    def season_voices(self) -> Path:
        """The season's voices (stt.season_voices), beside the voice bank."""
        return self.voice_bank.with_name(self.voice_bank.name.replace(".voices.json", ".season_voices.npz"))

    @property
    def cast(self) -> Path:
        """The season's cast from the web (stt.cast)."""
        return self.voice_bank.with_name(self.voice_bank.name.replace(".voices.json", ".cast.json"))

    def _find_cast(self) -> bool:
        """Look the season's cast up once (web search or the Wikipedia article); a failure only costs the evidence."""
        cfg = self.config.get("cast") or {}
        if not cfg.get("enabled", True):
            return False
        if not self.cast.exists():
            from stt import cast

            models = (self.config.get("llm") or {}).get("stages", {}).get("cast") or cast.DEFAULT_MODELS
            models = [models] if isinstance(models, str) else models
            try:
                cast.main([str(self.out), "--cast", str(self.cast), "--model", ",".join(self.route(m) for m in models)])
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
        gender = self.config.get("line_gender", True)  # a bool or {"enabled", "override", "voice_override"}
        gender = gender if isinstance(gender, dict) else {"enabled": bool(gender)}
        if gender.get("enabled", True):  # unknown speakers: their lines' own voices; known: sure-of-other-gender lines
            argv = [str(self.out), "--skip-speakers", ",".join(self._host_speakers())]
            argv += ["--voice-override", str(gender.get("voice_override", 0.9) if gender.get("override", True) else 0)]
            self._module("line_gender", "stt.line_gender", argv)  # GPU, own process

    def translate(self) -> None:
        from stt import translate

        cfg = self.config["translate"]
        argv = [str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model", self.llm("translate"),
                "--cps", str(cfg["cps"]), "--extend", str(cfg["extend"]), "--parallel", str(cfg["parallel"])]
        if cfg.get("tts_profile"):
            argv += ["--tts-profile", str(REPO_ROOT / cfg["tts_profile"])]
        if cfg.get("proofread", True):  # a second reader looks for misspelt words and broken grammar
            argv += ["--proofread-model", self.llm("translate_proofread")]
        translate.main(argv)
        self.start_onscreen()  # the dub does not need it: only the finished video does

    def on_screen(self) -> dict[str, Any]:
        """``on_screen`` settings: the voice of captioned lines and the Russian on-screen text.
        Voice ``text`` (captions only shown as Russian text) needs the track: without it the voice becomes
        ``neutral`` so the captioned lines are not lost."""
        settings = {"voice": "text", "neutral_gender": "male", "track": True, "burn_in_when_compressing": False,
                    **(self.config.get("on_screen") or {})}
        if settings["voice"] == "text" and not settings["track"]:
            log.warning("[%s] on_screen.voice is 'text' but on_screen.track is off: captioned lines get the "
                        "neutral voice instead", self.media.stem)
            settings["voice"] = "neutral"
        return settings

    def start_onscreen(self) -> None:
        """Translate the on-screen text in a background thread; ``translate_onscreen`` (the video) waits for it."""
        job: Future[Path | None] = Future()
        with _ONSCREEN_GUARD:
            _ONSCREEN_JOBS[self.out] = job

        def work() -> None:
            try:
                job.set_result(self._translate_onscreen())
            except BaseException as error:  # noqa: BLE001 - a failed translation only costs the track
                log.warning("[%s] on-screen text not translated (%s): no Russian on-screen track", self.media.stem,
                            error)
                job.set_result(None)

        threading.Thread(target=work, name=f"onscreen-{self.media.stem}").start()

    def translate_onscreen(self) -> Path | None:
        """``onscreen.ru.json``: on-screen captions, name cards and lower thirds in Russian (cached per text).
        None without OCR results or with the on-screen track off; a failed translation only costs the track.
        Waits for the background translation of this episode (``start_onscreen``) instead of asking again."""
        with _ONSCREEN_GUARD:
            job = _ONSCREEN_JOBS.get(self.out)
        if job is not None:
            return job.result()
        return self._translate_onscreen()

    def _translate_onscreen(self) -> Path | None:
        source, target = self.out / "onscreen.json", self.out / "onscreen.ru.json"
        if not source.exists() or not self.on_screen()["track"]:
            return None
        inputs = [p for p in (source, self.out / "subs.en.json", self.out / "speakers.json", self.out / "subs.ru.json")
                  if p.exists()]  # speakers: genders; subs.ru: the dub's wording the captions follow
        if target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):  # speakers: genders
            return target
        from stt import onscreen

        started = time.monotonic()
        try:
            onscreen.main(["translate", str(self.out), "--glossary", str(self.out / "glossary.txt"),
                           "--model", self.llm("translate")])
        except Exception as error:  # noqa: BLE001 - the video is still made, without the on-screen track
            log.warning("[%s] on-screen text not translated (%s): no Russian on-screen track", self.media.stem, error)
            return None
        log.info("[%s] on-screen text translated in %d s", self.media.stem, round(time.monotonic() - started))
        return target if target.exists() else None

    def _engine(self) -> dict[str, Any]:
        cfg = self.config["voiceover"]
        engine = dict(cfg.get("engines", {}).get(self.engine, {}))
        if "voices" not in engine:
            raise SystemExit(f"voiceover.engines.{self.engine}.voices missing in the config")
        engine.setdefault("listen_stress", bool(cfg.get("listen_stress", True)))  # takes heard for checked stress
        host = cfg.get("host_voice")
        if host and self.engine == "vosk":  # the host's lines in a TeraTTS voice inside the Vosk track
            tera = {k: v for k, v in cfg.get("engines", {}).get("teratts", {}).items() if k not in ("voices", "sample_rate")}
            engine["teratts"] = {"listen_stress": engine["listen_stress"], **tera, "voice": host}
        if self.engine == "f5":  # speakers without a clip: TeraTTS voices, or a Vosk voice cloned by F5
            from voiceover.engine_f5 import TERATTS, fallback_references, voice_id

            vo = load_json(REPO_ROOT / cfg.get("config", "voiceover_config.json"))
            vosk = cfg.get("engines", {}).get("vosk", {}).get("voices", {"female": 49, "male": 56})
            refs = fallback_references(self.references.parent / "fallback_voices",
                                       REPO_ROOT / vo["engine"]["model_path"], {g: int(v) for g, v in vosk.items()})
            auto = {g: voice_id(refs[g]) for g in refs}
            if engine.pop("fallback", "vosk") == "teratts":
                tera = {k: v for k, v in cfg.get("engines", {}).get("teratts", {}).items() if k != "sample_rate"}
                voices = tera.pop("voices", {"female": "ru_f1+ru_f2", "male": "ru_m5"})
                engine["teratts"] = {"listen_stress": engine["listen_stress"], **tera, "voice": voices["male"]}
                engine["voice"] = auto["male"]  # the default voice F5 clones (calibration probes it)
                auto = {g: TERATTS + str(voices.get(g, voices["male"])) for g in engine["voices"]}
            engine["voices"] = {g: auto[g] if v == "auto" else voice_id(v) for g, v in engine["voices"].items()}
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

    def _host_speakers(self) -> list[str]:
        """Speaker ids of the show's host: people of the season cast with a "host" role (else the
        speaker cards' own role)."""
        from stt.voice_bank import same_person

        cards = load_json(self.out / "speakers.json")
        cast = load_json(self.cast).get("cast", []) if self.cast.exists() else []
        hosts = [p["name"] for p in cast if "host" in str(p.get("role", "")).lower()]
        if hosts:
            return [c["id"] for c in cards if c.get("name") and any(same_person(c["name"], h) for h in hosts)]
        return [c["id"] for c in cards if re.search(r"\bhost\b", str(c.get("role", "")).lower())]

    def _host_voices(self) -> Path | None:
        """{speaker id: voice} giving the host ``voiceover.host_voice`` (TeraTTS) in the Vosk / TeraTTS track."""
        host = self.config["voiceover"].get("host_voice")
        if not host or self.engine not in ("vosk", "teratts"):
            return None
        voice = f"teratts:{host}" if self.engine == "vosk" else str(host)
        mapping = {sid: voice for sid in self._host_speakers()}
        path = self.out / f"speaker_voices.{self.engine}.json"
        save_json(path, mapping)
        log.info("[%s] host voice %s for %s", self.media.stem, host, ", ".join(mapping) or "nobody (no host found)")
        return path

    def dub(self) -> None:
        from stt import dub

        voices = ",".join(f"{k}={v}" for k, v in self._engine()["voices"].items())
        argv = [str(self.out), "--voices", voices, "--interjections", self.config["dub"]["interjections"],
                "--output", f"dub.ru.{self.engine}.json"]
        if self.engine == "f5":
            argv += ["--speaker-voices", str(self._speaker_voices())]
        elif (host := self._host_voices()) is not None:
            argv += ["--speaker-voices", str(host)]
        on_screen = self.on_screen()
        argv += ["--on-screen-voice", str(on_screen["voice"]), "--neutral-gender", str(on_screen["neutral_gender"])]
        dub.main(argv)

    def _module(self, name: str, module: str, args: list[str], *, oom_retries: int = 2,
                oom_wait_s: float = 60.0) -> None:
        """Run a module of this repo in the current interpreter, logging to logs/<name>.log.

        A run that ran out of GPU memory is started again after a pause; the voice-over keeps
        every line already synthesised, so it goes on where it stopped.
        """
        from stt.worker import gpu_out_of_memory

        env = {**os.environ, **dotenv(), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        tracker = progress.current()
        if tracker is not None:  # the module's progress ticks go to this stage's progress file
            env.update(tracker.environ())
        log_path = self.out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(oom_retries + 1):
            with log_path.open("w", encoding="utf-8") as log_file:
                command = [sys.executable, "-m", module, *args]
                if tracker is None:  # outside a tracked stage there is no progress to watch
                    code = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log_file,
                                          stderr=subprocess.STDOUT).returncode
                else:
                    code = self._run_watched(command, env, log_file, tracker)
            if code == 0:
                return
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if attempt == oom_retries or not gpu_out_of_memory(log_text):
                break
            log.warning("[%s] %s ran out of GPU memory; retry %d/%d in %.0f s", self.media.stem, name, attempt + 1,
                        oom_retries, oom_wait_s)
            log_path.with_suffix(f".oom{attempt + 1}.log").write_text(log_text, encoding="utf-8")
            time.sleep(oom_wait_s)
        raise SystemExit(f"{name} failed (exit {code}), see {log_path}")

    def _run_watched(self, command: list[str], env: dict[str, str], log_file: Any,
                     tracker: progress.Tracker) -> int:
        """Run ``command`` (output to ``log_file``) and return its exit code; while it runs, copy its progress
        (``[<episode>] voiceover (f5): 25/640 lines``) into this log every ``PROGRESS_LOG_LINES`` lines."""
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        last = 0
        try:
            while True:
                try:
                    return process.wait(timeout=PROGRESS_POLL_S)
                except subprocess.TimeoutExpired:
                    last = self._log_progress(tracker, last)
        except BaseException:  # Ctrl+C or an error here: do not leave the module running on its own
            process.terminate()
            raise

    def _log_progress(self, tracker: progress.Tracker | None, last: int) -> int:
        """Log the stage's progress if it moved far enough since ``last`` was logged; returns the new ``last``."""
        if tracker is None:
            return last
        try:
            record = json.loads(tracker.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return last
        done, total, unit = record.get("done"), record.get("total"), record.get("unit")
        if done is None or not total or not unit:
            return last
        if done < last or done - last >= (PROGRESS_LOG_LINES if unit in ("lines", "clips") else 1):
            if done >= last:
                label = f"{tracker.stage} ({tracker.engine})" if tracker.engine else tracker.stage
                log.info("[%s] %s: %d/%d %s", self.media.stem, label, done, total, unit)
            return done
        return last

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
                               "voice": engine.get("voice") or self._engine()["voices"]["male"]}
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

    def _stress_input_unchanged(self) -> bool:
        """The stress check was made on exactly the lines of this engine's dubbing script."""
        from voiceover.stress_check import lines_sha256

        dub, result = self.out / f"dub.ru.{self.engine}.json", self.out / "stress.json"
        try:
            recorded = load_json(result).get("lines_sha256")
            return bool(recorded) and dub.exists() and recorded == lines_sha256(load_json(dub))
        except (OSError, ValueError):
            return False

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
        if (self.out / "glossary.json").exists():  # the show's names, checked in every case form
            args += ["--names", str((self.out / "glossary.json").resolve())]
        if cfg.get("season_stress", True):  # one stress per name for the whole season
            args += ["--season", str(self.season_stress.resolve())]
        if cfg.get("stress_proofread", False):  # sol reads every line; its proposals go to the vote
            proofreader = self.llm("stress_proofread")
            args += ["--proofread", proofreader[0] if isinstance(proofreader, list) else proofreader,
                     "--proofread-passes", str(int(cfg.get("stress_proofread_passes", 2)))]
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
        with self.lock or nullcontext():  # the other lane may be writing it
            if self.config.get("review", True) and (
                    not review.exists() or review.stat().st_mtime < (self.out / "subs.ru.json").stat().st_mtime):
                try:  # lines to check by ear: a help for a person, never a reason to stop the episode
                    self._module("review", "stt.review", [str(self.out.resolve())])
                except SystemExit as error:
                    log.warning("[%s] %s", self.media.stem, error)

    def track(self, engine: str) -> Path:
        """The engine's finished voice-over mix (AAC), kept for the videos."""
        return self.out / f"{self.media.stem}.ru.{engine}.m4a"

    @property
    def finished(self) -> Path:
        """Folder of the finished video: ``paths.finished`` / ``paths.finished_folder`` (``{season}``:
        the video's folder name), else the episode folder (always for a test piece)."""
        folder = finished_folder(self.config, self.media)
        return self.out if folder is None or self.args.start or self.args.duration else folder

    def finished_video(self, engine: str | None = None) -> Path:
        """The video with all voice-overs (``mix.tracks: combined``) or with ``engine``'s (separate)."""
        combined = (self.config.get("mix") or {}).get("tracks", "combined") == "combined"
        return self.finished / f"{self.media.stem}.ru{'' if combined else f'.{engine or self.engine}'}.mkv"

    def mux(self) -> None:
        """The finished video: the voice-overs finished so far (combined) or this engine's (separate)."""
        from stt import mix

        cfg = self.config["mix"]
        if cfg.get("tracks", "combined") == "combined":
            order = [*cfg.get("track_order", []), *self.config["voiceover"].get("engines", {}), self.engine]
            engines = [e for e in dict.fromkeys(order) if self.track(e).exists()]
        else:
            engines = [self.engine]
        titles = cfg.get("track_titles") or {}
        tracks = [(self.track(e), titles.get(e, f"Русский ({e})")) for e in engines]
        srt = None
        if cfg.get("subtitles", True):  # the voiced lines (the same in every engine); the translation if none
            voiced = self.out / f"voiceover.{engines[0]}" / "export" / "display.srt"
            srt = voiced if voiced.exists() else self.out / "subs.ru.srt"
        out = self.finished_video()
        out.parent.mkdir(parents=True, exist_ok=True)
        video_cfg = cfg.get("video") or {}
        compress = bool(video_cfg.get("compress")) and not (self.args.start or self.args.duration)
        video_source = self.small_video(video_cfg, len(self.planned_tracks(engines))) if compress else None
        onscreen_ass, fonts = None, []
        if not (video_source and self.burn_onscreen()):  # burned into the compressed picture: no second copy
            onscreen_ass, fonts = self.onscreen_track()
        mix.mux_tracks(self.media, tracks, out, srt, self.args.start or 0.0, self.args.duration,
                       video_source=video_source,
                       original_audio_kbps=int(video_cfg.get("original_audio_kbps", 160)) if video_source else None,
                       onscreen_ass=onscreen_ass, fonts=fonts)
        log.info("[%s] %s: %s", self.media.stem, out, ", ".join(engines))
        if video_source:
            log.info("[%s] finished video %.2f GB", self.media.stem, out.stat().st_size / 1e9)

    def burn_onscreen(self) -> bool:
        return bool(self.on_screen()["burn_in_when_compressing"]) and self.translate_onscreen() is not None

    def onscreen_track(self) -> tuple[Path | None, list[Path]]:
        """The Russian on-screen text as an ASS track (default, forced) and the fonts to attach; (None, []) if none."""
        items = self.translate_onscreen()
        if items is None:
            return None, []
        from stt import onscreen_render, video_compress

        info = video_compress.probe(self.media)
        ass = onscreen_render.write_ass(onscreen_render.load_items(items), self.out / "onscreen.ass",
                                        int(info["width"]), int(info["height"]), offset_s=self.args.start or 0.0)
        return ass, list(onscreen_render.font_files() or []) if ass else []

    def planned_tracks(self, engines: list[str]) -> list[str]:
        """Voice-over tracks the finished video will have: those in it now and the engines of this run."""
        if (self.config["mix"].get("tracks", "combined") != "combined"):
            return engines
        return list(dict.fromkeys([*engines, *getattr(self.args, "planned_engines", [])]))

    def small_video(self, video_cfg: dict[str, Any], tracks: int) -> Path | None:
        """The episode's video with a smaller picture for ``mix.video.target_gb`` (made once, shared by the engines'
        videos); None when the picture is already small enough or the encoding failed (the video is copied then)."""
        from stt import video_compress

        codec = str(video_cfg.get("codec", "hevc_nvenc"))
        original_kbps = int(video_cfg.get("original_audio_kbps", 160))
        info = video_compress.probe(self.media)
        bitrate = video_compress.video_bitrate(info["duration"], float(video_cfg.get("target_gb", 2.0)),
                                               tracks * 192 + original_kbps, info["video_kbps"])
        if bitrate is None:
            log.info("[%s] the video is already close to the target size: copied as it is", self.media.stem)
            return None
        overlays = None
        if self.burn_onscreen():
            from stt import onscreen_render

            overlays = onscreen_render.render_overlays(onscreen_render.load_items(self.out / "onscreen.ru.json"),
                                                       int(info["width"]), int(info["height"]))
        # the burned-in text is part of the picture: a changed translation needs a new video
        burned = (f".onscreen-{hashlib.sha256((self.out / 'onscreen.ru.json').read_bytes()).hexdigest()[:8]}"
                  if overlays else "")
        small = self.out / f"video.{codec}.{bitrate}k{burned}.mkv"
        if not small.exists():
            try:
                video_compress.encode_video(self.media, small, bitrate, codec, overlays=overlays)
            except Exception as error:  # noqa: BLE001 - a big video is better than no video
                log.warning("[%s] compressing the video failed (%s): the video is copied as it is", self.media.stem, error)
                return None
        return small

    def mix(self) -> None:
        from stt import mix

        cfg = self.config["mix"]
        out_wav = self.out / f"{self.media.stem}.ru.{self.engine}.wav"
        argv = [str(self.media), str(self.vo_dir / "export" / "track.wav"), "--out", str(out_wav),
                "--duck-db", str(cfg["duck_db"])]
        if not self.args.start and not self.args.duration:
            argv.append("--full-duration")
        if cfg.get("duck") == "voices":  # lower only the original voices, keep music and effects
            argv += ["--duck-voices-only", "--rest-duck-db", str(cfg.get("rest_duck_db", 6.0)),
                     "--separation-cache", str(self.out / "separation")]  # one separation for all engines
        argv += ["--voice-gain-db", str(cfg.get("voice_gain_db", 3.0))]
        if cfg.get("loudness_lufs") is not None:  # the whole mix to this level, peaks limited (null: leave as mixed)
            argv += ["--loudness-lufs", str(cfg["loudness_lufs"])]
        if self.args.start:  # a test piece: the voice-over is on the episode timeline from 0
            argv += ["--start", str(self.args.start), "--voice-from-zero"]
        crowd = self.config.get("crowd", {})
        if crowd.get("enabled"):  # experimental: cheers/chants by several voices, under the voice-over
            from stt import crowd as crowd_layer

            crowd_layer.main([str(self.out), "--engine", self.engine, "--voices", str(crowd.get("voices", 3))])
            argv += ["--extra", f"{self.out / f'crowd.{self.engine}.wav'}:{crowd.get('gain_db', -4.0)}"]
        mix.main(argv)
        started = time.monotonic()
        mix.encode_track(out_wav, self.track(self.engine))
        log.info("[%s] mix (%s): AAC encode %.0f s", self.media.stem, self.engine, time.monotonic() - started)
        if not cfg.get("keep_wav"):  # ~0.5 GB per episode and engine; the AAC track is what the video gets
            out_wav.unlink()
        if cfg.get("mux"):  # a test piece gets a short re-encoded video of the piece
            self.mux_when_ready()

    def mux_when_ready(self) -> None:
        """Mux the finished video. ``combined``: once, by the engine whose track is the last of this run's engines
        (the others only log that the mux waits); ``separate``: this engine's own video."""
        combined = (self.config.get("mix") or {}).get("tracks", "combined") == "combined"
        with self.lock or nullcontext():  # one video for all engines: one writer at a time
            signature: tuple[tuple[str, int], ...] = ()
            if combined:
                planned = self.planned_tracks([self.engine])
                waiting = [e for e in planned if not self.track(e).exists()]
                if waiting:
                    log.info("[%s] mix (%s): mux deferred until %s finish", self.media.stem, self.engine,
                             ", ".join(waiting))
                    return
                signature = tuple(sorted((e, self.track(e).stat().st_mtime_ns) for e in planned))
                if _MUXED.get(self.out) == signature and self.finished_video().exists():
                    log.info("[%s] mix (%s): the video with these tracks is already made", self.media.stem, self.engine)
                    return
            started = time.monotonic()
            self.mux()
            log.info("[%s] mix (%s): mux %.0f s", self.media.stem, self.engine, time.monotonic() - started)
            if combined:
                _MUXED[self.out] = signature
            self._clear_separation()

    def _clear_separation(self) -> None:
        """Delete the cached voice separation (~1.5 GB per hour) once the video is made and every engine of this run
        has its track; before that (or after a failed mux, which never gets here) a rerun reuses it.
        Done whatever ``mix.keep_wav`` says: that setting is about the mixed WAVs."""
        cache = self.out / "separation"
        engines = dict.fromkeys([self.engine, *getattr(self.args, "planned_engines", [])])
        waiting = [e for e in engines if not self.track(e).exists()]
        if waiting or not cache.exists():
            return
        shutil.rmtree(cache, ignore_errors=True)
        log.info("[%s] mix (%s): voice separation cache removed (%s)", self.media.stem, self.engine, cache)


PROGRESS_POLL_S = 2.0  # how often a running module's progress file is looked at
PROGRESS_LOG_LINES = 25  # progress of lines / clips goes to the log every this many
SHARED_STAGES = {"transcribe", "glossary", "speakers", "translate", "stress"}  # one result for all engines
ENGINE_STAGES = {"dub", "voiceover", "check", "mix"}
CPU_ENGINES = {"vosk"}  # synthesise on the CPU: can run next to the GPU engines


def lanes(engines: list[str], parallel: bool) -> list[list[str]]:
    """Engines in the order they run: one lane, or the GPU engines and the CPU ones side by side."""
    gpu = [e for e in engines if e not in CPU_ENGINES]
    cpu = [e for e in engines if e in CPU_ENGINES]
    return [gpu, cpu] if parallel and gpu and cpu else [engines]


def run_lanes(files: list[Path], out_root: Path, args: argparse.Namespace, config: dict[str, Any],
              engines: list[str]) -> None:
    """Every episode in every engine; the second lane (CPU engines) starts an episode once the first
    has its shared stages, so the GPU-heavy recognition never runs twice at once."""
    groups = lanes(engines, bool((config.get("voiceover") or {}).get("parallel_cpu", True)))
    locks = {media: threading.Lock() for media in files}
    shared = {media: threading.Event() for media in files}
    errors: list[BaseException] = []

    def episode(media: Path, engine: str, until: str | None = None) -> Episode:
        lane_args = argparse.Namespace(**{**vars(args), "engine": engine, "until": until or args.until})
        return Episode(media, out_root, lane_args, config, lock=locks[media])

    def lane(group: list[str], first: bool) -> None:
        for media in files:
            if errors:
                return
            background = Background()  # this lane's mixes running while its next engine dubs and voices
            try:
                if first and len(groups) > 1:
                    try:
                        episode(media, group[0], STAGES[min(STAGES.index(args.until), STAGES.index("translate"))]).run()
                    finally:
                        shared[media].set()
                elif not first:
                    shared[media].wait()
                for engine in group:
                    if background.failed():
                        break
                    ep = episode(media, engine)
                    ep.args.mix_background = background
                    ep.run()
            except BaseException as error:  # noqa: BLE001 - stop both lanes, report after they end
                errors.append(error)
                for event in shared.values():
                    event.set()
            finally:
                try:
                    background.join()
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    for event in shared.values():
                        event.set()
            if errors:
                return

    if len(groups) == 1:
        lane(groups[0], True)
    else:
        from voiceover.engine import CPU_THREADS_ENV

        # Vosk on all cores starved the GPU engines' own CPU work (10-min piece: ESpeech 7.5 -> 12.3 min)
        os.environ.setdefault(CPU_THREADS_ENV, str(max(2, (os.cpu_count() or 4) // 2)))
        log.info("engines in two lanes: %s on the GPU, %s on the CPU alongside (%s CPU threads)", ", ".join(groups[0]),
                 ", ".join(groups[1]), os.environ[CPU_THREADS_ENV])
        threads = [threading.Thread(target=lane, args=(group, n == 0), name=f"lane-{n}") for n, group in
                   enumerate(groups)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    wait_onscreen()  # a run that ends at the translation still gets onscreen.ru.json
    if errors:
        raise errors[0]


VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm", ".ts"}


def season_voices_enabled(config: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool((config.get("season_voices") or {}).get("enabled", False)) and not getattr(args, "no_season_voices", False)


def season_voices_model(config: dict[str, Any]) -> str:
    """The word embedding model of the season's voices; the plain one when the trained weights are not on this computer."""
    from stt.refine_speakers import model_available

    plain = str((config.get("word_speakers") or {}).get("model", "resnet293"))
    model = str((config.get("season_voices") or {}).get("model") or plain)
    if model_available(model):
        return model
    log.warning("season voices: the weights of %s are not here, using %s (slightly less accurate; docs/stt_local.md)", model, plain)
    return plain


def fetch_voice_model(config: dict[str, Any]) -> None:
    """Download the trained weights of the season's voices once. A failed download only means the plain model is used."""
    from stt.refine_speakers import HUB_WEIGHTS, download_model, model_available

    model = str((config.get("season_voices") or {}).get("model") or "")
    if model not in HUB_WEIGHTS or model_available(model):
        return
    log.info("season voices: downloading the weights of %s ...", model)
    try:
        log.info("season voices: weights saved: %s", download_model(model))
    except Exception as error:  # noqa: BLE001 - the run goes on with the plain model
        log.warning("season voices: could not download %s: %s", model, error)


def run_season_voices(files: list[Path], out_root: Path, args: argparse.Namespace, config: dict[str, Any],
                      engines: list[str], first_pass: bool = True) -> None:
    """Every episode up to ``speakers`` first, then the words of each are assigned again over the people named in its whole
    season (the videos in the same folder that have a transcript: ``stt.season_voices``). A changed transcript gets a new
    ``speakers`` mark, so the stages after it run again and the earlier ones do not. With ``first_pass=False`` no episode
    is processed: only those that already have a transcript get their voices counted again."""
    from stt import season_voices

    if STAGES.index(args.until) < STAGES.index("speakers"):
        return
    forced = args.force in STAGES[: STAGES.index("speakers") + 1]
    first = argparse.Namespace(**{**vars(args), "engine": engines[0], "until": "speakers"})
    episodes = {media: Episode(media, out_root, first, config) for media in files}
    if first_pass:
        for episode in episodes.values():
            episode.run()
    else:
        files = [media for media in files if (episodes[media].out / "speakers.json").exists()]
        if not files:
            log.warning("season voices: no episode here has its transcript and names yet, nothing to count")
    if forced:
        args.force = None  # done above; the marks make the later stages follow
    cfg = config.get("season_voices") or {}
    model = season_voices_model(config)
    seasons: dict[Path, list[Path]] = {}
    for media in files:
        seasons.setdefault(media.parent, []).append(media)
    for folder, medias in seasons.items():
        every = [out_root / p.stem for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES]
        every = [f for f in every if (f / "speakers.json").exists()]
        # A translated episode stays: a new transcript would mean a new translation and voice-over, LLM costs included
        fresh = [m for m in medias if forced or not (episodes[m].out / "subs.ru.json").exists()]
        for media in medias:
            if media not in fresh:
                log.info("[%s] season voices: already translated, stays as it is (--force speakers assigns it again)", media.stem)
        if not fresh:
            continue
        targets = [episodes[m].out for m in fresh]
        results = season_voices.apply_season(every, targets, episodes[fresh[0]].season_voices, model=model,
                                             min_words=int(cfg.get("min_words", season_voices.MIN_WORDS)))
        for media in fresh:
            episode = episodes[media]
            result = results[episode.out.name]
            log.info("[%s] season voices: %s", media.stem, season_voices.summary(episode.out.name, result))
            if result.changed:
                episode.stamp("speakers").parent.mkdir(parents=True, exist_ok=True)
                episode.stamp("speakers").touch()


SUBSCRIPTION_DOWN = "The ChatGPT subscription login of OpenCode does not work (run `opencode auth login`)"


def subscription_models(config: dict[str, Any]) -> list[str]:
    """Model ids of the LLM steps that go through the OpenCode subscription (see ``Episode.llm``)."""
    llm = config.get("llm") or {}
    patterns = llm.get("opencode") or []
    names: list[str] = []
    for value in [llm.get("model"), *(llm.get("stages") or {}).values()]:
        names += value if isinstance(value, list) else [value] if value else []
    models = []
    for name in names:
        transports, bare = split_route(name)
        if "opencode" in transports or (not transports and any(fnmatch.fnmatchcase(bare, p) for p in patterns)):
            models.append(bare)
    return list(dict.fromkeys(models))


def check_subscription(config: dict[str, Any]) -> bool:
    """Ask the subscription once before the episodes; False (with a warning) when it does not answer.

    With ``llm.fallback`` false the run stops here instead of failing at its first LLM step.
    """
    from stt.llm import subscription_error

    for model in subscription_models(config):
        error = subscription_error(model)
        if error is None:
            continue
        if not (config.get("llm") or {}).get("fallback", True):
            raise SystemExit(f"{SUBSCRIPTION_DOWN}: {error}")
        router = (config.get("llm") or {}).get("router", "openrouter")
        log.warning("%s: %s. LLM steps with %s go through %s instead and cost money.", SUBSCRIPTION_DOWN, error,
                    model, " / ".join([router] if isinstance(router, str) else router))
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="English video -> Russian voice-over, end to end")
    parser.add_argument("media", nargs="+", help="Video files or glob patterns (a whole season works)")
    parser.add_argument("--out-root", default=None,
                        help="One sub-folder per episode (default: paths.work in the config, else output/stt)")
    parser.add_argument("--config", default=str(REPO_ROOT / "stt_config.json"))
    parser.add_argument("--preset", default=None, help="A named set of settings (presets in the config): main, upload, fast, previous")
    parser.add_argument("--mode", choices=["fast", "accurate"], default=None, help="Default from the config")
    parser.add_argument("--until", choices=STAGES, default="mix", help="Last stage to run")
    parser.add_argument("--force", choices=STAGES, default=None, help="Recompute this stage and the ones after it")
    parser.add_argument("--season-glossary", default=None, help="Default: <out-root>/<folder>.glossary.json")
    parser.add_argument("--engine", default=None,
                        help="TTS engine(s) (voiceover.engines in the config): vosk | teratts | f5, or several: f5,teratts,vosk")
    parser.add_argument("--start", type=float, default=None, help="Process from this time (tests), seconds")
    parser.add_argument("--duration", type=float, default=None, help="Process this many seconds (tests)")
    parser.add_argument("--no-subtitles", action="store_true",
                        help="Do not look for existing subtitles that name speakers (config: subtitles)")
    parser.add_argument("--no-season-voices", action="store_true",
                        help="Do not assign the words again over the season's named people (config: season_voices)")
    parser.add_argument("--season-voices-only", action="store_true",
                        help="Only count the season's voices again from the episodes that have a transcript; process nothing else")
    parser.add_argument("--download-voice-model", action="store_true",
                        help="Download the trained weights of the season's voices first, when they are not here (115 MB, once)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    config = apply_preset(load_config(Path(args.config)), args.preset)
    args.mode = args.mode or config["mode"]
    files: list[Path] = []
    for pattern in args.media:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        files += [Path(m) for m in matches]
    missing = [f for f in files if not f.is_file()]
    if missing or not files:
        raise SystemExit(f"Media not found: {missing or args.media}")
    out_root = Path(args.out_root or (config.get("paths") or {}).get("work") or "output/stt").expanduser()
    if args.download_voice_model and (args.season_voices_only or season_voices_enabled(config, args)):
        fetch_voice_model(config)
    if not args.season_voices_only:
        check_subscription(config)
    engines = [e.strip() for e in (args.engine or (config.get("voiceover") or {}).get("engine", "vosk")).split(",")
               if e.strip()]
    args.planned_engines = engines  # the finished video's size budget counts these voice-over tracks
    if args.season_voices_only:
        run_season_voices(files, out_root, args, config, engines, first_pass=False)
        return
    if season_voices_enabled(config, args):
        run_season_voices(files, out_root, args, config, engines)
    run_lanes(files, out_root, args, config, engines)


if __name__ == "__main__":
    main()
