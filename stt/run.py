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
import functools
import glob
import hashlib
import json
import logging
import os
import re
import subprocess  # noqa: F401 - tests patch stt.run.subprocess.run (the stage programs)
import sys
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from stt import gpu_budget, progress, run_llm, stage_runner
from stt.config import (apply_preset, apply_show, format_config_sources, load_config, read_config_layers,
                        resolve_config_layers, show_of)
from stt.episode_media import _ONSCREEN_GUARD, _ONSCREEN_JOBS, MediaMixin
from stt.episode_vo import VoiceoverMixin
from stt.run_llm import SUBSCRIPTION_DOWN, check_paid_steps, export_llm_env, subscription_models
from stt.runner import REPO_ROOT
from stt.stage_runner import ENGINE_STAGES, SHARED_STAGES, STAGES, Background, ProcessMixin
from utils.llm_route import COST_LOG_ENV, LLM_STOP_EXIT, PAID_MARK, LLMStop, SubscriptionDown, paid_line

# Names that lived here before the split (other code and tests reach them through ``stt.run``; patching
# ``stt.run.REPO_ROOT`` / ``load_json`` / ``dotenv`` still reaches the moved code, see ``stage_runner.run_module``).
from stt.config import deep_merge, effective_voiceover_policy  # noqa: F401,E402
from stt.episode_media import (_COMPRESS_GATE, _COMPRESS_GUARD, _COMPRESS_JOBS, _MIX_GATE, _MUXED,  # noqa: F401,E402
                               finished_folder)
from stt.episode_vo import _sha256  # noqa: F401,E402
from stt.gpu import VramPeak  # noqa: F401,E402
from stt.run_llm import llm_steps, paying_allowed  # noqa: F401,E402
from stt.runner import load_json, save_json  # noqa: F401,E402
from stt.stage_runner import (CPU_ENGINES, PROGRESS_LOG_LINES, PROGRESS_POLL_S, lanes,  # noqa: F401,E402
                              lookahead_enabled)
from utils.llm_route import (FALLBACK_AFTER_ENV, FALLBACK_AFTER_S, FALLBACK_ENV, PAID_ENV, ROUTERS,  # noqa: F401,E402
                             PaidNotAllowed, split_route, stop_message_in, with_route)

log = logging.getLogger("stt.run")


def dotenv(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """KEY=VALUE lines of the repo .env (parsed as data, never executed)."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2).strip().strip("'\"")
    return values


def wait_onscreen() -> None:
    """Wait for every on-screen translation started in the background (their errors are logged, never raised;
    an ``LLMStop`` is raised after all of them ended: the run stops)."""
    with _ONSCREEN_GUARD:
        jobs = list(_ONSCREEN_JOBS.values())
    errors: list[BaseException] = []
    for job in jobs:  # every job is joined before anything is raised
        try:
            job.result()
        except BaseException as error:  # noqa: BLE001 - collected, the first stop (else the first error) is raised
            errors.append(error)
    if errors:
        raise next((e for e in errors if isinstance(e, LLMStop)), errors[0])


class Episode(VoiceoverMixin, MediaMixin, ProcessMixin):
    """One video in one engine: the stages, their order and when one has to run again. The English side (transcript,
    glossary, names, translation) is here; the voice-over (``stt.episode_vo``), the on-screen text, mix and video
    (``stt.episode_media``) and the stage programs (``stt.stage_runner``) come from the mixins."""

    def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any],
                 lock: threading.Lock | None = None) -> None:
        self.media = media
        self.lock = lock  # shared with the other lane's Episode of the same video (parallel engines)
        self.out = out_root / media.stem
        self.args = args
        self.show = show_of(config, media, getattr(args, "show", None))
        self.config = config = apply_show(config, self.show)
        if self.show:
            log.info("[%s] show profile: %s", media.stem, (config["shows"][self.show].get("label") or self.show))
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
        self._kept_subtitles_logged = False

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
        ``llm.router`` ("openrouter", or a list tried in order) as a per-request fallback only when
        ``llm.fallback`` is true (default false: the subscription is retried, never silently replaced by a paid
        router); the rest go to ``llm.router``. Ids that already carry a route
        keep it.
        """
        llm = self.config.get("llm") or {"model": self.config.get("llm_model", "openai/gpt-6-luna")}
        stages = llm.get("stages") or {}
        model = stages[stage] if stages.get(stage) is not None else llm["model"]
        return [self.route(m) for m in model] if isinstance(model, list) else self.route(model)


    def route(self, name: str) -> str:
        """A model id with its route (``llm``): see ``llm()``; web search (``:online``) is OpenRouter's."""
        return self.route_in(self.config, name)

    @classmethod
    def route_in(cls, config: dict[str, Any], name: str) -> str:
        """``route`` for a config (also used before any episode exists, by the startup check of paid steps)."""
        return run_llm.route_in(config, name)


    def stamp(self, stage: str) -> Path:
        """Marks when a stage last finished (per engine for the engine's stages)."""
        engine_stage = stage in ENGINE_STAGES
        return self.out / ".stages" / (f"{stage}.{self.engine}" if engine_stage else stage)

    def named_after_translation(self) -> bool:
        """The speakers stage finished after the translation (its finish marks): the translation is made again."""
        named, translated = self.stamp("speakers"), self.stamp("translate")
        return named.exists() and translated.exists() and named.stat().st_mtime > translated.stat().st_mtime

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
                if (self.out / "subs.ru.json").exists():
                    # A translated episode keeps the subtitles it was assembled with: assembling again would mean a new
                    # translation and a new voice-over. A forced run (--force transcribe) takes the new ones.
                    if not self._kept_subtitles_logged:
                        self._kept_subtitles_logged = True
                        log.info("[%s] subtitles changed (%s -> %s), but the episode is translated: it keeps the old ones "
                                 "(--force transcribe takes the new)", self.media.stem, used, self.subtitle_key())
                else:
                    log.info("[%s] subtitles changed (%s -> %s): speakers are assigned again", self.media.stem, used,
                             self.subtitle_key())
                    return True
        mark = self.stamp(stage)
        if mark.exists() and any(p.exists() and p.stat().st_mtime > mark.stat().st_mtime for p in inputs or []):
            return True
        if previous is None or not self.stamp(previous).exists():
            return False  # runs from before the marks existed
        return not mark.exists() or mark.stat().st_mtime < self.stamp(previous).stat().st_mtime

    def steps(self) -> dict[str, tuple[Any, ...]]:
        """Per stage: its function, its outputs and (optionally) the manual inputs that make it run again."""
        return {
            "transcribe": (self.transcribe, [self.out / "subs.en.json"]),
            "glossary": (self.glossary, [self.out / "glossary.json"]),
            "speakers": (self.speakers, [self.out / "speakers.llm.json"], [self.out / "speakers.override.json"]),
            "translate": (self.translate, [self.out / "subs.ru.json"]),
            "dub": (self.dub, [self.out / f"dub.ru.{self.engine}.json"],
                    [self.out / "corrections.json"]  # manual fixes; for F5 also new or replaced voice clips
                    + ([self.references / "references.json", self.references / "ru_samples.json"]
                       if self.engine == "f5" else [])),
            "stress": (self.stress, [self.out / "stress.json"]),
            "voiceover": (self.voiceover, [self.vo_dir / "export" / "track.wav"]),
            "check": (self.check, [self.vo_dir / "speech_check.json"]),
            "mix": (self.mix, [self.track(self.engine)]),
        }

    def gpu_stage_needed(self) -> bool:
        """Would a run of this episode start with a GPU stage (recognition, glossary, names)? Same rules as ``run``."""
        steps = self.steps()
        for position, stage in enumerate(STAGES[: STAGES.index("speakers") + 1]):
            _, outputs, *inputs = steps[stage]
            if self.needed(stage, outputs, STAGES[position - 1] if position else None, *inputs):
                return True
        return False

    def run(self) -> None:
        until = STAGES.index(self.args.until)
        steps = self.steps()
        background: Background | None = getattr(self.args, "mix_background", None)
        for position, stage in enumerate(STAGES[: until + 1]):
            func, outputs, *inputs = steps[stage]
            with self.lock if self.lock and stage in SHARED_STAGES else nullcontext():
                if not self.needed(stage, outputs, STAGES[position - 1] if position else None, *inputs):
                    log.info("[%s] %s: done earlier", self.media.stem, stage)
                    continue
                if stage == "dub":
                    self.start_compression()  # the first stage of the voicing: the picture is made beside it
                if stage == "mix" and background is not None and self.gpu_overlap("mix"):
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

    def gpu_overlap(self, name: str, *, uses_gpu: bool = True) -> bool:
        """May the GPU job ``name`` run beside others (``gpu.parallel``: auto looks at the card, see stt.gpu_budget)?"""
        return gpu_budget.allow_overlap(name, (self.config.get("gpu") or {}).get("parallel", "auto"), uses_gpu=uses_gpu)

    def episode_minutes(self) -> float | None:
        """Length of the episode's audio (``audio.wav``) in minutes, None if there is none yet."""
        try:
            import soundfile as sf

            return sf.info(str(self.out / "audio.wav")).duration / 60
        except Exception as error:  # noqa: BLE001 - no audio.wav or unreadable: the length is unknown
            log.debug("[%s] episode length unknown: %s", self.media.stem, error)
            return None

    @property
    def vram_peaks(self) -> Path:
        """Peak video memory of the jobs of this episode (stt.gpu_budget.record_peak)."""
        return self.out / "logs" / "vram_peaks.json"

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
        argv += ["--gpu-parallel", gpu_budget.normalize_mode((self.config.get("gpu") or {}).get("parallel", "auto"))]
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
        fill = self.config.get("fill") or {}
        if fill.get("enabled", True):
            argv += ["--fill", fill.get("model", "whisper-large-v3"), "--fill-min-s", str(fill.get("min_s", 1.0)),
                     "--fill-min-score", str(fill.get("min_score", 0.7))]
        if fill.get("subtitles", True):
            argv.append("--fill-subtitles")  # has an effect only with --subtitles below
            if fill.get("edges", True):
                argv.append("--fill-edges")
        subtitles = self.subtitle_source()
        if subtitles:
            argv += ["--subtitles", str(self.out / "subs.source.en.srt"), "--subtitle-turn-bonus",
                     str((self.config.get("subtitles") or {}).get("turn_bonus", 0.3))]
        if self.args.start:
            argv += ["--start", str(self.args.start)]
        if self.args.duration:
            argv += ["--duration", str(self.args.duration)]
        transcribe.main(argv)
        if subtitles:
            self._try_other_cuts(lambda: transcribe.main(argv))
        self.subtitle_stamp.parent.mkdir(parents=True, exist_ok=True)
        self.subtitle_stamp.write_text(self.subtitle_key() + "\n", encoding="utf-8")

    def pinned_share(self) -> float | None:
        """Share of the named subtitle lines the audio confirmed (``subtitle_hints.json``); None without such lines."""
        try:
            report = json.loads((self.out / "subtitle_hints.json").read_text(encoding="utf-8"))
            pinned, lost = int(report["pinned_lines"]), int(report["not_confirmed"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return pinned / (pinned + lost) if pinned + lost else None

    def _try_other_cuts(self, assemble: Callable[[], None]) -> None:
        """The base's file fits the audio badly (named lines mostly not confirmed): it is probably from another cut of
        the episode. Try the base's alternates (at most 2), keep the one with the best share of confirmed lines
        (``assemble`` redoes the assembly only), and record every file tried but not kept as ``rejected`` in
        ``subs.source.json``. A good enough alternate ends the search; when none is good enough, the best is kept."""
        from stt import subtitles as subs

        found, cfg = self.subtitle_source(), self.config.get("subtitles") or {}
        share = self.pinned_share()
        threshold = float(cfg.get("min_pinned", 0.3))
        code = subs.episode_code(self.media)
        if not found or found.get("origin") != "base" or not found.get("labels") or not cfg.get("base") or not code \
                or share is None or share >= threshold:
            return
        alternates = subs.base_alternates(Path(str(cfg["base"])).expanduser(), self.media,
                                          subs.imdb_of(self.media, cfg.get("imdb", "tt0239195")), *code)[:2]
        if not alternates:
            return
        log.info("[%s] subtitles from the base: only %.0f%% of the named lines fit the audio; trying other cuts",
                 self.media.stem, 100 * share)
        path = self.out / subs.SOURCE_SRT
        original = path.read_text(encoding="utf-8")
        # (share, file, text, facts) of every file tried; the base's own file first
        tried: list[tuple[float, str, str, dict[str, Any]]] = [(share, str(found["path_or_url"]), original, found)]
        for alternate in alternates:
            path.write_text(alternate.text, encoding="utf-8")
            assemble()
            tried.append((self.pinned_share() or 0.0, alternate.where, alternate.text, alternate.info))
            if tried[-1][0] >= threshold:
                break
        best = max(tried, key=lambda t: t[0])  # the first of equals: the base's own file wins ties
        if best[0] < threshold:
            log.warning("[%s] no file of the base fits the audio (best: %.0f%% of the named lines confirmed); "
                        "keeping the best", self.media.stem, 100 * best[0])
        if best is not tried[-1]:
            path.write_text(best[2], encoding="utf-8")
            assemble()
        info = best[3]
        result = {**found, "origin": "base", "path_or_url": best[1],
                  **{k: info[k] for k in ("kind", "labels", "turns", "cues")},
                  "sha1": hashlib.sha1(best[2].encode("utf-8")).hexdigest()[:10],
                  "rejected": [{"path": t[1], "pinned_share": round(t[0], 3)} for t in tried if t is not best]}
        (self.out / subs.SOURCE_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        self._subtitles = result  # the stamp written next must name the file now in use

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
            except LLMStop:
                raise  # the subscription does not answer (or paying is not allowed): the run stops
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
        from stt import season_voices, speakers

        if season_voices.restore_first_pass(self.out):  # else the season voices would start from the old names
            log.info("[%s] speakers: named again from the first pass; the season voices follow", self.media.stem)
        argv =[str(self.out), "--glossary", str(self.out / "glossary.txt"), "--model", self.llm("speakers")]
        if (self.config.get("voice_bank") or {}).get("enabled", True):
            argv += ["--voice-bank", str(self.voice_bank)]
        if not (self.config.get("voice_bank") or {}).get("identities", True):
            argv.append("--no-identities")
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


def run_lanes(files: list[Path], out_root: Path, args: argparse.Namespace, config: dict[str, Any],
              engines: list[str]) -> None:
    """Every episode in every engine (``stt.stage_runner.run_lanes``) with this module's ``Episode`` and
    ``wait_onscreen``."""
    stage_runner.run_lanes(files, out_root, args, config, engines, episode_cls=Episode, wait_onscreen=wait_onscreen)




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
        # A translated episode stays: a new transcript would mean a new translation and voice-over, LLM costs included;
        # unless its naming ran again since the translation (an answer of «Кто это?»): it is translated again anyway
        fresh = [m for m in medias if forced or not (episodes[m].out / "subs.ru.json").exists()
                 or episodes[m].named_after_translation()]
        for media in medias:
            if media not in fresh:
                log.info("[%s] season voices: already translated, stays as it is (--force speakers assigns it again)", media.stem)
        if not fresh:
            continue
        targets = [episodes[m].out for m in fresh]
        results = season_voices.apply_season(every, targets, episodes[fresh[0]].season_voices, model=model,
                                             min_words=int(cfg.get("min_words", season_voices.MIN_WORDS)),
                                             caption_bonus=float(cfg.get("caption_bonus", season_voices.TURN_BONUS)),
                                             tribe_prior=float(cfg.get("tribe_prior", 0.0)),
                                             dissolve=str(cfg.get("dissolve", "none")),
                                             voice_source=str(cfg.get("voice_source", "all")),
                                             guests=str(cfg.get("guests", "everywhere")))
        for media in fresh:
            episode = episodes[media]
            result = results[episode.out.name]
            log.info("[%s] season voices: %s", media.stem, season_voices.summary(episode.out.name, result))
            if result.changed:
                episode.stamp("speakers").parent.mkdir(parents=True, exist_ok=True)
                episode.stamp("speakers").touch()


_EPISODE_CODE = re.compile(r"S\d{1,2}E\d{1,3}", re.IGNORECASE)


def prepare_summary(files: list[Path], out_root: Path) -> list[str]:
    """Lines telling what ``--prepare`` left in each source folder of ``files``: episodes with ``speakers.json`` out of
    all its videos, those already translated (kept as they are), the unnamed speaker groups per episode and the next step."""
    from stt import who

    def label(stem: str) -> str:
        code = _EPISODE_CODE.search(stem)
        return code.group(0).upper() if code else stem

    lines: list[str] = []
    unnamed_total = 0
    for folder in dict.fromkeys(media.parent for media in files):
        stems = [p.stem for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES]
        ready = [s for s in stems if (out_root / s / "speakers.json").exists()]
        lines.append(f"{folder}: {len(ready)} of {len(stems)} episodes have speakers.json")
        translated = [s for s in ready if (out_root / s / "subs.ru.json").exists()]
        if translated:
            lines.append("  already translated, kept as they are: " + ", ".join(label(s) for s in translated))
        groups = who.episode_groups(out_root, ready) if ready else []
        by_episode: dict[str, list[str]] = {}
        for group in groups:
            by_episode.setdefault(group.episode, []).append(f"{group.speaker} {group.cues} lines")
        unnamed_total += len(groups)
        for stem in ready:
            if stem in by_episode:
                lines.append(f"  {label(stem)}: unnamed {', '.join(by_episode[stem])}")
    if unnamed_total:
        lines.append("Next: optionally name the unnamed voices in the UI tab «Кто это?», then voice each episode "
                     "with the normal command (stt.run without --prepare).")
    else:
        lines.append("Next: voice each episode with the normal command (stt.run without --prepare).")
    return lines




def check_subscription(config: dict[str, Any]) -> bool:
    """Ask the subscription before the episodes; False (with a warning) when it does not answer.

    Per model: first the model must be offered by the login (a renamed model stops the run at once, naming the
    available ones), then one tiny request goes the way real requests do: the whole retry budget and the stand-in
    model of the subscription (``llm.subscription_fallback``). If only the stand-in answers the run starts with a
    warning. If nothing answers the run stops (``SubscriptionDown``) when it may not pay; otherwise the steps go
    through the paid router and cost money.
    """
    from stt.llm import subscription_offered, subscription_probe
    from utils.llm_route import subscription_fallback

    llm = config.get("llm") or {}
    export_llm_env(config)
    for model in subscription_models(config):
        error = subscription_offered(model)
        stood_in = False
        if error is not None:  # the login works but its model list lacks the model (renamed by an OpenCode update?)
            error = (f"the model {model} is not in OpenCode's model list (renamed? pick another one in the settings) "
                     f"/ модели {model} нет в списке OpenCode (переименована? выберите другую в настройках): {error}")
            if not llm.get("fallback", False):
                raise SubscriptionDown(error)
        else:
            error, stood_in = subscription_probe(model)
        if error is None:
            if stood_in:
                log.warning("%s is not answering, the run starts; requests go to %s when %s does not answer / "
                            "%s не отвечает, запуск начинается; запросы пойдут в %s, когда %s не отвечает", model,
                            subscription_fallback(model), model, model, subscription_fallback(model), model)
            continue
        if not llm.get("fallback", False):
            raise SubscriptionDown(f"{SUBSCRIPTION_DOWN}: {error}")
        router = llm.get("router", "openrouter")
        log.warning("%s: %s. LLM steps with %s go through %s instead and cost money.", SUBSCRIPTION_DOWN, error,
                    model, " / ".join([router] if isinstance(router, str) else router))
        return False
    return True


def expand_media(patterns: list[str]) -> tuple[list[Path], list[str]]:
    """Files named on the command line: an existing file is taken as it is (names like ``[EZTVx.to]`` hold glob
    characters), anything else is a glob pattern. Returns the files and the arguments that matched no file."""
    files: list[Path] = []
    missing: list[str] = []
    for pattern in patterns:
        if Path(pattern).is_file():
            files.append(Path(pattern))
            continue
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else []
        files += [Path(m) for m in matches if Path(m).is_file()]
        if not matches:
            missing.append(pattern)
    return files, missing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="English video -> Russian voice-over, end to end")
    parser.add_argument("media", nargs="*", help="Video files or glob patterns (a whole season works)")
    parser.add_argument("--out-root", default=None,
                        help="One sub-folder per episode (default: paths.work in the config, else output/stt)")
    parser.add_argument("--config", default=str(REPO_ROOT / "stt_config.json"))
    parser.add_argument("--dump-config", action="store_true",
                        help="Print effective settings and leaf sources without running models (media optional); the "
                             "layers are the command line's (base/local/preset/show/cli), the season_settings.json "
                             "that the UI saves separately is not included")
    parser.add_argument("--preset", default=None, help="A named set of settings (presets in the config): main, upload, fast, previous")
    parser.add_argument("--show", default=None,
                        help="Settings profile of a show (shows in the config; auto by the video's path; none: no profile)")
    parser.add_argument("--mode", choices=["fast", "accurate"], default=None, help="Default from the config")
    parser.add_argument("--until", choices=STAGES, default=None, help="Last stage to run (default: mix)")
    parser.add_argument("--prepare", action="store_true",
                        help="Prepare the season: every episode up to speakers (transcript, on-screen text, who speaks, names, "
                             "season voices), no translation or voice-over")
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
    if not args.media and not args.dump_config:
        parser.error("the following arguments are required: media")
    if args.prepare and args.until not in (None, "speakers"):
        parser.error(f"--prepare stops after speakers, it cannot be combined with --until {args.until}")
    args.until = "speakers" if args.prepare else args.until or "mix"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    layers = read_config_layers(Path(args.config)) if args.dump_config else []
    base = resolve_config_layers(layers)[0] if args.dump_config else load_config(Path(args.config))
    config = apply_preset(base, args.preset)
    if args.dump_config:
        if args.preset:
            layers.append((f"preset:{args.preset}", base["presets"][args.preset].get("settings") or {}))
        cli: dict[str, Any] = {}
        if args.mode:
            cli["mode"] = args.mode
        if args.engine:
            cli["voiceover"] = {"engine": args.engine}
        if args.out_root:
            cli["paths"] = {"work": args.out_root}
        if args.no_subtitles:
            cli["subtitles"] = {"enabled": False}
        if args.no_season_voices:
            cli["season_voices"] = {"enabled": False}
        # Episode applies the show after the preset; CLI options remain authoritative.
        media, _ = expand_media(args.media)
        for path in media or [Path(p) for p in args.media] or [Path()]:
            show = show_of(config, path, args.show)
            per_media = list(layers)
            if show:
                per_media.append((f"show:{show}", config["shows"][show].get("settings") or {}))
            per_media.append(("cli", cli))
            resolved, sources = resolve_config_layers(per_media)
            print(format_config_sources(resolved, sources))
        return
    args.mode = args.mode or config["mode"]
    files, missing = expand_media(args.media)
    if missing or not files:
        raise SystemExit(f"Media not found: {missing or args.media}")
    out_root = Path(args.out_root or (config.get("paths") or {}).get("work") or "output/stt").expanduser()
    cost_log = out_root / ".llm_cost" / (time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    os.environ[COST_LOG_ENV] = str(cost_log)  # one line per paid LLM answer of this run (also from stage processes)
    export_llm_env(config)  # also when the subscription probe is skipped: the stage processes inherit it
    try:
        run_all(args, config, files, out_root)
    except LLMStop as error:  # no LLM answer without paying: say what happened, once, as the last thing
        log_paid(cost_log)
        log.error("%s", error.message)
        raise SystemExit(LLM_STOP_EXIT) from None
    except BaseException:
        log_paid(cost_log)
        raise
    log_paid(cost_log)


def log_paid(cost_log: Path) -> None:
    """The one line about money at the end of a run (warning when something was paid)."""
    line = paid_line(cost_log)
    (log.warning if PAID_MARK in line else log.info)("%s", line)


def run_all(args: argparse.Namespace, config: dict[str, Any], files: list[Path], out_root: Path) -> None:
    """The run after the arguments are read: voice model, LLM checks, environments, season voices, the lanes."""
    if args.download_voice_model and (args.season_voices_only or season_voices_enabled(config, args)):
        fetch_voice_model(config)
    if not args.season_voices_only:
        check_paid_steps(config)
        check_subscription(config)
    engines = [e.strip() for e in (args.engine or (config.get("voiceover") or {}).get("engine", "vosk")).split(",")
               if e.strip()]
    args.planned_engines = engines  # the finished video's size budget counts these voice-over tracks
    from stt import preflight

    absent = preflight.missing_envs(REPO_ROOT, config, engines, args.until, STAGES)
    if absent:  # stop now instead of a quietly worse result hours later
        raise SystemExit(preflight.message(absent))
    if args.season_voices_only:
        run_season_voices(files, out_root, args, config, engines, first_pass=False)
        return
    if season_voices_enabled(config, args):
        run_season_voices(files, out_root, args, config, engines)
    run_lanes(files, out_root, args, config, engines)
    if args.prepare:
        for line in prepare_summary(files, out_root):
            log.info("prepare: %s", line)
            print(line)


if __name__ == "__main__":
    # ``python -m stt.run`` runs this file as ``__main__``: run the importable ``stt.run`` instead, so the code split
    # out of it (``stt.stage_runner.run_module``) and this entry point share one module and its state.
    from stt import run as _run

    _run.main()