"""Stages and processes of ``stt.run``: the stage names, the stage modules started as programs of their own
(``Episode._module``: logs, progress, retry after running out of GPU memory) and the lanes that run every episode in
every engine (``run_lanes``: GPU and CPU engines side by side, the next episode translated ahead).

Split out of ``stt/run.py``; ``stt.run`` re-exports every name, and its log lines keep the ``stt.run`` logger.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from stt import gpu_budget, progress
from stt.gpu import VramPeak
from utils.llm_route import LLM_STOP_EXIT, LLMStop, SubscriptionDown, stop_message_in

log = logging.getLogger("stt.run")

STAGES = ["transcribe", "glossary", "speakers", "translate", "dub", "stress", "voiceover", "check", "mix"]
PROGRESS_POLL_S = 2.0  # how often a running module's progress file is looked at
PROGRESS_LOG_LINES = 25  # progress of lines / clips goes to the log every this many
SHARED_STAGES = {"transcribe", "glossary", "speakers", "translate", "stress"}  # one result for all engines
ENGINE_STAGES = {"dub", "voiceover", "check", "mix"}
CPU_ENGINES = {"vosk"}  # synthesise on the CPU: can run next to the GPU engines


def run_module() -> ModuleType:
    """``stt.run`` looked up when called: the settings and helpers tests patch there (``REPO_ROOT``, ``load_json``,
    ``dotenv``) keep their effect on the code split out of it. ``python -m stt.run`` runs this same module."""
    return importlib.import_module("stt.run")


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



class ProcessMixin:
    """``Episode`` methods that start a stage module as a program of its own and follow its progress."""

    out: Path
    media: Path

    if TYPE_CHECKING:
        @property
        def vram_peaks(self) -> Path: ...

    def _module(self, name: str, module: str, args: list[str], *, oom_retries: int = 2,
                oom_wait_s: float = 60.0) -> None:
        """Run a module of this repo in the current interpreter, logging to logs/<name>.log.

        A run that ran out of GPU memory is started again after a pause; the voice-over keeps
        every line already synthesised, so it goes on where it stopped.
        """
        from stt.worker import gpu_out_of_memory

        stt_run = run_module()  # names tests patch there: dotenv, REPO_ROOT
        env = {**os.environ, **stt_run.dotenv(),
               "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        env.update({k: v for k, v in os.environ.items() if k.startswith("STT_LLM_")})  # the run's, never ``.env``'s
        tracker = progress.current()
        if tracker is not None:  # the module's progress ticks go to this stage's progress file
            env.update(tracker.environ())
        log_path = self.out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(oom_retries + 1):
            with log_path.open("w", encoding="utf-8") as log_file:
                command = [sys.executable, "-m", module, *args]
                if tracker is None:  # outside a tracked stage there is no progress to watch
                    code = subprocess.run(command, cwd=stt_run.REPO_ROOT, env=env, stdout=log_file,
                                          stderr=subprocess.STDOUT).returncode
                else:
                    code = self._run_watched(command, env, log_file, tracker, name)
            if code == 0:
                return
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if code == LLM_STOP_EXIT:  # the stage stopped on purpose: no LLM answer without paying (see utils.llm_route)
                raise SubscriptionDown(f"stage {name} stopped (exit {code}), see {log_path}", stop_message_in(log_text))
            if attempt == oom_retries or not gpu_out_of_memory(log_text):
                break
            log.warning("[%s] %s ran out of GPU memory; retry %d/%d in %.0f s", self.media.stem, name, attempt + 1,
                        oom_retries, oom_wait_s)
            log_path.with_suffix(f".oom{attempt + 1}.log").write_text(log_text, encoding="utf-8")
            time.sleep(oom_wait_s)
        raise SystemExit(f"{name} failed (exit {code}), see {log_path}")

    def _run_watched(self, command: list[str], env: dict[str, str], log_file: Any,
                     tracker: progress.Tracker, name: str = "") -> int:
        """Run ``command`` (output to ``log_file``) and return its exit code; while it runs, copy its progress
        (``[<episode>] voiceover (f5): 25/640 lines``) into this log every ``PROGRESS_LOG_LINES`` lines."""
        process = subprocess.Popen(command, cwd=run_module().REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        last = 0
        code: int | None = None
        try:
            with VramPeak(interval_s=0.5, pid=process.pid) as peak:  # the module's own process (not its children)
                while code is None:
                    try:
                        code = process.wait(timeout=PROGRESS_POLL_S)
                    except subprocess.TimeoutExpired:
                        last = self._log_progress(tracker, last)
        except BaseException:  # Ctrl+C or an error here: do not leave the module running on its own
            process.terminate()
            raise
        if name:
            gpu_budget.record_peak(self.vram_peaks, name, peak.peak_mb)
        return code

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


def lanes(engines: list[str], parallel: bool) -> list[list[str]]:
    """Engines in the order they run: one lane, or the GPU engines and the CPU ones side by side."""
    gpu = [e for e in engines if e not in CPU_ENGINES]
    cpu = [e for e in engines if e in CPU_ENGINES]
    return [gpu, cpu] if parallel and gpu and cpu else [engines]


def lookahead_enabled(files: list[Path], args: argparse.Namespace, config: dict[str, Any]) -> bool:
    """Translate the next episode while the current one is voiced (``overlap_episodes``, several episodes, run
    reaches the translation). Off when a stage up to the translation is forced: the lookahead and the lane would both run it."""
    if not config.get("overlap_episodes", True) or len(files) < 2:
        return False
    translate = STAGES.index("translate")
    if STAGES.index(args.until) < translate:
        return False
    force = getattr(args, "force", None)
    if force and STAGES.index(force) <= translate:
        log.info("episodes overlap is off: --force %s is at or before the translation and would run twice", force)
        return False
    return True


def run_lanes(files: list[Path], out_root: Path, args: argparse.Namespace, config: dict[str, Any],
              engines: list[str], *, episode_cls: Callable[..., Any], wait_onscreen: Callable[[], None]) -> None:
    """Every episode in every engine; the second lane (CPU engines) starts an episode once the first
    has its shared stages, so the GPU-heavy recognition never runs twice at once.

    Lookahead (``lookahead_enabled``): a third thread runs the next episode up to ``translate`` (LLM only) while
    the first lane voices the current one. The first lane waits for it before it touches that episode. Events:
    ``voicing[i]`` (lane 0 started the engines of episode i), ``ahead_done[i]`` (the lookahead is done with i or
    skipped it). An error anywhere is appended to ``errors`` and sets every event, so no thread waits forever.

    ``episode_cls`` makes an episode (``stt.run.Episode``) and ``wait_onscreen`` waits for the background on-screen
    translations (``stt.run.wait_onscreen``); ``stt.run.run_lanes`` passes both."""
    groups = lanes(engines, bool((config.get("voiceover") or {}).get("parallel_cpu", True)))
    locks = {media: threading.Lock() for media in files}
    shared = {media: threading.Event() for media in files}
    voicing = {media: threading.Event() for media in files}
    ahead_done = {media: threading.Event() for media in files}
    errors: list[BaseException] = []
    ahead = lookahead_enabled(files, args, config)
    translate_stage = STAGES[min(STAGES.index(args.until), STAGES.index("translate"))]
    shared_forced: set[Path] = set()  # episodes whose forced shared stages have run already
    forced_guard = threading.Lock()
    # A forced shared stage up to the translation runs once per episode; the finish marks make every engine's later stages
    # run. The stress check of an engine whose dubbing lines did not change then keeps its result (same lines, same marks).
    force_shared = bool(getattr(args, "force", None)) and STAGES.index(args.force) <= STAGES.index("translate")

    def release_all() -> None:
        for events in (shared, voicing, ahead_done):
            for event in events.values():
                event.set()

    def fail(error: BaseException) -> None:
        errors.append(error)
        release_all()

    def episode(media: Path, engine: str, until: str | None = None) -> Any:
        lane_args = argparse.Namespace(**{**vars(args), "engine": engine, "until": until or args.until})
        if force_shared:
            with forced_guard:  # the forced shared stages run once per episode, the finish marks pull the rest along
                if media in shared_forced:
                    lane_args.force = None
                shared_forced.add(media)
        return episode_cls(media, out_root, lane_args, config, lock=locks[media])

    def lane(group: list[str], first: bool) -> None:
        for media in files:
            if errors:
                return
            background = Background()  # this lane's mixes running while its next engine dubs and voices
            try:
                if first and ahead:
                    ahead_done[media].wait()  # the lookahead may be on this episode
                    if errors:
                        return
                if first and (len(groups) > 1 or ahead):
                    try:
                        episode(media, group[0], translate_stage).run()
                    finally:
                        shared[media].set()
                elif not first:
                    shared[media].wait()
                if errors:
                    return
                voicing[media].set()
                for engine in group:
                    if background.failed() or errors:  # another thread failed: no new engine starts
                        break
                    ep = episode(media, engine)
                    ep.args.mix_background = background
                    ep.run()
            except BaseException as error:  # noqa: BLE001 - stop all threads, report after they end
                fail(error)
            finally:
                try:
                    background.join()
                except BaseException as error:  # noqa: BLE001
                    fail(error)
            if errors:
                return

    def lookahead() -> None:
        try:
            for previous, media in zip(files, files[1:]):
                voicing[previous].wait()
                if errors:
                    return
                probe = episode(media, groups[0][0], "translate")
                if probe.gpu_stage_needed():
                    log.info("[%s] not translated ahead: it needs recognition or names first", media.stem)
                else:
                    log.info("[%s] translate ahead while %s is voiced", media.stem, previous.stem)
                    probe.run()
                ahead_done[media].set()
        except BaseException as error:  # noqa: BLE001
            fail(error)
        finally:
            for event in ahead_done.values():
                event.set()

    ahead_done[files[0]].set()  # nothing runs before the first episode
    helper = threading.Thread(target=lookahead, name="lookahead") if ahead else None
    threads: list[threading.Thread] = []
    if helper:
        helper.start()
    try:
        if len(groups) == 1:
            lane(groups[0], True)
        else:
            from voiceover.engine import CPU_THREADS_ENV

            # Vosk on all cores starved the GPU engines' own CPU work (10-min piece: ESpeech 7.5 -> 12.3 min)
            os.environ.setdefault(CPU_THREADS_ENV, str(max(2, (os.cpu_count() or 4) // 2)))
            log.info("engines in two lanes: %s on the GPU, %s on the CPU alongside (%s CPU threads)",
                     ", ".join(groups[0]), ", ".join(groups[1]), os.environ[CPU_THREADS_ENV])
            for n, group in enumerate(groups):
                thread = threading.Thread(target=lane, args=(group, n == 0), name=f"lane-{n}")
                thread.start()
                threads.append(thread)
    finally:  # whatever happened above: nothing may stay waiting
        try:
            for thread in threads:
                thread.join()
        finally:
            if helper:
                release_all()  # a lane that returned early (or never started) must not leave the lookahead waiting
                helper.join()
    try:
        wait_onscreen()  # a run that ends at the translation still gets onscreen.ru.json
    except LLMStop as error:
        errors.append(error)
    if errors:  # the reason the run stops (no money, no subscription) is reported before any other error
        raise next((e for e in errors if isinstance(e, LLMStop)), errors[0])
