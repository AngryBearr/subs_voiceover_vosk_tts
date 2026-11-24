"""stt.run after its split into stt.stage_runner, stt.episode_vo, stt.episode_media and stt.run_llm: every name stays
reachable through stt.run, and patching stt.run still reaches the moved code."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from stt import episode_media, episode_vo, run, run_llm, stage_runner

# Names other code and tests import from stt.run (module level) and Episode methods they call or patch.
RUN_NAMES = [
    "Background", "CPU_ENGINES", "ENGINE_STAGES", "Episode", "PROGRESS_LOG_LINES", "PROGRESS_POLL_S", "REPO_ROOT",
    "SHARED_STAGES", "STAGES", "SUBSCRIPTION_DOWN", "SubscriptionDown", "VIDEO_SUFFIXES", "_COMPRESS_JOBS",
    "_ONSCREEN_JOBS", "apply_preset", "apply_show", "check_paid_steps", "check_subscription", "dotenv",
    "expand_media", "export_llm_env", "fetch_voice_model", "finished_folder", "lanes", "llm_steps", "load_config",
    "load_json", "lookahead_enabled", "main", "paying_allowed", "prepare_summary", "run_all", "run_lanes",
    "run_season_voices", "save_json", "season_voices_enabled", "season_voices_model", "show_of",
    "subscription_models", "subprocess", "time", "wait_onscreen",
]
EPISODE_METHODS = [
    "_engine", "_execute", "_find_cast", "_host_voices", "_module", "_own_voice_mapping", "_reference_clip",
    "_speaker_voices", "_stress_input_unchanged", "_translate_onscreen", "_try_other_cuts", "_voiceover_config",
    "check", "dub", "finished_video", "llm", "mix", "mux", "mux_when_ready", "needed", "on_screen", "route",
    "route_in", "run", "small_video", "speakers", "start_compression", "start_onscreen", "steps", "stress", "track",
    "transcribe", "translate", "translate_onscreen", "voiceover", "widened_track",
]


@pytest.mark.parametrize("name", RUN_NAMES)
def test_name_is_still_in_stt_run(name: str) -> None:
    assert hasattr(run, name)


@pytest.mark.parametrize("name", EPISODE_METHODS)
def test_episode_still_has_method(name: str) -> None:
    assert callable(getattr(run.Episode, name))


def test_moved_state_is_the_same_object() -> None:
    """The background jobs are shared: what the mixins write is what stt.run (and the tests) read."""
    assert run._COMPRESS_JOBS is episode_media._COMPRESS_JOBS
    assert run._ONSCREEN_JOBS is episode_media._ONSCREEN_JOBS
    assert run.STAGES is stage_runner.STAGES
    assert run.Background is stage_runner.Background
    assert stage_runner.run_module() is run


def test_moved_code_logs_as_stt_run() -> None:
    """The UI and the log readers look for "stt.run" in the log lines."""
    for module in (stage_runner, episode_vo, episode_media, run_llm):
        assert module.log.name == "stt.run"


def test_route_in_is_the_same_on_episode_and_in_run_llm() -> None:
    config = {"llm": {"opencode": ["openai/*"], "router": ["orcarouter", "openrouter"]}}
    for name in ("openai/gpt-6-luna", "anthropic/claude-sonnet-5.5", "orcarouter+openrouter:x/y", "a/b:online"):
        assert run.Episode.route_in(config, name) == run_llm.route_in(config, name)


def test_patched_dotenv_and_repo_root_reach_the_stage_programs(tmp_path: Path) -> None:
    episode = mock.Mock(out=tmp_path, media=Path("ep.mkv"))
    with mock.patch.object(run, "dotenv", return_value={"FROM_DOTENV": "1"}), \
            mock.patch.object(run, "REPO_ROOT", tmp_path), \
            mock.patch.object(run.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as started:
        run.Episode._module(episode, "x", "stt.x", [])
    assert started.call_args.kwargs["env"]["FROM_DOTENV"] == "1"
    assert started.call_args.kwargs["cwd"] == tmp_path


def test_patched_load_json_reaches_the_voiceover_config(tmp_path: Path) -> None:
    episode = run.Episode.__new__(run.Episode)
    episode.out, episode.engine, episode.media = tmp_path, "vosk", Path("ep.mkv")
    episode.config = {"voiceover": {"engines": {"vosk": {"voices": {"male": 1, "female": 2}}}}, "llm": {"model": "m"}}
    base: dict[str, Any] = {"engine": {"model_path": "vosk"}}
    with mock.patch.object(run, "load_json", return_value=base) as loaded:
        path = episode._voiceover_config()
    assert loaded.called and run.load_json(path)["engine"]["name"] == "vosk"


def test_run_lanes_uses_the_episode_of_stt_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[str] = []

    class FakeEpisode:
        def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any],
                     lock: Any = None) -> None:
            self.args = args
            made.append(f"{media.stem}:{args.engine}")

        def run(self) -> None:
            pass

    monkeypatch.setattr(run, "Episode", FakeEpisode)
    run.run_lanes([tmp_path / "e1.mkv"], tmp_path, argparse.Namespace(engine="f5", until="mix", force=None),
                  {"overlap_episodes": False}, ["f5"])
    assert made == ["e1:f5"]


def test_dump_config_help_names_the_layers(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        run.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "base/local/preset/show/cli" in text and "season_settings.json" in text


def test_logger_name_constant() -> None:
    assert logging.getLogger("stt.run") is run.log
