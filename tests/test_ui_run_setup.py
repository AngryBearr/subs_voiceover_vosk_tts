"""``ui.run_setup``: the JSON command line over the settings layers and the job launcher (no network, GPU or systemd)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ui import dub_app, run_setup, season_settings, seasons, settings
from ui import state as ui_state


def _run(*argv: str) -> tuple[int, dict[str, Any]]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_setup.main(list(argv))
    return code, json.loads(buffer.getvalue())


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Season_01 with three fake episodes (E01 finished), its results folder, a global file in tmp."""
    source = tmp_path / "media" / "Season_01"
    source.mkdir(parents=True)
    for n in (1, 2, 3):
        (source / f"Show.S01E0{n}.Title.720p.mkv").write_text("")
    results = tmp_path / "media" / "STT_Season_01"
    results.mkdir()
    (results / "Show.S01E01.Title.720p.ru.mkv").write_text("")
    ui_state.update({"seasons": [{"source": str(source), "results": str(results)}], "current_season": str(source)})
    monkeypatch.setattr(run_setup, "LAST_JOB", tmp_path / "out" / "ui_last_job.json")
    monkeypatch.setattr(run_setup, "stt_run_processes", lambda: [])
    monkeypatch.setattr(run_setup, "video_minutes", lambda path: 40.0)
    return {"source": source, "results": results, "tmp": tmp_path}


def test_target_resolution(world: dict[str, Path]) -> None:
    source, results, tmp = world["source"], world["results"], world["tmp"]
    assert run_setup.resolve_target("").season.source == source  # current
    assert run_setup.resolve_target("season_01").season.results == results  # by name
    assert run_setup.resolve_target(str(source)).season.results == results
    assert run_setup.resolve_target(str(results)).season.source == source
    new = tmp / "other" / "Season_77"
    new.mkdir(parents=True)
    (new / "a.S77E01.mkv").write_text("")
    assert run_setup.resolve_target(str(new)).season.results == tmp / "other" / "STT_Season_77"
    res = tmp / "other" / "STT_Season_77"
    res.mkdir()
    assert run_setup.resolve_target(str(res)).season.source == new
    target = run_setup.resolve_target(str(source / "Show.S01E02.Title.720p.mkv"))
    assert target.season.source == source and target.episode == "Show.S01E02.Title.720p.mkv"
    with pytest.raises(run_setup.CliError):
        run_setup.resolve_target(str(tmp / "nothing"))


def test_show_layers_sources_and_durations(world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    glob = tmp_path / "global.json"
    glob.write_text(json.dumps({"mode": "fast"}), encoding="utf-8")
    ui_state.update({"global_settings": str(glob)})
    season_settings.save_own(settings.form_values(season_settings.global_config()), world["results"],
                             season_settings.global_config(), only=None)
    own = json.loads((world["results"] / "season_settings.json").read_text(encoding="utf-8"))
    own["mix"]["duck_db"] = 5.0
    (world["results"] / "season_settings.json").write_text(json.dumps(own), encoding="utf-8")
    config = season_settings.global_config()
    config["shows"] = {"season 01": {"label": "Test show", "settings": {"mix": {"duck_db": 11}}}}
    monkeypatch.setattr(season_settings, "global_config", lambda *a, **k: config)
    code, out = _run("show")
    assert code == 0
    rows = {r["key"]: r for r in out["settings"]}
    assert rows["mode"]["source"] == "season" and rows["mode"]["display"].startswith("Быстрый")
    assert rows["mix.duck_db"]["season"] == 5.0 and rows["mix.duck_db"]["show_override"] == 11
    assert out["season"]["show"] == "season 01" and out["season"]["show_settings"] == {"mix.duck_db": 11}
    assert out["season"]["total_unfinished_min"] == 80.0
    assert [e["finished"] for e in out["episodes"]] == [True, False, False]
    assert out["episodes"][0]["code"] == "S01E01" and out["episodes"][1]["duration_min"] == 40.0
    assert {p["name"] for p in out["presets"]} >= {"main", "fast"}


def test_source_global_vs_standard(world: dict[str, Path], tmp_path: Path) -> None:
    glob = tmp_path / "global.json"
    glob.write_text(json.dumps({"mix": {"duck_db": 33}}), encoding="utf-8")
    ui_state.update({"global_settings": str(glob)})
    _, out = _run("show")
    rows = {r["key"]: r for r in out["settings"]}
    assert rows["mix.duck_db"]["source"] == "global" and rows["mix.duck_db"]["season"] is None
    assert rows["mix.rest_duck_db"]["source"] == "standard"
    assert rows["voiceover.engines.vosk.voices.female"]["choices"] is None


def test_save_global_writes_only_differences(world: dict[str, Path], tmp_path: Path) -> None:
    glob = tmp_path / "global.json"
    glob.write_text(json.dumps({"keep": 1}), encoding="utf-8")
    ui_state.update({"global_settings": str(glob)})
    code, out = _run("save", "--scope", "global", "mode=fast", "mix.duck_db=20", 'ui.engines=["vosk","f5"]')
    assert code == 0 and out["saved"]["mode"] == "fast"
    data = json.loads(glob.read_text(encoding="utf-8"))
    assert data["mode"] == "fast" and data["mix"] == {"duck_db": 20.0} and data["keep"] == 1
    assert data["ui"]["engines"] == ["vosk", "f5"]
    assert glob.with_name("global.json.bak").is_file()
    _run("save", "--scope", "global", "mode=accurate")
    assert "mode" not in json.loads(glob.read_text(encoding="utf-8"))  # equal to the standard: dropped


def test_save_season_snapshot_and_note(world: dict[str, Path]) -> None:
    code, out = _run("save", "--scope", "season", "Season_01", "mix.duck_db=9")
    assert code == 0 and out["created"] and "note" in out
    own = json.loads((world["results"] / "season_settings.json").read_text(encoding="utf-8"))
    assert own["mix"]["duck_db"] == 9.0 and "mode" in own  # a full snapshot
    code, out = _run("save", "--scope", "season", "mix.duck_db=10")
    assert not out["created"] and "note" not in out
    code, out = _run("forget-season")
    assert out["removed"] is True
    assert _run("forget-season")[1]["removed"] is False


def test_save_errors(world: dict[str, Path]) -> None:
    code, out = _run("save", "--scope", "global", "no.such=1")
    assert code == 2 and out["error"] == "bad_key" and "mode" in out["allowed"]
    code, out = _run("save", "--scope", "global", "mode=slow")
    assert code == 2 and out["error"] == "bad_value" and "accurate" in out["message"]
    code, out = _run("save", "--scope", "global", "mix.duck_db=loud")
    assert code == 2 and out["error"] == "bad_value"
    code, out = _run("save", "--scope", "global", "ui.engines=[\"nope\"]")
    assert out["error"] == "bad_value"


def test_launch_dry_run_config(world: dict[str, Path], tmp_path: Path) -> None:
    glob = tmp_path / "global.json"
    glob.write_text(json.dumps({"mix": {"duck_db": 20, "loudness_lufs": -10}}), encoding="utf-8")
    ui_state.update({"global_settings": str(glob)})
    (world["results"] / "season_settings.json").write_text(json.dumps({"mix": {"duck_db": 9}}), encoding="utf-8")
    code, out = _run("launch", "--dry-run")
    assert code == 0 and out["dry_run"] and not out["started"]
    assert [Path(f).name for f in out["files"]] == ["Show.S01E02.Title.720p.mkv", "Show.S01E03.Title.720p.mkv"]  # unfinished
    assert out["changed_from_standard"]["mix.duck_db"] == 9 and out["changed_from_standard"]["mix.loudness_lufs"] == -10
    assert not (world["results"] / ".ui").exists()
    code, out = _run("launch", "--dry-run", "--episodes", "s1e3", "--preset", "fast", "--transport", "openrouter",
                     "--set", "foo.bar=[1]", "mix.duck_db=3", "--preview-from", "1", "--preview-minutes", "2",
                     "--engines", "vosk,f5", "--no-season-voices", "--prepare")
    assert code == 0
    assert [Path(f).name for f in out["files"]] == ["Show.S01E03.Title.720p.mkv"]
    assert out["engines"] == ["f5", "vosk"] and out["preset"] == "fast"  # the heroes' voices run first
    assert out["out_root"] == str(world["results"] / "_проба")
    cmd = out["command"]
    assert cmd[cmd.index("--start") + 1] == "60.0" and cmd[cmd.index("--duration") + 1] == "120.0"
    assert "--no-season-voices" in cmd and "--prepare" in cmd and cmd[cmd.index("--engine") + 1] == "f5,vosk"
    changed = out["changed_from_standard"]
    assert changed["mix.duck_db"] == 3 and changed["foo.bar"] == [1] and changed["mode"] == "fast"
    config = run_setup.build_config(seasons.Season(world["source"], world["results"]), "fast", "openrouter",
                                    [("mix.duck_db", 3)])
    assert config["llm"]["opencode"] == [] and config["llm"]["router"] == "openrouter"
    assert config["paths"]["work"] == str(world["results"]) and config["paths"]["finished_folder"] == "."
    assert _run("launch", "--dry-run", "--preset", "fast")[1]["engines"] == ["vosk"]  # the preset's engines
    assert _run("launch", "--dry-run")[1]["engines"] == ["f5", "teratts", "vosk"]
    assert len(_run("launch", "--dry-run", "--all")[1]["episodes"]) == 3
    assert _run("launch", "--dry-run", "--episodes", "S09E09")[1]["error"] == "bad_episode"
    assert _run("launch", "--dry-run", "--preset", "zzz")[1]["error"] == "bad_preset"
    file_target = str(world["source"] / "Show.S01E01.Title.720p.mkv")
    assert [Path(f).name for f in _run("launch", file_target, "--dry-run")[1]["files"]] == ["Show.S01E01.Title.720p.mkv"]


def test_launch_starts_registers_and_ui_reads_it_back(world: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []

    def fake(cmd: list[str], log_path: Path, unit: str, wait_for: str = "stt.run") -> dict[str, Any]:
        started.append((cmd, log_path, unit))
        return {"pid": 4242, "detached": "systemd", "warnings": []}

    monkeypatch.setattr(run_setup, "start_process", fake)
    code, out = _run("launch", "--episodes", "S01E02")
    assert code == 0 and out["started"] and out["pid"] == 4242 and out["stop"].startswith("systemctl --user stop dub-")
    config_path = Path(out["config"])
    assert config_path.parent == world["results"] / ".ui" and json.loads(config_path.read_text(encoding="utf-8"))
    assert started[0][0][:3] == [sys.executable, "-m", "stt.run"] and started[0][2] == out["unit"]
    saved = json.loads(run_setup.LAST_JOB.read_text(encoding="utf-8"))
    assert saved["state"] == "идёт" and saved["pid"] == 4242 and saved["finished"] is None and saved["cmd"] is None
    job = dub_app.restore_last_job(run_setup.LAST_JOB)
    assert job is not None and job.title == run_setup.JOB_TITLE and job.engines == out["engines"]
    assert job.files == [Path(f) for f in out["files"]]


def test_busy_refusal(world: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_setup, "stt_run_processes", lambda: [{"pid": 77, "cmd": "python -m stt.run x"}])
    monkeypatch.setattr(run_setup, "start_process", lambda *a, **k: pytest.fail("must not start"))
    code, out = _run("launch")
    assert code == 3 and out["error"] == "busy" and out["pids"] == [77]
    code, out = _run("launch", "--dry-run")
    assert code == 0 and out["warnings"]


def test_status_reads_last_job_log(world: dict[str, Path]) -> None:
    log = world["tmp"] / "job.log"
    log.write_text("\n".join(f"line {n}" for n in range(30)) + "\n\n", encoding="utf-8")
    run_setup.LAST_JOB.parent.mkdir(parents=True, exist_ok=True)
    run_setup.LAST_JOB.write_text(json.dumps({"files": [], "state": "идёт", "title": "t", "pid": 999999999, "log_path": str(log)}),
                                  encoding="utf-8")
    _, out = _run("status")
    last = out["running"]["last_job"]
    assert last["alive"] is False and len(last["log_tail"]) == 20 and last["log_tail"][-1] == "line 29"


@pytest.fixture
def external() -> Iterator[subprocess.Popen[bytes]]:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "stt.run"])
    yield process
    process.kill()
    process.wait()


def test_adopt_external_job(world: dict[str, Path], external: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    from stt import progress

    monkeypatch.setattr(dub_app, "JOB", None)
    path = world["tmp"] / "ext.json"
    data = {"files": [str(world["source"] / "a.mkv")], "out_root": str(world["results"]), "engines": ["vosk"],
            "config_path": "c.json", "log_path": "l.log", "extra": [], "pid": external.pid, "boot": progress.boot_id(),
            "started": time.time(), "finished": None, "state": "идёт", "title": "Запущено из Claude", "cmd": None, "result": ""}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert dub_app.adopt_external_job(path) is True
    assert dub_app.JOB is not None and dub_app.JOB.pid == external.pid
    assert dub_app._busy() is not None  # a second job is refused
    assert dub_app.adopt_external_job(path) is False  # already known
    # a dead job in the file is not adopted
    monkeypatch.setattr(dub_app, "JOB", None)
    path.write_text(json.dumps({**data, "pid": 999999999}), encoding="utf-8")
    assert dub_app.adopt_external_job(path) is False and dub_app.JOB is None
    # a process that is not stt.run is not adopted
    path.write_text(json.dumps({**data, "pid": 1}), encoding="utf-8")
    assert dub_app.adopt_external_job(path) is False


def test_video_minutes_unknown_for_a_broken_file(tmp_path: Path) -> None:
    broken = tmp_path / "x.mkv"
    broken.write_text("not a video")
    assert run_setup.video_minutes(broken) is None
    assert run_setup.video_minutes(tmp_path / "missing.mkv") is None
