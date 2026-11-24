"""Shared worker commands, subprocess compatibility, failures and cancellation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from stt import runner
from stt.runner import ENV_WORKERS, WorkerSpec, run_worker, worker_command, worker_python


@pytest.mark.parametrize("name", ["f5", "xvoice", "diarizen", "lava"])
def test_worker_commands_preserve_entry_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    spec = ENV_WORKERS[name]
    python = tmp_path / spec.env / "bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    entry = (["-I", str(tmp_path / "voiceover/xvoice_worker.py")] if name == "xvoice"
             else ["-m", spec.module])
    assert worker_python(spec) == python
    assert worker_command(spec, ["job.json", "out.json"]) == [str(python), *entry, "job.json", "out.json"]
    assert worker_command(spec, [], Path(sys.executable)) == [sys.executable, *entry]


def test_missing_worker_environment_and_override_explain_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="f5_env.*setup_envs.sh f5"):
        worker_command(ENV_WORKERS["f5"], [])
    with pytest.raises(FileNotFoundError, match="missing.*setup_envs.sh xvoice"):
        worker_python(ENV_WORKERS["xvoice"], tmp_path / "missing")
    (tmp_path / "f5_env/bin").mkdir(parents=True)
    (tmp_path / "f5_env/bin/python").symlink_to(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="f5_env"):
        worker_python(ENV_WORKERS["f5"])


def test_run_module_keeps_combined_log_tail_and_subprocess_run_mock(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_SPEAKER_CHECKPOINT", "chosen-checkpoint")
    log_path = tmp_path / "logs/worker.log"

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command == [sys.executable, "-m", "stt.worker", "diarize"]
        assert kwargs["cwd"] == runner.REPO_ROOT
        assert kwargs["env"]["STT_SPEAKER_CHECKPOINT"] == "chosen-checkpoint"
        assert kwargs["env"]["PYTHONPATH"] == str(runner.REPO_ROOT)
        assert kwargs["stderr"] == subprocess.STDOUT
        kwargs["stdout"].write("\n".join(f"line {n}" for n in range(8)) + "\n")
        return subprocess.CompletedProcess(command, 7)

    with patch.object(runner, "env_python", return_value=Path(sys.executable)), \
            patch.object(runner.subprocess, "run", side_effect=fake_run):
        assert runner.run_module("diarizen_env", "stt.worker", ["diarize"], log_path) == (
            7, "line 3 | line 4 | line 5 | line 6 | line 7")


def test_worker_failure_includes_only_log_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    (tmp_path / "fake.py").write_text(
        "import sys\nsys.stderr.write('discard this' + 'x' * 2500 + '\\nCUDA failure\\n')\nsys.exit(3)\n",
        encoding="utf-8")
    spec = WorkerSpec("unused", "fake", "Fake worker", isolated=True)
    log_path = tmp_path / "logs/worker.log"
    with pytest.raises(RuntimeError, match="Fake worker failed \\(exit 3\\)") as error:
        run_worker(spec, [], log_path, python=Path(sys.executable))
    assert "CUDA failure" in str(error.value) and "discard this" not in str(error.value)
    assert str(log_path) in str(error.value)
    assert log_path.read_text(encoding="utf-8").startswith("discard this")
    done = run_worker(spec, [], python=Path(sys.executable), capture_output=True, check=False)
    assert done.returncode == 3 and done.stderr.endswith("CUDA failure\n")
    with pytest.raises(RuntimeError, match="CUDA failure"):
        run_worker(spec, [], python=Path(sys.executable), capture_output=True)


def test_polled_worker_reports_progress_and_uses_stderr_log(tmp_path: Path) -> None:
    process = Mock(returncode=0)
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
    callback = Mock()
    with patch.object(runner.subprocess, "Popen", return_value=process) as popen:
        done = run_worker(ENV_WORKERS["f5"], ["job", "out"], tmp_path / "worker.log",
                          python=Path(sys.executable), on_poll=callback, poll_s=0.1)
    assert done.returncode == 0
    callback.assert_called_once_with()
    assert popen.call_args.kwargs["stdout"] is None
    assert popen.call_args.kwargs["stderr"].name == str(tmp_path / "worker.log")
    assert popen.call_args.kwargs["env"] is None
    assert process.wait.call_args_list == [call(timeout=0.1), call(timeout=0.1)]
    process.terminate.assert_not_called()


@pytest.mark.parametrize("kill_needed", [False, True])
def test_cancellation_terminates_and_reaps_worker(tmp_path: Path, kill_needed: bool) -> None:
    process = Mock(returncode=None)
    process.poll.return_value = None
    waits: list[Any] = [KeyboardInterrupt()]
    waits += [subprocess.TimeoutExpired("worker", 5), 0] if kill_needed else [0]
    process.wait.side_effect = waits
    with patch.object(runner.subprocess, "Popen", return_value=process), pytest.raises(KeyboardInterrupt):
        run_worker(ENV_WORKERS["f5"], [], tmp_path / "worker.log", python=Path(sys.executable), on_poll=lambda: None)
    process.terminate.assert_called_once_with()
    assert process.kill.call_count == int(kill_needed)
    assert process.wait.call_count == (3 if kill_needed else 2)


def test_progress_callback_failure_stops_worker(tmp_path: Path) -> None:
    process = Mock(returncode=None)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
    with patch.object(runner.subprocess, "Popen", return_value=process), pytest.raises(ValueError, match="progress"):
        run_worker(ENV_WORKERS["f5"], [], tmp_path / "worker.log", python=Path(sys.executable),
                   on_poll=Mock(side_effect=ValueError("progress")))
    process.terminate.assert_called_once_with()
    process.wait.assert_called_with(timeout=5)


def test_xvoice_keeps_isolation_cwd_and_subprocess_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stt import ru_samples

    monkeypatch.setattr(ru_samples, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    python = tmp_path / "xvoice_env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    repo = tmp_path / "models/X-Voice"
    repo.mkdir(parents=True)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    items = [{"text": "Пример"}]

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[:3] == [str(python), "-I", str(tmp_path / "voiceover/xvoice_worker.py")]
        assert kwargs["cwd"] == repo.resolve() and kwargs["env"] is None
        assert kwargs["stderr"].name == str(job_dir / "worker.log") and kwargs["stdout"] is None
        job = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        assert job["xvoice_repo"] == str(repo.resolve()) and job["items"] == items
        Path(command[4]).write_text(json.dumps(items), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with patch.object(ru_samples.subprocess, "run", side_effect=fake_run):
        assert ru_samples.run_worker(items, job_dir) == items


def test_lava_keeps_command_output_and_subprocess_mock(tmp_path: Path) -> None:
    from stt import widen

    out = tmp_path / "track.wide.wav"
    track = tmp_path / "track.wav"

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command == [sys.executable, "-m", "stt.widen", "--worker", str(track),
                           str(tmp_path / "track.wide.tmp.wav"), "--edge", "5200"]
        assert kwargs["cwd"] == widen.REPO_ROOT and kwargs["capture_output"] is True and kwargs["text"] is True
        Path(command[5]).write_bytes(b"widened")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch.object(widen.subprocess, "run", side_effect=fake_run):
        assert widen.widen_track(track, out, 5200.0, Path(sys.executable))
    assert out.read_bytes() == b"widened"
    assert not (tmp_path / "track.wide.tmp.wav").exists()
