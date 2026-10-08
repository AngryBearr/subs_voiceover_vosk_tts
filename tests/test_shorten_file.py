from __future__ import annotations

import asyncio
import json
import signal
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils.llm_types import DetailedResponse
from utils.shortening_runner import ModelSpec
from utils.vosk_calibration import model_metadata, profile_identity
from utils.shorten_file import MODEL_CHOICES, _LazyRequest, execute_run, main, prepare_run


def _fixture(tmp_path: Path, *, text: str = "one two three four five") -> tuple[Namespace, Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"fake")
    (model / "dictionary").write_text("fake", encoding="utf-8")
    (model / "config.json").write_text(json.dumps({"num_speakers": 9}), encoding="utf-8")
    source = tmp_path / "source.JSON"
    source.write_bytes(b"\xef\xbb\xbf" + json.dumps([{
        "index": 77, "start": 0, "end": 1000, "duration": 1000,
        "text": text, "unknown": {"keep": True},
    }]).encode())
    identity = profile_identity(model_metadata(model), 8, 1.5)
    profile = {
        "rate": 1.5,
        "status": "ok",
        "normalizer": "vosk_punctuation_v1_no_ruaccent",
        "input_character_basis": "normalized_tts_text",
        "chosen_model": "char_linear",
        "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.0, 0.1]}},
        "char_linear_guard_sec": 0.0,
        "identity": identity,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    args = Namespace(
        input=source, output_dir=tmp_path / "run", model="openai/gpt-5.6-luna",
        profile=profile_path, model_path=model, voice=8, rate=1.5,
        text_mode="punctuation", variant=None, length_tools=False, dry_run=False,
        timeout=300.0, max_tokens=None, gen_batch_size=6, review_batch_size=12,
        max_repairs=1, max_context_chars=20000,
    )
    return args, source, profile_path


def test_prepare_is_offline_and_preserves_source_fields(tmp_path: Path) -> None:
    args, source, _ = _fixture(tmp_path)
    prepared = prepare_run(args)
    assert prepared.input_format == "json"
    assert prepared.items[0]["unknown"] == {"keep": True}
    assert prepared.items[0]["index"] == 77
    assert prepared.target_ids == ["cue-0"]
    assert not args.output_dir.exists()
    assert prepared.model == ModelSpec("opencode", "openai/gpt-5.6-luna", "medium")
    assert prepared.input_bytes_sha256
    assert source.read_bytes().startswith(b"\xef\xbb\xbf")
    assert prepared.metadata["input"]["sha256"] == prepared.input_bytes_sha256
    assert "input_hash" not in prepared.metadata["input"]
    assert "sha256" not in prepared.metadata["calibration"]
    assert "profile_hash" not in prepared.metadata["calibration"]


def test_prepare_zero_calibration_keeps_forecast_and_excludes_already_short_text(tmp_path: Path) -> None:
    args, source, profile_path = _fixture(tmp_path, text="Да.")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["char_linear_guard_sec"] = 2.0
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    before = source.read_bytes()
    prepared = prepare_run(args)
    plan = prepared.plans[0]
    assert plan.calibrated_max_chars == 0 and plan.max_chars == 19
    assert plan.status == "fits" and plan.short_timing_risk
    assert prepared.target_ids == []  # No forced generation for text already within target.
    assert source.read_bytes() == before
    assert not args.output_dir.exists()


def test_dry_run_does_not_construct_live_components(tmp_path: Path, monkeypatch: Any) -> None:
    args, _, _ = _fixture(tmp_path)
    args.dry_run = True
    monkeypatch.setattr("utils.shorten_file.LiveBackend", lambda *_a, **_k: pytest.fail("backend constructed"))
    monkeypatch.setattr("utils.shorten_file.verify_original_timing", lambda *_a, **_k: pytest.fail("inference"))
    assert main([str(args.input), "--output-dir", str(args.output_dir), "--model", args.model,
                 "--profile", str(args.profile), "--model-path", str(args.model_path), "--dry-run"]) == 0
    report = json.loads((args.output_dir / "preflight.json").read_text(encoding="utf-8"))
    assert report["run_state"] == "dry_run"
    assert report["selected_count"] == 1
    assert "No LLM" in report["evidence_statement"]
    assert not list(args.output_dir.glob(".preflight.json.*.tmp"))


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--budget-usd", "10", "--budget-usd is for the primary OpenRouter route"),
        ("--max-requests", "1", "max-requests is for the primary OpenRouter route"),
    ],
)
def test_legacy_cost_guards_fail_before_server_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    message: str,
) -> None:
    args, _, _ = _fixture(tmp_path)
    monkeypatch.setattr("utils.shorten_file.CleanOpenCodeServer", lambda *_a, **_k: pytest.fail("server constructed"))
    monkeypatch.setattr("utils.shorten_file.LiveBackend", lambda *_a, **_k: pytest.fail("backend constructed"))
    status = main([
        str(args.input), "--output-dir", str(args.output_dir), "--model", args.model,
        "--profile", str(args.profile), "--model-path", str(args.model_path), option, value,
    ])
    assert status == 1
    assert message in capsys.readouterr().err
    assert not args.output_dir.exists()


def test_primary_default_request_cap_remains_unchanged(tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    args.model = None
    args.dry_run = True
    prepared = prepare_run(args)
    assert prepared.primary is True
    assert prepared.metadata["budget"]["max_requests"] == 1000


def test_profile_identity_mismatch_fails_before_output(tmp_path: Path) -> None:
    args, _, profile_path = _fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["identity"]["voice"] = 7
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "change",
    [
        ("identity", "model_fingerprint", "wrong-model"),
        ("identity", "normalizer", "wrong-normalizer"),
        ("identity", "normalizer_sha256", "wrong-normalizer-hash"),
        ("top", "normalizer", "wrong-normalizer"),
        ("top", "input_character_basis", "display_text"),
    ],
)
def test_profile_provenance_mismatches_fail_before_output(
    tmp_path: Path, change: tuple[str, str, str]
) -> None:
    args, _, profile_path = _fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    target, field, value = change
    (profile if target == "top" else profile["identity"])[field] = value
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_run(args)
    assert not args.output_dir.exists()


def test_report_profile_selects_numeric_rate(tmp_path: Path) -> None:
    args, _, profile_path = _fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_path.write_text(json.dumps({"schema_version": 1, "status": "complete", "profiles": {"1.0": profile, "1.5": profile}}), encoding="utf-8")
    prepared = prepare_run(args)
    assert prepared.profile["rate"] == 1.5


def test_bom_srt_is_loaded_and_malformed_block_is_refused(tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    srt = tmp_path / "source.sRt"
    srt.write_bytes("\ufeff1\n00:00:00,000 --> 00:00:01,000\nHello\n".encode("utf-8-sig"))
    args.input = srt
    prepared = prepare_run(args)
    assert prepared.input_format == "srt"
    assert prepared.items[0]["index"] == 1
    srt.write_bytes("1\nnot-a-timestamp\nHello\n".encode())
    with pytest.raises(ValueError):
        prepare_run(args)
    assert not args.output_dir.exists()


def test_malformed_srt_cli_keeps_block_diagnostics(tmp_path: Path, capsys: Any) -> None:
    args, _, _ = _fixture(tmp_path)
    args.input = tmp_path / "broken.srt"
    args.input.write_text("1\nnot-a-timestamp\nHello\n", encoding="utf-8")
    assert main([str(args.input), "--output-dir", str(args.output_dir), "--model", args.model,
                 "--profile", str(args.profile), "--model-path", str(args.model_path)]) == 1
    error = capsys.readouterr().err
    assert "SRTParseError" in error
    assert "Block #1" in error
    assert "Invalid timestamp" in error


def test_empty_exception_message_returns_cli_error(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr("utils.shorten_file.prepare_run", lambda _args: (_ for _ in ()).throw(ValueError()))
    assert main(["source.json", "--output-dir", str(tmp_path / "run"), "--model", MODEL_CHOICES[0],
                 "--profile", "profile.json"]) == 1
    error = capsys.readouterr().err
    assert "no error message" in error


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_primary_first_signal_requests_graceful_stop_without_cancelling(
    tmp_path: Path, monkeypatch: Any, signum: signal.Signals,
) -> None:
    prepared = SimpleNamespace(primary=True, dry_run=False, run_time_limit_sec=None, output_dir=tmp_path / "fresh")
    monkeypatch.setattr("utils.shorten_file.prepare_run", lambda _args: prepared)
    initial_handler = signal.getsignal(signum)

    async def fake_execute(_prepared: Any, *, stop_requested: Any) -> dict[str, Any]:
        assert not stop_requested()
        signal.raise_signal(signum)
        assert stop_requested()
        await asyncio.sleep(0)
        return {"run_state": "incomplete", "summary": {}, "outcome_summary": {}, "budget": {}}

    monkeypatch.setattr("utils.shorten_file.execute_run", fake_execute)
    assert main(["fake.json", "--profile", "profile.json", "--output-dir", str(tmp_path / "fresh")]) == 1
    assert signal.getsignal(signum) == initial_handler


def test_primary_deadline_checked_without_background_watchdog(tmp_path: Path, monkeypatch: Any) -> None:
    prepared = SimpleNamespace(primary=True, dry_run=False, run_time_limit_sec=0.001, output_dir=tmp_path / "fresh")
    monkeypatch.setattr("utils.shorten_file.prepare_run", lambda _args: prepared)
    async def fake_execute(_prepared: Any, *, stop_requested: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        assert stop_requested()
        return {"run_state": "incomplete", "summary": {}, "outcome_summary": {}, "budget": {}}
    monkeypatch.setattr("utils.shorten_file.execute_run", fake_execute)
    assert main(["fake.json", "--profile", "profile.json", "--output-dir", str(tmp_path / "fresh")]) == 1


def test_second_signal_cancels_in_flight_primary(tmp_path: Path, monkeypatch: Any) -> None:
    prepared = SimpleNamespace(primary=True, dry_run=False, run_time_limit_sec=None, output_dir=tmp_path / "fresh")
    monkeypatch.setattr("utils.shorten_file.prepare_run", lambda _args: prepared)
    async def fake_execute(_prepared: Any, *, stop_requested: Any) -> dict[str, Any]:
        signal.raise_signal(signal.SIGINT)
        assert stop_requested()
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(0)
        pytest.fail("second signal did not cancel the task")
    monkeypatch.setattr("utils.shorten_file.execute_run", fake_execute)
    assert main(["fake.json", "--profile", "profile.json", "--output-dir", str(tmp_path / "fresh")]) == 130


def test_invalid_top_level_and_existing_output_are_rejected(tmp_path: Path) -> None:
    args, source, _ = _fixture(tmp_path)
    source.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        prepare_run(args)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        prepare_run(args)


def test_live_same_model_and_preserved_json_with_mock_transport(tmp_path: Path, monkeypatch: Any) -> None:
    args, _, _ = _fixture(tmp_path)
    prepared = prepare_run(args)
    seen: list[ModelSpec] = []

    class FakeBackend:
        async def __call__(self, spec: ModelSpec, system: str, user: str, schema: dict[str, Any], allow: bool) -> DetailedResponse:
            seen.append(spec)
            if "verdict" in json.dumps(schema):
                payload = {"results": [{"cue_id": "cue-0", "verdict": "accept", "issues": [], "advice": ""}]}
            else:
                payload = {"results": [{"cue_id": "cue-0", "text": "short", "status": "ok"}]}
            return DetailedResponse(json.dumps(payload), [])

    monkeypatch.setattr("utils.shorten_file._LazyRequest._make_backend", lambda self: FakeBackend())
    report = asyncio.run(execute_run(prepared))
    assert report["run_state"] == "completed"
    assert seen and all(spec == prepared.model for spec in seen)
    exported = json.loads((args.output_dir / "subtitles.json").read_text(encoding="utf-8"))
    assert exported[0]["unknown"] == {"keep": True}
    assert exported[0]["index"] == 77 and exported[0]["start"] == 0


def test_deepseek_does_not_construct_opencode(monkeypatch: Any, tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    args.model = "deepseek/deepseek-flash"
    prepared = prepare_run(args)
    monkeypatch.setattr("utils.shorten_file.LiveBackend", lambda server=None, **kw: object())
    lazy = __import__("utils.shorten_file", fromlist=["_LazyRequest"])._LazyRequest(prepared)
    backend = lazy._make_backend()
    assert backend is not None and lazy.server is None


def test_lazy_opencode_cleanup_on_interrupted_start(monkeypatch: Any, tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    prepared = prepare_run(args)
    state: dict[str, Any] = {}

    class FakeServer:
        def __init__(self, **kwargs: Any) -> None:
            state["kwargs"] = kwargs
            state["instance"] = self

        def start(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            state["stopped"] = True

    monkeypatch.setattr("utils.shorten_file.CleanOpenCodeServer", FakeServer)
    lazy = _LazyRequest(prepared)
    with pytest.raises(KeyboardInterrupt):
        lazy._make_backend()
    lazy.close()
    assert state["stopped"] is True
    assert state["kwargs"]["timeout"] == 45.0
    assert state["kwargs"]["required_models"] == {prepared.model.model: {prepared.model.variant}}
    assert lazy.server is None and lazy.backend is None and not lazy.server_started


def test_numeric_and_variant_controls_fail_offline(tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    args.timeout = float("nan")
    with pytest.raises(ValueError, match="timeout"):
        prepare_run(args)
    args.timeout = 300.0
    args.model = "openai/gpt-5.6-sol"
    args.variant = "none"
    with pytest.raises(ValueError, match="variant"):
        prepare_run(args)


@pytest.mark.parametrize("rate", [0.9, 1.6, float("nan"), float("inf")])
def test_unsupported_rates_fail_before_output(tmp_path: Path, rate: float) -> None:
    args, _, _ = _fixture(tmp_path)
    args.rate = rate
    with pytest.raises(ValueError, match="rate"):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_nonpositive_timeout_fails_before_output(tmp_path: Path, timeout: float) -> None:
    args, _, _ = _fixture(tmp_path)
    args.timeout = timeout
    with pytest.raises(ValueError, match="timeout"):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("field", ["gen_batch_size", "review_batch_size", "max_context_chars"])
def test_invalid_batch_sizes_fail_before_output(tmp_path: Path, field: str) -> None:
    args, _, _ = _fixture(tmp_path)
    setattr(args, field, 0)
    with pytest.raises(ValueError, match=field):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("max_tokens", [0, -1, 384001, True])
def test_invalid_token_limits_fail_before_output(tmp_path: Path, max_tokens: Any) -> None:
    args, _, _ = _fixture(tmp_path)
    args.max_tokens = max_tokens
    with pytest.raises(ValueError, match="max_tokens"):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("required_name", ["model.onnx", "dictionary", "config.json"])
def test_model_required_directories_are_not_accepted_as_files(tmp_path: Path, required_name: str) -> None:
    args, _, _ = _fixture(tmp_path)
    required_path = args.model_path / required_name
    required_path.unlink()
    required_path.mkdir()
    with pytest.raises(ValueError, match=required_name):
        prepare_run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "running"},
        {"schema_version": 2},
    ],
)
def test_incomplete_or_wrong_schema_profile_report_fails_before_output(tmp_path: Path, change: dict[str, Any]) -> None:
    args, _, profile_path = _fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = {"schema_version": 1, "status": "complete", "profiles": {"1.5": profile}}
    report.update(change)
    profile_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration report"):
        prepare_run(args)
    assert not args.output_dir.exists()


def test_profile_report_rejects_missing_rate_and_duplicate_numeric_keys(tmp_path: Path) -> None:
    args, _, profile_path = _fixture(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.pop("rate")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="rate"):
        prepare_run(args)
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    args, _, profile_path = _fixture(duplicate_dir)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_path.write_text(json.dumps({"schema_version": 1, "status": "complete", "profiles": {
        "1.5": profile, "1.50": profile,
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_run(args)
    assert not args.output_dir.exists()


def test_short_timing_preflight_fit_needs_no_backend(tmp_path: Path, monkeypatch: Any) -> None:
    args, source, _ = _fixture(tmp_path, text="one two")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    profile["models"]["char_linear"]["coefficients"] = [0.0, 1.0]
    args.profile.write_text(json.dumps(profile), encoding="utf-8")
    source.write_bytes(b"\xef\xbb\xbf" + json.dumps([{
        "index": 77, "start": 0, "end": 1000, "text": "one two",
    }]).encode())
    prepared = prepare_run(args)
    assert prepared.target_ids == ["cue-0"]
    async def fail_backend(*_: Any, **__: Any) -> DetailedResponse:
        raise AssertionError("backend must not be called")
    monkeypatch.setattr("utils.shorten_file._LazyRequest._make_backend", lambda self: fail_backend)
    monkeypatch.setattr("utils.shorten_file.verify_original_timing", lambda plans, output_dir, **kw: {
        plan.cue_id: {"original": plan.text, "available_ms": plan.available_ms, "duration_sec": 0.1}
        for plan in plans
    })
    report = asyncio.run(execute_run(prepared))
    assert report["summary"]["actual_fits_count"] == 1
    assert not (args.output_dir / "requests.jsonl").exists()


def test_incomplete_run_keeps_checkpoint_and_returns_incomplete(tmp_path: Path, monkeypatch: Any) -> None:
    args, _, _ = _fixture(tmp_path)
    prepared = prepare_run(args)
    class ErrorBackend:
        async def __call__(self, *_: Any, **__: Any) -> DetailedResponse:
            return DetailedResponse(None, [], "transport")
    monkeypatch.setattr("utils.shorten_file._LazyRequest._make_backend", lambda self: ErrorBackend())
    report = asyncio.run(execute_run(prepared))
    assert report["run_state"] == "incomplete"
    assert (args.output_dir / "report.json").exists()
    assert (args.output_dir / "subtitles.json").exists()


@pytest.mark.parametrize("flags", [["--retry-unknown", "id"], ["--resume", "--dry-run"], ["--resume"]])
def test_resume_cli_invalid_flag_combinations_fail_offline(tmp_path: Path, flags: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, _ = _fixture(tmp_path)
    monkeypatch.setattr("utils.shorten_file.LiveBackend", lambda *a, **k: pytest.fail("backend constructed"))
    assert main([str(args.input), "--output-dir", str(args.output_dir), "--model", args.model,
                 "--profile", str(args.profile), "--model-path", str(args.model_path), *flags]) != 0
    assert not args.output_dir.exists()


@pytest.mark.parametrize("change", ["unknown", "source_bytes", "profile"])
def test_resume_cli_checks_real_prepared_contract_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], change: str,
) -> None:
    args, source, profile_path = _fixture(tmp_path)
    async def cancel(*a: Any, **k: Any) -> DetailedResponse:
        raise asyncio.CancelledError
    monkeypatch.setattr(_LazyRequest, "__call__", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute_run(prepare_run(args)))
    before = (args.output_dir / "requests.jsonl").read_bytes()
    request_id = json.loads(before.splitlines()[0])["request_id"]
    if change == "source_bytes":
        source.write_bytes(source.read_bytes() + b"\n")
    elif change == "profile":
        profile = json.loads(profile_path.read_text())
        profile["models"]["char_linear"]["coefficients"] = [0.0, 0.2]
        profile_path.write_text(json.dumps(profile))
    async def forbidden(*a: Any, **k: Any) -> DetailedResponse:
        pytest.fail("resume sent a new request")
    monkeypatch.setattr(_LazyRequest, "__call__", forbidden)
    status = main([str(source), "--output-dir", str(args.output_dir), "--model", args.model,
                   "--profile", str(profile_path), "--model-path", str(args.model_path), "--resume"])
    assert status == (2 if change == "unknown" else 1)
    error = capsys.readouterr().err
    if change == "unknown":
        assert request_id in error and "--retry-unknown" in error
    assert (args.output_dir / "requests.jsonl").read_bytes() == before
