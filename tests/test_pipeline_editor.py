"""Offline full-editor cycle with real synthetic WAV processing/mixing, no inference."""

from __future__ import annotations

import copy
import json
import re
import wave
from pathlib import Path
from typing import Any

import pytest

from utils import pipeline_batch as batch
from utils import pipeline_editor as editor
from utils.pipeline_text import json_sha256, text_sha256


def write_wav(path: Path, frames: int = 400) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(1000)
        stream.writeframes(b"\x10\x00" * frames)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, list[str]]:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {"offline": True})
    monkeypatch.setattr(batch, "publish_report", lambda folder, state: None)
    calls: list[str] = []

    class Engine:
        def __init__(self, config: dict[str, Any]) -> None:
            assert config["tts"]["voice"] == 8

        def normalize(self, cue: dict[str, Any]) -> str:
            if cue["text"] == "BAD":
                raise ValueError("RUNorm lexical change: крут")
            return cue["text"]

        def synthesize(self, text: str, path: Path, inputs: list[str]) -> None:
            calls.append(text)
            inputs.append(batch.normalize_vosk_text(text))
            write_wav(path)

    monkeypatch.setattr(batch, "AudioEngine", Engine)
    source = tmp_path / "source.json"
    batch.save_json(source, [{"index": index, "start": position * 2000, "end": position * 2000 + 1000,
                             "text": text, "gender": "female", "custom": position}
                            for position, (index, text) in enumerate(zip([31, 7, 99, 4],
                                ["Первая очень длинная исходная реплика для авторской проверки.",
                                 "Да.", "Третья.", "Четвёртая."]))])
    config = tmp_path / "config.json"
    batch.save_json(config, {"tts": {"voice": 8, "base_speech_rate": 1.25, "max_total_speed": 1.5},
                            "reduce_silence": {"enabled": False}, "resample": {"enabled": False},
                            "mix": {"sample_rate": 1000, "channels": 1}})
    folder = tmp_path / "editor"
    editor.create_project(source, folder, config)
    return folder, source, config, calls


def approve_all(folder: Path) -> dict[str, Any]:
    state = editor.load_project(folder)
    return editor.approve_cues(folder, [cue["cue_id"] for cue in state["cues"]], expected_revision=state["revision"])


def partial_batch(source: Path, config: Path, start: int = 0, end: int = 1) -> Path:
    folder = source.parent / f"batch-{start}-{end}"
    state = batch.prepare(source, folder, editor.load_pipeline_config(config), start, end)
    request = state["requests"]["1"]
    if request["entries"]:
        result = {key: copy.deepcopy(request[key]) for key in
                  ("round", "request_sha256", "binding_sha256", "source_sha256", "plan_sha256", "policy")}
        result.update(schema="pipeline-batch-reviewed-v1", execution="native_live", entries=[])
        proposed = copy.deepcopy(state)
        for row in request["entries"]:
            text = "Коротко."
            result["entries"].append({"cue_id": row["cue_id"], "original": row["original"], "candidate": text,
                "writer": {"model": "mock", "mode": "unit", "invocation_id": "writer"},
                "review": {"verdict": "accept", "text_sha256": text_sha256(text),
                           "request_sha256": request["request_sha256"], "model": "mock", "mode": "unit",
                           "invocation_id": "reviewer"}})
            next(cue for cue in proposed["cues"] if cue["cue_id"] == row["cue_id"])["chosen_text"] = text
        result["assembly_review"] = {"verdict": "accept", "assembly_sha256": json_sha256(batch.assembled(proposed)),
                                     "request_sha256": request["request_sha256"], "model": "mock", "mode": "unit",
                                     "invocation_id": "reviewer"}
        path = folder / "synthetic_review.json"
        batch.save_json(path, result)
        batch.ingest_reviewed(folder, path)
    batch.audio(folder)
    return folder


def test_full_cycle_partial_native_srt_one_edit_selective_audio_and_full_mix(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, source, config, calls = workspace
    original_source = source.read_bytes()
    native_folder = partial_batch(source, config)
    before_batch = {str(path): path.read_bytes() for path in native_folder.rglob("*") if path.is_file()}
    state = editor.adopt_batch(folder, native_folder)
    assert [cue["review_status"] for cue in state["cues"]] == ["native_approved", "native_approved", "pending", "pending"]
    assert len(state["cues"]) == 4 and [cue["index"] for cue in state["cues"]] == [31, 7, 99, 4]
    assert not state["ready_for_mix"]
    unchanged = Path(state["cues"][0]["audio_path"])
    unchanged_hash = batch.file_sha256(unchanged)
    exports = editor.export_project(folder)
    assert [cue["index"] for cue in editor._items(Path(exports["srt_path"]))] == [31, 7, 99, 4]
    metadata = batch.read_json(Path(exports["json_path"]))
    assert metadata["baseline_source_sha256"] == state["baseline_source_sha256"]
    assert metadata["cues"][3]["needs_human"] and metadata["cues"][3]["audio_path"] is None
    edited = source.parent / "edited.srt"
    edited.write_text(Path(exports["srt_path"]).read_text().replace("Да.", "Исправлено."), encoding="utf-8")
    state = editor.import_srt(folder, edited, expected_revision=state["revision"])
    assert state["cues"][1]["status"] == "needs_review" and state["cues"][1]["audio_path"] is None
    assert state["cues"][1]["gender"] == "female"
    count = len(calls)
    editor.process_pending(folder)
    assert len(calls) == count  # New text and remaining source need explicit approval.
    approve_all(folder)
    state = editor.process_pending(folder)
    assert calls[count:] == ["Исправлено.", "Третья.", "Четвёртая."]
    assert state["ready_for_mix"] and batch.file_sha256(unchanged) == unchanged_hash
    count = len(calls)
    revision = state["revision"]
    assert editor.process_pending(folder)["revision"] == revision and len(calls) == count
    output = Path(editor.mix_ready(folder))
    with wave.open(str(output)) as stream:
        assert stream.getnframes() == 7000 and stream.getframerate() == 1000
    assert source.read_bytes() == original_source
    assert before_batch == {str(path): path.read_bytes() for path in native_folder.rglob("*") if path.is_file()}
    assert not list(folder.rglob("*.html"))


def test_timing_neighbor_invalidated_raw_replayed_not_resynthesized(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    original_raw = Path(state["cues"][0]["attempts"][-1]["raw"]["path"])
    original_hash = batch.file_sha256(original_raw)
    previous_count = len(calls)
    state = editor.apply_edits(folder, [{"cue_id": "cue-1", "start": 1500}], expected_revision=state["revision"])
    assert state["cues"][0]["available_ms"] == 1300
    assert [cue["status"] for cue in state["cues"][:2]] == ["pending_recheck", "pending_recheck"]
    assert all(cue["review_status"] == "pending" for cue in state["cues"][:2])
    approve_all(folder)
    state = editor.process_pending(folder)
    assert len(calls) == previous_count and state["ready_for_mix"]
    assert batch.file_sha256(original_raw) == original_hash
    assert state["cues"][0]["attempts"][-1]["reused_raw_sha256"] == original_hash
    assert Path(editor.mix_ready(folder)).is_file()


def test_metadata_only_and_noops_preserve_pcm_and_approval(workspace: tuple[Path, Path, Path, list[str]]) -> None:
    folder, _, _, calls = workspace
    state = approve_all(folder)
    state = editor.process_pending(folder)
    assert editor.apply_edits(folder, [{"cue_id": "cue-0", "text": state["cues"][0]["text"]}],
                              expected_revision=state["revision"])["revision"] == state["revision"]
    old_path = state["cues"][0]["audio_path"]
    new = editor.apply_edits(folder, [{"cue_id": "cue-0", "speaker_id": "person-A"}], expected_revision=state["revision"])
    assert new["revision"] != state["revision"] and new["ready_for_mix"]
    assert new["cues"][0]["audio_path"] == old_path and new["cues"][0]["review_status"] == "human_approved"
    assert len(calls) == 4
    with pytest.raises(ValueError, match="Stale"):
        editor.apply_edits(folder, [], expected_revision=state["revision"])


@pytest.mark.parametrize("edits", [
    [{"cue_id": "cue-0", "index": 2}], [{"cue_id": "unknown", "text": "X"}],
    [{"cue_id": "cue-0"}, {"cue_id": "cue-0"}], [{"cue_id": "cue-1", "start": 500}],
    [{"cue_id": "cue-0", "start": -1}], [{"cue_id": "cue-0", "end": 0}],
    [{"cue_id": "cue-0", "start": 0.1}], [{"cue_id": "cue-0", "text": ""}],
    [{"cue_id": "cue-0", "speaker_id": 8}],
])
def test_bad_edits_atomic(workspace: tuple[Path, Path, Path, list[str]], edits: list[dict[str, Any]]) -> None:
    folder, _, _, _ = workspace
    state = editor.load_project(folder)
    before = (folder / "project.json").read_bytes()
    with pytest.raises(ValueError):
        editor.apply_edits(folder, edits, expected_revision=state["revision"])
    assert (folder / "project.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reorder", "foreign", "malformed", "negative_time"])
def test_incomplete_foreign_srt_atomic(workspace: tuple[Path, Path, Path, list[str]], mutation: str) -> None:
    folder, source, _, _ = workspace
    state = editor.load_project(folder)
    exported = Path(editor.export_project(folder)["srt_path"]).read_text()
    blocks = exported.strip().split("\n\n")
    if mutation == "missing":
        blocks.pop()
    elif mutation == "duplicate":
        blocks[1] = blocks[0]
    elif mutation == "reorder":
        blocks.reverse()
    elif mutation == "foreign":
        blocks[0] = blocks[0].replace("31\n", "123\n")
    elif mutation == "malformed":
        blocks[0] = blocks[0].replace("31\n", "cue31\n")
    else:
        blocks[0] = blocks[0].replace("00:00:00,000", "-00:00:00,100")
    edited = source.parent / "invalid.srt"
    edited.write_text("\n\n".join(blocks) + "\n")
    with pytest.raises(ValueError):
        editor.import_srt(folder, edited, expected_revision=state["revision"])
    assert editor.load_project(folder)["revision"] == state["revision"]


def test_failed_normalization_and_interruption_never_auto_retry(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, _, _, calls = workspace
    state = editor.load_project(folder)
    state = editor.apply_edits(folder, [{"cue_id": "cue-0", "text": "BAD"}], expected_revision=state["revision"])
    approve_all(folder)
    state = editor.process_pending(folder)
    assert state["cues"][0]["status"] == "normalization_failed" and len(calls) == 3
    editor.process_pending(folder)
    assert len(calls) == 3
    state = editor.apply_edits(folder, [{"cue_id": "cue-0", "text": "Исправлено."}], expected_revision=state["revision"])
    approve_all(folder)
    original = batch.AudioEngine.normalize

    def interrupt(self: Any, cue: dict[str, Any]) -> str:
        raise KeyboardInterrupt()

    monkeypatch.setattr(batch.AudioEngine, "normalize", interrupt)
    with pytest.raises(KeyboardInterrupt):
        editor.process_pending(folder)
    state = editor.load_project(folder)
    assert state["cues"][0]["status"] == "needs_reconciliation"
    monkeypatch.setattr(batch.AudioEngine, "normalize", original)
    editor.process_pending(folder)
    assert len(calls) == 3
    state = editor.retry_cues(folder, ["cue-0"], expected_revision=state["revision"])
    assert editor.process_pending(folder)["ready_for_mix"] and len(calls) == 4


@pytest.mark.parametrize("mutation", ["missing", "changed", "duration", "speed", "approval", "coverage", "config"])
def test_mix_fails_closed_current_audio_and_full_coverage(
    workspace: tuple[Path, Path, Path, list[str]], mutation: str,
) -> None:
    folder, _, _, _ = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    raw_state = batch.read_json(folder / "project.json")
    cue = raw_state["cues"][0]
    attempt = cue["attempts"][-1]
    if mutation == "missing":
        Path(cue["attempts"][-1]["final"]["path"]).unlink()
    elif mutation == "changed":
        write_wav(Path(attempt["final"]["path"]), 401)
    elif mutation == "duration":
        path = Path(attempt["final"]["path"])
        write_wav(path, 1900)
        attempt["final"] = batch.wav_info(path)
        attempt["wav_sha256"][str(path)] = batch.file_sha256(path)
    elif mutation == "speed":
        attempt["effective_factor"] = 2
        attempt["effective_total_speed"] = 2.5
    elif mutation == "approval":
        cue["human_approval"] = {}
    elif mutation == "coverage":
        raw_state["cues"].pop()
    else:
        raw_state["config"]["tts"]["voice"] = 9
    if mutation not in {"missing", "changed"}:
        editor._persist(folder, raw_state)
    with pytest.raises((ValueError, FileNotFoundError)):
        editor.mix_ready(folder)
    assert not list(folder.rglob("voiceover.wav"))
    assert state["ready_for_mix"]  # A saved status is not trusted by the mix gate.


def test_runtime_change_blocks_audio_adopt_mix_but_draft_remains_visible(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {"changed": True})
    assert not editor.load_project(folder)["runtime_valid"]
    assert Path(editor.export_project(folder)["srt_path"]).is_file()
    for action in (editor.process_pending, editor.mix_ready):
        with pytest.raises(ValueError, match="Runtime/model assets changed"):
            action(folder)
    assert not calls


def test_disjoint_adoption_and_human_conflict_no_merely_fits_approval(workspace: tuple[Path, Path, Path, list[str]]) -> None:
    folder, source, config, _ = workspace
    first = partial_batch(source, config)
    state = editor.adopt_batch(folder, first)
    assert editor.adopt_batch(folder, first)["revision"] == state["revision"]
    second = partial_batch(source, config, 2, 3)
    state = editor.adopt_batch(folder, second)
    assert all(cue["status"] == "fits" for cue in state["cues"])
    assert [cue["review_status"] for cue in state["cues"][2:]] == ["pending", "pending"]
    with pytest.raises(ValueError, match="approval"):
        editor.mix_ready(folder)
    another = source.parent / "human-conflict"
    state = editor.create_project(source, another, config)
    state = editor.apply_edits(another, [{"cue_id": "cue-0", "text": "Человек."}], expected_revision=state["revision"])
    with pytest.raises(ValueError, match="overwrite human"):
        editor.adopt_batch(another, first)


def test_new_project_freeze_original_and_cli_no_inference(
    workspace: tuple[Path, Path, Path, list[str]], capsys: Any,
) -> None:
    folder, source, config, calls = workspace
    with pytest.raises(FileExistsError):
        editor.create_project(source, folder, config)
    source.write_text("[]")  # The editor owns an immutable copy, not the original file.
    before = (folder / "project.json").read_bytes()
    editor.main(["status", "--project", str(folder)])
    assert len(json.loads(capsys.readouterr().out)["cues"]) == 4
    editor.main(["export", "--project", str(folder)])
    assert Path(json.loads(capsys.readouterr().out)["srt_path"]).is_file()
    assert before == (folder / "project.json").read_bytes() and not calls
    frozen = folder / "config.json"
    frozen.write_text("{}")
    with pytest.raises(ValueError, match="Frozen config"):
        editor.load_project(folder)


def test_srt_initialization_and_editor_writer_preserve_position_order(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, source, config, _ = workspace
    srt = Path(editor.export_project(folder)["srt_path"])
    other = source.parent / "from-srt"
    state = editor.create_project(srt, other, config)
    assert [cue["index"] for cue in state["cues"]] == [31, 7, 99, 4]
    exported = Path(editor.export_project(other)["srt_path"])
    assert editor.import_srt(other, exported, expected_revision=state["revision"])["revision"] == state["revision"]


def test_foreign_and_stale_batch_rejected_without_import(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, source, config, _ = workspace
    native = partial_batch(source, config)
    before = editor.load_project(folder)["revision"]
    with batch.locked(native):
        state = batch.load_state(native)
        state["binding"]["runtime"] = {"old": True}
        state["binding_sha256"] = json_sha256(state["binding"])
        batch.persist(native, state)
    with pytest.raises(ValueError, match="Runtime/model assets changed"):
        editor.adopt_batch(folder, native)
    assert editor.load_project(folder)["revision"] == before and not (folder / "imports").exists()
    # Restoring runtime with different full origin is still not an adoption permit.
    foreign_source = source.parent / "foreign.json"
    items = batch.read_json(source)
    items[2]["text"] = "Другой исходник."
    batch.save_json(foreign_source, items)
    foreign = batch.prepare(foreign_source, source.parent / "foreign-batch", editor.load_pipeline_config(config), 2, 3)
    assert foreign["range"] == [2, 3]
    with pytest.raises(ValueError, match="full origin"):
        editor.adopt_batch(folder, source.parent / "foreign-batch")
    assert editor.load_project(folder)["revision"] == before


def test_writer_assembly_rejection_does_not_become_approval_from_audio_fit(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, source, config, _ = workspace
    native = partial_batch(source, config)
    with batch.locked(native):
        state = batch.load_state(native)
        state["imports"]["1"]["result"]["assembly_review"]["verdict"] = "revise"
        batch.persist(native, state)
    state = editor.adopt_batch(folder, native)
    assert [cue["status"] for cue in state["cues"][:2]] == ["fits", "fits"]
    assert all(cue["review_status"] == "pending" for cue in state["cues"])


def test_explicit_audio_uses_real_pause_and_edge_processing_with_stubbed_vad(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils import silero_pauses as pauses
    import numpy as np

    _, source, config, calls = workspace
    settings = batch.read_json(config)
    settings.update(silero_pauses={"enabled": True}, silero_edges={"enabled": True})
    batch.save_json(config, settings)
    detections: list[int] = []
    initializations: list[int] = []

    def init(self: pauses.PauseProcessor) -> None:
        initializations.append(1)
        self.identity = {"policy": pauses.POLICY, "fixture": True}

    def detect(self: pauses.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        detections.append(len(audio))
        return []

    monkeypatch.setattr(pauses.PauseProcessor, "__init__", init)
    monkeypatch.setattr(pauses.PauseProcessor, "detect", detect)
    folder = source.parent / "with-pauses"
    editor.create_project(source, folder, config)
    editor.process_pending(folder)
    assert initializations == [] and detections == [] and calls == []
    approve_all(folder)
    state = editor.process_pending(folder)
    assert state["ready_for_mix"] and len(calls) == 4 and len(detections) == 4
    assert initializations == [1]
    editor.process_pending(folder)
    assert initializations == [1] and len(detections) == 4
    for cue in state["cues"]:
        attempt = cue["attempts"][-1]
        assert attempt["silero_edges"]["available_ms"] == cue["available_ms"]
        assert attempt["silero_edges"]["chosen_text_sha256"] == text_sha256(cue["text"])
        assert set(attempt) >= {"raw", "pretempo", "pauses", "edges", "final"}


def test_retiming_too_small_rechecks_raw_without_tts_and_blocks_mix(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    count = len(calls)
    state = editor.apply_edits(folder, [{"cue_id": "cue-3", "end": 6100}], expected_revision=state["revision"])
    approve_all(folder)
    state = editor.process_pending(folder)
    assert len(calls) == count and state["cues"][3]["status"] == "unresolved"
    assert state["cues"][3]["duration_ms"] == 400 and state["cues"][3]["audio_path"] is not None
    with pytest.raises(ValueError, match="measured fit"):
        editor.mix_ready(folder)


def test_single_writer_lock_and_cli_bulk_approval(
    workspace: tuple[Path, Path, Path, list[str]], capsys: Any,
) -> None:
    folder, _, _, calls = workspace
    state = editor.load_project(folder)
    with batch.locked(folder):
        with pytest.raises(ValueError, match="another process"):
            editor.approve_cues(folder, ["cue-0"], expected_revision=state["revision"])
    editor.main(["approve", "--project", str(folder), "--all", "--expected-revision", state["revision"]])
    state = json.loads(capsys.readouterr().out)
    assert all(cue["review_status"] == "human_approved" for cue in state["cues"]) and not calls


def test_real_native_ui_backend_integration_full_srt_selective_queue_and_mix(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    """Run real UI callbacks/backend; only expensive inference and report publication are stubbed."""
    from ui import native_editor as ui

    initial_folder, source_json, config, calls = workspace
    source = source_json.parent / "integration_source.srt"
    source.write_text(Path(editor.export_project(initial_folder)["srt_path"]).read_text(), encoding="utf-8")
    original = source.read_bytes()
    folder = source.parent / "ui-editor"
    ui.use_backend_for_tests(None)
    assert ui.get_backend() is editor  # Not the fake backend used in UI unit tests.

    def rendered(result: tuple[Any, ...]) -> dict[str, Any]:
        assert len(result) == 10
        assert not result[1].startswith("Ошибка"), result[1]
        assert result[0]["revision"] == result[0]["project"]["revision"]
        return result[0]

    state = rendered(ui.on_create(None, str(source), str(folder), str(config), {"only_attention": True}))
    assert state["visible_ids"] == ["cue-0", "cue-1", "cue-2", "cue-3"]
    assert [row[0] for row in ui.cue_rows(ui.project_cues(state["project"]))] == [31, 7, 99, 4]
    assert not calls

    native = partial_batch(source, config)
    assert calls == ["Коротко.", "Да."]
    batch_bytes = {str(path): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    state = rendered(ui.on_adopt(str(folder), str(native), state))
    assert state["visible_ids"] == ["cue-2", "cue-3"]
    first_audio = Path(state["project"]["cues"][0]["audio_path"])
    first_hash = batch.file_sha256(first_audio)

    export_message, files_update = ui.on_export(state)
    assert "черновик" in export_message and len(files_update["value"]) == 2
    exported_srt, exported_json = [Path(path) for path in files_update["value"]]
    assert [item["index"] for item in editor._items(exported_srt)] == [31, 7, 99, 4]
    assert len(batch.read_json(exported_json)["cues"]) == 4
    edited = source.parent / "integration_edited.srt"
    edited.write_text(exported_srt.read_text().replace("Да.", "Изменена одна реплика."), encoding="utf-8")
    state = rendered(ui.on_import_srt(str(edited), state))
    assert state["visible_ids"] == ["cue-1", "cue-2", "cue-3"]
    assert state["project"]["cues"][1]["status"] == "needs_review"
    state = rendered(ui.on_select("cue-1", state))
    assert state["cue_id"] == "cue-1" and ui.find_cue(state["project"], state["cue_id"])["index"] == 7
    state = rendered(ui.on_approve(state))
    state = rendered(ui.on_approve_all(True, state))
    assert len(calls) == 2  # Create/import/approval/export never synthesize.

    process_updates = list(ui.on_process(state))
    assert len(process_updates) == 2 and all(len(update) == 10 for update in process_updates)
    state = rendered(process_updates[-1])
    assert calls == ["Коротко.", "Да.", "Изменена одна реплика.", "Третья.", "Четвёртая."]
    assert state["project"]["ready_for_mix"] and state["visible_ids"] == [] and state["cue_id"] is None
    assert batch.file_sha256(first_audio) == first_hash
    for cue in state["project"]["cues"]:
        assert Path(cue["audio_path"]).is_file() and cue["duration_ms"] == 400

    state = rendered(ui.on_filter(False, state))
    assert state["visible_ids"] == ["cue-0", "cue-1", "cue-2", "cue-3"]
    selected = ui.on_select("cue-1", state)
    state = rendered(selected)
    assert selected[5:9] == ("Изменена одна реплика.", 2000, 3000, "")
    assert Path(selected[9]["value"]).is_file()
    hidden = ui.on_filter(True, state)
    assert rendered(hidden)["cue_id"] is None

    mix_message, mix_update = ui.on_mix(state)
    assert "Озвучка собрана" in mix_message and Path(mix_update["value"]).is_file()
    with wave.open(mix_update["value"]) as stream:
        assert stream.getnframes() == 7000 and stream.getframerate() == 1000
    saved_revision = state["revision"]
    state = rendered(list(ui.on_process(state))[-1])
    assert state["revision"] == saved_revision and len(calls) == 5
    export_message, final_update = ui.on_export(state)
    assert not export_message.startswith("Ошибка")
    metadata = batch.read_json(Path(final_update["value"][1]))
    assert metadata["ready_for_mix"] and all(not cue["needs_human"] for cue in metadata["cues"])
    assert source.read_bytes() == original and batch.file_sha256(first_audio) == first_hash
    assert batch_bytes == {str(path): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    assert not list(folder.rglob("*.html"))


@pytest.mark.parametrize("error", [RuntimeError("Expected silero-vad==6.2.3; found 6.2.4"),
                                  FileNotFoundError("Missing runtime code asset: pipeline.py")])
def test_unavailable_runtime_keeps_drafts_accessible_and_strict_actions_blocked(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    folder, source, config, calls = workspace
    native = partial_batch(source, config)
    before = (folder / "project.json").read_bytes()
    count = len(calls)

    def unavailable(config: dict[str, Any]) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(batch, "runtime_binding", unavailable)
    state = editor.load_project(folder)
    assert state["runtime_valid"] is False and state["runtime_reason"] == str(error)
    assert all(cue["reason"] == str(error) and cue["audio_path"] is None for cue in state["cues"])
    exported = editor.export_project(folder)
    assert len(editor._items(Path(exported["srt_path"]))) == 4
    assert batch.read_json(Path(exported["json_path"]))["runtime_reason"] == str(error)
    for action in (lambda: editor.process_pending(folder), lambda: editor.adopt_batch(folder, native),
                   lambda: editor.mix_ready(folder)):
        with pytest.raises(type(error), match=re.escape(str(error))):
            action()
    assert before == (folder / "project.json").read_bytes() and len(calls) == count
    assert not (folder / "imports").exists()


def test_failed_batch_copy_is_unadopted_and_retry_preserves_orphans_and_user_files(
    workspace: tuple[Path, Path, Path, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, source, config, _ = workspace
    native = partial_batch(source, config)
    digest = batch.load_state(native)["state_sha256"]
    user_folder = folder / "imports" / digest
    user_folder.mkdir(parents=True)
    user_file = user_folder / "user.txt"
    user_file.write_text("Preserve this unrelated historical file.")
    native_before = {str(path): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    project_before = (folder / "project.json").read_bytes()
    copyfile = editor.shutil.copyfile
    copied: list[Path] = []

    def fail_after_first_wav(source: Any, destination: Any) -> Any:
        result = copyfile(source, destination)
        if Path(destination).suffix == ".wav":
            copied.append(Path(destination))
            if len(copied) == 1:
                raise OSError("Injected import copy failure after first WAV")
        return result

    monkeypatch.setattr(editor.shutil, "copyfile", fail_after_first_wav)
    with pytest.raises(OSError, match="Injected import copy failure"):
        editor.adopt_batch(folder, native)
    assert (folder / "project.json").read_bytes() == project_before
    unchanged = editor.load_project(folder)
    assert unchanged["imports"] == {} and all(cue["review_status"] == "pending" for cue in unchanged["cues"])
    assert all(cue["audio_path"] is None for cue in unchanged["cues"])
    assert len(copied) == 1 and copied[0].is_file()
    orphan_before = {str(path): path.read_bytes() for path in copied[0].parents[2].rglob("*") if path.is_file()}
    monkeypatch.setattr(editor.shutil, "copyfile", copyfile)
    adopted = editor.adopt_batch(folder, native)
    assert adopted["cues"][0]["review_status"] == "native_approved"
    assert Path(adopted["cues"][0]["audio_path"]).is_file()
    assert user_file.read_text() == "Preserve this unrelated historical file."
    assert all(Path(path).read_bytes() == data for path, data in orphan_before.items())
    assert native_before == {str(path): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    directories = sorted(path.name for path in (folder / "imports").iterdir())
    assert editor.adopt_batch(folder, native)["revision"] == adopted["revision"]
    assert sorted(path.name for path in (folder / "imports").iterdir()) == directories


def test_missing_final_explicit_retry_replays_raw_without_new_tts_or_deleting_history(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    count = len(calls)
    cue = state["cues"][0]
    old_final = Path(cue["audio_path"])
    old_raw = cue["attempts"][-1]["raw"]
    old_final.unlink()
    remaining = {str(path): path.read_bytes() for path in old_final.parent.rglob("*") if path.is_file()}
    state = editor.load_project(folder)
    assert state["cues"][0]["status"] == "audio_invalid"
    state = editor.retry_cues(folder, ["cue-0"], expected_revision=state["revision"])
    assert state["cues"][0]["status"] == "pending_retry"
    state = editor.process_pending(folder)
    assert len(calls) == count and state["ready_for_mix"]
    assert Path(state["cues"][0]["audio_path"]).is_file() and state["cues"][0]["audio_path"] != str(old_final)
    assert state["cues"][0]["attempts"][-1]["reused_raw_sha256"] == old_raw["sha256"]
    assert all(Path(path).read_bytes() == data for path, data in remaining.items())
    before = (folder / "project.json").read_bytes()
    with pytest.raises(ValueError, match="valid current fit"):
        editor.retry_cues(folder, ["cue-0"], expected_revision=state["revision"])
    assert (folder / "project.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_invalid_completed_raw_cannot_authorize_silent_resynthesis(
    workspace: tuple[Path, Path, Path, list[str]], damage: str,
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    count = len(calls)
    raw = Path(state["cues"][0]["attempts"][-1]["raw"]["path"])
    if damage == "missing":
        raw.unlink()
    else:
        write_wav(raw, 401)
    before = (folder / "project.json").read_bytes()
    with pytest.raises(ValueError, match="Immutable RAW changed/missing"):
        editor.retry_cues(folder, ["cue-0"], expected_revision=state["revision"])
    assert (folder / "project.json").read_bytes() == before
    state = editor.process_pending(folder)
    assert state["cues"][0]["status"] == "audio_invalid" and len(calls) == count


def test_raw_changed_after_retry_fails_closed_before_tts(
    workspace: tuple[Path, Path, Path, list[str]],
) -> None:
    folder, _, _, calls = workspace
    approve_all(folder)
    state = editor.process_pending(folder)
    count = len(calls)
    cue = state["cues"][0]
    Path(cue["audio_path"]).unlink()
    state = editor.retry_cues(folder, ["cue-0"], expected_revision=state["revision"])
    write_wav(Path(cue["attempts"][-1]["raw"]["path"]), 401)
    state = editor.process_pending(folder)
    assert state["cues"][0]["status"] == "audio_failed" and len(calls) == count
    assert "Immutable RAW changed/missing" in state["cues"][0]["reason"]


@pytest.mark.parametrize("text", ["Привет\n \nПока", "\nПривет", "Привет\n", " \nПривет", "Привет\n ",
                                  "Привет\r\n \r\nПока", "Привет\r\n", "Привет\r \rПока"])
def test_empty_multiline_srt_lines_rejected_by_real_ui_and_json_creation_atomically(
    workspace: tuple[Path, Path, Path, list[str]], text: str,
) -> None:
    from ui import native_editor as ui

    folder, source, config, calls = workspace
    ui.use_backend_for_tests(None)
    state = ui.on_open(str(folder), {"only_attention": True})[0]
    state = ui.on_select("cue-0", state)[0]
    before = (folder / "project.json").read_bytes()
    result = ui.on_save(text, 0, 1000, "", state)
    assert len(result) == 10 and result[0] == state
    assert result[1].startswith("Ошибка") and "empty SRT lines" in result[1]
    assert (folder / "project.json").read_bytes() == before

    items = batch.read_json(source)
    items[0]["text"] = text
    invalid_source = source.parent / "invalid_multiline.json"
    batch.save_json(invalid_source, items)
    target = source.parent / "invalid_multiline_project"
    with pytest.raises(ValueError, match="empty SRT lines"):
        editor.create_project(invalid_source, target, config)
    assert not target.exists() and not calls


@pytest.mark.parametrize("text", ["Привет\nПока", "Привет\r\nПока"])
def test_valid_multiline_text_exports_and_imports_complete_srt(
    workspace: tuple[Path, Path, Path, list[str]], text: str,
) -> None:
    folder, _, _, calls = workspace
    state = editor.load_project(folder)
    state = editor.apply_edits(folder, [{"cue_id": "cue-0", "text": text}], expected_revision=state["revision"])
    exported = editor.export_project(folder)
    parsed = editor._items(Path(exported["srt_path"]))
    assert [cue["index"] for cue in parsed] == [31, 7, 99, 4]
    assert parsed[0]["text"] == ["Привет", "Пока"]
    assert batch.read_json(Path(exported["json_path"]))["cues"][0]["text"] == text
    imported = editor.import_srt(folder, Path(exported["srt_path"]), expected_revision=state["revision"])
    assert len(imported["cues"]) == 4 and imported["cues"][0]["text"] == "Привет Пока"
    assert not calls
