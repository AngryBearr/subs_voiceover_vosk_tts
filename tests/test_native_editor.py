"""Unit tests for the minimal native editor UI: fake backend, no server, no inference.

The backend contract is faked in memory; the real ``utils.pipeline_editor`` is
never imported here. ``build_demo`` is exercised against the installed Gradio
only to validate the component/event structure (no launch, no network).
"""

from __future__ import annotations

import copy
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import native_editor as ne  # noqa: E402

TMP_ROOT = Path("/tmp/opencode")


def make_cue(cue_id: str, index: int, start: int, end: int, text: str, *, status: str = "pending_audio",
             review_status: str = "pending", needs_human: bool | None = True, audio_path: str | None = None,
             available_ms: int | None = 2000) -> dict[str, Any]:
    cue: dict[str, Any] = {"cue_id": cue_id, "index": index, "start": start, "end": end, "text": text,
                           "speaker_id": None, "status": status, "review_status": review_status,
                           "reason": status, "available_ms": available_ms, "audio_path": audio_path,
                           "duration_ms": None}
    if needs_human is not None:
        cue["needs_human"] = needs_human
    return cue


def make_project(revision: str = "rev-1") -> dict[str, Any]:
    return {"revision": revision, "ready_for_mix": False, "runtime_valid": True,
            "cues": [make_cue("cue-0", 0, 0, 1000, "Первая реплика", status="pending_audio",
                              review_status="pending"),
                     make_cue("cue-1", 1, 1200, 2200, "Вторая реплика", status="fits",
                              review_status="human_approved", needs_human=False)]}


class FakeBackend:
    """In-memory stand-in for utils.pipeline_editor; records every call and payload."""

    def __init__(self, tmp: Path, project: dict[str, Any] | None = None) -> None:
        self.tmp = tmp
        self.project = project or make_project()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: dict[str, Exception] = {}
        self._sequence = 0

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def _bump(self, label: str) -> str:
        self._sequence += 1
        self.project["revision"] = f"{label}-{self._sequence}"
        return str(self.project["revision"])

    def _record(self, name: str, **payload: Any) -> None:
        self.calls.append((name, payload))
        if name in self.fail:
            raise self.fail[name]

    def load_project(self, folder: Path) -> dict[str, Any]:
        self._record("load_project", folder=str(folder))
        return copy.deepcopy(self.project)

    def create_project(self, source: Path, folder: Path, config: Path) -> dict[str, Any]:
        self._record("create_project", source=str(source), folder=str(folder), config=str(config))
        return copy.deepcopy(self.project)

    def adopt_batch(self, folder: Path, batch_folder: Path) -> dict[str, Any]:
        self._record("adopt_batch", folder=str(folder), batch=str(batch_folder))
        return copy.deepcopy(self.project)

    def import_srt(self, folder: Path, srt_path: Path, *, expected_revision: str) -> dict[str, Any]:
        self._record("import_srt", folder=str(folder), srt=str(srt_path), expected_revision=expected_revision)
        self.project["cues"][0]["text"] = "Из SRT"
        self._bump("rev-import")
        return copy.deepcopy(self.project)

    def apply_edits(self, folder: Path, edits: list[dict[str, Any]], *,
                    expected_revision: str) -> dict[str, Any]:
        self._record("apply_edits", folder=str(folder), edits=copy.deepcopy(edits),
                     expected_revision=expected_revision)
        for edit in edits:
            cue = next(cue for cue in self.project["cues"] if cue["cue_id"] == edit["cue_id"])
            cue.update({key: value for key, value in edit.items() if key != "cue_id"})
            cue.update(audio_path=None, duration_ms=None, status="needs_review", review_status="pending",
                       needs_human=True)
        self._bump("rev-edit")
        return copy.deepcopy(self.project)

    def approve_cues(self, folder: Path, cue_ids: list[str], *, expected_revision: str) -> dict[str, Any]:
        self._record("approve_cues", folder=str(folder), cue_ids=list(cue_ids),
                     expected_revision=expected_revision)
        for cue in self.project["cues"]:
            if cue["cue_id"] in cue_ids:
                cue.update(review_status="human_approved", needs_human=False)
        self._bump("rev-approve")
        return copy.deepcopy(self.project)

    def process_pending(self, folder: Path) -> dict[str, Any]:
        self._record("process_pending", folder=str(folder))
        self._bump("rev-audio")
        return copy.deepcopy(self.project)

    def export_project(self, folder: Path) -> dict[str, str]:
        self._record("export_project", folder=str(folder))
        srt, meta = self.tmp / "subtitles.srt", self.tmp / "project.json"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nТест\n", encoding="utf-8")
        meta.write_text("{}", encoding="utf-8")
        return {"srt_path": str(srt), "json_path": str(meta)}

    def mix_ready(self, folder: Path) -> str:
        self._record("mix_ready", folder=str(folder))
        output = self.tmp / "voiceover.wav"
        output.write_bytes(b"RIFF0000WAVE")
        return str(output)


@pytest.fixture()
def workdir() -> Iterator[Path]:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="native_editor_test_", dir=str(TMP_ROOT)))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def backend(workdir: Path) -> Iterator[FakeBackend]:
    fake = FakeBackend(workdir)
    ne.use_backend_for_tests(fake)
    try:
        yield fake
    finally:
        ne.use_backend_for_tests(None)


def opened(backend: FakeBackend, folder: str = "/tmp/opencode/project", *, only_attention: bool = False,
           cue_id: str | None = None) -> dict[str, Any]:
    project = backend.load_project(Path(folder))
    visible_ids = [cue["cue_id"] for cue in ne.filtered_cues(project, only_attention)]
    return {"folder": folder, "revision": project["revision"], "project": project, "cue_id": cue_id,
            "only_attention": only_attention, "visible_ids": visible_ids}


def test_filter_uses_needs_human_flag_with_status_fallback() -> None:
    project = {"revision": "rev-1", "cues": [
        make_cue("c0", 0, 0, 100, "a", status="fits", review_status="pending", needs_human=True),
        make_cue("c1", 1, 200, 300, "b", status="fits", review_status="human_approved", needs_human=False),
        make_cue("c2", 2, 400, 500, "c", status="weird_status", review_status="pending", needs_human=None),
        make_cue("c3", 3, 600, 700, "d", status="pending_audio", review_status="pending", needs_human=False),
    ]}
    assert [cue["cue_id"] for cue in ne.filtered_cues(project, True)] == ["c0", "c2"]
    assert ne.cue_rows(ne.filtered_cues(project, True))[1][4] == "weird_status"
    assert "требуют внимания 2" in ne.summary_text(project)


def test_select_and_save_edit_the_selected_cue_by_id(backend: FakeBackend) -> None:
    state = opened(backend)
    backend.calls.clear()
    selected = ne.on_select("cue-1", state)
    assert len(selected) == 10
    assert selected[0]["cue_id"] == "cue-1"
    assert selected[5] == "Вторая реплика"
    assert selected[7] == 2200
    result = ne.on_save("Новый текст", 1200, 2200, "", selected[0])
    assert len(result) == 10
    name, payload = backend.calls[-1]
    assert name == "apply_edits"
    assert payload["edits"] == [{"cue_id": "cue-1", "text": "Новый текст", "start": 1200, "end": 2200,
                                 "speaker_id": None}]
    assert payload["expected_revision"] == "rev-1"
    assert "сохранена" in result[1]


def test_save_conflict_keeps_form_and_asks_reload(backend: FakeBackend) -> None:
    state = opened(backend, cue_id="cue-1")
    backend.fail["apply_edits"] = ValueError("Stale expected_revision; reload before editing")
    backend.project["revision"] = "rev-2"
    backend.calls.clear()
    result = ne.on_save("Правка", 1200, 2200, "", state)
    assert len(result) == 10
    assert "конфликт" in result[1] and "«Открыть»" in result[1] and "не применялись" in result[1]
    assert result[0] is state
    assert all(result[index] == gr.update() for index in range(2, 10))
    assert backend.names() == ["apply_edits", "load_project"]


def test_open_error_keeps_current_form(backend: FakeBackend) -> None:
    state = opened(backend, cue_id="cue-1")
    backend.fail["load_project"] = RuntimeError("boom")
    backend.calls.clear()
    result = ne.on_open("/tmp/opencode/broken", state)
    assert result[0] is state and "Ошибка" in result[1]
    assert all(result[index] == gr.update() for index in range(2, 10))


def test_save_without_selection_makes_no_backend_call(backend: FakeBackend) -> None:
    state = opened(backend)
    backend.calls.clear()
    result = ne.on_save("x", 0, 10, "", state)
    assert "Выберите кью" in result[1]
    assert backend.calls == []


def test_save_validation_rejects_empty_text_and_bad_timing(backend: FakeBackend) -> None:
    state = opened(backend, cue_id="cue-1")
    backend.calls.clear()
    assert "пустым" in ne.on_save("   ", 0, 10, "", state)[1]
    assert "больше начала" in ne.on_save("ok", 10, 10, "", state)[1]
    assert "числами" in ne.on_save("ok", "abc", 10, "", state)[1]
    assert backend.calls == []


def test_select_rejects_a_cue_id_hidden_by_the_filter(backend: FakeBackend) -> None:
    state = opened(backend, only_attention=True)  # visible_ids == ["cue-0"]
    backend.calls.clear()
    rejected = ne.on_select("cue-1", state)
    assert len(rejected) == 10 and rejected[0] is state
    assert "недоступен" in rejected[1] and "фильтр" in rejected[1]
    assert all(rejected[index] == gr.update() for index in range(2, 10))
    accepted = ne.on_select("cue-0", state)
    assert accepted[0]["cue_id"] == "cue-0" and accepted[5] == "Первая реплика"


def test_filter_hiding_the_selection_clears_it_and_blocks_wrong_edit(backend: FakeBackend) -> None:
    state = opened(backend, only_attention=False, cue_id="cue-1")
    backend.calls.clear()
    result = ne.on_filter(True, state)
    assert result[0]["cue_id"] is None
    assert "скрыт" in result[1]
    assert result[5] == "" and result[7] is None and result[9] == gr.update(value=None)
    assert backend.calls == []
    blocked = ne.on_save("Старый текст", 0, 10, "", result[0])
    assert "Выберите кью" in blocked[1]
    assert backend.calls == []


def test_import_srt_sends_file_and_revision_to_backend(backend: FakeBackend) -> None:
    state = opened(backend)
    backend.calls.clear()
    result = ne.on_import_srt("/tmp/opencode/edited.srt", state)
    name, payload = backend.calls[-1]
    assert name == "import_srt"
    assert payload["srt"] == "/tmp/opencode/edited.srt"
    assert payload["expected_revision"] == "rev-1"
    assert result[2][0][3] == "Из SRT"
    assert result[0]["revision"].startswith("rev-import")


def test_approvals_are_explicit_and_batch_respects_confirmation(backend: FakeBackend) -> None:
    state = opened(backend, cue_id="cue-0")
    backend.calls.clear()
    without_confirmation = ne.on_approve_all(False, state)
    assert backend.calls == [] and "галочку" in without_confirmation[1]

    assert "Выберите кью" in ne.on_approve({**state, "cue_id": None})[1]
    assert backend.calls == []

    approved = ne.on_approve(state)
    name, payload = backend.calls[-1]
    assert name == "approve_cues" and payload["cue_ids"] == ["cue-0"]
    assert payload["expected_revision"] == "rev-1" and "подтверждён" in approved[1]

    backend.calls.clear()
    batch = ne.on_approve_all(True, state)
    name, payload = backend.calls[-1]
    assert name == "approve_cues" and payload["cue_ids"] == ["cue-0"]
    assert "1" in batch[1]


def test_save_and_approve_uses_the_saved_revision(backend: FakeBackend) -> None:
    state = opened(backend, cue_id="cue-1")
    backend.calls.clear()
    result = ne.on_save_approve("Исправленный текст", 1200, 2200, "SPK1", state)
    assert backend.names() == ["apply_edits", "approve_cues"]
    assert backend.calls[0][1]["edits"][0]["speaker_id"] == "SPK1"
    assert backend.calls[1][1]["cue_ids"] == ["cue-1"]
    assert backend.calls[1][1]["expected_revision"] == "rev-edit-1"
    assert "подтверждён" in result[1]

    backend.fail["approve_cues"] = RuntimeError("binding mismatch")
    backend.calls.clear()
    partial = ne.on_save_approve("Ещё текст", 1200, 2200, "", state)
    assert "Правка сохранена" in partial[1] and "подтверждение" in partial[1]
    assert partial[0]["revision"].startswith("rev-edit")


def test_process_is_explicit_only_and_generator_streams_status(backend: FakeBackend) -> None:
    state = opened(backend)
    backend.calls.clear()
    ne.on_open("/tmp/opencode/project", state)
    ne.on_save("Текст", 0, 1000, "", {**state, "cue_id": "cue-0"})
    assert "process_pending" not in backend.names()

    fresh = opened(backend)  # revision from the fake project after the fake save
    backend.calls.clear()
    stream = ne.on_process(fresh)
    first = next(stream)
    assert len(first) == 10 and "TTS" in first[1]
    rest = list(stream)
    assert backend.names() == ["load_project", "process_pending"]
    assert "обработана" in rest[-1][1]


def test_missing_audio_is_none_and_existing_audio_is_bound(backend: FakeBackend, workdir: Path) -> None:
    sound = workdir / "cue.wav"
    sound.write_bytes(b"RIFF0000WAVE")
    backend.project["cues"][0]["audio_path"] = str(sound)
    state = opened(backend)
    with_audio = ne.on_select("cue-0", state)
    assert with_audio[9] == gr.update(value=str(sound))
    assert ne.on_select("cue-1", state)[9] == gr.update(value=None)

    backend.calls.clear()
    saved = ne.on_save("Другой текст", 0, 1000, "", with_audio[0])
    assert saved[9] == gr.update(value=None)  # no stale audio after a text edit


def test_export_returns_full_files_and_mix_separates_voiceover(backend: FakeBackend, workdir: Path) -> None:
    state = opened(backend)
    backend.calls.clear()
    message, files = ne.on_export(state)
    assert "Экспорт не означает готовность" in message
    assert files == gr.update(value=[str(workdir / "subtitles.srt"), str(workdir / "project.json")])
    assert (workdir / "subtitles.srt").is_file() and (workdir / "project.json").is_file()

    not_ready_message, not_ready_file = ne.on_mix(state)
    assert backend.names() == ["export_project"]
    assert "не все кью" in not_ready_message and not_ready_file == gr.update(value=None)

    backend.project["ready_for_mix"] = True
    state = opened(backend)  # refresh the cached view so the ready flag is visible
    backend.fail["mix_ready"] = RuntimeError("not ready")
    failed_message, failed_file = ne.on_mix(state)
    assert "не собрана" in failed_message and failed_file == gr.update(value=None)

    backend.fail.pop("mix_ready")
    state = opened(backend)
    ok_message, ok_file = ne.on_mix(state)
    assert ok_file == gr.update(value=str(workdir / "voiceover.wav")) and "отдельный файл" in ok_message


def test_create_and_adopt_pass_paths_and_guard_revision(backend: FakeBackend, workdir: Path) -> None:
    backend.calls.clear()
    created = ne.on_create(None, "/tmp/opencode/source.srt", str(workdir / "new"), "pipeline_config_vosk.json", {})
    assert backend.calls[-1] == ("create_project", {"source": "/tmp/opencode/source.srt",
                                                    "folder": str(workdir / "new"),
                                                    "config": "pipeline_config_vosk.json"})
    assert created[0]["folder"] == str(workdir / "new")
    assert "Укажите исходный" in ne.on_create(None, "", str(workdir / "new"), "", {})[1]

    state = opened(backend, "/tmp/opencode/project")
    backend.project["revision"] = "rev-2"
    backend.calls.clear()
    blocked = ne.on_adopt("/tmp/opencode/project", "/tmp/opencode/batch", state)
    assert "Проект изменён" in blocked[1] and backend.names() == ["load_project"]

    backend.project["revision"] = state["revision"]
    backend.calls.clear()
    adopted = ne.on_adopt("/tmp/opencode/project", "/tmp/opencode/batch", state)
    assert "Batch импортирован" in adopted[1]
    assert backend.names() == ["load_project", "adopt_batch"]


def test_clear_text_is_local_only(backend: FakeBackend) -> None:
    backend.calls.clear()
    text, message = ne.on_clear_text("длинный текст")
    assert text == "" and "не запускалась" in message
    assert backend.calls == []


def test_build_demo_config_has_expected_components_and_events() -> None:
    demo = ne.build_demo()
    config = demo.get_config_file()
    buttons = [component["props"].get("value") for component in config["components"]
               if component["type"] == "button"]
    assert "Озвучить сейчас (локальная TTS, только по кнопке)" in buttons
    assert "Сохранить и подтвердить" in buttons
    labels = [component["props"].get("label") for component in config["components"]]
    assert "Только требующие внимания" in labels
    assert "Говорящий (ручная метка, не голос TTS)" in labels
    assert "Подтверждаю текст и тайминги всех требующих внимания" in labels
    assert {component["type"] for component in config["components"]} >= {"dataframe", "dropdown", "audio", "file"}

    dependencies = config["dependencies"]
    assert len(dependencies) == 14
    shapes = sorted((len(row["inputs"]), len(row["outputs"])) for row in dependencies)
    assert shapes == sorted([(5, 10), (2, 10), (3, 10), (2, 10), (2, 10), (2, 10), (5, 10), (5, 10),
                             (1, 10), (2, 10), (1, 2), (1, 10), (1, 2), (1, 2)])
    assert {"on_create", "on_save", "on_save_approve", "on_process", "on_export", "on_mix"} <= {
        row["api_name"] for row in dependencies}


def test_select_and_filter_listen_to_user_input_not_change() -> None:
    config = ne.build_demo().get_config_file()
    by_label = {component["props"].get("label"): component["id"] for component in config["components"]}
    events: dict[int, list[str]] = {}
    api_by_target: dict[tuple[int, str], str] = {}
    for row in config["dependencies"]:
        for target in row["targets"]:
            block_id, event = tuple(target)
            events.setdefault(block_id, []).append(event)
            api_by_target[(block_id, event)] = row["api_name"]
    selector_id = by_label["Кью (индекс — текст)"]
    filter_id = by_label["Только требующие внимания"]
    assert events[selector_id] == ["input"]  # human-only: programmatic refresh must not re-enter
    assert events[filter_id] == ["input"]
    assert api_by_target[(selector_id, "input")] == "on_select"
    assert api_by_target[(filter_id, "input")] == "on_filter"
    assert (selector_id, "change") not in api_by_target
