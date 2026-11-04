"""Voice choices and samples ("▶ Послушать") on the "Настройки" tab, and the order of the form."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ui import dub_app, settings, voice_preview


def _model_config(tmp_path: Path) -> Path:
    names = {"andrienko_aleksandr": 0, "archibasova_alina": 1, "chiffa": 9, "trofimova_ekaterina": 49,
             "zlatoustov_ivan": 56}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"speaker_id_map": names}), encoding="utf-8")
    return path


def test_vosk_choices_have_names_marks_and_gender_first(tmp_path: Path) -> None:
    config = _model_config(tmp_path)
    female = settings.vosk_voice_choices("female", config)
    assert female == [("1 · Archibasova Alina (жен.)", 1), ("49 · Trofimova Ekaterina (жен.)", 49), ("9 · Chiffa", 9),
                      ("0 · Andrienko Aleksandr (муж.)", 0), ("56 · Zlatoustov Ivan (муж.)", 56)]
    male = settings.vosk_voice_choices("male", config)
    assert [v for _, v in male] == [0, 56, 9, 1, 49]
    assert all(isinstance(v, int) for _, v in male)


def test_vosk_choices_without_the_model_are_plain_numbers(tmp_path: Path) -> None:
    choices = settings.vosk_voice_choices("male", tmp_path / "missing.json")
    assert len(choices) == settings.VOSK_SPEAKERS and choices[0] == ("0", 0)


def test_real_model_gender_marks() -> None:
    names = settings.vosk_speaker_names()
    if not names:
        pytest.skip("no Vosk model here")
    female = {n for _, n in settings.vosk_voice_choices("female") if n in settings.VOSK_FEMALE}
    assert female == settings.VOSK_FEMALE and len(names) == settings.VOSK_SPEAKERS
    assert settings.vosk_voice_choices("female")[0][1] in settings.VOSK_FEMALE


def test_a_chosen_vosk_voice_is_saved_as_an_integer(tmp_path: Path) -> None:
    setting = next(s for s in settings.SETTINGS if s.kind == "vosk_voice")
    assert settings.from_form(setting, 49) == 49 and isinstance(settings.from_form(setting, 49.0), int)
    with pytest.raises(ValueError):
        settings.from_form(setting, 99)
    assert settings.to_form(setting, 49) == 49


def test_settings_components_follow_the_order_of_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the layout, save and reset get the components in ``SETTINGS`` order."""
    app = dub_app.build()
    save = next(fn for fn in app.fns.values() if fn.fn is dub_app.save_settings)
    labels = [str(getattr(block, "label", "")) for block in save.inputs[3:]]  # after folder, results folder, target
    assert labels == [s.label for s in settings.SETTINGS]
    views = [fn for fn in app.fns.values() if fn.fn is dub_app.settings_view]
    assert views and all([str(getattr(b, "label", "")) for b in fn.outputs[:len(labels)]] == labels for fn in views)


def test_every_voice_setting_has_a_listen_button() -> None:
    app = dub_app.build()
    listens = [fn for fn in app.fns.values()
               if any(getattr(app.blocks.get(i), "value", None) == "▶ Послушать" for i, _ in fn.targets)]
    assert len(listens) == len(dub_app.VOICE_ENGINE) == 5
    voice_paths = {s.path for s in settings.SETTINGS if s.kind in ("voice", "vosk_voice")}
    assert voice_paths == set(dub_app.VOICE_ENGINE)


# ---------------------------------------------------------------- the sample

def test_preview_path_is_per_engine_voice_and_phrase(tmp_path: Path) -> None:
    path = voice_preview.preview_path("teratts", "ru_f1+ru_f2", tmp_path)
    assert path == tmp_path / "teratts" / f"ru_f1+ru_f2.{voice_preview.phrase_hash()}.wav"
    assert voice_preview.preview_path("vosk", 49, tmp_path).name.startswith("49.")
    assert voice_preview.preview_path("vosk", 49, tmp_path, "another") != voice_preview.preview_path("vosk", 49, tmp_path)


def test_command_runs_this_module_in_a_subprocess(tmp_path: Path) -> None:
    cmd = voice_preview.build_command("vosk", 49, tmp_path / "o.wav", "cpu")
    assert cmd == [sys.executable, "-m", "ui.voice_preview", "vosk", "49", str(tmp_path / "o.wav"), "--device", "cpu"]


def test_preview_makes_the_sample_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(cmd[5]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(voice_preview.subprocess, "run", fake_run)
    wav, message = voice_preview.preview("teratts", "ru_m5", root=tmp_path)
    assert wav == voice_preview.preview_path("teratts", "ru_m5", tmp_path) and wav.read_bytes() == b"RIFF"
    assert voice_preview.preview("teratts", "ru_m5", root=tmp_path)[0] == wav
    assert len(calls) == 1  # the second listen comes from the kept file
    assert not list((tmp_path / "teratts").glob(".*part*"))


def test_preview_errors_are_short_russian_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    error = "Traceback (most recent call last):\n  File \"x\", line 1\nValueError: TeraTTS voice not found: ru_x\n"
    monkeypatch.setattr(voice_preview.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", error))
    wav, message = voice_preview.preview("teratts", "ru_x", root=tmp_path)
    assert wav is None and message == "Не получилось озвучить образец: ValueError: TeraTTS voice not found: ru_x"
    assert "Traceback" not in message and not any(tmp_path.rglob("*.wav"))

    def timeout(cmd: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(voice_preview.subprocess, "run", timeout)
    assert voice_preview.preview("vosk", 3, root=tmp_path)[0] is None
    assert voice_preview.preview("vosk", "", root=tmp_path)[1] == "Сначала выберите голос."
    assert voice_preview.preview("piper", "x", root=tmp_path)[0] is None


def test_listen_voice_uses_the_form_value(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, Any]] = []
    monkeypatch.setattr(dub_app.voice_preview, "preview", lambda engine, voice: seen.append((engine, voice)) or (Path("/x.wav"), "ok"))
    path = ("voiceover", "engines", "vosk", "voices", "male")
    assert dub_app.listen_voice(path, 12) == ("/x.wav", "ok") and seen == [("vosk", 12)]
    assert dub_app.listen_voice(("voiceover", "host_voice"), "")[0] is None


def test_all_vosk_samples_are_made_with_one_model_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ui import voice_preview

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        folder = Path(cmd[cmd.index("--output-folder") + 1])
        for voice in cmd[cmd.index("--voices") + 1].split(","):
            (folder / f"v{voice}.wav").write_bytes(b"RIFF")
        return type("Done", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(voice_preview.subprocess, "run", fake_run)
    root = tmp_path / "voice_previews"
    voice_preview.preview_path("vosk", 3, root).parent.mkdir(parents=True)
    voice_preview.preview_path("vosk", 3, root).write_bytes(b"kept")
    made = voice_preview.prepare_vosk_voices(root)
    assert made == 56 and len(calls) == 1  # one process for every voice not kept yet
    assert "3" not in calls[0][calls[0].index("--voices") + 1].split(",")
    assert voice_preview.prepare_vosk_voices(root) == 0 and len(calls) == 1


def test_a_click_waits_for_the_background_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import threading

    root = tmp_path / "voice_previews"
    target = voice_preview.preview_path("vosk", 7, root)
    target.parent.mkdir(parents=True)
    monkeypatch.setattr(voice_preview.subprocess, "run", lambda *a, **k: pytest.fail("a second model load"))
    voice_preview.VOSK_PREPARING.set()
    try:
        threading.Timer(0.3, lambda: target.write_bytes(b"RIFF")).start()  # the background run makes it
        wav, message = voice_preview.preview("vosk", 7, root=root)
    finally:
        voice_preview.VOSK_PREPARING.clear()
    assert wav == target and message == "Готово."


def test_the_click_says_at_once_what_happens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(voice_preview, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(voice_preview, "preview_path", lambda engine, voice, root=tmp_path: tmp_path / f"{voice}.wav")
    button, note, player = dub_app._listen_started(("voiceover", "engines", "teratts", "voices", "male"), "ru_m5")
    assert button["interactive"] is False and "⏳" in note["value"] and "15 с" in note["value"]
    (tmp_path / "ru_m5.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(voice_preview, "preview", lambda engine, voice: (tmp_path / "ru_m5.wav", "Готово."))
    button, note, player = dub_app._listen_outputs(("voiceover", "engines", "teratts", "voices", "male"), "ru_m5")
    assert button["interactive"] is True and note["value"].startswith("✅") and player["visible"] is True
