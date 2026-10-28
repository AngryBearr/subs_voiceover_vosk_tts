from __future__ import annotations

import json
from pathlib import Path

from ui import dub_app


def test_job_config_applies_the_form(monkeypatch) -> None:
    monkeypatch.setattr(dub_app, "base_config", lambda: {"llm": {"model": "openai/gpt-6.1-sol", "opencode": ["openai/*"],
                                                                 "router": "orcarouter"}, "voiceover": {}})
    via_api = dub_app.job_config("Быстрый: только Whisper", "Sol: лучше перевод, дороже", True, True, True, True,
                                 transport="API OpenRouter")
    assert via_api["llm"] == {"model": "openai/gpt-6.1-sol", "opencode": [], "router": "openrouter"}
    subscription = dub_app.job_config("Быстрый: только Whisper", "openai/gpt-6.1-sol", True, True, True, True,
                                      transport="Подписка (opencode)")
    assert subscription["llm"]["opencode"] == ["openai/*"] and subscription["llm"]["router"] == "orcarouter"
    config = dub_app.job_config("Быстрый: только Whisper", "openai/gpt-6-luna", subtitles=False, duck_voices=False,
                                stress=False, refine=False)
    assert config["mode"] == "fast" and config["llm"]["model"] == "openai/gpt-6-luna"
    assert config["mix"]["subtitles"] is False and config["mix"]["duck"] == "all"
    assert config["voiceover"]["stress_check"] is False and config["speaker_refine"]["model"] == "none"
    assert config["voiceover"]["engines"]["f5"]["nfe"] == 32
    fast = dub_app.job_config("Быстрый: только Whisper", "openai/gpt-6-luna", True, True, True, True, f5_fast=True)
    assert fast["voiceover"]["engines"]["f5"]["nfe"] == 7


def test_videos_and_progress(tmp_path: Path) -> None:
    for name in ("b.mkv", "a.mp4", "notes.txt"):
        (tmp_path / name).write_text("")
    assert dub_app.list_videos(str(tmp_path)) == ["a.mp4", "b.mkv"]
    out = tmp_path / "out"
    stages = out / "a" / ".stages"
    stages.mkdir(parents=True)
    for mark in ("transcribe", "glossary", "dub.vosk"):
        (stages / mark).write_text("")
    log = tmp_path / "job.log"
    log.write_text("INFO stt.run: [a] speakers ...\n")
    job = dub_app.Job(files=[tmp_path / "a.mp4"], out_root=out, engines=["vosk", "teratts"],
                      config_path=tmp_path / "c.json", log_path=log, engine="vosk")
    assert dub_app.progress_rows(job) == [["a", "⏳ 3/9: спикеры", "2/9"]]
    for mark in ("speakers", "translate", "stress", "voiceover.vosk", "check.vosk", "mix.vosk"):
        (stages / mark).write_text("")
    (out / "a" / "a.ru.vosk.mkv").write_text("")
    job.state = "готово"
    assert dub_app.progress_rows(job)[0][1] == "✅ готово"
    assert dub_app.results_rows(job) == [["a", "vosk", str(out / "a" / "a.ru.vosk.mkv")]]
    # one video with every voice-over, in the folder of finished videos (paths.finished/<season folder>)
    done = tmp_path / "done" / f"STT_{tmp_path.name}"
    done.mkdir(parents=True)
    (done / "a.ru.mkv").write_text("")
    (out / "a" / "a.ru.vosk.m4a").write_text("")
    job.config_path.write_text(json.dumps({"paths": {"finished": str(tmp_path / "done"), "finished_folder": "STT_{season}"}}))
    assert dub_app.results_rows(job) == [["a", "vosk", str(done / "a.ru.mkv")]]


def test_voice_bank_rows(tmp_path: Path) -> None:
    import json

    from stt.voice_bank import VoiceBank

    bank = VoiceBank(tmp_path / "S1.voices.json")
    bank.add_episode("ep1", [{"id": "S01", "name": "Julie McGee", "label_ru": "Джули", "gender": "female",
                              "speech_s": 120.0, "llm_confidence": 0.9, "embedding": [1.0, 0.0]}], [])
    bank.save()
    assert dub_app.list_banks(str(tmp_path)) == ["S1.voices.json"]
    assert dub_app.people_rows(str(tmp_path), "S1.voices.json") == [["Julie McGee", "Джули", "female", 1, 2.0, "нет"]]
    refs = dub_app.refs_dir(str(tmp_path), "S1.voices.json")
    refs.mkdir()
    (refs / "references.json").write_text(json.dumps([{"name": "Julie McGee", "wav": "Julie_McGee.wav",
                                                        "episode": "S29E01x", "start": 1.0, "end": 9.0, "score": 0.93,
                                                        "music_db": -10.0, "text": "I'm dating John Rocker"}]))
    audio, info = dub_app.person_view(str(tmp_path), "S1.voices.json", "Julie McGee")
    assert audio.endswith("Julie_McGee.wav") and "S29E01" in info and "0.93" in info


def test_review_rows_and_corrections(tmp_path: Path) -> None:
    import json

    episode = tmp_path / "ep"
    episode.mkdir()
    (episode / "review.json").write_text(json.dumps([{"line": 19, "time": "00:01:00,265", "speaker": "Джефф",
                                                      "voice": "male", "sounds_like": "John Rocker (0.64)",
                                                      "text_ru": "Как фамилия?"}]))
    (episode / "speakers.json").write_text(json.dumps([{"id": "S02", "name": "John Rocker", "gender": "male",
                                                        "cues": 3}]))
    assert dub_app.list_reviewed(str(tmp_path)) == ["ep"]
    assert dub_app.review_rows(str(tmp_path), "ep")[0][:5] == [19, "00:01:00", "Джефф", "male",
                                                              "похоже на John Rocker (0.64)"]
    assert "1 правок" in dub_app.add_correction(str(tmp_path), "ep", 19, {"speaker": "S02"})
    dub_app.add_correction(str(tmp_path), "ep", 19, {"text": "Как твоя фамилия?"})
    assert json.loads((episode / "corrections.json").read_text()) == {"19": {"speaker": "S02",
                                                                             "text": "Как твоя фамилия?"}}
    assert dub_app.review_rows(str(tmp_path), "ep")[0][6].startswith('{"speaker": "S02"')
    assert dub_app.speaker_choices(str(tmp_path), "ep") == ["S02: John Rocker (male)"]


def test_status_warns_when_the_subscription_login_fails(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "job.log"
    job = dub_app.Job(files=[], out_root=tmp_path, engines=["f5"], config_path=tmp_path / "c.json", log_path=log)
    monkeypatch.setattr(dub_app, "JOB", job)
    log.write_text("2026 INFO stt.run: [ep] asr ...\n", encoding="utf-8")
    assert dub_app.SUBSCRIPTION_WARNING not in dub_app.refresh()[0]
    log.write_text("WARNING stt.llm: LLM openai/gpt-6.1-sol via opencode failed (token expired); using orcarouter\n",
                   encoding="utf-8")
    assert dub_app.SUBSCRIPTION_WARNING in dub_app.refresh()[0]


def test_settings_keep_only_changes_and_other_personal_keys(tmp_path: Path) -> None:
    from ui import settings

    defaults = json.loads((dub_app.REPO_ROOT / "stt_config.json").read_text(encoding="utf-8"))
    (tmp_path / "stt_config.json").write_text(json.dumps(defaults), encoding="utf-8")
    local = tmp_path / "stt_config.local.json"
    local.write_text(json.dumps({"llm": {"opencode": ["openai/*"]}, "mode": "fast"}), encoding="utf-8")
    values = settings.form_values({**defaults, "mode": "fast"})
    by_path = {s.path: n for n, s in enumerate(settings.SETTINGS)}
    assert values[by_path[("mode",)]] == "Быстрый: только Whisper"
    values[by_path[("mode",)]] = "Точный: Cohere + Whisper + LLM"  # back to the default: leaves the local file
    values[by_path[("mix", "subtitles")]] = False
    values[by_path[("voiceover", "engines", "vosk", "voices", "female")]] = 12
    own, warnings = settings.save(values, tmp_path / "stt_config.json", local)
    assert own == 2 and warnings == []
    saved = json.loads(local.read_text(encoding="utf-8"))
    assert saved["llm"]["opencode"] == ["openai/*"] and "mode" not in saved
    assert saved["mix"] == {"subtitles": False} and saved["voiceover"]["engines"]["vosk"]["voices"] == {"female": 12}
    assert (tmp_path / "stt_config.local.json.bak").exists()
    settings.reset(local)
    assert json.loads(local.read_text(encoding="utf-8")) == {"llm": {"opencode": ["openai/*"]}}


def test_settings_reject_bad_values() -> None:
    from ui import settings

    vosk = next(s for s in settings.SETTINGS if s.path[-1] == "female" and "vosk" in s.path)
    tera = next(s for s in settings.SETTINGS if s.path[-1] == "female" and "teratts" in s.path)
    for setting, value in ((vosk, 80), (vosk, 1.5), (tera, "ru_f1+nobody"), (tera, "")):
        try:
            settings.from_form(setting, value)
        except ValueError:
            continue
        raise AssertionError(f"{setting.label}: {value!r} accepted")
    assert settings.from_form(tera, "ru_f1+ru_f2") == "ru_f1+ru_f2"
    values = settings.form_values(json.loads((dub_app.REPO_ROOT / "stt_config.json").read_text(encoding="utf-8")))
    values[[s.path for s in settings.SETTINGS].index(("mix", "duck_db"))] = None
    try:
        settings.save(values, dub_app.REPO_ROOT / "stt_config.json", Path("/nonexistent/never_written.json"))
    except settings.SettingsError as exc:
        assert exc.problems == ["«Насколько приглушать, дБ»: введите число"]
    else:
        raise AssertionError("an empty number was saved")
