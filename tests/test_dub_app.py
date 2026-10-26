from __future__ import annotations

from pathlib import Path

from ui import dub_app


def test_job_config_applies_the_form() -> None:
    config = dub_app.job_config("Быстрый: только Whisper", "openai/gpt-6-luna", subtitles=False, duck_voices=False,
                                stress=False, refine=False)
    assert config["mode"] == "fast" and config["llm"]["model"] == "openai/gpt-6-luna"
    assert config["mix"]["subtitles"] is False and config["mix"]["duck"] == "all"
    assert config["voiceover"]["stress_check"] is False and config["speaker_refine"]["model"] == "none"


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
