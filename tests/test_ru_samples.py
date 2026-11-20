"""stt.ru_samples: Russian text of a reference clip, take choice; stt.run: which clip F5 gets per language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stt import ru_samples
from stt.run import Episode
from voiceover.text import syllables


def _row(start_s: float, end_s: float, text: str, speaker: str = "S1") -> dict[str, Any]:
    return {"start": int(start_s * 1000), "end": int(end_s * 1000), "text": [text], "speaker": speaker}


def test_text_of_rows_over_the_span_in_ms_and_without_stress_marks() -> None:
    rows = [_row(1, 3, "Это совсем другая реплика."), _row(10, 12, "Я дум+аю, что мы справимся."),
            _row(12, 14, "Всё будет хорошо, вот увидишь."), _row(30, 32, "Далеко отсюда.")]
    text = ru_samples.russian_text(rows, {"start": 10.5, "end": 13.0, "speaker": "S1"})
    assert text == "Я думаю, что мы справимся. Всё будет хорошо, вот увидишь."


def test_pieces_replace_the_span_and_other_speakers_are_ignored() -> None:
    rows = [_row(1, 3, "Первая часть будет вот такой."), _row(2, 3, "Чужая реплика поверх.", "S2"),
            _row(20, 22, "Вторая часть будет совсем другой."), _row(8, 9, "Между частями ничего нет.")]
    ref = {"start": 1.0, "end": 22.0, "pieces": [[1.0, 3.0], [20.0, 22.0]], "speaker": "S1"}
    assert ru_samples.russian_text(rows, ref) == "Первая часть будет вот такой. Вторая часть будет совсем другой."


def test_long_text_is_cut_to_whole_sentences() -> None:
    sentence = "Мы пойдём туда вместе и всё посмотрим."
    rows = [_row(0, 10, " ".join([sentence] * 6))]
    text = ru_samples.russian_text(rows, {"start": 0, "end": 10})
    assert text == " ".join([sentence] * 3) and syllables(text) <= ru_samples.MAX_SYLLABLES


def test_short_text_is_extended_with_the_persons_nearest_rows() -> None:
    rows = [_row(0, 2, "Далёкая реплика человека номер один."), _row(8, 9, "Да."),
            _row(11, 13, "Ближайшая реплика этого же человека."), _row(11, 13, "Чужая реплика рядом.", "S2")]
    text = ru_samples.russian_text(rows, {"start": 8.0, "end": 9.0, "speaker": "S1"})
    assert "Да. Ближайшая реплика этого же человека." in text
    assert "Чужая" not in text and syllables(text) >= ru_samples.EXTEND_SYLLABLES


def test_empty_when_the_person_has_no_rows() -> None:
    assert ru_samples.russian_text([], {"start": 0, "end": 5, "speaker": "S1"}) == ""


def _take(cer: float, speech_s: float, similarity: float, take: int = 0) -> dict[str, Any]:
    return {"take": take, "cer": cer, "speech_s": speech_s, "similarity": similarity}


def test_select_take_prefers_similarity_inside_the_gates() -> None:
    takes = [_take(0.0, 6.0, 0.5, 0), _take(0.04, 6.0, 0.7, 1), _take(0.01, 2.0, 0.9, 2), _take(0.2, 6.0, 0.95, 3)]
    assert ru_samples.select_take(takes)["take"] == 1  # type: ignore[index]


def test_select_take_falls_back_to_fewest_errors() -> None:
    takes = [_take(0.12, 6.0, 0.9, 0), _take(0.08, 6.0, 0.5, 1)]
    assert ru_samples.select_take(takes)["take"] == 1  # type: ignore[index]
    assert ru_samples.select_take([]) is None


def test_cer_and_finish() -> None:
    assert ru_samples.cer("Привет, мир", "привет мир") == 0.0
    assert ru_samples.cer("абвг", "абвд") == 0.25
    out = ru_samples.finish(np.ones(24000, dtype=np.float32))
    assert len(out) == 48000 and out[0] == 0.0 and out[-1] == 0.0 and out[24000] == 0.0 and out[12000] == 1.0


def _episode(tmp_path: Path) -> Episode:
    episode = Episode.__new__(Episode)
    episode.media = Path("e.mkv")
    episode.voice_bank = tmp_path / "Season 1.voices.json"
    refs = episode.references
    refs.mkdir(exist_ok=True)
    (refs / "Amy.wav").write_bytes(b"en")
    (refs / "Amy.txt").write_text("english", encoding="utf-8")
    return episode


def test_reference_language_picks_the_russian_sample_when_it_exists(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    ref = {"name": "Amy", "wav": "Amy.wav"}
    english = (episode.references / "Amy.wav").resolve()
    assert episode._reference_clip(ref, "en", set()) == english  # the default
    warned: set[str] = set()
    assert episode._reference_clip(ref, "ru", warned) == english and warned == {"Amy.wav"}  # no sample yet
    (episode.references / "Amy.ru.wav").write_bytes(b"ru")
    assert episode._reference_clip(ref, "ru", set()) == english  # the text is missing too
    (episode.references / "Amy.ru.txt").write_text("русский", encoding="utf-8")
    assert episode._reference_clip(ref, "ru", set()) == english.with_suffix(".ru.wav")
    assert episode._reference_clip(ref, "en", set()) == english


def test_speaker_voices_follow_reference_language(tmp_path: Path) -> None:
    from unittest import mock

    def voices(language: str | None) -> dict[str, str]:
        episode = _episode(tmp_path)
        f5 = {"reference_language": language} if language else {}
        episode.config = {"voiceover": {"engines": {"f5": f5}}}
        episode.out = tmp_path / "ep"
        episode.out.mkdir(exist_ok=True)
        (episode.out / "speakers.json").write_text(json.dumps([{"id": "S1", "name": "Amy", "gender": "female"}]),
                                                   encoding="utf-8")
        (episode.references / "references.json").write_text(json.dumps([{"name": "Amy", "wav": "Amy.wav"}]),
                                                            encoding="utf-8")
        with mock.patch("voiceover.engine_f5.voice_id", side_effect=lambda p: Path(p).name), \
                mock.patch.object(Episode, "_module"):
            return json.loads(episode._speaker_voices().read_text(encoding="utf-8"))

    assert voices(None) == {"S1": "Amy.wav"}
    (tmp_path / "Season 1.references" / "Amy.ru.wav").write_bytes(b"ru")
    (tmp_path / "Season 1.references" / "Amy.ru.txt").write_text("русский", encoding="utf-8")
    assert voices("en") == {"S1": "Amy.wav"}
    assert voices("ru") == {"S1": "Amy.ru.wav"}


def test_speaker_voices_make_samples_only_for_ru_and_survive_failure(tmp_path: Path) -> None:
    from unittest import mock

    def run(language: str, side_effect: BaseException | None = None) -> mock.MagicMock:
        episode = _episode(tmp_path)
        episode.config = {"voiceover": {"engines": {"f5": {"reference_language": language}}}}
        episode.out = tmp_path / "ep"
        episode.out.mkdir(exist_ok=True)
        (episode.out / "speakers.json").write_text("[]", encoding="utf-8")
        with mock.patch("stt.run.REPO_ROOT", tmp_path), \
                mock.patch.object(Episode, "_module", side_effect=side_effect) as module:
            episode._speaker_voices()
        return module

    (tmp_path / "xvoice_env" / "bin").mkdir(parents=True)
    (tmp_path / "xvoice_env" / "bin" / "python").write_text("")
    assert run("en").call_count == 0
    module = run("ru")
    assert [c.args[:2] for c in module.call_args_list] == [("ru_samples", "stt.ru_samples")]
    run("ru", SystemExit("ru_samples failed"))  # does not fail the episode
    (tmp_path / "xvoice_env" / "bin" / "python").unlink()
    assert run("ru").call_count == 0  # no environment: skipped with a warning


def _season(tmp_path: Path) -> tuple[Path, Path]:
    refs = tmp_path / "S.references"
    refs.mkdir()
    (tmp_path / "E01").mkdir()
    (tmp_path / "E01" / "subs.ru.json").write_text(
        json.dumps([_row(0, 6, "Это достаточно длинная русская реплика.")]), encoding="utf-8")
    (refs / "Amy.wav").write_bytes(b"english")
    (refs / "references.json").write_text(json.dumps(
        [{"name": "Amy", "wav": "Amy.wav", "episode": "E01", "start": 0.0, "end": 6.0}]), encoding="utf-8")
    return refs, tmp_path


def test_failure_is_remembered_skipped_and_retried_when_the_clip_changes(tmp_path: Path) -> None:
    from unittest import mock

    refs, season = _season(tmp_path)
    (season / "E01" / "subs.ru.json").write_text("[]", encoding="utf-8")  # no Russian rows
    ru_samples.make_samples(refs, season_dir=season)
    entry = json.loads((refs / "ru_samples.json").read_text(encoding="utf-8"))["Amy"]
    assert "failed" in entry and entry["clip"]["sha"]
    stamp = (refs / "ru_samples.json").stat().st_mtime_ns
    ru_samples.make_samples(refs, season_dir=season)  # skipped, nothing rewritten
    assert (refs / "ru_samples.json").stat().st_mtime_ns == stamp
    (refs / "Amy.wav").write_bytes(b"another clip")
    (season / "E01" / "subs.ru.json").write_text(
        json.dumps([_row(0, 6, "Это достаточно длинная русская реплика.")]), encoding="utf-8")
    with mock.patch.object(ru_samples, "run_worker", side_effect=RuntimeError("tried again")):
        try:
            ru_samples.make_samples(refs, season_dir=season)
        except RuntimeError as error:
            assert str(error) == "tried again"  # the changed clip was tried again
        else:
            raise AssertionError("not tried again")


def test_clip_from_another_season_is_remembered(tmp_path: Path) -> None:
    refs, season = _season(tmp_path)
    (refs / "references.json").write_text(json.dumps(
        [{"name": "Jeff", "wav": "Amy.wav", "episode": "S51E01", "start": 0.0, "end": 0.0}]), encoding="utf-8")
    ru_samples.make_samples(refs, season_dir=season)
    assert json.loads((refs / "ru_samples.json").read_text(encoding="utf-8"))["Amy"]["failed"] == \
        "clip from another season"


def test_current_sample_is_kept_and_index_not_rewritten_but_stale_one_is_remade(tmp_path: Path) -> None:
    from unittest import mock

    refs, season = _season(tmp_path)
    ref = json.loads((refs / "references.json").read_text(encoding="utf-8"))[0]
    (refs / "Amy.ru.wav").write_bytes(b"ru")
    index = {"Amy": {"text": "x"}}  # made before clip ids existed: current
    (refs / "ru_samples.json").write_text(json.dumps(index), encoding="utf-8")
    stamp = (refs / "ru_samples.json").stat().st_mtime_ns
    with mock.patch.object(ru_samples, "run_worker", side_effect=AssertionError("must not run")):
        ru_samples.make_samples(refs, season_dir=season)
        index["Amy"]["clip"] = ru_samples.clip_id(refs / "Amy.wav", ref)
        (refs / "ru_samples.json").write_text(json.dumps(index), encoding="utf-8")
        stamp = (refs / "ru_samples.json").stat().st_mtime_ns
        ru_samples.make_samples(refs, season_dir=season)
        assert (refs / "ru_samples.json").stat().st_mtime_ns == stamp
        (refs / "Amy.wav").write_bytes(b"replaced by voice_bank")  # stale now
        try:
            ru_samples.make_samples(refs, season_dir=season)
        except AssertionError as error:
            assert str(error) == "must not run"  # the worker was reached: the sample is remade
        else:
            raise AssertionError("stale sample kept")
