"""stt.own_voice: clean prompt spans (no laughter, no stretched words, one voice) and the Russian sample of a prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stt import own_voice
from stt.own_voice import capped_end, collect_spans, cut_ranges, embed_spans, select_spans, speech_spans


def _w(text: str, start: float, end: float, speaker: str = "S1") -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def test_laughter_is_dropped_and_close_words_merge() -> None:
    words = [_w("Ha", 1.0, 1.4), _w("What", 1.5, 1.8), _w("happened?", 1.9, 2.5), _w("haha", 2.6, 3.4),
             _w("Well,", 5.0, 5.4), _w("ok.", 5.5, 5.8)]
    assert speech_spans(words, "S1", 0.9, 6.0) == [(1.5, 2.5), (5.0, 5.8)]


def test_a_word_stretched_over_laughter_is_capped_by_its_letters() -> None:
    word = _w("happened?", 624.1, 626.9)  # the real case: laughter after the line
    assert capped_end(word) == 624.1 + 0.30 + 0.11 * 8  # letters only: the "?" does not count
    assert capped_end(_w("time.", 452.2, 459.0)) < 453.0
    assert capped_end(_w("no", 1.0, 1.2)) == 1.2  # a normal word is not touched
    assert speech_spans([_w("happened?", 624.1, 626.9)], "S1", 623.0, 627.0) == [(624.1, 624.1 + 0.30 + 0.88)]


def test_other_speakers_short_spans_and_words_outside_the_cue_are_left_out() -> None:
    words = [_w("Hi", 1.0, 1.2), _w("you", 1.0, 1.5, "S2"), _w("a", 4.0, 4.1), _w("outside", 9.0, 9.5)]
    assert speech_spans(words, "S1", 0.5, 5.0) == []  # "Hi" is 0.2 s: under the shortest span
    assert speech_spans(words, "S2", 0.5, 5.0) == [(1.0, 1.5)]


def test_collect_skips_overlap_fill_ocr_cues_and_words_under_another_voice() -> None:
    words = [_w("hello", 1.0, 1.6), _w("there", 1.7, 2.2), _w("again", 3.0, 3.6), _w("yes", 3.1, 3.7, "S2"),
             _w("fine", 5.0, 5.6), _w("more", 7.0, 7.6), _w("text", 9.0, 9.6)]
    cues = [{"speaker": "S1", "start": 900, "end": 2300}, {"speaker": "S1", "start": 2900, "end": 3800},
            {"speaker": "S1", "start": 4900, "end": 5700, "overlap": True},
            {"speaker": "S1", "start": 6900, "end": 7700, "fill": True},
            {"speaker": "S1", "start": 8900, "end": 9700, "ocr": True}]
    assert collect_spans(words, cues, "S1") == [(1.0, 2.2)]


def _vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_embedding_pads_short_spans_and_normalises() -> None:
    seen: list[int] = []

    def embed(piece: np.ndarray) -> np.ndarray:
        seen.append(len(piece))
        return np.array([2.0, 0.0])

    vectors = embed_spans(np.ones(16_000 * 5, dtype=np.float32), [(0.0, 0.4), (1.0, 3.0)], embed)
    assert seen == [16_000, 32_000] and np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_an_outlier_span_is_dropped_and_the_best_are_chosen_up_to_the_limit() -> None:
    spans = [(0.0, 3.0), (10.0, 13.0), (20.0, 23.0), (30.0, 33.0), (40.0, 41.0)]
    vectors = np.stack([_vec(1, 0.1), _vec(1, 0.2), _vec(1, 0.3), _vec(1, 0.0), _vec(0, 1)])  # the last: another voice
    result = select_spans(spans, vectors, min_s=1.5, max_s=7.0)
    assert result.reason is None
    assert (40.0, 41.0) in [s for s, _ in result.dropped]
    assert len(result.kept) == 2 and result.seconds <= 7.0  # 3 + 0.3 + 3 fits, a third span would not
    assert [c for _, c in result.kept] == sorted((c for _, c in result.kept), reverse=True)


def test_a_mixed_cluster_or_too_little_speech_gives_no_prompt() -> None:
    spans = [(0.0, 3.0), (10.0, 12.0), (20.0, 22.0), (30.0, 32.0)]
    four_voices = np.eye(4)  # 3 + 2 + 2 + 2 s: the centroid is near the first voice, the other 6 of 9 s are far from it
    result = select_spans(spans, four_voices, min_s=1.5)
    assert result.reason is not None and "mixed" in result.reason and result.kept == []
    few = select_spans([(0.0, 1.0)], np.stack([_vec(1, 0)]), min_s=1.5)
    assert few.reason is not None and "1.0 s" in few.reason
    assert select_spans([], np.zeros((0, 0))).reason is not None


def test_chosen_spans_are_padded_joined_and_short_ranges_dropped() -> None:
    assert cut_ranges([(30.0, 31.0), (10.0, 10.2), (11.0, 12.0)]) == [(10.95, 12.05), (29.95, 31.05)]
    assert cut_ranges([(11.0, 12.0), (12.3, 13.0)]) == [(10.95, 13.05)]  # 0.2 s apart: one range
    assert own_voice.MIN_RANGE_S == 0.5


def test_two_evenly_mixed_voices_give_no_prompt_and_a_clear_majority_keeps_its_voice() -> None:
    a, b = _vec(1, 0.0), _vec(0.25, 1.0)  # two people: cosine ~0.24, both ~0.8 to the joint centroid
    spans = [(0.0, 2.0), (10.0, 12.0), (20.0, 22.0), (30.0, 32.0)]
    even = select_spans(spans, np.stack([a, b, a, b]), min_s=1.5)
    assert even.reason is not None and "two voices" in even.reason
    spans6 = spans + [(40.0, 42.0), (50.0, 52.0)]
    major = select_spans(spans6, np.stack([a, a, a, a, b, b]), min_s=1.5, max_s=20.0)
    assert major.reason is None
    assert {s for s, _ in major.kept} == set(spans) and {s for s, _ in major.dropped} == {(40.0, 42.0), (50.0, 52.0)}


def test_one_voice_with_some_spread_is_not_split() -> None:
    spans = [(0.0, 2.0), (10.0, 12.0), (20.0, 22.0), (30.0, 32.0)]
    vectors = np.stack([_vec(1, 0.0), _vec(1, 0.3), _vec(1, 0.6), _vec(1, 0.2)])  # one person, calm and shouting
    result = select_spans(spans, vectors, min_s=1.5)
    assert result.reason is None and len(result.kept) == 4


# --- the Russian sample (X-Voice) of a prompt ---

class _Embedder:
    def batch(self, windows: np.ndarray) -> np.ndarray:
        return np.ones((len(windows), 4), dtype=np.float32)


def _episode(tmp_path: Path, offset: float = 100.0) -> Path:
    import json

    import soundfile as sf

    folder = tmp_path / "ep"
    (folder / "own_voice").mkdir(parents=True)
    (folder / "source.json").write_text(json.dumps({"media": "m.mkv", "start": offset}), encoding="utf-8")
    rows = [{"speaker": "S1", "start": 1000, "end": 4000, "text": ["Это наша первая длинная реплика."]},
            {"speaker": "S1", "start": 20000, "end": 23000, "text": ["А это совсем другая реплика."]},
            {"speaker": "S2", "start": 1000, "end": 4000, "text": ["Чужая реплика."]}]
    (folder / "subs.ru.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    sf.write(str(folder / "own_voice" / "S1.wav"), np.full(16000, 0.1, dtype=np.float32), 16000)
    (folder / "own_voice" / "S1.json").write_text(json.dumps({"ranges": [[offset + 1.0, offset + 4.0]]}), encoding="utf-8")
    return folder


def _fake_scoring(monkeypatch: Any, cers: dict[int, float]) -> None:
    """Takes scored by their seed (the file stem) from ``cers`` instead of a recognizer; a prompt read as 1 s of sound."""
    from stt import ru_samples

    def score(path: Path, text: str, recognizer: Any, embed: Any, reference_print: np.ndarray) -> dict[str, Any]:
        seed = int(path.stem)
        return {"cer": cers[seed], "speech_s": 5.0, "similarity": 0.5 + seed / 10, "cut_s": 0.0,
                "audio": np.full(24000, 0.1, dtype=np.float32)}

    monkeypatch.setattr(ru_samples, "score_take", score)
    monkeypatch.setattr(ru_samples, "_read_mono", lambda path, rate: np.full(rate, 0.1, dtype=np.float32))


def test_the_sample_is_made_from_the_translation_over_the_prompt_ranges_and_the_best_take_is_kept(
        tmp_path: Path, monkeypatch: Any) -> None:
    import json

    folder = _episode(tmp_path)
    _fake_scoring(monkeypatch, {0: 0.5, 1: 0.01, 2: 0.02})
    seen: list[list[dict[str, Any]]] = []

    def worker(items: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]]:
        import soundfile as sf

        seen.append(items)
        for item in items:
            sf.write(item["out_wav"], np.zeros(24000, dtype=np.float32), 24000)
        return [{"out_wav": item["out_wav"]} for item in items]

    reasons: dict[str, str] = {}
    prompt = folder / "own_voice" / "S1.wav"
    wav = own_voice.russian_sample(folder, "S1", prompt, takes=3, recognizer=None, embedder=_Embedder(),
                                   run_worker=worker, reasons=reasons)
    assert wav == folder / "own_voice" / "S1.ru.wav" and wav.is_file() and not reasons
    text = (folder / "own_voice" / "S1.ru.txt").read_text(encoding="utf-8").strip()
    assert text == "Это наша первая длинная реплика."  # the S1 row over 1-4 s of the audio (media 101-104 s, offset 100)
    assert [i["seed"] for i in seen[0]] == [0, 1, 2] and all(i["text"] == text for i in seen[0])
    assert seen[0][0]["ref_wav"] == str(prompt.resolve())
    sample = json.loads((folder / "own_voice" / "S1.json").read_text(encoding="utf-8"))["sample"]
    assert sample["take"] == 2 and sample["cer"] == 0.02 and sample["takes"] == 3 and sample["text"] == text
    assert len(sample["prompt_sha256"]) == 64


def test_a_second_call_reuses_the_sample_until_the_prompt_or_the_text_changes(tmp_path: Path, monkeypatch: Any) -> None:
    import json

    folder = _episode(tmp_path)
    _fake_scoring(monkeypatch, {0: 0.01})
    calls: list[int] = []

    def worker(items: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]]:
        import soundfile as sf

        calls.append(len(items))
        for item in items:
            sf.write(item["out_wav"], np.zeros(24000, dtype=np.float32), 24000)
        return [{"out_wav": item["out_wav"]} for item in items]

    prompt = folder / "own_voice" / "S1.wav"
    args: dict[str, Any] = {"takes": 1, "recognizer": None, "embedder": _Embedder(), "run_worker": worker}
    first = own_voice.russian_sample(folder, "S1", prompt, **args)
    assert own_voice.russian_sample(folder, "S1", prompt, **args) == first and calls == [1]
    rows = json.loads((folder / "subs.ru.json").read_text(encoding="utf-8"))
    rows[0]["text"] = ["Теперь совсем другой текст реплики."]
    (folder / "subs.ru.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert own_voice.russian_sample(folder, "S1", prompt, **args) == first and calls == [1, 1]  # text changed


def test_no_russian_text_or_a_missing_xvoice_means_no_sample_with_a_reason(tmp_path: Path, monkeypatch: Any) -> None:
    folder = _episode(tmp_path)
    _fake_scoring(monkeypatch, {0: 0.01})
    prompt = folder / "own_voice" / "S1.wav"
    reasons: dict[str, str] = {}
    (folder / "subs.ru.json").write_text("[]", encoding="utf-8")

    def never(items: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]]:
        raise AssertionError("the worker must not run without a text")

    assert own_voice.russian_sample(folder, "S1", prompt, takes=1, recognizer=None, embedder=_Embedder(),
                                    run_worker=never, reasons=reasons) is None
    assert reasons == {"S1": "no Russian text"}

    (folder / "subs.ru.json").write_text('[{"speaker": "S1", "start": 1000, "end": 4000, "text": ["Это наша реплика."]}]',
                                         encoding="utf-8")

    def missing(items: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]]:
        raise SystemExit("xvoice_env or models/X-Voice missing")

    assert own_voice.russian_sample(folder, "S1", prompt, takes=1, recognizer=None, embedder=_Embedder(),
                                    run_worker=missing, reasons=reasons) is None
    assert "X-Voice failed" in reasons["S1"] and not (folder / "own_voice" / "S1.ru.wav").exists()
