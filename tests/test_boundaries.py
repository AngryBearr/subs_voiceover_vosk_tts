"""Word-level Viterbi speaker relabelling with a fake embedder (no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stt.boundaries import RelabelParams, embed_windows, relabel_words, word_windows
from stt.types import Word

RATE = 16_000
DIM = 8
PERIOD = 0.6  # 0.3 s words, 0.3 s gaps


class FakeEmbedder:
    """The audio of a word holds its true speaker index + 1 as a constant; embed = unit vector + noise."""

    def __init__(self, seed: int = 0, noise: float = 0.05) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.calls = 0

    def batch(self, wavs: np.ndarray) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(wavs), DIM), dtype=np.float32)
        for i, wav in enumerate(wavs):
            out[i, int(round(wav[len(wav) // 2])) - 1] = 1.0
        out += self.rng.normal(0, self.noise, out.shape).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def scene(diar: str, true: str, source: dict[int, str] | None = None) -> tuple[list[Word], np.ndarray]:
    """Words at a 0.6 s period; ``diar`` / ``true`` are one letter per word ('-' in diar: no label)."""
    words = [Word(i * PERIOD, i * PERIOD + 0.3, f"w{i}", speaker=None if d == "-" else d,
                  source=(source or {}).get(i)) for i, d in enumerate(diar)]
    audio = np.zeros(int((len(words) * PERIOD + 1) * RATE), dtype=np.float32)
    for w, t in zip(words, true):
        audio[int(w.start * RATE):int(w.end * RATE)] = ord(t) - ord("A") + 1
    return words, audio


def labels(words: list[Word]) -> str:
    return "".join(w.speaker or "-" for w in words)


def test_one_word_shout_is_relabelled() -> None:
    # B has its own block far enough away to get a centroid
    words, audio = scene("AAAAAA" + "BBBBBB", "AABAAA" + "BBBBBB")
    out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert labels(out) == "AABAAA" + "BBBBBB"
    assert stats["relabelled"] == 1
    assert stats["changes_before"] == 1 and stats["changes_after"] == 3
    assert stats["speakers_with_centroid"] == 2
    assert labels(words) == "AAAAAA" + "BBBBBB"  # the input is not modified


def test_one_word_shout_kept_with_large_penalty() -> None:
    words, audio = scene("AAAAAA" + "BBBBBB", "AABAAA" + "BBBBBB")
    out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams(switch_penalty=2.0))
    assert labels(out) == "AAAAAA" + "BBBBBB"
    assert stats["relabelled"] == 0


def test_shifted_boundary_moves() -> None:
    words, audio = scene("A" * 8 + "B" * 8, "A" * 6 + "B" * 10)
    out, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert labels(out) == "A" * 6 + "B" * 10


def test_flicker_is_smoothed() -> None:
    words, audio = scene("A" * 6 + "ABABAB" + "A" * 4 + "B" * 6, "A" * 16 + "B" * 6)
    out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert labels(out) == "A" * 16 + "B" * 6
    assert stats["changes_after"] < stats["changes_before"]


def test_ocr_words_keep_label_and_missing_labels_get_one() -> None:
    words, audio = scene("AAAAAAAA" + "AA-AAAAAA" + "BBBBBB", "A" * 17 + "BBBBBB",
                         source={13: "ocr"})
    words[13].speaker = "B"  # an OCR word the audio says is A, labelled B
    out, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert out[13].speaker == "B"
    assert out[10].speaker == "A"
    assert all(w.speaker == "A" for i, w in enumerate(out[:17]) if i != 13)


def test_speaker_without_centroid_keeps_its_words() -> None:
    words, audio = scene("AAAAAA" + "BB", "AAAAAA" + "AA")  # B has 2 words: no centroid
    out, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert labels(out) == "AAAAAA" + "BB"


def test_word_windows_shapes_padding_and_crop() -> None:
    audio = np.arange(1, 3 * RATE + 1, dtype=np.float32)
    words = [Word(0.0, 0.1, "a"), Word(1.4, 1.6, "b"), Word(2.95, 3.0, "c"), Word(0.5, 2.5, "long")]
    win = word_windows(words, audio, 1.0)
    assert win.shape == (4, RATE)
    first = int(round(0.05 * RATE)) - RATE // 2  # window starts before the audio: zero padded
    assert np.all(win[0, :-first] == 0) and win[0, -first] == audio[0]
    assert np.array_equal(win[1], audio[int(1.5 * RATE) - RATE // 2:int(1.5 * RATE) + RATE // 2])
    last = int(round(2.975 * RATE)) - RATE // 2  # window runs past the audio end
    assert win[2, len(audio) - last - 1] == audio[-1] and np.all(win[2, len(audio) - last:] == 0)
    centre = int(round(1.5 * RATE))  # longer than the window: the centre crop
    assert np.array_equal(win[3], audio[centre - RATE // 2:centre + RATE // 2])


def test_embed_windows_falls_back_to_per_item_call() -> None:
    windows = np.ones((3, 10), dtype=np.float32)
    out = embed_windows(windows, lambda w: np.array([1.0, 0.0], dtype=np.float32))
    assert out.shape == (3, 2)


def test_cache_round_trip_and_invalidation(tmp_path: Path) -> None:
    cache = tmp_path / "words.emb.fake.npz"
    words, audio = scene("AAAAAA" + "BBBBBB", "AABAAA" + "BBBBBB")
    first = FakeEmbedder()
    out1, _ = relabel_words(words, audio, first, RelabelParams(), cache=cache)
    assert cache.exists() and first.calls == 1
    second = FakeEmbedder()
    out2, stats = relabel_words(words, audio, second, RelabelParams(), cache=cache)
    assert second.calls == 0 and stats["embed_s"] == 0.0
    assert labels(out1) == labels(out2)
    words[3].start += 0.01  # a word time changed: recompute and overwrite
    third = FakeEmbedder()
    relabel_words(words, audio, third, RelabelParams(), cache=cache)
    assert third.calls == 1
    fourth = FakeEmbedder()
    relabel_words(words, audio, fourth, RelabelParams(window_s=0.8), cache=cache)  # window differs
    assert fourth.calls == 1


def test_change_prior_keeps_a_short_interjection() -> None:
    """A one-word 'yep' of B inside A's speech: its 1-s window sounds like A, so without a change prior
    the Viterbi gives it to A; with the prior the diarizer's change is kept."""
    from stt.boundaries import RelabelParams, _viterbi

    words = [Word(i * 0.5, i * 0.5 + 0.4, "w", speaker=s) for i, s in enumerate("AAAABAAAA")]
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    emb = np.stack([a if s == "A" else 0.7 * a + 0.3 * b for s in "AAAABAAAA"])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    centroids = {"A": a, "B": b}
    own = [w.speaker for w in words]
    without = _viterbi(words, own, emb, centroids, RelabelParams(switch_penalty=0.2, prior=0.1, change_prior=0.0))
    assert without == list("AAAAAAAAA")
    with_prior = _viterbi(words, own, emb, centroids, RelabelParams(switch_penalty=0.2, prior=0.1, change_prior=0.5))
    assert with_prior == list("AAAABAAAA")


def _unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_missing_people() -> None:
    from stt.boundaries import missing_people

    bank = {"Ann": _unit(1, 0, 0, 0), "Ben": _unit(1, 0.3, 0, 0), "Cy": _unit(0, 0, 1, 0), "Di": _unit(0, 0, 0, 1)}
    clusters = {"0": _unit(1, 0.05, 0, 0),  # Ann's own cluster (Ben is a little further)
                "1": _unit(0, 0, 0.4, 1)}  # Di: closest, cosine 0.93; Cy is only 0.37 away
    missing = missing_people(bank, clusters, 0.5)
    assert sorted(missing) == ["Ben", "Cy"]  # Ben shares Ann's cluster, Cy is below `represented`
    assert missing["Cy"] is bank["Cy"]
    assert sorted(missing_people(bank, clusters, 0.9999)) == ["Ann", "Ben", "Cy", "Di"]
    assert missing_people(bank, {}, 0.5) == bank and missing_people({}, clusters, 0.5) == {}


def _merged_scene(diar: str, true: str) -> tuple[list[Word], np.ndarray, np.ndarray, RelabelParams]:
    words, audio = scene(diar, true)
    # person C (axis 2) was merged into A's cluster: the bank knows C's voice
    bank = np.zeros(DIM, dtype=np.float32)
    bank[2] = 1.0
    return words, audio, bank, RelabelParams()


def test_bank_state_wins_the_merged_persons_words() -> None:
    words, audio, bank, params = _merged_scene("A" * 16, "A" * 12 + "C" * 4)
    out, stats = relabel_words(words, audio, FakeEmbedder(), params, bank={"Cee": bank})
    assert [w.speaker for w in out] == ["A"] * 12 + ["bank:Cee"] * 4
    assert stats["bank_states"] == ["Cee"] and stats["bank_words"] == 4
    big, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams(bank_margin=1.0), bank={"Cee": bank})
    assert [w.speaker for w in big] == ["A"] * 16
    assert stats["bank_states"] == ["Cee"] and stats["bank_words"] == 0


def test_no_bank_gives_todays_labels() -> None:
    words, audio, bank, params = _merged_scene("A" * 16, "A" * 12 + "C" * 4)
    plain, stats = relabel_words(words, audio, FakeEmbedder(), params)
    assert [w.speaker for w in plain] == ["A"] * 16
    assert "bank_states" not in stats and "bank_words" not in stats
    empty, _ = relabel_words(words, audio, FakeEmbedder(), params, bank={})
    assert labels(empty) == labels(plain)


def test_bank_min_s_drops_a_one_word_run() -> None:
    words, audio, bank, _ = _merged_scene("A" * 16, "A" * 12 + "C" + "A" * 3)
    kept, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams(), bank={"Cee": bank})
    assert [w.speaker for w in kept].count("bank:Cee") == 1
    out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams(bank_min_s=1.0), bank={"Cee": bank})
    assert [w.speaker for w in out] == ["A"] * 16 and stats["bank_words"] == 0


def test_bank_centroids_and_cache(tmp_path: Path) -> None:
    import json

    import soundfile as sf

    from stt.boundaries import bank_centroids, cached_bank_centroids

    ref = tmp_path / "refs"
    ref.mkdir()
    voice = np.full(24_000 * 3, 2.0, dtype=np.float32)  # speaker index 1 (value 2) at 24 kHz
    sf.write(str(ref / "a.wav"), np.concatenate([voice, np.zeros(24_000, dtype=np.float32)]), 24_000, subtype="FLOAT")
    (ref / "references.json").write_text(json.dumps([{"name": "Ann", "wav": "a.wav"}]), encoding="utf-8")
    found = bank_centroids(ref, FakeEmbedder(), 1.0)
    assert list(found) == ["Ann"] and int(np.argmax(found["Ann"])) == 1  # silence windows would add axis 0
    assert abs(float(np.linalg.norm(found["Ann"])) - 1.0) < 1e-5
    cache = tmp_path / "bank.npz"
    first = cached_bank_centroids(ref, FakeEmbedder(), 1.0, cache)
    second_embedder = FakeEmbedder()
    second = cached_bank_centroids(ref, second_embedder, 1.0, cache)
    assert second_embedder.calls == 0 and np.allclose(first["Ann"], second["Ann"])
    assert not (ref / "bank.npz").exists()
