"""Fusion keeps what only Whisper heard; CTC alignment is anchored to Whisper's word times where it drifts."""
from __future__ import annotations

from stt.align import anchor_to_reference
from stt.fusion import fuse, restore_dropped
from stt.types import Segment, Word


def _words(text: str, start: float, step: float = 0.5) -> list[Word]:
    return [Word(start + i * step, start + i * step + 0.3, t) for i, t in enumerate(text.split())]


class TestRestoreDropped:
    def test_puts_back_omitted_run(self) -> None:
        fused = "Keep going. There's one for Sabu."
        first = "Keep going. That's wheeled on the tail! There's one for Sabu."
        assert restore_dropped(fused, first) == "Keep going. That's wheeled on the tail! There's one for Sabu."

    def test_disagreement_keeps_llm_version(self) -> None:
        # words replaced by other words are not an omission: no doubled text
        fused = "Toka has everybody up top."
        first = "Toca has everybody up top."
        assert restore_dropped(fused, first) == fused

    def test_short_run_and_hesitations_stay_dropped(self) -> None:
        assert restore_dropped("We go now.", "We go yes now.") == "We go now."
        assert restore_dropped("We go now.", "We go um uh uh now.") == "We go now."

    def test_empty_inputs(self) -> None:
        assert restore_dropped("", "a b c d") == ""
        assert restore_dropped("a b c", "") == "a b c"


class _Client:
    def __init__(self, text: str) -> None:
        self.text = text

    def chat_json(self, system: str, prompt: str) -> dict:
        return {"chunks": [{"id": 0, "text": self.text}]}


def test_fuse_restores_first_model_words() -> None:
    chunks = [(0.0, 10.0)]
    hyp = {"whisper": [Segment(0.0, 10.0, "Go quickly now. Good job everybody. Come on")],
           "cohere": [Segment(0.0, 10.0, "Go quickly now. Come on")]}
    out = fuse(chunks, hyp, _Client("Go quickly now. Come on"))  # type: ignore[arg-type]
    assert out[0].text == "Go quickly now. Good job everybody. Come on"


class TestAnchorToReference:
    text = "one two three four five six seven eight nine ten"

    def _drifted(self, text: str) -> list[Word]:
        """The first four words are timed right, the rest lands 3 s late."""
        words = _words(text, 10.0)
        return words[:4] + [Word(w.start + 3.0, w.end + 3.0, w.text) for w in words[4:]]

    def test_drifted_run_takes_reference_times(self) -> None:
        reference = _words(self.text, 10.0)
        out = anchor_to_reference(self._drifted(self.text), reference)
        assert [round(w.start, 1) for w in out] == [round(w.start, 1) for w in reference]
        assert [w.text for w in out] == self.text.split()

    def test_agreeing_times_unchanged(self) -> None:
        words = _words(self.text, 10.0)
        assert anchor_to_reference(words, _words(self.text, 10.4)) == words

    def test_lone_mismatch_is_not_moved(self) -> None:
        words = _words(self.text, 10.0)
        reference = _words(self.text, 10.0)
        reference[2] = Word(30.0, 30.3, "three")
        assert anchor_to_reference(words, reference) == words

    def test_unmatched_words_are_kept_and_monotonic(self) -> None:
        drifted = self._drifted("one two three four five six extra seven eight nine ten")
        out = anchor_to_reference(drifted, _words(self.text, 10.0))
        assert [w.text for w in out] == [w.text for w in drifted]
        assert all(a.end <= b.start + 1e-6 for a, b in zip(out, out[1:]))
        extra = out[6]
        assert out[5].end <= extra.start and extra.end <= out[7].start + 1e-6
        assert extra.start < 13.5  # no longer left at the drifted time

    def test_no_reference(self) -> None:
        words = _words(self.text, 10.0)
        assert anchor_to_reference(words, []) == words



def test_fuse_restores_from_the_timed_recognition_not_the_first() -> None:
    """In the pipeline Cohere (text only) comes first and Whisper (word times) second: Whisper's dropped shout comes back."""
    from stt.fusion import fuse
    from stt.types import Segment, Word

    class Client:
        def chat_json(self, system, prompt):  # noqa: ANN001
            return {"chunks": [{"id": 0, "text": "We are going to win this."}]}

    cohere = [Segment(start=0.0, end=4.0, text="We are going to win this.")]
    whisper = [Segment(start=0.0, end=4.0, text="We are going to win this. Pull him up now!",
                       words=[Word(0.1, 0.3, "We"), Word(3.0, 3.3, "Pull"), Word(3.3, 3.5, "him"), Word(3.5, 3.7, "up"),
                              Word(3.7, 3.9, "now!")])]
    fused = fuse([(0.0, 4.0)], {"cohere-transcribe": cohere, "whisper-large-v3": whisper}, Client())
    assert fused[0].text == "We are going to win this. Pull him up now!"
