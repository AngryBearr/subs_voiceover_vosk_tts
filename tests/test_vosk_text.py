"""Punctuation cleanup of the synthesis text: no quote marks, brackets become commas."""

from __future__ import annotations

import pytest

from utils.vosk_text import normalize_vosk_text


@pytest.mark.parametrize("text, expected", [
    ("Как говорит сын, «раздавать».", "Как говорит сын, раздавать."),
    ('Он сказал "да" и ушёл.', "Он сказал да и ушёл."),
    ("“Да” и „нет‟ и ‚так‘", "Да и нет и так"),
    ("Don’t 'go'", "Dont go"),
    ("«Да» — нет…", "Да - нет..."),
    ("Автор (штат Мэн) жил.", "Автор, штат Мэн, жил."),
    ("(Тихо) он сказал", "Тихо, он сказал"),
    ("Он ушёл (давно).", "Он ушёл, давно."),
    ("Он ушёл, (давно), и вот", "Он ушёл, давно, и вот"),
    ("— (Эй!) — крикнул", "- Эй! - крикнул"),
    ("а [1] б {2}", "а, 1, б, 2"),
    ("а  \" б", "а б"),
])
def test_quotes_go_and_brackets_become_commas(text: str, expected: str) -> None:
    assert normalize_vosk_text(text) == expected


def test_no_quote_marks_or_doubled_punctuation_remain() -> None:
    for text in ["(a), (b).", "Слово ((да)) слово", "(Да)", "Да, ( нет ) , да", "«\"x\"» (“y”)"]:
        out = normalize_vosk_text(text)
        assert not any(ch in out for ch in "\"'«»“”„‟‘’‚()")
        assert ",," not in out and ", ," not in out and ",." not in out and not out.startswith(",") and "  " not in out


def test_an_ascii_quote_from_the_number_expansion_is_removed() -> None:
    # ru-normalizr turns every quote into an ASCII ``"``; the engine applies this function to its output
    assert normalize_vosk_text('Он сказал: "три" раза') == "Он сказал: три раза"
    with pytest.raises(TypeError):
        normalize_vosk_text(None)  # type: ignore[arg-type]
