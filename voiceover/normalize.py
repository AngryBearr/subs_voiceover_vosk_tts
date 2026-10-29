"""Rule-based text expansion for TTS with a fail-closed guard.

Numbers are spoken by ``ru-normalizr`` (rules + pymorphy3: the case of the words around them),
sentence by sentence where a sentence holds a digit; ranges ("3-4 года") have their own rule (each
number said in the sentence alone, joined: "три-четыре года", "около трёх-четырёх лет"); times and
scores ("10:07", "2:1"), abbreviations, symbols and Latin words go to ``rutextnorm`` (pure rules,
deterministic, no model) as before. On 64 subtitle-like phrases rutextnorm alone got ~15 numbers
wrong ("Мне 22 года" -> "двадцать второго года", "2 недели" -> "два недели", "с 10 часов" -> "с
десять"), this way none (a year range "1990–2000 гг." is ru-normalizr's own and still clumsy). The
guard makes sure that ordinary
Russian words survive untouched: every Cyrillic word of the input that is not
an abbreviation candidate (followed by a dot) and not a unit after a number
must reappear, in order, in the output.  If the guard fails, the caller keeps
the original text, so no expansion step can ever invent words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rutextnorm import flag_uncertain, normalize_russian

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|[^\w\s]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]+(?:[-'’][А-Яа-яЁё]+)*")
_NUMBER_RE = re.compile(r"\d")
_GLUED_ELLIPSIS_RE = re.compile(r"(\.{3}|…)(?=[^\W\d_])")
# Abbreviations read letter by letter although they hold a vowel; rutextnorm keeps such words as
# they are and the TTS then says them as one syllable (США -> "ша"; found by voiceover.speech_check).
SPELLED = {"США": "Сэ-Шэ-А", "НЛО": "Эн-Эл-О", "ЮАР": "Ю-А-Эр", "ОАЭ": "О-А-Э"}
_SPELLED_RE = re.compile(r"(?<![А-Яа-яЁё])(" + "|".join(SPELLED) + r")(?![А-Яа-яЁё])")


@dataclass(frozen=True)
class Expanded:
    text: str
    changed: bool
    uncertain: tuple[str, ...]  # human-readable notes for spans the rules were unsure about


def _fold(word: str) -> str:
    return word.casefold().replace("ё", "е")


def _must_keep(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    keep: list[str] = []
    for position, token in enumerate(tokens):
        if not _CYRILLIC_RE.fullmatch(token):
            continue
        followed_by_dot = position + 1 < len(tokens) and tokens[position + 1] == "."
        after_number = position > 0 and bool(_NUMBER_RE.search(tokens[position - 1]))
        if not followed_by_dot and not after_number:
            keep.append(_fold(token))
    return keep


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    index = 0
    for item in haystack:
        if index < len(needle) and item == needle[index]:
            index += 1
    return index == len(needle)


def guard(before: str, after: str) -> None:
    """Raise ``ValueError`` when expansion changed a plain Russian word."""
    if not isinstance(after, str):
        raise ValueError("expansion returned non-text")
    expected = _must_keep(before)
    actual = [_fold(token) for token in _TOKEN_RE.findall(after) if _CYRILLIC_RE.fullmatch(token)]
    if not _is_subsequence(expected, actual):
        raise ValueError(f"expansion changed plain words: {expected!r} -> {actual!r}")


def space_after_ellipsis(text: str) -> str:
    """``хотят…хотят`` -> ``хотят… хотят``: the accentizer otherwise glues the words."""
    return _GLUED_ELLIPSIS_RE.sub(r"\1 ", text)


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_CLOCK_RE = re.compile(r"\d+:\d+")  # times and scores: rutextnorm says them better
# "3-4", "3 или 4": both numbers take the sentence's case (ru-normalizr inflects only the first)
_RANGE_RE = re.compile(r"(?<![\d.,:])(\d+)(\s*[-–—]\s*|\s+(?:или|либо)\s+)(\d+)(?![\d:]|[.,]\d|-[А-Яа-яЁё])")
_SAID_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*|\d+|\S")
_YEARS_RE = re.compile(r"\s*(?:г\.|гг\.|год|годы|годах|годов|годам)")  # "1990-2000 гг.": ru-normalizr's own
_KEEP = "QQKEEP{}QQ"  # a placeholder ru-normalizr leaves alone (letters only: it says digits)
_NORMALIZER: Any = None


def _say_numbers(sentence: str) -> str:
    global _NORMALIZER
    if _NORMALIZER is None:
        from ru_normalizr import NormalizeOptions, Normalizer

        _NORMALIZER = Normalizer(NormalizeOptions())
    clocks = _CLOCK_RE.findall(sentence)
    marks = [_KEEP.format(chr(ord("A") + n % 26) * (n // 26 + 1)) for n in range(len(clocks))]
    for clock, mark in zip(clocks, marks):
        sentence = sentence.replace(clock, mark, 1)
    sentence = _NORMALIZER.normalize(sentence)
    for clock, mark in zip(clocks, marks):
        sentence = sentence.replace(mark, clock, 1)
    return sentence


def _say_range(sentence: str, match: re.Match[str]) -> str | None:
    """ "3-4" of a sentence as words, each number in the sentence's case ("трёх-четырёх")."""
    said = [_SAID_TOKEN_RE.findall(_say_numbers(sentence[:match.start()] + number + sentence[match.end():]))
            for number in (match.group(1), match.group(3))]  # words and punctuation apart
    head = 0
    while head < min(map(len, said)) and said[0][head] == said[1][head]:
        head += 1
    tail = 0
    while tail < min(map(len, said)) - head and said[0][-1 - tail] == said[1][-1 - tail]:
        tail += 1
    middles = [words[head:len(words) - tail] for words in said]
    if not all(middles) or any(re.search(r"\d", w) for words in middles for w in words):
        return None
    joint = "-" if match.group(2).strip() in "-–—" else f" {match.group(2).strip()} "
    return " ".join(middles[0]) + joint + " ".join(middles[1])


def say_numbers(text: str) -> str:
    """The digits of a text as Russian words agreeing with their sentence (ru-normalizr)."""
    out = []
    for sentence in _SENTENCE_RE.split(text):
        if re.search(r"\d", sentence):
            position = 0
            while match := _RANGE_RE.search(sentence, position):
                years = max(int(match.group(1)), int(match.group(3))) >= 1000 \
                    and _YEARS_RE.match(sentence, match.end())
                words = None if years else _say_range(sentence, match)
                if words is None:
                    position = match.end()
                    continue
                sentence = sentence[:match.start()] + words + sentence[match.end():]
                position = match.start() + len(words)
            if re.search(r"\d", _CLOCK_RE.sub("", sentence)):
                sentence = _say_numbers(sentence)
        out.append(sentence)
    return " ".join(out)


def expand(text: str) -> Expanded:
    """Expand digits/abbreviations/Latin; raises ``ValueError`` if a plain word changed."""
    text = _SPELLED_RE.sub(lambda match: SPELLED[match.group(1)], text)
    spoken = say_numbers(text) if re.search(r"\d", text) else text
    result = normalize_russian(spoken)
    guard(text, result)
    notes = tuple(f"{original!r}: {reason}" for _start, _end, original, reason in flag_uncertain(text))
    return Expanded(result, result != text, notes)


_ACCENTED_WORD_RE = re.compile(r"[А-Яа-яЁё+]+")
_SOURCE_WORD_RE = re.compile(r"[А-Яа-яЁё]+")


def keep_yo_spelling(source: str, accented: str, yo_homographs: dict[str, str], accents: dict[str, str]) -> str:
    """Undo RUAccent's е -> ё in words that are spelled both ways (все / всё), keeping the source.

    For text whose author already writes ё (the LLM translation) the source spelling is the
    better guess: RUAccent's model turned "Все хвалили", "Где все?" into "всё".
    """
    source_words = iter(_SOURCE_WORD_RE.findall(source))

    def fix(match: re.Match[str]) -> str:
        token = match.group(0)
        plain = token.replace("+", "")
        original = next(source_words, None)
        if original is None or "ё" in original.lower() or "ё" not in plain.lower() \
                or plain.lower().replace("ё", "е") != original.lower() or original.lower() not in yo_homographs:
            return token
        stressed = accents.get(original.lower())
        if not stressed or stressed.replace("+", "") != original.lower():
            yo = plain.lower().index("ё")
            stressed = original.lower()[:yo] + "+" + original.lower()[yo:]
        position = stressed.index("+")
        return original[:position] + "+" + original[position:]

    return _ACCENTED_WORD_RE.sub(fix, accented)
