"""Deterministic repair of OCR text: stray dots and spaces, words glued together.

    vocab = Vocabulary.build(asr_words, ocr_texts, names)
    repair_text("Ifeel really good about..", vocab)   # -> "I feel really good about..."

The OCR of burned-in captions sometimes drops the space between words ("yourname", "honoringwhat",
"Ifeel", "TOKATRIBE"), turns an ellipsis into ".." or a stray "." and puts spaces before punctuation.
A token is split only when it is not a known word and every part is a known word, and then only if
a part is a very common short word or the parts occur far more often than the glued token (a glued
form that repeats in the OCR is judged by those counts, not by its own count). The vocabulary is the
episode's ASR words, the OCR texts themselves and a small built-in list; names are never split.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

log = logging.getLogger("stt.ocr_repair")

# Very common short English words: one of them next to a known word is enough to trust a split.
FUNCTION_WORDS = frozenset(
    "i a an the you your yours my me we our us he she his her him they them their it its is am are was were be been "
    "to of in on at by for from with and or but so if as not no yes do does did what that this these those who how "
    "when where why all can will just up out about into than then there here too very".split())

# Real words made of known parts that a short episode may never say: never split them.
CONNECTORS = frozenset("of to in on at".split())
COMPOUNDS = frozenset(
    "inside outside into onto today tonight anyone everyone someone something anything nothing everything myself "
    "yourself itself himself herself themselves without within because maybe always already another together tomorrow "
    "yesterday anyway however whatever whoever whenever somebody everybody nobody anybody upon cannot somewhere "
    "everywhere anywhere nowhere sometimes somehow meanwhile alright airline wedding baseball hobby teammate "
    "tribemate camp campfire beach shelter backyard downtown overnight underwater afternoon bathroom bedroom "
    "newspaper homework girlfriend boyfriend grandmother grandfather".split())

_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_GLUED = re.compile(r"(?<![\w'])([^\W\d_]{4,30})(?![\w'])", re.UNICODE)
MAX_PARTS = 3


@dataclass
class Vocabulary:
    """Word counts of the episode: `asr` (spoken words), `ocr` (other on-screen texts), `names` (never split)."""

    asr: Counter[str] = field(default_factory=Counter)
    ocr: Counter[str] = field(default_factory=Counter)
    names: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, asr_words: Iterable[str] = (), ocr_texts: Iterable[str] = (),
              names: Iterable[str] = ()) -> Vocabulary:
        vocab = cls()
        for word in asr_words:
            vocab.asr.update(t.lower() for t in _TOKEN.findall(word))
        for text in ocr_texts:
            vocab.ocr.update(t.lower() for t in _TOKEN.findall(text))
        vocab.names = {n.lower() for name in names for n in _TOKEN.findall(name)}
        return vocab

    def count(self, word: str) -> int:
        """How often the word was seen (built-in words count as very frequent)."""
        word = word.lower()
        return self.asr[word] + self.ocr[word] + (1000 if word in FUNCTION_WORDS else 0)

    def is_known(self, word: str) -> bool:
        """A real word or a name: it is never split."""
        word = word.lower()
        return word in self.asr or word in FUNCTION_WORDS or word in COMPOUNDS or word in self.names


def proper_names(words: Iterable[str]) -> set[str]:
    """Words the speech always writes with a capital letter (names, places); lower case."""
    seen: dict[str, bool] = {}  # lower-case word -> has it ever been written in lower case
    for word in words:
        for token in re.findall(r"[^\W\d_]{3,}", word):
            seen[token.lower()] = seen.get(token.lower(), False) or token[0].islower()
    return {w for w, lower in seen.items() if not lower and w not in FUNCTION_WORDS and w not in COMPOUNDS}


def _good_part(part: str, vocab: Vocabulary) -> bool:
    """A usable piece of a split: a short function word, or a longer word the episode has seen."""
    part = part.lower()
    if len(part) <= 2:
        return part in FUNCTION_WORDS
    return vocab.count(part) > 0


def _split_plans(token: str, vocab: Vocabulary, parts: int = MAX_PARTS) -> list[list[str]]:
    """All ways to cut `token` into 2..`parts` good pieces."""
    plans: list[list[str]] = []
    for cut in range(1, len(token)):
        head, tail = token[:cut], token[cut:]
        if not _good_part(head, vocab):
            continue
        if _good_part(tail, vocab):
            plans.append([head, tail])
        if parts > 2:
            plans += [[head, *rest] for rest in _split_plans(tail, vocab, parts - 1)]
    return plans


def split_token(token: str, vocab: Vocabulary) -> list[str]:
    """The pieces of a glued token, or [token] when it is a word, a name or cannot be split with confidence."""
    if vocab.is_known(token) or len(token) < 4:
        return [token]
    best: tuple[int, float] | None = None
    chosen = [token]
    own = vocab.count(token)
    for plan in _split_plans(token, vocab):
        counts = [vocab.count(p) for p in plan]
        if token.isupper():  # names and credits are in capitals: only long parts the episode has seen again
            seen = all(vocab.asr[p.lower()] + vocab.ocr[p.lower()] >= 2 for p in plan)
            trusted = seen and all(len(p) >= 3 for p in plan) or (  # "HEADOF": a long word and a joining "of"
                seen and all(len(p) >= 4 or p.lower() in CONNECTORS for p in plan))
        else:  # a clear function word ("your", "what", "I") or parts the episode has seen several times
            trusted = any(p.lower() in FUNCTION_WORDS and (len(p) >= 3 or p.lower() in ("i", "a")) for p in plan) \
                or min(counts) >= 3
        if not trusted or (own >= 2 and min(counts) <= own):  # a repeated glued form needs more frequent parts
            continue
        score = (-len(plan), sum(math.log(c) for c in counts))
        if best is None or score > best:
            best, chosen = score, plan
    if best:
        return chosen
    return _damaged_head(token, vocab) if token.isupper() else [token]


def _damaged_head(token: str, vocab: Vocabulary) -> list[str]:
    """"VOIGEACTRESS": a known word with a damaged word glued before it -> ["VOICE", "ACTRESS"] when the
    damaged part is one letter away from a word the OCR saw elsewhere."""
    seen = [w for w in vocab.ocr if len(w) >= 4]
    for cut in range(3, len(token) - 4):
        head, tail = token[:cut], token[cut:]
        if len(tail) < 5 or vocab.asr[tail.lower()] + vocab.ocr[tail.lower()] < 1 or vocab.is_known(head):
            continue
        close = [w for w in seen if len(w) == len(head) and sum(a != b for a, b in zip(w, head.lower())) == 1]
        if len(close) == 1:
            return [close[0].upper(), tail]
    return [token]


def _glued_words(text: str, vocab: Vocabulary) -> tuple[str, list[tuple[str, str]]]:
    changes: list[tuple[str, str]] = []

    def split(match: re.Match[str]) -> str:
        token = match.group(1)
        pieces = split_token(token, vocab)
        if len(pieces) == 1:
            return token
        out, at = [], 0
        for piece in pieces:  # cut the original letters (keeps the case) unless a damaged part was corrected
            out.append(token[at : at + len(piece)] if "".join(pieces).lower() == token.lower() else piece)
            at += len(piece)
        fixed = " ".join(out)
        changes.append((token, fixed))
        return fixed

    return _GLUED.sub(split, text), changes


def fix_punctuation(text: str) -> str:
    """Stray dots and spaces: ".going" -> "going", "about.." -> "about...", "word ." -> "word."."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[^\W\d_])[·•](?=[^\W\d_])", " ", text)  # "TOKA·TRIBE"
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)  # no space before punctuation
    text = re.sub(r",(?=[^\W\d_])", ", ", text)  # "RESIDENT,OBGYN"
    text = re.sub(r"\s*-$", "", text)  # a cut hyphen at the end
    text = re.sub(r"(?<=[^\W\d_])\.{2}(?!\.)", "...", text)  # "about.." is a cut ellipsis
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"^\.{1,2}(?=[^\W\d_])", "", text)  # a leading stray dot or two
    text = re.sub(r"(?<=\s)\.{1,2}(?=[^\W\d_])", "", text)  # the same after a space
    return text


def repair_text(text: str, vocab: Vocabulary | None = None) -> str:
    """Repaired OCR text; without a vocabulary only the punctuation is fixed."""
    text = fix_punctuation(text)
    if vocab is not None:
        text, changes = _glued_words(text, vocab)
        for before, after in changes:
            log.debug("OCR glued words: %s -> %s", before, after)
    return text
