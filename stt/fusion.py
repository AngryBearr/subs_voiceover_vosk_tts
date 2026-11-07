"""LLM fusion of several ASR hypotheses of the same VAD chunks (generative error correction).

Different ASR models make different mistakes; an LLM sees all hypotheses of a chunk plus the
show glossary and writes the most likely transcript. The result is text per chunk (no word
times) and goes through forced alignment (`stt.align`) afterwards.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from stt.llm import OpenRouterClient
from stt.textfix import collapse_repeats
from stt.types import Segment
from stt.vad import Chunk

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You fix automatic speech recognition (ASR) output of an English TV show.
For each audio chunk you get transcripts of the SAME audio from several ASR systems. They make
different mistakes. Produce the single most likely transcript of what was actually said:
- prefer words supported by most systems; use the context of neighbouring chunks;
- fix misheard names and terms using the glossary; keep the speakers' own words, do not
  paraphrase, summarize or add content; drop pure hesitation sounds (um, uh);
- use normal sentence punctuation and capitalization;
- if systems disagree and none is plausible, keep the first system's words.
Answer JSON only: {"chunks": [{"id": <id>, "text": "<transcript>"}, ...]}"""


def chunk_texts(segments: list[Segment], chunks: list[Chunk]) -> list[str]:
    """Text of each chunk from a model's segments (assigned by segment midpoint)."""
    cleaned, _ = collapse_repeats(segments)
    out: list[list[str]] = [[] for _ in chunks]
    for seg in cleaned:
        mid = (seg.start + seg.end) / 2
        for i, (start, end) in enumerate(chunks):
            if start - 0.05 <= mid <= end + 0.05:
                out[i].append(seg.text.strip())
                break
    return [" ".join(t) for t in out]


_KEY = re.compile(r"[^\w']+")
HESITATIONS = {"um", "uh", "umm", "uhh", "hmm", "mm", "mmm", "er", "erm", "ah", "huh"}
MIN_RESTORED_WORDS = 3


def _key(token: str) -> str:
    return _KEY.sub("", token.lower())


def restore_dropped(fused: str, first: str, min_words: int = MIN_RESTORED_WORDS) -> str:
    """Put back runs of the first model's words that the fused text left out completely.

    The LLM follows the majority and drops what only one system heard: shouts, side remarks and
    chants over the show's voice-over (measured on Survivor S51: about a third of the lines that
    Whisper alone had were gone after the fusion). Only pure omissions come back: a run of at least
    `min_words` words with no counterpart at all in the fused text. Where the fused text has other
    words in their place (a disagreement) the LLM's version stays, so nothing is said twice.
    """
    a, b = first.split(), fused.split()
    if not a or not b:
        return fused
    matcher = difflib.SequenceMatcher(a=[_key(t) for t in a], b=[_key(t) for t in b], autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        out.extend(b[j1:j2])
        if tag == "delete":
            run = a[i1:i2]
            if len(run) >= min_words and any(_key(t) not in HESITATIONS for t in run):
                out.extend(run)
    return " ".join(out)


def fuse(
    chunks: list[Chunk],
    hypotheses: dict[str, list[Segment]],
    client: OpenRouterClient,
    *,
    glossary: str = "",
    batch_size: int = 8,
    workers: int = 6,
) -> list[Segment]:
    """One fused text segment per non-empty chunk; the first model is the fallback.

    Runs of words that the LLM dropped completely are put back (`restore_dropped`) from the models whose
    recognition has word times (Whisper: it hears the shouts the others miss, and its times anchor the
    alignment); from the first model when none has word times.

    Batches are sent in parallel; each gets the first model's text of the two preceding
    chunks as context, so batches do not depend on each other.
    """
    names = list(hypotheses)
    texts = {name: chunk_texts(segs, chunks) for name, segs in hypotheses.items()}
    first = texts[names[0]]
    timed = [name for name in names if any(seg.words for seg in hypotheses[name])] or names[:1]

    def restored(text: str, j: int) -> str:
        for name in timed:
            text = restore_dropped(text, texts[name][j])
        return text

    def run(start: int) -> list[str]:
        ids = range(start, min(start + batch_size, len(chunks)))
        items = [{"id": j, **{f"asr{k + 1}": texts[n][j] for k, n in enumerate(names)}} for j in ids]
        context = " ".join(first[max(0, start - 2) : start])
        prompt = (
            (f"Glossary:\n{glossary}\n" if glossary else "")
            + f"Previous context: {context}\nChunks:\n"
            + "\n".join(json.dumps(x, ensure_ascii=False) for x in items)
        )
        try:
            answer = client.chat_json(SYSTEM_PROMPT, prompt)
        except RuntimeError as error:  # a refusal (S01E05: "content_filter" on every route) or a dead transport
            log.warning("fusion: chunks %d-%d keep the first recognition (%s)", ids[0], ids[-1], error)
            failed.append(start)
            answer = {}
        got = {c.get("id"): str(c.get("text", "")).strip() for c in answer.get("chunks", []) if isinstance(c, dict)}
        return [restored(got[j], j) if got.get(j) else first[j] for j in ids]

    failed: list[int] = []
    starts = range(0, len(chunks), batch_size)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(run, starts))
    if starts and len(failed) == len(starts):  # nothing fused at all: the LLM is down, not one passage refused
        raise RuntimeError(f"fusion: every batch failed ({len(failed)})")
    fused = [text for batch in batches for text in batch]
    return [Segment(start=s, end=e, text=t) for (s, e), t in zip(chunks, fused) if t.strip()]
