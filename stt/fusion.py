"""LLM fusion of several ASR hypotheses of the same VAD chunks (generative error correction).

Different ASR models make different mistakes; an LLM sees all hypotheses of a chunk plus the
show glossary and writes the most likely transcript. The result is text per chunk (no word
times) and goes through forced alignment (`stt.align`) afterwards.
"""

from __future__ import annotations

import json
import logging
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

    Batches are sent in parallel; each gets the first model's text of the two preceding
    chunks as context, so batches do not depend on each other.
    """
    names = list(hypotheses)
    texts = {name: chunk_texts(segs, chunks) for name, segs in hypotheses.items()}
    first = texts[names[0]]

    def run(start: int) -> list[str]:
        ids = range(start, min(start + batch_size, len(chunks)))
        items = [{"id": j, **{f"asr{k + 1}": texts[n][j] for k, n in enumerate(names)}} for j in ids]
        context = " ".join(first[max(0, start - 2) : start])
        prompt = (
            (f"Glossary:\n{glossary}\n" if glossary else "")
            + f"Previous context: {context}\nChunks:\n"
            + "\n".join(json.dumps(x, ensure_ascii=False) for x in items)
        )
        answer = client.chat_json(SYSTEM_PROMPT, prompt)
        got = {c.get("id"): str(c.get("text", "")).strip() for c in answer.get("chunks", []) if isinstance(c, dict)}
        return [got.get(j) or first[j] for j in ids]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(run, range(0, len(chunks), batch_size)))
    fused = [text for batch in batches for text in batch]
    return [Segment(start=s, end=e, text=t) for (s, e), t in zip(chunks, fused) if t.strip()]
