from __future__ import annotations

import re
from typing import List


_QUOTE_MAP = {
    "«": '"',
    "»": '"',
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‚": "'",
    "’": "'",
    "‘": "'",
}

_DASH_MAP = {
    "—": "-",
    "–": "-",
    "‒": "-",
    "−": "-",
}


def _remove_control_and_emoji(s: str) -> str:
    # Remove control chars (Cc), format (Cf), surrogates and most emojis
    # Basic filter: keep BMP printable except control, and remove surrogate pairs.
    return "".join(
        ch
        for ch in s
        if ch.isprintable()
        and not (0xD800 <= ord(ch) <= 0xDFFF)  # surrogates
        and ord(ch) not in (0x00AD,)  # soft hyphen
    )


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    # Normalize quotes and dashes
    for k, v in _QUOTE_MAP.items():
        s = s.replace(k, v)
    for k, v in _DASH_MAP.items():
        s = s.replace(k, v)
    # Remove control/emojis and collapse whitespace
    s = _remove_control_and_emoji(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def count_symbols_for_sps(text: str) -> int:
    if not text:
        return 0
    # Remove spaces/newlines and RUAccent stress markers '+', keep punctuation
    s = text.replace(" ", "").replace("\n", "").replace("\r", "")
    s = s.replace("+", "")
    return len(s)


_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?;:…])\s+")


def split_into_chunks(text: str, max_len: int = 1200) -> List[str]:
    if not text or len(text) <= max_len:
        return [text or ""]
    # Split by sentences; then pack greedily into chunks up to max_len
    sentences = _SENT_SPLIT_RE.split(text)
    chunks: List[str] = []
    cur = []
    cur_len = 0
    for sent in sentences:
        if not sent:
            continue
        if cur_len + len(sent) + (1 if cur else 0) <= max_len:
            cur.append(sent)
            cur_len += len(sent) + (1 if cur_len > 0 else 0)
        else:
            if cur:
                chunks.append(" ".join(cur))
            # If one sentence is longer than max_len, hard-split it
            if len(sent) > max_len:
                for i in range(0, len(sent), max_len):
                    part = sent[i : i + max_len]
                    if part:
                        chunks.append(part)
                cur = []
                cur_len = 0
            else:
                cur = [sent]
                cur_len = len(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
