"""Measurement of shortened candidates: which rungs to synthesise and how rows look."""

from __future__ import annotations

from typing import Any

from voiceover.engine import PreparedText, SynthResult
from voiceover.text import syllables


def rung_key(unit_id: str, rung: int) -> str:
    """Engine key of one candidate rung (the engine's ``cue_id`` field carries it)."""
    return f"{unit_id}#r{rung}"


MANUAL_RUNG = "manual"


def manual_key(unit_id: str) -> str:
    """Engine key of a manual override text (``rung`` is the string ``"manual"`` in its row)."""
    return f"{unit_id}#manual"


def row_key(unit_id: str, rung: int | str) -> str:
    """Engine key of a durations1 row: a ladder rung or the manual text."""
    return manual_key(unit_id) if rung == MANUAL_RUNG else rung_key(unit_id, int(rung))


def manual_items(texts: dict[str, str], indices: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Engine items for manual override texts, in unit id order."""
    return [{"cue_id": manual_key(uid), "index": (indices or {}).get(uid, 0), "text": text}
            for uid, text in sorted(texts.items())]


def candidate_items(ledger: list[dict[str, Any]], indices: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Engine items for every rung that passed the hard checks, in ledger order.

    ``indices`` maps unit ids to their subtitle index (the engine passes it through to its text tools).
    """
    items: list[dict[str, Any]] = []
    for entry in ledger:
        for rung in entry["rungs"]:
            if rung["checks"]:
                continue
            items.append({"cue_id": rung_key(entry["unit_id"], rung["rung"]), "index": (indices or {}).get(entry["unit_id"], 0), "text": rung["text"]})
    return items


def measure_row(unit_id: str, rung: int | str, text: str, prepared: PreparedText, result: SynthResult) -> dict[str, Any]:
    return {"unit_id": unit_id, "rung": rung, "text": text, "tts_text": prepared.tts_text,
            "normalization": prepared.normalization, "normalization_error": prepared.error, "path": result.path,
            "sha256": result.sha256, "speech_sec": result.speech_sec, "raw_sec": result.raw_sec,
            "syllables": syllables(text), "error": result.error}


def split_key(key: str) -> tuple[str, int | str]:
    """Inverse of ``row_key``: ``("unit-1", 2)`` for ``unit-1#r2`` and ``("unit-1", "manual")`` for ``unit-1#manual``."""
    if key.endswith(f"#{MANUAL_RUNG}"):
        return key[:-len(MANUAL_RUNG) - 1], MANUAL_RUNG
    unit_id, _, rung = key.rpartition("#r")
    return unit_id, int(rung)
