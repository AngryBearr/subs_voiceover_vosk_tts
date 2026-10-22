"""Shared builders for the phase 4 tests: a three-unit episode with one deficit unit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

from voiceover.oracle import Oracle
from voiceover.placement import solve_placements
from voiceover.timeline import TimelinePolicy

SR = 22050
POLICY = TimelinePolicy()
ORACLE = Oracle("x", {"intercept": 0.1, "syllables": 0.17, "pauses": 0.0}, 50, 0.1, 0.5, 0.1, 5.0)
ORIGINAL = "Ну, это очень сложный вопрос для нас всех, правда."
ANNOTATION = {"units": [{"id": "u1", "text": "Ну,", "role": "decor"},
                        {"id": "u2", "text": "это очень сложный вопрос", "role": "core"},
                        {"id": "u3", "text": "для нас всех, правда.", "role": "support"}],
              "refs": {}, "act": "statement", "warnings": []}
RUNG_TEXTS = {1: "Это очень сложный вопрос для нас всех, правда.", 2: "Это очень сложный вопрос для нас всех.",
              3: "Это очень сложный вопрос для нас.", 4: "Это сложный вопрос.", 5: "Это вопрос."}
DROPPED = {1: ["u1"], 2: ["u1"], 3: ["u1"], 4: ["u1", "u3"], 5: ["u1", "u2", "u3"]}
# Schedule0 target of unit-0 in this episode is 4.288 s of base-rate speech.
TARGET = 4.288


def make_rung(number: int, text: str, *, checks: list[str] | None = None, dropped: list[str] | None = None,
              estimated: float | None = None) -> dict[str, Any]:
    return {"rung": number, "text": text, "syllables": 5, "estimated_sec": estimated, "fits": False,
            "checks": checks or [], "note": "", "kept": [], "dropped": dropped or [], "compressed": []}


def write_wav(path: Path, seconds: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(SR * seconds), dtype=np.float32), SR)
    return str(path)


def build_world(tmp_path: Path, rung_sec: dict[int, float], *, ok: dict[int, bool] | None = None,
                checks: dict[int, list[str]] | None = None, errors: set[int] | None = None,
                slack: float = 0.0, status: str = "resolved", estimate_choice: int = 3) -> SimpleNamespace:
    """Episode: unit-0 (5.0 s of speech, deficit), unit-1 and unit-2 (1.0 s each).

    ``rung_sec`` is the measured speech of each synthesised rung (missing rungs have no clip).
    """
    texts = [(0, ORIGINAL, 5.0), (3000, "Да, конечно.", 1.0), (8000, "Хорошо, пойдём.", 1.0)]
    cues = [{"cue_id": f"cue-{i}", "position": i, "index": i + 1, "start": start, "end": start + 2000, "text": text}
            for i, (start, text, _) in enumerate(texts)]
    units = [{"unit_id": f"unit-{i}", "cue_ids": [f"cue-{i}"], "position": i, "index": i + 1, "start": start,
              "end": start + 2000, "text": text, "scene_id": "scene-0", "merged": False, "reasons": []}
             for i, (start, text, _) in enumerate(texts)]
    rows0 = [{"unit_id": f"unit-{i}", "cue_ids": [f"cue-{i}"], "position": i, "index": i + 1, "start": start,
              "end": start + 2000, "merged": False, "scene_id": "scene-0", "display_text": text, "tts_text": text,
              "normalization": "ok", "normalization_error": None, "notes": [], "path": write_wav(
                  tmp_path / "clips" / f"u{i}.wav", seconds), "sha256": "", "raw_sec": seconds, "speech_sec": seconds,
              "edge_trimmed_ms": 0, "reduced_pauses": 0, "error": None}
             for i, (start, text, seconds) in enumerate(texts)]
    placements0, _, _ = solve_placements(rows0, POLICY, ORACLE, {})
    checks = checks or {}
    rungs = [make_rung(n, RUNG_TEXTS[n], checks=checks.get(n), dropped=DROPPED[n]) for n in range(1, 6)]
    ledger = [{"unit_id": "unit-0", "cue_ids": ["cue-0"], "scene_id": "scene-0", "text_band": "easy",
               "original": ORIGINAL, "original_syllables": 14, "current_syllables": 14, "target_syllables": 10,
               "speech_sec": 5.0, "target_base_sec": TARGET, "annotation": ANNOTATION, "rungs": rungs,
               "chosen_rung": estimate_choice, "chosen_text": RUNG_TEXTS[estimate_choice], "chosen_syllables": 8,
               "chosen_estimated_sec": 3.5, "status": status, "lossy": False, "llm_cached": False, "error": None}]
    rows1 = []
    for n, sec in rung_sec.items():
        failed = n in (errors or set())
        rows1.append({"unit_id": "unit-0", "rung": n, "text": RUNG_TEXTS[n], "tts_text": RUNG_TEXTS[n],
                      "normalization": "ok", "normalization_error": None,
                      "path": "" if failed else write_wav(tmp_path / "clips" / f"r{n}.wav", sec),
                      "sha256": "", "speech_sec": 0.0 if failed else sec, "raw_sec": sec, "syllables": 5,
                      "error": "boom" if failed else None})
    ok = ok or {}
    verdicts = [{"unit_id": "unit-0", "rung": row["rung"], "ok": ok.get(row["rung"], True), "core_lost": [],
                 "support_lost": [], "reasons": [] if ok.get(row["rung"], True) else ["core_lost"], "answers": [],
                 "llm_cached": False} for row in rows1]
    return SimpleNamespace(units=units, rows0=rows0, rows1=rows1, verdicts=verdicts, ledger=ledger,
                           placements0=placements0, policy=POLICY, oracle=ORACLE, cues=cues, fit_slack_sec=slack)


def add_manual_row(world: SimpleNamespace, tmp_path: Path, text: str, seconds: float, unit_id: str = "unit-0") -> None:
    """Append the measurement row the pipeline writes for a manual override text."""
    world.rows1.append({"unit_id": unit_id, "rung": "manual", "text": text, "tts_text": text, "normalization": "ok",
                        "normalization_error": None, "path": write_wav(tmp_path / "clips" / f"{unit_id}-manual.wav", seconds),
                        "sha256": "", "speech_sec": seconds, "raw_sec": seconds, "syllables": 5, "error": None})
