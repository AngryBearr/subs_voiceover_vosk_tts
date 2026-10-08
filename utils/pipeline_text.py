"""Offline native-assistant handoff and the shared bounded Vosk speed policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from utils.length_check import prepare_length_text
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.analyze_text import join_text_lines


@dataclass(frozen=True)
class SpeedPolicy:
    base_speech_rate: float = 1.25
    max_total_speed: float = 1.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_total_speed) or not 1 <= self.max_total_speed <= 1.5:
            raise ValueError("max_total_speed must be finite and between 1 and 1.5")
        if not math.isfinite(self.base_speech_rate) or not 0 < self.base_speech_rate <= self.max_total_speed:
            raise ValueError("base_speech_rate must be finite, positive and <= max_total_speed")


def resolve_speed_policy(config: dict[str, Any]) -> SpeedPolicy:
    """Resolve one cap; the analyzer's old cap is only a compatibility input."""
    tts = config.get("tts", {})
    return SpeedPolicy(
        float(tts.get("base_speech_rate", 1.25)),
        float(tts.get("max_total_speed", config.get("analysis", {}).get("max_mismatch_ratio", 1.5))),
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_eligibility(original: str, candidate: str | None, source_status: str) -> str:
    """Mechanical proposal guard, NOT semantic approval or an audio fit check."""
    old_text = prepare_length_text(original, "normalized_tts_text")
    new_text = prepare_length_text(candidate, "normalized_tts_text") if candidate is not None else ""
    if not new_text:
        return "no_candidate"
    if source_status == "invalid":
        return "source_invalid"
    if len(new_text) >= len(old_text):
        return "not_shorter"
    if Counter(re.findall(r"\d+(?:[.,]\d+)*", new_text)) - Counter(re.findall(r"\d+(?:[.,]\d+)*", old_text)):
        return "new_numeric_representation"
    return "eligible"


def prepare_text(items: list[dict[str, Any]], policy: SpeedPolicy, cps: float) -> dict[str, Any]:
    """Freeze full source context and advisory nominal targets, without inference."""
    planner = PlannerConfig(cps=cps, max_speed=policy.max_total_speed,
                            length_basis="normalized_tts_text")
    plans = plan_subtitles(items, planner)
    settings = {**asdict(policy), "cps": cps, "gap_ms": 200,
                "length_basis": "normalized_tts_text"}
    return {
        "schema": "pipeline-native-text-v1",
        "source_sha256": json_sha256(items),
        "plan_sha256": json_sha256({"source": items, "policy": settings}),
        "policy": settings,
        "review_kind": "model_only_not_human_or_audio_approval",
        "entries": [{"cue_id": plan.cue_id, "original": copy.deepcopy(item),
                     "target": {"available_ms": plan.available_ms, "max_chars": plan.max_chars,
                                "status": plan.status, "estimate_source": plan.estimate_source},
                     "candidate": None, "review": None}
                    for item, plan in zip(items, plans)],
    }


def apply_reviewed_text(
    items: list[dict[str, Any]], artifact: Any, policy: SpeedPolicy, cps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the entire subset before replacing any text; preserve source fields."""
    prepared = prepare_text(items, policy, cps)
    if not isinstance(artifact, dict) or artifact.get("schema") != prepared["schema"]:
        raise ValueError("Expected pipeline-native-text-v1, not an experimental results file")
    for key in ("source_sha256", "plan_sha256", "policy"):
        if artifact.get(key) != prepared[key]:
            raise ValueError(f"Reviewed text {key} mismatch; prepare again for this source/policy")
    rows = artifact.get("entries")
    if not isinstance(rows, list):
        raise ValueError("entries must be a list")
    known = {row["cue_id"]: row for row in prepared["entries"]}
    seen: set[str] = set()
    chosen = copy.deepcopy(items)
    report: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each reviewed entry must be an object")
        cue_id = row.get("cue_id")
        if not isinstance(cue_id, str) or cue_id not in known or cue_id in seen:
            raise ValueError(f"Unknown or duplicate cue_id: {cue_id}")
        seen.add(cue_id)
        source = known[cue_id]
        if json_sha256(row.get("original")) != json_sha256(source["original"]):
            raise ValueError(f"Original source mismatch: {cue_id}")
        candidate, review = row.get("candidate"), row.get("review")
        if candidate is not None and not isinstance(candidate, str):
            raise ValueError(f"candidate must be text or null: {cue_id}")
        if review is not None:
            if not isinstance(review, dict) or review.get("verdict") not in {"accept", "revise", "refusal"}:
                raise ValueError(f"Invalid review: {cue_id}")
            expected_hash = text_sha256(candidate) if candidate is not None else None
            if review.get("text_sha256") != expected_hash:
                raise ValueError(f"Review candidate hash mismatch: {cue_id}")
            if review["verdict"] == "accept" and not all(
                isinstance(review.get(key), str) and review[key].strip() for key in ("model", "mode")
            ):
                raise ValueError(f"Accepted review must identify model and mode: {cue_id}")
        position = int(cue_id.split("-")[1])
        original = join_text_lines(items[position]["text"])
        new_text = prepare_length_text(candidate, "normalized_tts_text") if candidate is not None else ""
        reason = "no_candidate" if not new_text else "not_reviewed"
        if new_text and review is not None:
            reason = review["verdict"]
            if reason == "accept":
                reason = candidate_eligibility(original, candidate, source["target"]["status"])
                if reason == "eligible":
                    chosen[position]["text"] = candidate
                    reason = "applied"
        report.append({"cue_id": cue_id, "reason": reason,
                       "normalized_chars": len(new_text), "max_chars": source["target"]["max_chars"],
                       "over_advisory_target": len(new_text) > source["target"]["max_chars"],
                       "review": copy.deepcopy(review)})
    report.extend({"cue_id": cue_id, "reason": "not_supplied"} for cue_id in known if cue_id not in seen)
    return chosen, report
