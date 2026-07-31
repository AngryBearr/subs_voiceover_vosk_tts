"""Strict parsing models for the quality-v3 original-only semantic planner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping
from typing import Any

from utils.semantic_requirements import PLANNER_ISSUE_CODES, SemanticRequirement


@dataclass(frozen=True)
class AtomicProposition:
    """One material proposition decomposed from the original source."""

    proposition_id: str
    source_span: str
    meaning: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.proposition_id, "source_span": self.source_span, "meaning": self.meaning}


@dataclass(frozen=True)
class SemanticPlan:
    """Validated planner output for one source item."""

    index: int
    source_text: str
    propositions: tuple[AtomicProposition, ...]
    requirements: tuple[SemanticRequirement, ...]
    planner_version: str = "quality-v3"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_text": self.source_text,
            "propositions": [proposition.as_dict() for proposition in self.propositions],
            "requirements": [requirement.as_dict() for requirement in self.requirements],
            "planner_version": self.planner_version,
        }


PLANNER_PROMPT = """You are the quality-v3 original-only semantic planner. Return exactly this JSON shape:
{"results":[{"index":1,"source_text":"...","propositions":[{"id":"p1","source_span":"...","meaning":"..."}],"requirements":[{"issue":"...","source_anchor":"...","requirement":"..."}]}]}.
Every result has exactly these four keys: index, source_text, propositions, requirements. No extra keys, including planner_version.
Every proposition has exactly id, source_span, meaning. Every requirement has exactly issue, source_anchor, requirement.
Allowed issue values only: """ + ", ".join(sorted(PLANNER_ISSUE_CODES)) + """.
source_span and source_anchor must each be an exact non-empty substring of that result's source_text. Keep stable unique ids.
For every indexed original source, decompose every material clause into concise atomic propositions (maximum 4 propositions)
and requirements (maximum 12). Preserve agent/person, predicate, object, polarity, tense, modality, speech act,
alternatives, causal and comparison relations, references, spatial relations, and clause boundaries.
Context may resolve references only; it must not add facts. Do not inspect or infer from any candidate, edit history,
lineage, critic, editor, or review result. The planner input contains original text only.
Compact valid example:
{"results":[{"index":7,"source_text":"Им должны уйти.","propositions":[{"id":"p1","source_span":"Им должны уйти","meaning":"The people must leave."}],"requirements":[{"issue":"agent","source_anchor":"Им","requirement":"Preserve the agent."}]}]}
"""


def _normalise(value: str) -> str:
    return value.lower().replace("ё", "е")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def parse_semantic_plans(response: str | None, expected: Mapping[int, str]) -> dict[int, SemanticPlan]:
    """Parse and validate the planner's complete strict-JSON response."""
    if response is None:
        raise ValueError("planner response is missing")
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("planner response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("planner response must have exact top-level key results")

    parsed: dict[int, SemanticPlan] = {}
    for raw_result in payload["results"]:
        if not isinstance(raw_result, dict) or set(raw_result) != {"index", "source_text", "propositions", "requirements"}:
            raise ValueError("result has unknown or missing fields")
        index = raw_result["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected or index in parsed:
            raise ValueError("result index is invalid, unexpected, or duplicated")
        source_text = _require_string(raw_result["source_text"], "source_text")
        if source_text != expected[index]:
            raise ValueError("source_text does not match expected source")
        raw_propositions = raw_result["propositions"]
        if not isinstance(raw_propositions, list) or not raw_propositions:
            raise ValueError("propositions must be a nonempty list")
        propositions: list[AtomicProposition] = []
        proposition_ids: set[str] = set()
        normalised_source = _normalise(source_text)
        for raw_proposition in raw_propositions:
            if not isinstance(raw_proposition, dict) or set(raw_proposition) != {"id", "source_span", "meaning"}:
                raise ValueError("proposition has unknown or missing fields")
            proposition_id = _require_string(raw_proposition["id"], "proposition id")
            source_span = _require_string(raw_proposition["source_span"], "source_span")
            meaning = _require_string(raw_proposition["meaning"], "meaning")
            if proposition_id in proposition_ids or _normalise(source_span) not in normalised_source:
                raise ValueError("proposition id is duplicated or source_span is absent")
            proposition_ids.add(proposition_id)
            propositions.append(AtomicProposition(proposition_id, source_span, meaning))

        raw_requirements = raw_result["requirements"]
        if not isinstance(raw_requirements, list):
            raise ValueError("requirements must be a list")
        requirements: list[SemanticRequirement] = []
        requirement_keys: set[tuple[str, str]] = set()
        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, dict) or set(raw_requirement) != {"issue", "source_anchor", "requirement"}:
                raise ValueError("requirement has unknown or missing fields")
            issue = _require_string(raw_requirement["issue"], "issue")
            source_anchor = _require_string(raw_requirement["source_anchor"], "source_anchor")
            requirement = _require_string(raw_requirement["requirement"], "requirement")
            key = (issue, _normalise(source_anchor))
            if issue not in PLANNER_ISSUE_CODES or _normalise(source_anchor) not in normalised_source or key in requirement_keys:
                raise ValueError("requirement is invalid or duplicated")
            requirement_keys.add(key)
            requirements.append(SemanticRequirement(issue, source_anchor, requirement))
        parsed[index] = SemanticPlan(index, source_text, tuple(propositions), tuple(requirements))

    if set(parsed) != set(expected):
        raise ValueError("planner response is incomplete")
    return parsed
