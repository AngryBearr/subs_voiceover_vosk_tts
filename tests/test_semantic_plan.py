"""Tests for strict quality-v3 semantic plan parsing."""

from __future__ import annotations

import json

import pytest

from utils.semantic_plan import AtomicProposition, SemanticPlan, parse_semantic_plans


def test_parse_semantic_plan_and_round_trip() -> None:
    response = json.dumps({"results": [{"index": 1, "source_text": "Им нужно будет уйти.", "propositions": [{"id": "p1", "source_span": "Им нужно будет уйти", "meaning": "The dative person must leave in the future."}], "requirements": [{"issue": "agent", "source_anchor": "Им нужно", "requirement": "Preserve the person."}]}]})
    plan = parse_semantic_plans(response, {1: "Им нужно будет уйти."})[1]
    assert plan == SemanticPlan(1, "Им нужно будет уйти.", (AtomicProposition("p1", "Им нужно будет уйти", "The dative person must leave in the future."),), plan.requirements)
    assert plan.as_dict()["propositions"][0]["id"] == "p1"


@pytest.mark.parametrize("response", [None, "{}", '{"results": []}', '{"results": [{"index": 1}]}'])
def test_parser_rejects_missing_or_partial_responses(response: str | None) -> None:
    with pytest.raises(ValueError):
        parse_semantic_plans(response, {1: "Источник."})


def test_parser_rejects_unknown_fields_and_duplicate_requirement() -> None:
    result = {"index": 1, "source_text": "Источник.", "propositions": [{"id": "p", "source_span": "Источник", "meaning": "source"}], "requirements": [{"issue": "agent", "source_anchor": "Источник", "requirement": "keep"}, {"issue": "agent", "source_anchor": "источник", "requirement": "keep"}]}
    with pytest.raises(ValueError):
        parse_semantic_plans(json.dumps({"results": [result]}), {1: "Источник."})


@pytest.mark.parametrize(
    ("issue", "source_anchor"),
    [("person", "Источник"), ("agent", "отсутствует")],
)
def test_parser_rejects_unknown_issue_and_missing_source_anchor(issue: str, source_anchor: str) -> None:
    result = {
        "index": 1,
        "source_text": "Источник.",
        "propositions": [{"id": "p", "source_span": "Источник", "meaning": "source"}],
        "requirements": [{"issue": issue, "source_anchor": source_anchor, "requirement": "keep"}],
    }
    with pytest.raises(ValueError):
        parse_semantic_plans(json.dumps({"results": [result]}), {1: "Источник."})
