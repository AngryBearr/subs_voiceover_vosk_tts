"""Tests for versioned shortening prompts and strict parsers."""

from __future__ import annotations

import json

import pytest

from utils.shortening_prompts import (
    ARBITER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_generation_prompt,
    build_review_prompt,
    generation_schema,
    parse_generation,
    parse_review,
    review_schema,
)


def test_versioned_prompts_and_strict_schemas() -> None:
    assert PROMPT_VERSION == "shortening-v2"
    assert "FULL essential meaning" in GENERATOR_SYSTEM_PROMPT
    assert "JSON only" in ARBITER_SYSTEM_PROMPT
    for schema, required in ((generation_schema(), {"cue_id", "text", "status"}), (review_schema(), {"cue_id", "verdict", "issues", "advice"})):
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]["results"]["items"]["properties"]) == required
        assert schema["properties"]["results"]["items"]["additionalProperties"] is False


def test_prompt_payload_fields_and_no_history() -> None:
    context = [{"cue_id": "cue-0", "text": "before", "is_target": False, "provider": "omit"}]
    generation = build_generation_prompt(
        context,
        [{"cue_id": "cue-1", "original": "original", "max_chars": 5, "length_basis": "normalized_tts_text", "prior_candidate": "old", "feedback": "short"}],
    )
    review = build_review_prompt(
        context,
        [{"cue_id": "cue-1", "original": "original", "candidate": "new", "max_chars": 5, "actual_chars": 3, "length_basis": "normalized_tts_text", "history": "omit"}],
    )
    assert "provider" not in generation
    assert "history" not in review
    assert "prior_candidate" in generation
    assert "actual_chars" in review
    generation_payload = json.loads(generation.split("Input JSON:\n", 1)[1])
    review_payload = json.loads(review.split("Input JSON:\n", 1)[1])
    assert generation_payload["schema"] == generation_schema()
    assert review_payload["schema"] == review_schema()
    assert generation_payload["targets"][0]["length_basis"] == "normalized_tts_text"
    assert review_payload["targets"][0]["length_basis"] == "normalized_tts_text"


def test_generation_parse_and_deterministic_order() -> None:
    payload = {"results": [
        {"cue_id": "b", "text": " B\n\t", "status": "ok"},
        {"cue_id": "a", "text": "A", "status": "cannot_shorten"},
    ]}
    parsed = parse_generation(json.dumps(payload), ["a", "b"])
    assert list(parsed) == ["a", "b"]
    assert parsed["a"]["text"] == "A"
    assert parsed["b"]["text"] == " B\n\t"


def test_review_parse_accept_requires_empty_issues() -> None:
    good = {"results": [{"cue_id": "a", "verdict": "accept", "issues": [], "advice": ""}]}
    assert parse_review(json.dumps(good), ["a"])["a"]["verdict"] == "accept"
    bad = {"results": [{"cue_id": "a", "verdict": "accept", "issues": ["issue"], "advice": ""}]}
    with pytest.raises(ValueError, match="empty issues"):
        parse_review(json.dumps(bad), ["a"])


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"cue_id": "a", "text": "", "status": "ok"}]},
        {"results": [{"cue_id": "a", "text": " \t\n", "status": "ok"}]},
        {"results": [{"cue_id": "a", "text": "safe\x00text", "status": "ok"}]},
        {"results": [{"cue_id": "a", "text": "safe\x01text", "status": "cannot_shorten"}]},
        {"results": [{"cue_id": "a", "text": "A", "status": "ok", "extra": 1}]},
        {"results": [{"cue_id": "a", "text": "A", "status": "ok"}, {"cue_id": "a", "text": "B", "status": "ok"}]},
        {"results": [{"cue_id": "other", "text": "A", "status": "ok"}]},
    ],
)
def test_generation_rejects_malformed_results(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_generation(json.dumps(payload), ["a"])


def test_parsers_reject_code_fences_unknown_top_level_and_non_strings() -> None:
    with pytest.raises(ValueError):
        parse_generation("```json {} ```", ["a"])
    with pytest.raises(ValueError):
        parse_generation(json.dumps({"results": [], "extra": 1}), [])
    payload = {"results": [{"cue_id": "a", "text": "A", "status": True}]}
    with pytest.raises(ValueError):
        parse_generation(json.dumps(payload), ["a"])


@pytest.mark.parametrize("parser", [parse_generation, parse_review])
def test_strict_and_partial_parsers_fail_closed_on_repeated_raw_cue_id_keys(parser: object) -> None:
    raw = '{"results":[{"cue_id":"a","cue_id":"a","text":"A","status":"ok"}]}'
    if parser is parse_review:
        raw = '{"results":[{"cue_id":"a","cue_id":"a","verdict":"accept","issues":[],"advice":""}]}'
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        parser(raw, ["a"])  # type: ignore[operator]


@pytest.mark.parametrize("parser", [parse_generation, parse_review])
def test_strict_parsers_validate_expected_id_type_and_shape(parser: object) -> None:
    for expected_ids in (("a",), ["a", "a"], ["a", 1]):
        with pytest.raises(ValueError):
            parser('{"results":[]}', expected_ids)  # type: ignore[operator]
    with pytest.raises(ValueError):
        parser('{"results":{}}', ["a"])  # type: ignore[operator]
