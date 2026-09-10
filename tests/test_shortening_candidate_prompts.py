from __future__ import annotations

import json

import pytest

from utils.shortening_candidate_prompts import (
    CANDIDATE_PROMPT_VERSION,
    build_candidate_prompt,
    build_selection_prompt,
    parse_candidates,
    parse_selection,
)
from utils.shortening_experiment import _run_candidates
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_runner import ModelSpec
from utils.llm_types import DetailedResponse


def test_candidate_prompt_is_versioned_and_masks_history() -> None:
    prompt = build_candidate_prompt(
        [{"cue_id": "cue-0", "text": "context", "is_target": True}],
        [{"cue_id": "cue-0", "original": "long text", "max_chars": 8, "history": "secret"}],
        5,
    )
    payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
    assert payload["version"] == CANDIDATE_PROMPT_VERSION
    assert "history" not in prompt
    assert payload["schema"]["additionalProperties"] is False


def test_candidates_require_complete_unique_ids_and_preserve_text() -> None:
    value = {"results": [{"cue_id": "cue-0", "status": "ok", "candidates": [
        {"candidate_id": "x", "text": " A "}, {"candidate_id": "y", "text": "B"}
    ]}]}
    parsed = parse_candidates(json.dumps(value), ["cue-0"])
    assert parsed["cue-0"]["candidates"][0]["text"] == " A "
    with pytest.raises(ValueError):
        parse_candidates(json.dumps({"results": [{"cue_id": "cue-0", "status": "cannot_shorten", "candidates": [{"candidate_id": "x", "text": "x"}]}]}), ["cue-0"])


def test_selection_parser_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        parse_selection(json.dumps({"results": [{"cue_id": "cue-0", "selected_candidate_id": None, "issues": [], "advice": "", "extra": 1}]}), ["cue-0"])


def test_candidate_dedup_marks_duplicate_oversize_rows_without_changing_rendered_text(tmp_path: object) -> None:
    from pathlib import Path
    path = Path(str(tmp_path)) / "candidates"
    items = [{"start": 0, "end": 1000, "text": "A deliberately long subtitle"}]
    plans = plan_subtitles(items, PlannerConfig(cps=5, max_speed=1.0))
    case = {"case_id": "s:cue-0", "source_id": "s", "cue_id": "cue-0", "original": plans[0].text,
            "max_chars": plans[0].max_chars, "available_ms": plans[0].available_ms,
            "context": []}

    async def fake(_: ModelSpec, __: str, user: str, schema: dict[str, object], ___: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        cue_id = payload["targets"][0]["cue_id"]
        answer = {"results": [{"cue_id": cue_id, "status": "ok", "candidates": [
            {"candidate_id": "first", "text": "too long"}, {"candidate_id": "second", "text": "too long"}
        ]}]}
        return DetailedResponse(json.dumps(answer), [])

    import asyncio
    records, _ = asyncio.run(_run_candidates(items=items, plans=plans, cases=[case], repetition=0, seed=1,
                                             generator=ModelSpec("deepseek", "deepseek-flash", "off"),
                                             request_fn=fake, output_dir=path))
    assert records[0]["candidate_counts"]["distinct_n"] == 1
    assert records[0]["candidates"][1]["rejection_reason"] == "duplicate_normalized_text"
    assert records[0]["candidates"][0]["text"] == "too long"
