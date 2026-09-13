from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_resume import canonical_hash
from utils.shortening_review import (
    OUTCOME_ACCEPTED_CORRECTION,
    OUTCOME_INVALID,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_NOT_CHECKED,
    OUTCOME_UNCHANGED_AND_FITS,
    ReviewError,
    _proposed_action,
    build_review_state,
    make_review_queue,
    render_outcomes_markdown,
    render_review_markdown,
    validate_edit_queue,
    validate_review_state,
)


def _items(texts: list[Any], *, starts: list[int] | None = None) -> list[dict[str, Any]]:
    starts = starts or [i * 2000 for i in range(len(texts))]
    return [{"index": i, "start": start, "end": start + 1000, "text": text, "analysis": {"cue": i}} for i, (start, text) in enumerate(zip(starts, texts))]


def _snapshot(source: list[dict[str, Any]], current: list[dict[str, Any]] | None = None,
              records: list[dict[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
    config = kwargs.pop("planner_config", PlannerConfig(cps=5, max_speed=1.0))
    current = current or source
    plans = plan_subtitles(source, config)
    records = records if records is not None else [{"cue_id": plan.cue_id, "status": "pending", "history": [], "final_review": None} for plan in plans]
    return build_review_state(source_items=source, current_items=current, plans=plans, records=records,
                              planner_config=config, metadata={"test": True}, selected_ids=kwargs.pop("selected_ids", []), **kwargs)


def test_mixed_outcomes_flags_and_measured_evidence() -> None:
    source = _items(["ok", "this is a very long subtitle cue", "short timing cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [
        {"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None},
        {"cue_id": "cue-1", "status": "accepted", "history": [{"candidate": "short"}], "final_review": {"verdict": "accept", "advice": "fine"}, "candidate": "short", "final_text": "short"},
        {"cue_id": "cue-2", "status": "pending", "history": [], "timing_evidence": {"original": "short timing cue", "available_ms": plans[2].available_ms, "duration_sec": 0.5}, "final_review": None},
    ]
    current = _items(["ok", "short", "short timing cue"])
    state = build_review_state(source_items=source, current_items=current, plans=plans, records=records,
                               planner_config=config, metadata=None, selected_ids=[plans[1].cue_id])
    assert [row["outcome"] for row in state["outcomes"]] == [OUTCOME_UNCHANGED_AND_FITS, OUTCOME_ACCEPTED_CORRECTION, OUTCOME_UNCHANGED_AND_FITS]
    assert state["outcomes"][1]["text_changed"] is True
    assert state["outcomes"][2]["validation_source"] == "measured"
    assert state["outcomes"][2]["measured_fit"] is True
    assert state["source_sha256"] != state["current_sha256"]


def test_queue_only_unresolved_and_markdown_groups_all_arbitrary_text() -> None:
    source = _items(["fine", "long long long long", "bad text", "x"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [{"cue_id": plan.cue_id, "status": "pending", "history": [], "final_review": None} for plan in plans]
    records[2]["status"] = "error"
    records[2]["failure"] = {"type": "schema", "message": "bad | ` input"}
    state = _snapshot(source, records=records, planner_config=config)
    queue = make_review_queue(state)
    assert {row["cue_id"] for row in queue["entries"]} == {row["cue_id"] for row in state["outcomes"] if row["outcome"] not in {OUTCOME_UNCHANGED_AND_FITS, OUTCOME_ACCEPTED_CORRECTION}}
    assert "Protocol errors" in render_review_markdown(state, queue)
    assert "|" in render_outcomes_markdown(state)
    assert "bad \\| \\` input" in render_review_markdown(state, queue)


def test_model_evidence_binds_original_context_not_final_neighbors() -> None:
    source = _items(["one", "long original cue", "three"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [{"cue_id": plan.cue_id, "status": "pending", "history": [], "final_review": None} for plan in plans]
    records[1] = {"cue_id": "cue-1", "status": "accepted", "candidate": "short", "final_text": "short", "history": [],
                  "final_review": {"verdict": "accept", "advice": "accepted"}}
    current = _items(["one changed", "short", "three changed"])
    state = build_review_state(source_items=source, current_items=current, plans=plans, records=records,
                               planner_config=config, metadata=None, selected_ids=["cue-1"])
    row = state["outcomes"][1]
    assert row["outcome"] == OUTCOME_ACCEPTED_CORRECTION
    assert row["evidence"]["semantic"]["validated_context_kind"] == "original"
    assert row["evidence"]["semantic"]["final_context_checked"] is False
    assert row["evidence"]["semantic"]["validated_context_fingerprint"] == row["original_context_fingerprint"]
    assert row["evidence"]["semantic"]["reuse_guard_context_fingerprint"] == row["current_context_fingerprint"]


def test_reanalysis_reuses_only_matching_cache_and_force_recheck_stales_it() -> None:
    source = _items(["long source cue", "neighbor"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [{"cue_id": "cue-0", "status": "accepted", "candidate": "short", "final_text": "short", "history": [],
                "final_review": {"verdict": "accept"}}, {"cue_id": "cue-1", "status": "pending", "history": [], "final_review": None}]
    first = _snapshot(source, _items(["short", "neighbor"]), records=records, planner_config=config, selected_ids=["cue-0"])
    again = _snapshot(source, _items(["short", "neighbor"]), records=[{"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None}, {"cue_id": "cue-1", "status": "pending", "history": [], "final_review": None}],
                      planner_config=config, selected_ids=[], previous_state=first)
    assert again["outcomes"][0]["reusable_checks"]["semantic"] is True
    changed = _items(["short", "neighbor moved"], starts=[0, 2500])
    stale = _snapshot(source, changed, records=[{"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None}, {"cue_id": "cue-1", "status": "pending", "history": [], "final_review": None}],
                      planner_config=config, previous_state=first)
    assert stale["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION
    forced = _snapshot(source, _items(["short", "neighbor"]), records=[{"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None}, {"cue_id": "cue-1", "status": "pending", "history": [], "final_review": None}],
                       planner_config=config, previous_state=first, force_recheck=True)
    assert forced["outcomes"][0]["reusable_checks"] == {"semantic": False, "measured": False}
    assert forced["outcomes"][0]["measured_duration_sec"] is None


def test_reanalysis_metadata_change_invalidates_semantic_evidence() -> None:
    source = _items(["long source cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [{"cue_id": "cue-0", "status": "accepted", "candidate": "short", "final_text": "short",
                "history": [], "final_review": {"verdict": "accept"}}]
    first = build_review_state(source_items=source, current_items=_items(["short"]), plans=plans,
                               records=records, planner_config=config, metadata={"runtime": 1}, selected_ids=["cue-0"])
    changed = build_review_state(source_items=source, current_items=_items(["short"]), plans=plans,
                                 records=[{"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None}],
                                 planner_config=config, metadata={"runtime": 2}, selected_ids=[], previous_state=first)
    assert changed["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION
    assert changed["outcomes"][0]["reason"] == "evidence_stale"


def test_missing_actual_does_not_count_as_measured_and_external_fit_is_unresolved() -> None:
    source = _items(["long source cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    state = _snapshot(source, _items(["short"]), records=[{"cue_id": "cue-0", "status": "pending", "history": [], "final_review": None}],
                      planner_config=config)
    assert state["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION
    assert state["outcomes"][0]["measured_duration_sec"] is None
    assert state["outcomes"][0]["measured_fit"] is None


def test_fixed_initial_denominator_grows_for_regressions() -> None:
    source = _items(["long long cue", "fine"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    first = _snapshot(source, planner_config=config, selected_ids=["cue-0"])
    assert first["summary"]["initial_intervention_count"] >= 1
    current = _items(["long long cue", "now changed"])
    second = _snapshot(source, current, planner_config=config, previous_state=first)
    assert second["summary"]["initial_intervention_count"] == first["summary"]["initial_intervention_count"]
    assert second["summary"]["intervention_count"] >= first["summary"]["intervention_count"]
    assert second["summary"]["unresolved_outside_initial_count"] >= 1


def test_human_approval_requires_queue_context_and_adjacent_edit_stales_it() -> None:
    source = _items(["long original cue", "neighbor"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    base = _snapshot(source, _items(["short", "neighbor"]), planner_config=config)
    entry = make_review_queue(base)["entries"][0]
    approved = validate_edit_queue(base, {"schema_version": 1, "snapshot_id": base["snapshot_id"], "source_sha256": base["source_sha256"], "current_sha256": base["current_sha256"], "entries": [{**entry, "approved": True, "decision": "keep", "replacement_text": "short"}]})
    resolved = _snapshot(source, _items(["short", "neighbor"]), planner_config=config, human_approvals=approved)
    assert resolved["outcomes"][0]["validation_source"] == "human"
    adjacent = _items(["short", "neighbor changed"])
    stale = _snapshot(source, adjacent, planner_config=config, human_approvals=approved)
    assert stale["outcomes"][0]["reason"] == "context_changed"
    assert stale["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION


@pytest.mark.parametrize("bad", ["schema", "unknown", "duplicate", "budget", "context", "format", "overlong"])
def test_edit_queue_rejects_malformed_variants(bad: str) -> None:
    source = _items(["a long long cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    state = _snapshot(source, planner_config=config)
    queue = make_review_queue(state)
    entry = queue["entries"][0]
    if bad == "schema":
        queue["schema_version"] = 2
    elif bad == "unknown":
        queue["entries"] = [{**entry, "cue_id": "unknown"}]
    elif bad == "duplicate":
        queue["entries"] = [entry, copy.deepcopy(entry)]
    elif bad == "budget":
        queue["entries"][0]["max_chars"] = 999
    elif bad == "context":
        queue["entries"][0]["current_context_fingerprint"] = "stale"
    elif bad == "format":
        queue["entries"][0]["approved"] = "yes"
    elif bad == "overlong":
        queue["entries"][0].update({"approved": True, "decision": "replace", "replacement_text": "this is much too long for the cue"})
    with pytest.raises(ReviewError):
        validate_edit_queue(state, queue)


def test_normalized_lengths_lists_and_inputs_are_preserved() -> None:
    source = _items([["Hello,", "world!"]])
    before = copy.deepcopy(source)
    config = PlannerConfig(cps=13, max_speed=1.0, length_basis="normalized_tts_text")
    state = _snapshot(source, planner_config=config)
    assert source == before
    assert state["source_items"][0]["text"] == ["Hello,", "world!"]
    assert state["outcomes"][0]["source_text"] == "Hello, world!"
    assert state["outcomes"][0]["actual_chars"] == len("Hello, world!")
    validate_review_state(state)


def test_sparse_records_are_keyed_by_cue_id_and_plans_are_canonical() -> None:
    source = _items(["fine", "long source cue", "ok", "another long source cue", "fine"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(source, config)
    records = [
        {"cue_id": "cue-3", "position": 3, "status": "pending", "history": [], "final_review": None},
        {"cue_id": "cue-1", "status": "pending", "history": [], "final_review": None},
    ]
    state = _snapshot(source, records=records, planner_config=config, selected_ids=["cue-3"])
    assert state["records"] == records
    assert [row["cue_id"] for row in state["outcomes"]] == [f"cue-{i}" for i in range(5)]
    assert state["outcomes"][0]["outcome"] == OUTCOME_UNCHANGED_AND_FITS
    with pytest.raises(ReviewError):
        _snapshot(source, records=[{"cue_id": "cue-99"}], planner_config=config)


def test_empty_record_snapshot_all_fitting_targets_is_valid() -> None:
    source = _items(["a", "b", "c", "d", "e"])
    state = _snapshot(source, records=[], planner_config=PlannerConfig(cps=13, max_speed=1.0), selected_ids=[])
    assert all(row["outcome"] == OUTCOME_UNCHANGED_AND_FITS for row in state["outcomes"])
    assert all(row["validation_source"] == "prediction" for row in state["outcomes"])
    validate_review_state(state)


def test_stale_or_overlong_model_acceptance_cannot_approve_current_text() -> None:
    source = _items(["a deliberately long source cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    stale = _snapshot(
        source, _items(["short current text"]), planner_config=config,
        records=[{"cue_id": "cue-0", "status": "accepted", "final_text": "old text", "final_review": {"verdict": "accept"}}],
    )
    assert stale["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION
    overlong = _snapshot(
        source, _items(["this remains far too long for the available window"]), planner_config=config,
        records=[{"cue_id": "cue-0", "status": "accepted", "candidate": "short", "final_text": "this remains far too long for the available window", "final_review": {"verdict": "accept"}}],
    )
    assert overlong["outcomes"][0]["outcome"] != OUTCOME_ACCEPTED_CORRECTION


def test_human_approval_requires_fit_and_invalid_data_wins() -> None:
    source = _items(["a deliberately long source cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    base = _snapshot(source, _items(["short"]), planner_config=config)
    fingerprint = base["outcomes"][0]["current_context_fingerprint"]
    with pytest.raises(ReviewError):
        _snapshot(source, _items(["short"]), planner_config=config, human_approvals={"cue-0": {"text": "this is too long", "reviewed_context_fingerprint": fingerprint}})
    invalid_source = _items(["bad"])
    invalid_source[0]["end"] = invalid_source[0]["start"]
    with pytest.raises(ReviewError, match="invalid cue cue-0 cannot be approved"):
        _snapshot(invalid_source, planner_config=config, human_approvals={"cue-0": {"text": "bad", "reviewed_context_fingerprint": fingerprint}})


def test_prediction_reanalysis_and_external_fit_rules() -> None:
    source = _items(["fine"])
    first = _snapshot(source, records=[], planner_config=PlannerConfig(cps=13, max_speed=1.0))
    again = _snapshot(source, records=[], planner_config=PlannerConfig(cps=13, max_speed=1.0), previous_state=first, force_recheck=True)
    assert again["outcomes"][0]["validation_source"] == "prediction"
    edited = _snapshot(_items(["a long source cue"]), _items(["short"]), records=[], planner_config=PlannerConfig(cps=5, max_speed=1.0))
    assert edited["outcomes"][0]["outcome"] == OUTCOME_NEEDS_ATTENTION


def test_current_context_uses_displayed_neighbors_and_exact_fingerprint() -> None:
    source = _items(["long source cue", ["old", "neighbor"]])
    current = _items(["short", ["new", "neighbor"]])
    state = _snapshot(source, current, records=[], planner_config=PlannerConfig(cps=5, max_speed=1.0))
    queue = make_review_queue(state)
    entry = queue["entries"][0]
    assert entry["context_after"][0]["text"] == "new neighbor"
    assert entry["current_context_fingerprint"] == canonical_hash({"before": entry["context_before"], "after": entry["context_after"]})
    approved = validate_edit_queue(state, {**queue, "entries": [{**entry, "approved": True, "decision": "keep", "replacement_text": "short"}]})
    adjacent = _snapshot(source, _items(["short", ["changed", "neighbor"]]), records=[], planner_config=PlannerConfig(cps=5, max_speed=1.0), human_approvals=approved)
    assert adjacent["outcomes"][0]["reason"] == "context_changed"


def test_duration_is_preserved_and_measurements_are_fail_closed() -> None:
    source = _items(["long source cue"])
    source[0]["duration"] = 1.0
    current = copy.deepcopy(source)
    current[0]["duration"] = 2.0
    config = PlannerConfig(cps=5, max_speed=1.0)
    plan = plan_subtitles(source, config)[0]
    evidence = {"cue_id": "cue-0", "original": "long source cue", "available_ms": plan.available_ms, "duration_sec": 0.1}
    first = _snapshot(source, source, records=[{"cue_id": "cue-0", "status": "pending", "timing_evidence": evidence}], planner_config=config)
    state = _snapshot(source, current, records=[{"cue_id": "cue-0", "status": "pending", "timing_evidence": {**evidence, "status": "error"}}], planner_config=config, previous_state=first)
    assert state["outcomes"][0]["source_timing"]["duration"] == 1.0
    assert state["outcomes"][0]["current_timing"]["duration"] == 2.0
    assert state["outcomes"][0]["measured_fit"] is None
    assert state["outcomes"][0]["outcome"] != OUTCOME_UNCHANGED_AND_FITS


def test_effective_plan_cap_is_not_expanded_and_queue_keep_is_strict() -> None:
    source = _items(["a long source cue"])
    config = PlannerConfig(cps=13, max_speed=1.0)
    nominal = plan_subtitles(source, config)[0]
    capped = replace(nominal, max_chars=max(0, nominal.max_chars - 1), status="needs_shortening")
    state = build_review_state(source_items=source, current_items=source, plans=[capped], records=[], planner_config=config, metadata=None, selected_ids=[])
    assert state["outcomes"][0]["max_chars"] == capped.max_chars
    reanalysis = build_review_state(source_items=source, current_items=source, plans=[nominal], records=[], planner_config=config, metadata=None, selected_ids=[], previous_state=state)
    assert reanalysis["outcomes"][0]["max_chars"] == capped.max_chars
    queue = make_review_queue(state)
    entry = queue["entries"][0]
    with pytest.raises(ReviewError):
        validate_edit_queue(state, {**queue, "entries": [{**entry, "approved": True, "decision": "keep", "replacement_text": "different"}]})
    with pytest.raises(ReviewError):
        validate_edit_queue(state, {**queue, "schema_version": True})


def test_markdown_renders_every_category_and_critique_fields() -> None:
    source = _items(["long long timing cue", "another long cue", "third long cue"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    records = [
        {"cue_id": "cue-0", "status": "timing_unverified", "timing_evidence": {"error": "timing failed"}},
        {"cue_id": "cue-1", "status": "manual", "final_review": {"issues": "issue | one", "advice": "advice `two`"}},
        {"cue_id": "cue-2", "status": "protocol_misuse", "failure": "protocol"},
    ]
    state = _snapshot(source, records=records, planner_config=config)
    queue = make_review_queue(state)
    markdown = render_review_markdown(state, queue)
    assert all(markdown.count(f"### cue-{i} ") == 1 for i in range(3))
    assert "Issues: issue \\| one" in markdown and "Advice: advice \\`two\\`" in markdown
    outcomes = render_outcomes_markdown(state)
    assert "Validation source" in outcomes and "Measured fit" in outcomes


def test_reanalysis_preserves_unresolved_runner_evidence_by_exact_snapshot() -> None:
    source = _items(["long cue", "a very long subtitle cue", "another very long subtitle cue",
                     "one more very long subtitle cue", "yet another very long subtitle cue", "last very long subtitle cue"])
    source[0]["end"] = 100
    source[1]["start"] = 300
    records = [
        {"cue_id": "cue-0", "status": "short_timing", "history": [], "final_review": None},
        {"cue_id": "cue-1", "status": "timing_unverified", "history": [], "final_review": None},
        {"cue_id": "cue-2", "status": "manual", "history": [], "final_review": None},
        {"cue_id": "cue-3", "status": "transport_error", "failure": "network_timeout", "history": [], "final_review": None},
        {"cue_id": "cue-4", "status": "manual", "failure": "protocol_misuse", "history": [], "final_review": None},
        {"cue_id": "cue-5", "status": "pending", "history": [], "final_review": None},
    ]
    first = _snapshot(source, records=records, planner_config=PlannerConfig(cps=5, max_speed=1.0))
    expected = [
        (OUTCOME_NEEDS_ATTENTION, "short_timing", "short_timing"),
        (OUTCOME_NEEDS_ATTENTION, "timing_unverified", "timing_unverified"),
        (OUTCOME_NEEDS_ATTENTION, "semantic_unverified", "manual"),
        (OUTCOME_INVALID, "transport_error", "network_timeout"),
        (OUTCOME_NEEDS_ATTENTION, "protocol_error", "protocol_misuse"),
        (OUTCOME_NOT_CHECKED, "semantic_unverified", "not_checked"),
    ]
    assert [(row["outcome"], row["category"], row["reason"]) for row in first["outcomes"]] == expected

    second = _snapshot(source, records=[], planner_config=PlannerConfig(cps=5, max_speed=1.0), previous_state=first)
    assert [(row["outcome"], row["category"], row["reason"]) for row in second["outcomes"]] == expected


@pytest.mark.parametrize(
    ("initial_record", "historical_record"),
    [
        (
            {"cue_id": "cue-0", "status": "transport_error", "failure": "network_timeout"},
            {"cue_id": "cue-0", "status": "error", "failure": "network_timeout"},
        ),
        (
            {"cue_id": "cue-0", "status": "manual", "failure": "protocol_misuse"},
            {"cue_id": "cue-0", "status": "error", "failure": "protocol_misuse"},
        ),
    ],
)
def test_human_approval_wins_over_historical_runner_error(
    initial_record: dict[str, Any], historical_record: dict[str, Any],
) -> None:
    source = _items(["a deliberately long source cue"])
    current = _items(["short"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    first = _snapshot(source, current, records=[initial_record], planner_config=config)
    entry = make_review_queue(first)["entries"][0]
    approved = validate_edit_queue(
        first,
        {**make_review_queue(first), "entries": [{**entry, "approved": True, "decision": "keep", "replacement_text": "short"}]},
    )
    second = _snapshot(source, current, records=[historical_record], planner_config=config,
                       previous_state=first, human_approvals=approved)
    row = second["outcomes"][0]
    assert row["outcome"] == OUTCOME_ACCEPTED_CORRECTION
    assert row["category"] == "semantic"
    assert row["reason"] == "human_approved"
    assert row["validation_source"] == "human"


def test_protocol_misuse_uses_runner_manual_failure_shape() -> None:
    source = _items(["a deliberately long source cue"])
    state = _snapshot(source, records=[{"cue_id": "cue-0", "status": "manual", "failure": "protocol_misuse"}])
    row = state["outcomes"][0]
    assert (row["outcome"], row["category"], row["reason"]) == (
        OUTCOME_NEEDS_ATTENTION, "protocol_error", "protocol_misuse"
    )
    assert make_review_queue(state)["entries"][0]["category"] == "protocol_error"


def test_timing_reanalysis_recomputes_first_cue_budget_from_next_start() -> None:
    source = _items(["a long source cue", "neighbor"])
    config = PlannerConfig(cps=5, max_speed=1.0)
    first = _snapshot(source, planner_config=config)
    edited = _items(["a long source cue", "neighbor"], starts=[0, 1200])
    second = _snapshot(source, edited, planner_config=config, previous_state=first)
    edited_plan = plan_subtitles(edited, config)[0]
    assert second["outcomes"][0]["max_chars"] == edited_plan.max_chars
    assert second["outcomes"][0]["max_chars"] < first["outcomes"][0]["max_chars"]
    assert second["outcomes"][0]["timing_fingerprint"] != first["outcomes"][0]["timing_fingerprint"]

    third = _snapshot(source, edited, planner_config=config, previous_state=second)
    first_row, second_row = second["outcomes"][0], third["outcomes"][0]
    assert third["outcomes"][0]["max_chars"] == second["outcomes"][0]["max_chars"]
    assert third["outcomes"][0]["timing_fingerprint"] == second["outcomes"][0]["timing_fingerprint"]
    assert third["outcomes"][0]["evidence"] == second["outcomes"][0]["evidence"]
    assert third["outcomes"][0]["reusable_checks"] == second["outcomes"][0]["reusable_checks"]
    assert first_row["current_timing"] == second_row["current_timing"]


def test_config_change_recomputes_first_cue_budget() -> None:
    source = _items(["a long source cue"])
    old_config = PlannerConfig(cps=5, max_speed=1.0)
    new_config = PlannerConfig(cps=10, max_speed=1.0)
    first = _snapshot(source, planner_config=old_config)
    second = _snapshot(source, planner_config=new_config, previous_state=first)
    assert second["outcomes"][0]["max_chars"] == plan_subtitles(source, new_config)[0].max_chars
    assert second["outcomes"][0]["max_chars"] != first["outcomes"][0]["max_chars"]


def _calibration_config(slope: float) -> PlannerConfig:
    identity = {
        "engine": "vosk", "voice": 8, "rate": 1.5,
        "model_config_sha256": None, "model_fingerprint": "fingerprint",
        "normalizer": "vosk_punctuation_v1_no_ruaccent", "normalizer_sha256": "hash",
    }
    profile = {
        "status": "ok", "identity": identity,
        "normalizer": "vosk_punctuation_v1_no_ruaccent",
        "input_character_basis": "normalized_tts_text", "rate": 1.5,
        "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.0, slope]}},
        "char_linear_guard_sec": 0.0,
    }
    return PlannerConfig(cps=13.0, max_speed=1.5, length_basis="normalized_tts_text",
                         calibration_profile=profile, calibration_identity=identity)


def test_profile_change_recomputes_first_cue_budget() -> None:
    source = _items(["a long source cue"])
    old_config = _calibration_config(0.1)
    new_config = _calibration_config(0.05)
    first = _snapshot(source, planner_config=old_config)
    second = _snapshot(source, planner_config=new_config, previous_state=first)
    assert second["outcomes"][0]["max_chars"] == plan_subtitles(source, new_config)[0].max_chars
    assert second["outcomes"][0]["max_chars"] != first["outcomes"][0]["max_chars"]


def test_invalid_whitespace_is_invalid_text_and_cannot_be_approved() -> None:
    source = _items([" bad "])
    state = _snapshot(source)
    row = state["outcomes"][0]
    assert row["outcome"] == OUTCOME_INVALID
    assert row["category"] == "invalid_text"
    assert _proposed_action(row) == "fix_source_externally_then_start_new_run"
    approval = {"cue-0": {"text": " bad ", "reviewed_context_fingerprint": row["current_context_fingerprint"]}}
    with pytest.raises(ReviewError, match="invalid cue cue-0 cannot be approved"):
        _snapshot(source, human_approvals=approval)
