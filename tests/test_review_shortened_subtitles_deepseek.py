"""Offline tests for the separated DeepSeek Pro quality stages."""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

import utils.review_shortened_subtitles_deepseek as reviewer
from utils.review_shortened_subtitles_deepseek import (
    CRITIC_PROMPT,
    COMPACTION_PROMPT,
    EDITOR_PROMPT,
    VERIFIER_PROMPT,
    ReviewTarget,
    build_context,
    build_required_checks,
    build_parser,
    parse_assessments,
    parse_verifications,
    pro_usage_summary,
    review_max_tokens,
    review_subtitles,
    review_validation_error,
    route_targets,
    semantic_risk_issues,
    select_changed_targets,
)
from utils.semantic_plan import PLANNER_PROMPT
from utils.shortening_domain import CalibratedDurationModel


def item(index: int, text: str, duration: float = 10.0) -> dict[str, Any]:
    return {"index": index, "text": [text], "analysis": {"available_duration_sec": duration,
            "effective_duration_sec": duration}}


def duration_model(char_seconds: float = 1.0) -> CalibratedDurationModel:
    return CalibratedDurationModel(
        intercept_sec=0.0, seconds_per_budget_char=char_seconds, seconds_per_punctuation=0.0,
        measurement="edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction",
        voice="test-voice", rate="+0%", sample_count=30, episode_count=2,
        validation_method="leave_one_episode_out", validation_mae_sec=0.1,
        validation_rmse_sec=0.2, validation_r_squared=0.8,
    )


def write_duration_profile(path: Any) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "profile_type": "calibrated_post_silence_duration",
        "measurement": "edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction",
        "voice": "test-voice", "rate": "+0%",
        "coefficients": {"intercept_sec": 0.0, "seconds_per_budget_char": 1.0,
                         "seconds_per_punctuation": 0.0},
        "training": {"sample_count": 30, "episode_count": 2},
        "validation": {"method": "leave_one_episode_out", "mae_sec": 0.1,
                        "rmse_sec": 0.2, "r_squared": 0.8},
    }), encoding="utf-8")


class FakeCompletions:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses, self.calls = responses, []
    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_cache_hit_tokens=1, prompt_cache_miss_tokens=2,
                                completion_tokens=3, completion_tokens_details=SimpleNamespace(reasoning_tokens=4))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))], usage=usage)


class FakeClient:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def assessment(index: int, verdict: str, issues: list[str]) -> str:
    return json.dumps({"results": [{"index": index, "verdict": verdict, "issues": issues}]})


def editor(index: int, candidate: str, decision: str = "candidate") -> str:
    return json.dumps({"results": [{"index": index, "decision": decision, "candidate": candidate}]})


def verification(index: int, verdict: str = "pass", issues: list[str] | None = None,
                 required: list[str] | None = None, failed: set[str] | None = None) -> str:
    required, failed = required or [], failed or set()
    return json.dumps({"results": [{"index": index, "verdict": verdict, "issues": issues or [],
        "required_checks": [{"issue": issue, "verdict": "fail" if issue in failed else "pass"}
                              for issue in required]}]})


def multi_assessment(values: list[tuple[int, str, list[str]]]) -> str:
    return json.dumps({"results": [
        {"index": index, "verdict": verdict, "issues": issues}
        for index, verdict, issues in values
    ]})


def multi_editor(values: list[tuple[int, str, str]]) -> str:
    return json.dumps({"results": [
        {"index": index, "decision": decision, "candidate": candidate}
        for index, candidate, decision in values
    ]})


def multi_verification(values: list[tuple[int, str, list[str], list[str], set[str]]]) -> str:
    return json.dumps({"results": [
        {"index": index, "verdict": verdict, "issues": issues,
         "required_checks": [{"issue": issue, "verdict": "fail" if issue in failed else "pass"}
                              for issue in required]}
        for index, verdict, issues, required, failed in values
    ]})


def unit_item(index: int, text: str) -> dict[str, Any]:
    value = item(index, text)
    value.update({"start": (index - 1) * 1100, "end": index * 1000})
    return value


def planner_response(index: int, source_text: str,
                     propositions: list[tuple[str, str]],
                     requirements: list[tuple[str, str]]) -> str:
    return json.dumps({"results": [{
        "index": index,
        "source_text": source_text,
        "propositions": [{"id": f"p{position}", "source_span": span, "meaning": meaning}
                         for position, (span, meaning) in enumerate(propositions, 1)],
        "requirements": [{"issue": issue, "source_anchor": anchor,
                           "requirement": f"Preserve {issue}."}
                          for issue, anchor in requirements],
    }]})


def test_strict_assessment_parser_rejects_formal_errors() -> None:
    assert parse_assessments(assessment(1, "pass", []), {1}) == {1: ("pass", [])}
    assert parse_assessments(assessment(1, "fail", ["agent", "agent"]), {1}) == {1: ("fail", ["agent"])}
    assert parse_assessments(assessment(1, "pass", ["agent"]), {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["bogus"]}]}', {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["agent"],"x":1}]}', {1}) == {}
    assert parse_assessments('{"results":[]}', {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["agent"]},{"index":1,"verdict":"fail","issues":["agent"]}]}', {1}) == {}
    assert parse_verifications(verification(1, required=["agent"]), {1: ["agent"]})[1][2] == {"agent": "pass"}
    assert parse_verifications(verification(1), {1: ["agent"]}) == {}
    assert parse_verifications(verification(1, required=["spatial"]), {1: ["spatial"]})[1][2] == {"spatial": "pass"}


def test_selection_union_includes_remaining_critical_and_366() -> None:
    original = [item(365, "same"), item(366, "unchanged critical")]
    current = [item(365, "changed"), item(366, "unchanged critical")]
    current[1]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    assert [target.index for target in select_changed_targets(original, current)] == [365, 366]


def test_ratio_selection_and_compatibility_route_helper() -> None:
    original = [item(1, "Long unchanged phrase"), item(2, "Changed original phrase")]
    current = [item(1, "Long unchanged phrase"), item(2, "Changed phrase")]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 1.0,
                                    "extended_mismatch_ratio": 1.6})
    targets = select_changed_targets(original, current)
    assert [target.index for target in targets] == [1, 2]
    routed = route_targets(targets, "auto", 99.0)
    assert [target.index for target in routed["high"]] == [1]
    assert [target.index for target in routed[None]] == [2]


def test_formal_validation_numbers_negation_and_short_natural_lines() -> None:
    target = ReviewTarget(0, 1, "Это не 42 важных слова", "Это не 42 слова", 30, 2, 20, 2, False)
    assert review_validation_error(target, "Не 42 слова") is None
    assert review_validation_error(target, "Это не слова") == "numbers_missing:42"
    assert review_validation_error(target, "Это 42 слова") == "negations_missing"
    polarity = ReviewTarget(0, 3, "Нет, пользы не будет", "Пользы нет", 30, 2, 20, 3, False)
    assert review_validation_error(polarity, "Пользы нет") is None
    assert review_validation_error(polarity, "Пользы будет") == "negations_missing"
    compact = ReviewTarget(0, 4, "Осторожно, острое лезвие", "Лезвию каюк", 20, 2, 20, 3, False)
    assert review_validation_error(compact, "Лезвию каюк") is None
    slash = ReviewTarget(0, 5, "Резать или шлифовать", "Резать/шлифовать", 30, 2, 20, 2, False)
    assert review_validation_error(slash, "Резать/шлифовать") == "slash_added"
    original_slash = ReviewTarget(0, 6, "Ввод/вывод работает", "Ввод/вывод", 30, 2, 20, 2, False)
    assert review_validation_error(original_slash, "Ввод/вывод") is None


def test_duration_gate_is_strict_and_calibrated() -> None:
    model = duration_model()
    target = ReviewTarget(0, 1, "Original", "short", 100, 5.0, 0, 1, False)
    assert review_validation_error(target, "1234", duration_profile=model) is None
    assert review_validation_error(target, "12345", duration_profile=model) is None
    assert review_validation_error(target, "123456", duration_profile=model) == "duration_too_long"
    assert review_validation_error(target, "123456", duration_profile=model, duration_fit_ratio=2.0) is None
    zero = ReviewTarget(0, 2, "Original", "short", 100, 0.0, 0, 1, False)
    assert review_validation_error(zero, "x", duration_profile=model) == "duration_too_long"


def test_duration_profile_cli_flags_and_legacy_signature() -> None:
    args = build_parser().parse_args(["original.json", "shortened.json"])
    assert args.duration_profile is None and args.duration_fit_ratio == 1.0
    assert args.multi_cue_editor is False
    assert build_parser().parse_args(
        ["original.json", "shortened.json", "--multi-cue-editor"]
    ).multi_cue_editor is True
    args = build_parser().parse_args(["original.json", "shortened.json", "--duration-profile", "profile.json",
                                      "--duration-fit-ratio", "1.25"])
    assert str(args.duration_profile) == "profile.json" and args.duration_fit_ratio == 1.25
    target = ReviewTarget(0, 1, "Original", "short", 100, 1, 0, 1, False)
    assert review_validation_error(target, "short") is None


def test_multi_cue_editor_commits_adjacent_targets_atomically_and_freezes_evidence() -> None:
    original = [
        unit_item(1, "Мы говорим,"),
        unit_item(2, "что это важно"),
        unit_item(3, "сегодня."),
    ]
    current = [
        unit_item(1, "Говорим,"),
        unit_item(2, "это важно"),
        unit_item(3, "сегодня."),
    ]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
        multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        multi_verification([
            (1, "pass", [], ["object"], set()), (2, "pass", [], ["object"], set()),
        ]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_multi_cue_editor=True,
    ))
    assert len(client.chat.completions.calls) == 3
    editor_call = client.chat.completions.calls[1]
    assert "unit_context:" in editor_call["messages"][1]["content"]
    unit_payload = json.loads(editor_call["messages"][1]["content"].split("unit_context:", 1)[1])
    assert unit_payload["cue_indices"] == [1, 2, 3]
    assert unit_payload["editable_indices"] == [1, 2]
    assert unit_payload["evidence_indices"] == [3]
    assert [entry["index"] for entry in unit_payload["cues"]] == [1, 2, 3]
    assert [entry["index"] for entry in unit_payload["cues"] if entry["editable"]] == [1, 2]
    assert [entry["text"] for entry in output] == [["Говорим кратко,"], ["Важно"], ["сегодня."]]
    assert report["unresolved_indices"] == []


@pytest.mark.parametrize("response", [
    multi_editor([(1, "one", "candidate")]),
    multi_editor([(1, "one", "unresolved"), (2, "two", "candidate")]),
    multi_editor([(1, "one/", "candidate"), (2, "two", "candidate")]),
])
def test_multi_cue_editor_rejects_incomplete_unresolved_or_formal_invalid_as_one_transaction(
    response: str,
) -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "это важно")]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]), response,
        multi_verification([
            (1, "pass", [], ["object"], set()), (2, "pass", [], ["object"], set()),
        ]),
        multi_editor([(1, "repair", "unresolved"), (2, "repair", "unresolved")]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_multi_cue_editor=True,
    ))
    assert [entry["text"] for entry in output] == [["Говорим,"], ["это важно"]]
    assert all("editor" in report["outcomes"][index - 1]["stage_errors"] for index in (1, 2))


def test_multi_cue_editor_evidence_editor_target_downgrades_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно"), unit_item(3, "и продолжаем")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "что это важно"), unit_item(3, "продолжаем")]
    current[1]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "pass", []), (3, "fail", ["object"])]),
        json.dumps({"results": [
            {"index": 1, "decision": "candidate", "candidate": "Говорим кратко,"},
            {"index": 2, "decision": "candidate", "candidate": "Важно"},
        ]}),
        editor(3, "Продолжаем"),
        multi_verification([
            (1, "pass", [], ["object"], set()), (2, "pass", [], ["budget"], set()),
            (3, "pass", [], ["object"], set()),
        ]),
    ])
    asyncio.run(review_subtitles(original, current, "key", client=client, enable_multi_cue_editor=True))
    editor_prompts = [call["messages"][1]["content"] for call in client.chat.completions.calls
                      if call["messages"][0]["content"] == EDITOR_PROMPT]
    assert editor_prompts and all("unit_context:" not in prompt for prompt in editor_prompts)
    assert len(editor_prompts) >= 2


def test_atomic_repair_without_initial_verifier_assessment_rolls_back_to_uncertain() -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Мы/говорим,"), unit_item(2, "что/это важно")]
    client = FakeClient([
        multi_assessment([(1, "pass", []), (2, "pass", [])]),
        multi_editor([(1, "", "unresolved"), (2, "", "unresolved")]),
        multi_editor([(1, "Мы говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        verification(1),
        verification(2, "fail", ["grammar"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_multi_cue_editor=True,
    ))
    assert [entry["text"] for entry in output] == [["Мы/говорим,"], ["что/это важно"]]
    outcomes = {outcome["index"]: outcome for outcome in report["outcomes"]}
    assert report["unresolved_indices"] == [1, 2]
    for index in (1, 2):
        assert outcomes[index]["verifier_verdict"] == "uncertain"
        assert outcomes[index]["verifier_history"] == []


def test_multi_cue_editor_oversized_group_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "это важно")]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
        multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        multi_verification([
            (1, "pass", [], ["object"], set()), (2, "pass", [], ["object"], set()),
        ]),
    ])
    original_estimator = reviewer.estimate_input_tokens
    monkeypatch.setattr(reviewer, "estimate_input_tokens", lambda system, prompt:
                        50_001 if "unit_context:" in prompt else original_estimator(system, prompt))
    asyncio.run(review_subtitles(original, current, "key", client=client,
                                 enable_multi_cue_editor=True))
    editor_prompts = [call["messages"][1]["content"] for call in client.chat.completions.calls
                      if call["messages"][0]["content"] == EDITOR_PROMPT]
    assert len(editor_prompts) == 1
    assert "unit_context:" not in editor_prompts[0]


def test_atomic_repair_uses_full_unit_and_only_failed_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "это важно")]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
        multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        multi_verification([
            (1, "fail", ["object"], ["object"], {"object"}),
            (2, "pass", [], ["object"], set()),
        ]),
        multi_editor([(1, "Говорим очень кратко,", "candidate")]),
        verification(1, required=["object"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_multi_cue_editor=True,
    ))
    repair_call = client.chat.completions.calls[3]
    repair_payload = json.loads(repair_call["messages"][1]["content"].split("unit_context:", 1)[1])
    editor_payload = json.loads(
        client.chat.completions.calls[1]["messages"][1]["content"].split("unit_context:", 1)[1]
    )
    assert repair_payload["unit_id"] == editor_payload["unit_id"]
    assert repair_payload["cue_indices"] == [1, 2]
    assert repair_payload["editable_indices"] == [1]
    assert repair_payload["evidence_indices"] == [2]
    assert output[0]["text"] == ["Говорим очень кратко,"]
    assert output[1]["text"] == ["Важно"]
    assert report["unresolved_indices"] == []


def test_atomic_repair_failure_rolls_back_every_member_without_key_error() -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "это важно")]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
        multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        multi_verification([
            (1, "fail", ["object"], ["object"], {"object"}),
            (2, "fail", ["object"], ["object"], {"object"}),
        ]),
        multi_editor([(1, "Говорим очень кратко,", "candidate"),
                      (2, "Важно очень кратко", "candidate")]),
        verification(1, required=["object"]),
        verification(2, "fail", ["object"], ["object"], {"object"}),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_multi_cue_editor=True,
    ))
    assert [entry["text"] for entry in output] == [["Говорим кратко,"], ["Важно"]]
    outcomes = {entry["index"]: entry for entry in report["outcomes"]}
    for index in (1, 2):
        assert outcomes[index]["lineage"]["repair"] is None
        assert len(outcomes[index]["verifier_history"]) == 1
        assert outcomes[index]["verifier_required_checks"] == {"object": "fail"}
    assert report["unresolved_indices"] == [1, 2]


def test_atomic_repair_oversized_group_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    original = [unit_item(1, "Мы говорим,"), unit_item(2, "что это важно")]
    current = [unit_item(1, "Говорим,"), unit_item(2, "это важно")]
    client = FakeClient([
        multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
        multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
        multi_verification([
            (1, "fail", ["object"], ["object"], {"object"}),
            (2, "fail", ["object"], ["object"], {"object"}),
        ]),
        editor(1, "Говорим очень кратко,"), editor(2, "Важно очень кратко"),
        verification(1, required=["object"]), verification(2, required=["object"]),
    ])
    original_estimator = reviewer.estimate_input_tokens
    monkeypatch.setattr(reviewer, "estimate_input_tokens", lambda system, prompt:
                        50_001 if "unit_context:" in prompt else original_estimator(system, prompt))
    asyncio.run(review_subtitles(original, current, "key", client=client,
                                 enable_multi_cue_editor=True))
    repair_prompts = [call["messages"][1]["content"] for call in client.chat.completions.calls
                      if call["messages"][0]["content"] == EDITOR_PROMPT][1:]
    assert repair_prompts and all("unit_context:" not in prompt for prompt in repair_prompts)
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=FakeClient([
            multi_assessment([(1, "fail", ["object"]), (2, "fail", ["object"])]),
            multi_editor([(1, "Говорим кратко,", "candidate"), (2, "Важно", "candidate")]),
            multi_verification([
                (1, "fail", ["object"], ["object"], {"object"}),
                (2, "fail", ["object"], ["object"], {"object"}),
            ]),
            editor(1, "Говорим очень кратко,"), editor(2, "Важно очень кратко"),
            verification(1, required=["object"]), verification(2, required=["object"]),
        ]), enable_multi_cue_editor=True,
    ))
    assert report["unresolved_indices"] == []
    assert all("repair" not in outcome["stage_errors"] for outcome in report["outcomes"])
    assert [entry["text"] for entry in output] == [["Говорим очень кратко,"], ["Важно очень кратко"]]


@pytest.mark.parametrize("fit_ratio", [True, 0, -1, float("inf"), float("nan")])
def test_duration_diagnostic_rejects_invalid_fit_ratio(fit_ratio: Any) -> None:
    target = ReviewTarget(0, 1, "Original", "short", 100, 1.0, 0, 1, False)
    with pytest.raises(ValueError, match="duration_fit_ratio"):
        reviewer.duration_diagnostic(target, "short", duration_model(), fit_ratio)


def test_duration_diagnostic_is_absent_without_profile() -> None:
    target = ReviewTarget(0, 1, "Original", "short", 100, 1.0, 0, 1, False)
    assert reviewer.duration_diagnostic(target, "short", None, 1.0) is None


def test_workflow_initial_duration_invalid_is_recovery_only() -> None:
    original, current = [item(1, "Original phrase", 10.0)], [item(1, "12345678901", 10.0)]
    client = FakeClient([assessment(1, "pass", []), editor(1, "", "unresolved"), editor(1, "", "unresolved")])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    outcome = report["outcomes"][0]
    assert report["unresolved_indices"] == [1]
    assert outcome["outcome"] == "unresolved"
    assert outcome["duration_diagnostic"]["predicted_duration_sec"] == 11.0
    assert len(client.chat.completions.calls) == 3
    assert report["stages"]["compaction"]["requests"] == 0


def test_workflow_duration_invalid_editor_is_rejected_without_compaction() -> None:
    original, current = [item(1, "Original phrase with object", 10.0)], [item(1, "short", 10.0)]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    invalid = "12345678901"
    client = FakeClient([assessment(1, "fail", ["object"]), editor(1, invalid), editor(1, "", "unresolved")])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    outcome = report["outcomes"][0]
    assert outcome["stage_errors"]["editor"] == "formal_error:duration_too_long"
    assert outcome["lineage"]["rejected"][0]["reason"] == "duration_too_long"
    assert set(outcome["lineage"]["rejected"][0]["duration_diagnostic"]) == {
        "predicted_duration_sec", "duration_limit_sec", "effective_duration_sec", "fit_ratio"}
    assert report["stages"]["compaction"]["requests"] == 0


def test_workflow_character_too_long_still_compacts_with_profile() -> None:
    original, current = [item(1, "Original phrase with alternatives", 2.0)], [item(1, "short", 2.0)]
    too_long = "This candidate is intentionally longer than the character budget for this subtitle"
    compact = "Compact alternatives"
    client = FakeClient([assessment(1, "fail", ["alternative"]), editor(1, too_long),
                         editor(1, compact), verification(1, required=["alternative"])])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client,
        duration_profile=duration_model(0.01)))
    assert report["stages"]["compaction"]["requests"] == 1
    assert report["outcomes"][0]["lineage"]["compaction"] == compact


def test_workflow_duration_valid_final_updates_diagnostic() -> None:
    original, current = [item(1, "Original phrase", 10.0)], [item(1, "short", 10.0)]
    accepted = "valid"
    client = FakeClient([assessment(1, "fail", ["object"]), editor(1, accepted),
                         verification(1, required=["object"])])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    outcome = report["outcomes"][0]
    assert outcome["outcome"] == "verified"
    assert outcome["duration_diagnostic"]["predicted_duration_sec"] == float(len(accepted))
    assert outcome["lineage"]["duration_diagnostic"] == outcome["duration_diagnostic"]


def test_workflow_duration_invalid_repair_is_rejected_before_reverify() -> None:
    original, current = [item(1, "Original phrase with object", 10.0)], [item(1, "short", 10.0)]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    invalid = "12345678901"
    client = FakeClient([assessment(1, "fail", ["object"]), editor(1, "", "unresolved"),
                         verification(1, "fail", ["object"]), editor(1, invalid)])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    outcome = report["outcomes"][0]
    assert outcome["stage_errors"]["repair"] == "formal_error:duration_too_long"
    assert report["stages"]["reverify"]["requests"] == 0
    assert outcome["lineage"]["rejected"][-1]["reason"] == "duration_too_long"


def test_workflow_reverify_failure_rolls_back_and_final_gate_rechecks_duration() -> None:
    original, current = [item(1, "Original phrase with object", 10.0)], [item(1, "short", 10.0)]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    editor_candidate, repair_candidate = "edited", "repaired"
    client = FakeClient([assessment(1, "fail", ["object"]), editor(1, editor_candidate),
                         verification(1, "fail", ["object"]), editor(1, repair_candidate),
                         verification(1, "uncertain", ["uncertain"])])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    outcome = report["outcomes"][0]
    assert outcome["outcome"] == "unresolved"
    assert outcome["lineage"]["final"] == editor_candidate
    assert outcome["duration_diagnostic"]["predicted_duration_sec"] == float(len(editor_candidate))
    assert outcome["lineage"]["rejected"][-1]["stage"] == "reverify"


def test_workflow_unchanged_critical_zero_duration_cannot_verify() -> None:
    original = [item(1, "Same critical phrase", 0.1)]
    current = [item(1, "Same critical phrase", 0.1)]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    client = FakeClient([assessment(1, "pass", []), editor(1, "Same critical phrase")])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, duration_profile=duration_model()))
    assert report["selected_indices"] == [1]
    assert report["unresolved_indices"] == [1]
    assert report["stages"]["verifier"]["requests"] == 0


def test_no_profile_workflow_has_no_duration_fields() -> None:
    client = FakeClient([assessment(1, "pass", []), verification(1)])
    _, report, usage = asyncio.run(review_subtitles(
        [item(1, "Original phrase")], [item(1, "Changed phrase")], "key", client=client))
    assert "duration_profile" not in report and "duration_profile" not in usage
    assert "duration_diagnostic" not in report["outcomes"][0]
    assert "duration_diagnostic" not in report["outcomes"][0]["lineage"]


def test_cli_rejects_profile_before_api_key_lookup(tmp_path: Path, monkeypatch: Any) -> None:
    profile = tmp_path / "bad.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reviewer, "load_api_key", lambda _: pytest.fail("API key lookup must not run"))
    assert reviewer.main(["original.json", "shortened.json", "--duration-profile", str(profile)]) == 2


def test_cli_rejects_invalid_ratio_before_client_or_review(tmp_path: Path, monkeypatch: Any) -> None:
    profile = tmp_path / "profile.json"
    write_duration_profile(profile)
    monkeypatch.setattr(reviewer, "load_api_key", lambda _: pytest.fail("API key lookup must not run"))
    assert reviewer.main(["original.json", "shortened.json", "--duration-profile", str(profile),
                          "--duration-fit-ratio", "0"]) == 2


def test_cli_loads_profile_once_and_writes_nonsecret_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    profile = tmp_path / "profile.json"
    write_duration_profile(profile)
    output_dir = tmp_path / "out"
    captured: dict[str, Any] = {}
    original_loader = reviewer.load_calibrated_duration_profile
    loads = 0

    def load_once(path: Path) -> CalibratedDurationModel:
        nonlocal loads
        loads += 1
        return original_loader(path)

    async def fake_review(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return [], {"duration_profile": {"coefficients": {"x": 1}, "fit_ratio": 1.25}}, {
            "duration_profile": {"coefficients": {"x": 1}, "fit_ratio": 1.25}}

    monkeypatch.setattr(reviewer, "load_calibrated_duration_profile", load_once)
    monkeypatch.setattr(reviewer, "load_api_key", lambda _: "secret-key")
    monkeypatch.setattr(reviewer, "load_json", lambda _: [])
    monkeypatch.setattr(reviewer, "review_subtitles", fake_review)
    assert reviewer.main(["original.json", "shortened.json", "--duration-profile", str(profile),
                          "--duration-fit-ratio", "1.25", "--output-dir", str(output_dir)]) == 0
    assert loads == 1 and captured["duration_profile"].voice == "test-voice"
    report = json.loads((output_dir / "shortened_review.report.json").read_text(encoding="utf-8"))
    usage = json.loads((output_dir / "shortened_review.usage.json").read_text(encoding="utf-8"))
    for metadata in (report["duration_profile"], usage["duration_profile"]):
        assert metadata["fit_ratio"] == 1.25 and "coefficients" in metadata
        assert "secret-key" not in json.dumps(metadata) and str(profile) not in json.dumps(metadata)


def test_cli_legacy_omits_duration_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    output_dir = tmp_path / "out"
    async def fake_review(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        return [], {}, {}
    monkeypatch.setattr(reviewer, "load_api_key", lambda _: "secret-key")
    monkeypatch.setattr(reviewer, "load_json", lambda _: [])
    monkeypatch.setattr(reviewer, "review_subtitles", fake_review)
    assert reviewer.main(["original.json", "shortened.json", "--output-dir", str(output_dir)]) == 0
    assert "duration_profile" not in json.loads(
        (output_dir / "shortened_review.report.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("original", "candidate", "expected"),
    [
        ("Сегодня мы закончим", "Мы закончим", "time"),
        ("По-моему, это верно", "Это верно", "modality"),
        ("Уйдем, потому что поздно", "Уйдем поздно", "causality"),
        ("Столько лет, сколько ему", "Много лет", "comparison"),
        ("Можно резать, шлифовать или красить", "Можно резать", "alternative"),
        ("Судим по ним", "Так и судим", "reference"),
        ("Берем этот профиль", "Берем профиль", "reference"),
    ],
)
def test_semantic_risk_hints_cover_high_value_anchors(
    original: str, candidate: str, expected: str
) -> None:
    assert expected in semantic_risk_issues(original, candidate)
    one_word = ReviewTarget(0, 2, "Привет", "Привет", 10, 1, 0, 1, False)
    assert review_validation_error(one_word, "Привет") is None


def test_usage_pricing_and_reasoning_token_reservation() -> None:
    summary = pro_usage_summary({"prompt_cache_miss_tokens": 1_000_000,
                                 "prompt_cache_hit_tokens": 1_000_000,
                                 "completion_tokens": 1_000_000})
    assert summary["estimated_cost_usd"] == 1.308625
    assert summary["pricing_usd_per_million"]["output"] == 0.87
    target = ReviewTarget(0, 1, "Original words", "Short words", 30, 2, 20, 2, False)
    assert review_max_tokens([target], None) < 4096
    assert review_max_tokens([target], "high") == 8192
    assert review_max_tokens([target], "max") == 16384


def test_context_uses_full_source_position_and_overlay() -> None:
    source = [item(350, "source 350"), item(351, "real 351"), item(352, "original 352")]
    original, current = [item(352, "original 352")], [item(352, "flash 352")]
    target = ReviewTarget(0, 352, "original 352", "flash 352", 100, 2, 20, 2, False)
    payload = json.loads(build_context(original, current, [target], 1, source))
    assert [value["index"] for value in payload] == [351, 352]
    assert payload[0]["current_text"] == "real 351"
    assert payload[1]["current_text"] == "flash 352"


def test_prompts_cover_regressions_without_literal_overstrictness() -> None:
    for marker in ("modality", "causality", "alternatives", "cross-line references", "не требуется"):
        assert marker in CRITIC_PROMPT
    assert "Естественный" in CRITIC_PROMPT and "фрагмент" in VERIFIER_PROMPT
    for prompt in (CRITIC_PROMPT, EDITOR_PROMPT, VERIFIER_PROMPT):
        assert "JSON" in prompt
    for prompt in (CRITIC_PROMPT, VERIFIER_PROMPT):
        assert '"index"' in prompt and '"verdict"' in prompt and '"issues"' in prompt
        assert "pass|fail|uncertain" in prompt
        assert "pass требует пустой issues" in prompt
        assert "fail/uncertain — непустой" in prompt
        assert "alternative" in prompt and "polarity" in prompt and "uncertain" in prompt
    assert '"decision"' in EDITOR_PROMPT and '"candidate"' in EDITOR_PROMPT
    for marker in ("склей", "падеж", "object/goal", "antecedents", "slash notation"):
        assert marker in EDITOR_PROMPT
    for marker in ("вслух", "broken case attachment", "ambiguous antecedent", "editor notation"):
        assert marker in VERIFIER_PROMPT
    assert "JSON" in CRITIC_PROMPT and "JSON" in EDITOR_PROMPT and "JSON" in VERIFIER_PROMPT


def test_pass_skips_editor_and_blind_verifier_is_isolated() -> None:
    original, current = [item(1, "Original phrase here")], [item(1, "Flash phrase here")]
    client = FakeClient([assessment(1, "pass", []), verification(1)])
    output, report, usage = asyncio.run(review_subtitles(original, current, "key", client=client))
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in calls)
    verifier = calls[1]["messages"][1]["content"]
    assert "critic" not in verifier.lower() and "issues" not in verifier and "editor" not in verifier.lower()
    assert output[0]["text"] == ["Flash phrase here"]
    assert report["automated_verified_count"] == 1 and report["unresolved_indices"] == []
    assert set(usage["stages"]) == {"critic", "editor", "compaction", "verifier", "repair", "reverify"}


def test_critic_failure_routes_editor_high_then_verifier_disabled() -> None:
    original, current = [item(1, "Original phrase with object")], [item(1, "Flash phrase wrong")]
    client = FakeClient([assessment(1, "fail", ["object"]), editor(1, "Correct compact phrase"),
                         verification(1, required=["object"])])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert client.chat.completions.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert client.chat.completions.calls[2]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert output[0]["text"] == ["Correct compact phrase"]
    assert report["outcomes"][0]["lineage"]["editor"] == "Correct compact phrase"


def test_remaining_critical_editor_payload_adds_budget_issue() -> None:
    original = [item(1, "Original semantic relation stays")]
    current = [item(1, "Original semantic relation stays")]
    current[0]["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    client = FakeClient([
        assessment(1, "pass", []),
        editor(1, "Relation stays compact"),
        verification(1),
    ])
    asyncio.run(review_subtitles(original, current, "key", client=client))
    editor_call = client.chat.completions.calls[1]
    payload = json.loads(editor_call["messages"][1]["content"].split("\nTargets:", 1)[1])
    assert payload[0]["issues"] == ["budget"]
    assert payload[0]["requires_shortening"] is True
    for marker in ("budget-only", "сравнение/reference", "alternatives", "predicate"):
        assert marker in editor_call["messages"][0]["content"]


def test_critic_pass_with_risk_hint_routes_high_editor_and_resolves() -> None:
    original = [item(1, "Сегодня мы достигнем цели")]
    current = [item(1, "Мы достигнем цели")]
    client = FakeClient([
        assessment(1, "pass", []),
        editor(1, "Сегодня достигнем цели"),
        verification(1, required=["time"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    editor_call = client.chat.completions.calls[1]
    assert editor_call["extra_body"] == {"thinking": {"type": "disabled"}}
    editor_payload = json.loads(
        editor_call["messages"][1]["content"].split("\nTargets:", 1)[1]
    )
    assert editor_payload[0]["issues"] == ["time"]
    verifier_prompt = client.chat.completions.calls[2]["messages"][1]["content"]
    assert "risk_hints" not in verifier_prompt
    assert '"issues"' not in verifier_prompt
    assert output[0]["text"] == ["Сегодня достигнем цели"]
    assert report["outcomes"][0]["risk_hints"] == ["time"]
    assert report["unresolved_indices"] == []


def test_repair_payload_merges_verifier_issues_and_risk_hints() -> None:
    original = [item(1, "Столько лет, сколько указано по ним")]
    current = [item(1, "Лет как указано")]
    client = FakeClient([
        assessment(1, "pass", []),
        editor(1, "Лет как указано"),
        verification(1, "fail", ["grammar"], ["comparison", "reference"]),
        editor(1, "Столько лет, сколько указано"),
        verification(1, required=["comparison", "reference"]),
    ])
    _, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    repair_payload = json.loads(
        client.chat.completions.calls[3]["messages"][1]["content"].split("\nTargets:", 1)[1]
    )
    assert repair_payload[0]["issues"] == ["grammar", "comparison", "reference"]
    assert "semantic_requirements" in repair_payload[0]
    assert report["outcomes"][0]["verifier_verdict"] == "pass"
    assert client.chat.completions.calls[3]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_alternative_regression_critic_editor_verifier_flow() -> None:
    original = [item(394, "Можно резать, шлифовать или полировать деталь")]
    current = [item(394, "Можно резать деталь")]
    compact_three_way = "Деталь режут, шлифуют или полируют"
    client = FakeClient([
        assessment(394, "fail", ["alternative"]),
        editor(394, compact_three_way),
        verification(394, required=["alternative"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert output[0]["text"] == [compact_three_way]
    assert report["outcomes"][0]["critic_issues"] == ["alternative"]
    assert report["outcomes"][0]["verifier_verdict"] == "pass"
    assert report["unresolved_indices"] == []


def test_too_long_editor_candidate_gets_one_targeted_compaction() -> None:
    original = [item(394, "Можно резать, шлифовать или полировать деталь", 2.0)]
    current = [item(394, "Можно резать деталь", 2.0)]
    rejected = "Деталь можно аккуратно резать, шлифовать или полировать"
    compact = "Режьте, шлифуйте или полируйте деталь"
    client = FakeClient([
        assessment(394, "fail", ["alternative"]), editor(394, rejected),
        editor(394, compact), verification(394, required=["alternative"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    outcome = report["outcomes"][0]
    assert output[0]["text"] == [compact]
    assert report["stages"]["compaction"]["requests"] == 1
    assert report["retries"]["targeted_compactions"] == 1
    assert outcome["lineage"]["rejected"] == [
        {"stage": "editor", "candidate": rejected, "reason": f"too_long:{len(rejected)}>39"}
    ]
    prompt = client.chat.completions.calls[2]["messages"][1]["content"]
    assert client.chat.completions.calls[2]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert client.chat.completions.calls[2]["max_tokens"] < 8192
    for marker in (rejected, '"current_length"', '"max_chars": 39', '"excess"', '"original"', '"context"'):
        assert marker in prompt
    assert '"semantic_requirements"' in prompt
    assert report["unresolved_indices"] == []


def test_planner_malformed_compaction_gets_one_disabled_schema_retry() -> None:
    original = [item(394, "Можно резать, шлифовать или полировать деталь", 2.0)]
    current = [item(394, "Можно резать деталь", 2.0)]
    rejected = "Деталь можно аккуратно резать, шлифовать или полировать"
    compact = "Режьте, шлифуйте или полируйте деталь"
    client = FakeClient([
        planner_response(394, original[0]["text"][0], [(original[0]["text"][0], "The phrase")],
                        [("alternative", "или")]),
        assessment(394, "fail", ["alternative"]), editor(394, rejected),
        "malformed", editor(394, compact), verification(394, required=["alternative"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_planner=True,
    ))
    compaction_calls = [call for call in client.chat.completions.calls
                        if call["messages"][0]["content"] == COMPACTION_PROMPT]
    assert len(compaction_calls) == 2
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in compaction_calls)
    assert all(call["max_tokens"] < 8192 for call in compaction_calls)
    assert output[0]["text"] == [compact]
    assert report["stages"]["compaction"]["requests"] == 2
    assert report["outcomes"][0]["verifier_verdict"] == "pass"


def test_planner_compaction_api_none_has_no_schema_retry() -> None:
    original = [item(394, "Можно резать, шлифовать или полировать деталь", 2.0)]
    current = [item(394, "Можно резать деталь", 2.0)]
    rejected = "Деталь можно аккуратно резать, шлифовать или полировать"
    client = FakeClient([
        planner_response(394, original[0]["text"][0], [(original[0]["text"][0], "The phrase")],
                        [("alternative", "или")]),
        assessment(394, "fail", ["alternative"]), editor(394, rejected), None,
        verification(394, required=["alternative"]), editor(394, "", "unresolved"),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, enable_planner=True,
    ))
    assert report["stages"]["compaction"]["requests"] == 1
    assert report["outcomes"][0]["stage_errors"]["compaction"] == "api_failure"


def test_planner_malformed_repair_gets_one_disabled_schema_retry() -> None:
    source = "Original phrase with object"
    client = FakeClient([
        planner_response(1, source, [(source, "The phrase")], []), assessment(1, "pass", []),
        verification(1, "fail", ["object"]),
        "malformed", editor(1, "Repaired phrase here"), verification(1),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        [item(1, source)], [item(1, "Flash phrase here")], "key", client=client,
        enable_planner=True,
    ))
    repair_calls = [call for call in client.chat.completions.calls
                    if call["messages"][0]["content"] == EDITOR_PROMPT]
    assert len(repair_calls) == 2
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in repair_calls)
    assert all(call["max_tokens"] < 8192 for call in repair_calls)
    assert report["stages"]["repair"]["requests"] == 2
    assert output[0]["text"] == ["Repaired phrase here"]
    assert report["automated_verified_count"] == 1


def test_planner_repair_api_none_has_no_schema_retry() -> None:
    source = "Original phrase with object"
    client = FakeClient([
        planner_response(1, source, [(source, "The phrase")], []), assessment(1, "pass", []),
        editor(1, "Edited phrase here"), verification(1, "fail", ["object"]), None,
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, source)], [item(1, "Flash phrase here")], "key", client=client,
        enable_planner=True,
    ))
    assert report["stages"]["repair"]["requests"] == 1
    assert report["outcomes"][0]["stage_errors"]["repair"] == "api_failure"


def test_failed_compaction_and_critic_failure_cannot_generic_pass() -> None:
    original = [item(1, "Можно резать, шлифовать или полировать деталь", 2.0)]
    current = [item(1, "Можно резать деталь", 2.0)]
    rejected = "Деталь можно аккуратно резать, шлифовать или полировать"
    client = FakeClient([
        assessment(1, "fail", ["alternative"]), editor(1, rejected),
        editor(1, "", "unresolved"), verification(1, required=["alternative"]),
        editor(1, "", "unresolved"),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert output[0]["text"] == ["Можно резать деталь"]
    assert report["unresolved_indices"] == [1]
    assert report["outcomes"][0]["compaction_decision"] == "unresolved"
    assert report["stages"]["compaction"]["requests"] == 1


def test_required_check_must_be_present_and_pass() -> None:
    assert parse_verifications(verification(1), {1: ["alternative"]}) == {}
    failed = verification(1, "fail", ["alternative"], ["alternative"], {"alternative"})
    parsed = parse_verifications(failed, {1: ["alternative"]})
    assert parsed[1][2] == {"alternative": "fail"}


@pytest.mark.parametrize("verifier_response", [
    verification(1),
    verification(1, "fail", ["time"], ["time"], {"time"}),
])
def test_workflow_required_check_failure_remains_unresolved(verifier_response: str) -> None:
    original = [item(1, "Сегодня мы достигнем цели")]
    current = [item(1, "Мы достигнем цели")]
    client = FakeClient([
        assessment(1, "pass", []), editor(1, "Сегодня достигнем цели"),
        verifier_response, editor(1, "", "unresolved"),
    ])
    _, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert report["retries"]["bounded_repairs"] == 1
    assert report["automated_verified_count"] == 0
    assert report["unresolved_indices"] == [1]


def test_repair_is_committed_only_after_passing_reverify() -> None:
    original, current = [item(1, "Original phrase with object")], [item(1, "Flash phrase here")]
    success = FakeClient([assessment(1, "pass", []), verification(1, "fail", ["object"]),
                          editor(1, "Repaired phrase here"), verification(1)])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=success))
    assert len(success.chat.completions.calls) == 4 and report["automated_verified_count"] == 1
    assert output[0]["text"] == ["Repaired phrase here"]
    assert report["outcomes"][0]["lineage"]["final"] == "Repaired phrase here"
    assert report["outcomes"][0]["verifier_history"] == [
        {"stage": "verifier", "verdict": "fail", "issues": ["object"], "required_checks": {}, "error": "fail"},
        {"stage": "reverify", "verdict": "pass", "issues": [], "required_checks": {}, "error": None},
    ]
    assert report["outcomes"][0]["verifier_verdict"] == "pass"


def test_repair_rolls_back_when_reverify_fails() -> None:
    original, current = [item(1, "Original phrase with object")], [item(1, "Flash phrase here")]
    failure = FakeClient([assessment(1, "pass", []), verification(1, "fail", ["object"]),
                          editor(1, "Still questionable phrase"), verification(1, "uncertain", ["uncertain"])])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=failure))
    assert len(failure.chat.completions.calls) == 4
    assert report["unresolved_indices"] == [1] and report["automated_verified_count"] == 0
    assert output[0]["text"] == ["Flash phrase here"]
    outcome = report["outcomes"][0]
    assert outcome["verifier_verdict"] == "fail"
    assert outcome["verifier_issues"] == ["object"]
    assert outcome["lineage"]["repair"] == "Still questionable phrase"
    assert outcome["lineage"]["final"] == "Flash phrase here"
    assert outcome["lineage"]["rejected"] == [{
        "stage": "reverify", "candidate": "Still questionable phrase", "reason": "uncertain",
    }]
    assert [entry["verdict"] for entry in outcome["verifier_history"]] == ["fail", "uncertain"]


def test_naturalness_only_reverify_keeps_semantic_repair_unresolved() -> None:
    original = [item(342, "А сегодня утром пробежалась по ним рукой")]
    current = [item(342, "Пробежалась рукой")]
    repaired = "Сегодня утром пробежалась по ним"
    client = FakeClient([
        assessment(342, "pass", []),
        editor(342, "", "unresolved"),
        verification(342, "fail", ["reference"], ["time", "reference"]),
        editor(342, repaired),
        verification(342, "fail", ["naturalness"], ["time", "reference"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    outcome = report["outcomes"][0]
    assert output[0]["text"] == [repaired]
    assert outcome["lineage"]["final"] == repaired
    assert outcome["outcome"] == "unresolved"
    assert outcome["verifier_issues"] == ["naturalness"]
    assert not outcome["lineage"]["rejected"]


def test_semantic_required_check_failure_rolls_back_repair() -> None:
    original = [item(381, "Ловушки вокруг лагеря")]
    current = [item(381, "Ловушки")]
    repaired = "Ловушки вокруг лагеря"
    client = FakeClient([
        assessment(381, "pass", []),
        editor(381, "Ловушки"),
        verification(381, "fail", ["spatial"], ["spatial"], {"spatial"}),
        editor(381, repaired),
        verification(381, "fail", ["spatial"], ["spatial"], {"spatial"}),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    outcome = report["outcomes"][0]
    assert output[0]["text"] == ["Ловушки"]
    assert outcome["lineage"]["final"] == "Ловушки"
    assert outcome["lineage"]["rejected"] == [{
        "stage": "reverify", "candidate": repaired, "reason": "fail",
    }]


def test_failed_only_repair_preserves_verified_peer() -> None:
    original = [item(1, "Original first phrase"), item(2, "Original second phrase")]
    current = [item(1, "Flash first phrase"), item(2, "Flash second phrase")]
    critic = json.dumps({"results": [
        {"index": 1, "verdict": "pass", "issues": []},
        {"index": 2, "verdict": "pass", "issues": []},
    ]})
    verifier = json.dumps({"results": [
        {"index": 1, "verdict": "pass", "issues": [], "required_checks": []},
        {"index": 2, "verdict": "fail", "issues": ["reference"], "required_checks": []},
    ]})
    client = FakeClient([critic, verifier, editor(2, "", "unresolved")])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert report["unresolved_indices"] == [2]
    assert report["outcomes"][0]["outcome"] == "verified"
    assert output[0]["text"] == ["Flash first phrase"]
    assert output[1]["text"] == ["Flash second phrase"]


def test_oversized_critic_target_is_not_sent_and_reports_stage_error() -> None:
    original, current = [item(1, "Original phrase here")], [item(1, "Flash phrase here")]
    client = FakeClient([])
    output, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, max_input_tokens=100
    ))
    assert client.chat.completions.calls == []
    assert output[0]["text"] == ["Flash phrase here"]
    assert report["unresolved_indices"] == [1]
    assert report["outcomes"][0]["stage_errors"] == {"critic": "input_too_large"}


def test_actual_critic_and_verifier_batches_split_by_real_prompt_size() -> None:
    original = [item(1, "Original phrase one"), item(2, "Original phrase two")]
    current = [item(1, "Flash phrase one"), item(2, "Flash phrase two")]
    client = FakeClient([
        json.dumps({"results": [
            {"index": 1, "verdict": "pass", "issues": []},
            {"index": 2, "verdict": "pass", "issues": []},
        ]}),
        json.dumps({"results": [
            {"index": 1, "verdict": "pass", "issues": [], "required_checks": []},
            {"index": 2, "verdict": "pass", "issues": [], "required_checks": []},
        ]}),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, batch_size=2, max_input_tokens=4500
    ))
    assert report["stages"]["critic"]["requests"] == 1
    assert report["stages"]["verifier"]["requests"] == 1
    assert report["automated_verified_count"] == 2


def test_transport_retries_then_reports_api_failure(monkeypatch: Any) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    original, current = [item(1, "Original phrase here")], [item(1, "Flash phrase here")]
    client = FakeClient([TimeoutError() for _ in range(12)])
    _, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert len(client.chat.completions.calls) == 12
    assert all(call["messages"][0]["content"] == CRITIC_PROMPT
               for call in client.chat.completions.calls[:3])
    assert report["outcomes"][0]["stage_errors"]["critic"] == "api_failure"
    assert report["unresolved_indices"] == [1]


def test_paid_malformed_response_reports_parse_failure() -> None:
    original, current = [item(1, "Original phrase here")], [item(1, "Flash phrase here")]
    client = FakeClient([
        '{"results":[{"index":1,"verdict":"pass"}]}',
        editor(1, "", "unresolved"),
        verification(1, "uncertain", ["uncertain"], ["uncertain"]),
        editor(1, "", "unresolved"),
    ])
    _, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    assert report["outcomes"][0]["stage_errors"]["critic"] == "parse_failure"


def test_review_api_default_concurrency_is_three() -> None:
    assert inspect.signature(review_subtitles).parameters["concurrency"].default == 3
    assert build_parser().parse_args(["original.json", "shortened.json"]).concurrency == 3


def test_owned_pro_client_disables_sdk_retries(monkeypatch: Any) -> None:
    created: list[dict[str, Any]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions([]))

        async def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    unchanged = [item(1, "Same short phrase")]
    asyncio.run(review_subtitles(unchanged, unchanged, "key"))
    assert created == [{
        "api_key": "key",
        "base_url": "https://api.deepseek.com",
        "timeout": 60.0,
        "max_retries": 0,
    }]


def test_stage_requests_use_bounded_concurrency() -> None:
    class TrackingCompletions:
        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0

        async def create(self, **kwargs: Any) -> Any:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.01)
            payload = json.loads(kwargs["messages"][1]["content"].split("\nTargets:", 1)[1])
            is_verifier = kwargs["messages"][0]["content"] == VERIFIER_PROMPT
            results = [
                ({"index": target["index"], "verdict": "pass", "issues": [],
                  "required_checks": [{"issue": issue, "verdict": "pass"}
                                      for issue in target.get("required_checks", [])]}
                 if is_verifier else
                 {"index": target["index"], "verdict": "pass", "issues": []})
                for target in payload
            ]
            self.active -= 1
            usage = SimpleNamespace()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"results": results})
                ))],
                usage=usage,
            )

    completions = TrackingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    original = [item(i, f"Original phrase number {i}") for i in range(1, 5)]
    current = [item(i, f"Compact phrase number {i}") for i in range(1, 5)]
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, batch_size=1, concurrency=2
    ))
    assert completions.maximum == 2
    assert report["automated_verified_count"] == 4


def test_semantic_requirements_are_audited_and_sent_to_stages() -> None:
    original = [item(381, "Ловушки вокруг лагеря в первую ночь")]
    current = [item(381, "Ловушки")]
    client = FakeClient([
        assessment(381, "pass", []),
        editor(381, "Ловушки вокруг лагеря в первую ночь"),
        verification(381, required=["spatial", "time"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(original, current, "key", client=client))
    payloads = [json.loads(call["messages"][1]["content"].split("\nTargets:", 1)[1])
                for call in client.chat.completions.calls]
    assert all("semantic_requirements" in payload[0] for payload in payloads)
    requirements = report["outcomes"][0]["semantic_requirements"]
    assert {entry["source_anchor"] for entry in requirements} == {"вокруг лагеря", "в первую ночь"}
    assert output[0]["text"] == ["Ловушки вокруг лагеря в первую ночь"]


def test_planner_index21_is_original_only_and_reaches_safe_verification() -> None:
    propositions = [("Им нужно будет научиться адаптироваться", "The agent must learn to adapt."),
                    ("они будут изгнаны", "They will be expelled.")]
    source = "Им нужно будет научиться адаптироваться или они будут изгнаны!"
    lossy = "Адаптируйся или изгнание."
    safe = "Им надо адаптироваться, иначе их изгонят."
    planner_requirements = [("agent", "Им"), ("modality", "нужно будет"), ("tense", "будут изгнаны"),
                             ("speech_act", source), ("alternative", "или"), ("predicate", "научиться"),
                             ("object", "изгнаны"), ("entity", "они")]
    required = ["agent", "modality", "tense", "speech_act", "alternative"]
    client = FakeClient([
        planner_response(21, source, propositions, planner_requirements),
        assessment(21, "fail", ["alternative"]),
        editor(21, safe),
        verification(21, required=required),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        [item(21, source)], [item(21, lossy)], "key", client=client, enable_planner=True,
    ))
    planner_call, critic_call, editor_call, verifier_call = client.chat.completions.calls
    planner_user = planner_call["messages"][1]["content"]
    assert source in planner_user and lossy not in planner_user
    assert "current_text" not in planner_user
    assert all(marker not in planner_user.lower() for marker in ("critic", "editor", "lineage"))
    assert planner_call["messages"][0]["content"] == PLANNER_PROMPT
    assert planner_call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert planner_call["max_tokens"] >= 8192
    assert output[0]["text"] == [safe]
    outcome = report["outcomes"][0]
    assert outcome["outcome"] == "verified"
    assert outcome["planner_status"] == "success"
    assert len(outcome["semantic_plan"]["propositions"]) == 2
    assert list(outcome["verifier_required_checks"]) == required
    assert len(outcome["verifier_required_checks"]) < len(outcome["semantic_plan"]["requirements"])
    assert report["stages"]["planner"]["requests"] == 1
    assert report["semantic_plans"]["21"]["index"] == 21
    for call in (critic_call, editor_call, verifier_call):
        assert '"semantic_plan"' in call["messages"][1]["content"]
    verifier_user = verifier_call["messages"][1]["content"].lower()
    assert "semantic_plan" in verifier_user
    assert "critic_verdict" not in verifier_user and "editor history" not in verifier_user
    assert "lineage" not in verifier_user and "rejected" not in verifier_user


def test_planner_partial_response_fails_closed_without_later_stage_calls() -> None:
    source = "Мы должны прийти."
    client = FakeClient([json.dumps({"results": []}), json.dumps({"results": []})])
    output, report, _ = asyncio.run(review_subtitles(
        [item(21, source)], [item(21, "Мы прийти")], "key", client=client, enable_planner=True,
    ))
    assert len(client.chat.completions.calls) == 2
    assert output[0]["text"] == ["Мы прийти"]
    outcome = report["outcomes"][0]
    assert outcome["outcome"] == "unresolved"
    assert outcome["planner_status"] == "parse_failure"
    assert outcome["stage_errors"] == {"planner": "parse_failure"}


def test_planner_failed_batch_retries_each_target_once_without_blocking_peer() -> None:
    source_one = "Сегодня мы достигнем цели"
    source_two = "Они должны завершить работу"
    planner_valid = planner_response(1, source_one, [(source_one, "They will reach the goal today.")], [])
    client = FakeClient([
        json.dumps({"results": []}),
        planner_valid,
        json.dumps({"results": []}),
        assessment(1, "pass", []),
        editor(1, "Сегодня достигнем цели"),
        verification(1, required=["time"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        [item(1, source_one), item(2, source_two)],
        [item(1, "Мы достигнем цели"), item(2, "Они работу")],
        "key", client=client, enable_planner=True,
    ))
    assert len(client.chat.completions.calls) == 6
    assert report["stages"]["planner"]["requests"] == 3
    assert output[0]["text"] == ["Сегодня достигнем цели"]
    assert output[1]["text"] == ["Они работу"]
    assert report["outcomes"][0]["outcome"] == "verified"
    assert report["outcomes"][1]["outcome"] == "unresolved"
    assert report["outcomes"][1]["planner_status"] == "parse_failure"
    assert report["outcomes"][1]["stage_errors"] == {"planner": "parse_failure"}
    assert [json.loads(call["messages"][1]["content"].split("\nTargets:", 1)[1])
            for call in client.chat.completions.calls[:3]]
    assert all(len(json.loads(call["messages"][1]["content"].split("\nTargets:", 1)[1])) == 1
               for call in client.chat.completions.calls[1:3])


def test_planner_initial_batches_are_capped_at_two_targets() -> None:
    original = [item(index, f"Original phrase {index}") for index in range(1, 5)]
    current = [item(index, f"Flash phrase {index}") for index in range(1, 5)]
    planner = json.dumps({"results": [
        {"index": index, "source_text": f"Original phrase {index}",
         "propositions": [{"id": "p1", "source_span": f"Original phrase {index}", "meaning": "phrase"}],
         "requirements": []}
        for index in range(1, 3)
    ]})
    planner_two = json.dumps({"results": [
        {"index": index, "source_text": f"Original phrase {index}",
         "propositions": [{"id": "p1", "source_span": f"Original phrase {index}", "meaning": "phrase"}],
         "requirements": []}
        for index in range(3, 5)
    ]})
    client = FakeClient([planner, planner_two] + [assessment(index, "pass", []) for index in range(1, 5)]
                        + [verification(index) for index in range(1, 5)])
    asyncio.run(review_subtitles(original, current, "key", client=client, enable_planner=True, batch_size=4))
    planner_calls = [call for call in client.chat.completions.calls if call["messages"][0]["content"] == PLANNER_PROMPT]
    assert len(planner_calls) == 2
    assert all(len(json.loads(call["messages"][1]["content"].split("\nTargets:", 1)[1])) <= 2
               for call in planner_calls)


def test_planner_enabled_malformed_verifier_batch_retries_each_target_once() -> None:
    planner = json.dumps({"results": [
        {"index": index, "source_text": f"Original {index}",
         "propositions": [{"id": "p1", "source_span": f"Original {index}", "meaning": "phrase"}],
         "requirements": []}
        for index in (1, 2)
    ]})
    critic = json.dumps({"results": [
        {"index": index, "verdict": "pass", "issues": []} for index in (1, 2)
    ]})
    malformed = "not json"
    client = FakeClient([
        planner,
        critic,
        malformed,
        verification(1),
        malformed,
        editor(2, "", "unresolved"),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, "Original 1"), item(2, "Original 2")],
        [item(1, "Changed 1"), item(2, "Changed 2")],
        "key", client=client, enable_planner=True,
    ))
    assert report["stages"]["verifier"]["requests"] == 3
    assert report["outcomes"][0]["outcome"] == "verified"
    assert report["outcomes"][1]["stage_errors"]["verifier"] == "parse_failure"
    assert len(client.chat.completions.calls) == 6
    verifier_calls = [call for call in client.chat.completions.calls
                      if call["messages"][0]["content"] == VERIFIER_PROMPT]
    assert [len(json.loads(call["messages"][1]["content"].split("\nTargets:", 1)[1]))
            for call in verifier_calls] == [2, 1, 1]


def test_planner_enabled_reverify_malformed_response_gets_one_schema_retry() -> None:
    source = "Сегодня мы придем"
    client = FakeClient([
        planner_response(1, source, [(source, "We will arrive today.")], []),
        assessment(1, "pass", []),
        editor(1, "Мы придем сегодня"),
        verification(1, "fail", ["time"], ["time"], {"time"}),
        editor(1, "Мы придем сегодня"),
        "not json",
        verification(1, required=["time"]),
    ])
    output, report, _ = asyncio.run(review_subtitles(
        [item(1, source)], [item(1, "Мы придем")], "key", client=client, enable_planner=True,
    ))
    assert output[0]["text"] == ["Мы придем сегодня"]
    assert report["stages"]["reverify"]["requests"] == 2
    assert report["outcomes"][0]["outcome"] == "verified"


def test_planner_index279_boundary_and_modality_reach_verifier() -> None:
    source = 'Тропа, размытая водой". То есть должна быть размытая водой тропинка.'
    lossy = "Тропа, размытая водой."
    safe = source
    requirements = [("boundary", "То есть должна быть размытая водой тропинка"), ("modality", "должна быть")]
    required = [issue for issue, _ in requirements]
    propositions = [("Тропа, размытая водой", "A trail washed by water."),
                    ("То есть должна быть размытая водой тропинка", "The clarified trail must be washed by water.")]
    client = FakeClient([
        planner_response(279, source, propositions, requirements),
        assessment(279, "pass", []),
        editor(279, safe),
        verification(279, required=required),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(279, source)], [item(279, lossy)], "key", client=client, enable_planner=True,
    ))
    assert set(report["outcomes"][0]["verifier_required_checks"]) >= set(required)
    assert report["outcomes"][0]["planner_status"] == "success"


def test_build_required_checks_caps_and_orders_deterministic_labels() -> None:
    assert build_required_checks(
        ["agent", "uncertain", "predicate", "agent", "object"],
        ["time", "predicate", "polarity", "modality", "causality", "comparison", "alternative"],
    ) == ["agent", "predicate", "object", "time", "polarity", "modality"]


def test_planner_critic_malformed_batch_retries_each_target_once() -> None:
    sources = ["Original first", "Original second"]
    planner = json.dumps({"results": [
        {"index": index, "source_text": source,
         "propositions": [{"id": "p1", "source_span": source, "meaning": "phrase"}],
         "requirements": []}
        for index, source in enumerate(sources, 1)
    ]})
    verifier = json.dumps({"results": [
        {"index": index, "verdict": "pass", "issues": [], "required_checks": []}
        for index in (1, 2)
    ]})
    client = FakeClient([
        planner, "malformed", assessment(1, "pass", []), "malformed",
        editor(2, "", "unresolved"), verifier, editor(2, "", "unresolved"),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, sources[0]), item(2, sources[1])],
        [item(1, "Changed first"), item(2, "Changed second")],
        "key", client=client, enable_planner=True,
    ))
    assert report["stages"]["critic"]["requests"] == 3
    assert report["outcomes"][0]["critic_verdict"] == "pass"
    assert report["outcomes"][1]["critic_verdict"] == "uncertain"
    assert report["outcomes"][1]["stage_errors"]["critic"] == "parse_failure"


def test_planner_critic_api_none_has_no_schema_retry() -> None:
    source = "Original phrase"
    planner = planner_response(1, source, [(source, "A phrase")], [])
    client = FakeClient([planner, None, editor(1, "", "unresolved"), verification(1, "uncertain", ["uncertain"])])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, source)], [item(1, "Changed phrase")], "key", client=client, enable_planner=True,
    ))
    assert report["stages"]["critic"]["requests"] == 1
    assert report["outcomes"][0]["stage_errors"]["critic"] == "api_failure"


def test_planner_editor_malformed_batch_retries_each_target_disabled() -> None:
    sources = ["Original first", "Original second"]
    planner = json.dumps({"results": [
        {"index": index, "source_text": source,
         "propositions": [{"id": "p1", "source_span": source, "meaning": "phrase"}],
         "requirements": []}
        for index, source in enumerate(sources, 1)
    ]})
    critic = json.dumps({"results": [
        {"index": index, "verdict": "pass", "issues": []} for index in (1, 2)
    ]})
    verifier = json.dumps({"results": [
        {"index": index, "verdict": "pass", "issues": [], "required_checks": []}
        for index in (1, 2)
    ]})
    current = [item(1, "Changed first"), item(2, "Changed second")]
    for current_item in current:
        current_item["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    client = FakeClient([
        planner, critic, "malformed", editor(1, "Recovered first"), "malformed", verifier,
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, sources[0]), item(2, sources[1])],
        current,
        "key", client=client, enable_planner=True,
    ))
    editor_calls = [call for call in client.chat.completions.calls
                    if call["messages"][0]["content"] == EDITOR_PROMPT]
    assert report["stages"]["editor"]["requests"] == 3
    assert len(editor_calls) == 3
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in editor_calls)
    assert all(call["max_tokens"] < 8192 for call in editor_calls)
    assert report["outcomes"][0]["lineage"]["editor"] == "Recovered first"
    assert report["outcomes"][1]["stage_errors"]["editor"] == "parse_failure"


def test_planner_editor_api_none_has_no_schema_retry() -> None:
    source = "Original phrase"
    planner = planner_response(1, source, [(source, "A phrase")], [])
    current = item(1, "Changed phrase")
    current["analysis"].update({"is_checked": True, "mismatch_ratio": 2.0})
    client = FakeClient([planner, assessment(1, "pass", []), None, verification(1)])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, source)], [current], "key", client=client, enable_planner=True,
    ))
    assert report["stages"]["editor"]["requests"] == 1
    assert report["outcomes"][0]["stage_errors"]["editor"] == "api_failure"


def test_repair_payload_keeps_uncapped_verifier_guidance() -> None:
    issues = ["agent", "predicate", "object", "polarity", "modality", "time", "causality"]
    required = issues[:6]
    client = FakeClient([
        assessment(1, "fail", issues), editor(1, "Recovered phrase"),
        verification(1, "fail", issues, required), editor(1, "Repaired phrase"),
        verification(1, required=required),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        [item(1, "Original phrase")], [item(1, "Changed phrase")], "key", client=client,
    ))
    repair_call = client.chat.completions.calls[3]
    payload = json.loads(repair_call["messages"][1]["content"].split("\nTargets:", 1)[1])
    assert payload[0]["issues"][:7] == issues
    assert len(payload[0]["issues"]) > 6
    assert report["outcomes"][0]["verifier_required_checks"] == {
        issue: "pass" for issue in required
    }
