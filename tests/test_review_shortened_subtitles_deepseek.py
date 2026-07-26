"""Offline tests for the separated DeepSeek Pro quality stages."""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from utils.review_shortened_subtitles_deepseek import (
    CRITIC_PROMPT,
    EDITOR_PROMPT,
    VERIFIER_PROMPT,
    ReviewTarget,
    build_context,
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


def item(index: int, text: str, duration: float = 10.0) -> dict[str, Any]:
    return {"index": index, "text": [text], "analysis": {"available_duration_sec": duration,
            "effective_duration_sec": duration}}


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


def test_strict_assessment_parser_rejects_formal_errors() -> None:
    assert parse_assessments(assessment(1, "pass", []), {1}) == {1: ("pass", [])}
    assert parse_assessments(assessment(1, "pass", ["agent"]), {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["bogus"]}]}', {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["agent"],"x":1}]}', {1}) == {}
    assert parse_assessments('{"results":[]}', {1}) == {}
    assert parse_assessments('{"results":[{"index":1,"verdict":"fail","issues":["agent"]},{"index":1,"verdict":"fail","issues":["agent"]}]}', {1}) == {}
    assert parse_verifications(verification(1, required=["agent"]), {1: ["agent"]})[1][2] == {"agent": "pass"}
    assert parse_verifications(verification(1), {1: ["agent"]}) == {}


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
    assert client.chat.completions.calls[1]["extra_body"]["reasoning_effort"] == "high"
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
    assert editor_call["extra_body"]["reasoning_effort"] == "high"
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
    assert report["outcomes"][0]["verifier_verdict"] == "pass"


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
    for marker in (rejected, '"current_length"', '"max_chars": 39', '"excess"', '"original"', '"context"'):
        assert marker in prompt
    assert report["unresolved_indices"] == []


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
        verification(1),
        verification(2),
    ])
    _, report, _ = asyncio.run(review_subtitles(
        original, current, "key", client=client, batch_size=2, max_input_tokens=3900
    ))
    assert report["stages"]["critic"]["requests"] == 1
    assert report["stages"]["verifier"]["requests"] == 2
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


