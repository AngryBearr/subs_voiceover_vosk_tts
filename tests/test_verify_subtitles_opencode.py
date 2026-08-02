import asyncio
import json

import pytest

import utils.verify_subtitles_opencode as verifier
from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage
from utils.semantic_units import SemanticUnit
from utils.verify_subtitles_opencode import parse_unit_verification_response, parse_verification_response, verify_items


def response(index=1, verdict="pass", issues=None, severity="none"):
    return '{"results":[{"index":%d,"verdict":"%s","issues":%s,"severity":"%s","explanation":"checked"}]}' % (index, verdict, repr(issues or []).replace("'", '"'), severity)


def test_strict_parser_accepts_and_rejects():
    parsed = parse_verification_response(response(), [1])
    assert parsed[1]["verdict"] == "pass"
    with pytest.raises(ValueError):
        parse_verification_response('{"results":[]}', [1])


@pytest.mark.parametrize("raw", ["", "not json", "{}", '{"results":[]}', '{"results": [{"index": true, "verdict": "pass", "issues": [], "severity": "none", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "wat", "issues": [], "severity": "none", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "pass", "issues": ["bad"], "severity": "none", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "pass", "issues": [], "severity": "minor", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "fail", "issues": [], "severity": "major", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "uncertain", "issues": ["predicate"], "severity": "none", "explanation": "x"}]}' , '{"results": [{"index": 1, "verdict": "pass", "issues": [], "severity": "none", "explanation": ""}]}' , '{"results": [{"index": 1, "verdict": "pass", "issues": [], "severity": "none", "explanation": "x", "extra": 1}]}' ])
def test_parser_rejection_matrix(raw):
    with pytest.raises(ValueError):
        parse_verification_response(raw, [1])


def test_payload_objects_context_hints_and_system_prompt(monkeypatch):
    seen = {}
    async def request(prompt, model):
        seen["payload"] = json.loads(prompt)
        return response()
    asyncio.run(verify_items([{"index": 1, "text": "Мне нужно уйти сейчас."}], [{"index": 1, "text": "Уйти."}], request_callable=request))
    target = seen["payload"]["targets"][0]
    assert isinstance(target, dict) and isinstance(target["context"], list) and isinstance(target["risk_hints"], list)


@pytest.mark.parametrize("verdict,issues,severity", [("fail", ["predicate"], "major"), ("uncertain", ["reference"], "minor")])
def test_nonpass_verdicts_restore(verdict, issues, severity):
    async def request(prompt, model):
        return response(verdict=verdict, issues=issues, severity=severity)
    output, _, _ = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], request_callable=request))
    assert output[0]["text"] == "old"


@pytest.mark.parametrize("report", [{"outcomes": [{"index": 1, "outcome": "fallback"}]}, {"outcomes": []}, {"outcomes": [{"index": 1, "outcome": "missing"}]}])
def test_prior_nonverified_never_requests(report):
    calls = 0
    async def request(prompt, model):
        nonlocal calls
        calls += 1
        return response()
    output, _, _ = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], review_report=report, request_callable=request))
    assert calls == 0 and output[0]["text"] == "old"


@pytest.mark.parametrize("bad", [{"outcomes": [{"index": True, "outcome": "verified"}]}, {"outcomes": [{"index": 9, "outcome": "verified"}]}, {"outcomes": [{"index": 1, "outcome": 4}]}])
def test_prior_bad_schema_fails_closed_without_request(bad):
    calls = 0
    async def request(prompt, model):
        nonlocal calls
        calls += 1
        return response()
    output, report, _ = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], review_report=bad, request_callable=request))
    assert calls == 0 and output[0]["text"] == "old" and report["status"] == "completed_with_unresolved"


@pytest.mark.parametrize("raw", [
    '{"results":[{"index":1,"verdict":"pass","issues":[],"severity":"none","explanation":"x"},{"index":1,"verdict":"pass","issues":[],"severity":"none","explanation":"x"}]}',
    '{"results":[{"index":2,"verdict":"pass","issues":[],"severity":"none","explanation":"x"}]}',
    '{"results":[{"index":1,"verdict":"pass","issues":[],"severity":"none"}]}',
    '{"results":[{"index":1,"verdict":"pass","issues":"","severity":"none","explanation":"x"}]}',
    '{"results":[{"index":1,"verdict":"fail","issues":["predicate"],"severity":"none","explanation":"x"}]}',
])
def test_parser_exact_index_and_shape_rejections(raw):
    with pytest.raises(ValueError):
        parse_verification_response(raw, [1])


@pytest.mark.parametrize("candidate_text", [["new"], "new"])
def test_restore_shape_helper_via_failure(candidate_text):
    async def request(prompt, model):
        return response(verdict="uncertain", issues=["predicate"], severity="minor")
    output, _, _ = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": candidate_text, "meta": {"x": 1}}], request_callable=request))
    assert output[0]["text"] == (["old"] if isinstance(candidate_text, list) else "old")


def test_changed_pass_is_kept_and_fail_is_restored():
    calls = []

    async def request(prompt, model):
        calls.append(prompt)
        return response(verdict="fail", issues=["predicate"], severity="major")

    original = [{"index": 1, "text": ["old"]}]
    candidate = [{"index": 1, "text": ["new"], "meta": 3}]
    output, report, _ = asyncio.run(verify_items(original, candidate, request_callable=request))
    assert output[0]["text"] == ["old"]
    assert output[0]["meta"] == 3
    assert len(calls) == 1
    assert report["fallback_indices"] == [1]


def test_prior_unresolved_is_not_sent():
    calls = []

    async def request(prompt, model):
        calls.append(prompt)
        return response()

    output, report, _ = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        review_report={"outcomes": [{"index": 1, "outcome": "unresolved"}]}, request_callable=request,
    ))
    assert not calls
    assert output[0]["text"] == "old"
    assert report["fallback_indices"] == [1]


def test_prior_unresolved_with_matching_evidence_is_not_sent():
    calls = []

    async def request(prompt, model):
        calls.append(prompt)
        return response()

    output, report, _ = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        review_report={"outcomes": [{"index": 1, "outcome": "unresolved", "candidate_text": "new"}]},
        request_callable=request,
    ))
    assert calls == []
    assert output[0]["text"] == "old"
    assert report["targets"][0]["reason"] == "prior_not_verified"


def test_changed_input_without_request_callable_fails_before_transport(monkeypatch):
    def unexpected_transport(*args, **kwargs):
        raise AssertionError("transport called")

    monkeypatch.setattr(verifier, "_request_live", unexpected_transport)
    with pytest.raises(ValueError, match="^request_callable_required$"):
        asyncio.run(verify_items(
            [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        ))


def test_prior_verified_matching_lineage_final_requests_once_and_keeps_candidate():
    calls = []

    async def request(prompt, model):
        calls.append(prompt)
        return response()

    output, report, _ = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        review_report={"outcomes": [{"index": 1, "outcome": "verified", "lineage": {"final": "new"}}]},
        request_callable=request,
    ))
    assert len(calls) == 1
    assert output[0]["text"] == "new"
    assert report["verified_indices"] == [1]


def test_prior_text_mismatch_and_missing_evidence_fail_closed_without_requests():
    for prior in (
        {"outcomes": [{"index": 1, "outcome": "verified", "candidate_text": "other"}]},
        {"outcomes": [{"index": 1, "outcome": "verified"}]},
    ):
        calls = []

        async def request(prompt, model):
            calls.append(prompt)
            return response()

        output, report, _ = asyncio.run(verify_items(
            [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
            review_report=prior, request_callable=request,
        ))
        assert calls == []
        assert output[0]["text"] == "old"
        assert report["targets"][0]["reason"] == "prior_schema_failure"
        assert report["targets"][0]["error"] in {"prior_schema_text_mismatch", "prior_schema_missing_evidence"}


def test_transport_retry_backoff_and_success(monkeypatch):
    calls = 0
    sleeps = []

    async def request(prompt, model):
        nonlocal calls
        calls += 1
        return None if calls == 1 else response()

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(verifier.asyncio, "sleep", sleep)
    output, report, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        request_callable=request, transport_retries=1,
    ))
    assert output[0]["text"] == "new"
    assert report["requests"] == 2
    assert report["transport_retries"] == 1
    assert usage["transport_failures"] == 1
    assert sleeps == [1]


def test_exhausted_transport_retries_falls_back_and_counts(monkeypatch):
    sleeps = []

    async def request(prompt, model):
        return None

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(verifier.asyncio, "sleep", sleep)
    output, report, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        request_callable=request, transport_retries=2,
    ))
    assert output[0]["text"] == "old"
    assert report["requests"] == 3
    assert report["transport_retries"] == 2
    assert usage["transport_failures"] == 3
    assert sleeps == [1, 2]


def test_malformed_batch_splits_once_and_isolates_results():
    async def request(prompt, model):
        indices = [target["index"] for target in json.loads(prompt)["targets"]]
        if len(indices) > 1:
            return '{"results":[]}'
        return response(indices[0], "pass" if indices[0] == 1 else "fail", ["predicate"] if indices[0] == 2 else [], "major" if indices[0] == 2 else "none")

    output, report, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old1"}, {"index": 2, "text": "old2"}],
        [{"index": 1, "text": "new1"}, {"index": 2, "text": "new2"}],
        batch_size=2, schema_retries=1, request_callable=request,
    ))
    assert output[0]["text"] == "new1" and output[1]["text"] == "old2"
    assert report["requests"] == 3 and report["schema_split_retry"] is True
    assert usage["successful_responses"] == 2


def test_malformed_batch_with_no_schema_split_falls_back():
    async def request(prompt, model):
        return '{"results":[]}'

    output, report, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old1"}, {"index": 2, "text": "old2"}],
        [{"index": 1, "text": "new1"}, {"index": 2, "text": "new2"}],
        batch_size=2, schema_retries=0, request_callable=request,
    ))
    assert [item["text"] for item in output] == ["old1", "old2"]
    assert report["requests"] == 1 and report["schema_split_retry"] is False
    assert usage["successful_responses"] == 0


def test_schema_retries_only_accepts_zero_or_one():
    with pytest.raises(ValueError):
        verifier._validate_runtime_args("model", 3, 1, 1, 1, 2)
    with pytest.raises(SystemExit):
        verifier.build_parser().parse_args(["original", "candidate", "--schema-retries", "2"])


@pytest.mark.parametrize("server_url,expected_auth", [(None, None), ("http://configured", {"Authorization": "configured"})])
def test_main_auth_header_only_for_explicit_server(monkeypatch, tmp_path, server_url, expected_auth):
    original_path = tmp_path / "original.json"
    candidate_path = tmp_path / "candidate.json"
    original_path.write_text('[{"index": 1, "text": "old"}]')
    candidate_path.write_text('[{"index": 1, "text": "new"}]')
    seen = []

    class FakeServer:
        base_url = "http://started"

        def __init__(self, port, hostname):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    async def request_live(prompt, model, base_url, auth_header, structured_output=True):
        seen.append((base_url, auth_header, structured_output))
        return response()

    monkeypatch.setattr(verifier, "OpenCodeServer", FakeServer)
    monkeypatch.setattr(verifier, "_request_live", request_live)
    monkeypatch.setattr(verifier, "_get_auth_header", lambda: {"Authorization": "configured"})
    args = [str(original_path), str(candidate_path), "--output-dir", str(tmp_path / "out")]
    if server_url:
        args.extend(["--server-url", server_url])
    assert verifier.main(args) == 0
    assert seen == [(server_url or "http://started", expected_auth, False)]


def test_request_live_deletes_session_after_send_exception(monkeypatch):
    deleted = []

    async def create(base_url, title, auth_header):
        return "session"

    async def send(*args, **kwargs):
        raise RuntimeError("send failed")

    async def delete(base_url, session, auth_header):
        deleted.append((base_url, session, auth_header))

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt_result", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    with pytest.raises(RuntimeError):
        asyncio.run(verifier._request_live("prompt", "model", "http://server", {"Authorization": "x"}))
    assert deleted == [("http://server", "session", {"Authorization": "x"})]


def test_verifier_aggregates_two_opencode_results_exactly():
    results = [
        OpenCodePromptResult(response(), OpenCodePromptUsage(10, 20, 3, 30, 4, 5, 0.125, "p1", "m1", "stop")),
        OpenCodePromptResult(response(), OpenCodePromptUsage(2, 4, 1, 6, 7, 8, 0.25, "p2", "m2", "length")),
    ]

    async def request(prompt, model):
        return results.pop(0)

    _, _, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}, {"index": 2, "text": "old2"}],
        [{"index": 1, "text": "new"}, {"index": 2, "text": "new2"}],
        batch_size=1, request_callable=request,
    ))
    assert usage["token_usage_available"] is True
    assert usage["usage_response_count"] == 2
    assert {key: usage[key] for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")} == {"input_tokens": 12, "output_tokens": 24, "reasoning_tokens": 4, "total_tokens": 36, "cache_read_tokens": 11, "cache_write_tokens": 13}
    assert usage["provider_reported_cost_usd"] == 0.375
    assert usage["provider_cost_is_billing_authoritative"] is False
    assert usage["provider_ids"] == ["p1", "p2"] and usage["model_ids"] == ["m1", "m2"] and usage["finishes"] == ["length", "stop"]


def test_verifier_counts_usage_from_schema_failed_response():
    async def request(prompt, model):
        return OpenCodePromptResult('{"results":[]}', OpenCodePromptUsage(7, 8, 2, 15, 1, 0, 0.75, "provider", "model", "stop"))

    _, report, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}],
        schema_retries=0, request_callable=request,
    ))
    assert report["status"] == "completed_with_unresolved"
    assert usage["schema_failures"] == 1
    assert usage["usage_response_count"] == 1 and usage["token_usage_available"] is True
    assert usage["total_tokens"] == 15 and usage["provider_reported_cost_usd"] == 0.75


def test_verifier_string_fake_omits_provider_usage_fields_and_secrets():
    secret = "/private/secret/provider-token"

    async def request(prompt, model):
        return response()

    _, _, usage = asyncio.run(verify_items(
        [{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], request_callable=request,
    ))
    assert usage["token_usage_available"] is False
    forbidden = {"input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens", "provider_reported_cost_usd", "provider_cost_is_billing_authoritative", "usage_response_count", "provider_ids", "model_ids", "finishes"}
    assert forbidden.isdisjoint(usage)
    assert secret not in json.dumps(usage)


@pytest.mark.parametrize("kwargs", [{"backend": ""}, {"accounting": " "}, {"cost_is_billing_authoritative": "yes"}])
def test_verifier_backend_accounting_validation(kwargs):
    async def request(prompt, model):
        return response()
    with pytest.raises(ValueError, match="invalid_backend_accounting"):
        asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], request_callable=request, **kwargs))


def test_verifier_default_shape_and_openrouter_override():
    async def request(prompt, model):
        return OpenCodePromptResult(response(), OpenCodePromptUsage(1, 1, None, 2, None, None, None, "provider", "model", "stop"))
    _, default_report, default_usage = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], request_callable=request))
    assert default_report["backend"] == "opencode" and default_report["accounting"] == "subscription"
    assert default_usage["backend"] == "opencode" and default_usage["accounting"] == "subscription"
    assert default_usage["provider_cost_is_billing_authoritative"] is False
    _, report, usage = asyncio.run(verify_items([{"index": 1, "text": "old"}], [{"index": 1, "text": "new"}], request_callable=request, backend="openrouter", accounting="metered_api", cost_is_billing_authoritative=True))
    assert report["backend"] == "openrouter" and report["accounting"] == "metered_api"
    assert usage["provider_cost_is_billing_authoritative"] is True


def unit_result(unit: SemanticUnit, verdict: str = "pass", issues: list[str] | None = None, severity: str = "none") -> dict[str, object]:
    return {"cue_indices": list(unit.cue_indices), "verdict": verdict,
            "issues": issues or [], "severity": severity, "explanation": "checked"}


def unit_raw(units: list[SemanticUnit], **kwargs: object) -> str:
    return json.dumps({"results": [unit_result(unit, **kwargs) for unit in units]})


def test_unit_parser_accepts_pair_and_empty_and_rejects_strict_shapes():
    units = (SemanticUnit("u1", (1, 2), (1,)), SemanticUnit("u2", (3,), (3,)))
    assert parse_unit_verification_response(unit_raw(list(units)), units)["u1"]["verdict"] == "pass"
    assert parse_unit_verification_response('{"results":[]}', ()) == {}
    malformed = [
        '{"results": {}}',
        '{"results":[{"cue_indices":[9],"verdict":"pass","issues":[],"severity":"none","explanation":"x"}]}',
        unit_raw([units[0]]),
        json.dumps({"results": [unit_result(units[0]), unit_result(units[0])]}),
        json.dumps({"results": [{**unit_result(units[0]), "cue_indices": [2, 1]}]}),
        json.dumps({"results": [{**unit_result(units[0]), "cue_indices": [1, "2"]}]}),
        json.dumps({"results": [{**unit_result(units[0]), "issues": ["predicate"], "severity": "none"}]}),
        json.dumps({"results": [{**unit_result(units[0]), "verdict": "fail", "issues": [], "severity": "major"}]}),
    ]
    for raw in malformed:
        with pytest.raises(ValueError):
            parse_unit_verification_response(raw, units)


def test_unit_parser_maps_exact_cue_tuples_not_internal_ids():
    units = (SemanticUnit("hash-a", (10, 11), (10,)), SemanticUnit("hash-b", (20,), (20,)))
    raw = json.dumps({"results": [unit_result(units[1]), unit_result(units[0])]})
    parsed = parse_unit_verification_response(raw, units)
    assert set(parsed) == {"hash-a", "hash-b"}
    assert parsed["hash-a"]["cue_indices"] == [10, 11]
    assert parsed["hash-b"]["cue_indices"] == [20]


def test_unit_payload_supports_context_only_member_and_exact_context():
    seen: dict[str, object] = {}
    original = [{"index": 1, "text": "Он сказал,"}]
    candidate = [{"index": 1, "text": "Он сказал"}]
    context = [{"index": 99, "text": "начало", "start": -100, "end": -20}, {"index": 1, "text": "Он сказал,", "start": 0, "end": 100}, {"index": 2, "text": "что это", "start": 150, "end": 250}]

    async def request(prompt: str, model: str) -> str:
        seen["payload"] = json.loads(prompt)
        payload = seen["payload"]
        target = payload["targets"][0]
        return unit_raw([SemanticUnit("response-id-is-ignored", tuple(target["cue_indices"]), tuple(target["changed_indices"]))])

    output, _, _ = asyncio.run(verify_items(original, candidate, context_items=context, context_window=1, semantic_units=True, request_callable=request))
    assert output[0]["text"] == "Он сказал"
    target = seen["payload"]["targets"][0]
    assert "unit_id" not in target
    assert set(target) == {"cue_indices", "changed_indices", "original_cues", "candidate_cues", "context", "risk_hints"}
    assert target["original_cues"] == [{"index": 1, "text": "Он сказал,"}, {"index": 2, "text": "что это"}]
    assert target["candidate_cues"] == [{"index": 1, "text": "Он сказал"}, {"index": 2, "text": "что это"}]
    assert target["context"] == [{"index": 99, "original_text": "начало", "current_text": "начало"}]
    assert all(row["index"] == 1 for row in target["risk_hints"])


def test_unit_payload_context_never_leaks_sibling_candidates():
    seen: list[dict[str, object]] = []
    original = [
        {"index": 1, "text": "first original", "start": 0, "end": 100},
        {"index": 2, "text": "second original", "start": 1000, "end": 1100},
        {"index": 3, "text": "third original", "start": 2000, "end": 2100},
    ]
    candidate = [
        {**original[0], "text": "first candidate"},
        {**original[1], "text": "second candidate"},
        {**original[2], "text": "third candidate"},
    ]

    async def request(prompt: str, model: str) -> str:
        payload = json.loads(prompt)
        seen.extend(payload["targets"])
        units = [SemanticUnit("response-id-is-ignored", tuple(target["cue_indices"]), tuple(target["changed_indices"]))
                 for target in payload["targets"]]
        return unit_raw(units)

    asyncio.run(verify_items(original, candidate, context_window=1, semantic_units=True,
                             semantic_unit_max_cues=1, request_callable=request))
    assert len(seen) == 3
    for target in seen:
        assert all(row["original_text"] == row["current_text"] for row in target["context"])
        assert all(row["current_text"] == original[row["index"] - 1]["text"] for row in target["context"])


@pytest.mark.parametrize("verdict,issues,severity", [("fail", ["predicate"], "major"), ("uncertain", ["reference"], "minor")])
def test_unit_verdict_is_atomic_and_reports_unit_metadata(verdict: str, issues: list[str], severity: str):
    original = [{"index": 1, "text": "Он сказал,", "start": 0, "end": 100}, {"index": 2, "text": "что ушел", "start": 150, "end": 250}]
    candidate = [{**original[0], "text": "Он утверждает,"}, {**original[1], "text": "что приехал"}]

    async def request(prompt: str, model: str) -> str:
        payload = json.loads(prompt)
        units = [SemanticUnit("response-id-is-ignored", tuple(t["cue_indices"]), tuple(t["changed_indices"])) for t in payload["targets"]]
        return unit_raw(units, verdict=verdict, issues=issues, severity=severity)

    output, report, _ = asyncio.run(verify_items(original, candidate, semantic_units=True, request_callable=request))
    assert [item["text"] for item in output] == ["Он сказал,", "что ушел"]
    assert report["units"][0]["cue_indices"] == [1, 2]
    assert report["units"][0]["changed_indices"] == [1, 2]
    assert report["units"][0]["verdict"] == verdict
    assert [target["unit_id"] for target in report["targets"]] == [report["units"][0]["unit_id"]] * 2
    assert all(target["unit_indices"] == [1, 2] for target in report["targets"])


def test_unit_evidence_is_not_restored_or_reported_as_target():
    original = [{"index": 1, "text": "a", "meta": 1}, {"index": 2, "text": "b", "meta": 2}]
    candidate = [{"index": 1, "text": "changed", "meta": 3}, {"index": 2, "text": "b", "meta": 4}]

    async def request(prompt: str, model: str) -> str:
        target = json.loads(prompt)["targets"][0]
        unit = SemanticUnit("response-id-is-ignored", tuple(target["cue_indices"]), tuple(target["changed_indices"]))
        return unit_raw([unit], verdict="fail", issues=["predicate"], severity="major")

    output, report, _ = asyncio.run(verify_items(original, candidate, semantic_units=True, request_callable=request))
    assert output[0]["text"] == "a" and output[0]["meta"] == 3
    assert output[1] == candidate[1]
    assert [target["index"] for target in report["targets"]] == [1]


def test_unit_single_schema_failure_does_not_split_and_multi_batch_splits_units_only():
    single_calls = 0

    async def bad_single(prompt: str, model: str) -> str:
        nonlocal single_calls
        single_calls += 1
        return '{"results":[]}'

    asyncio.run(verify_items([{"index": 1, "text": "a"}], [{"index": 1, "text": "b"}], semantic_units=True, schema_retries=1, request_callable=bad_single))
    assert single_calls == 1

    calls: list[int] = []

    async def bad_batch(prompt: str, model: str) -> str:
        payload = json.loads(prompt)
        calls.append(len(payload["targets"]))
        if len(payload["targets"]) > 1:
            return '{"results":[]}'
        target = payload["targets"][0]
        unit = SemanticUnit("response-id-is-ignored", tuple(target["cue_indices"]), tuple(target["changed_indices"]))
        return unit_raw([unit])

    original = [{"index": 1, "text": "a."}, {"index": 2, "text": "b."}]
    candidate = [{**original[0], "text": "x"}, {**original[1], "text": "y"}]
    _, report, _ = asyncio.run(verify_items(original, candidate, semantic_units=True, batch_size=2, schema_retries=1, request_callable=bad_batch))
    assert calls == [2, 1, 1] and report["schema_split_retry"] is True


def test_unit_prior_restores_nonverified_candidate_before_prompt_and_prior_error_skips_units():
    seen: dict[str, object] = {}
    original = [{"index": 1, "text": "a,", "start": 0, "end": 100}, {"index": 2, "text": "b", "start": 150, "end": 250}]
    candidate = [{"index": 1, "text": "x", "start": 0, "end": 100}, {"index": 2, "text": "y", "start": 150, "end": 250}]
    prior = {"outcomes": [{"index": 1, "outcome": "verified", "candidate_text": "x"}, {"index": 2, "outcome": "unresolved", "candidate_text": "y"}]}

    async def request(prompt: str, model: str) -> str:
        payload = json.loads(prompt)
        seen["candidate_cues"] = payload["targets"][0]["candidate_cues"]
        target = payload["targets"][0]
        unit = SemanticUnit("response-id-is-ignored", tuple(target["cue_indices"]), tuple(target["changed_indices"]))
        return unit_raw([unit])

    output, report, _ = asyncio.run(verify_items(original, candidate, review_report=prior, semantic_units=True, request_callable=request))
    assert {row["index"]: row["text"] for row in seen["candidate_cues"]} == {1: "x", 2: "b"}
    assert output[0]["text"] == "x" and output[1]["text"] == "b"
    assert report["targets"][1]["reason"] == "prior_not_verified"

    calls = 0
    async def unexpected(prompt: str, model: str) -> str:
        nonlocal calls
        calls += 1
        return unit_raw([])
    _, bad_report, _ = asyncio.run(verify_items(original, candidate, review_report={"outcomes": [{"index": 1, "outcome": "verified"}]}, semantic_units=True, request_callable=unexpected))
    assert calls == 0 and bad_report["units"] == [] and all(target["reason"] == "prior_schema_failure" for target in bad_report["targets"])


def test_unit_missing_timing_stays_singleton_and_legacy_shape_is_unchanged():
    seen: list[dict[str, object]] = []
    async def request(prompt: str, model: str) -> str:
        payload = json.loads(prompt)
        seen.extend(payload["targets"])
        units = [SemanticUnit("response-id-is-ignored", tuple(t["cue_indices"]), tuple(t["changed_indices"])) for t in payload["targets"]]
        return unit_raw(units)
    original = [{"index": 1, "text": "a,"}, {"index": 2, "text": "b"}]
    candidate = [{"index": 1, "text": "x"}, {"index": 2, "text": "y"}]
    _, report, _ = asyncio.run(verify_items(original, candidate, semantic_units=True, request_callable=request))
    assert [target["cue_indices"] for target in seen] == [[1], [2]]
    async def legacy(prompt: str, model: str) -> str:
        assert "unit_id" not in json.loads(prompt)["targets"][0]
        return response()
    _, legacy_report, _ = asyncio.run(verify_items([{"index": 1, "text": "a"}], [{"index": 1, "text": "b"}], request_callable=legacy))
    assert "semantic_units_enabled" not in legacy_report and "units" not in legacy_report


def test_unit_prompt_runtime_defaults_live_validation_and_usage(monkeypatch):
    parser_args = verifier.build_parser().parse_args(["original", "candidate"])
    assert parser_args.model == "openai/gpt-5.6-sol"
    assert verifier.build_parser().parse_args(["original", "candidate", "--model", "openai/gpt-5.6-luna"]).model == "openai/gpt-5.6-luna"
    assert parser_args.semantic_units is False and parser_args.semantic_unit_max_cues == 3 and parser_args.semantic_unit_max_gap_sec == 0.3
    assert parser_args.structured_output is False
    assert verifier.build_parser().parse_args(["original", "candidate", "--structured-output"]).structured_output is True
    assert verifier.build_parser().parse_args(["original", "candidate", "--no-structured-output"]).structured_output is False
    with pytest.raises(ValueError):
        verifier._validate_runtime_args("model", 3, 1, 1, 1, 1, 0, 0.3)
    with pytest.raises(ValueError):
        verifier._validate_runtime_args("model", 3, 1, 1, 1, 1, 3, float("inf"))
    started = False
    class Server:
        base_url = "http://server"
        def __init__(self, port: int | None, hostname: str) -> None:
            nonlocal started
            started = True
        def start(self) -> None:
            pass
        def stop(self) -> None:
            pass
    monkeypatch.setattr(verifier, "OpenCodeServer", Server)
    with pytest.raises(ValueError):
        verifier.verify_items_live([], [], semantic_units=True, semantic_unit_max_cues=6)
    assert started is False


def test_verify_items_uses_sol_default_and_accepts_explicit_luna():
    models = []

    async def request(prompt, model):
        models.append(model)
        return response()

    original = [{"index": 1, "text": "old"}]
    candidate = [{"index": 1, "text": "new"}]
    asyncio.run(verifier.verify_items(original, candidate, request_callable=request))
    asyncio.run(verifier.verify_items(original, candidate, model="openai/gpt-5.6-luna", request_callable=request))
    assert models == ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"]

    results = [OpenCodePromptResult('{"results":[]}', OpenCodePromptUsage(10, 20, 3, 30, 4, 5, 0.125, "p", "m", "stop")),
               OpenCodePromptResult('{"results":[]}', OpenCodePromptUsage(2, 4, 1, 6, 7, 8, 0.25, "p", "m", "length"))]
    async def usage_request(prompt: str, model: str) -> OpenCodePromptResult:
        payload = json.loads(prompt)
        units = [SemanticUnit("response-id-is-ignored", tuple(t["cue_indices"]), tuple(t["changed_indices"])) for t in payload["targets"]]
        return OpenCodePromptResult(unit_raw(units), results.pop(0).usage)
    _, _, usage = asyncio.run(verify_items([{"index": 1, "text": "a"}, {"index": 2, "text": "b"}], [{"index": 1, "text": "x"}, {"index": 2, "text": "y"}], semantic_units=True, batch_size=1, request_callable=usage_request))
    assert usage["input_tokens"] == 12 and usage["output_tokens"] == 24 and usage["total_tokens"] == 36 and usage["provider_reported_cost_usd"] == 0.375


def test_request_live_selects_unit_prompt_from_new_target_shape(monkeypatch):
    seen: dict[str, str] = {}

    async def create(base_url, title, auth_header):
        return "session"

    async def send(base_url, session, system_prompt, prompt, model, auth_header):
        seen["system"] = system_prompt
        return OpenCodePromptResult('{"results":[]}', None)

    async def delete(base_url, session, auth_header):
        pass

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt_result", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    prompt = json.dumps({"targets": [{"cue_indices": [1], "original_cues": [], "changed_indices": [], "candidate_cues": [], "context": [], "risk_hints": []}]})
    asyncio.run(verifier._request_live(prompt, "model", "http://server", None))
    assert seen["system"] == verifier.UNIT_VERIFIER_SYSTEM_PROMPT
    assert "exactly one root key, results" in seen["system"]
    assert "exactly these keys and no others: cue_indices, verdict, issues, severity, explanation" in seen["system"]
    for field in ("unit_id", "changed_indices", "original_cues", "candidate_cues", "context", "risk_hints", "markdown", "tools", "rewrites", "reasoning"):
        assert field in seen["system"]
    assert "Filler deletion, natural paraphrase, and natural compression are allowed" in seen["system"]
    assert "literal substring preservation is not required" in seen["system"]
    assert "only for material semantic loss, material semantic change, or material ambiguity" in seen["system"]


@pytest.mark.parametrize("payload", [[{"targets": []}], {"targets": []}, {"targets": ["not-a-target"]}, {"targets": [None]}])
def test_request_live_rejects_malformed_unit_payload_shapes(monkeypatch, payload):
    seen: dict[str, str] = {}

    async def create(base_url, title, auth_header):
        return "session"

    async def send(base_url, session, system_prompt, prompt, model, auth_header):
        seen["system"] = system_prompt
        return OpenCodePromptResult('{"results":[]}', None)

    async def delete(base_url, session, auth_header):
        pass

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt_result", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    asyncio.run(verifier._request_live(json.dumps(payload), "model", "http://server", None))
    assert seen["system"] == verifier.INDEPENDENT_VERIFIER_SYSTEM_PROMPT


def test_unit_verification_schema_is_exact_and_fresh():
    schema = verifier.unit_verification_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["results"] and schema["additionalProperties"] is False
    item = schema["properties"]["results"]["items"]
    assert item["required"] == ["cue_indices", "verdict", "issues", "severity", "explanation"]
    assert item["additionalProperties"] is False
    assert item["properties"]["cue_indices"] == {"type": "array", "items": {"type": "integer"}, "minItems": 1}
    assert item["properties"]["verdict"]["enum"] == ["pass", "fail", "uncertain"]
    assert item["properties"]["issues"]["items"]["enum"] == sorted(verifier.ISSUE_CODES)
    assert item["properties"]["severity"]["enum"] == ["none", "minor", "major"]
    assert item["properties"]["explanation"] == {"type": "string", "minLength": 1}
    assert verifier.unit_verification_schema() is not schema


def test_request_live_passes_unit_schema_to_transport(monkeypatch):
    seen = {}

    async def create(base_url, title, auth_header):
        return "session"

    async def send(*args, **kwargs):
        seen["schema"] = kwargs.get("output_schema")
        return OpenCodePromptResult('{"results":[]}', None)

    async def delete(base_url, session, auth_header):
        pass

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt_result", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    prompt = json.dumps({"targets": [{"cue_indices": [1], "original_cues": []}]})
    asyncio.run(verifier._request_live(prompt, "model", "http://server", None))
    assert seen["schema"] == verifier.unit_verification_schema()


def test_request_live_unit_text_fallback_keeps_unit_prompt_and_omits_schema(monkeypatch):
    seen = {}

    async def create(base_url, title, auth_header):
        return "session"

    async def send(*args, **kwargs):
        seen["system"] = args[2]
        seen["schema"] = kwargs.get("output_schema", "missing")
        return OpenCodePromptResult('{"results":[]}', None)

    async def delete(base_url, session, auth_header):
        pass

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt_result", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    prompt = json.dumps({"targets": [{"cue_indices": [1], "original_cues": []}]})
    asyncio.run(verifier._request_live(prompt, "model", "http://server", None, structured_output=False))
    assert seen["system"] == verifier.UNIT_VERIFIER_SYSTEM_PROMPT
    assert seen["schema"] is None


def test_live_structured_output_defaults_and_explicit_flag(monkeypatch):
    captured = []

    async def fake_request(prompt, model, base_url, auth_header, structured_output=True):
        captured.append(structured_output)
        return OpenCodePromptResult(response(), None)

    monkeypatch.setattr(verifier, "_request_live", fake_request)
    original = [{"index": 1, "text": "old"}]
    candidate = [{"index": 1, "text": "new"}]
    verifier.verify_items_live(original, candidate, server_url="http://server")
    verifier.verify_items_live(original, candidate, server_url="http://server", structured_output=False)
    verifier.verify_items_live(original, candidate, server_url="http://server", structured_output=True)
    assert captured == [False, False, True]
