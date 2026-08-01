import asyncio
import json

import pytest

import utils.verify_subtitles_opencode as verifier
from utils.verify_subtitles_opencode import parse_verification_response, verify_items


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

    async def request_live(prompt, model, base_url, auth_header):
        seen.append((base_url, auth_header))
        return response()

    monkeypatch.setattr(verifier, "OpenCodeServer", FakeServer)
    monkeypatch.setattr(verifier, "_request_live", request_live)
    monkeypatch.setattr(verifier, "_get_auth_header", lambda: {"Authorization": "configured"})
    args = [str(original_path), str(candidate_path), "--output-dir", str(tmp_path / "out")]
    if server_url:
        args.extend(["--server-url", server_url])
    assert verifier.main(args) == 0
    assert seen == [(server_url or "http://started", expected_auth)]


def test_request_live_deletes_session_after_send_exception(monkeypatch):
    deleted = []

    async def create(base_url, title, auth_header):
        return "session"

    async def send(*args, **kwargs):
        raise RuntimeError("send failed")

    async def delete(base_url, session, auth_header):
        deleted.append((base_url, session, auth_header))

    monkeypatch.setattr(verifier, "create_session", create)
    monkeypatch.setattr(verifier, "send_prompt", send)
    monkeypatch.setattr(verifier, "delete_session", delete)
    with pytest.raises(RuntimeError):
        asyncio.run(verifier._request_live("prompt", "model", "http://server", {"Authorization": "x"}))
    assert deleted == [("http://server", "session", {"Authorization": "x"})]
