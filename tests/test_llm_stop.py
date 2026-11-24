"""LLM requests stop the run instead of paying silently: fallback model, paid guard, stop exit code, cost log."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import stt.cast
import stt.llm
from stt import run
from utils import llm_route
from utils.llm_route import LLM_STOP_EXIT, PaidNotAllowed, SubscriptionDown
from utils.opencode_clean import OpenCodeError
from voiceover.llm import LLMClient, LLMConfig

SOL, LUNA = "openai/gpt-6.1-sol", "openai/gpt-6-luna"


def busy() -> OpenCodeError:
    return OpenCodeError("HTTP 503: busy", transient=True)


def quota() -> OpenCodeError:
    return OpenCodeError("HTTP 502: usage limit reached", quota=True)


class ModelServer:
    """Answers per model from a script (an exception instance is raised, a string is returned); records the calls."""

    def __init__(self, script: dict[str, list[object]]) -> None:
        self.script, self.calls = {k: list(v) for k, v in script.items()}, []

    def generate(self, prompt: str, model: str, variant: Any = None, timeout: float = 0) -> str:
        self.calls.append(model)
        queue = self.script.get(model, [])
        item = queue.pop(0) if queue else busy()
        if isinstance(item, Exception):
            raise item
        return str(item)


@pytest.fixture()
def sub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Subscription with Sol -> Luna fallback after 10 s of a 60 s budget; sleeping is recorded, not done."""
    state: dict[str, Any] = {"sleeps": [], "server": ModelServer({})}
    monkeypatch.setenv(llm_route.FALLBACK_ENV, json.dumps({SOL: LUNA}))
    monkeypatch.setenv(llm_route.FALLBACK_AFTER_ENV, "10")
    monkeypatch.setenv("STT_LLM_SUBSCRIPTION_RETRY_S", "60")
    monkeypatch.setattr(stt.llm.time, "sleep", state["sleeps"].append)
    monkeypatch.setattr(stt.llm, "shared_server", lambda model: state["server"])
    monkeypatch.setattr(stt.llm, "load_api_key", lambda *a, **k: "k")
    return state


def test_luna_stands_in_for_one_request_and_sol_is_tried_first_again(sub: dict[str, Any],
                                                                      caplog: pytest.LogCaptureFixture) -> None:
    sub["server"] = ModelServer({SOL: [busy()] * 50 + ['{"sol": 2}'], LUNA: ['{"luna": 1}']})
    client = stt.llm.OpenRouterClient("opencode:" + SOL)
    with caplog.at_level(logging.WARNING, logger="utils.llm_route"):
        assert client.chat_json("s", "u") == {"luna": 1}
    assert "this request went to openai/gpt-6-luna (subscription, free)" in caplog.text
    assert sum(sub["sleeps"]) == pytest.approx(10.0, abs=0.1)  # Sol got only the fallback delay, not the whole budget
    assert client.usage.subscription_fallback_calls == 1
    sub["server"].script[SOL] = ['{"sol": 2}']
    assert client.chat_json("s", "u2") == {"sol": 2}  # no sticky switching
    assert sub["server"].calls[-1] == SOL and client.usage.subscription_fallback_calls == 1


def test_a_model_without_a_fallback_gets_the_whole_budget(sub: dict[str, Any]) -> None:
    sub["server"] = ModelServer({})
    client = stt.llm.OpenRouterClient("opencode:" + LUNA)
    with pytest.raises(SubscriptionDown):
        client.chat_json("s", "u")
    assert sum(sub["sleeps"]) == pytest.approx(60.0, abs=0.1) and set(sub["server"].calls) == {LUNA}


def test_sol_and_luna_both_failing_stops_the_run(sub: dict[str, Any]) -> None:
    sub["server"] = ModelServer({})
    client = stt.llm.OpenRouterClient("opencode:" + SOL)
    with pytest.raises(SubscriptionDown, match="not answering"):
        client.chat_json("s", "u")
    assert set(sub["server"].calls) == {SOL, LUNA} and sum(sub["sleeps"]) == pytest.approx(60.0, abs=0.1)


def test_quota_goes_to_luna_and_both_used_up_stops(sub: dict[str, Any]) -> None:
    sub["server"] = ModelServer({SOL: [quota()], LUNA: [quota()]})
    client = stt.llm.OpenRouterClient("opencode:" + SOL)
    with pytest.raises(SubscriptionDown, match="usage limit"):
        client.chat_json("s", "u")
    assert sub["server"].calls == [SOL, LUNA] and sub["sleeps"] == []
    sub["server"] = ModelServer({SOL: [quota()], LUNA: ['{"ok": 1}']})
    assert client.chat_json("s", "u2") == {"ok": 1}


def test_a_paid_route_after_the_subscription_still_covers_one_request(sub: dict[str, Any]) -> None:
    sub["server"] = ModelServer({})
    posted: list[str] = []

    def post(self: Any, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        posted.append(router)
        return {"model": "m", "choices": [{"message": {"content": '{"paid": 1}'}}], "usage": {"cost": 0.25}}

    with mock.patch.object(stt.llm.OpenRouterClient, "_post", post):
        client = stt.llm.OpenRouterClient("opencode+openrouter:" + SOL)
        assert client.chat_json("s", "u") == {"paid": 1}
    assert posted == ["openrouter"]


def test_deny_raises_before_anything_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm_route.PAID_ENV, "deny")
    monkeypatch.setattr(stt.llm, "load_api_key", lambda *a, **k: "k")
    posted: list[str] = []
    monkeypatch.setattr(stt.llm.OpenRouterClient, "_post", lambda self, body, router="openrouter": posted.append(router))
    with pytest.raises(PaidNotAllowed, match="openrouter"):
        stt.llm.OpenRouterClient("openrouter:" + LUNA).chat_json("s", "u")
    assert posted == []
    # a subscription failure is not turned into a paid request either
    monkeypatch.setattr(stt.llm, "shared_server", lambda model: ModelServer({}))
    monkeypatch.setattr(stt.llm.time, "sleep", lambda s: None)
    client = stt.llm.OpenRouterClient("opencode+openrouter:" + LUNA, subscription_retry_s=5)
    with pytest.raises(SubscriptionDown):
        client.chat_json("s", "u")
    assert posted == []


def test_voiceover_guard_and_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import utils.opencode_clean as opencode
    import voiceover.llm as vllm

    monkeypatch.setenv(llm_route.PAID_ENV, "deny")
    monkeypatch.setattr(vllm, "load_api_key", lambda name, env_file=None: "key")
    sent: list[str] = []
    monkeypatch.setattr(vllm, "openai_transport", lambda config, key: lambda body: sent.append("x") or {})
    for model in ("openrouter:" + LUNA, LUNA):  # a routed paid model and a plain provider model
        with pytest.raises(PaidNotAllowed):
            vllm.route_transport(LLMConfig(model=model, cache_dir=tmp_path))({"model": "x", "messages": []})
    assert sent == []
    # Sol -> Luna inside the subscription, counted; both down: SubscriptionDown and the client stays stopped
    monkeypatch.setenv(llm_route.FALLBACK_ENV, json.dumps({SOL: LUNA}))
    monkeypatch.setenv(llm_route.FALLBACK_AFTER_ENV, "10")
    monkeypatch.setenv(opencode.SUBSCRIPTION_RETRY_ENV, "30")
    monkeypatch.setattr(opencode.time, "sleep", lambda s: None)
    server = ModelServer({SOL: [busy()] * 20, LUNA: ['{"a": 1}']})
    monkeypatch.setattr(opencode, "shared_server", lambda model: server)
    client = LLMClient(LLMConfig(model="opencode:" + SOL, cache_dir=tmp_path), sleep=lambda s: None)
    assert client.complete_json("s", "u", prompt_version="v", tag="t").content == {"a": 1}
    assert client.stats()["subscription_fallback_calls"] == 1
    server.script[LUNA] = []
    with pytest.raises(SubscriptionDown):
        client.complete_json("s", "u2", prompt_version="v", tag="t")
    with pytest.raises(SubscriptionDown):  # no more calls once the subscription gave up
        client.complete_json("s", "u3", prompt_version="v", tag="t")
    assert client.exhausted


class FakeCastClient:
    """Stands in for OpenRouterClient in ``find_cast``: records the models, raises what the script says."""

    built: list[str] = []
    errors: dict[str, Exception] = {}

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        FakeCastClient.built.append(model)


def test_cast_never_reaches_a_paid_model_when_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeCastClient.built = []
    models = ["opencode:" + SOL, "openrouter:" + LUNA + ":online"]

    def lookup(client: FakeCastClient, *args: Any) -> dict[str, Any]:
        raise SubscriptionDown("not answering")

    monkeypatch.setattr(stt.cast, "OpenRouterClient", FakeCastClient)
    monkeypatch.setattr(stt.cast, "wiki_cast", lookup)
    monkeypatch.setattr(stt.cast, "fetch_cast", lookup)
    monkeypatch.setenv(llm_route.PAID_ENV, "deny")
    with pytest.raises(SubscriptionDown):
        stt.cast.find_cast(models, "Survivor", "S01E01.mkv", [])
    assert FakeCastClient.built == ["opencode:" + SOL]
    # allowed to pay: the next (paid) model is tried; when the last one cannot answer either, the run stops
    FakeCastClient.built = []
    monkeypatch.delenv(llm_route.PAID_ENV)
    with pytest.raises(SubscriptionDown):
        stt.cast.find_cast(models, "Survivor", "S01E01.mkv", [])
    assert len(FakeCastClient.built) == 2
    # another subscription model may still answer; PaidNotAllowed ends the lookup at once
    FakeCastClient.built = []
    monkeypatch.setenv(llm_route.PAID_ENV, "deny")
    answers: list[Exception | dict[str, Any]] = [SubscriptionDown("x"), {"cast": [{}] * 20}]

    def scripted(client: FakeCastClient, *args: Any) -> dict[str, Any]:
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(stt.cast, "wiki_cast", scripted)
    assert stt.cast.find_cast(["opencode:" + SOL, "opencode:" + LUNA], "S", "f", [])["cast"]
    monkeypatch.setattr(stt.cast, "wiki_cast", lambda *a: (_ for _ in ()).throw(PaidNotAllowed("cast")))
    with pytest.raises(PaidNotAllowed):
        stt.cast.find_cast(["opencode:" + SOL, "opencode:" + LUNA], "S", "f", [])


CONFIG: dict[str, Any] = {"llm": {"model": SOL, "opencode": ["openai/*"], "router": ["openrouter"], "fallback": False,
                                  "stages": {"cast": [SOL, "openrouter:" + LUNA + ":online"],
                                             "stress": [SOL, "anthropic/claude-sonnet-5.5"]}}}


def test_paying_is_allowed_only_without_a_subscription_or_with_the_fallback() -> None:
    assert not run.paying_allowed(CONFIG)
    assert run.paying_allowed({"llm": {**CONFIG["llm"], "opencode": []}})
    assert run.paying_allowed({"llm": {**CONFIG["llm"], "fallback": True}})
    run.export_llm_env({"llm": {**CONFIG["llm"], "subscription_fallback": {SOL: LUNA}}})
    import os
    assert os.environ[llm_route.PAID_ENV] == "deny" and json.loads(os.environ[llm_route.FALLBACK_ENV]) == {SOL: LUNA}


def test_startup_drops_a_paid_backup_and_stops_on_a_paid_primary(caplog: pytest.LogCaptureFixture) -> None:
    import copy

    config = copy.deepcopy(CONFIG)
    with caplog.at_level(logging.WARNING, logger="stt.run"):
        run.check_paid_steps(config)
    assert config["llm"]["stages"]["cast"] == [SOL] and config["llm"]["stages"]["stress"] == [SOL]
    assert "may not pay" in caplog.text
    config["llm"]["stages"]["cast"] = ["openrouter:" + LUNA + ":online", SOL]
    config["llm"]["stages"]["translate_proofread"] = "anthropic/claude-sonnet-5.5"
    with pytest.raises(PaidNotAllowed) as caught:
        run.check_paid_steps(config)
    assert "cast: openrouter:" in caught.value.reason and "translate_proofread" in caught.value.reason
    config["llm"]["opencode"] = []  # this run was set up to pay: nothing to check
    run.check_paid_steps(config)


def test_stage_exit_75_stops_the_run_with_the_stage_message(tmp_path: Path) -> None:
    message = llm_route.stop_message(SubscriptionDown("usage limit reached"), 0.0)
    episode = mock.Mock(out=tmp_path, media=Path("ep.mkv"))

    def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
        kwargs["stdout"].write(f"2026-11-23 10:00:00 INFO x: working\n2026-11-23 10:00:01 ERROR {message}\n")
        return mock.Mock(returncode=LLM_STOP_EXIT)

    with mock.patch.object(run.subprocess, "run", fake_run), mock.patch.object(run, "dotenv", return_value={}), \
            pytest.raises(SubscriptionDown) as caught:
        run.Episode._module(episode, "stress", "voiceover.stress_check", [])
    assert caught.value.message == message and "usage limit reached" in caught.value.message


def test_main_logs_the_stop_last_and_exits_75(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    video = tmp_path / "S01E01.mkv"
    video.write_text("")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**CONFIG, "mode": "fast"}), encoding="utf-8")

    def stop(*args: Any) -> None:
        llm_route.record_paid("openai/gpt-6-luna", "openrouter", 0.5)  # something was paid before the stop
        raise SubscriptionDown("not answering")

    monkeypatch.setattr(run, "run_all", stop)
    with caplog.at_level(logging.INFO, logger="stt.run"), pytest.raises(SystemExit) as caught:
        run.main([str(video), "--out-root", str(tmp_path / "R"), "--config", str(config)])
    assert caught.value.code == LLM_STOP_EXIT
    last = caplog.records[-1].getMessage()
    assert last.startswith(llm_route.STOP_MARK) and "Paid in this run before the stop: $0.50" in last
    assert any("LLM paid in this run: $0.50" in r.getMessage() for r in caplog.records[:-1])
    assert (tmp_path / "R" / ".llm_cost").is_dir()


def test_stage_entry_point_turns_a_stop_into_exit_75(caplog: pytest.LogCaptureFixture) -> None:
    @llm_route.exiting_on_stop
    def main() -> int:
        raise PaidNotAllowed("stress: model via openrouter")

    with caplog.at_level(logging.ERROR):
        assert main() == LLM_STOP_EXIT
    text = caplog.records[-1].getMessage()
    assert "may not pay" in text and "stress: model via openrouter" in text and "OpenRouter" in text
    assert llm_route.stop_message_in("2026 ERROR x: " + text + "\n2026 INFO later\n").startswith(llm_route.STOP_MARK)


def test_cost_log_is_written_and_summed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cost_log = tmp_path / "cost" / "run.jsonl"
    monkeypatch.setenv(llm_route.COST_LOG_ENV, str(cost_log))
    assert llm_route.paid_line(cost_log) == llm_route.NOTHING_PAID_MARK + " / в этом запуске ничего не оплачено"
    monkeypatch.setattr(stt.llm, "load_api_key", lambda *a, **k: "k")
    monkeypatch.setattr(stt.llm, "shared_server", lambda model: ModelServer({LUNA: ['{"sub": 1}']}))

    def post(self: Any, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        return {"model": "m", "choices": [{"message": {"content": '{"paid": 1}'}}], "usage": {"cost": 0.25}}

    monkeypatch.setattr(stt.llm.OpenRouterClient, "_post", post)
    stt.llm.OpenRouterClient("opencode:" + LUNA).chat_json("s", "u")  # the subscription is free: no line
    assert not cost_log.exists()
    paid = stt.llm.OpenRouterClient("openrouter:" + SOL)
    paid.chat_json("s", "a")
    paid.chat_json("s", "b")
    rows = [json.loads(line) for line in cost_log.read_text().splitlines()]
    assert [r["cost_usd"] for r in rows] == [0.25, 0.25] and rows[0]["transport"] == "openrouter"
    assert rows[0]["model"] == SOL and {"time", "stage"} <= set(rows[0])
    assert llm_route.paid_total(cost_log) == (0.5, 2)
    assert "$0.50 (2 requests" in llm_route.paid_line(cost_log) and SOL in llm_route.paid_line(cost_log)


def test_voiceover_client_reads_the_cost_of_a_paid_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voiceover.llm as vllm

    cost_log = tmp_path / "cost.jsonl"
    monkeypatch.setenv(llm_route.COST_LOG_ENV, str(cost_log))
    monkeypatch.setattr(vllm, "load_api_key", lambda name, env_file=None: "key")
    monkeypatch.setattr(vllm, "openai_transport", lambda config, key: lambda body: {
        "model": body["model"], "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.125}})
    client = LLMClient(LLMConfig(model="openrouter:" + LUNA, cache_dir=tmp_path / "c"), sleep=lambda s: None)
    client.complete_json("s", "u", prompt_version="v", tag="t")
    assert llm_route.paid_total(cost_log) == (0.125, 1)


class JuryClient:
    """A voter: answers every question (``reply``) or fails with ``failure`` (then it is exhausted)."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.config = mock.Mock(model="voter")
        self.failure, self.exhausted = failure, None

    def complete_json(self, *args: Any, **kwargs: Any) -> Any:
        if self.failure:
            self.exhausted = str(self.failure)
            raise self.failure
        return mock.Mock(content={"items": [{"id": 0, "stressed": "гол+оса"}]})


def test_jury_leaves_a_failed_voter_out_but_stops_when_none_answers(caplog: pytest.LogCaptureFixture) -> None:
    from voiceover import stress_check

    suspect = stress_check.Suspect(word="голоса", accented="голос+а", source="homograph", contexts=["Два голоса."])
    down, fine = JuryClient(SubscriptionDown("usage limit")), JuryClient()
    with caplog.at_level(logging.WARNING):
        assert stress_check.ask([(suspect, "Два голоса.")], down) == [None]  # left out of the vote
        stress_check._require_answered([down, fine], "voter", ["a", "b"])  # one voter is enough
    assert "left out of the vote" in caplog.text and "gave no votes" in caplog.text
    other = JuryClient(SubscriptionDown("x"))
    stress_check.ask([(suspect, "Два голоса.")], other)
    with pytest.raises(SubscriptionDown, match="voter"):
        stress_check._require_answered([down, other], "voter", ["a", "b"])
    with pytest.raises(PaidNotAllowed):  # never left out
        stress_check.ask([(suspect, "Два голоса.")], JuryClient(PaidNotAllowed("stress: x via openrouter")))


def test_only_real_outages_stop_the_run() -> None:
    """A failure of one request (bad request, refusal) is not "the subscription is down"; the limit, an exhausted
    retry budget and a refused login are."""
    from utils.llm_route import is_subscription_down
    from utils.opencode_clean import OpenCodeError

    assert not is_subscription_down(OpenCodeError("HTTP 400: bad request", status=400))
    assert not is_subscription_down(OpenCodeError("answer without text", transient=True))
    assert is_subscription_down(OpenCodeError("HTTP 502: usage limit reached", quota=True, status=502))
    assert is_subscription_down(OpenCodeError("subscription not answering after 600 s", exhausted=True))
    assert is_subscription_down(OpenCodeError("HTTP 401: token expired", status=401))


# ------------------------------------------------------------------ review fixes


def test_dotenv_never_overrides_the_runs_llm_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm_route.PAID_ENV, "deny")
    seen: dict[str, str] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
        seen.update(kwargs["env"])
        return mock.Mock(returncode=0)

    episode = mock.Mock(out=tmp_path, media=Path("ep.mkv"))
    with mock.patch.object(run.subprocess, "run", fake_run), \
            mock.patch.object(run, "dotenv", return_value={"STT_LLM_PAID": "allow", "OPENROUTER_API_KEY": "k"}):
        run.Episode._module(episode, "stress", "voiceover.stress_check", [])
    assert seen[llm_route.PAID_ENV] == "deny" and seen["OPENROUTER_API_KEY"] == "k"


def test_the_budget_counts_request_time_and_the_stand_in_gets_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode

    clock = [0.0]
    monkeypatch.setattr(opencode.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(opencode.time, "sleep", lambda s: None)
    monkeypatch.setenv(llm_route.FALLBACK_ENV, json.dumps({SOL: LUNA}))
    monkeypatch.setenv(llm_route.FALLBACK_AFTER_ENV, "100")

    class Slow:
        """Every failing request takes 50 s; Luna answers after one more failure."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str, model: str, variant: Any = None, timeout: float = 0) -> str:
            self.calls.append(model)
            clock[0] += 50.0
            if model == LUNA and self.calls.count(LUNA) > 1:
                return "ok"
            raise busy()

    server = Slow()
    # without a fallback: 50 s per request + pauses 5, 15, 30 against 120 s -> gives up after about 3 tries
    with pytest.raises(OpenCodeError) as caught:
        opencode.generate_with_retry(lambda: server, "p", "openai/other", None, 300.0, 120.0)
    assert caught.value.exhausted and server.calls == ["openai/other"] * 3 and caught.value.spent >= 120.0
    # with the stand-in: Sol used its 100 s (2 requests), Luna gets the rest of 180 s, not a fresh budget
    server.calls.clear()
    clock[0] = 0.0
    text, used = llm_route.generate_via_subscription(lambda m: server, "p", SOL, None, 300.0, 180.0)
    assert (text, used) == ("ok", LUNA) and server.calls == [SOL, SOL, LUNA, LUNA]


def test_a_server_that_cannot_start_is_retried_then_counts_as_subscription_down(
        sub: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode
    import voiceover.llm as vllm

    starts = [RuntimeError("OpenAI OAuth auth preflight failed"), ModelServer({LUNA: ['{"ok": 1}']})]

    def shared(model: str) -> Any:
        item = starts.pop(0) if starts else RuntimeError("OpenAI OAuth auth preflight failed")
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(stt.llm, "shared_server", shared)
    assert stt.llm.OpenRouterClient("opencode:" + LUNA).chat_json("s", "u") == {"ok": 1}  # retried within the budget
    with pytest.raises(SubscriptionDown, match="not answering"):
        stt.llm.OpenRouterClient("opencode:" + LUNA).chat_json("s", "u")
    # in the voice-over the stress jury leaves the voter out instead of swallowing a plain error
    from voiceover import stress_check

    monkeypatch.setattr(opencode.time, "sleep", lambda s: None)
    monkeypatch.setenv(opencode.SUBSCRIPTION_RETRY_ENV, "10")
    monkeypatch.setattr(opencode, "shared_server", shared)
    client = LLMClient(LLMConfig(model="opencode:" + LUNA, cache_dir=Path("/tmp/none-cache-unused")),
                       sleep=lambda s: None)
    suspect = stress_check.Suspect(word="голоса", accented="голос+а", source="homograph", contexts=["Два голоса."])
    assert stress_check.ask([(suspect, "Два голоса.")], client) == [None]
    assert client.exhausted  # recorded: ``check`` can stop the run when nobody answers


def test_an_empty_answer_fails_that_request_only(sub: dict[str, Any]) -> None:
    empty = OpenCodeError("answer without text", transient=True, empty=True)
    sub["server"] = ModelServer({LUNA: [empty, empty, '{"ok": 1}']})
    assert stt.llm.OpenRouterClient("opencode:" + LUNA).chat_json("s", "u") == {"ok": 1}  # retried twice
    sub["server"] = ModelServer({LUNA: [empty] * 10})
    client = stt.llm.OpenRouterClient("opencode:" + LUNA)
    with pytest.raises(RuntimeError, match="answer without text") as caught:
        client.chat_json("s", "u")
    assert not isinstance(caught.value, llm_route.LLMStop) and sub["server"].calls.count(LUNA) == 3
    assert sum(sub["sleeps"]) < 60  # it never ate the budget


def test_wait_onscreen_joins_every_job_before_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import Future

    first, second, third = Future(), Future(), Future()
    first.set_exception(RuntimeError("other"))
    second.set_exception(SubscriptionDown("down"))
    third.set_result(None)
    joined: list[str] = []
    for name, job in (("a", first), ("b", second), ("c", third)):
        real = job.result
        monkeypatch.setattr(job, "result", lambda real=real, name=name: (joined.append(name), real())[1])
    monkeypatch.setattr(run, "_ONSCREEN_JOBS", {Path("a"): first, Path("b"): second, Path("c"): third})
    with pytest.raises(SubscriptionDown):
        run.wait_onscreen()
    assert joined == ["a", "b", "c"]


def test_stress_json_names_the_voters_that_were_left_out() -> None:
    import inspect

    from voiceover import stress_check

    assert "voters_left_out" in inspect.getsource(stress_check.check)


# ------------------------------------------------------------------ a run that may pay: the paid transport fails too


def _http_error(code: int, body: str) -> Exception:
    import io
    import urllib.error

    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]


def test_a_paid_run_stops_with_paid_down_when_openrouter_refuses(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    monkeypatch.setenv(llm_route.PAID_ENV, "allow")
    monkeypatch.setenv(llm_route.COST_LOG_ENV, str(tmp_path / "cost.jsonl"))
    monkeypatch.setattr(stt.llm, "load_api_key", lambda *a, **k: "k")
    monkeypatch.setattr(stt.llm, "shared_server", lambda model: ModelServer({}))
    monkeypatch.setattr(stt.llm.time, "sleep", lambda s: None)

    def post(self: Any, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        raise _http_error(402, '{"error": {"message": "Insufficient credits"}}')

    monkeypatch.setattr(stt.llm.OpenRouterClient, "_post", post)
    client = stt.llm.OpenRouterClient("opencode+openrouter:" + LUNA, subscription_retry_s=5)
    with pytest.raises(llm_route.PaidDown) as caught:
        client.chat_json("s", "u")
    assert caught.value.subscription_tried
    message = caught.value.message
    assert "did not answer and OpenRouter did not answer either" in message and "Nothing was paid." in message
    assert "allow paying" not in message and "--transport openrouter" not in message  # paying already failed
    assert "OpenRouter тоже не ответил" in llm_route.stop_message_ru(message)
    # a plain refusal of one request (400) is not "for good"
    monkeypatch.setattr(stt.llm.OpenRouterClient, "_post",
                        lambda self, body, router="openrouter": (_ for _ in ()).throw(_http_error(400, "bad")))
    with pytest.raises(RuntimeError) as other:
        stt.llm.OpenRouterClient("openrouter:" + LUNA).chat_json("s", "u")
    assert not isinstance(other.value, llm_route.LLMStop)


def test_voiceover_paid_failure_is_paid_down_and_the_jury_goes_on(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    import voiceover.llm as vllm
    from voiceover import stress_check

    monkeypatch.setenv(llm_route.PAID_ENV, "allow")
    monkeypatch.setattr(vllm, "load_api_key", lambda name, env_file=None: "key")

    def refuse(config: LLMConfig, key: str) -> Any:
        def call(body: dict[str, Any]) -> dict[str, Any]:
            raise vllm.LLMError("request rejected with HTTP 402: Insufficient credits")
        return call

    monkeypatch.setattr(vllm, "openai_transport", refuse)
    client = LLMClient(LLMConfig(model="openrouter:" + LUNA, cache_dir=tmp_path / "c"), sleep=lambda s: None)
    with pytest.raises(llm_route.PaidDown):
        client.complete_json("s", "u", prompt_version="v", tag="t")
    suspect = stress_check.Suspect(word="голоса", accented="голос+а", source="homograph", contexts=["Два голоса."])
    assert stress_check.ask([(suspect, "Два голоса.")], client) == [None]  # that voter is left out
    fine = JuryClient()
    stress_check._require_answered([client, fine], "voter", ["a", "b"])  # the run goes on with the others
    with pytest.raises(llm_route.PaidDown):  # nobody answers: stop
        stress_check._require_answered([client], "voter", ["a"])


def test_luna_answer_is_not_kept_under_sol_and_the_restart_asks_sol(sub: dict[str, Any], tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from stt import progress

    monkeypatch.setattr(stt.llm, "_KEPT_NOW", set())
    partial = tmp_path / ".partial" / "llm" / "translate"
    sub["server"] = ModelServer({SOL: [busy()] * 50, LUNA: ['{"luna": 1}']})
    with pytest.raises(KeyboardInterrupt):  # the stage stops after the stand-in answered
        with progress.track(tmp_path, "translate"):
            client = stt.llm.OpenRouterClient("opencode:" + SOL)
            assert client.chat_json("s", "u") == {"luna": 1}
            raise KeyboardInterrupt
    assert not list(partial.glob("*.json"))  # Luna's answer is not kept as Sol's
    record = stt.llm.read_provenance(tmp_path)["stages"]["translate"]
    assert record["fallback_calls"] == 1 and record["fallback_models"] == ["opencode:" + LUNA]
    assert record["requested_models"] == [SOL]
    assert client.usage.to_dict()["fallback_used"] is True
    assert client.usage.to_dict()["fallback_models"] == ["opencode:" + LUNA]

    monkeypatch.setattr(stt.llm, "_KEPT_NOW", set())  # a new process
    sub["server"] = ModelServer({SOL: ['{"sol": 2}']})
    with progress.track(tmp_path, "translate"):  # the restarted stage asks Sol again
        again = stt.llm.OpenRouterClient("opencode:" + SOL)
        assert again.chat_json("s", "u") == {"sol": 2}
    assert sub["server"].calls == [SOL] and len(list(partial.glob("*.json"))) == 1  # Sol's answer is kept
    assert again.usage.to_dict()["fallback_used"] is False


def test_fallback_provenance_is_per_stage_attempt_and_only_inside_a_stage(sub: dict[str, Any],
                                                                          tmp_path: Path) -> None:
    from stt import progress

    assert stt.llm.note_fallback(SOL, "opencode:" + LUNA) is None  # outside a stage: nothing is written
    assert stt.llm.read_provenance(tmp_path) == {"stages": {}}
    with progress.track(tmp_path, "glossary"):
        stt.llm.note_fallback(SOL, "opencode:" + LUNA)
        path = stt.llm.note_fallback(SOL, "opencode:" + LUNA)
    with progress.track(tmp_path, "voiceover", engine="vosk"):
        stt.llm.note_fallback(SOL, "opencode:" + LUNA)
    assert path == tmp_path / "llm_provenance.json"
    stages = stt.llm.read_provenance(tmp_path)["stages"]
    assert stages["glossary"]["fallback_calls"] == 2 and stages["voiceover.vosk"]["fallback_calls"] == 1
    with progress.track(tmp_path, "glossary") as tracker:  # the stage starts again: its old record is replaced
        tracker.started += 1.0
        stt.llm.note_fallback(SOL, "opencode:" + LUNA)
    assert stt.llm.read_provenance(tmp_path)["stages"]["glossary"]["fallback_calls"] == 1


def test_usage_report_keeps_its_old_fields() -> None:
    usage = stt.llm.Usage()
    usage.add({"model": "opencode:" + SOL, "usage": {"prompt_tokens": 2, "cost": 0.5}})
    assert usage.to_dict() == {"calls": 1, "prompt_tokens": 2, "completion_tokens": 0, "cost_usd": 0.5,
                               "subscription_fallback_calls": 0, "fallback_used": False, "fallback_models": [],
                               "models": ["opencode:" + SOL]}
    usage.add({"model": "opencode:" + LUNA, "subscription_fallback": True})
    report = usage.to_dict()
    assert report["subscription_fallback_calls"] == 1 and report["fallback_used"] is True
    assert report["fallback_models"] == ["opencode:" + LUNA] and report["models"] == sorted(["opencode:" + SOL,
                                                                                             "opencode:" + LUNA])
