from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.fakes import FakeTransport, make_client
from tests.phase4_fixtures import build_world
from voiceover.verify import VerifyPolicy, verify_rung, verify_rungs


def _entry(tmp_path: Path) -> dict[str, Any]:
    return build_world(tmp_path, {}).ledger[0]


def _rung(entry: dict[str, Any], number: int) -> dict[str, Any]:
    return next(r for r in entry["rungs"] if r["rung"] == number)


def _answer(**kept: bool) -> dict[str, Any]:
    return {"fragments": [{"id": key, "kept": value, "why": "x"} for key, value in kept.items()]}


def test_core_lost_fails_the_rung(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    transport = FakeTransport(lambda body: _answer(u2=False, u3=True))
    verdict = verify_rung(make_client(tmp_path, transport), VerifyPolicy(), entry, _rung(entry, 1))
    assert not verdict["ok"] and verdict["core_lost"] == ["u2"] and verdict["support_lost"] == []
    assert "core_lost" in verdict["reasons"] and transport.calls == 1
    prompt = transport.bodies[0]["messages"][1]["content"]
    assert "u2: это очень сложный вопрос (core)" in prompt and "u3:" in prompt and "u1:" not in prompt  # u1 is dropped


def test_support_lost_is_recorded_but_ok(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    client = make_client(tmp_path, FakeTransport(lambda body: _answer(u2=True, u3=False)))
    verdict = verify_rung(client, VerifyPolicy(), entry, _rung(entry, 2))
    assert verdict["ok"] and verdict["support_lost"] == ["u3"] and verdict["core_lost"] == []


def test_no_core_left_needs_no_call(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    transport = FakeTransport(lambda body: _answer())
    verdict = verify_rung(make_client(tmp_path, transport), VerifyPolicy(), entry, _rung(entry, 5))
    assert not verdict["ok"] and verdict["reasons"] == ["no_core_left"] and transport.calls == 0


def test_disabled_makes_no_calls(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    transport = FakeTransport(lambda body: _answer())
    verdicts = verify_rungs(make_client(tmp_path, transport), VerifyPolicy(enabled=False), [entry],
                            {("unit-0", 1), ("unit-0", 5)})
    assert transport.calls == 0 and all(v["ok"] and v["reasons"] == ["skipped"] for v in verdicts)
    assert [v["rung"] for v in verdicts] == [1, 5]


def test_incomplete_answer_is_not_ok(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    client = make_client(tmp_path, FakeTransport(lambda body: _answer(u3=True)))  # u2 (core) is missing
    verdict = verify_rung(client, VerifyPolicy(), entry, _rung(entry, 1))
    assert not verdict["ok"] and verdict["core_lost"] == ["u2"]
    assert "answer_incomplete" in verdict["reasons"]


def test_bad_response_fails_the_rung_without_raising(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    client = make_client(tmp_path, FakeTransport(lambda body: "not json at all"))
    verdict = verify_rung(client, VerifyPolicy(), entry, _rung(entry, 1))
    assert not verdict["ok"] and verdict["reasons"] == ["bad_response"]


def test_second_run_is_served_from_the_cache(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    measured = {("unit-0", 1), ("unit-0", 2), ("unit-0", 4)}
    transport = FakeTransport(lambda body: _answer(u2=True, u3=True))
    first = verify_rungs(make_client(tmp_path, transport), VerifyPolicy(workers=2), [entry], measured)
    assert transport.calls == 3 and [v["rung"] for v in first] == [1, 2, 4]
    second_transport = FakeTransport(lambda body: _answer())
    second = verify_rungs(make_client(tmp_path, second_transport), VerifyPolicy(), [entry], measured)
    assert second_transport.calls == 0 and all(v["llm_cached"] and v["ok"] for v in second)


def test_support_only_annotation_is_checked_as_core(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    for fragment in entry["annotation"]["units"]:
        fragment["role"] = "support" if fragment["role"] == "core" else fragment["role"]
    transport = FakeTransport(lambda body: _answer(u2=False, u3=True))
    verdict = verify_rung(make_client(tmp_path, transport), VerifyPolicy(), entry, _rung(entry, 1))
    assert transport.calls == 1 and not verdict["ok"] and verdict["core_lost"] == ["u2"]
    assert "(core)" in transport.bodies[0]["messages"][1]["content"]


def test_fragment_both_kept_and_dropped_counts_as_kept(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    rung = dict(_rung(entry, 5))
    rung["kept"] = list(rung["dropped"])  # the model contradicts itself about the core fragment
    transport = FakeTransport(lambda body: _answer(**{fid: True for fid in rung["kept"]}))
    verdict = verify_rung(make_client(tmp_path, transport), VerifyPolicy(), entry, rung)
    assert transport.calls == 1 and verdict["ok"]
