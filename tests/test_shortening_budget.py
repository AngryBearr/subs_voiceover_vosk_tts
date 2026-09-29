from __future__ import annotations

from decimal import Decimal

import pytest

from utils.llm_types import CallRecord
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_budget import BudgetPolicy, BudgetStop
from utils.shortening_resume import canonical_hash
from utils.shortening_runner import ModelSpec


def _snapshot() -> dict[str, object]:
    models = {
        "meta/muse-spark-1.3-contributor": {
            "model": "meta/muse-spark-1.3-contributor", "canonical_model": "meta/muse-spark-1.3-contributor",
            "effort": "low", "provider_slug": "meta", "provider_name": "Meta",
            "context_length": 1_016_384, "max_input_tokens": 1_000_000, "max_output_tokens": 16_384,
            "max_input_rate": "0.0000001", "max_output_rate": "0.0000002",
            "request_fee": "0", "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": {}, "endpoint_metadata": {},
        },
        "openai/gpt-5.6-sol": {
            "model": "openai/gpt-5.6-sol", "canonical_model": "openai/gpt-5.6-sol",
            "effort": "medium", "provider_slug": "openai", "provider_name": "OpenAI",
            "context_length": 1_016_384, "max_input_tokens": 1_000_000, "max_output_tokens": 16_384,
            "max_input_rate": "0.00000461", "max_output_rate": "0.000015",
            "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": {}, "endpoint_metadata": {},
        },
    }
    snapshot: dict[str, object] = {"schema": "openrouter_production.v1", "id": "", "captured_at": "2026-09-20T00:00:00Z", "models": models}
    snapshot["id"] = canonical_hash({key: value for key, value in snapshot.items() if key != "id"})
    return snapshot


def _policy(limit: str | Decimal = "10", *, max_requests: int = 1000) -> BudgetPolicy:
    return BudgetPolicy.from_snapshot(limit, _snapshot(), max_requests=max_requests)


def _call(model: str, variant: str, cost: float | None, *, provider: str = "openrouter") -> CallRecord:
    usage = OpenCodePromptUsage(input_tokens=12, output_tokens=3, reasoning_tokens=2, cost=cost)
    return CallRecord(provider, model, variant, "2026-09-20T00:00:00Z", 0.1, usage, "success",
                      requested_model=model, requested_variant=variant)


def test_reservations_use_published_ceiling_and_show_zero_fee_assumption() -> None:
    policy = _policy()
    reservation = policy.reservation(ModelSpec("openrouter", "openai/gpt-5.6-sol", "medium"))
    assert reservation["usd"] == "4.85576"
    assert reservation["model"] == {"provider": "openrouter", "model": "openai/gpt-5.6-sol", "variant": "medium"}
    assert policy.summary([], []) ["request_fee_assumption"] == "absent_request_fee_assumed_zero"


def test_decimal_boundary_and_next_reservation_are_protected() -> None:
    policy = _policy("4.85576")
    model = ModelSpec("openrouter", "openai/gpt-5.6-sol", "medium")
    assert policy.admit(model, [], []) ["usd"] == "4.85576"
    with pytest.raises(BudgetStop) as stopped:
        policy.admit(model, [_call("openai/gpt-5.6-sol", "medium", 0.00001)], [])
    assert stopped.value.reason == "blocked_by_budget"
    assert stopped.value.summary["next_reservation_usd"] == "4.85576"


def test_unknown_cost_and_request_limit_stop_before_admission() -> None:
    policy = _policy(max_requests=1)
    model = ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")
    with pytest.raises(BudgetStop) as stopped:
        policy.admit(model, [_call("meta/muse-spark-1.3-contributor", "low", None)], [])
    assert stopped.value.reason == "blocked_unknown_cost"
    assert Decimal(stopped.value.summary["unknown_reserved_usd"]) > 0
    with pytest.raises(BudgetStop) as request_stopped:
        policy.admit(model, [_call("meta/muse-spark-1.3-contributor", "low", 0)], [])
    assert request_stopped.value.reason == "blocked_request_limit"


def test_known_failure_does_not_make_known_call_unknown_and_old_replay_is_not_added() -> None:
    policy = _policy()
    call = _call("openai/gpt-5.6-sol", "medium", 1.25)
    failure = {"request_id": "r1", "model": {"provider": "openrouter", "model": "openai/gpt-5.6-sol", "variant": "medium"}, "physical_call_count": 1, "accounting_incomplete": False}
    summary = policy.summary([call], [failure])
    assert summary["known_spent_usd"] == "1.25"
    assert summary["unknown_reserved_usd"] == "0"
    assert summary["physical_calls"] == 1


def test_canonical_actual_model_is_admitted_for_requested_role_but_unrelated_is_rejected() -> None:
    snapshot = _snapshot()
    models = snapshot["models"]
    assert isinstance(models, dict)
    entry = models["openai/gpt-5.6-sol"]
    assert isinstance(entry, dict)
    entry["canonical_model"] = "openai/gpt-5.6-sol-2026"
    snapshot["id"] = canonical_hash({key: value for key, value in snapshot.items() if key != "id"})
    policy = BudgetPolicy.from_snapshot("10", snapshot)

    admitted = _call("openai/gpt-5.6-sol-2026", "medium", 1.25)
    admitted.requested_model = "openai/gpt-5.6-sol"
    admitted.requested_variant = "medium"
    assert policy.admit(ModelSpec("openrouter", "openai/gpt-5.6-sol", "medium"), [admitted], [])

    wrong = _call("unrelated/model", "medium", 1.25)
    wrong.requested_model = "openai/gpt-5.6-sol"
    wrong.requested_variant = "medium"
    assert policy.summary([wrong], []) ["status"] == "blocked_unknown_cost"

    mismatched_request = _call("openai/gpt-5.6-sol-2026", "medium", 1.25)
    mismatched_request.requested_model = "meta/muse-spark-1.3-contributor"
    mismatched_request.requested_variant = "low"
    assert policy.summary([mismatched_request], []) ["status"] == "blocked_unknown_cost"


def test_failure_claims_add_only_physical_calls_without_saved_receipts() -> None:
    policy = _policy()
    call = _call("openai/gpt-5.6-sol", "medium", 1.25)
    failure = {
        "request_id": "partial", "model": {"provider": "openrouter", "model": "openai/gpt-5.6-sol", "variant": "medium"},
        "physical_call_count": 3, "accounting_incomplete": False,
    }
    assert policy.summary([call], [failure])["physical_calls"] == 3


def test_foreign_routes_and_invalid_snapshot_fail_closed() -> None:
    policy = _policy()
    foreign = _call("openai/gpt-5.6-sol", "medium", 0, provider="openai")
    assert policy.summary([foreign], []) ["status"] == "blocked_unknown_cost"
    invalid = _snapshot()
    invalid["models"] = dict(invalid["models"])  # type: ignore[index]
    invalid["models"]["openai/gpt-5.6-sol"] = dict(invalid["models"]["openai/gpt-5.6-sol"])  # type: ignore[index]
    invalid["models"]["openai/gpt-5.6-sol"]["max_output_rate"] = "NaN"  # type: ignore[index]
    invalid["id"] = canonical_hash({key: value for key, value in invalid.items() if key != "id"})
    with pytest.raises(ValueError):
        BudgetPolicy.from_snapshot("10", invalid)
