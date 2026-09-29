"""Pure, fail-closed reservations for the two OpenRouter shortening models."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from utils.llm_types import CallRecord
from utils.shortening_resume import canonical_hash

if TYPE_CHECKING:
    from utils.shortening_runner import ModelSpec


_SCHEMA = "openrouter_production.v1"
_MAX_OUTPUT_TOKENS = 16_384
_RESERVATION_BASIS = "published_context_and_max_rates"
_COST_BASIS = "openrouter_reported_usd"
_MIN_UNKNOWN_RESERVATION = Decimal("0.000000000001")
_ROLES: dict[tuple[str, str], tuple[str, str]] = {
    ("meta/muse-spark-1.3-contributor", "low"): ("meta", "Meta"),
    ("openai/gpt-5.6-sol", "medium"): ("openai", "OpenAI"),
}
_NEW_ROLES = {("meta/muse-spark-1.3-contributor", "low"): ("meta", "Meta"),
              ("openai/gpt-6-sol", "high"): ("openai", "OpenAI")}


class BudgetStop(RuntimeError):
    """Admission stopped before starting another physical request."""

    def __init__(self, reason: str, summary: dict[str, Any]) -> None:
        self.reason = reason
        self.summary = summary
        super().__init__(reason)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _decimal(value: Any, name: str, *, nonnegative: bool = True) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite decimal") from None
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _roles(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    models = snapshot.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("snapshot models must be a mapping")
    if set(models) == {model for model, _ in _ROLES}:
        return _ROLES
    if set(models) == {model for model, _ in _NEW_ROLES}:
        return _NEW_ROLES
    raise ValueError("snapshot must contain the exact active model roles")


def _snapshot_model(snapshot: Mapping[str, Any], model: str, variant: str) -> Mapping[str, Any]:
    models = snapshot.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("snapshot models must be a mapping")
    roles = _roles(snapshot)
    if (model, variant) not in roles:
        raise ValueError("snapshot role mismatch")
    matches: list[Mapping[str, Any]] = []
    for key, entry in models.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ValueError("snapshot model entries must be mappings")
        if key == model or entry.get("model") == model:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f"snapshot missing exact role {model}:{variant}")
    entry = matches[0]
    if entry.get("model") != model or entry.get("effort") != variant:
        raise ValueError(f"snapshot role mismatch for {model}:{variant}")
    provider_slug, provider_name = roles[(model, variant)]
    if entry.get("provider_slug") != provider_slug:
        raise ValueError(f"snapshot provider mismatch for {model}:{variant}")
    if entry.get("provider_name") != provider_name:
        raise ValueError(f"snapshot provider name mismatch for {model}:{variant}")
    required = {
        "model", "canonical_model", "effort", "provider_slug", "provider_name",
        "context_length", "max_input_tokens", "max_output_tokens", "max_input_rate",
        "max_output_rate", "model_metadata", "endpoint_metadata",
    }
    if not required.issubset(entry):
        raise ValueError(f"snapshot role missing fields for {model}:{variant}")
    _positive_int(entry["context_length"], "context_length")
    max_input = _positive_int(entry["max_input_tokens"], "max_input_tokens")
    if entry["max_output_tokens"] != _MAX_OUTPUT_TOKENS:
        raise ValueError("max_output_tokens must be 16384")
    if max_input > entry["context_length"] - _MAX_OUTPUT_TOKENS:
        raise ValueError("max_input_tokens exceeds context/output bound")
    for rate_name in ("max_input_rate", "max_output_rate"):
        if not isinstance(entry[rate_name], str):
            raise ValueError(f"{rate_name} must be a decimal string")
        _decimal(entry[rate_name], rate_name)
    if not isinstance(entry["model_metadata"], Mapping) or not isinstance(entry["endpoint_metadata"], Mapping):
        raise ValueError("snapshot metadata must be mappings")
    if not isinstance(entry["canonical_model"], str) or not entry["canonical_model"].strip():
        raise ValueError("snapshot canonical_model is required")
    request_fee = entry.get("request_fee")
    if request_fee is not None:
        if not isinstance(request_fee, str):
            raise ValueError("request_fee must be a decimal string")
        if _decimal(request_fee, "request_fee") > 0:
            raise ValueError("positive request_fee is not supported by route policy")
    request_fee_source = entry.get("request_fee_source")
    if request_fee_source is not None and not isinstance(request_fee_source, str):
        raise ValueError("request_fee_source must be a string")
    return entry


def _validate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != _SCHEMA:
        raise ValueError("invalid OpenRouter snapshot schema")
    snapshot_id = snapshot.get("id")
    if not isinstance(snapshot_id, str) or len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
        raise ValueError("invalid snapshot id")
    if canonical_hash({key: value for key, value in snapshot.items() if key != "id"}) != snapshot_id:
        raise ValueError("snapshot id does not match canonical hash")
    if not isinstance(snapshot.get("captured_at"), str) or not snapshot["captured_at"].strip():
        raise ValueError("snapshot captured_at is required")
    for model, variant in _roles(snapshot):
        _snapshot_model(snapshot, model, variant)
    return snapshot


def _role_for_model(model: Any, variant: Any, roles: Mapping[tuple[str, str], Any]) -> tuple[str, str] | None:
    if not isinstance(model, str) or not isinstance(variant, str):
        return None
    return (model, variant) if (model, variant) in roles else None


def _record_role(record: Any, roles: Mapping[tuple[str, str], Any]) -> tuple[str, str] | None:
    if _value(record, "provider") == "openrouter":
        return _role_for_model(_value(record, "requested_model"), _value(record, "requested_variant"), roles)
    return _role_for_model(_value(record, "model"), _value(record, "variant"), roles)


def _failure_role(failure: Mapping[str, Any], roles: Mapping[tuple[str, str], Any]) -> tuple[str, str] | None:
    model = failure.get("model")
    if isinstance(model, Mapping):
        return _role_for_model(model.get("model"), model.get("variant"), roles)
    return _role_for_model(failure.get("model"), failure.get("variant"), roles)


def _reported_cost(record: Any) -> Decimal | None:
    usage = _value(record, "usage")
    cost = _value(usage, "cost") if usage is not None else None
    if cost is None or isinstance(cost, bool) or not isinstance(cost, (str, int, float, Decimal)):
        return None
    try:
        parsed = Decimal(str(cost))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _reservation_for(entry: Mapping[str, Any], model: str, variant: str, snapshot_id: str) -> dict[str, Any]:
    input_tokens = int(entry["max_input_tokens"])
    output_tokens = _MAX_OUTPUT_TOKENS
    input_rate = _decimal(entry["max_input_rate"], "max_input_rate")
    output_rate = _decimal(entry["max_output_rate"], "max_output_rate")
    request_fee = _decimal(entry.get("request_fee", "0"), "request_fee") if entry.get("request_fee") is not None else Decimal(0)
    amount = Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate + request_fee
    return {
        "usd": _decimal_string(amount),
        "pricing_snapshot_id": snapshot_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": {"provider": "openrouter", "model": model, "variant": variant},
    }


class BudgetPolicy:
    """A stateless budget policy; the runner remains the sole journal owner."""

    def __init__(self, limit_usd: Decimal, snapshot: dict[str, Any], max_requests: int, max_unknown_calls: int = 0) -> None:
        self.limit_usd = limit_usd
        self.snapshot = snapshot
        self.max_requests = max_requests
        self.max_unknown_calls = max_unknown_calls

    @classmethod
    def from_snapshot(cls, limit_usd: str | Decimal, snapshot: dict[str, Any], *, max_requests: int = 1000,
                      max_unknown_calls: int = 0) -> "BudgetPolicy":
        limit = _decimal(limit_usd, "limit_usd")
        if limit <= 0:
            raise ValueError("limit_usd must be positive")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        validated = _validate_snapshot(snapshot)
        if type(max_unknown_calls) is not int or not 0 <= max_unknown_calls <= max_requests:
            raise ValueError("max_unknown_calls must be between 0 and max_requests")
        if max_unknown_calls and _roles(validated) is _ROLES:
            raise ValueError("legacy snapshots require strict unknown-cost policy")
        return cls(limit, validated, max_requests, max_unknown_calls)

    def reservation(self, model: ModelSpec) -> dict[str, Any]:
        provider = _value(model, "provider")
        model_name = _value(model, "model")
        variant = _value(model, "variant")
        role = _role_for_model(model_name, variant, _roles(self.snapshot))
        if provider != "openrouter" or role is None:
            raise ValueError("model is not an active OpenRouter budget role")
        entry = _snapshot_model(self.snapshot, model_name, variant)
        return _reservation_for(entry, model_name, variant, self.snapshot["id"])

    def _analysis(self, calls: Sequence[CallRecord], failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
        known = Decimal(0)
        unknown_by_role: dict[tuple[str, str], int] = {}
        receipt_counts: dict[tuple[str, str], int] = {}
        unknown_count = 0
        invalid_count = 0
        roles = _roles(self.snapshot)
        unknown_reserved = Decimal(0)
        receipts_by_request: dict[str, int] = {}
        for call in calls:
            role = _record_role(call, roles)
            valid_actual = False
            if role is not None:
                entry = _snapshot_model(self.snapshot, role[0], role[1])
                valid_actual = _value(call, "model") in {entry["model"], entry["canonical_model"]}
            if _value(call, "provider") != "openrouter" or role is None or not valid_actual:
                invalid_count += 1
                continue
            no_receipt = (_value(call, "status") in {"timeout", "error", "malformed"}
                          and not _value(call, "generation_id") and _value(call, "provider_name") is None
                          and _value(call, "is_byok") is None)
            if self.max_unknown_calls and not no_receipt and (
                _value(call, "provider_name") != entry["provider_name"]
                or _value(call, "is_byok") is not False or not _value(call, "generation_id")
            ):
                invalid_count += 1
                continue
            receipt_counts[role] = receipt_counts.get(role, 0) + 1
            receipt_request_id = _value(call, "request_id")
            if isinstance(receipt_request_id, str) and receipt_request_id:
                receipts_by_request[receipt_request_id] = receipts_by_request.get(receipt_request_id, 0) + 1
            raw_cost = _value(_value(call, "usage"), "cost")
            if raw_cost is not None and _reported_cost(call) is None:
                invalid_count += 1
                continue
            cost = _reported_cost(call)
            if cost is None:
                unknown_by_role[role] = unknown_by_role.get(role, 0) + 1
                if self.max_unknown_calls:
                    entry = _snapshot_model(self.snapshot, *role)
                    usage = _value(call, "usage")
                    input_tokens = _value(usage, "input_tokens")
                    output_tokens = _value(usage, "output_tokens")
                    if (type(input_tokens) is int and type(output_tokens) is int
                            and 0 <= input_tokens <= entry["max_input_tokens"]
                            and 0 <= output_tokens <= _MAX_OUTPUT_TOKENS):
                        unknown_reserved += max(
                            input_tokens * _decimal(entry["max_input_rate"], "max_input_rate")
                            + output_tokens * _decimal(entry["max_output_rate"], "max_output_rate"),
                            _MIN_UNKNOWN_RESERVATION,
                        )
                    else:
                        unknown_reserved += Decimal(_reservation_for(entry, *role, self.snapshot["id"])["usd"])
            else:
                known += cost

        covered_unknown: dict[tuple[str, str], int] = {}
        covered_complete: dict[tuple[str, str], int] = {}
        failure_claims: dict[tuple[str, str], int] = {}
        unmatched_failures: dict[tuple[str, str], int] = {}
        seen_failure_ids: set[str] = set()
        physical_calls = len(calls)
        for failure in failures:
            if not isinstance(failure, dict):
                invalid_count += 1
                continue
            request_id = failure.get("request_id")
            if isinstance(request_id, str) and request_id:
                if request_id in seen_failure_ids:
                    continue
                seen_failure_ids.add(request_id)
            role = _failure_role(failure, roles)
            physical_count = failure.get("physical_call_count", 0)
            if isinstance(physical_count, bool) or not isinstance(physical_count, int) or physical_count < 0:
                invalid_count += 1
                physical_count = 0
            if role is None or _failure_provider(failure) != "openrouter":
                invalid_count += 1
                physical_calls += max(physical_count, 1)
                continue
            failure_claims[role] = failure_claims.get(role, 0) + max(physical_count, 1)
            if self.max_unknown_calls:
                matched = receipts_by_request.get(request_id, 0) if isinstance(request_id, str) else 0
                missing = max(0, max(physical_count, 1) - matched)
                unmatched_failures[role] = unmatched_failures.get(role, 0) + missing
                if missing and failure.get("accounting_incomplete", True) is True:
                    unknown_count += missing
                    entry = _snapshot_model(self.snapshot, *role)
                    unit = _reservation_for(entry, *role, self.snapshot["id"])
                    unknown_reserved += max(Decimal(unit["usd"]), _MIN_UNKNOWN_RESERVATION) * missing
            if failure.get("accounting_incomplete", True) is True:
                covered_unknown[role] = covered_unknown.get(role, 0) + physical_count
                if not self.max_unknown_calls:
                    unknown_count += max(physical_count, 1)
                    entry = _snapshot_model(self.snapshot, *role)
                    unit = _reservation_for(entry, *role, self.snapshot["id"])
                    unknown_reserved += max(Decimal(unit["usd"]), _MIN_UNKNOWN_RESERVATION) * max(physical_count, 1)
            else:
                covered_complete[role] = covered_complete.get(role, 0) + physical_count

        physical_calls += (sum(unmatched_failures.values()) if self.max_unknown_calls else sum(
            max(0, claimed - receipt_counts.get(role, 0)) for role, claimed in failure_claims.items()
        ))

        if self.max_unknown_calls:
            unknown_count += sum(unknown_by_role.values())
        else:
            for role, count in unknown_by_role.items():
                uncovered = max(0, count - covered_unknown.get(role, 0) - covered_complete.get(role, 0))
                unknown_count += uncovered
                entry = _snapshot_model(self.snapshot, *role)
                unit = _reservation_for(entry, *role, self.snapshot["id"])
                unknown_reserved += max(Decimal(unit["usd"]), _MIN_UNKNOWN_RESERVATION) * uncovered

        if invalid_count:
            # A foreign route cannot be treated as free; reserve the larger known
            # role rather than making an unsupported provider balance claim.
            role_reservations = [
                Decimal(_reservation_for(
                    _snapshot_model(self.snapshot, model, variant), model, variant, self.snapshot["id"],
                )["usd"])
                for model, variant in roles
            ]
            unknown_reserved += max(max(role_reservations, default=Decimal(0)), _MIN_UNKNOWN_RESERVATION) * invalid_count
        return {
            "known_spent": known,
            "unknown_reserved": unknown_reserved,
            "physical_calls": physical_calls,
            "invalid_count": invalid_count,
            "unknown_count": unknown_count,
        }

    def summary(self, calls: Sequence[CallRecord], failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
        analysis = self._analysis(calls, failures)
        known = analysis["known_spent"]
        unknown = analysis["unknown_reserved"]
        if analysis["invalid_count"] or (analysis["unknown_count"] and not self.max_unknown_calls):
            status = "blocked_unknown_cost"
        elif analysis["unknown_count"] >= self.max_unknown_calls and self.max_unknown_calls:
            status = "blocked_unknown_limit"
        elif analysis["physical_calls"] >= self.max_requests:
            status = "blocked_request_limit"
        elif known >= self.limit_usd:
            status = "blocked_by_budget"
        else:
            status = "ok"
        return {
            "limit_usd": _decimal_string(self.limit_usd),
            "known_spent_usd": _decimal_string(known),
            "unknown_reserved_usd": _decimal_string(unknown),
            "remaining_usd": _decimal_string(max(Decimal(0), self.limit_usd - known - unknown)),
            "physical_calls": analysis["physical_calls"],
            "max_requests": self.max_requests,
            "unknown_call_count": analysis["unknown_count"],
            "max_unknown_calls": self.max_unknown_calls,
            "status": status,
            "pricing_snapshot_id": self.snapshot["id"],
            "reservation_basis": _RESERVATION_BASIS,
            "cost_basis": _COST_BASIS,
            "request_fee_assumption": "absent_request_fee_assumed_zero",
        }

    def admit(self, model: ModelSpec, calls: Sequence[CallRecord], failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
        summary = self.summary(calls, failures)
        try:
            next_reservation = self.reservation(model)
        except ValueError as exc:
            summary.update({"blocked_reason": "invalid_policy_model", "error": str(exc)})
            raise BudgetStop("blocked_unknown_cost", summary) from exc
        summary["next_reservation_usd"] = next_reservation["usd"]
        if summary["status"] in {"blocked_unknown_cost", "blocked_unknown_limit"}:
            summary["blocked_reason"] = "unknown_or_foreign_cost"
            raise BudgetStop(summary["status"], summary)
        if summary["physical_calls"] >= self.max_requests:
            summary["blocked_reason"] = "max_requests"
            raise BudgetStop("blocked_request_limit", summary)
        projected = Decimal(summary["known_spent_usd"]) + Decimal(summary["unknown_reserved_usd"]) + Decimal(next_reservation["usd"])
        if projected > self.limit_usd:
            summary["blocked_reason"] = "reservation_exceeds_limit"
            raise BudgetStop("blocked_by_budget", summary)
        return next_reservation


def _failure_provider(failure: Mapping[str, Any]) -> Any:
    model = failure.get("model")
    return model.get("provider") if isinstance(model, Mapping) else failure.get("provider")
