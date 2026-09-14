"""Small, auditable transports for the authorized SH08 matrix.

This module deliberately contains no matrix scheduling or inference policy.  It
only validates one request, performs at most one transport operation, and
returns a JSON-safe receipt.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.opencode_clean import CleanOpenCodeServer
from utils.sh08_matrix_models import MAX_TOKENS, MODEL_SPECS, OPENCODE_REQUIRED_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


_SECRET_PATTERNS = (
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"(?:api[_ -]?key|authorization)\s*[:=]\s*[^\s,;]+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
)


def _safe_error(value: Any, secrets: tuple[str, ...] = ()) -> str:
    message = value if isinstance(value, str) else "request failed"
    message = " ".join(message.split())
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted-key]")
    for pattern, replacement in ((pattern, "[redacted-credential]") for pattern in _SECRET_PATTERNS):
        message = pattern.sub(replacement, message)
    return message[:300] or "request failed"


def _safe_raw(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        # Raw provider data is evidence, so redaction must not normalize or
        # truncate it.  Error messages use _safe_error and its 300-char cap.
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[redacted-key]")
        for pattern, replacement in ((pattern, "[redacted-credential]") for pattern in _SECRET_PATTERNS):
            redacted = pattern.sub(replacement, redacted)
        return redacted
    if isinstance(value, dict):
        return {str(_safe_raw(str(key), secrets)): _safe_raw(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_raw(item, secrets) for item in value]
    return value


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0


def _valid_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _usage_dict(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")
    result: dict[str, Any] = {}
    for name in names:
        values = [record["usage"].get(name) for record in records if isinstance(record.get("usage"), dict)]
        values = [value for value in values if isinstance(value, int) and not isinstance(value, bool) and value >= 0]
        if values:
            result[name] = sum(values)
    return result


def _cost_from_usage(spec: dict[str, Any], usage: dict[str, Any]) -> Optional[float]:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not _finite_nonnegative(input_tokens) or not _finite_nonnegative(output_tokens):
        return None
    cost = (float(input_tokens) * float(spec["input_per_million"]) + float(output_tokens) * float(spec["output_per_million"])) / 1_000_000
    return cost if _finite_nonnegative(cost) else None


def _usage_error(spec: dict[str, Any], usage: dict[str, Any], *, require_tokens: bool = True) -> Optional[str]:
    """Validate billing usage without counting reasoning tokens twice."""
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    for name in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
        value = usage.get(name)
        if value is not None and not _valid_token(value):
            return "usage_tokens_invalid"
    if require_tokens and (not _valid_token(input_tokens) or not _valid_token(output_tokens)):
        return "usage_tokens_missing"
    reasoning = usage.get("reasoning_tokens")
    effective_input: Optional[int] = None
    completion: Optional[int] = None
    if spec.get("provider") == "opencode":
        if _valid_token(input_tokens) and _valid_token(output_tokens):
            effective_input = input_tokens + (usage.get("cache_read_tokens") or 0) + (usage.get("cache_write_tokens") or 0)
            completion = output_tokens + (reasoning or 0)
        if effective_input is not None and effective_input > spec["context_limit"]:
            return "usage_input_exceeds_context_limit"
        if completion is not None and completion > spec["max_tokens"]:
            return "usage_output_exceeds_max_tokens"
    else:
        if _valid_token(input_tokens) and input_tokens > spec["context_limit"]:
            return "usage_input_exceeds_context_limit"
        if _valid_token(output_tokens) and output_tokens > spec["max_tokens"]:
            return "usage_output_exceeds_max_tokens"
        if _valid_token(reasoning) and _valid_token(output_tokens) and reasoning > output_tokens:
            return "usage_reasoning_exceeds_output"
    total = usage.get("total_tokens")
    if total is not None and _valid_token(input_tokens) and _valid_token(output_tokens):
        expected_total = input_tokens + output_tokens
        if spec.get("provider") == "opencode" and effective_input is not None and completion is not None:
            expected_total = effective_input + completion
        if total != expected_total:
            return "usage_total_inconsistent"
    return None


def _returned_provider(data: dict[str, Any]) -> Any:
    provider = data.get("provider", data.get("provider_name"))
    if provider is None:
        return "unknown"
    if isinstance(provider, (str, int, float, bool)):
        return provider
    return copy.deepcopy(provider)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MatrixCleanOpenCodeServer(CleanOpenCodeServer):
    """OpenCode server with one neutral, tool-free subtitle step."""

    def _make_config(self) -> None:
        super()._make_config()
        assert self._temp_root is not None
        config_path = self._temp_root / "opencode" / "opencode.json"
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        config["permission"] = {"*": "deny"}
        config["tools"] = {}
        agents = config.setdefault("agent", {})
        agents.setdefault("subtitle", {})["steps"] = 1
        for name in ("title", "summary", "compaction"):
            agents.setdefault(name, {})["disable"] = True
        config["compaction"] = {"auto": False}
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)

    def _environment(self) -> dict[str, str]:
        env = super()._environment()
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(MAX_TOKENS)
        for key in list(env):
            if key in {"DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"} or "ORC" in key.upper() and "ROUTER" in key.upper():
                env.pop(key, None)
        return env


class MatrixBackend:
    """Dispatch one SH08 request and return a serializable accounting receipt."""

    def __init__(self, *, allow_muse_data: bool = False, openrouter_no_byok: bool = False,
                 timeout: float = 300.0) -> None:
        if not isinstance(allow_muse_data, bool) or not isinstance(openrouter_no_byok, bool):
            raise TypeError("transport flags must be bool")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.allow_muse_data = allow_muse_data
        self.openrouter_no_byok = openrouter_no_byok
        self.timeout = float(timeout)
        self._credentials: Optional[dict[str, str]] = None
        self._server: Optional[MatrixCleanOpenCodeServer] = None
        self._secrets: tuple[str, ...] = ()

    async def __aenter__(self) -> "MatrixBackend":
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def _load_credentials(self) -> dict[str, str]:
        if self._credentials is not None:
            return self._credentials
        try:
            from dotenv import dotenv_values
            file_values = dotenv_values(str(REPO_ROOT / ".env"))
        except (OSError, ImportError):
            file_values = {}
        values: dict[str, str] = {}
        for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
            value = os.environ.get(name, file_values.get(name))
            if isinstance(value, str) and value.strip():
                values[name] = value.strip()
        self._credentials = values
        self._secrets = tuple(values.values())
        return values

    @staticmethod
    def _check_opencode_version() -> None:
        try:
            result = subprocess.run(["opencode", "--version"], capture_output=True, text=True, timeout=10, check=False)
        except Exception as exc:
            raise RuntimeError("opencode_version_unavailable") from exc
        output = f"{result.stdout}\n{result.stderr}"
        version_pattern = rf"(?<![0-9.]){re.escape(OPENCODE_REQUIRED_VERSION)}(?![0-9.])"
        if result.returncode != 0 or re.search(version_pattern, output) is None:
            raise RuntimeError(f"opencode_version_required:{OPENCODE_REQUIRED_VERSION}")

    def preflight(self, model_key: Optional[str] = None) -> dict[str, Any]:
        """Run only local checks; this method never starts a server or makes a request."""
        if model_key is not None and model_key not in MODEL_SPECS:
            return {"ok": False, "error": "unknown_model", "missing_varnames": []}
        spec = MODEL_SPECS.get(model_key) if model_key is not None else None
        missing: list[str] = []
        if spec is not None and spec["provider"] in {"deepseek", "openrouter"}:
            credentials = self._load_credentials()
            key_name = "DEEPSEEK_API_KEY" if spec["provider"] == "deepseek" else "OPENROUTER_API_KEY"
            if key_name not in credentials:
                missing.append(key_name)
        if spec is not None and spec["provider"] == "opencode":
            try:
                self._check_opencode_version()
            except RuntimeError as exc:
                return {"ok": False, "error": _safe_error(str(exc)), "missing_varnames": []}
        return {"ok": not missing, "error": "missing_credentials" if missing else None, "missing_varnames": missing}

    def _receipt(self, key: str, *, started: float, raw: Optional[dict[str, Any]] = None,
                 calls: Optional[list[dict[str, Any]]] = None, usage: Optional[dict[str, Any]] = None,
                 cost: Optional[float] = None, cost_known: bool = False, cost_kind: Optional[str] = None,
                 error: Optional[str] = None, physical: Optional[int] = 0,
                 returned_model: Any = "unknown", returned_provider: Any = "unknown",
                 observed_assistant_steps: Optional[int] = None, text: Optional[str] = None) -> dict[str, Any]:
        spec = MODEL_SPECS.get(key, {})
        safe_raw = _safe_raw(raw, self._secrets) if raw is not None else {}
        safe_calls = _safe_raw(calls or [], self._secrets)
        return {
            "text": text, "raw": safe_raw, "calls": safe_calls, "usage": usage or {},
            "cost_usd": cost, "cost_known": cost_known, "cost_kind": cost_kind,
            "error": _safe_error(error, self._secrets) if error else None,
            "physical_calls_observed": physical, "model": spec.get("model", "unknown"),
            "provider": spec.get("provider", "unknown"), "returned_model": returned_model,
            "returned_provider": returned_provider, "observed_assistant_steps": observed_assistant_steps,
            "elapsed_sec": max(0.0, time.monotonic() - started),
        }

    async def __call__(self, model_key: str, system_prompt: str, user_prompt: str,
                       schema: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        if model_key not in MODEL_SPECS:
            return self._receipt(model_key, started=started, error="unknown_model", physical=0)
        spec = MODEL_SPECS[model_key]
        if not isinstance(system_prompt, str) or not system_prompt.strip() or not isinstance(user_prompt, str) or not user_prompt.strip() or not isinstance(schema, dict):
            return self._receipt(model_key, started=started, error="invalid_request", physical=0)
        if model_key == "muse" and not self.allow_muse_data:
            return self._receipt(model_key, started=started, error="muse_data_consent_required", physical=0)
        if spec["provider"] == "openrouter" and not self.openrouter_no_byok:
            return self._receipt(model_key, started=started, error="openrouter_no_byok_confirmation_required", physical=0)
        if spec["provider"] in {"deepseek", "openrouter"}:
            credentials = self._load_credentials()
            key_name = "DEEPSEEK_API_KEY" if spec["provider"] == "deepseek" else "OPENROUTER_API_KEY"
            if key_name not in credentials:
                return self._receipt(model_key, started=started, error=f"missing_credentials:{key_name}", physical=0)
        if spec["provider"] == "opencode":
            try:
                self._check_opencode_version()
                if self._server is None:
                    self._server = MatrixCleanOpenCodeServer(
                        enable_length_tool=False,
                        required_models={"openai/gpt-5.6-luna": {"medium"}, "openai/gpt-5.6-sol": {"medium"}},
                        timeout=self.timeout,
                    )
                    self._server.start()
                receipt = await self._call_opencode(spec, system_prompt, user_prompt, schema, started)
                return receipt
            except Exception as exc:
                return self._receipt(model_key, started=started, error=_safe_error(str(exc), self._secrets), physical=None)
        if spec["provider"] == "deepseek":
            return await self._call_deepseek(model_key, spec, system_prompt, user_prompt, schema, started)
        return await self._call_openrouter(model_key, spec, system_prompt, user_prompt, schema, started)

    async def _call_deepseek(self, key: str, spec: dict[str, Any], system: str, user: str,
                             schema: dict[str, Any], started: float) -> dict[str, Any]:
        from utils.deepseek_transport import request_json_detailed
        try:
            response = await request_json_detailed(
                api_key=self._load_credentials()["DEEPSEEK_API_KEY"], model=spec["model"],
                system_prompt=system, user_prompt=user, timeout=self.timeout,
                reasoning_effort="off", max_tokens=MAX_TOKENS, allow_length_tool=False,
            )
            calls = [asdict(record) for record in response.calls]
            usage = _usage_dict(calls)
            usage_errors = [
                _usage_error(spec, record["usage"])
                for record in calls
                if isinstance(record.get("usage"), dict)
            ]
            usage_missing = any(not isinstance(record.get("usage"), dict) for record in calls)
            usage_error = next((error for error in usage_errors if error is not None), None)
            billable_usage = bool(calls) and not usage_missing and all(
                _valid_token(record["usage"].get("input_tokens")) and
                _valid_token(record["usage"].get("output_tokens"))
                for record in calls
                if isinstance(record.get("usage"), dict)
            )
            valid_usage = billable_usage and usage_error is None
            cost = _cost_from_usage(spec, usage) if billable_usage else None
            actual_model = calls[0].get("model", "unknown") if calls else "unknown"
            actual_provider = calls[0].get("provider", "unknown") if calls else "unknown"
            model_mismatch = any(record.get("model") != spec["model"] for record in calls)
            provider_mismatch = any(record.get("provider") != spec["provider"] for record in calls)
            for record in calls:
                if record.get("model") != spec["model"]:
                    record["status"] = "mismatch"
                    record["error"] = "returned_model_mismatch"
            structural_error: Optional[str] = None
            if model_mismatch:
                structural_error = "returned_model_mismatch"
            elif provider_mismatch:
                structural_error = "returned_provider_mismatch"
            elif response.error:
                structural_error = response.error
            elif len(calls) != 1:
                structural_error = "deepseek_requires_one_physical_call"
            elif calls[0].get("status") != "success":
                structural_error = "deepseek_call_not_success"
            elif calls[0].get("tool_calls", 0) != 0:
                structural_error = "unexpected_tool_calls"
            elif not isinstance(response.text, str) or not response.text.strip():
                structural_error = "missing_response"
            elif not valid_usage:
                structural_error = usage_error or "usage_invalid"
            receipt = self._receipt(
                key, started=started,
                raw={"detailed_response_snapshot": {"text": response.text, "error": response.error, "calls": calls}},
                calls=calls, usage=usage, cost=cost,
                cost_known=cost is not None,
                cost_kind="conservative_token_upper" if cost is not None else None,
                error=structural_error, physical=len(calls), returned_model=actual_model,
                returned_provider=actual_provider, text=response.text,
            )
            return receipt
        except Exception as exc:
            return self._receipt(key, started=started, error=type(exc).__name__, physical=1)

    async def _call_opencode(self, spec: dict[str, Any], system: str, user: str,
                             schema: dict[str, Any], started: float) -> dict[str, Any]:
        from utils import opencode_transport
        session = await opencode_transport.create_session(self._server.base_url, title="subtitle-review", auth_header=self._server.auth_header)  # type: ignore[union-attr]
        response = None
        try:
            response = await opencode_transport.send_prompt_detailed(
                self._server.base_url, session, system, user, spec["model"], spec["variant"],  # type: ignore[union-attr]
                auth_header=self._server.auth_header, output_schema=schema, allow_length_tool=False,
                timeout=self.timeout, native_structured_output=False,
            )
            calls = [asdict(record) for record in response.calls]
            raw_snapshot = {"text": response.text, "error": response.error, "calls": calls}
        finally:
            try:
                await opencode_transport.delete_session(
                    self._server.base_url, session, auth_header=self._server.auth_header,  # type: ignore[union-attr]
                )
            except Exception:
                pass
        usage = _usage_dict(calls)
        key = "luna" if spec["model"].endswith("luna") else "sol"
        actual_model = calls[0].get("model", "unknown") if calls else "unknown"
        actual_provider = calls[0].get("provider", "unknown") if calls else "unknown"
        usage_error: Optional[str] = None
        if len(calls) == 1:
            usage_data = calls[0].get("usage")
            usage_error = _usage_error(spec, usage_data or {}) if isinstance(usage_data, dict) else "usage_invalid"
        error = response.error
        if error is None:
            if not calls:
                error = "missing_response"
            elif len(calls) != 1:
                error = "opencode_requires_one_assistant_step"
            elif actual_model not in {spec["model"], spec["model"].rsplit("/", 1)[-1]}:
                error = "returned_model_mismatch"
            elif calls[0].get("status") != "success":
                error = "opencode_call_not_success"
            elif calls[0].get("tool_calls", 0) != 0:
                error = "unexpected_tool_calls"
            elif not isinstance(response.text, str) or not response.text.strip():
                error = "missing_response"
            elif usage_error is not None:
                error = usage_error
        valid = error is None
        receipt = self._receipt(
            key, started=started,
            raw={"detailed_response_snapshot": raw_snapshot},
            calls=calls, usage=usage,
            cost=0.0 if valid else None, cost_known=valid,
            cost_kind="subscription_not_priced" if valid else None, error=error, physical=None,
            returned_model=actual_model, returned_provider=actual_provider,
            observed_assistant_steps=len(calls), text=response.text,
        )
        return receipt

    async def _post_openrouter(self, *, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, bytes]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_URL, headers=headers, json=body,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                return response.status, await response.read()

    async def _call_openrouter(self, key: str, spec: dict[str, Any], system: str, user: str,
                               schema: dict[str, Any], started: float) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": spec["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": MAX_TOKENS, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "shortening_result", "strict": True, "schema": copy.deepcopy(schema),
            }},
            "provider": {
                "require_parameters": True, "allow_fallbacks": False, "only": [spec["endpoint_tag"]],
                "max_price": {"prompt": spec["input_per_million"], "completion": spec["output_per_million"]},
            },
            "reasoning": {"effort": "low"},
        }
        headers = {"Authorization": f"Bearer {self._load_credentials()['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
        try:
            status, raw_bytes = await self._post_openrouter(headers=headers, body=body)
            try:
                raw_value = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                call = {"provider": "openrouter", "model": "unknown", "variant": spec["variant"], "started_at": _utc_now(),
                        "elapsed_sec": max(0.0, time.monotonic() - started), "usage": None, "status": "error",
                        "requested_model": spec["model"], "requested_variant": spec["variant"], "request_max_tokens": MAX_TOKENS}
                return self._receipt(key, started=started, calls=[call], error=f"http_{status}:malformed_provider_response", physical=1)
            if not isinstance(raw_value, dict):
                call = {"provider": "openrouter", "model": "unknown", "variant": spec["variant"],
                        "started_at": _utc_now(), "elapsed_sec": max(0.0, time.monotonic() - started),
                        "usage": None, "status": "error", "requested_model": spec["model"],
                        "requested_variant": spec["variant"], "request_max_tokens": MAX_TOKENS}
                return self._receipt(key, started=started, calls=[call], error="malformed_provider_response", physical=1)
            return self._parse_openrouter(key, spec, raw_value, _safe_raw(raw_value, self._secrets), started,
                                          status=status)
        except Exception as exc:
            return self._receipt(key, started=started, error=type(exc).__name__, physical=1)

    def _parse_openrouter(self, key: str, spec: dict[str, Any], data: dict[str, Any],
                          raw: dict[str, Any], started: float, call: Optional[dict[str, Any]] = None,
                          *, status: int = 200) -> dict[str, Any]:
        """Parse billing metadata before applying any response validity checks."""
        actual_model = data.get("model") if isinstance(data.get("model"), str) else "unknown"
        returned_provider = _returned_provider(data)
        usage_data = data.get("usage")
        usage: dict[str, Any] = {}
        usage_invalid = not isinstance(usage_data, dict)
        if isinstance(usage_data, dict):
            for source, target in (
                ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"), ("cache_read_tokens", "cache_read_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
            ):
                value = usage_data.get(source)
                if value is not None and not _valid_token(value):
                    usage_invalid = True
                elif _valid_token(value):
                    usage[target] = value
            prompt_details = usage_data.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens")
                if cached is not None and not _valid_token(cached):
                    usage_invalid = True
                elif _valid_token(cached):
                    usage["cache_read_tokens"] = cached
            elif prompt_details is not None:
                usage_invalid = True
            details = usage_data.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning = details.get("reasoning_tokens")
                if reasoning is not None and not _valid_token(reasoning):
                    usage_invalid = True
                elif _valid_token(reasoning):
                    usage["reasoning_tokens"] = reasoning
            elif details is not None:
                usage_invalid = True
            raw_cost = usage_data.get("cost", data.get("cost"))
        else:
            raw_cost = data.get("cost")
        if raw_cost is not None and not _finite_nonnegative(raw_cost):
            usage_invalid = True
        elif _finite_nonnegative(raw_cost):
            usage["cost"] = float(raw_cost)
        cost = usage.get("cost")
        byok = data.get("is_byok") is True or isinstance(usage_data, dict) and usage_data.get("is_byok") is True
        usage_error = _usage_error(spec, usage)
        if isinstance(usage_data, dict) and usage_error is not None:
            usage_invalid = True
        known = _finite_nonnegative(cost) and not byok

        call = call or {
            "provider": "openrouter", "model": actual_model, "variant": spec["variant"],
            "started_at": _utc_now(), "elapsed_sec": max(0.0, time.monotonic() - started),
            "requested_model": spec["model"], "requested_variant": spec["variant"],
            "request_max_tokens": MAX_TOKENS,
        }
        call["model"] = actual_model
        call["returned_provider"] = returned_provider
        call["usage"] = usage or None
        call["status"] = "error" if not 200 <= status < 300 else "success"
        if usage_invalid:
            call["status"] = "error" if not 200 <= status < 300 else "malformed"
            call["error"] = usage_error or "invalid_usage"

        expected = spec["model"]
        dated = f"{expected}-20260902"
        error: Optional[str] = None
        if not 200 <= status < 300:
            error_data = data.get("error")
            message = error_data.get("message") if isinstance(error_data, dict) else "provider request failed"
            error = f"http_{status}:{_safe_error(message, self._secrets)}"
            call["error"] = error
        elif actual_model not in {expected, dated}:
            call["status"] = "mismatch"
            call["error"] = "returned_model_mismatch"
            error = "returned_model_mismatch"
        else:
            provider = data.get("provider")
            provider_name = data.get("provider_name")
            if provider is not None or provider_name is not None:
                provider_text = json.dumps(provider if provider is not None else provider_name, sort_keys=True).lower()
                endpoint = re.sub(r"[^a-z0-9]", "", str(spec["endpoint_tag"]).split("/", 1)[0].lower())
                normalized_provider = re.sub(r"[^a-z0-9]", "", provider_text)
                if endpoint not in normalized_provider:
                    call["status"] = "mismatch"
                    call["error"] = "returned_provider_mismatch"
                    error = "returned_provider_mismatch"
            service_tier = data.get("service_tier")
            if error is None and service_tier is not None and (not isinstance(service_tier, str) or (key == "gemini" and service_tier != "flex")):
                call["status"] = "mismatch"
                call["error"] = "returned_service_tier_mismatch"
                error = "returned_service_tier_mismatch"

        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        tools = message.get("tool_calls") if isinstance(message, dict) else None
        function_call = message.get("function_call") if isinstance(message, dict) else None
        if error is None and (not isinstance(choices, list) or len(choices) != 1 or choice is None):
            call["status"] = "malformed"
            call["error"] = "invalid_choices"
            error = "invalid_choices"
        elif error is None and isinstance(message, dict) and (
            (tools is not None and not isinstance(tools, list)) or tools or function_call is not None
        ):
            call["status"] = "malformed"
            call["error"] = "unexpected_tool_calls"
            error = "unexpected_tool_calls"
        elif error is None and (not isinstance(choice, dict) or choice.get("finish_reason") != "stop" or
                                not isinstance(text, str) or not text.strip() or
                                not isinstance(message, dict) or message.get("refusal") is not None):
            call["status"] = "malformed"
            call["error"] = "invalid_assistant_response"
            error = "invalid_assistant_response"
        if error is None and usage_invalid:
            call["status"] = "malformed"
            call["error"] = usage_error or "invalid_usage"
            error = usage_error or "invalid_usage"
        elif error is None and not known:
            call["status"] = "unknown_usage"
            call["error"] = "openrouter_byok_attribution_unknown" if byok else "usage_cost_unknown"
            error = call["error"]
        receipt = self._receipt(
            key, started=started, raw=raw, usage=usage, cost=float(cost) if known else None,
            calls=[call], cost_known=known, cost_kind="provider_reported" if known else None,
            error=error, physical=1, returned_model=actual_model, returned_provider=returned_provider,
            text=text if isinstance(text, str) else None,
        )
        return receipt
