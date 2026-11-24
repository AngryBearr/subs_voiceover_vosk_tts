"""Shared core of the two LLM clients (``stt.llm`` and ``voiceover.llm``).

The clients keep their own retry loops and caches (``stt.llm`` keeps answers of an unfinished stage under
``<episode>/.partial/llm/<stage>/``; ``voiceover.llm`` keeps a permanent cache in ``output/llm_cache/``). What they
share lives here:

- API keys: ``parse_env_file`` / ``env_value`` read the environment and the repo ``.env`` (parsed as data);
- JSON answers: ``extract_json`` (a fenced block anywhere, then the first object or array) and
  ``strip_code_fence`` (an answer that is one fenced block);
- routing to a paid router: ``routed_model`` (the id the router knows), ``post_chat_completions`` (one HTTP POST),
  ``response_cost`` / ``record_answer_cost`` (the run's cost log);
- the subscription: ``subscription_chat`` asks OpenCode (with its free per-request stand-in model) and returns the
  answer in the OpenAI response shape, marked ``subscription_fallback`` when the stand-in answered.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

from utils.llm_route import generate_via_subscription, record_paid, router_model
from utils.opencode_clean import flatten_messages

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)
_FENCE_ANYWHERE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# ------------------------------------------------------------------ API keys


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse plain ``KEY=VALUE`` lines (``export`` allowed, quotes removed); comments and blank lines are ignored."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def env_value(name: str, env_file: Path | None = None) -> str:
    """``name`` from the environment, else from ``env_file`` (the repo ``.env`` by default); "" when unset."""
    value = os.environ.get(name, "").strip()
    if not value:
        value = parse_env_file(env_file if env_file is not None else DEFAULT_ENV_FILE).get(name, "").strip()
    return value


# ------------------------------------------------------------------ JSON answers


def strip_code_fence(text: str) -> str:
    """The inside of an answer that is one fenced block (```json ... ```), else the stripped text."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def extract_json(text: str) -> Any:
    """Parse the JSON of a free-form answer: a fenced block anywhere, from its first ``{`` or ``[``.

    Raises ``json.JSONDecodeError`` when there is no JSON.
    """
    text = text.strip()
    fenced = _FENCE_ANYWHERE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=0)
    return json.loads(text[start:])


def parse_json_object(text: str) -> dict[str, Any]:
    """The JSON object of an answer (an optional fence around it); ``ValueError`` otherwise."""
    value = json.loads(strip_code_fence(text))
    if not isinstance(value, dict):
        raise ValueError("JSON answer is not an object")
    return value


# ------------------------------------------------------------------ paid routers


def routed_model(bare: str, error: Callable[[str], Exception] = RuntimeError) -> str:
    """The id to send to a router for ``bare``; raises ``error`` (nothing is sent) when the router has no such id."""
    model = router_model(bare)
    if model is None:
        raise error(f"no OpenRouter id for {bare}")
    return model


def post_chat_completions(base_url: str, api_key: str, body: dict[str, Any], timeout_s: float,
                          title: str = "subs-voiceover-stt") -> dict[str, Any]:
    """One POST to ``<base_url>/chat/completions``; HTTP and network errors are raised as ``urllib`` raises them."""
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": title},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]


def response_cost(response: dict[str, Any]) -> float:
    """The price of one answer in USD as the router reported it (``usage.cost``); 0 when absent."""
    return float((response.get("usage") or {}).get("cost") or 0.0)


def record_answer_cost(model: str, transport: str, response: dict[str, Any]) -> None:
    """Append the price of a paid answer to the run's cost log (``llm_route.record_paid``)."""
    record_paid(model, transport, response_cost(response))


# ------------------------------------------------------------------ the subscription


def subscription_chat(get_server: Callable[[str], Any], messages: list[dict[str, str]], model: str,
                      variant: str | None, timeout_s: float, retry_s: float) -> dict[str, Any]:
    """Ask the subscription (OpenCode) and return the answer in the OpenAI response shape (no usage: it is free).

    A model with a free stand-in (``llm_route.subscription_fallback``) may hand this request to the weaker model;
    the answer then carries ``subscription_fallback: True`` and ``model`` names the model that really answered.
    OpenCode failures are raised as ``OpenCodeError`` (each client turns them into its own errors).
    """
    text, used = generate_via_subscription(get_server, flatten_messages(messages), model, variant, timeout_s, retry_s)
    return {"model": "opencode:" + used, "subscription_fallback": used != model,
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {}}


def answered_by_fallback(response: dict[str, Any]) -> bool:
    """True when the weaker stand-in model of the subscription answered instead of the requested one."""
    return bool(response.get("subscription_fallback"))
