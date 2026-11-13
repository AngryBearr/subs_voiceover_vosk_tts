"""Where an LLM request goes: transports named in front of the model id.

``opencode+openrouter:openai/gpt-6.1-sol`` asks OpenCode (the ChatGPT subscription) first and
OpenRouter when OpenCode fails; ``openrouter:google/gemini-4-argon`` goes to OpenRouter only. A
model id without a known transport prefix keeps the client's own default.

OpenCode Go models (``opencode-go/<name>``) have other ids on OpenRouter: ``router_model`` maps them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Router:
    """An OpenAI-compatible paid router."""

    base_url: str
    api_key_env: str


ROUTERS: dict[str, Router] = {
    "openrouter": Router("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}
TRANSPORTS = ("opencode", *ROUTERS)

OPENCODE_GO_PREFIX = "opencode-go/"
# OpenCode Go model name -> the same model's OpenRouter id (only verified pairs).
OPENCODE_GO_MODELS: dict[str, str] = {
    "gemini-4-argon": "google/gemini-4-argon",
    "claude-haiku-5-5": "anthropic/claude-haiku-5.5",
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "gpt-6-luna": "openai/gpt-6-luna",
    "minimax-m3": "minimax/minimax-m3",
    "mimo-v2.6-pro": "xiaomi/mimo-v2.6-pro",
    "step-5-preview": "stepfun/step-5-preview",
    "hy4": "tencent/hy4",
    "muse-spark-1.3-contributor": "meta/muse-spark-1.3-contributor",
    "glm-5.3": "z-ai/glm-5.3",
    "kimi-k3": "moonshotai/kimi-k3",
    "grok-4.7": "x-ai/grok-4.7",
}


def router_model(bare: str) -> str | None:
    """The id to send to a router for a bare model id; None for an OpenCode Go model without an OpenRouter id."""
    if not bare.startswith(OPENCODE_GO_PREFIX):
        return bare
    return OPENCODE_GO_MODELS.get(bare[len(OPENCODE_GO_PREFIX):])


def split_route(model: str) -> tuple[list[str], str]:
    """(transports in order, bare model id); no prefix -> ([], model)."""
    head, sep, rest = model.partition(":")
    names = head.split("+") if sep else []
    if names and rest and all(name in TRANSPORTS for name in names):
        return names, rest
    return [], model


def with_route(model: str, transports: list[str]) -> str:
    """The model id with these transports in front (an existing prefix is kept)."""
    if split_route(model)[0] or not transports:
        return model
    return "+".join(transports) + ":" + model
