"""Where an LLM request goes: transports named in front of the model id.

``opencode+orcarouter:openai/gpt-6.1-sol`` asks OpenCode (the ChatGPT subscription) first and
OrcaRouter when OpenCode fails; ``openrouter:google/gemini-4-argon`` goes to OpenRouter only. A
model id without a known transport prefix keeps the client's own default.
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
    "orcarouter": Router("https://api.orcarouter.ai/v1", "ORCAROUTER_API_KEY"),
}
TRANSPORTS = ("opencode", *ROUTERS)


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
