"""LLM settings of an ``stt.run`` run: the route of every model id (subscription, paid router), the steps and models
of the run, whether it may pay, and the check at the start that stops a run which would pay without permission.

Split out of ``stt/run.py``; ``stt.run`` re-exports every name (``Episode.route_in`` is ``route_in``), and its log
lines keep the ``stt.run`` logger. The check that asks the subscription (``check_subscription``) stays in ``stt.run``.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from typing import Any

from utils.llm_route import (FALLBACK_AFTER_ENV, FALLBACK_AFTER_S, FALLBACK_ENV, PAID_ENV, ROUTERS, PaidNotAllowed,
                             split_route, with_route)

log = logging.getLogger("stt.run")
_WARNED_ROUTES: set[str] = set()  # unsupported router names already warned about


def known_router(name: str) -> bool:
    """False (with one warning per name) for a router that no longer exists, e.g. an old "orcarouter"."""
    if name in ROUTERS:
        return True
    if name not in _WARNED_ROUTES:
        _WARNED_ROUTES.add(name)
        log.warning("llm: router %r is not supported any more and is ignored (OpenRouter is used)", name)
    return False


def known_route(name: str) -> str:
    """The model id without route names that no longer exist (``orcarouter+openrouter:x`` -> ``openrouter:x``)."""
    head, sep, rest = name.partition(":")
    if not (sep and rest and re.fullmatch(r"[a-z]+(\+[a-z]+)*", head)):
        return name
    names = head.split("+")
    kept = [n for n in names if n == "opencode" or known_router(n)]
    if len(kept) == len(names):
        return name
    return "+".join(kept) + ":" + rest if kept else rest


def route_in(config: dict[str, Any], name: str) -> str:
    """A model id with its route for a config (``Episode.llm``; also used before any episode exists, by the startup
    check of paid steps); web search (``:online``) is OpenRouter's."""
    llm = config.get("llm") or {}
    if name.endswith(":online"):
        return with_route(name, ["openrouter"])
    name = known_route(name)
    router = llm.get("router", "openrouter")
    routers = [router] if isinstance(router, str) else list(router)
    routers = [r for r in routers if known_router(r)] or ["openrouter"]
    if any(fnmatch.fnmatchcase(name, p) for p in llm.get("opencode") or []):
        return with_route(name, ["opencode", *routers] if llm.get("fallback", False) else ["opencode"])
    return with_route(name, routers)


SUBSCRIPTION_DOWN = "The ChatGPT subscription login of OpenCode does not work (run `opencode auth login`)"


def subscription_models(config: dict[str, Any]) -> list[str]:
    """Model ids of the LLM steps that go through the OpenCode subscription (see ``Episode.llm``)."""
    llm = config.get("llm") or {}
    patterns = llm.get("opencode") or []
    names: list[str] = []
    for value in [llm.get("model"), *(llm.get("stages") or {}).values()]:
        names += value if isinstance(value, list) else [value] if value else []
    models = []
    for name in names:
        transports, bare = split_route(name)
        if "opencode" in transports or (not transports and any(fnmatch.fnmatchcase(bare, p) for p in patterns)):
            models.append(bare)
    return list(dict.fromkeys(models))


def paying_allowed(config: dict[str, Any]) -> bool:
    """May this run pay for LLM requests? Yes when no model goes through the subscription (``llm.opencode`` empty: the
    run was set up for OpenRouter, or there is no subscription) or when ``llm.fallback`` allows the paid fallback."""
    llm = config.get("llm") or {}
    return not llm.get("opencode") or bool(llm.get("fallback", False))


def export_llm_env(config: dict[str, Any]) -> None:
    """Settings of the LLM clients for this process and the stage processes it starts (environment variables):
    the subscription's retry budget, its per-request fallback model and whether paying is allowed."""
    from stt.llm import SUBSCRIPTION_RETRY_ENV, SUBSCRIPTION_RETRY_S

    llm = config.get("llm") or {}
    os.environ[SUBSCRIPTION_RETRY_ENV] = str(llm.get("subscription_retry_s", SUBSCRIPTION_RETRY_S))
    os.environ[FALLBACK_ENV] = json.dumps(llm.get("subscription_fallback") or {})
    os.environ[FALLBACK_AFTER_ENV] = str(llm.get("subscription_fallback_after_s", FALLBACK_AFTER_S))
    os.environ[PAID_ENV] = "allow" if paying_allowed(config) else "deny"


def llm_steps(config: dict[str, Any]) -> dict[str, list[str]]:
    """The LLM steps of this run: name -> model ids with their routes (a list is a chain of models; ``model`` is the
    default of the steps without their own). Steps that will not run are left out."""
    llm = config.get("llm") or {}
    voiceover, stages = config.get("voiceover") or {}, llm.get("stages") or {}
    off = set()
    if not voiceover.get("stress_check", True):
        off |= {"stress", "stress_screen", "stress_proofread"}
    if not voiceover.get("stress_proofread", False):
        off.add("stress_proofread")
    if not (config.get("cast") or {}).get("enabled", True):
        off.add("cast")
    if not (config.get("translate") or {}).get("proofread", True):
        off.add("translate_proofread")
    steps = {"model": llm.get("model", config.get("llm_model", "openai/gpt-6-luna")), **stages}
    steps = {k: v for k, v in steps.items() if k not in off and v}
    return {k: [route_in(config, m) for m in (v if isinstance(v, list) else [v])] for k, v in steps.items()}


def check_paid_steps(config: dict[str, Any]) -> None:
    """When this run may not pay: drop paid models that come after a subscription one in a step's list (with a
    warning) and stop (``PaidNotAllowed``) when a step's only or first model is paid, naming the steps."""
    if paying_allowed(config):
        return
    blocked: list[str] = []
    stages = (config.get("llm") or {}).get("stages") or {}
    for step, models in llm_steps(config).items():
        paid = [split_route(m)[0][:1] != ["opencode"] for m in models]
        if paid[0]:
            blocked.append(f"{step}: {models[0]}")
        elif any(paid):
            kept = [m for m, is_paid in zip(models, paid) if not is_paid]
            log.warning("llm: %s is paid and this run may not pay: left out of step %r / платная модель убрана из "
                        "шага %r в этом запуске", ", ".join(m for m, is_paid in zip(models, paid) if is_paid), step,
                        step)
            if isinstance(stages.get(step), list):
                stages[step] = [m for m in stages[step] if route_in(config, m) in kept]
    if blocked:
        raise PaidNotAllowed("these steps use a paid model: " + "; ".join(blocked)
                             + " (pick a subscription model for them in the settings, or allow paying for this run)")
