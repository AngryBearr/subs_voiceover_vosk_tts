"""Where an LLM request goes: transports named in front of the model id.

``opencode+openrouter:openai/gpt-6.1-sol`` asks OpenCode (the ChatGPT subscription) first and
OpenRouter when OpenCode fails; ``openrouter:google/gemini-4-argon`` goes to OpenRouter only. A
model id without a known transport prefix keeps the client's own default.

OpenCode Go models (``opencode-go/<name>``) have other ids on OpenRouter: ``router_model`` maps them.

The rule for money: requests go to the subscription; a paid router is used only when the run was allowed to pay
(``STT_LLM_PAID``). Otherwise a request that cannot be answered by the subscription raises ``SubscriptionDown`` and
the whole run stops (``LLM_STOP_EXIT``) instead of paying silently. Before stopping, a weaker model of the same
subscription may answer one request (``subscription_fallback``, free).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


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


# ------------------------------------------------------------------ stop: the run ends instead of paying

LLM_STOP_EXIT = 75  # exit code of a stage process that stopped because of an LLMStop
STOP_MARK = "LLM STOPPED / LLM ОСТАНОВЛЕН"  # starts the stop message in logs (the UI and ``stt.run`` look for it)
PAID_ENV = "STT_LLM_PAID"  # "deny": a request to a paid transport raises PaidNotAllowed; unset or "allow": as before
COST_LOG_ENV = "STT_LLM_COST_LOG"  # JSON lines file that gets one line per paid answer
FALLBACK_ENV = "STT_LLM_SUBSCRIPTION_FALLBACK"  # JSON {model: weaker model of the same subscription}
FALLBACK_AFTER_ENV = "STT_LLM_SUBSCRIPTION_FALLBACK_AFTER_S"  # how long the main model gets before the weaker one
FALLBACK_AFTER_S = 120.0
PAID_MARK = "LLM paid in this run"
NOTHING_PAID_MARK = "LLM: nothing paid in this run"


class LLMStop(RuntimeError):
    """An LLM request cannot be answered without paying and the run must stop (never swallowed by a stage).

    ``reason`` says what happened; ``text`` (when set) is a ready stop message, e.g. one read from a stage's log.
    """

    def __init__(self, reason: str, text: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.text = text

    @property
    def message(self) -> str:
        return self.text or stop_message(self)


class SubscriptionDown(LLMStop):
    """The subscription does not answer (login, busy for the whole budget) or its usage limit is used up."""


class PaidNotAllowed(LLMStop):
    """A request would go to a paid transport, but this run may not pay."""


class PaidDown(LLMStop):
    """A run that may pay: the paid transport failed for good too (credits used up, key refused, still failing after
    its retries). Nothing more can be tried, and the message does not suggest paying."""

    def __init__(self, reason: str, text: str | None = None, subscription_tried: bool = False) -> None:
        super().__init__(reason, text)
        self.subscription_tried = subscription_tried  # the subscription did not answer before OpenRouter was asked


PAID_DEAD_RE = re.compile(r"HTTP (401|402|403)\b|insufficient[ _]credits?|no credits?|credit limit|key limit|"
                          r"quota exceeded|payment required", re.IGNORECASE)


def paid_run() -> bool:
    """True when ``stt.run`` explicitly allowed this run to pay (a run through OpenRouter or with the paid fallback)."""
    return os.environ.get(PAID_ENV, "").strip().lower() == "allow"


def is_paid_dead(detail: str) -> bool:
    """An answer of a paid transport that will not get better by waiting: refused key, no money, key limit."""
    return bool(PAID_DEAD_RE.search(detail))


def stop_message(error: LLMStop, paid_usd: float | None = None) -> str:
    """What the user reads when a run stops: what happened, that nothing was paid, what is kept, what to do next.

    Five lines: the marker, two English, two Russian (the interface shows the last two)."""
    if paid_usd is None:
        paid_usd = paid_total(os.environ.get(COST_LOG_ENV))[0]
    money_en = ("Nothing was paid." if paid_usd <= 0 else f"Paid in this run before the stop: ${paid_usd:.2f}.")
    money_ru = ("Ничего не оплачено." if paid_usd <= 0 else f"Оплачено в этом запуске до остановки: ${paid_usd:.2f}.")
    kept_en = "Finished steps and answers already received are kept and reused on the next run."
    kept_ru = "Законченные шаги и уже полученные ответы сохранены и будут использованы при следующем запуске."
    if isinstance(error, PaidDown):
        before_en = "The subscription did not answer and " if error.subscription_tried else ""
        before_ru = "Подписка не ответила, и " if error.subscription_tried else ""
        en1 = (f"{before_en}OpenRouter did not answer either (or the money there ran out): {error.reason} "
               f"The run stopped. {money_en}")
        en2 = f"{kept_en} Check the OpenRouter balance and key, then run again."
        ru1 = (f"{before_ru}OpenRouter тоже не ответил (или там закончились деньги): {error.reason} "
               f"Запуск остановлен. {money_ru}")
        ru2 = f"{kept_ru} Проверьте баланс и ключ OpenRouter и запустите снова."
        return "\n".join([STOP_MARK, en1, en2, ru1, ru2])
    if isinstance(error, PaidNotAllowed):
        what_en = f"A step wanted to use a paid service ({error.reason}), but this run may not pay."
        what_ru = f"Шаг хотел обратиться к платному сервису ({error.reason}), но этому запуску платить нельзя."
        fix_en = "Choose a subscription model for that step in the settings, or allow paying for this one run (below)."
        fix_ru = "Выберите для этого шага модель подписки в настройках или разрешите оплату для этого запуска (ниже)."
    else:
        what_en = f"OpenCode (the ChatGPT subscription) is not answering or its limit is used up: {error.reason}"
        what_ru = f"OpenCode (подписка ChatGPT) не отвечает или лимит исчерпан: {error.reason}"
        fix_en = "To continue, run again later (free)."
        fix_ru = "Чтобы продолжить, запустите снова позже (бесплатно)."
    return (f"{STOP_MARK}\n"
            f"{what_en} The run stopped. {money_en} {kept_en} {fix_en}\n"
            "Only if you allow paying: launch this one run through OpenRouter (in the UI choose \"API OpenRouter\" "
            "for the launch, or `ui.run_setup launch ... --transport openrouter`).\n"
            f"{what_ru} Запуск остановлен. {money_ru} {kept_ru} {fix_ru}\n"
            "Только если вы разрешаете оплату: запустите этот один запуск через OpenRouter (в интерфейсе выберите "
            "«API OpenRouter» при запуске или `ui.run_setup launch ... --transport openrouter`).")


def stop_message_ru(message: str) -> str:
    """The Russian half of a stop message (its last two lines), for the interface."""
    return "\n".join(message.splitlines()[-2:])


def stop_message_in(log_text: str) -> str | None:
    """The last stop message in a log (from its marker to the next log record), else None."""
    start = log_text.rfind(STOP_MARK)
    if start < 0:
        return None
    lines = log_text[start:].splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        if re.match(r"\d{4}-\d\d-\d\d |=== |Traceback", line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def exiting_on_stop(main: Callable[..., int]) -> Callable[..., int]:
    """Entry point of a stage process: an ``LLMStop`` is logged (message last) and becomes exit code ``LLM_STOP_EXIT``."""
    @functools.wraps(main)
    def wrapper(*args: Any, **kwargs: Any) -> int:
        try:
            return main(*args, **kwargs)
        except LLMStop as error:
            log.error("%s", error.message)
            return LLM_STOP_EXIT
    return wrapper


def paid_allowed() -> bool:
    """False when ``stt.run`` decided that this run may not pay (``STT_LLM_PAID=deny``)."""
    return os.environ.get(PAID_ENV, "allow").strip().lower() != "deny"


def current_stage() -> str | None:
    """The ``stt.run`` stage this process works for, if it runs inside one."""
    try:
        from stt import progress

        tracker = progress.current()
        return tracker.stage if tracker is not None else None
    except Exception:  # noqa: BLE001 - the stage is only a label
        return None


def require_paid(transport: str, model: str) -> None:
    """Raise ``PaidNotAllowed`` before anything is sent when ``transport`` is paid and this run may not pay."""
    if transport != "opencode" and not paid_allowed():
        raise PaidNotAllowed(f"{current_stage() or 'LLM step'}: {model} via {transport}")


# ------------------------------------------------------------------ cost log

_COST_LOCK = threading.Lock()


def record_paid(model: str, transport: str, cost_usd: float = 0.0) -> None:
    """Append one JSON line for an answer that cost money (cost > 0 or any non-subscription transport)."""
    path = os.environ.get(COST_LOG_ENV)
    if not path or (transport == "opencode" and cost_usd <= 0):
        return
    row = {"time": datetime.now().isoformat(timespec="seconds"), "stage": current_stage(), "model": model,
           "transport": transport, "cost_usd": round(float(cost_usd), 6)}
    try:
        with _COST_LOCK:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as error:  # a full disk must not fail an answer that was just paid for
        log.warning("could not write the LLM cost log %s: %s", path, error)


def paid_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """The rows of a cost log (none for a missing file; unreadable lines are skipped)."""
    if not path or not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def paid_total(path: str | Path | None) -> tuple[float, int]:
    """(dollars, requests) in a cost log."""
    rows = paid_rows(path)
    return sum(float(r.get("cost_usd") or 0.0) for r in rows), len(rows)


def paid_line(path: str | Path | None) -> str:
    """One line for the end of a run: nothing paid, or the sum with the requests by stage and model."""
    rows = paid_rows(path)
    if not rows:
        return NOTHING_PAID_MARK + " / в этом запуске ничего не оплачено"
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row.get('stage') or '-'}/{row.get('model')}"
        entry = groups.setdefault(key, [0.0, 0.0])
        entry[0] += 1
        entry[1] += float(row.get("cost_usd") or 0.0)
    detail = ", ".join(f"{key} x{int(n)} ${cost:.2f}" for key, (n, cost) in sorted(groups.items()))
    total = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    return f"{PAID_MARK}: ${total:.2f} ({len(rows)} requests: {detail})"


# ------------------------------------------------------------------ free fallback inside the subscription


def subscription_fallback(model: str) -> str | None:
    """The weaker model of the same subscription that stands in for ``model`` (env ``STT_LLM_SUBSCRIPTION_FALLBACK``)."""
    try:
        mapping = json.loads(os.environ.get(FALLBACK_ENV) or "{}")
    except ValueError:
        return None
    other = mapping.get(model) if isinstance(mapping, dict) else None
    return str(other) if other and other != model else None


def subscription_fallback_after_s() -> float:
    """Seconds the main model gets before one request goes to its fallback model."""
    try:
        return max(0.0, float(os.environ.get(FALLBACK_AFTER_ENV, FALLBACK_AFTER_S)))
    except ValueError:
        return FALLBACK_AFTER_S


def is_subscription_down(error: BaseException) -> bool:
    """True for an OpenCode failure that means "the subscription does not work now": the usage limit, the retry
    budget used up (``generate_with_retry``), or the login refused (HTTP 401/403). A failure of one request (a
    refusal, a bad request) is none of these: it fails that request only."""
    return (bool(getattr(error, "quota", False)) or bool(getattr(error, "exhausted", False))
            or getattr(error, "status", None) in (401, 403))


def generate_via_subscription(get_server: Callable[[str], Any], prompt: str, model: str, variant: str | None,
                              timeout: float, retry_s: float) -> tuple[str, str]:
    """``generate`` on the subscription with the free per-request fallback model; returns (text, model that answered).

    With a fallback configured for ``model`` the main model gets ``min(fallback_after_s, retry_s)`` of waiting; when
    it is still not answering (or its usage limit is reached) THIS request goes to the fallback model with the rest
    of the budget. The next request starts with the main model again. Without a fallback the main model gets the
    whole budget. A failure of the last model is raised as an ``OpenCodeError`` (the client turns it into
    ``SubscriptionDown`` or a paid fallback).
    """
    import utils.opencode_clean as opencode

    def attempt(name: str, budget: float) -> str:
        return opencode.generate_with_retry(lambda: get_server(name), prompt, name, variant, timeout, budget)

    fallback = subscription_fallback(model)
    if fallback is None:
        return attempt(model, retry_s), model
    first = min(subscription_fallback_after_s(), retry_s)
    try:
        return attempt(model, first), model
    except opencode.OpenCodeError as error:
        if not is_subscription_down(error):
            raise
        log.warning("%s is not answering (%s), this request went to %s (subscription, free) / %s не отвечает, "
                    "этот запрос отправлен в %s (подписка, бесплатно)", model, str(error)[:200], fallback, model,
                    fallback)
        used = 0.0 if getattr(error, "quota", False) else float(getattr(error, "spent", 0.0) or first)
        rest = max(0.0, retry_s - used)  # what the main model really used (pauses and requests) is gone
        return attempt(fallback, rest), fallback
