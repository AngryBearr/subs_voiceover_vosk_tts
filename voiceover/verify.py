"""Entailment check: does a shortened line still state the meaning it claims to keep?

One model call per measured rung.  Only core fragments decide the verdict; lost support
fragments are recorded for the human queue.  Functions here do no file I/O.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from voiceover.compress import Annotation, Fragment
from voiceover.llm import LLMBadResponse, LLMClient

KEPT_ROLES = ("core", "support")
# Bump when the verdict rules change without a prompt change, so cached stage artifacts are rebuilt.
VERIFY_RULES_VERSION = 2


@dataclass(frozen=True)
class VerifyPolicy:
    enabled: bool = True
    prompt_version: str = "v2"
    workers: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifyPolicy":
        defaults = cls()
        return cls(enabled=bool(data.get("enabled", defaults.enabled)),
                   prompt_version=str(data.get("prompt_version", defaults.prompt_version)),
                   workers=max(1, int(data.get("workers", defaults.workers))))


VERIFY_SYSTEM = (
    "Ты проверяющий сокращённых русских субтитров для закадровой озвучки. Тебе дают исходную реплику, "
    "её сокращённый вариант и список смысловых фрагментов, которые сокращение якобы сохраняет. Для каждого "
    "фрагмента реши, передаёт ли сокращённая реплика его суть.\n"
    "Фрагмент сохранён, если остались его событие или утверждение и его участники: кто, что, с кем, "
    "отрицание, время, причина. Перефразирование допустимо.\n"
    "Можно опускать без потери смысла: усилители и степень (совершенно, небольшой, очень, действительно), "
    "вводные и оценочные слова (честно говоря, знаешь, чистосердечно), повторы, уточнения способа, "
    "которые не меняют факт (написано → было, поделиться → рассказать).\n"
    "Фрагмент потерян только если сокращение меняет или убирает факт: другой субъект или объект, "
    "пропало отрицание, условие, причина, время, или высказывание стало обобщённым так, что зритель "
    "поймёт его иначе. Сомневаешься — считай сохранённым. Отвечай только JSON."
)


def verify_user_prompt(original: str, candidate: str, claimed: list[Fragment]) -> str:
    lines = [f"Исходная реплика: {original}", f"Сокращённая реплика: {candidate}", "",
             "Фрагменты, которые сокращение должно сохранять:"]
    lines += [f"- {f.id}: {f.text} ({f.role})" for f in claimed]
    lines += ["", "Для каждого фрагмента ответь, передаёт ли сокращённая реплика этот смысл (kept: true/false) "
              "и кратко почему.",
              'Формат: {"fragments": [{"id": "u1", "kept": true, "why": "не более 15 слов"}]}']
    return "\n".join(lines)


def claimed_fragments(annotation: Annotation, dropped: list[str], kept: list[str] | None = None) -> list[Fragment]:
    """Fragments the rung claims to keep: core and support ones not reported as dropped.

    A fragment listed as both kept and dropped counts as kept (the model is contradicting
    itself, so the checker decides).  When the annotation has no core fragment at all,
    the support fragments stand in for the core; when it has neither, the whole original
    line is the core, so the rung is still checked against something.
    """
    gone = set(dropped) - set(kept or [])
    claimed = [f for f in annotation.fragments if f.id not in gone and f.role in KEPT_ROLES]
    if any(f.role == "core" for f in annotation.fragments):
        return claimed
    if any(f.role == "support" for f in annotation.fragments):
        return [Fragment(f.id, f.text, "core") for f in claimed]
    return [Fragment("line", " ".join(f.text for f in annotation.fragments), "core")]


def _verdict(unit_id: str, rung: int, ok: bool, core_lost: list[str], support_lost: list[str], reasons: list[str],
             answers: list[Any], cached: bool) -> dict[str, Any]:
    return {"unit_id": unit_id, "rung": rung, "ok": ok, "core_lost": core_lost, "support_lost": support_lost,
            "reasons": reasons, "answers": answers, "llm_cached": cached}


def verify_rung(client: LLMClient | None, policy: VerifyPolicy, entry: dict[str, Any],
                rung: dict[str, Any]) -> dict[str, Any]:
    """Verdict for one rung of a serialised ledger entry; never raises on a bad answer."""
    unit_id, number = entry["unit_id"], rung["rung"]
    if not policy.enabled or client is None:
        return _verdict(unit_id, number, True, [], [], ["skipped"], [], False)
    if entry.get("annotation"):
        annotation = Annotation.from_dict(unit_id, entry["annotation"])
    else:
        annotation = Annotation(unit_id, [Fragment("u1", entry["original"], "core")], {}, "statement")
    claimed = claimed_fragments(annotation, rung["dropped"], rung.get("kept"))
    if not any(f.role == "core" for f in claimed):
        return _verdict(unit_id, number, False, [], [], ["no_core_left"], [], False)
    try:
        result = client.complete_json(VERIFY_SYSTEM, verify_user_prompt(entry["original"], rung["text"], claimed),
                                      prompt_version=f"verify-{policy.prompt_version}",
                                      tag=f"verify:{unit_id}#r{number}")
    except LLMBadResponse:
        return _verdict(unit_id, number, False, [], [], ["bad_response"], [], False)
    raw = result.content.get("fragments")
    answers = [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []
    kept = {str(a.get("id")): a.get("kept") is True for a in answers}
    reasons: list[str] = []
    if any(f.id not in kept for f in claimed):
        reasons.append("answer_incomplete")
    core_lost = [f.id for f in claimed if f.role == "core" and not kept.get(f.id, False)]
    support_lost = [f.id for f in claimed if f.role == "support" and not kept.get(f.id, False)]
    if core_lost:
        reasons.append("core_lost")
    return _verdict(unit_id, number, not core_lost, core_lost, support_lost, reasons, answers, result.cached)


def verify_rungs(client: LLMClient | None, policy: VerifyPolicy, ledger: list[dict[str, Any]],
                 measured: set[tuple[str, int]]) -> list[dict[str, Any]]:
    """Verdicts for every measured rung, in ledger order."""
    jobs = [(entry, rung) for entry in ledger for rung in entry["rungs"] if (entry["unit_id"], rung["rung"]) in measured]
    with ThreadPoolExecutor(max_workers=policy.workers) as pool:
        return list(pool.map(lambda job: verify_rung(client, policy, job[0], job[1]), jobs))
