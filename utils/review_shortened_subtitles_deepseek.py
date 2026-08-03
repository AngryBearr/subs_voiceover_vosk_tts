"""Production DeepSeek Pro critic, editor, and blind-verifier workflow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from utils.analyze_text import join_text_lines
from utils.shorten_helpers import calculate_budget, load_api_key, load_json, save_json, target_min_words, validate_context_source
from utils.shorten_subtitles_deepseek import dynamic_max_tokens, shorten_via_deepseek_async
from utils.semantic_plan import PLANNER_PROMPT, SemanticPlan, parse_semantic_plans
from utils.semantic_requirements import PLANNER_ISSUE_CODES, SemanticRequirement, extract_semantic_requirements
from utils.shortening_domain import CalibratedDurationModel, load_calibrated_duration_profile
from utils.semantic_units import build_semantic_units

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_INPUT_TOKENS = 50_000
MAX_EXPLICIT_VERIFIER_CHECKS = 6
ThinkingMode = Literal["disabled", "high", "max", "auto"]
ISSUE_CODES = set(PLANNER_ISSUE_CODES) | {"grammar", "naturalness", "budget", "uncertain"}

SEMANTIC_CONTRACT = (
    "Оценивай материальную смысловую эквивалентность при необходимом сжатии: agent, predicate/object, "
    "polarity, modality, time, causality/concession, comparison, число и охват alternatives, entities, "
    "numbers и cross-line references. semantic_requirements are concrete source obligations. "
    "Spatial/reference anchors нельзя заменять расплывчатым там/туда/это без однозначного эквивалента; "
    "буквальное совпадение не требуется, допускается эквивалентная парафраза; fillers и детали можно убрать. Естественный "
    "фрагмент допустим в соседнем контексте. Читай вслух и в склейке; missing required object/goal, "
    "broken case attachment, ambiguous antecedent и editor notation "
    "являются predicate/grammar/reference/naturalness ошибками; запрещено импортировать факт из соседней строки. "
    "semantic_plan и semantic_requirements являются только входными свидетельствами: в output возвращай только "
    "схему ответа текущего этапа и не копируй вложенные поля входа."
)
CRITIC_PROMPT = SEMANTIC_CONTRACT + (
    " Ты независимый критик и не редактируешь. Для каждой цели верни строго JSON: "
    '{"results":[{"index":1,"verdict":"pass|fail|uncertain","issues":[]}]}. '
    "pass требует пустой issues; fail/uncertain — непустой список: "
    + ",".join(sorted(ISSUE_CODES)) + "."
)
EDITOR_PROMPT = SEMANTIC_CONTRACT + (
    " Ты редактор. Исправь только указанные issues, не показывай рассуждения. budget-only означает: "
    "сохрани всю семантику при уплотнении, особенно межстрочное сравнение/reference; сначала попробуй "
    "компактный список alternatives. Мысленно склей цель с реальными соседними строками и сохрани "
    "грамматическое присоединение и падеж. Не оставляй переходный predicate без обязательного object/goal. "
    "При конкурирующих antecedents предпочитай компактное явное существительное неоднозначному местоимению. "
    "Сохраняй материальные time/modality/comparison anchors и все alternatives естественными союзами; "
    "никогда не используй slash notation. Нельзя удалять центральное отношение или predicate ради лимита. "
    "Верни строго JSON: "
    '{"results":[{"index":1,"decision":"candidate|unresolved","candidate":"текст"}]}. '
    "candidate должен быть естественным и не длиннее max_chars."
)
VERIFIER_PROMPT = SEMANTIC_CONTRACT + (
    " Ты слепой финальный проверяющий. Естественный фрагмент в соседнем контексте допустим. "
    "required_checks are evaluated exactly once; all must pass for overall pass. Two-way entailment: every original proposition is recoverable from candidate "
    "and every candidate claim is supported by original; preserve agent/person, tense, and speech act/mood. "
    "Omitted or non-listed semantic-plan obligations are still evaluated holistically by two-way entailment; "
    "an omitted label is never assumed passed. "
    "JSON: "
    '{"results":[{"index":1,"verdict":"pass|fail|uncertain","issues":[],"required_checks":'
    '[{"issue":"alternative","verdict":"pass|fail"}]}]}. '
    "pass требует пустой issues; fail/uncertain — непустой список только из фиксированных issue codes: "
    + ",".join(sorted(ISSUE_CODES)) + "."
)
COMPACTION_PROMPT = SEMANTIC_CONTRACT + (
    " Ты выполняешь единственную точечную попытку уплотнения отклонённого editor candidate. Сократи его "
    "ровно настолько, насколько нужно для max_chars, сохранив все required_issues и смысловые anchors. "
    "Не возвращай исходный lossy candidate. Верни строго JSON в editor schema: "
    '{"results":[{"index":1,"decision":"candidate|unresolved","candidate":"текст"}]}.'
)
SYSTEM_PROMPT = CRITIC_PROMPT


@dataclass(frozen=True)
class ReviewTarget:
    position: int
    index: int
    original_text: str
    shortened_text: str
    max_chars: int
    effective_duration_sec: float
    compression_percent: float
    min_words: int
    requires_shortening: bool


@dataclass(frozen=True)
class EditGroup:
    """Stable editor membership, retained across the initial edit and repair."""

    unit_id: str
    cue_indices: tuple[int, ...]
    editable_indices: tuple[int, ...]


class DurationDiagnostic(TypedDict):
    predicted_duration_sec: float
    duration_limit_sec: float
    effective_duration_sec: float
    fit_ratio: float


def _indexed(items: Sequence[Dict[str, Any]], label: str) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for item in items:
        index = item.get("index")
        if not isinstance(index, int):
            raise ValueError(f"{label}: missing_or_invalid_index")
        if index in result:
            raise ValueError(f"{label}: duplicate_index:{index}")
        result[index] = item
    return result


def select_changed_targets(original: List[Dict[str, Any]], shortened: List[Dict[str, Any]],
                           avg_chars_per_sec: float = 13.0, target_ratio: float = 1.5,
                           min_words: int = 3,
                           duration_profile: Optional[CalibratedDurationModel] = None,
                           duration_fit_ratio: float = 1.0) -> List[ReviewTarget]:
    """Select changed union remaining-critical, once and in target order."""
    originals, current = _indexed(original, "original"), _indexed(shortened, "shortened")
    if set(originals) != set(current):
        raise ValueError(f"subtitle index sets differ: {sorted(set(originals) ^ set(current))}")
    targets: List[ReviewTarget] = []
    for position, item in enumerate(shortened):
        index = int(item["index"])
        old, now = join_text_lines(originals[index].get("text", "")).strip(), join_text_lines(item.get("text", "")).strip()
        analysis = item.get("analysis", {})
        budget_source = item if any(isinstance(analysis.get(k), (int, float)) and analysis[k] > 0
                                    for k in ("effective_duration_sec", "available_duration_sec", "duration_sec")) else originals[index]
        budget = calculate_budget(budget_source, target_ratio, avg_chars_per_sec)
        if duration_profile is None:
            ratio = analysis.get("extended_mismatch_ratio")
            ratio = ratio if isinstance(ratio, (int, float)) else analysis.get("mismatch_ratio")
            critical = bool(analysis.get("is_checked") and isinstance(ratio, (int, float)) and ratio > target_ratio)
        else:
            if (isinstance(duration_fit_ratio, bool) or not isinstance(duration_fit_ratio, (int, float))
                    or not math.isfinite(duration_fit_ratio) or duration_fit_ratio <= 0):
                raise ValueError("duration_fit_ratio must be finite and positive")
            critical = bool(
                analysis.get("is_checked") and budget.effective_duration_sec > 0
                and duration_profile.predict_seconds(now) > budget.effective_duration_sec * duration_fit_ratio
            )
        if old == now and not critical:
            continue
        targets.append(ReviewTarget(position, index, old, now, budget.max_chars, budget.effective_duration_sec,
                                    round(max(0.0, 100 * (1 - len(now) / max(1, len(old)))), 1),
                                    1 if len(old.split()) <= 1 else min(2, target_min_words(old, min_words)),
                                    critical))
    return targets


def route_targets(targets: Sequence[ReviewTarget], mode: ThinkingMode,
                  high_risk_threshold: float) -> Dict[Optional[str], List[ReviewTarget]]:
    """Compatibility helper; quality workflow itself uses fixed stage routing."""
    if mode != "auto":
        return {None if mode == "disabled" else mode: list(targets)}
    return {None: [t for t in targets if not t.requires_shortening and t.compression_percent < high_risk_threshold],
            "high": [t for t in targets if t.requires_shortening or t.compression_percent >= high_risk_threshold]}


def duration_diagnostic(target: ReviewTarget, candidate: str, duration_profile: Optional[CalibratedDurationModel],
                        duration_fit_ratio: float) -> Optional[DurationDiagnostic]:
    if duration_profile is None:
        return None
    if (isinstance(duration_fit_ratio, bool) or not isinstance(duration_fit_ratio, (int, float))
            or not math.isfinite(duration_fit_ratio)
            or duration_fit_ratio <= 0):
        raise ValueError("duration_fit_ratio must be finite and positive")
    return {
        "predicted_duration_sec": duration_profile.predict_seconds(candidate),
        "duration_limit_sec": target.effective_duration_sec * duration_fit_ratio,
        "effective_duration_sec": target.effective_duration_sec,
        "fit_ratio": duration_fit_ratio,
    }


def review_validation_error(target: ReviewTarget, candidate: str, max_chars: Optional[int] = None,
                            duration_profile: Optional[CalibratedDurationModel] = None,
                            duration_fit_ratio: float = 1.0) -> Optional[str]:
    text, limit = candidate.strip(), target.max_chars if max_chars is None else max_chars
    if not text:
        return "missing"
    if len(text) > limit:
        return f"too_long:{len(text)}>{limit}"
    if "/" in text and "/" not in target.original_text:
        return "slash_added"
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", target.original_text))
    if numbers - set(re.findall(r"\d+(?:[.,]\d+)?", text)):
        return "numbers_missing:" + ",".join(sorted(numbers - set(re.findall(r"\d+(?:[.,]\d+)?", text))))
    negative_markers = ("не", "нет", "никогда", "нельзя")
    original_is_negative = any(
        re.search(rf"\b{marker}\b", target.original_text, re.I)
        for marker in negative_markers
    )
    candidate_is_negative = any(
        re.search(rf"\b{marker}\b", text, re.I)
        for marker in negative_markers
    )
    if original_is_negative and not candidate_is_negative:
        return "negations_missing"
    diagnostic = duration_diagnostic(target, text, duration_profile, duration_fit_ratio)
    if diagnostic is not None and diagnostic["predicted_duration_sec"] > diagnostic["duration_limit_sec"]:
        return "duration_too_long"
    return None


def semantic_risk_issues(original: str, candidate: str) -> List[str]:
    """Project extracted semantic requirements to legacy unique issue labels."""
    return list(dict.fromkeys(requirement.issue for requirement in extract_semantic_requirements(original, candidate)))


def build_required_checks(risk_hints: Sequence[str], critic_issues: Sequence[str]) -> List[str]:
    """Build the bounded, stable ordered list of explicit verifier diagnostics."""
    result: List[str] = []
    for issue in list(risk_hints) + list(critic_issues):
        if issue == "uncertain" or issue in result:
            continue
        result.append(issue)
        if len(result) == MAX_EXPLICIT_VERIFIER_CHECKS:
            break
    return result


def build_context(original: List[Dict[str, Any]], items: List[Dict[str, Any]], targets: Sequence[ReviewTarget],
                  window: int, context_source: Optional[List[Dict[str, Any]]] = None) -> str:
    source = context_source or original
    positions = {int(item["index"]): pos for pos, item in enumerate(source)}
    current = {int(item["index"]): join_text_lines(item.get("text", "")) for item in items}
    wanted: set[int] = set()
    for target in targets:
        pos = positions[target.index]
        wanted.update(range(max(0, pos - window), min(len(source), pos + window + 1)))
    payload = [{"index": source[pos]["index"], "original_text": join_text_lines(source[pos].get("text", "")),
                "current_text": current.get(int(source[pos]["index"]), join_text_lines(source[pos].get("text", "")))}
               for pos in sorted(wanted)]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_prompt(original: List[Dict[str, Any]], items: List[Dict[str, Any]],
                 targets: Sequence[ReviewTarget], window: int, retry: bool = False,
                 retry_errors: Optional[Dict[int, str]] = None) -> str:
    """Compatibility prompt builder used by token-aware callers."""
    payload = [{"index": t.index, "original_text": t.original_text, "shortened_text": t.shortened_text,
                "max_chars": t.max_chars, "min_words": t.min_words} for t in targets]
    prompt = "Контекст:" + build_context(original, items, targets, window) + "\nЦели:" + json.dumps(payload, ensure_ascii=False)
    if retry:
        prompt += "\nОшибки:" + json.dumps({str(t.index): (retry_errors or {}).get(t.index, "invalid_result") for t in targets}, ensure_ascii=False)
    return prompt


def estimate_input_tokens(system_prompt: str, user_prompt: str) -> int:
    return len(system_prompt) + len(user_prompt) + 2_000


def split_batches(original: List[Dict[str, Any]], items: List[Dict[str, Any]],
                  targets: Sequence[ReviewTarget], batch_size: int, max_input_tokens: int,
                  window: int) -> tuple[List[List[ReviewTarget]], List[ReviewTarget]]:
    batches: List[List[ReviewTarget]] = []
    oversized: List[ReviewTarget] = []
    pending = _chunks(targets, batch_size)
    while pending:
        group = pending.pop(0)
        if estimate_input_tokens(SYSTEM_PROMPT, build_prompt(original, items, group, window)) <= max_input_tokens:
            batches.append(group)
        elif len(group) == 1:
            oversized.extend(group)
        else:
            middle = len(group) // 2
            pending[0:0] = [group[:middle], group[middle:]]
    return batches, oversized


def parse_assessments(response: Optional[str], expected: set[int]) -> Dict[int, tuple[str, List[str]]]:
    """Strictly parse complete, unique critic/verifier results."""
    try:
        data = json.loads(response or "")
    except (json.JSONDecodeError, TypeError):
        return {}
    values = data.get("results") if isinstance(data, dict) and set(data) == {"results"} else None
    if not isinstance(values, list):
        return {}
    parsed: Dict[int, tuple[str, List[str]]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"index", "verdict", "issues"}:
            return {}
        index, verdict, issues = value["index"], value["verdict"], value["issues"]
        if not isinstance(index, int) or index not in expected or index in parsed or verdict not in {"pass", "fail", "uncertain"}:
            return {}
        if not isinstance(issues, list) or any(not isinstance(x, str) or x not in ISSUE_CODES for x in issues):
            return {}
        issues = list(dict.fromkeys(issues))
        if (verdict == "pass") != (not issues):
            return {}
        parsed[index] = (verdict, issues)
    return parsed if set(parsed) == expected else {}


def parse_verifications(
    response: Optional[str], required: Dict[int, List[str]]
) -> Dict[int, tuple[str, List[str], Dict[str, str]]]:
    """Strictly parse verifier results and require one explicit check per known risk."""
    try:
        data = json.loads(response or "")
    except (json.JSONDecodeError, TypeError):
        return {}
    values = data.get("results") if isinstance(data, dict) and set(data) == {"results"} else None
    if not isinstance(values, list):
        return {}
    parsed: Dict[int, tuple[str, List[str], Dict[str, str]]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"index", "verdict", "issues", "required_checks"}:
            return {}
        index = value["index"]
        if not isinstance(index, int) or index not in required or index in parsed:
            return {}
        base = parse_assessments(json.dumps({"results": [{
            "index": index, "verdict": value["verdict"], "issues": value["issues"]
        }]}), {index})
        checks_raw = value["required_checks"]
        if not base or not isinstance(checks_raw, list):
            return {}
        checks: Dict[str, str] = {}
        for check in checks_raw:
            if (not isinstance(check, dict) or set(check) != {"issue", "verdict"}
                    or check.get("issue") not in required[index]
                    or check["issue"] in checks or check.get("verdict") not in {"pass", "fail"}):
                return {}
            checks[check["issue"]] = check["verdict"]
        if set(checks) != set(required[index]):
            return {}
        verdict, issues = base[index]
        if verdict == "pass" and any(result != "pass" for result in checks.values()):
            return {}
        parsed[index] = (verdict, issues, checks)
    return parsed if set(parsed) == set(required) else {}


def _parse_editor(response: Optional[str], expected: set[int]) -> Dict[int, tuple[str, str]]:
    try:
        data = json.loads(response or "")
    except (json.JSONDecodeError, TypeError):
        return {}
    values = data.get("results") if isinstance(data, dict) and set(data) == {"results"} else None
    result: Dict[int, tuple[str, str]] = {}
    if not isinstance(values, list):
        return result
    for value in values:
        if (not isinstance(value, dict) or set(value) != {"index", "decision", "candidate"}
                or value.get("index") not in expected or value["index"] in result
                or value.get("decision") not in {"candidate", "unresolved"} or not isinstance(value.get("candidate"), str)):
            return {}
        result[value["index"]] = (value["decision"], value["candidate"].strip())
    return result if set(result) == expected else {}


def pro_usage_summary(total: Dict[str, int]) -> Dict[str, Any]:
    cost = (total.get("prompt_cache_miss_tokens", 0) * .435 + total.get("prompt_cache_hit_tokens", 0) * .003625
            + total.get("completion_tokens", 0) * .87) / 1_000_000
    return {
        **total,
        "estimated_cost_usd": round(cost, 8),
        "pricing_model": DEFAULT_MODEL,
        "pricing_usd_per_million": {
            "input_cache_miss": 0.435,
            "input_cache_hit": 0.003625,
            "output": 0.87,
        },
    }


def duration_profile_metadata(profile: Optional[CalibratedDurationModel], fit_ratio: float) -> Optional[Dict[str, Any]]:
    if profile is None:
        return None
    return {
        "schema_version": 1,
        "profile_type": "calibrated_post_silence_duration",
        "measurement": profile.measurement,
        "voice": profile.voice,
        "rate": profile.rate,
        "coefficients": {
            "intercept_sec": profile.intercept_sec,
            "seconds_per_budget_char": profile.seconds_per_budget_char,
            "seconds_per_punctuation": profile.seconds_per_punctuation,
        },
        "training": {"sample_count": profile.sample_count, "episode_count": profile.episode_count},
        "validation": {
            "method": profile.validation_method,
            "mae_sec": profile.validation_mae_sec,
            "rmse_sec": profile.validation_rmse_sec,
            "r_squared": profile.validation_r_squared,
        },
        "fit_ratio": fit_ratio,
    }


def review_max_tokens(targets: Sequence[ReviewTarget], effort: Optional[str]) -> int:
    base = dynamic_max_tokens([target.max_chars for target in targets])
    return max(base, 8192) if effort == "high" else max(base, 16384) if effort == "max" else base


def _chunks(values: Sequence[ReviewTarget], size: int) -> List[List[ReviewTarget]]:
    return [list(values[i:i + max(1, size)]) for i in range(0, len(values), max(1, size))]


def _split_stage_batches(
    targets: Sequence[ReviewTarget],
    batch_size: int,
    system_prompt: str,
    prompt_factory: Any,
    max_input_tokens: int,
) -> tuple[List[List[ReviewTarget]], List[ReviewTarget]]:
    """Split using the exact stage prompt, reporting oversized single targets."""
    batches: List[List[ReviewTarget]] = []
    oversized: List[ReviewTarget] = []
    pending = _chunks(targets, batch_size)
    while pending:
        group = pending.pop(0)
        if estimate_input_tokens(system_prompt, prompt_factory(group)) <= max_input_tokens:
            batches.append(group)
        elif len(group) == 1:
            oversized.extend(group)
        else:
            middle = len(group) // 2
            pending[0:0] = [group[:middle], group[middle:]]
    return batches, oversized


async def review_subtitles(original: List[Dict[str, Any]], shortened: List[Dict[str, Any]], api_key: str,
                           model: str = DEFAULT_MODEL, thinking_mode: ThinkingMode = "auto",
                           high_risk_threshold: float = 35.0, context_window: int = 3,
                           batch_size: int = DEFAULT_BATCH_SIZE, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
                            min_words: int = 3, avg_chars_per_sec: float = 13.0, target_ratio: float = 1.5,
                             base_url: str = "https://api.deepseek.com", client: Any = None,
                             context_source: Optional[List[Dict[str, Any]]] = None,
                             concurrency: int = 3, enable_planner: bool = False,
                             duration_profile: Optional[CalibratedDurationModel] = None,
                             duration_fit_ratio: float = 1.0,
                             enable_multi_cue_editor: bool = False) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Run critic, conditional editor, blind verifier, and one bounded repair."""
    del thinking_mode, high_risk_threshold
    if duration_profile is not None and (isinstance(duration_fit_ratio, bool)
                                         or not isinstance(duration_fit_ratio, (int, float))
                                         or not math.isfinite(duration_fit_ratio)
                                         or duration_fit_ratio <= 0):
        raise ValueError("duration_fit_ratio must be finite and positive")
    if context_source is not None:
        validate_context_source(original, context_source)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    targets = select_changed_targets(original, shortened, avg_chars_per_sec, target_ratio, min_words,
                                     duration_profile, duration_fit_ratio)
    deterministic_requirements: Dict[int, tuple[SemanticRequirement, ...]] = {
        target.index: tuple(extract_semantic_requirements(target.original_text, target.shortened_text))
        for target in targets
    }
    semantic_plans: Dict[int, Optional[SemanticPlan]] = {target.index: None for target in targets}
    semantic_requirements: Dict[int, tuple[SemanticRequirement, ...]] = dict(deterministic_requirements)
    risk_hints = {
        target.index: list(dict.fromkeys(requirement.issue for requirement in semantic_requirements[target.index]))
        for target in targets
    }
    output, usage = copy.deepcopy(shortened), {}
    stage_names = (("planner", "critic", "editor", "compaction", "verifier", "repair", "reverify")
                   if enable_planner else ("critic", "editor", "compaction", "verifier", "repair", "reverify"))
    stage_usage: Dict[str, Dict[str, int]] = {name: {} for name in stage_names}
    stage_requests: Dict[str, int] = {name: 0 for name in stage_names}
    lineage = {t.index: {"flash": t.shortened_text, "editor": None, "compaction": None,
                          "repair": None, "rejected": [], "final": t.shortened_text} for t in targets}
    if duration_profile is not None:
        for target in targets:
            lineage[target.index]["duration_diagnostic"] = duration_diagnostic(
                target, target.shortened_text, duration_profile, duration_fit_ratio)
    assessments: Dict[str, Dict[int, tuple[str, List[str]]]] = {"critic": {}, "verifier": {}}
    verifier_history: Dict[int, List[Dict[str, Any]]] = {target.index: [] for target in targets}
    verifier_checks: Dict[int, Dict[str, str]] = {target.index: {} for target in targets}
    stage_errors: Dict[str, Dict[int, str]] = {name: {} for name in stage_names}
    decisions: Dict[str, Dict[int, Optional[str]]] = {"editor": {}, "compaction": {}, "repair": {}}
    valid_recovery: set[int] = set()
    blocked: set[int] = set()
    owns_client = client is None
    if owns_client:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0
        )
    semaphore = asyncio.Semaphore(concurrency)

    async def call(stage: str, system: str, prompt: str, effort: Optional[str], group: Sequence[ReviewTarget],
                   max_tokens_override: Optional[int] = None) -> Optional[str]:
        stage_requests[stage] += 1
        request_number = stage_requests[stage]
        indices = [target.index for target in group]
        print(f"Pro {stage} request {request_number} start: {indices}", flush=True)
        response = await shorten_via_deepseek_async(model, system, prompt, api_key, base_url,
                                                    semaphore=semaphore,
                                                     reasoning_effort=effort,
                                                     max_tokens=(max_tokens_override if max_tokens_override is not None
                                                                 else review_max_tokens(group, effort)),
                                                    client=client, usage_total=stage_usage[stage], api_retries=2)
        status = "completed" if response is not None else "error"
        print(f"Pro {stage} request {request_number} {status}: {indices}", flush=True)
        return response

    def context(group: Sequence[ReviewTarget], current_items: List[Dict[str, Any]]) -> str:
        return build_context(original, current_items, group, context_window, context_source)

    def planner_context(group: Sequence[ReviewTarget]) -> str:
        source = context_source or original
        positions = {int(item["index"]): pos for pos, item in enumerate(source)}
        wanted: set[int] = set()
        for target in group:
            pos = positions[target.index]
            wanted.update(range(max(0, pos - context_window), min(len(source), pos + context_window + 1)))
        payload = [{"index": source[pos]["index"],
                    "original_text": join_text_lines(source[pos].get("text", ""))}
                   for pos in sorted(wanted)]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def plan_field(target: ReviewTarget) -> Dict[str, Any]:
        return {"semantic_plan": semantic_plans[target.index].as_dict()} if enable_planner and semantic_plans[target.index] else {}

    def merged_requirements(target: ReviewTarget) -> tuple[SemanticRequirement, ...]:
        values = (list(semantic_plans[target.index].requirements) if semantic_plans[target.index] else [])
        values.extend(deterministic_requirements[target.index])
        result: list[SemanticRequirement] = []
        seen: set[tuple[str, str]] = set()
        for requirement in values:
            key = (requirement.issue, requirement.source_anchor.lower().replace("ё", "е"))
            if key not in seen:
                seen.add(key)
                result.append(requirement)
        return tuple(result)

    try:
        if enable_planner:
            def planner_prompt(group: Sequence[ReviewTarget]) -> str:
                payload = [{"index": t.index, "source_text": t.original_text,
                            "max_chars": t.max_chars,
                            "deterministic_requirements": [r.as_dict() for r in deterministic_requirements[t.index]]}
                           for t in group]
                return "Original-only context:" + planner_context(group) + "\nTargets:" + json.dumps(payload, ensure_ascii=False)

            planner_batches, oversized = _split_stage_batches(
                targets, min(2, batch_size), PLANNER_PROMPT, planner_prompt, max_input_tokens
            )
            for target in oversized:
                stage_errors["planner"][target.index] = "input_too_large"
                blocked.add(target.index)
            planner_responses = await asyncio.gather(*(
                call("planner", PLANNER_PROMPT, planner_prompt(group), None, group, max_tokens_override=8192)
                for group in planner_batches
            ))
            failed_planner_groups: List[List[ReviewTarget]] = []
            for group, response in zip(planner_batches, planner_responses):
                expected = {t.index: t.original_text for t in group}
                try:
                    parsed_plans = parse_semantic_plans(response, expected)
                except ValueError:
                    parsed_plans = {}
                if not parsed_plans:
                    failed_planner_groups.append(group)
                    provisional_error = "api_failure" if response is None else "parse_failure"
                    for target in group:
                        stage_errors["planner"][target.index] = provisional_error
                else:
                    for target in group:
                        plan = parsed_plans[target.index]
                        semantic_plans[target.index] = plan
                        semantic_requirements[target.index] = merged_requirements(target)

            retry_targets = [target for group in failed_planner_groups for target in group]
            retry_responses = await asyncio.gather(*(
                call("planner", PLANNER_PROMPT, planner_prompt([target]), None, [target], max_tokens_override=8192)
                for target in retry_targets
            ))
            for target, response in zip(retry_targets, retry_responses):
                expected = {target.index: target.original_text}
                try:
                    parsed_plans = parse_semantic_plans(response, expected)
                except ValueError:
                    parsed_plans = {}
                if parsed_plans:
                    semantic_plans[target.index] = parsed_plans[target.index]
                    semantic_requirements[target.index] = merged_requirements(target)
                    stage_errors["planner"].pop(target.index, None)
                else:
                    stage_errors["planner"][target.index] = (
                        "api_failure" if response is None else "parse_failure"
                    )
                    blocked.add(target.index)
            for target in targets:
                if target.index in blocked:
                    assessments["critic"][target.index] = ("uncertain", ["uncertain"])
                    assessments["verifier"][target.index] = ("uncertain", ["uncertain"])

        def critic_prompt(group: Sequence[ReviewTarget]) -> str:
            payload = [{"index": t.index, "original": t.original_text, "candidate": t.shortened_text,
                        "max_chars": t.max_chars, "semantic_requirements": [r.as_dict() for r in semantic_requirements[t.index]],
                        **plan_field(t)} for t in group]
            return "Context:" + context(group, shortened) + "\nTargets:" + json.dumps(payload, ensure_ascii=False)

        critic_targets = [target for target in targets if target.index not in blocked]
        critic_batches, oversized = _split_stage_batches(
            critic_targets, batch_size, CRITIC_PROMPT, critic_prompt, max_input_tokens
        )
        for target in oversized:
            assessments["critic"][target.index] = ("uncertain", ["uncertain"])
            stage_errors["critic"][target.index] = "input_too_large"
            blocked.add(target.index)
        critic_responses = await asyncio.gather(*(
            call("critic", CRITIC_PROMPT, critic_prompt(group), None, group)
            for group in critic_batches
        ))
        for group, response in zip(critic_batches, critic_responses):
            parsed = parse_assessments(response, {t.index for t in group})
            if not parsed and response is not None and enable_planner:
                retry_responses = await asyncio.gather(*(
                    call("critic", CRITIC_PROMPT, critic_prompt([target]), None, [target])
                    for target in group
                ))
                retried: Dict[int, tuple[str, List[str]]] = {}
                for target, retry_response in zip(group, retry_responses):
                    retry_parsed = parse_assessments(retry_response, {target.index})
                    if retry_parsed:
                        retried[target.index] = retry_parsed[target.index]
                parsed = retried
            for t in group:
                assessments["critic"][t.index] = parsed.get(t.index, ("uncertain", ["uncertain"]))
                if not parsed:
                    stage_errors["critic"][t.index] = (
                        "api_failure" if response is None else "parse_failure"
                    )
                elif t.index not in parsed:
                    stage_errors["critic"][t.index] = "parse_failure"
                else:
                    stage_errors["critic"].pop(t.index, None)

        editor_targets = [
            target for target in targets
            if (target.index not in blocked and (target.requires_shortening
                or assessments["critic"][target.index][0] != "pass"
                 or review_validation_error(target, target.shortened_text, duration_profile=duration_profile,
                                            duration_fit_ratio=duration_fit_ratio) is not None
                or risk_hints[target.index]))
        ]
        for target in editor_targets:
            initial_error = review_validation_error(target, target.shortened_text, duration_profile=duration_profile,
                                                    duration_fit_ratio=duration_fit_ratio)
            if initial_error is not None:
                stage_errors["editor"][target.index] = f"formal_error:{initial_error}"
        editor_targets = [t for t in editor_targets if t.index not in blocked]

        target_by_index = {target.index: target for target in editor_targets}
        edit_groups: Dict[int, EditGroup] = {
            target.index: EditGroup(f"cue:{target.index}", (target.index,), (target.index,))
            for target in editor_targets
        }
        multi_groups: List[EditGroup] = []
        if enable_multi_cue_editor and editor_targets:
            changed_editor_indices = [
                target.index for target in editor_targets
                if join_text_lines(
                    next(item for item in original if int(item["index"]) == target.index).get("text", "")
                ) != join_text_lines(
                    next(item for item in output if int(item["index"]) == target.index).get("text", "")
                )
            ]
            units = build_semantic_units(
                original, output, context_items=context_source,
                target_indices=changed_editor_indices,
            )
            all_editor_indices = set(target_by_index)
            for unit in units:
                editable = tuple(index for index in unit.changed_indices if index in all_editor_indices)
                evidence = set(unit.cue_indices) - set(editable)
                if len(editable) < 2 or evidence & all_editor_indices:
                    continue
                group = EditGroup(unit.unit_id, unit.cue_indices, editable)
                multi_groups.append(group)
                for index in editable:
                    edit_groups[index] = group

        def unit_context(group: EditGroup, current_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            by_original = {int(item["index"]): item for item in original}
            by_current = {int(item["index"]): item for item in current_items}
            return {
                "unit_id": group.unit_id,
                "cue_indices": list(group.cue_indices),
                "editable_indices": list(group.editable_indices),
                "evidence_indices": [index for index in group.cue_indices if index not in group.editable_indices],
                "cues": [{
                    "index": index,
                    "original_text": join_text_lines(by_original[index].get("text", "")),
                    "current_text": join_text_lines(by_current[index].get("text", "")),
                    "editable": index in group.editable_indices,
                    **({
                        "max_chars": target_by_index[index].max_chars,
                        "min_words": target_by_index[index].min_words,
                        "issues": normalized_editor_issues(target_by_index[index]),
                        "requires_shortening": target_by_index[index].requires_shortening,
                        "anchors": [r.source_anchor for r in semantic_requirements[index]],
                        "semantic_requirements": [r.as_dict() for r in semantic_requirements[index]],
                        **plan_field(target_by_index[index]),
                    } if index in target_by_index else {})
                } for index in group.cue_indices],
                "instruction": "Evidence indices are frozen. Return rows only for editable_indices. Do not move content between cue indices or change evidence.",
            }

        def normalized_editor_issues(target: ReviewTarget) -> List[str]:
            issues = list(assessments["critic"][target.index][1]) + risk_hints[target.index]
            formal_error = review_validation_error(target, target.shortened_text, duration_profile=duration_profile,
                                                   duration_fit_ratio=duration_fit_ratio)
            if (target.requires_shortening or
                    (formal_error is not None and formal_error.startswith("too_long:"))):
                issues.append("budget")
            return list(dict.fromkeys(issues))

        def editor_prompt(group: Sequence[ReviewTarget]) -> str:
            payload = [{"index": t.index, "original": t.original_text, "candidate": lineage[t.index]["final"],
                        "issues": normalized_editor_issues(t), "max_chars": t.max_chars,
                        "requires_shortening": t.requires_shortening,
                        "semantic_requirements": [r.as_dict() for r in semantic_requirements[t.index]],
                        **plan_field(t)} for t in group]
            return "Context:" + context(group, output) + "\nTargets:" + json.dumps(payload, ensure_ascii=False)

        atomic_groups: List[EditGroup] = []
        for group in multi_groups:
            prompt = "unit_context:" + json.dumps(unit_context(group, output), ensure_ascii=False)
            if estimate_input_tokens(EDITOR_PROMPT, prompt) <= max_input_tokens:
                atomic_groups.append(group)

        atomic_unit_ids = {group.unit_id for group in atomic_groups}

        atomic_prompts = [
            "unit_context:" + json.dumps(unit_context(group, output), ensure_ascii=False)
            for group in atomic_groups
        ]
        atomic_responses = await asyncio.gather(*(
            call("editor", EDITOR_PROMPT, prompt, None,
                 [target_by_index[index] for index in group.editable_indices])
            for group, prompt in zip(atomic_groups, atomic_prompts)
        ))
        for group, prompt, response in zip(atomic_groups, atomic_prompts, atomic_responses):
            expected = set(group.editable_indices)
            parsed = _parse_editor(response, expected)
            if not parsed and response is not None and enable_planner:
                retry_response = await call(
                    "editor", EDITOR_PROMPT, prompt, None,
                    [target_by_index[index] for index in group.editable_indices],
                )
                parsed = _parse_editor(retry_response, expected)
                response = retry_response
            errors: Dict[int, Optional[str]] = {}
            for index in group.editable_indices:
                target = target_by_index[index]
                decision, candidate = parsed.get(index, ("unresolved", ""))
                errors[index] = (
                    review_validation_error(target, candidate, duration_profile=duration_profile,
                                            duration_fit_ratio=duration_fit_ratio)
                    if decision == "candidate" else ("unresolved" if parsed and decision == "unresolved" else "parse_failure")
                )
            if not parsed or any(errors[index] is not None for index in group.editable_indices):
                for index in group.editable_indices:
                    target = target_by_index[index]
                    decisions["editor"][index] = None if not parsed else parsed[index][0]
                    error = errors[index]
                    stage_errors["editor"][index] = (
                        ("api_failure" if response is None else "parse_failure") if not parsed
                        else (f"formal_error:{error}" if error and error not in {"unresolved", "parse_failure"}
                              else error or "parse_failure")
                    )
                    if parsed and error and error not in {"unresolved", "parse_failure"}:
                        rejected: Dict[str, Any] = {"stage": "editor", "candidate": parsed[index][1], "reason": error}
                        if error == "duration_too_long":
                            rejected["duration_diagnostic"] = duration_diagnostic(
                                target, parsed[index][1], duration_profile, duration_fit_ratio)
                        lineage[index]["rejected"].append(rejected)
                continue
            for index in group.editable_indices:
                target = target_by_index[index]
                decision, candidate = parsed[index]
                decisions["editor"][index] = decision
                stage_errors["editor"].pop(index, None)
                lineage[index]["editor"] = candidate
                lineage[index]["final"] = candidate
                if duration_profile is not None:
                    lineage[index]["duration_diagnostic"] = duration_diagnostic(
                        target, candidate, duration_profile, duration_fit_ratio)
                output[target.position]["text"] = [candidate] if isinstance(output[target.position].get("text"), list) else candidate
                valid_recovery.add(index)

        editor_batches, oversized = _split_stage_batches(
            [target for target in editor_targets if edit_groups[target.index].unit_id not in atomic_unit_ids],
            min(2, batch_size), EDITOR_PROMPT, editor_prompt, max_input_tokens
        )
        for target in oversized:
            stage_errors["editor"][target.index] = "input_too_large"
            decisions["editor"][target.index] = None
            blocked.add(target.index)
        editor_prompts = [editor_prompt(group) for group in editor_batches]
        editor_responses = await asyncio.gather(*(
            call("editor", EDITOR_PROMPT, prompt, None, group)
            for group, prompt in zip(editor_batches, editor_prompts)
        ))
        for group, response in zip(editor_batches, editor_responses):
            parsed = _parse_editor(response, {t.index for t in group})
            if not parsed and response is not None and enable_planner:
                retry_responses = await asyncio.gather(*(
                    call("editor", EDITOR_PROMPT, editor_prompt([target]), None, [target])
                    for target in group
                ))
                retried: Dict[int, tuple[str, str]] = {}
                for target, retry_response in zip(group, retry_responses):
                    retry_parsed = _parse_editor(retry_response, {target.index})
                    if retry_parsed:
                        retried[target.index] = retry_parsed[target.index]
                parsed = retried
            for t in group:
                decision, candidate = parsed.get(t.index, ("unresolved", ""))
                decisions["editor"][t.index] = decision if parsed else None
                error = (review_validation_error(t, candidate, duration_profile=duration_profile,
                                                 duration_fit_ratio=duration_fit_ratio)
                         if decision == "candidate" else None)
                if not parsed:
                    stage_errors["editor"][t.index] = (
                        "api_failure" if response is None else "parse_failure"
                    )
                elif t.index not in parsed:
                    decisions["editor"][t.index] = None
                    stage_errors["editor"][t.index] = "parse_failure"
                elif decision == "unresolved":
                    stage_errors["editor"][t.index] = "unresolved"
                elif error is not None:
                    stage_errors["editor"][t.index] = f"formal_error:{error}"
                    rejected: Dict[str, Any] = {"stage": "editor", "candidate": candidate, "reason": error}
                    if error == "duration_too_long":
                        rejected["duration_diagnostic"] = duration_diagnostic(
                            t, candidate, duration_profile, duration_fit_ratio)
                    lineage[t.index]["rejected"].append(rejected)
                else:
                    stage_errors["editor"].pop(t.index, None)
                    lineage[t.index]["editor"] = candidate
                    lineage[t.index]["final"] = candidate
                    if duration_profile is not None:
                        lineage[t.index]["duration_diagnostic"] = duration_diagnostic(
                            t, candidate, duration_profile, duration_fit_ratio)
                    output[t.position]["text"] = [candidate] if isinstance(output[t.position].get("text"), list) else candidate
                    valid_recovery.add(t.index)

        compaction_targets = [
            t for t in editor_targets
            if stage_errors["editor"].get(t.index, "").startswith("formal_error:too_long:")
        ]

        def compaction_prompt(target: ReviewTarget) -> str:
            rejected = lineage[target.index]["rejected"][-1]["candidate"]
            payload = [{"index": target.index, "original": target.original_text,
                        "context": json.loads(context([target], output)),
                        "rejected_candidate": rejected, "current_length": len(rejected),
                        "max_chars": target.max_chars, "excess": len(rejected) - target.max_chars,
                         "required_issues": normalized_editor_issues(target),
                         "semantic_requirements": [r.as_dict() for r in semantic_requirements[target.index]],
                         **plan_field(target)}]
            return "Targets:" + json.dumps(payload, ensure_ascii=False)

        compaction_requests = [(target, compaction_prompt(target)) for target in compaction_targets]
        compaction_responses = await asyncio.gather(*(
            call("compaction", COMPACTION_PROMPT, prompt, None, [target])
            for target, prompt in compaction_requests
        ))
        for (target, prompt), response in zip(compaction_requests, compaction_responses):
            parsed = _parse_editor(response, {target.index})
            if not parsed and response is not None and enable_planner:
                retry_response = await call("compaction", COMPACTION_PROMPT, prompt, None, [target])
                parsed = _parse_editor(retry_response, {target.index})
                response = retry_response
            decision, candidate = parsed.get(target.index, ("unresolved", ""))
            decisions["compaction"][target.index] = decision if parsed else None
            error = (review_validation_error(target, candidate, duration_profile=duration_profile,
                                             duration_fit_ratio=duration_fit_ratio)
                     if decision == "candidate" else None)
            if not parsed:
                stage_errors["compaction"][target.index] = "api_failure" if response is None else "parse_failure"
            elif decision == "unresolved":
                stage_errors["compaction"][target.index] = "unresolved"
            elif error is not None:
                stage_errors["compaction"][target.index] = f"formal_error:{error}"
                rejected: Dict[str, Any] = {"stage": "compaction", "candidate": candidate, "reason": error}
                if error == "duration_too_long":
                    rejected["duration_diagnostic"] = duration_diagnostic(
                        target, candidate, duration_profile, duration_fit_ratio)
                lineage[target.index]["rejected"].append(rejected)
            else:
                lineage[target.index]["compaction"] = candidate
                lineage[target.index]["final"] = candidate
                if duration_profile is not None:
                    lineage[target.index]["duration_diagnostic"] = duration_diagnostic(
                        target, candidate, duration_profile, duration_fit_ratio)
                output[target.position]["text"] = ([candidate] if isinstance(output[target.position].get("text"), list)
                                                    else candidate)
                valid_recovery.add(target.index)

        async def verify(group: Sequence[ReviewTarget], stage: str = "verifier") -> None:
            # Deliberately contains no lineage, critic result, source label, or editor history.
            required = {t.index: build_required_checks(risk_hints[t.index], assessments["critic"][t.index][1])
                        for t in group}
            def verification_prompt(targets_for_prompt: Sequence[ReviewTarget]) -> str:
                payload = [{"index": t.index, "original": t.original_text,
                            "candidate": lineage[t.index]["final"], "max_chars": t.max_chars,
                            "required_checks": required[t.index],
                            "semantic_requirements": [r.as_dict() for r in semantic_requirements[t.index]],
                            **plan_field(t)} for t in targets_for_prompt]
                return "Context:" + context(targets_for_prompt, output) + "\nTargets:" + json.dumps(
                    payload, ensure_ascii=False
                )

            response = await call(stage, VERIFIER_PROMPT, verification_prompt(group), None, group)
            parsed = parse_verifications(response, required)
            if not parsed and response is not None and enable_planner:
                retry_responses = await asyncio.gather(*(
                    call(stage, VERIFIER_PROMPT, verification_prompt([target]), None, [target])
                    for target in group
                ))
                retried: Dict[int, tuple[str, List[str], Dict[str, str]]] = {}
                for target, retry_response in zip(group, retry_responses):
                    retry_parsed = parse_verifications(retry_response, {target.index: required[target.index]})
                    if retry_parsed:
                        retried[target.index] = retry_parsed[target.index]
                parsed = retried
            for t in group:
                parsed_value = parsed.get(t.index)
                assessments["verifier"][t.index] = (parsed_value[0], parsed_value[1]) if parsed_value else ("uncertain", ["uncertain"])
                verifier_checks[t.index] = parsed_value[2] if parsed_value else {}
                if parsed_value is None:
                    error = "api_failure" if response is None else "parse_failure"
                    stage_errors[stage][t.index] = error
                elif parsed_value[0] != "pass":
                    error = parsed_value[0]
                    stage_errors[stage][t.index] = error
                else:
                    error = None
                    stage_errors[stage].pop(t.index, None)
                verdict, issues = assessments["verifier"][t.index]
                verifier_history[t.index].append({
                    "stage": stage,
                    "verdict": verdict,
                    "issues": issues,
                    "required_checks": verifier_checks[t.index],
                    "error": error,
                })

        publishable = [t for t in targets if t.index not in blocked and
                       review_validation_error(t, str(lineage[t.index]["final"]), duration_profile=duration_profile,
                                               duration_fit_ratio=duration_fit_ratio) is None]

        def verifier_prompt(group: Sequence[ReviewTarget]) -> str:
            required = {t.index: build_required_checks(risk_hints[t.index], assessments["critic"][t.index][1])
                        for t in group}
            payload = [{"index": t.index, "original": t.original_text,
                        "candidate": lineage[t.index]["final"], "max_chars": t.max_chars,
                        "required_checks": required[t.index],
                        "semantic_requirements": [r.as_dict() for r in semantic_requirements[t.index]],
                        **plan_field(t)} for t in group]
            return "Context:" + context(group, output) + "\nTargets:" + json.dumps(payload, ensure_ascii=False)

        verifier_batches, oversized = _split_stage_batches(
            publishable, batch_size, VERIFIER_PROMPT, verifier_prompt, max_input_tokens
        )
        for target in oversized:
            stage_errors["verifier"][target.index] = "input_too_large"
            blocked.add(target.index)
        await asyncio.gather(*(verify(group) for group in verifier_batches))
        failed = [
            t for t in targets
            if (t.index not in blocked
                and (assessments["verifier"].get(t.index, ("uncertain", []))[0] != "pass"
                     or ((risk_hints[t.index] or assessments["critic"][t.index][0] != "pass")
                         and t.index not in valid_recovery)))
        ]
        failed_by_index = {target.index: target for target in failed}
        atomic_repair_groups: List[EditGroup] = []
        for group in multi_groups:
            members = tuple(index for index in group.editable_indices if index in failed_by_index)
            if not members:
                continue
            repair_group = EditGroup(group.unit_id, group.cue_indices, members)
            repair_prompt = "unit_context:" + json.dumps(unit_context(repair_group, output), ensure_ascii=False)
            if estimate_input_tokens(EDITOR_PROMPT, repair_prompt) <= max_input_tokens:
                atomic_repair_groups.append(repair_group)
        atomic_repair_ids = {group.unit_id for group in atomic_repair_groups}
        atomic_repair_snapshots: Dict[int, Dict[str, Any]] = {}
        atomic_repair_requests = [
            (group, "unit_context:" + json.dumps(unit_context(group, output), ensure_ascii=False))
            for group in atomic_repair_groups
        ]
        atomic_repair_responses = await asyncio.gather(*(
            call("repair", EDITOR_PROMPT, prompt, None,
                 [target_by_index[index] for index in group.editable_indices])
            for group, prompt in atomic_repair_requests
        ))
        repaired: List[ReviewTarget] = []
        for (group, prompt), response in zip(atomic_repair_requests, atomic_repair_responses):
            expected = set(group.editable_indices)
            parsed = _parse_editor(response, expected)
            if not parsed and response is not None and enable_planner:
                response = await call("repair", EDITOR_PROMPT, prompt, None,
                                      [target_by_index[index] for index in group.editable_indices])
                parsed = _parse_editor(response, expected)
            errors: Dict[int, Optional[str]] = {}
            for index in group.editable_indices:
                target = target_by_index[index]
                decision, candidate = parsed.get(index, ("unresolved", ""))
                errors[index] = (review_validation_error(
                    target, candidate, duration_profile=duration_profile, duration_fit_ratio=duration_fit_ratio
                ) if decision == "candidate" else ("unresolved" if parsed else "parse_failure"))
            if not parsed or any(errors[index] is not None for index in group.editable_indices):
                for index in group.editable_indices:
                    decisions["repair"][index] = None if not parsed else parsed[index][0]
                    error = errors[index]
                    stage_errors["repair"][index] = (
                        ("api_failure" if response is None else "parse_failure") if not parsed
                        else (f"formal_error:{error}" if error not in {"unresolved", "parse_failure"}
                              else error)
                    )
                continue
            for index in group.editable_indices:
                target = target_by_index[index]
                atomic_repair_snapshots[index] = {
                    "lineage": copy.deepcopy(lineage[index]),
                    "output_text": copy.deepcopy(output[target.position].get("text")),
                    "assessment": copy.deepcopy(
                        assessments["verifier"].get(index, ("uncertain", ["uncertain"]))
                    ),
                    "checks": copy.deepcopy(verifier_checks[index]),
                    "history": copy.deepcopy(verifier_history[index]),
                    "valid_recovery": index in valid_recovery,
                }
                decisions["repair"][index] = parsed[index][0]
                candidate = parsed[index][1]
                lineage[index]["repair"], lineage[index]["final"] = candidate, candidate
                output[target.position]["text"] = [candidate] if isinstance(output[target.position].get("text"), list) else candidate
                repaired.append(target)

        repair_requests: List[tuple[ReviewTarget, str]] = []
        for t in failed:
            if edit_groups.get(t.index, EditGroup(f"cue:{t.index}", (t.index,), (t.index,))).unit_id in atomic_repair_ids:
                continue
            issues = list(dict.fromkeys(
                assessments["verifier"].get(t.index, ("uncertain", ["uncertain"]))[1]
                + risk_hints[t.index] + assessments["critic"][t.index][1]
            ))
            payload = [{"index": t.index, "original": t.original_text, "candidate": lineage[t.index]["final"],
                        "issues": issues, "max_chars": t.max_chars,
                        "requires_shortening": t.requires_shortening,
                        "semantic_requirements": [r.as_dict() for r in semantic_requirements[t.index]],
                        **plan_field(t)}]
            repair_requests.append((t, "Context:" + context([t], output) + "\nTargets:" + json.dumps(payload, ensure_ascii=False)))
        repair_responses = await asyncio.gather(*(
            call("repair", EDITOR_PROMPT, prompt, None, [target])
            for target, prompt in repair_requests
        ))
        repair_snapshots: Dict[int, Dict[str, Any]] = {}
        for (t, prompt), response in zip(repair_requests, repair_responses):
            parsed = _parse_editor(response, {t.index})
            if not parsed and response is not None and enable_planner:
                retry_response = await call("repair", EDITOR_PROMPT, prompt, None, [t])
                parsed = _parse_editor(retry_response, {t.index})
                response = retry_response
            decision, candidate = parsed.get(t.index, ("unresolved", ""))
            decisions["repair"][t.index] = decision if parsed else None
            error = (review_validation_error(t, candidate, duration_profile=duration_profile,
                                             duration_fit_ratio=duration_fit_ratio)
                     if decision == "candidate" else None)
            if not parsed:
                stage_errors["repair"][t.index] = (
                    "api_failure" if response is None else "parse_failure"
                )
            elif decision == "unresolved":
                stage_errors["repair"][t.index] = "unresolved"
            elif error is not None:
                stage_errors["repair"][t.index] = f"formal_error:{error}"
                rejected: Dict[str, Any] = {"stage": "repair", "candidate": candidate, "reason": error}
                if error == "duration_too_long":
                    rejected["duration_diagnostic"] = duration_diagnostic(
                        t, candidate, duration_profile, duration_fit_ratio)
                lineage[t.index]["rejected"].append(rejected)
            else:
                stage_errors["repair"].pop(t.index, None)
                repair_snapshots[t.index] = {
                    "final": lineage[t.index]["final"],
                    "output_text": copy.deepcopy(output[t.position].get("text")),
                    "assessment": assessments["verifier"].get(t.index, ("uncertain", ["uncertain"])),
                    "checks": dict(verifier_checks[t.index]),
                    "valid_recovery": t.index in valid_recovery,
                }
                lineage[t.index]["repair"], lineage[t.index]["final"] = candidate, candidate
                if duration_profile is not None:
                    lineage[t.index]["duration_diagnostic"] = duration_diagnostic(
                        t, candidate, duration_profile, duration_fit_ratio)
                output[t.position]["text"] = [candidate] if isinstance(output[t.position].get("text"), list) else candidate
                repaired.append(t)
        await asyncio.gather(*(verify([target], "reverify") for target in repaired))
        for target in repaired:
            if target.index in atomic_repair_snapshots:
                continue
            required = build_required_checks(risk_hints[target.index], assessments["critic"][target.index][1])
            verdict, reverify_issues = assessments["verifier"][target.index]
            checks = verifier_checks[target.index]
            formal_error = review_validation_error(target, str(lineage[target.index]["final"]),
                                                   duration_profile=duration_profile,
                                                   duration_fit_ratio=duration_fit_ratio)
            reverify_passed = (
                verdict == "pass"
                and set(checks) == set(required)
                and all(value == "pass" for value in checks.values())
                and formal_error is None
            )
            if reverify_passed:
                valid_recovery.add(target.index)
                continue
            naturalness_only = (
                verdict == "fail" and required and set(checks) == set(required)
                and all(value == "pass" for value in checks.values())
                and formal_error is None and reverify_issues
                and set(reverify_issues) <= {"naturalness", "grammar"}
            )
            if naturalness_only:
                continue
            snapshot = repair_snapshots[target.index]
            attempted = str(lineage[target.index]["repair"])
            history_error = verifier_history[target.index][-1]["error"]
            reason = str(history_error or (f"formal_error:{formal_error}" if formal_error else f"verdict:{verdict}"))
            rejected: Dict[str, Any] = {"stage": "reverify", "candidate": attempted, "reason": reason}
            if formal_error == "duration_too_long":
                rejected["duration_diagnostic"] = duration_diagnostic(
                    target, attempted, duration_profile, duration_fit_ratio)
            lineage[target.index]["rejected"].append(rejected)
            lineage[target.index]["final"] = snapshot["final"]
            if duration_profile is not None:
                lineage[target.index]["duration_diagnostic"] = duration_diagnostic(
                    target, str(snapshot["final"]), duration_profile, duration_fit_ratio)
            output[target.position]["text"] = snapshot["output_text"]
            assessments["verifier"][target.index] = snapshot["assessment"]
            verifier_checks[target.index] = snapshot["checks"]
            if not snapshot["valid_recovery"]:
                valid_recovery.discard(target.index)
        failed_atomic_repair_ids: set[str] = set()
        for group in atomic_repair_groups:
            for index in group.editable_indices:
                target = target_by_index[index]
                required = build_required_checks(risk_hints[index], assessments["critic"][index][1])
                verdict = assessments["verifier"][index][0]
                checks = verifier_checks[index]
                formal_error = review_validation_error(
                    target, str(lineage[index]["final"]), duration_profile=duration_profile,
                    duration_fit_ratio=duration_fit_ratio,
                )
                if not (verdict == "pass" and set(checks) == set(required)
                        and all(value == "pass" for value in checks.values()) and formal_error is None):
                    failed_atomic_repair_ids.add(group.unit_id)
                    break
        for group in atomic_repair_groups:
            if group.unit_id not in failed_atomic_repair_ids:
                for index in group.editable_indices:
                    valid_recovery.add(index)
                continue
            for index in group.editable_indices:
                snapshot = atomic_repair_snapshots[index]
                target = target_by_index[index]
                lineage[index] = snapshot["lineage"]
                output[target.position]["text"] = snapshot["output_text"]
                assessments["verifier"][index] = snapshot["assessment"]
                verifier_checks[index] = snapshot["checks"]
                verifier_history[index] = snapshot["history"]
                if snapshot["valid_recovery"]:
                    valid_recovery.add(index)
                else:
                    valid_recovery.discard(index)
    finally:
        if owns_client:
            await client.close()

    unresolved: List[int] = []
    outcomes: List[Dict[str, Any]] = []
    for target in targets:
        final = str(lineage[target.index]["final"])
        verdict, issues = assessments["verifier"].get(target.index, ("uncertain", ["uncertain"]))
        required = build_required_checks(risk_hints[target.index], assessments["critic"][target.index][1])
        hard_gate_needs_recovery = bool(required) or assessments["critic"][target.index][0] != "pass"
        checks_pass = set(verifier_checks[target.index]) == set(required) and all(
            value == "pass" for value in verifier_checks[target.index].values()
        )
        final_duration_diagnostic = (duration_diagnostic(target, final, duration_profile, duration_fit_ratio)
                                     if duration_profile is not None else None)
        verified = (verdict == "pass" and checks_pass and review_validation_error(
            target, final, duration_profile=duration_profile, duration_fit_ratio=duration_fit_ratio) is None
                    and (not hard_gate_needs_recovery or target.index in valid_recovery))
        if not verified:
            unresolved.append(target.index)
        output[target.position]["text"] = [final] if isinstance(output[target.position].get("text"), list) else final
        lineage[target.index]["final"] = final
        if duration_profile is not None:
            lineage[target.index]["duration_diagnostic"] = final_duration_diagnostic
        outcomes.append({"index": target.index, "outcome": "verified" if verified else "unresolved",
                         "critic_verdict": assessments["critic"][target.index][0],
                          "critic_issues": assessments["critic"][target.index][1],
                          "planner_status": ("disabled" if not enable_planner else
                                              ("success" if semantic_plans[target.index] is not None else
                                               stage_errors["planner"].get(target.index, "unresolved"))),
                          "semantic_plan": (semantic_plans[target.index].as_dict()
                                             if semantic_plans[target.index] is not None else None),
                          "semantic_requirements": [r.as_dict() for r in semantic_requirements[target.index]],
                         "risk_hints": risk_hints[target.index],
                         "verifier_verdict": verdict, "verifier_issues": issues,
                         "verifier_required_checks": verifier_checks[target.index],
                         "verifier_history": verifier_history[target.index],
                         "editor_decision": decisions["editor"].get(target.index),
                         "compaction_decision": decisions["compaction"].get(target.index),
                         "repair_decision": decisions["repair"].get(target.index),
                         "stage_errors": {stage: errors[target.index] for stage, errors in stage_errors.items()
                                          if target.index in errors},
                          "lineage": lineage[target.index],
                          **({"duration_diagnostic": final_duration_diagnostic}
                             if final_duration_diagnostic is not None else {})})
    for totals in stage_usage.values():
        for key, value in totals.items():
            usage[key] = usage.get(key, 0) + value
    report = {"total": len(shortened), "selected": len(targets), "selected_indices": [t.index for t in targets],
              "automated_verified_count": len(targets) - len(unresolved),
              "unresolved_indices": unresolved, "fallback": unresolved, "outcomes": outcomes,
              "risk_hints": {str(index): hints for index, hints in risk_hints.items() if hints},
              "semantic_plans": {str(index): plan.as_dict() for index, plan in semantic_plans.items()
                                  if plan is not None},
              "stages": {name: {"usage": values, "requests": stage_requests[name]}
                         for name, values in stage_usage.items()},
              "retries": {"targeted_compactions": stage_requests["compaction"],
                          "bounded_repairs": stage_requests["repair"],
                          "fresh_reverifications": stage_requests["reverify"]},
               "context_mode": "full" if context_source is not None else "sparse"}
    if duration_profile is not None:
        report["duration_profile"] = duration_profile_metadata(duration_profile, duration_fit_ratio)
    print(
        f"Pro review completed: {len(targets) - len(unresolved)}/{len(targets)} verified, "
        f"{len(unresolved)} unresolved",
        flush=True,
    )
    usage_report = {**pro_usage_summary(usage), "stages": {k: pro_usage_summary(v) for k, v in stage_usage.items()}}
    if duration_profile is not None:
        usage_report["duration_profile"] = duration_profile_metadata(duration_profile, duration_fit_ratio)
    return output, report, usage_report


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Pro review CLI parser."""
    parser = argparse.ArgumentParser(description="Review shortened subtitles with separated DeepSeek Pro roles.")
    parser.add_argument("original")
    parser.add_argument("shortened")
    parser.add_argument("--context-source")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--thinking-mode",
        choices=["disabled", "high", "max", "auto"],
        default="auto",
        help="Deprecated compatibility option; fixed stage routing is always enforced",
    )
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--semantic-planner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-cue-editor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--avg-chars-per-sec", type=float, default=13.0)
    parser.add_argument("--target-ratio", type=float, default=1.5)
    parser.add_argument("--duration-profile", type=Path)
    parser.add_argument("--duration-fit-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", "-o", default="output/reviewed")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        duration_profile = (load_calibrated_duration_profile(args.duration_profile)
                            if args.duration_profile is not None else None)
        if duration_profile is not None and (isinstance(args.duration_fit_ratio, bool)
                                             or not isinstance(args.duration_fit_ratio, (int, float))
                                             or not math.isfinite(args.duration_fit_ratio)
                                             or args.duration_fit_ratio <= 0):
            raise ValueError("duration_fit_ratio must be finite and positive")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    api_key = args.api_key or load_api_key("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DeepSeek API key not found.", file=sys.stderr)
        return 1
    original_path, shortened_path = Path(args.original), Path(args.shortened)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output, report, usage = asyncio.run(review_subtitles(load_json(original_path), load_json(shortened_path), api_key,
            args.model, args.thinking_mode, 35.0, args.context_window, args.batch_size, args.max_input_tokens,
            args.min_words, args.avg_chars_per_sec, args.target_ratio, args.base_url,
             context_source=load_json(Path(args.context_source)) if args.context_source else None,
             concurrency=args.concurrency, enable_planner=args.semantic_planner,
             enable_multi_cue_editor=args.multi_cue_editor,
             duration_profile=duration_profile, duration_fit_ratio=args.duration_fit_ratio))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    stem = shortened_path.stem
    save_json(output_dir / f"{stem}_reviewed.json", output)
    report["context_source"] = str(Path(args.context_source)) if args.context_source else None
    (output_dir / f"{stem}_review.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{stem}_review.usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
