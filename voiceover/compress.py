"""Annotation, ladder compression, local checks and the per-unit ledger.

The model proposes five cumulative rungs per unit; the duration oracle and a
set of hard local checks decide which rung (if any) is used.  Functions here
take data and a client and do no file I/O.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from voiceover.llm import LLMBadResponse, LLMClient
from voiceover.oracle import Oracle
from voiceover.text import plain, syllables

ROLES = ("core", "support", "decor", "address")
ACTS = ("statement", "question", "exclamation", "command")
RUNG_COUNT = 5
LOSSY_RUNG = 5
MIN_COVERAGE = 0.8
# Only particles: negative pronouns ("ничего не ела" -> "не ела") may go in a compression.
NEGATIONS = frozenset({"не", "ни", "нет", "нельзя"})

_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_DIGITS_RE = re.compile(r"\d+")
_CARDINALS = frozenset(
    # "один/одна/одно" are left out: in speech they mostly work as "a certain", not as a count.
    "ноль два две три четыре пять шесть семь восемь девять десять одиннадцать двенадцать "
    "тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать двадцать тридцать "
    "сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто сто двести триста четыреста пятьсот шестьсот "
    "семьсот восемьсот девятьсот тысяча миллион".split())
# Ordinals are compared on their stem so that any case or gender ending matches.
_ORDINAL_RE = re.compile(
    r"^(перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят)"
    r"(ый|ого|ому|ым|ом|ая|ой|ую|ое|ые|ых|ыми|ий|ья|ье|ьи|ьего|ьему|ьим|ьем|ьей|ью)$")


# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class CompressionPolicy:
    bands: tuple[str, ...] = ("easy", "medium", "hard")
    context_units: int = 2
    prompt_version: str = "v2"
    fit_slack_sec: float = 0.0
    workers: int = 4  # parallel model calls; results are always assembled in episode order

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressionPolicy":
        defaults = cls()
        return cls(bands=tuple(str(b) for b in data.get("bands", defaults.bands)),
                   context_units=int(data.get("context_units", defaults.context_units)),
                   prompt_version=str(data.get("prompt_version", defaults.prompt_version)),
                   fit_slack_sec=float(data.get("fit_slack_sec", defaults.fit_slack_sec)),
                   workers=max(1, int(data.get("workers", defaults.workers))))

    def to_dict(self) -> dict[str, Any]:
        return {"bands": list(self.bands), "context_units": self.context_units,
                "prompt_version": self.prompt_version, "fit_slack_sec": self.fit_slack_sec}  # workers is not a result input


@dataclass(frozen=True)
class SceneUnit:
    unit_id: str
    text: str


@dataclass(frozen=True)
class UnitJob:
    """One unit that needs text work (or the human queue), with its numeric target."""

    unit_id: str
    cue_ids: tuple[str, ...]
    scene_id: str
    text_band: str
    text: str
    speech_sec: float
    removed_sec: float
    tempo_factor: float
    current_syllables: int
    target_syllables: int

    @property
    def target_base_sec(self) -> float:
        """Base-rate seconds the line may last: removal is measured at tempo, speech at base."""
        return self.speech_sec - self.removed_sec * self.tempo_factor


@dataclass(frozen=True)
class Fragment:
    id: str
    text: str
    role: str


@dataclass
class Annotation:
    unit_id: str
    fragments: list[Fragment]
    refs: dict[str, str]
    act: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"units": [{"id": f.id, "text": f.text, "role": f.role} for f in self.fragments],
                "refs": dict(self.refs), "act": self.act, "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, unit_id: str, data: dict[str, Any]) -> "Annotation":
        return cls(unit_id, [Fragment(str(u["id"]), str(u["text"]), str(u["role"])) for u in data["units"]],
                   {str(k): str(v) for k, v in data.get("refs", {}).items()}, str(data.get("act", "statement")),
                   [str(w) for w in data.get("warnings", [])])


@dataclass
class RungResult:
    rung: int
    text: str
    syllables: int
    estimated_sec: float | None
    fits: bool
    checks: list[str]
    note: str
    kept: list[str]
    dropped: list[str]
    compressed: list[str]

    @property
    def candidate(self) -> bool:
        return not self.checks

    def to_dict(self) -> dict[str, Any]:
        return {"rung": self.rung, "text": self.text, "syllables": self.syllables,
                "estimated_sec": None if self.estimated_sec is None else round(self.estimated_sec, 3),
                "fits": self.fits, "checks": list(self.checks), "note": self.note, "kept": list(self.kept),
                "dropped": list(self.dropped), "compressed": list(self.compressed)}


@dataclass
class LedgerEntry:
    unit_id: str
    cue_ids: list[str]
    scene_id: str
    text_band: str
    original: str
    original_syllables: int
    current_syllables: int
    target_syllables: int
    speech_sec: float
    target_base_sec: float
    annotation: dict[str, Any] | None
    rungs: list[RungResult]
    chosen_rung: int | None
    chosen_text: str
    chosen_syllables: int
    chosen_estimated_sec: float | None
    status: str
    lossy: bool
    llm_cached: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "cue_ids": list(self.cue_ids), "scene_id": self.scene_id,
                "text_band": self.text_band, "original": self.original,
                "original_syllables": self.original_syllables, "current_syllables": self.current_syllables,
                "target_syllables": self.target_syllables, "speech_sec": round(self.speech_sec, 3),
                "target_base_sec": round(self.target_base_sec, 3), "annotation": self.annotation,
                "rungs": [r.to_dict() for r in self.rungs], "chosen_rung": self.chosen_rung,
                "chosen_text": self.chosen_text, "chosen_syllables": self.chosen_syllables,
                "chosen_estimated_sec": None if self.chosen_estimated_sec is None
                else round(self.chosen_estimated_sec, 3),
                "status": self.status, "lossy": self.lossy, "llm_cached": self.llm_cached, "error": self.error}


# ---------------------------------------------------------------- jobs

def build_jobs(units: list[dict[str, Any]], placements: list[dict[str, Any]], rows: list[dict[str, Any]],
               policy: CompressionPolicy) -> tuple[list[UnitJob], dict[str, list[SceneUnit]]]:
    """Select the units to process (episode order) and group all units by scene."""
    placement_by_id = {p["unit_id"]: p for p in placements}
    row_by_id = {r["unit_id"]: r for r in rows}
    scenes: dict[str, list[SceneUnit]] = {}
    jobs: list[UnitJob] = []
    for unit in units:
        scenes.setdefault(unit["scene_id"], []).append(SceneUnit(unit["unit_id"], unit["text"]))
        placement = placement_by_id.get(unit["unit_id"])
        row = row_by_id.get(unit["unit_id"])
        if placement is None or row is None:
            continue
        band = placement["text_band"]
        if band != "timing_limited" and band not in policy.bands:
            continue
        jobs.append(UnitJob(unit["unit_id"], tuple(placement["cue_ids"]), unit["scene_id"], band,
                            str(row["display_text"]), float(row["speech_sec"]), float(placement["removed_sec"]),
                            float(placement["tempo_factor"]), int(placement["current_syllables"]),
                            int(placement["target_syllables"])))
    return jobs, scenes


def shortening_jobs(jobs: list[UnitJob]) -> list[UnitJob]:
    """Jobs that go to the model (timing-limited ones do not)."""
    return [job for job in jobs if job.text_band != "timing_limited"]


# ---------------------------------------------------------------- annotation

ANNOTATE_SYSTEM = (
    "Ты редактор, который готовит русский закадровый текст для субтитров. Твоя задача — разметить помеченные "
    "реплики на смысловые фрагменты.\n"
    "Правила:\n"
    "- Фрагменты — дословные подстроки реплики, идут в том же порядке и вместе покрывают всю реплику.\n"
    "- Роли: core — суждение, от которого зависит сцена; support — деталь, которую можно сжать или опустить; "
    "decor — вводные слова, оговорки, повторы; address — обращения.\n"
    "- refs: местоимения и неясные выражения реплики и то, на что они указывают (по соседним репликам).\n"
    "- act: statement, question, exclamation или command.\n"
    "- Размечай только помеченные звёздочкой реплики. Отвечай только JSON."
)


def annotation_user_prompt(scene: list[SceneUnit], marked: set[str]) -> str:
    lines = ["Сцена (реплики по порядку; звёздочкой помечены реплики для разметки):", ""]
    lines += [f"{'*' if unit.unit_id in marked else ' '} [{unit.unit_id}] {unit.text}" for unit in scene]
    lines += ["", "Верни JSON такого вида, только для помеченных реплик:",
              '{"units": {"<unit_id>": {"units": [{"id": "u1", "text": "<дословный фрагмент>", '
              '"role": "core|support|decor|address"}], "refs": {"<местоимение или выражение>": "<на что указывает>"}, '
              '"act": "statement|question|exclamation|command"}}}']
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def parse_annotation(unit_id: str, text: str, raw: Any) -> Annotation:
    """Validate one model annotation against the original text; never raises."""
    fallback = Annotation(unit_id, [Fragment("u1", text, "core")], {}, "statement", ["annotation_missing"])
    if not isinstance(raw, dict) or not isinstance(raw.get("units"), list):
        return fallback
    warnings: list[str] = []
    fragments: list[Fragment] = []
    for position, item in enumerate(raw["units"], start=1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            warnings.append("annotation_warning: unreadable fragment")
            continue
        fragment_text = item["text"].strip()
        if fragment_text not in text:
            warnings.append(f"annotation_warning: fragment is not a substring: {fragment_text!r}")
            continue
        role = str(item.get("role", "core"))
        if role not in ROLES:
            warnings.append(f"annotation_warning: unknown role {role!r}, treated as core")
            role = "core"
        fragments.append(Fragment(str(item.get("id") or f"u{position}"), fragment_text, role))
    if not fragments:
        return fallback
    covered = sum(_word_count(f.text) for f in fragments)
    total = _word_count(text)
    if total and covered / total < MIN_COVERAGE:
        warnings.append(f"annotation_warning: fragments cover {covered} of {total} words")
    refs = raw.get("refs") if isinstance(raw.get("refs"), dict) else {}
    act = str(raw.get("act", "statement"))
    return Annotation(unit_id, fragments, {str(k): str(v) for k, v in refs.items()},
                      act if act in ACTS else "statement", warnings)


def annotate_scenes(client: LLMClient, scenes: dict[str, list[SceneUnit]], jobs: list[UnitJob],
                    policy: CompressionPolicy) -> dict[str, Annotation]:
    """One model call per scene that holds at least one unit to shorten."""
    marked_by_scene: dict[str, dict[str, UnitJob]] = {}
    for job in shortening_jobs(jobs):
        marked_by_scene.setdefault(job.scene_id, {})[job.unit_id] = job

    def annotate_scene(scene_id: str) -> dict[str, Annotation]:
        marked = marked_by_scene[scene_id]
        try:
            result = client.complete_json(ANNOTATE_SYSTEM, annotation_user_prompt(scenes[scene_id], set(marked)),
                                          prompt_version=f"annotate-{policy.prompt_version}", tag=f"annotate:{scene_id}")
            answer = result.content.get("units")
        except LLMBadResponse:
            answer = None
        answers = answer if isinstance(answer, dict) else {}
        return {unit_id: parse_annotation(unit_id, job.text, answers.get(unit_id)) for unit_id, job in marked.items()}

    annotations: dict[str, Annotation] = {}
    scene_ids = [scene_id for scene_id in scenes if marked_by_scene.get(scene_id)]
    with ThreadPoolExecutor(max_workers=policy.workers) as pool:
        for part in pool.map(annotate_scene, scene_ids):
            annotations.update(part)
    return annotations


# ---------------------------------------------------------------- checks

def number_keys(text: str) -> set[str]:
    """Number tokens of a text: digit runs, cardinal words, ordinal stems."""
    keys = {f"d:{m}" for m in _DIGITS_RE.findall(text)}
    for word in _WORD_RE.findall(text.casefold()):
        if word in _CARDINALS:
            keys.add(f"c:{word}")
        else:
            match = _ORDINAL_RE.match(word)
            if match:
                keys.add(f"o:{match.group(1)}")
    return keys


def negation_count(text: str) -> int:
    return sum(1 for word in _WORD_RE.findall(text.casefold()) if word in NEGATIONS)


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    shortest = min(len(a), len(b))
    if shortest < 4:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= shortest - 2


def capitalised_words(text: str) -> list[str]:
    """Casefolded capitalised words that are not at the start of a sentence."""
    found: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if not word[0].isupper():
            continue
        before = text[:match.start()].rstrip(" \t\"'«“„(—–-")
        if not before or before[-1] in ".!?…":
            continue
        found.append(word.casefold())
    return found


def hard_checks(original: str, candidate: str, scene_text: str, dropped_text: str) -> list[str]:
    """Machine-readable reasons why a candidate is unusable (empty list means it passes)."""
    if not candidate.strip():
        return ["empty"]
    reasons: list[str] = []
    if "\n" in candidate.strip() or "\r" in candidate.strip():
        reasons.append("multi_line")
    if " ".join(candidate.split()) == " ".join(original.split()):
        reasons.append("unchanged")
    elif syllables(candidate) >= syllables(original):
        reasons.append("not_shorter")
    if number_keys(original) - number_keys(candidate):
        reasons.append("number_lost")
    scene_words = {w.casefold() for w in _WORD_RE.findall(scene_text)}
    if any(not any(_same_word(word, known) for known in scene_words) for word in capitalised_words(candidate)):
        reasons.append("new_name")
    if negation_count(candidate) < negation_count(original) - negation_count(dropped_text):
        reasons.append("negation_lost")
    return reasons


def estimate_sec(oracle: Oracle, speech_sec: float, original: str, candidate: str) -> float:
    """The line's own measured rate scaled by the oracle's candidate/original ratio."""
    base = oracle.predict(plain(original))
    if base <= 0:
        return oracle.predict(plain(candidate))
    return speech_sec * oracle.predict(plain(candidate)) / base


# ---------------------------------------------------------------- compression

COMPRESS_SYSTEM = (
    "Ты редактор русского закадрового текста. Сокращай реплику так, чтобы её было удобно произнести вслух.\n"
    "Правила:\n"
    "- Текст должен звучать естественно, как живая речь; не пиши телеграфным стилем и без сокращений.\n"
    "- Числа записывай словами только если в оригинале они были словами; цифры оставляй цифрами.\n"
    "- Не меняй числа, имена и отрицания, кроме случая, когда фрагмент с ними явно отброшен.\n"
    "- Сохраняй речевой акт (вопрос остаётся вопросом) и не добавляй новых фактов.\n"
    "- Каждая ступень — полная реплика в одну строку, без пояснений внутри текста. Отвечай только JSON."
)

LADDER = (
    "1. Убрать decor и address (вводные слова, оговорки, повторы, обращения).",
    "2. Сжать support-фрагменты (более короткие синонимы, местоимения там, где объект ясен из refs, "
    "действительный залог).",
    "3. Ужать формулировки core-фрагментов, не меняя суждения (короткие синонимы, лишние уточнения).",
    "4. Полностью убрать support-фрагмент.",
    "5. Переписать, оставив только core (с потерей смысла; такая ступень помечается как lossy).",
)
RUNG_RULES = (
    "Каждая следующая ступень должна быть короче предыдущей, если это вообще возможно: не перескакивай "
    "сразу к ступени 5. Ступень 5 — только отбрасывание core-содержания; если цель достигнута раньше, "
    "ступень 5 может повторять ступень 4. "
)


def compression_user_prompt(job: UnitJob, annotation: Annotation, scene: list[SceneUnit], context_units: int) -> str:
    index = next((i for i, u in enumerate(scene) if u.unit_id == job.unit_id), 0)
    before = scene[max(0, index - context_units):index]
    after = scene[index + 1:index + 1 + context_units]
    lines = ["Контекст сцены:"]
    lines += [f"[{u.unit_id}] {plain(u.text)}" for u in before]
    lines += [f">>> [{job.unit_id}] {plain(job.text)}  <<< (сокращаем эту реплику)"]
    lines += [f"[{u.unit_id}] {plain(u.text)}" for u in after]
    lines += ["", f"Оригинал: {job.text}", "", "Разметка оригинала:"]
    lines += [f"- {f.id} ({f.role}): {f.text}" for f in annotation.fragments]
    if annotation.refs:
        lines += ["", "Ссылки (refs):"] + [f"- {k} = {v}" for k, v in annotation.refs.items()]
    lines += ["", f"Речевой акт: {annotation.act}", "",
              f"Слогов сейчас: {job.current_syllables}; нужно не более: {job.target_syllables}; "
              f"убрать минимум: {job.current_syllables - job.target_syllables} "
              "(слог = гласная: а е ё и о у ы э ю я).", "",
              "Лестница сокращений (каждая ступень включает правки предыдущих):"]
    lines += list(LADDER)
    lines += ["", "Верни КАЖДУЮ из пяти ступеней, каждую как полную реплику в одну строку. Если ступень ничего не "
              "меняет, повтори текст предыдущей. " + RUNG_RULES +
              "Числа, имена и отрицания не меняй, если фрагмент с ними явно не "
              "отброшен. Речевой акт сохраняй, новых фактов не добавляй.", "",
              'Формат: {"rungs": [{"rung": 1, "text": "...", "kept": ["u1"], "dropped": ["u3"], '
              '"compressed": ["u2"], "note": "не более 15 слов"}, ... до rung 5]}']
    return "\n".join(lines)


def _id_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def score_rungs(job: UnitJob, annotation: Annotation, raw_rungs: Any, oracle: Oracle, scene_text: str,
                policy: CompressionPolicy) -> list[RungResult]:
    """Run the local checks and the oracle over the model's rungs (no choosing yet)."""
    by_number: dict[int, dict[str, Any]] = {}
    if isinstance(raw_rungs, list):
        for item in raw_rungs:
            if isinstance(item, dict) and isinstance(item.get("rung"), int):
                by_number.setdefault(item["rung"], item)
    fragment_text = {f.id: f.text for f in annotation.fragments}
    results: list[RungResult] = []
    previous = job.text
    for number in range(1, RUNG_COUNT + 1):
        item = by_number.get(number, {})
        text = item.get("text", "").strip() if isinstance(item.get("text"), str) else ""
        dropped = _id_list(item.get("dropped"))
        if not item:
            checks = ["missing"]
        elif number > 1 and " ".join(text.split()) == " ".join(previous.split()):
            checks = ["repeat_of_previous"]
        else:
            dropped_text = " ".join(fragment_text.get(i, "") for i in dropped)
            checks = hard_checks(job.text, text, scene_text, dropped_text)
        estimated = estimate_sec(oracle, job.speech_sec, job.text, text) if text else None
        fits = estimated is not None and estimated <= job.target_base_sec - policy.fit_slack_sec
        results.append(RungResult(number, text, syllables(text), estimated, fits, checks,
                                  str(item.get("note", "")), _id_list(item.get("kept")), dropped,
                                  _id_list(item.get("compressed"))))
        if text:
            previous = text
    return results


def choose_rung(rungs: list[RungResult]) -> tuple[RungResult | None, str]:
    """First rung that fits and passes; else the cheapest passing one (unresolved); else none (rejected)."""
    usable = [r for r in rungs if r.candidate]
    for rung in usable:
        if rung.fits:
            return rung, "resolved"
    if usable:
        return min(usable, key=lambda r: (r.estimated_sec if r.estimated_sec is not None else float("inf"),
                                          r.rung)), "unresolved"
    return None, "rejected"


def _entry(job: UnitJob, annotation: Annotation | None, rungs: list[RungResult], chosen: RungResult | None,
           status: str, cached: bool, error: str | None = None) -> LedgerEntry:
    return LedgerEntry(
        job.unit_id, list(job.cue_ids), job.scene_id, job.text_band, job.text, syllables(job.text),
        job.current_syllables, job.target_syllables, job.speech_sec, job.target_base_sec,
        annotation.to_dict() if annotation else None, rungs, chosen.rung if chosen else None,
        chosen.text if chosen else job.text, chosen.syllables if chosen else syllables(job.text),
        chosen.estimated_sec if chosen else None, status, _is_lossy(annotation, chosen), cached, error)


def _is_lossy(annotation: Annotation | None, chosen: RungResult | None) -> bool:
    """Rung 5, or any rung whose reported drops include a core fragment."""
    if chosen is None:
        return False
    if chosen.rung == LOSSY_RUNG:
        return True
    core_ids = {f.id for f in annotation.fragments if f.role == "core"} if annotation else set()
    return any(dropped in core_ids for dropped in chosen.dropped)


def compress_unit(client: LLMClient, job: UnitJob, annotation: Annotation | None, scene: list[SceneUnit],
                  oracle: Oracle, policy: CompressionPolicy) -> LedgerEntry:
    """Ask for the five rungs, score them locally and record the decision."""
    if job.text_band == "timing_limited":
        return _entry(job, None, [], None, "timing_limited", False)
    if annotation is None:
        annotation = Annotation(job.unit_id, [Fragment("u1", job.text, "core")], {}, "statement",
                                ["annotation_missing"])
    try:
        result = client.complete_json(COMPRESS_SYSTEM, compression_user_prompt(job, annotation, scene,
                                                                               policy.context_units),
                                      prompt_version=f"compress-{policy.prompt_version}", tag=f"compress:{job.unit_id}")
    except LLMBadResponse as error:
        return _entry(job, annotation, [], None, "rejected", False, str(error))
    scene_text = " ".join(u.text for u in scene)
    rungs = score_rungs(job, annotation, result.content.get("rungs"), oracle, scene_text, policy)
    chosen, status = choose_rung(rungs)
    return _entry(job, annotation, rungs, chosen, status, result.cached)


def run_compression(client: LLMClient, jobs: list[UnitJob], scenes: dict[str, list[SceneUnit]], oracle: Oracle,
                    policy: CompressionPolicy, annotations: dict[str, Annotation] | None = None) -> list[LedgerEntry]:
    """Annotate (unless given) then compress every job in episode order."""
    if annotations is None:
        annotations = annotate_scenes(client, scenes, jobs, policy)
    with ThreadPoolExecutor(max_workers=policy.workers) as pool:
        return list(pool.map(lambda job: compress_unit(client, job, annotations.get(job.unit_id),
                                                       scenes[job.scene_id], oracle, policy), jobs))


def summarize(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numbers over serialised ledger entries."""
    counts = {name: sum(1 for e in ledger if e["status"] == name)
              for name in ("resolved", "unresolved", "rejected", "timing_limited")}
    chosen = [e for e in ledger if e["chosen_rung"] is not None]
    histogram: dict[str, int] = {}
    for entry in chosen:
        key = str(entry["chosen_rung"])
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "units": len(ledger),
        "sent_to_model": counts["resolved"] + counts["unresolved"] + counts["rejected"],
        **counts,
        "rung_histogram": dict(sorted(histogram.items())),
        "lossy": sum(1 for e in ledger if e["lossy"]),
        "syllables_removed": sum(e["original_syllables"] - e["chosen_syllables"] for e in chosen),
        "estimated_sec_saved": round(sum(e["speech_sec"] - e["chosen_estimated_sec"] for e in chosen
                                         if e["chosen_estimated_sec"] is not None), 2),
        "annotation_warnings": sum(1 for e in ledger if e["annotation"] and e["annotation"]["warnings"]),
    }
