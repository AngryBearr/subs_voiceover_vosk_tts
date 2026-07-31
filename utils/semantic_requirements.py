"""Extract immutable, auditable semantic obligations from subtitle source text."""

from __future__ import annotations

from dataclasses import dataclass
import re


PLANNER_ISSUE_CODES: frozenset[str] = frozenset(
    {
        "agent",
        "predicate",
        "object",
        "polarity",
        "modality",
        "tense",
        "speech_act",
        "time",
        "causality",
        "comparison",
        "alternative",
        "entity",
        "number",
        "reference",
        "spatial",
        "boundary",
    }
)


@dataclass(frozen=True)
class SemanticRequirement:
    """A concrete source obligation that a shortened candidate should preserve."""

    issue: str
    source_anchor: str
    requirement: str

    def as_dict(self) -> dict[str, str]:
        """Return the stable JSON representation used in review prompts and reports."""
        return {"issue": self.issue, "source_anchor": self.source_anchor, "requirement": self.requirement}


def _normalise(value: str) -> str:
    return value.lower().replace("ё", "е")


def extract_semantic_requirements(original: str, candidate: str) -> list[SemanticRequirement]:
    """Extract material source anchors absent literally from a candidate.

    The function is advisory: it never rejects a candidate and accepts no mutable state.
    Requirements are emitted in source order, with duplicate anchors removed stably.
    """
    source = _normalise(original)
    result_text = _normalise(candidate)
    found: list[tuple[int, SemanticRequirement]] = []

    def add(issue: str, anchor: str, requirement: str, position: int) -> None:
        normalized_anchor = _normalise(anchor)
        if normalized_anchor and normalized_anchor not in result_text:
            source_anchor = original[position:position + len(anchor)]
            found.append((position, SemanticRequirement(issue, source_anchor, requirement)))

    patterns: list[tuple[str, str, str]] = [
        ("agent", r"(?:мне|нам|тебе|вам|ему|ей|им)\s+(?:нужно|надо|придётся|придется)(?:\s+будет)?", "Preserve the dative person as the source agent."),
        ("modality", r"(?:должен|должна|должны)(?:\s+быть)?|(?:нужно|надо|придётся|придется|можем|может|смогут)(?:\s+будет)?", "Preserve the source modality."),
        ("modality", r"дай\s+я|давайте", "Preserve the source modal or volitional force."),
        ("tense", r"\b(?:буду|будешь|будет|будем|будете|будут)\b(?:\s+[а-я-]+)?", "Preserve the source future tense."),
        ("tense", r"\b(?:буду|будешь|будет|будем|будете|будут)\b\s+[а-я-]+(?:\s+[а-я-]+)?", "Preserve the source future tense."),
        ("time", r"сегодня утром|этим утром|все время|сейчас|сначала|наконец|вчера|завтра|теперь|потом|сегодня", "Preserve the source time anchor."),
        ("time", r"в (?:первую|последнюю|эту|следующую|прошлую) (?:ночь|утро|день|неделю|месяц|год)", "Preserve the specific source temporal period."),
        ("modality", r"конечно|по-моему|по моему|пожалуй|может быть|возможно|вероятно|лучше был бы|все равно|несмотря на|насколько мне известно", "Preserve the source modality or concession."),
        ("causality", r"потому что|так как|поэтому|из-за|из за|чтобы|хотя|несмотря на", "Preserve the source causal or concessive relation."),
        ("comparison", r"по сравнению с|как будто", "Preserve the source comparison relation."),
        ("reference", r"по ним|у нас|у меня|с нами|с ним|с ней|для них|от него|от нее|к нему|к ней", "Preserve the source prepositional reference."),
        ("boundary", r"всё|все|всякие|только", "Preserve the source quantifier or scope."),
    ]
    for issue, expression, requirement in patterns:
        for match in re.finditer(r"(?<![а-я])(?:" + expression + r")(?![а-я])", source):
            add(issue, match.group(0), requirement, match.start())

    for match in re.finditer(r"(?<![а-я])([а-я-]+йте)(?![а-я])", source):
        add("speech_act", match.group(1), "Preserve the source imperative speech act.", match.start())

    for match in re.finditer(r"(?<![а-я])(давайте|дай\s+я)(?![а-я])", source):
        add("speech_act", match.group(1), "Preserve the source speech act.", match.start())

    for match in re.finditer(r"(?<![а-я])(то есть\s+[^.!?;]+)", source):
        add("boundary", match.group(1), "Preserve the boundary and the complete clause introduced by 'то есть'.", match.start())

    for match in re.finditer(
        r"(?<![а-я])((?:этот|эта|это|эти|этого|этой|этому|этим|этом)\s+[а-я-]+)(?![а-я])", source
    ):
        add("reference", match.group(1), "Preserve the demonstrative reference and its noun.", match.start())

    spatial = r"вокруг\s+[а-я-]+|рядом с\s+(?:нами|ним|ней|ними|[а-я-]+)|возле\s+[а-я-]+|около\s+[а-я-]+|внутри\s+[а-я-]+|между\s+[а-я-]+|под\s+[а-я-]+|над\s+[а-я-]+"
    for match in re.finditer(r"(?<![а-я])(" + spatial + r")(?![а-я])", source):
        add("spatial", match.group(1), "Preserve the source spatial relation and its landmark.", match.start())

    comparisons = [
        r"не\s+столько\s*,?\s*[^,.!?;]*\s*сколько",
        r"столько\s+[^,.!?;]*\s*,?\s*сколько",
        r"не\s+[^,.!?;]+\s+а\s+[^,.!?;]+",
        r"как\s+[^,.!?;]+\s+так\s+и\s+[^,.!?;]+",
        r"скорее\s+[^,.!?;]+\s+чем\s+[^,.!?;]+",
    ]
    for expression in comparisons:
        for match in re.finditer(r"(?<![а-я])(" + expression + r")(?![а-я])", source):
            add("comparison", match.group(1), "Preserve the complete comparison pair, including both poles.", match.start())

    alternatives = list(re.finditer(r"\bили\b", source))
    if alternatives:
        start = max(0, source.rfind(" ", 0, alternatives[0].start() - 1) + 1)
        end = min(len(source), source.find(".", alternatives[-1].end()) if source.find(".", alternatives[-1].end()) >= 0 else len(source))
        anchor = source[start:end].strip(" ,;:")
        anchor_start = start + len(source[start:end]) - len(source[start:end].lstrip(" ,;:"))
        add("alternative", anchor, "Preserve every source alternative branch and their disjunction.", anchor_start)

    source_question = "?" in original
    candidate_question = "?" in candidate
    if source_question and not candidate_question:
        question_start = max(original.rfind(".", 0, original.find("?") if "?" in original else 0) + 1, 0)
        question_anchor = original[question_start:].strip()
        add("speech_act", question_anchor, "Preserve the source question speech act.", question_start + len(original[question_start:]) - len(original[question_start:].lstrip()))
    elif candidate_question and not source_question and original.strip():
        statement_anchor = original.strip()
        found.append((original.find(statement_anchor), SemanticRequirement("speech_act", statement_anchor, "Preserve the source statement speech act.")))

    imperative = r"[а-я-]+(?:йте|йся|ись|йтесь|итесь)"
    source_has_imperative = bool(re.search(r"(?<![а-я])" + imperative + r"(?![а-я])", source))
    candidate_has_imperative = bool(re.search(r"(?<![а-я])" + imperative + r"(?![а-я])", result_text))
    if not source_has_imperative and candidate_has_imperative:
        statement_anchor = original.strip()
        if statement_anchor:
            add("speech_act", statement_anchor, "Preserve the source statement speech act rather than introducing an imperative.", original.find(statement_anchor))

    unique: list[SemanticRequirement] = []
    seen: set[tuple[str, str]] = set()
    for _, requirement in sorted(found, key=lambda value: value[0]):
        key = (requirement.issue, _normalise(requirement.source_anchor))
        if key not in seen:
            seen.add(key)
            unique.append(requirement)
    return unique
