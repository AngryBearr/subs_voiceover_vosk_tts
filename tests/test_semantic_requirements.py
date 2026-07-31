"""Pure tests for concrete semantic requirement extraction."""

from __future__ import annotations

from utils.semantic_requirements import PLANNER_ISSUE_CODES, SemanticRequirement, extract_semantic_requirements


def test_regression_anchors_and_issue_codes() -> None:
    cases = [
        ("а Сегодня Утром пробежалась По Ним рукой...", "Пробежалась рукой", {("time", "Сегодня Утром"), ("reference", "По Ним")}),
        ("а не столько, сколько мне есть на самом деле.", "не столько", {("comparison", "не столько, сколько")}),
        ("Мы порасставили крысиные ловушки вокруг лагеря в первую ночь...", "Ловушки", {("spatial", "вокруг лагеря"), ("time", "в первую ночь")}),
        ("Крыса пыталась забраться на плот или побегать рядом с нами или добраться до чего-то.", "Крыса пыталась", {("spatial", "рядом с нами"), ("reference", "с нами"), ("alternative", "плот или побегать рядом с нами или добраться до чего-то")}),
    ]
    for original, candidate, expected in cases:
        requirements = extract_semantic_requirements(original, candidate)
        actual = {(r.issue, r.source_anchor) for r in requirements}
        assert actual == expected


def test_requirement_is_frozen_and_preserves_source_spelling() -> None:
    requirement = SemanticRequirement("time", "Сегодня утром", "Preserve the source time anchor.")
    assert requirement.as_dict()["source_anchor"] == "Сегодня утром"
    try:
        requirement.issue = "reference"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("SemanticRequirement must be frozen")


def test_literal_equivalence_immutability_and_stable_deduplication() -> None:
    original = "Сегодня утром мы были рядом с Нами, сегодня утром."
    candidate = "сегодня УТРОМ мы были рядом с нами, сегодня утром."
    assert extract_semantic_requirements(original, candidate) == []
    source = "Сегодня утром, сегодня утром, сегодня утром"
    requirements = extract_semantic_requirements(source, "")
    assert [(r.issue, r.source_anchor) for r in requirements] == [("time", "Сегодня утром")]
    assert original == "Сегодня утром мы были рядом с Нами, сегодня утром."
    assert candidate == "сегодня УТРОМ мы были рядом с нами, сегодня утром."


def test_planner_codes_and_russian_person_mood_tense_coverage() -> None:
    original = "Им нужно будет адаптироваться, или они будут изгнаны."
    requirements = extract_semantic_requirements(original, "Адаптируйся или изгнание.")
    actual = {(requirement.issue, requirement.source_anchor) for requirement in requirements}
    assert isinstance(PLANNER_ISSUE_CODES, frozenset)
    assert {"agent", "modality", "tense", "alternative"}.issubset({issue for issue, _ in actual})
    assert any(issue == "agent" and "Им нужно" in anchor for issue, anchor in actual)
    assert any(issue == "modality" and "нужно будет" in anchor for issue, anchor in actual)


def test_boundary_and_speech_act_requirements_are_advisory() -> None:
    original = "То есть должна быть проверка, а не отказ."
    requirements = extract_semantic_requirements(original, "Проверка.")
    assert any(requirement.issue == "boundary" and "То есть должна быть" in requirement.source_anchor for requirement in requirements)
    assert any(requirement.issue == "modality" and "должна быть" in requirement.source_anchor for requirement in requirements)
    assert any(requirement.issue == "speech_act" for requirement in extract_semantic_requirements("Давайте уйдем.", "Уйдем."))


def test_candidate_reflexive_imperative_is_detected_without_overmarking_source_imperative() -> None:
    source = "Им нужно будет адаптироваться или они будут изгнаны."
    bad_candidate = "Адаптируйся или изгнание."
    assert any(requirement.issue == "speech_act" for requirement in extract_semantic_requirements(source, bad_candidate))
    assert not any(requirement.issue == "speech_act"
                   for requirement in extract_semantic_requirements("Адаптируйся сейчас.", "Адаптируйся сейчас."))
    assert not any(requirement.issue == "speech_act"
                   for requirement in extract_semantic_requirements("Сделайте это сейчас.", "сделайте это сейчас."))


def test_question_speech_act_anchor_is_case_insensitive() -> None:
    requirements = extract_semantic_requirements("Ты готов?", "ТЫ ГОТОВ?")
    assert not any(requirement.issue == "speech_act" for requirement in requirements)
