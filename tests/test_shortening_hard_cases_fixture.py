"""Integrity checks for the paid API hard-case smoke fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.shorten_helpers import validate_context_source


ROOT = Path(__file__).parent.parent
EXPECTED_INDICES = [333, 342, 346, 352, 358, 379, 380, 381, 391, 394]


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON subtitle list."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    assert isinstance(value, list)
    return value


def test_hard_case_fixture_matches_source_and_context() -> None:
    """Ensure hard cases are exact source items and valid full context targets."""
    hard_cases = load_json(ROOT / "skill_test/subs_analyzed_hard_cases.json")
    source = load_json(ROOT / "skill_test/subs_analyzed.json")
    context_source = load_json(ROOT / "skill_test/S01_E01_ru_analyzed.json")

    assert [item["index"] for item in hard_cases] == EXPECTED_INDICES
    source_by_index = {item["index"]: item for item in source}
    assert all(item == source_by_index[item["index"]] for item in hard_cases)
    validate_context_source(hard_cases, context_source)
