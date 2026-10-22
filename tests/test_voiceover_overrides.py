from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voiceover.overrides import OVERRIDES_SCHEMA, Overrides, load_overrides


def _write(tmp_path: Path, units: Any, schema: str = OVERRIDES_SCHEMA) -> Path:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"schema": schema, "units": units}, ensure_ascii=False), encoding="utf-8")
    return path


def test_good_file_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, {"unit-138": {"text": "Но она думает иначе.", "note": "смысл"},
                             "unit-45": {"accept": True}})
    overrides = load_overrides(path)
    assert overrides.texts == {"unit-138": "Но она думает иначе."}
    assert overrides.accepted == frozenset({"unit-45"}) and overrides.notes == {"unit-138": "смысл"}
    assert not overrides.is_empty and overrides.source == path
    overrides.check_units(["unit-45", "unit-138", "unit-1"])


def test_canonical_ignores_notes_and_order(tmp_path: Path) -> None:
    first = load_overrides(_write(tmp_path, {"b": {"accept": True}, "a": {"text": "x", "note": "one"}}))
    second = load_overrides(_write(tmp_path, {"a": {"text": "x", "note": "two"}, "b": {"accept": True}}))
    assert first.canonical() == second.canonical()


def test_no_file_means_empty() -> None:
    assert load_overrides(None) == Overrides.empty()
    assert Overrides.empty().is_empty


def test_explicit_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_overrides(tmp_path / "nope.json")


def test_unknown_unit_ids_are_named(tmp_path: Path) -> None:
    overrides = load_overrides(_write(tmp_path, {"unit-1": {"accept": True}, "unit-999": {"text": "x"},
                                                 "unit-998": {"accept": True}}))
    with pytest.raises(ValueError, match="unit-998, unit-999"):
        overrides.check_units(["unit-1"])


@pytest.mark.parametrize("entry,message", [
    ({"text": "x", "accept": True}, "both"),
    ({"text": ""}, "non-empty"),
    ({"text": 5}, "non-empty"),
    ({"accept": False}, "needs either"),
    ({"accept": "yes"}, "needs either"),
    ({"note": "only a note"}, "needs either"),
    ({"text": "x", "colour": 1}, "unknown keys"),
    ({"text": "x", "note": 3}, "note"),
    ("text", "must be an object"),
])
def test_bad_entries_are_rejected(tmp_path: Path, entry: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_overrides(_write(tmp_path, {"unit-1": entry}))


def test_bad_schema_and_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        load_overrides(_write(tmp_path, {}, schema="other"))
    with pytest.raises(ValueError, match="units"):
        load_overrides(_write(tmp_path, ["unit-1"]))
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_overrides(broken)
