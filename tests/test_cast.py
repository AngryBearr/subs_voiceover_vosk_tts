from __future__ import annotations

from typing import Any

from stt import cast as cast_module
from stt.cast import cast_text, check_speakers, find_cast, find_member
from stt.llm import OpenRouterClient

CAST: dict[str, Any] = {"show": "Survivor", "season": "29", "cast": [
    {"name": "Jeff Probst", "aliases": [], "gender": "male", "role": "host"},
    {"name": "Jon Misch", "aliases": ["John Misch"], "gender": "male", "role": "contestant",
     "relations": "dating Jaclyn Schultz"},
    {"name": "John Rocker", "aliases": [], "gender": "male", "role": "contestant"},
    {"name": "Jaclyn Schultz", "aliases": [], "gender": "female", "role": "contestant"},
    {"name": "Flick Palmateer", "aliases": ["Felicity Palmateer"], "gender": "female", "role": "contestant"},
    {"name": "Natalie Anderson", "aliases": [], "gender": "female", "role": "contestant",
     "relations": "twin of Nadiya Anderson"},
]}


def _speaker(sid: str, name: str | None, gender: str = "male") -> dict[str, Any]:
    return {"id": sid, "name": name, "gender": gender, "cues": 5}


def test_find_member_spellings_and_ambiguity() -> None:
    assert find_member("John Misch", CAST)["name"] == "Jon Misch"
    assert find_member("Jaclyn", CAST)["name"] == "Jaclyn Schultz"
    assert find_member("John Rocker", CAST)["name"] == "John Rocker"
    assert find_member("John", CAST) is None  # Jon Misch or John Rocker
    assert find_member("Nadiya Anderson", CAST) is None  # the twin is not in this cast


def test_check_speakers_spelling_gender_and_unknown() -> None:
    speakers = [_speaker("S1", "John Misch"), _speaker("S2", "Jaclyn Schultz", "male"),
                _speaker("S3", "Baylor Wilson", "female"), _speaker("S4", None), _speaker("S5", "Jeff Probst"),
                _speaker("S6", "Felicity Palmateer", "female")]
    notes = check_speakers(speakers, CAST)
    assert speakers[0]["name"] == "Jon Misch" and "cast_check" not in speakers[0]
    assert speakers[1]["cast_check"] == "cast says female" and speakers[1]["gender"] == "male"  # reported only
    assert speakers[2]["cast_check"] == "not in cast"
    assert "cast_check" not in speakers[3] and "cast_check" not in speakers[4]
    assert speakers[5]["name"] == "Felicity Palmateer"  # a nickname of the cast name: kept
    assert len(notes) == 3


def test_cast_text_lines() -> None:
    text = cast_text(CAST)
    assert text.startswith("Survivor 29:\n")
    assert "- Jon Misch (John Misch) - male, contestant; dating Jaclyn Schultz" in text


def test_online_model_without_json_mode(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    bodies: list[dict[str, Any]] = []

    def fake_post(self: OpenRouterClient, body: dict[str, Any], transport: str) -> dict[str, Any]:
        bodies.append(body)
        return {"choices": [{"message": {"content": '{"cast": []}'}}], "usage": {}}

    monkeypatch.setattr(OpenRouterClient, "_post", fake_post)
    OpenRouterClient("openrouter:openai/gpt-6-luna:online").chat_json("system", "user")
    OpenRouterClient("openrouter:openai/gpt-6-luna").chat_json("system", "user")
    assert "response_format" not in bodies[0] and bodies[1]["response_format"] == {"type": "json_object"}


def test_check_speakers_keeps_name_over_quoted_nickname() -> None:
    cast = {"cast": [{"name": "Bradley “Dale” Wentworth", "aliases": ["Dale Wentworth"], "gender": "male"}]}
    speakers = [_speaker("S1", "Dale Wentworth"), _speaker("S2", "Dale")]
    assert check_speakers(speakers, cast) == []
    assert [s["name"] for s in speakers] == ["Dale Wentworth", "Dale"]


def test_find_cast_falls_back_to_the_next_model(monkeypatch: Any) -> None:
    answers = {"a": RuntimeError("refused"), "b": {"cast": [{"name": "X"}]}, "c": CAST}

    def fake_fetch(client: Any, folder: str, file: str, heard: list[str]) -> dict[str, Any]:
        answer = answers[client]
        if isinstance(answer, Exception):
            raise answer
        return {**answer, "model": client, "usage": {}}

    monkeypatch.setattr(cast_module, "OpenRouterClient", lambda model, reasoning=None: model)
    monkeypatch.setattr(cast_module, "fetch_cast", fake_fetch)
    found = find_cast(["a", "b", "c"], "Season 29", "ep.mkv", [])
    assert found["model"] == "c" and len(found["failed"]) == 2
    import pytest

    with pytest.raises(RuntimeError, match="no model found the cast"):
        find_cast(["a", "b"], "Season 29", "ep.mkv", [])
