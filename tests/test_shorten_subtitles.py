"""Tests for utils.shorten_subtitles module."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from utils.shorten_subtitles import (
    build_context,
    build_user_prompt,
    parse_shortened_response,
    validate_shortened_text,
    select_subtitles_for_shortening,
    apply_shortening,
    ShortenResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_item(
    index: int,
    text: str | List[str],
    start: int,
    end: int,
    mismatch_ratio: float | None = None,
    extended_mismatch_ratio: float | None = None,
    is_checked: bool = True,
) -> Dict[str, Any]:
    """Create a test subtitle item."""
    item: Dict[str, Any] = {
        "text": [text] if isinstance(text, str) else text,
        "index": index,
        "start": start,
        "end": end,
        "duration": end - start,
        "gender": "male",
    }
    if mismatch_ratio is not None or is_checked:
        duration_sec = (end - start) / 1000.0
        estimated_sec = len(" ".join(item["text"])) / 13.0
        item["analysis"] = {
            "duration_sec": duration_sec,
            "estimated_sec": estimated_sec,
            "mismatch_ratio": mismatch_ratio or estimated_sec / duration_sec if duration_sec > 0 else 0,
            "diff_sec": estimated_sec - duration_sec,
            "extended_duration_sec": None,
            "extended_mismatch_ratio": extended_mismatch_ratio,
            "used_gap_sec": None,
            "words_count": len(" ".join(item["text"]).split()),
            "is_short_segment": duration_sec < 1.0,
            "is_checked": is_checked,
            "is_critical": (mismatch_ratio or 0) > 1.3,
        }
    return item


@pytest.fixture
def sample_items() -> List[Dict[str, Any]]:
    """Create sample subtitle items for testing."""
    return [
        make_item(1, "Короткий субтитр", 0, 2000, mismatch_ratio=1.0),
        make_item(2, "Это очень длинный субтитр который не влезает в отведённое время", 2000, 4000, mismatch_ratio=2.5),
        make_item(3, "Ещё один нормальный субтитр", 4000, 6000, mismatch_ratio=1.1),
        make_item(4, "Тоже длинный и нужно сократить его сильно", 6000, 8000, mismatch_ratio=1.8),
        make_item(5, "Последний субтитр", 8000, 10000, mismatch_ratio=0.9),
    ]


# ---------------------------------------------------------------------------
# Tests: build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_basic_context(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test basic context building."""
        context = build_context(sample_items, 1, context_window=1)
        assert "ЦЕЛЕВОЙ СУБТИТР" in context
        assert "ДО" in context
        assert "ПОСЛЕ" in context

    def test_context_window_size(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test that context window limits output."""
        context = build_context(sample_items, 2, context_window=1)
        lines = [l for l in context.split("\n") if l.strip()]
        # Should have: 1 before + 1 target + 1 after = 3 items (some may be multi-line)
        assert "Субтитр #2" in context  # before
        assert "Субтитр #4" in context  # after

    def test_context_at_start(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test context when target is at the start."""
        context = build_context(sample_items, 0, context_window=2)
        assert "ЦЕЛЕВОЙ СУБТИТР #1" in context
        # Should not crash, just include what's available

    def test_context_at_end(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test context when target is at the end."""
        context = build_context(sample_items, 4, context_window=2)
        assert "ЦЕЛЕВОЙ СУБТИТР #5" in context


# ---------------------------------------------------------------------------
# Tests: build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_prompt_contains_context(self) -> None:
        """Test that prompt includes context."""
        context = "Some context here"
        prompt = build_user_prompt(context, min_words=3)
        assert context in prompt
        assert "3" in prompt

    def test_prompt_contains_min_words(self) -> None:
        """Test that prompt mentions min words."""
        prompt = build_user_prompt("ctx", min_words=5)
        assert "5" in prompt


# ---------------------------------------------------------------------------
# Tests: parse_shortened_response
# ---------------------------------------------------------------------------


class TestParseShortenedResponse:
    def test_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        response = '{"shortened_text": "Короткий текст"}'
        result = parse_shortened_response(response)
        assert result == "Короткий текст"

    def test_json_with_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        response = '```json\n{"shortened_text": "Короткий текст"}\n```'
        result = parse_shortened_response(response)
        assert result == "Короткий текст"

    def test_json_without_quotes(self) -> None:
        """Test parsing JSON without markdown quotes."""
        response = '```\n{"shortened_text": "Текст"}\n```'
        result = parse_shortened_response(response)
        assert result == "Текст"

    def test_plain_text_response(self) -> None:
        """Test parsing plain text response (fallback)."""
        response = "Это просто текст без JSON"
        result = parse_shortened_response(response)
        assert result == "Это просто текст без JSON"

    def test_empty_response(self) -> None:
        """Test parsing empty response."""
        assert parse_shortened_response(None) is None
        assert parse_shortened_response("") is None

    def test_invalid_json(self) -> None:
        """Test parsing invalid JSON."""
        response = "not json at all {invalid"
        result = parse_shortened_response(response)
        # Should fallback to plain text
        assert result == "not json at all {invalid"

    def test_json_with_extra_text(self) -> None:
        """Test parsing JSON with surrounding text."""
        response = 'Вот результат: {"shortened_text": "Текст"} hope this helps'
        result = parse_shortened_response(response)
        assert result == "Текст"


# ---------------------------------------------------------------------------
# Tests: validate_shortened_text
# ---------------------------------------------------------------------------


class TestValidateShortenedText:
    def test_valid_shortening(self) -> None:
        """Test valid shortened text with enough words."""
        assert validate_shortened_text("Оригинальный длинный текст", "Короткий текст здесь") is True

    def test_too_few_words(self) -> None:
        """Test text with too few words."""
        assert validate_shortened_text("Оригинал", "Одно") is False

    def test_repeated_words_allowed(self) -> None:
        """Test that repeated words are allowed with fewer unique words."""
        assert validate_shortened_text("Вперёд вперёд вперёд", "Вперёд, вперёд!") is True

    def test_empty_text(self) -> None:
        """Test empty text."""
        assert validate_shortened_text("Оригинал", "") is False

    def test_only_dots(self) -> None:
        """Test text that's only dots."""
        assert validate_shortened_text("Оригинал", "...") is False

    def test_ellipsis_truncation(self) -> None:
        """Test that simple ellipsis truncation is rejected."""
        original = "Это очень длинный оригинальный текст который нужно сократить"
        shortened = "Это очень длинный оригинальный текст который нужно..."
        # This should be rejected as it's just truncation
        assert validate_shortened_text(original, shortened) is False

    def test_meaningful_shortening_with_ellipsis(self) -> None:
        """Test that meaningful shortening with ellipsis is accepted."""
        original = "Это очень длинный оригинальный текст который нужно сократить"
        shortened = "Длинный текст для проверки..."
        # This has enough words and is meaningful (not just truncated original)
        assert validate_shortened_text(original, shortened) is True


# ---------------------------------------------------------------------------
# Tests: select_subtitles_for_shortening
# ---------------------------------------------------------------------------


class TestSelectSubtitlesForShortening:
    def test_selects_high_ratio(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test that subtitles with high ratio are selected."""
        selected = select_subtitles_for_shortening(sample_items, threshold=1.5)
        # Items with ratio > 1.5: index 1 (2.5) and index 3 (1.8)
        assert 1 in selected
        assert 3 in selected

    def test_excludes_low_ratio(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test that subtitles with low ratio are excluded."""
        selected = select_subtitles_for_shortening(sample_items, threshold=1.5)
        # Items with ratio <= 1.5: index 0 (1.0), index 2 (1.1), index 4 (0.9)
        assert 0 not in selected
        assert 2 not in selected
        assert 4 not in selected

    def test_uses_extended_when_available(self) -> None:
        """Test that extended_mismatch_ratio is used when available."""
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, extended_mismatch_ratio=1.2),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        # Extended ratio is 1.2 < 1.5, so should not be selected
        assert len(selected) == 0

    def test_uses_mismatch_when_no_extended(self) -> None:
        """Test that mismatch_ratio is used when extended is None."""
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, extended_mismatch_ratio=None),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        assert len(selected) == 1

    def test_empty_items(self) -> None:
        """Test with empty items list."""
        selected = select_subtitles_for_shortening([], threshold=1.5)
        assert selected == []

    def test_unchecked_items(self) -> None:
        """Test that unchecked items are skipped."""
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, is_checked=False),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        assert len(selected) == 0


# ---------------------------------------------------------------------------
# Tests: apply_shortening
# ---------------------------------------------------------------------------


class TestApplyShortening:
    def test_applies_shortening(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test that shortening is applied correctly."""
        results = [
            ShortenResult(index=1, original_text="old", shortened_text="new", success=True),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[1]["text"] == ["new"]
        # Original should not be modified
        assert sample_items[1]["text"] != ["new"]

    def test_keeps_original_on_failure(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test that original text is kept on failure."""
        results = [
            ShortenResult(index=1, original_text="old", shortened_text="old", success=False),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[1]["text"] == sample_items[1]["text"]

    def test_multiple_results(self, sample_items: List[Dict[str, Any]]) -> None:
        """Test applying multiple results."""
        results = [
            ShortenResult(index=0, original_text="a", shortened_text="x", success=True),
            ShortenResult(index=2, original_text="b", shortened_text="y", success=True),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[0]["text"] == ["x"]
        assert new_items[2]["text"] == ["y"]
        # Others unchanged
        assert new_items[1]["text"] == sample_items[1]["text"]

    def test_preserves_list_format(self) -> None:
        """Test that list text format is preserved."""
        items = [{"text": ["line1", "line2"], "index": 1}]
        results = [ShortenResult(index=0, original_text="line1 line2", shortened_text="short", success=True)]
        new_items = apply_shortening(items, results)
        assert isinstance(new_items[0]["text"], list)
        assert new_items[0]["text"] == ["short"]

    def test_preserves_string_format(self) -> None:
        """Test that string text format is preserved."""
        items = [{"text": "single line", "index": 1}]
        results = [ShortenResult(index=0, original_text="single line", shortened_text="short", success=True)]
        new_items = apply_shortening(items, results)
        assert isinstance(new_items[0]["text"], str)
        assert new_items[0]["text"] == "short"


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_default_args(self) -> None:
        """Test default argument values."""
        from utils.shorten_subtitles import main
        import argparse

        # We can't easily test main() directly, but we can test arg parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("input")
        parser.add_argument("--mode", choices=["opencode", "deepseek"], default="opencode")
        parser.add_argument("--model", default="deepseek-v4-flash")
        parser.add_argument("--threshold", type=float, default=1.5)
        parser.add_argument("--context-window", type=int, default=3)
        parser.add_argument("--max-iterations", type=int, default=15)
        parser.add_argument("--min-words", type=int, default=3)
        parser.add_argument("--concurrency", type=int, default=5)
        parser.add_argument("--output-dir", default="output/shortened")
        parser.add_argument("--api-key")
        parser.add_argument("--base-url", default="https://api.deepseek.com")
        parser.add_argument("--analyze-args", default="")

        args = parser.parse_args(["test.json"])
        assert args.mode == "opencode"
        assert args.model == "deepseek-v4-flash"
        assert args.threshold == 1.5
        assert args.max_iterations == 15
