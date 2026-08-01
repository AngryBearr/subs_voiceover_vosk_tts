"""Tests for utils.shorten_helpers module."""

from __future__ import annotations

import json
import asyncio
import copy
import sys
import pytest
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List

from utils.shorten_subtitles_deepseek import (
    build_parser,
    main as deepseek_main,
    shorten_subtitles_deepseek,
    shorten_via_deepseek_async,
)
from utils.shortening_domain import (
    POST_SILENCE_MEASUREMENT_LABEL,
    CalibratedDurationModel,
    DurationSelectionPolicy,
)

from utils.shorten_helpers import (
    build_context,
    build_user_prompt,
    build_batch_context,
    build_batch_user_prompt,
    parse_shortened_response,
    parse_batch_response,
    validate_shortened_text,
    select_subtitles_for_shortening,
    apply_shortening,
    get_system_prompt,
    ShortenResult,
    SYSTEM_PROMPT_SHORTEN,
    SYSTEM_PROMPT_REPHRASE,
    calculate_budget,
    hydrate_context_timing,
    refresh_analysis,
    target_min_words,
    load_json,
    validate_context_source,
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
        context = build_context(sample_items, 1, context_window=1)
        assert "ЦЕЛЕВОЙ СУБТИТР" in context
        assert "ДО" in context
        assert "ПОСЛЕ" in context

    def test_context_window_size(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_context(sample_items, 2, context_window=1)
        assert "Субтитр #2" in context
        assert "Субтитр #4" in context

    def test_context_at_start(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_context(sample_items, 0, context_window=2)
        assert "ЦЕЛЕВОЙ СУБТИТР #1" in context

    def test_context_at_end(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_context(sample_items, 4, context_window=2)
        assert "ЦЕЛЕВОЙ СУБТИТР #5" in context

    def test_no_timestamps(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_context(sample_items, 1, context_window=1)
        assert "[0.0s" not in context
        assert "s -" not in context


# ---------------------------------------------------------------------------
# Tests: build_batch_context
# ---------------------------------------------------------------------------


class TestBuildBatchContext:
    def test_single_subtitle_batch(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_batch_context(sample_items, [1], context_window=1)
        assert json.loads(context)[1]["index"] == 2

    def test_multiple_subtitles_batch(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_batch_context(sample_items, [1, 3], context_window=1)
        payload = json.loads(context)
        assert [item["index"] for item in payload] == [1, 2, 3, 4, 5]
        assert context.count("Ещё один нормальный субтитр") == 1

    def test_empty_batch(self, sample_items: List[Dict[str, Any]]) -> None:
        context = build_batch_context(sample_items, [], context_window=1)
        assert context == "[]"

    def test_real_full_episode_neighbor_for_352(self) -> None:
        root = Path(__file__).parent.parent
        source = load_json(root / "skill_test/S01_E01_ru_analyzed.json")
        sparse = load_json(root / "skill_test/subs_analyzed.json")
        position = next(i for i, value in enumerate(sparse) if value["index"] == 352)
        payload = json.loads(build_batch_context(sparse, [position], 1, source))
        neighbor = next(value for value in payload if value["index"] == 351)
        source_neighbor = next(value for value in source if value["index"] == 351)
        assert neighbor["text"] == " ".join(source_neighbor["text"])

    def test_context_source_validation(self) -> None:
        target = [{"index": 2, "text": ["same"]}]
        with pytest.raises(ValueError, match="duplicate_index:2"):
            validate_context_source(target, [{"index": 2, "text": ["same"]}, {"index": 2, "text": ["same"]}])
        with pytest.raises(ValueError, match="text_mismatch:2"):
            validate_context_source(target, [{"index": 2, "text": ["different"]}])


# ---------------------------------------------------------------------------
# Tests: build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_prompt_contains_context(self) -> None:
        context = "Some context here"
        prompt = build_user_prompt(context, min_words=3)
        assert context in prompt
        assert "3" in prompt

    def test_prompt_contains_min_words(self) -> None:
        prompt = build_user_prompt("ctx", min_words=5)
        assert "5" in prompt


# ---------------------------------------------------------------------------
# Tests: build_batch_user_prompt
# ---------------------------------------------------------------------------


class TestBuildBatchUserPrompt:
    def test_shorten_strategy(self) -> None:
        prompt = build_batch_user_prompt("context", min_words=3, strategy="shorten")
        assert "Сократи" in prompt
        assert "min_words" in prompt

    def test_rephrase_strategy(self) -> None:
        prompt = build_batch_user_prompt("context", min_words=3, strategy="rephrase")
        assert "Перефразируй" in prompt
        assert "компактнее" in prompt

    def test_default_strategy(self) -> None:
        prompt = build_batch_user_prompt("context", min_words=3)
        assert "Сократи" in prompt


# ---------------------------------------------------------------------------
# Tests: get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_shorten_strategy(self) -> None:
        prompt = get_system_prompt("shorten", min_words=3)
        assert "JSON" in prompt
        assert "max_chars" in prompt

    def test_rephrase_strategy(self) -> None:
        prompt = get_system_prompt("rephrase", min_words=3)
        assert prompt == get_system_prompt("shorten", min_words=9)

    def test_min_words_substitution(self) -> None:
        prompt = get_system_prompt("shorten", min_words=5)
        assert "5" not in prompt


# ---------------------------------------------------------------------------
# Tests: parse_shortened_response
# ---------------------------------------------------------------------------


class TestParseShortenedResponse:
    def test_valid_json(self) -> None:
        response = '{"shortened_text": "Короткий текст"}'
        result = parse_shortened_response(response)
        assert result == "Короткий текст"

    def test_json_with_markdown(self) -> None:
        response = '```json\n{"shortened_text": "Короткий текст"}\n```'
        result = parse_shortened_response(response)
        assert result == "Короткий текст"

    def test_json_without_quotes(self) -> None:
        response = '```\n{"shortened_text": "Текст"}\n```'
        result = parse_shortened_response(response)
        assert result == "Текст"

    def test_plain_text_response(self) -> None:
        response = "Это просто текст без JSON"
        result = parse_shortened_response(response)
        assert result == "Это просто текст без JSON"

    def test_empty_response(self) -> None:
        assert parse_shortened_response(None) is None
        assert parse_shortened_response("") is None

    def test_invalid_json(self) -> None:
        response = "not json at all {invalid"
        result = parse_shortened_response(response)
        assert result == "not json at all {invalid"

    def test_json_with_extra_text(self) -> None:
        response = 'Вот результат: {"shortened_text": "Текст"} hope this helps'
        result = parse_shortened_response(response)
        assert result == "Текст"


# ---------------------------------------------------------------------------
# Tests: parse_batch_response
# ---------------------------------------------------------------------------


class TestParseBatchResponse:
    def test_valid_batch_response(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '{"results": [{"index": 2, "shortened_text": "Короткий"}, {"index": 4, "shortened_text": "Тоже короткий"}]}'
        result = parse_batch_response(response, [1, 3], sample_items)
        assert result == {1: "Короткий", 3: "Тоже короткий"}

    def test_empty_response(self, sample_items: List[Dict[str, Any]]) -> None:
        assert parse_batch_response(None, [1], sample_items) == {}
        assert parse_batch_response("", [1], sample_items) == {}

    def test_invalid_json(self, sample_items: List[Dict[str, Any]]) -> None:
        result = parse_batch_response("not json", [1], sample_items)
        assert result == {}

    def test_missing_index(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '{"results": [{"index": 2, "shortened_text": "Текст"}]}'
        result = parse_batch_response(response, [1, 3], sample_items)
        assert result == {1: "Текст"}
        assert 3 not in result

    def test_markdown_wrapped(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '```json\n{"results": [{"index": 2, "shortened_text": "Текст"}]}\n```'
        result = parse_batch_response(response, [1], sample_items)
        assert result == {1: "Текст"}

    def test_single_result_fallback(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '{"shortened_text": "Текст"}'
        result = parse_batch_response(response, [1], sample_items)
        assert result == {1: "Текст"}

    def test_string_index(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '{"results": [{"index": "2", "shortened_text": "Текст"}]}'
        result = parse_batch_response(response, [1], sample_items)
        assert result == {1: "Текст"}

    def test_empty_results_array(self, sample_items: List[Dict[str, Any]]) -> None:
        response = '{"results": []}'
        result = parse_batch_response(response, [1], sample_items)
        assert result == {}

    def test_unknown_subtitle_id_is_not_treated_as_array_position(self) -> None:
        items = [make_item(100, "Первый", 0, 1000), make_item(200, "Второй", 1000, 2000)]
        response = '{"results": [{"index": 1, "shortened_text": "Неверная цель"}]}'
        assert parse_batch_response(response, [1], items) == {}


# ---------------------------------------------------------------------------
# Tests: validate_shortened_text
# ---------------------------------------------------------------------------


class TestValidateShortenedText:
    def test_valid_shortening(self) -> None:
        assert validate_shortened_text("Оригинальный длинный текст", "Короткий текст здесь") is True

    def test_too_few_words(self) -> None:
        assert validate_shortened_text("Оригинал", "Одно") is False

    def test_repeated_words_allowed(self) -> None:
        assert validate_shortened_text("Вперёд вперёд вперёд", "Вперёд, вперёд!") is True

    def test_empty_text(self) -> None:
        assert validate_shortened_text("Оригинал", "") is False

    def test_only_dots(self) -> None:
        assert validate_shortened_text("Оригинал", "...") is False

    def test_ellipsis_rejected(self) -> None:
        original = "Это очень длинный оригинальный текст который нужно сократить"
        shortened = "Это очень длинный оригинальный текст..."
        assert validate_shortened_text(original, shortened) is False

    def test_ellipsis_in_original_allowed(self) -> None:
        original = "Это текст с многоточием... и продолжением"
        shortened = "Это текст с многоточием..."
        assert validate_shortened_text(original, shortened) is True

    def test_no_ellipsis_valid(self) -> None:
        original = "Это очень длинный оригинальный текст"
        shortened = "Короткий валидный текст"
        assert validate_shortened_text(original, shortened) is True


# ---------------------------------------------------------------------------
# Tests: select_subtitles_for_shortening
# ---------------------------------------------------------------------------


class TestSelectSubtitlesForShortening:
    def test_selects_high_ratio(self, sample_items: List[Dict[str, Any]]) -> None:
        selected = select_subtitles_for_shortening(sample_items, threshold=1.5)
        assert 1 in selected
        assert 3 in selected

    def test_excludes_low_ratio(self, sample_items: List[Dict[str, Any]]) -> None:
        selected = select_subtitles_for_shortening(sample_items, threshold=1.5)
        assert 0 not in selected
        assert 2 not in selected
        assert 4 not in selected

    def test_uses_extended_when_available(self) -> None:
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, extended_mismatch_ratio=1.2),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        assert len(selected) == 0

    def test_uses_mismatch_when_no_extended(self) -> None:
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, extended_mismatch_ratio=None),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        assert len(selected) == 1

    def test_empty_items(self) -> None:
        selected = select_subtitles_for_shortening([], threshold=1.5)
        assert selected == []

    def test_unchecked_items(self) -> None:
        items = [
            make_item(1, "Текст", 0, 2000, mismatch_ratio=2.0, is_checked=False),
        ]
        selected = select_subtitles_for_shortening(items, threshold=1.5)
        assert len(selected) == 0

    def test_skill_fixture_selects_expected_effective_ratio_targets(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "skill_test" / "subs_analyzed.json"
        items = json.loads(fixture.read_text(encoding="utf-8"))
        refreshed = refresh_analysis(items, avg_chars_per_sec=13.0, threshold=1.5)
        positions = select_subtitles_for_shortening(refreshed["items"], threshold=1.5)
        assert [refreshed["items"][position]["index"] for position in positions] == [
            333, 334, 336, 341, 342, 343, 346, 347, 348, 352, 356, 358, 362,
            365, 366, 379, 380, 381, 391, 393, 394,
        ]

    def test_exclude_indices(self, sample_items: List[Dict[str, Any]]) -> None:
        selected = select_subtitles_for_shortening(
            sample_items, threshold=1.5, exclude_indices={1}
        )
        assert 1 not in selected
        assert 3 in selected

    def test_exclude_all(self, sample_items: List[Dict[str, Any]]) -> None:
        selected = select_subtitles_for_shortening(
            sample_items, threshold=1.5, exclude_indices={1, 3}
        )
        assert len(selected) == 0

    def test_calibrated_selection_uses_effective_duration_and_strict_boundary(self) -> None:
        model = CalibratedDurationModel(
            0.0, 1.0, 0.0, POST_SILENCE_MEASUREMENT_LABEL, "v", "r", 30, 2,
            "leave_one_episode_out", 0.0, 0.0, 1.0,
        )
        items = [make_item(1, "12345", 0, 1000, mismatch_ratio=100.0)]
        items[0]["analysis"]["effective_duration_sec"] = 5.0
        assert select_subtitles_for_shortening(items, 999.0, selection_policy=DurationSelectionPolicy(model)) == []
        items[0]["text"] = ["123456"]
        assert select_subtitles_for_shortening(items, 999.0, selection_policy=DurationSelectionPolicy(model)) == [0]

    def test_calibrated_selection_checks_flags_exclusions_duration_and_list_punctuation(self) -> None:
        model = CalibratedDurationModel(
            0.0, 0.1, 0.2, POST_SILENCE_MEASUREMENT_LABEL, "v", "r", 30, 2,
            "leave_one_episode_out", 0.0, 0.0, 1.0,
        )
        policy = DurationSelectionPolicy(model)
        selected = [
            make_item(1, ["long", "text!"], 0, 1000, mismatch_ratio=0.1),
            make_item(2, "long text", 0, 1000, mismatch_ratio=99.0, is_checked=False),
            make_item(3, "long text", 0, 0, mismatch_ratio=99.0),
        ]
        selected[0]["analysis"]["effective_duration_sec"] = 1.0
        selected[1]["analysis"]["effective_duration_sec"] = 1.0
        selected[2]["analysis"]["effective_duration_sec"] = 0.0
        assert select_subtitles_for_shortening(selected, 0.0, {0}, policy) == []
        assert select_subtitles_for_shortening(selected, 0.0, selection_policy=policy) == [0]


# ---------------------------------------------------------------------------
# Tests: apply_shortening
# ---------------------------------------------------------------------------


class TestApplyShortening:
    def test_applies_shortening(self, sample_items: List[Dict[str, Any]]) -> None:
        results = [
            ShortenResult(index=1, original_text="old", shortened_text="new", success=True),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[1]["text"] == ["new"]
        assert sample_items[1]["text"] != ["new"]

    def test_keeps_original_on_failure(self, sample_items: List[Dict[str, Any]]) -> None:
        results = [
            ShortenResult(index=1, original_text="old", shortened_text="old", success=False),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[1]["text"] == sample_items[1]["text"]

    def test_multiple_results(self, sample_items: List[Dict[str, Any]]) -> None:
        results = [
            ShortenResult(index=0, original_text="a", shortened_text="x", success=True),
            ShortenResult(index=2, original_text="b", shortened_text="y", success=True),
        ]
        new_items = apply_shortening(sample_items, results)
        assert new_items[0]["text"] == ["x"]
        assert new_items[2]["text"] == ["y"]
        assert new_items[1]["text"] == sample_items[1]["text"]

    def test_preserves_list_format(self) -> None:
        items = [{"text": ["line1", "line2"], "index": 1}]
        results = [ShortenResult(index=0, original_text="line1 line2", shortened_text="short", success=True)]
        new_items = apply_shortening(items, results)
        assert isinstance(new_items[0]["text"], list)
        assert new_items[0]["text"] == ["short"]

    def test_preserves_string_format(self) -> None:
        items = [{"text": "single line", "index": 1}]
        results = [ShortenResult(index=0, original_text="single line", shortened_text="short", success=True)]
        new_items = apply_shortening(items, results)
        assert isinstance(new_items[0]["text"], str)
        assert new_items[0]["text"] == "short"


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing (deepseek)
# ---------------------------------------------------------------------------


class TestCLIDeepseek:
    def test_default_args(self) -> None:
        import argparse
        from utils.shorten_helpers import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--concurrency", type=int, default=5)
        parser.add_argument("--api-key")
        parser.add_argument("--base-url", default="https://api.deepseek.com")

        args = parser.parse_args(["test.json"])
        assert args.model == "deepseek-v4-flash"
        assert args.threshold == 1.5
        assert args.max_iterations == 3
        assert args.avg_chars_per_sec == 13.0
        assert args.concurrency == 5
        assert args.batch_size == 5
        assert args.stuck_threshold == 5
        assert args.early_stop_patience is None


def test_async_fallback_client_disables_sdk_retries(monkeypatch: Any) -> None:
    created: list[dict[str, Any]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)
            usage = SimpleNamespace()

            async def create(**call_kwargs: Any) -> Any:
                del call_kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"results": []}'))],
                    usage=usage,
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    result = asyncio.run(shorten_via_deepseek_async("model", "system", "user", "key"))
    assert result == '{"results": []}'
    assert created[0]["max_retries"] == 0


def test_owned_flash_shared_client_disables_sdk_retries(monkeypatch: Any) -> None:
    created: list[dict[str, Any]] = []
    closed: list[bool] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace())

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    result = asyncio.run(shorten_subtitles_deepseek(
        [], [], "model", 3, 3, "key", "https://api.deepseek.com", 3
    ))
    assert result == []
    assert created == [{
        "api_key": "key",
        "base_url": "https://api.deepseek.com",
        "timeout": 60.0,
        "max_retries": 0,
    }]
    assert closed == [True]


def test_budget_and_preserved_zero_timestamp_analysis() -> None:
    item = make_item(1, "Очень длинный исходный текст здесь", 0, 0)
    item["analysis"].update({"duration_sec": 2.0, "mismatch_ratio": 2.0})
    budget = calculate_budget(item, target_ratio=1.0, avg_chars_per_sec=10.0)
    assert budget.effective_duration_sec == 2.0
    assert budget.max_chars == 20
    refreshed = refresh_analysis([item], avg_chars_per_sec=10.0, threshold=1.5)
    assert refreshed["items"][0]["analysis"]["duration_sec"] == 2.0
    assert refreshed["items"][0]["analysis"]["mismatch_ratio"] != float("inf")


def test_fixture_style_effective_duration_stays_stable_after_text_change() -> None:
    item = make_item(1, "а" * 52, 0, 0)
    item["analysis"].update({
        "duration_sec": 2.0, "estimated_sec": 4.0, "mismatch_ratio": 2.0,
        "extended_duration_sec": None, "extended_mismatch_ratio": 1.6,
        "is_checked": True,
    })
    first = refresh_analysis([item], 13.0, 1.5)
    analysis = first["items"][0]["analysis"]
    assert analysis["effective_duration_sec"] == pytest.approx(2.5)
    assert analysis["extended_mismatch_ratio"] == pytest.approx(1.6)
    first["items"][0]["text"] = ["а" * 39]
    second = refresh_analysis(first["items"], 13.0, 1.5)
    updated = second["items"][0]["analysis"]
    assert updated["effective_duration_sec"] == pytest.approx(2.5)
    assert updated["extended_duration_sec"] == pytest.approx(2.5)
    assert updated["used_gap_sec"] == pytest.approx(0.5)
    assert updated["is_short_segment"] is False
    assert updated["extended_mismatch_ratio"] == pytest.approx(1.2)
    assert updated["is_critical"] is False
    assert select_subtitles_for_shortening(second["items"], 1.5) == []
    assert calculate_budget(second["items"][0], 1.0, 13.0).max_chars == 32
    assert second["checked_count"] == 1


def test_refresh_analysis_handles_mixed_timing_per_item() -> None:
    timed = make_item(1, "Текст для обычного тайминга", 0, 2000)
    saved = make_item(2, "Сохраненный текст для анализа", 0, 0)
    saved["analysis"].update({"duration_sec": 1.5, "estimated_sec": 3.0, "mismatch_ratio": 2.0, "is_checked": True})
    result = refresh_analysis([timed, saved], 13.0, 1.5)
    assert result["checked_count"] == 2
    assert result["items"][1]["analysis"]["mismatch_ratio"] != float("inf")


def test_hydrate_context_timing_uses_real_neighbors_and_stays_stable() -> None:
    text = "Очень длинный текст для субтитра"
    sparse = [make_item(2, text, 1000, 2000), make_item(4, "Короткая реплика", 100000, 101000)]
    context = [
        make_item(1, "До", 0, 1000),
        make_item(2, text, 1000, 2000),
        make_item(3, "После", 2200, 3200),
        make_item(4, "Короткая реплика", 3200, 4200),
    ]
    original_context = copy.deepcopy(context)
    old = refresh_analysis(copy.deepcopy(sparse), 13.0, 1.5)
    hydrated = hydrate_context_timing(sparse, context, 13.0, 1.5)
    first = refresh_analysis(hydrated, 13.0, 1.5, preserve_saved_timing=True)

    assert old["items"][0]["analysis"]["effective_duration_sec"] > 90.0
    assert first["items"][0]["analysis"]["effective_duration_sec"] == pytest.approx(1.0)
    assert first["items"][0]["analysis"]["is_critical"] is True
    assert select_subtitles_for_shortening(first["items"], 1.5) == [0]
    assert context == original_context

    first["items"][0]["text"] = ["Короткий текст"]
    second = refresh_analysis(first["items"], 13.0, 1.5, preserve_saved_timing=True)
    assert second["items"][0]["analysis"]["effective_duration_sec"] == pytest.approx(1.0)


def test_strict_budget_numbers_and_negation() -> None:
    from utils.shorten_helpers import validation_error
    original = "Я не куплю 15 очень дорогих билетов сегодня"
    assert validation_error(original, "Не куплю 15 билетов", 3, 20) is None
    assert validation_error(original, "Куплю 15 билетов", 3, 20) == "negation_missing"
    assert validation_error(original, "Не куплю билеты", 3, 20) == "number_missing"
    assert validation_error("Я без дорогого билета сегодня", "Я лишён билета", 3, 20) is None


def test_target_min_words_relaxes_three_word_original() -> None:
    assert target_min_words("Стой!", 3) == 1
    assert target_min_words("Уже поздно", 3) == 2
    assert target_min_words("Всё, лезвию каюк.", 3) == 2
    assert target_min_words("Это достаточно длинная исходная реплика", 3) == 3


def test_dynamic_tokens_and_usage_cost() -> None:
    from utils.shorten_subtitles_deepseek import dynamic_max_tokens, usage_summary
    assert dynamic_max_tokens([10]) == 192
    assert dynamic_max_tokens([100, 100]) == 392
    assert dynamic_max_tokens([5000]) == 4096
    summary = usage_summary({"prompt_cache_miss_tokens": 1_000_000, "prompt_cache_hit_tokens": 1_000_000, "completion_tokens": 1_000_000})
    assert summary["estimated_cost_usd"] == pytest.approx(0.4228)
    assert summary["pricing_model"] == "deepseek-v4-flash"


def test_default_deepseek_request_disables_thinking() -> None:
    import asyncio
    from types import SimpleNamespace
    from utils.shorten_subtitles_deepseek import shorten_via_deepseek_async

    class Completions:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] = {}
        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"results":[]}'))], usage=SimpleNamespace())

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    asyncio.run(shorten_via_deepseek_async("m", "s", "u", "secret", client=client))
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert "temperature" not in completions.kwargs


def test_retryable_api_error_classification() -> None:
    from utils.shorten_subtitles_deepseek import _is_retryable_api_error

    class StatusError(Exception):
        def __init__(self, status: int) -> None:
            self.status_code = status
    APIConnectionError = type("APIConnectionError", (Exception,), {})
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    assert _is_retryable_api_error(StatusError(429))
    assert _is_retryable_api_error(StatusError(500))
    assert _is_retryable_api_error(APIConnectionError())
    assert _is_retryable_api_error(APITimeoutError())
    assert not _is_retryable_api_error(StatusError(400))


def test_targeted_retry_only_resends_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import utils.shorten_subtitles_deepseek as module
    items = [make_item(1, "Первый очень длинный исходный текст", 0, 2000), make_item(2, "Второй очень длинный исходный текст", 2000, 4000)]
    prompts: List[str] = []

    async def fake_call(*args: Any, **kwargs: Any) -> str:
        prompts.append(args[2])
        if len(prompts) == 1:
            return '{"results":[{"index":1,"shortened_text":"Первый короткий текст"}]}'
        return '{"results":[{"index":2,"shortened_text":"Второй короткий текст"}]}'

    monkeypatch.setattr(module, "shorten_via_deepseek_async", fake_call)
    results = asyncio.run(module.shorten_subtitles_deepseek(
        items, [0, 1], "m", 0, 3, "key", "url", 1, batch_size=2,
        avg_chars_per_sec=13.0, target_ratio=1.5, client=object(),
    ))
    assert all(result.success for result in results)
    second_targets = json.loads(prompts[1].split("Цели и budgets:", 1)[1].split("\n", 1)[0])
    assert [target["index"] for target in second_targets] == [2]
    first_targets = json.loads(prompts[0].split("Цели и budgets:", 1)[1].split("\n", 1)[0])
    assert second_targets[0]["max_chars"] == first_targets[1]["max_chars"]


def test_unchanged_flash_response_is_terminal_safe_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import utils.shorten_subtitles_deepseek as module

    original = "Первый очень длинный исходный текст"
    items = [make_item(1, original, 0, 2000)]
    calls = 0

    async def fake_call(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"results": [{"index": 1, "shortened_text": original}]})

    monkeypatch.setattr(module, "shorten_via_deepseek_async", fake_call)
    results = asyncio.run(module.shorten_subtitles_deepseek(
        items, [0], "m", 0, 3, "key", "url", 1, batch_size=1,
        avg_chars_per_sec=13.0, target_ratio=1.5, client=object(),
    ))
    assert calls == 1
    assert results[0].success is False
    assert results[0].error == "safe_deferred_unchanged"


def test_iterative_loop_actual_adapter_call_shape(tmp_path: Path) -> None:
    from utils.shorten_helpers import run_iterative_shortening
    calls: List[tuple[float, float]] = []
    item = make_item(1, "Очень длинный исходный текст для теста", 0, 0)
    item["analysis"].update({"duration_sec": 1.0, "estimated_sec": 3.0, "mismatch_ratio": 3.0, "is_checked": True})

    def fake_shorten(items: List[Dict[str, Any]], targets: List[int], model: str,
                     context: int, min_words: int, effort: Any, strategy: str,
                     batch: int, cps: float, ratio: float) -> List[ShortenResult]:
        calls.append((cps, ratio))
        original = " ".join(items[0]["text"])
        return [ShortenResult(0, original, "Короткий текст здесь", True)]

    result = run_iterative_shortening(item and [item], fake_shorten, "m", 1.5, 1, 1, 3, tmp_path, "fixture", avg_chars_per_sec=13.0)
    assert calls == [(13.0, 1.5)]
    assert result[0]["analysis"]["mismatch_ratio"] < 3.0


def test_legacy_iterative_selection_matches_refresh_critical_items(tmp_path: Path) -> None:
    from utils.shorten_helpers import run_iterative_shortening

    items = [make_item(1, "short", 0, 1000, mismatch_ratio=1.0),
             make_item(2, "long text with extended timing", 1000, 2000, mismatch_ratio=1.0,
                       extended_mismatch_ratio=2.0),
             make_item(3, "unchecked long text", 2000, 3000, mismatch_ratio=3.0, is_checked=False)]
    expected = refresh_analysis(copy.deepcopy(items), 13.0, 1.5)
    expected_targets = select_subtitles_for_shortening(expected["items"], 1.5)
    captured: List[List[int]] = []

    def fake_shorten(current: List[Dict[str, Any]], targets: List[int], *args: Any) -> List[ShortenResult]:
        captured.append(targets[:])
        return [ShortenResult(index, " ".join(current[index]["text"]),
                              " ".join(current[index]["text"]), False) for index in targets]

    run_iterative_shortening(items, fake_shorten, "m", 1.5, 1, 1, 3,
                             tmp_path, "legacy")
    assert captured == [expected_targets]
    assert len(expected_targets) == len(expected["critical_items"])


def test_iterative_policy_selection_does_not_replace_legacy_budget_ratio(tmp_path: Path) -> None:
    from utils.shorten_helpers import run_iterative_shortening

    item = make_item(1, "Очень длинный текст для бюджета", 0, 1000, mismatch_ratio=0.1)
    item["analysis"].update({"effective_duration_sec": 1.0, "is_checked": True})
    policy = DurationSelectionPolicy(CalibratedDurationModel(
        0.0, 0.05, 0.0, POST_SILENCE_MEASUREMENT_LABEL, "v", "r", 30, 2,
        "leave_one_episode_out", 0.0, 0.0, 1.0,
    ))
    calls: List[tuple[List[int], float]] = []

    def fake_shorten(items: List[Dict[str, Any]], targets: List[int], model: str,
                     context: int, min_words: int, effort: Any, strategy: str,
                     batch: int, cps: float, ratio: float) -> List[ShortenResult]:
        calls.append((targets[:], ratio))
        original = " ".join(items[0]["text"])
        return [ShortenResult(0, original, original, True)]

    result = run_iterative_shortening([item], fake_shorten, "m", 1.5, 1, 2, 3,
                                      tmp_path, "policy", selection_policy=policy)
    assert calls == [([0], 1.5), ([0], 1.5)]
    assert result[0]["text"] == item["text"]


def _write_test_profile(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": 1, "profile_type": "calibrated_post_silence_duration",
        "measurement": POST_SILENCE_MEASUREMENT_LABEL, "voice": "v", "rate": "r",
        "coefficients": {"intercept_sec": 0.1, "seconds_per_budget_char": 0.01,
                         "seconds_per_punctuation": 0.01},
        "training": {"sample_count": 30, "episode_count": 2},
        "validation": {"method": "leave_one_episode_out", "mae_sec": 0.1,
                        "rmse_sec": 0.2, "r_squared": 0.8},
    }), encoding="utf-8")


def test_deepseek_parser_duration_flags() -> None:
    defaults = build_parser().parse_args(["input.json"])
    assert defaults.duration_profile is None and defaults.duration_fit_ratio == 1.0
    args = build_parser().parse_args(["input.json", "--duration-profile", "profile.json",
                                      "--duration-fit-ratio", "1.25"])
    assert args.duration_profile == Path("profile.json")
    assert args.duration_fit_ratio == 1.25


@pytest.mark.parametrize("argv", [["input.json", "--duration-fit-ratio", "0"],
                                   ["input.json", "--duration-fit-ratio", "nan"]])
def test_deepseek_invalid_duration_configuration_precedes_api(monkeypatch: pytest.MonkeyPatch,
                                                               argv: List[str]) -> None:
    monkeypatch.setattr("utils.shorten_subtitles_deepseek.load_api_key",
                        lambda *args: pytest.fail("API key lookup must not run"))
    assert deepseek_main(argv) == 2


@pytest.mark.parametrize("profile_name,profile_text", [("missing.json", None),
                                                        ("malformed.json", "not json"),
                                                        ("rejected.json", json.dumps({"schema_version": 1}))])
def test_deepseek_invalid_profile_precedes_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               profile_name: str, profile_text: str | None) -> None:
    profile_path = tmp_path / profile_name
    if profile_text is not None:
        profile_path.write_text(profile_text, encoding="utf-8")
    monkeypatch.setattr("utils.shorten_subtitles_deepseek.load_api_key",
                        lambda *args: pytest.fail("API key lookup must not run"))
    assert deepseek_main(["input.json", "--duration-profile", str(profile_path)]) == 2


def test_deepseek_valid_profile_metadata_and_policy_propagation(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([make_item(1, "A long subtitle", 0, 2000, mismatch_ratio=2.0)]), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    _write_test_profile(profile_path)
    captured: List[Any] = []

    def fake_run(**kwargs: Any) -> None:
        captured.append(kwargs)

    monkeypatch.setattr("utils.shorten_subtitles_deepseek.run_iterative_shortening", fake_run)
    monkeypatch.setattr("utils.shorten_subtitles_deepseek.load_api_key", lambda *args: "secret")
    assert deepseek_main([str(input_path), "--duration-profile", str(profile_path),
                          "--output-dir", str(tmp_path / "out")]) == 0
    assert captured[0]["shorten_func"] is not None
    assert "secret" not in repr(captured[0]["shorten_func"])
    assert str(profile_path) not in repr(captured[0])
    assert isinstance(captured[0]["selection_policy"], DurationSelectionPolicy)
    usage = json.loads((tmp_path / "out" / "input_final.usage.json").read_text(encoding="utf-8"))
    assert usage["selection_mode"] == "calibrated_post_silence_duration"
    assert usage["fit_ratio"] == 1.0
    assert usage["duration_profile"]["voice"] == "v"
    assert usage["duration_profile"]["coefficients"]["intercept_sec"] == 0.1
    assert usage["duration_profile"]["validation"]["method"] == "leave_one_episode_out"
    assert "secret" not in json.dumps(usage)
    assert str(profile_path) not in json.dumps(usage)


def test_deepseek_legacy_usage_mode_without_profile(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("utils.shorten_subtitles_deepseek.run_iterative_shortening", lambda **kwargs: None)
    monkeypatch.setattr("utils.shorten_subtitles_deepseek.load_api_key", lambda *args: "secret")
    assert deepseek_main([str(input_path), "--output-dir", str(tmp_path / "out")]) == 0
    usage = json.loads((tmp_path / "out" / "input_final.usage.json").read_text(encoding="utf-8"))
    assert usage["selection_mode"] == "legacy_mismatch_ratio"
    assert "secret" not in json.dumps(usage)
    assert "duration_profile" not in usage


# ---------------------------------------------------------------------------
# Tests: OpenCode helpers
# ---------------------------------------------------------------------------


class TestOpenCodeHelpers:
    def test_parse_model_string_with_provider(self) -> None:
        from utils.shorten_subtitles_opencode import _parse_model_string

        result = _parse_model_string("opencode-go/deepseek-v4-flash")
        assert result == {"providerID": "opencode-go", "modelID": "deepseek-v4-flash"}

    def test_parse_model_string_deepseek(self) -> None:
        from utils.shorten_subtitles_opencode import _parse_model_string

        result = _parse_model_string("deepseek-v4-flash")
        assert result == {"providerID": "opencode-go", "modelID": "deepseek-v4-flash"}

    def test_parse_model_string_other(self) -> None:
        from utils.shorten_subtitles_opencode import _parse_model_string

        result = _parse_model_string("mimo-v2.5")
        assert result == {"providerID": "opencode-go", "modelID": "mimo-v2.5"}
