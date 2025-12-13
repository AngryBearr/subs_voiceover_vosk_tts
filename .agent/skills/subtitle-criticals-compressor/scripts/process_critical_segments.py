import json
from typing import Any, Dict, List

from utils.analyze_text import (
    analyze_subtitles,
    join_text_lines,
    DEFAULT_AVG_CHARS_PER_SEC,
    DEFAULT_MAX_MISMATCH_RATIO,
)


def build_context(items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in items:
        text = item.get("text", [])
        if isinstance(text, list):
            parts.append(" ".join(str(t) for t in text))
        else:
            parts.append(str(text))
    return "\n".join(p for p in parts if p.strip())


def build_local_context(items: List[Dict[str, Any]], index: int, window: int = 2) -> str:
    """Построить локальный контекст вокруг сегмента с данным индексом.

    Берём до `window` сегментов до и после, включая сам сегмент.
    Используется только для подачи агенту (не влияет на основной JSON).
    """
    pos = None
    for i, item in enumerate(items):
        if item.get("index") == index:
            pos = i
            break
    if pos is None:
        return ""

    start = max(0, pos - window)
    end = min(len(items), pos + window + 1)
    slice_items = items[start:end]
    return build_context(slice_items)


def compress_text_with_agent(context: str, segment_text: str) -> str:
    """Заглушка: реальное сжатие делает агент по промпту.

    Вызов агента происходит на уровне Skill, а не внутри скрипта.
    Этот хук оставлен для удобства, но в рантайме будет перезаписан
    логикой самого скилла.
    """
    raise NotImplementedError("This function must be handled by the agent logic")


def recompute_analysis_for_item(
    item: Dict[str, Any],
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> None:
    """Пересчитать analysis только для одного элемента, используя
    ту же формулу, что и в analyze_subtitles.
    """
    # Оборачиваем в список и прогоняем через analyze_subtitles,
    # чтобы не дублировать формулы.
    tmp_items = [
        {
            **item,
            "analysis": item.get("analysis", {}),
        }
    ]
    result = analyze_subtitles(tmp_items, avg_chars_per_sec, max_mismatch_ratio)
    item.update(result["items"][0])


def process_json_items(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> List[Dict[str, Any]]:
    """Применить уже полученные от агента сжатые тексты к критичным сегментам
    и пересчитать analysis.

    compressed_texts: mapping index -> new_text
    """
    for item in items:
        analysis = item.get("analysis") or {}
        if not analysis.get("is_critical"):
            continue

        idx = item.get("index")
        if idx is None or idx not in compressed_texts:
            continue

        new_text = compressed_texts[idx]
        # кладём как список из одной строки, по конвенции проекта
        item["text"] = [new_text]

        # пересчитываем analysis с учётом нового текста
        recompute_analysis_for_item(
            item,
            avg_chars_per_sec=avg_chars_per_sec,
            max_mismatch_ratio=max_mismatch_ratio,
        )

    return items


def build_agent_view(
    items: List[Dict[str, Any]],
    window: int = 2,
    critical_ratio_threshold: float = 1.7,
) -> Dict[str, Any]:
    """Построить компактное представление для агента.

    Формат:
    {
      "context": "<общий текст>",
      "segments": [
        {
          "context": "локальный контекст вокруг сегмента",
          "index": 1,
          "text": "склеенный текст сегмента",
          "duration_sec": 2.28,
          "estimated_sec": 3.92
        },
        ...
      ]
    }

    В список попадают только действительно критичные сегменты с
    эффективным ratio > critical_ratio_threshold.
    """
    segments: List[Dict[str, Any]] = []

    for item in items:
        analysis = item.get("analysis") or {}
        if not analysis.get("is_critical"):
            continue

        # Используем extended_mismatch_ratio если есть, иначе обычный
        ratio = analysis.get("extended_mismatch_ratio")
        if ratio is None:
            ratio = analysis.get("mismatch_ratio")

        if ratio is None:
            continue
        if ratio <= critical_ratio_threshold:
            continue

        idx = item.get("index")
        if idx is None:
            continue

        segment_text = join_text_lines(item.get("text", ""))

        segments.append(
            {
                "index": idx,
                "text": segment_text,
                "duration_sec": analysis.get("duration_sec"),
                "estimated_sec": analysis.get("estimated_sec"),
            }
        )

    return {
        "segments": segments,
    }
