import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

# Локальная копия констант и логики анализа, чтобы скилл был полностью
# самодостаточным и не зависел от внешних модулей репозитория.

DEFAULT_AVG_CHARS_PER_SEC: float = 13.0
DEFAULT_MAX_MISMATCH_RATIO: float = 1.3
MIN_DURATION_SEC: float = 1.0
MIN_GAP_BEFORE_NEXT_SEC: float = 0.3
MIN_WORDS_SHORT_SEGMENT: int = 3

# Константы контроля уровня сокращения
# УБРАНО: ограничение MAX_SHORTENING_PERCENT = 30% — теперь модель может сокращать
# на любое значение, главное чтобы итоговый ratio <= TARGET_MISMATCH_RATIO
TARGET_MISMATCH_RATIO: float = 1.5  # Целевой порог ratio после сокращения
MIN_MISMATCH_RATIO_FOR_SHORTENING: float = 1.5  # Минимальный ratio для попадания в список на сокращение
MAX_COMPRESSION_ATTEMPTS: int = 5  # Максимум попыток сокращения


def join_text_lines(text_field: Any) -> str:
    """Normalize subtitle text field to a single string.

    This helper accepts both formats that commonly appear in subtitle JSON:
    - a list of text lines (e.g. ["line 1", "line 2"]),
    - a single string.

    It always returns one plain string with lines joined by a single space.

    Args:
        text_field: Value from the ``text`` field of a subtitle item.

    Returns:
        A single string containing the concatenated text.
    """
    if isinstance(text_field, list):
        return " ".join(str(t) for t in text_field)
    return str(text_field)


def _analyze_single_item(
    item: Dict[str, Any],
    idx: int,
    items: List[Dict[str, Any]],
    avg_chars_per_sec: float,
    max_mismatch_ratio: float,
) -> Dict[str, Any]:
    """Compute timing analysis for a single subtitle item.

    The function reproduces locally the timing logic used в анализе примеров
    из assets/subs_analyzed.json: оценивает длительность текста, сравнивает
    её с доступным слотом по времени и, при необходимости, пытается учесть
    свободный зазор до следующего субтитра (extended_* метрики).

    Args:
        item: Subtitle item with at least ``start``, ``end`` and ``text``.
        idx: Zero-based position of the item in the full ``items`` list.
        items: Full list of subtitle items (нужен для доступа к соседям).
        avg_chars_per_sec: Assumed average TTS speed (characters per second).
        max_mismatch_ratio: Threshold for considering a segment critical.

    Returns:
        Dict with computed ``analysis`` fields, включая duration, estimated
        duration, mismatch ratios и флаги критичности.
    """

    start = item.get("start")
    end = item.get("end")
    if start is None or end is None:
        return {
            "duration_sec": None,
            "estimated_sec": None,
            "mismatch_ratio": None,
            "diff_sec": None,
            "extended_duration_sec": None,
            "extended_mismatch_ratio": None,
            "used_gap_sec": None,
            "words_count": None,
            "is_short_segment": False,
            "is_checked": False,
            "is_critical": False,
        }

    duration_ms = end - start
    duration_sec = duration_ms / 1000.0

    text_value = join_text_lines(item.get("text", ""))
    words_count = len(text_value.split()) if text_value else 0
    is_short_segment = duration_sec < MIN_DURATION_SEC

    should_check = True
    if is_short_segment and words_count < MIN_WORDS_SHORT_SEGMENT:
        should_check = False

    if not should_check:
        return {
            "duration_sec": duration_sec,
            "estimated_sec": None,
            "mismatch_ratio": None,
            "diff_sec": None,
            "extended_duration_sec": None,
            "extended_mismatch_ratio": None,
            "used_gap_sec": None,
            "words_count": words_count,
            "is_short_segment": is_short_segment,
            "is_checked": False,
            "is_critical": False,
        }

    estimated_sec = (
        len(text_value) / avg_chars_per_sec if avg_chars_per_sec > 0 else 0.0
    )

    if duration_sec > 0:
        mismatch_ratio = estimated_sec / duration_sec
    else:
        mismatch_ratio = float("inf")

    diff_sec = estimated_sec - duration_sec

    # Try to extend into the gap before next subtitle if needed
    extended_duration_sec: float | None = None
    extended_mismatch_ratio: float | None = None
    used_gap_sec: float | None = None

    if mismatch_ratio > max_mismatch_ratio and idx + 1 < len(items):
        next_item = items[idx + 1]
        next_start = next_item.get("start")
        if next_start is not None:
            gap_ms = next_start - end
            gap_sec = gap_ms / 1000.0
            extra_sec = gap_sec - MIN_GAP_BEFORE_NEXT_SEC
            if extra_sec > 0:
                extended_duration_sec = duration_sec + extra_sec
                used_gap_sec = extra_sec
                if extended_duration_sec > 0:
                    extended_mismatch_ratio = estimated_sec / extended_duration_sec

    effective_ratio = (
        extended_mismatch_ratio if extended_mismatch_ratio is not None else mismatch_ratio
    )
    is_critical = effective_ratio > max_mismatch_ratio

    return {
        "duration_sec": duration_sec,
        "estimated_sec": estimated_sec,
        "mismatch_ratio": mismatch_ratio,
        "diff_sec": diff_sec,
        "extended_duration_sec": extended_duration_sec,
        "extended_mismatch_ratio": extended_mismatch_ratio,
        "used_gap_sec": used_gap_sec,
        "words_count": words_count,
        "is_short_segment": is_short_segment,
        "is_checked": True,
        "is_critical": is_critical,
    }


def analyze_subtitles(
    items: List[Dict[str, Any]],
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> Dict[str, Any]:
    """Run timing analysis for a list of subtitle items.

    Для каждого элемента в ``items`` рассчитывается поле ``analysis`` по тем
    же принципам, что и в примере assets/subs_analyzed.json: длительность
    текстовой реплики, оценочная длительность на основе скорости чтения,
    mismatch‑ratio, возможное расширение слота за счёт зазора до следующего
    субтитра и финальный флаг ``is_critical``.

    Args:
        items: Список JSON‑объектов субтитров (модифицируется на месте).
        avg_chars_per_sec: Средняя скорость чтения в символах/секунду.
        max_mismatch_ratio: Порог, выше которого сегмент считается критичным.

    Returns:
        Словарь с ключами:
            items: исходный список с обновлённым полем ``analysis``;
            checked_count: количество реально проанализированных сегментов;
            critical_items: список ссылок на критичные элементы.
    """

    critical_items: List[Dict[str, Any]] = []
    checked_count = 0

    for idx, item in enumerate(items):
        analysis = _analyze_single_item(
            item,
            idx,
            items,
            avg_chars_per_sec=avg_chars_per_sec,
            max_mismatch_ratio=max_mismatch_ratio,
        )
        item["analysis"] = analysis
        if analysis.get("is_checked"):
            checked_count += 1
        if analysis.get("is_critical"):
            critical_items.append(item)

    return {
        "items": items,
        "checked_count": checked_count,
        "critical_items": critical_items,
    }


def load_items(path: str) -> List[Dict[str, Any]]:
    """Load subtitle items list from JSON file.

    Args:
        path: Path to JSON file with an array of subtitle items.

    Returns:
        Parsed list of items (list[dict]).
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_items(path: str, data: List[Dict[str, Any]]) -> None:
    """Save updated subtitle items list to JSON file.

    Args:
        path: Destination file path.
        data: List of subtitle items to serialize.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_combined_mismatch_ratio(analysis: Dict[str, Any]) -> float | None:
    """Return the best-case mismatch ratio for compression decisions.

    Для каждого сегмента мы храним две оценки несоответствия:
    - ``mismatch_ratio`` — только текст против длительности слота;
    - ``extended_mismatch_ratio`` — та же оценка, но с учётом возможного
      расширения слота за счёт зазора до следующего субтитра.

    Для принятия решений о сжатии используется минимальное из двух значений,
    поскольку оно отражает лучший достижимый вариант при учёте зазора.

    Args:
        analysis: Поле ``analysis`` одного сегмента.

    Returns:
        Число (float) или None, если обе метрики отсутствуют.
    """

    mismatch_ratio = analysis.get("mismatch_ratio")
    extended_ratio = analysis.get("extended_mismatch_ratio")

    if mismatch_ratio is None and extended_ratio is None:
        return None
    if mismatch_ratio is None:
        return extended_ratio
    if extended_ratio is None:
        return mismatch_ratio
    return min(mismatch_ratio, extended_ratio)


def ensure_shortening_meta(items: List[Dict[str, Any]]) -> None:
    """Ensure `shortened` field exists on all items.

    Args:
        items: List of subtitle items.
    """
    for item in items:
        if "shortened" not in item:
            item["shortened"] = False


def build_context_window(
    items: List[Dict[str, Any]],
    pos: int,
    window: int = 3,
) -> str:
    """Build local text context around a given position.

    Контекст строится как конкатенация текстов соседних сегментов в диапазоне
    ``[pos-window, pos+window]`` (границы обрезаются по краям списка). Тексты
    собираются в одну строку, чтобы экономить место в payload для агента.

    Args:
        items: Полный список сегментов.
        pos: Позиция целевого сегмента в списке (0-based).
        window: Количество соседей до и после, включаемых в контекст.

    Returns:
        Строка с локальным контекстом вокруг выбранного сегмента.
    """

    start = max(0, pos - window)
    end = min(len(items), pos + window + 1)
    parts: List[str] = []
    for i in range(start, end):
        text = join_text_lines(items[i].get("text", ""))
        if text.strip():
            parts.append(text)
    return " ".join(parts)


@dataclass
class AgentSegment:
    context: str
    index: int
    text: str
    duration_sec: float | None
    estimated_sec: float | None
    mismatch_ratio: float | None
    extended_mismatch_ratio: float | None

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass instance to plain dict for JSON serialization."""
        return {
            "context": self.context,
            "index": self.index,
            "text": self.text,
            "duration_sec": self.duration_sec,
            "estimated_sec": self.estimated_sec,
            "mismatch_ratio": self.mismatch_ratio,
            "extended_mismatch_ratio": self.extended_mismatch_ratio,
        }


def collect_segments_for_agent(
    items: List[Dict[str, Any]],
    window: int = 3,
    min_ratio: float = MIN_MISMATCH_RATIO_FOR_SHORTENING,
) -> Dict[int, AgentSegment]:
    """Select segments that require compression based on mismatch ratio.

    Сегмент попадает в выборку, если:
    - mismatch_ratio >= min_ratio, ИЛИ
    - extended_mismatch_ratio >= min_ratio (если есть)
    
    Флаг is_critical НЕ используется — проверяем только значения ratio.
    Это гарантирует, что все субтитры с высоким ratio будут обработаны.

    Args:
        items: Полный список сегментов с полем ``analysis``.
        window: Размер оконного контекста (количество соседей до/после).
        min_ratio: Минимальное значение ratio для попадания в выборку
            (по умолчанию 1.5).

    Returns:
        Словарь ``index -> AgentSegment`` для всех выбранных сегментов.
    """

    segments: Dict[int, AgentSegment] = {}
    for pos, item in enumerate(items):
        analysis = item.get("analysis") or {}
        
        # Проверяем mismatch_ratio и extended_mismatch_ratio независимо
        mismatch_ratio = analysis.get("mismatch_ratio")
        extended_ratio = analysis.get("extended_mismatch_ratio")
        
        # Сегмент нуждается в сокращении если ЛЮБОЙ из ratio >= min_ratio
        needs_shortening = False
        if mismatch_ratio is not None and mismatch_ratio >= min_ratio:
            needs_shortening = True
        if extended_ratio is not None and extended_ratio >= min_ratio:
            needs_shortening = True
        
        if not needs_shortening:
            continue
        
        # Используем combined ratio для отображения (минимум из двух)
        combined = get_combined_mismatch_ratio(analysis)

        idx = item.get("index")
        if idx is None:
            continue

        text = join_text_lines(item.get("text", ""))
        context = build_context_window(items, pos, window=window)

        duration_sec = analysis.get("duration_sec")
        estimated_sec = analysis.get("estimated_sec")
        extended_ratio = analysis.get("extended_mismatch_ratio")

        seg = AgentSegment(
            context=context,
            index=idx,
            text=text,
            duration_sec=duration_sec,
            estimated_sec=estimated_sec,
            mismatch_ratio=combined,
            extended_mismatch_ratio=extended_ratio,
        )
        segments[idx] = seg

    return segments


def build_agent_payload(
    items: List[Dict[str, Any]],
    window: int = 3,
    min_ratio: float = 1.5,
) -> str:
    """Build compact JSON payload for the compression agent.

    Функция подготавливает минимальный по размеру JSON‑объект, который
    содержит только критичные сегменты, выбранные collect_segments_for_agent,
    а также служебные поля управления силой сжатия. JSON сериализуется без
    лишних пробелов для экономии контекста.

    Args:
        items: Список сегментов с заполненным ``analysis``.
        window: Количество соседей до/после для текстового контекста.
        min_ratio: Порог комбинированного mismatch_ratio для отбора.

    Returns:
        Строка JSON, готовая к вставке в промпт модели.
    """

    ensure_shortening_meta(items)
    segments = collect_segments_for_agent(items, window=window, min_ratio=min_ratio)
    payload = {"segments": [seg.to_dict() for seg in segments.values()]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_agent_response(raw: str) -> Dict[int, str]:
    """Parse model response into mapping index -> new text.

    Ожидается, что модель вернёт один JSON‑объект без обёрток вида::

        {"1": "новый текст", "8": "..."}

    Ключи приводятся к int, значения — к строкам.

    Args:
        raw: Сырый JSON‑текст из ответа модели.

    Returns:
        Словарь ``index -> compressed_text``.
    """

    data = json.loads(raw)
    result: Dict[int, str] = {}
    for k, v in data.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        result[idx] = str(v)
    return result


def apply_compressed_texts(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Apply compressed texts from the agent to subtitle items.

    Для каждого элемента, чей ``index`` присутствует в ``compressed_texts``,
    функция:
    - заменяет ``text`` на список из одной строки с новым содержимым;
    - отмечает сегмент как ``shortened = True``.

    Args:
        items: Список исходных сегментов (модифицируется на месте).
        compressed_texts: Словарь ``index -> new_text`` из ответа модели.

    Returns:
        Тот же список ``items`` для удобства чейнинга вызовов.
    """

    for item in items:
        idx = item.get("index")
        if idx is None or idx not in compressed_texts:
            continue
        new_text = compressed_texts[idx]
        item["text"] = [new_text]
        item["shortened"] = True
    return items


def recompute_analysis_for_position(
    items: List[Dict[str, Any]],
    pos: int,
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> None:
    """Recompute timing analysis only for one position.

    Так как ``extended_mismatch_ratio`` зависит от времени начала следующего
    сегмента, здесь анализируется небольшой срез из 1–2 элементов, после
    чего поле ``analysis`` первого элемента переносится обратно в оригинальный
    список ``items``.

    Args:
        items: Полный список сегментов.
        pos: Позиция (0-based) элемента, чьё ``analysis`` нужно обновить.
        avg_chars_per_sec: Предполагаемая скорость чтения.
        max_mismatch_ratio: Порог критичности для ratio.
    """

    if pos < 0 or pos >= len(items):
        return

    end = min(len(items), pos + 2)
    slice_items = [deepcopy(items[i]) for i in range(pos, end)]
    result = analyze_subtitles(
        slice_items,
        avg_chars_per_sec=avg_chars_per_sec,
        max_mismatch_ratio=max_mismatch_ratio,
    )
    items[pos]["analysis"] = result["items"][0].get("analysis")


def should_continue_compression(
    analysis: Dict[str, Any],
    target_ratio: float = 1.5,
) -> bool:
    """Decide whether a segment still needs further compression.

    Сигнал к продолжению — комбинированный ratio всё ещё превышает
    целевой порог ``target_ratio``.

    Args:
        analysis: Текущее поле ``analysis`` сегмента.
        target_ratio: Желаемый верхний предел комбинированного ratio.

    Returns:
        True, если имеет смысл отправить сегмент на ещё одно сжатие.
    """

    combined = get_combined_mismatch_ratio(analysis)
    if combined is None:
        return False
    return combined > target_ratio


def is_too_aggressive_compression(
    old_analysis: Dict[str, Any],
    new_analysis: Dict[str, Any],
    low_ratio_threshold: float = 1.0,
    strong_overflow_threshold: float = 2.1,
) -> bool:
    """Heuristic to detect overly aggressive compression.

    Идея: слишком сильное сокращение текста плохо, если исходный
    перегруз по длительности был умеренным. В таком случае не стоит
    доводить комбинированный ratio до очень низких значений.

    Args:
        old_analysis: Поле ``analysis`` до сжатия.
        new_analysis: Поле ``analysis`` после сжатия и пересчёта.
        low_ratio_threshold: Нижний безопасный предел комбинированного ratio.
        strong_overflow_threshold: Порог, выше которого исходный перегруз
            считается настолько сильным, что агрессивное сжатие допустимо.

    Returns:
        True, если сжатие считается чрезмерным и стоит ослабить его.
    """

    old_combined = get_combined_mismatch_ratio(old_analysis)
    new_combined = get_combined_mismatch_ratio(new_analysis)

    if old_combined is None or new_combined is None:
        return False

    if new_combined < low_ratio_threshold and old_combined <= strong_overflow_threshold:
        return True
    return False


@dataclass
class ShorteningResult:
    """Result of text shortening with metrics.
    
    Новая логика: нет ограничения на процент сокращения.
    Главный критерий успеха — итоговый mismatch_ratio <= TARGET_MISMATCH_RATIO.
    """
    
    index: int
    original_text: str
    shortened_text: str
    shortening_percent: float
    new_mismatch_ratio: float | None  # ratio после сокращения
    target_achieved: bool  # True если new_mismatch_ratio <= TARGET_MISMATCH_RATIO
    attempt_number: int


def calculate_shortening_percent(original_text: str, shortened_text: str) -> float:
    """Calculate the percentage of text reduction.

    Вычисляет, на сколько процентов был сокращён текст относительно оригинала.
    Положительное значение означает сокращение, отрицательное — удлинение.

    Args:
        original_text: Исходный текст до сокращения.
        shortened_text: Текст после сокращения.

    Returns:
        Процент сокращения (например, 25.0 означает сокращение на 25%).
    """
    original_len = len(original_text)
    if original_len == 0:
        return 0.0
    shortened_len = len(shortened_text)
    reduction = original_len - shortened_len
    return (reduction / original_len) * 100.0


def format_shortened_subtitles(
    items: List[Dict[str, Any]],
) -> Dict[int, str]:
    """Format shortened subtitles as {index: shortened_text} dictionary.

    Собирает все субтитры, которые были сокращены (имеют флаг shortened=True),
    и возвращает их в виде словаря для быстрой проверки результатов.

    Args:
        items: Список субтитров с полем `shortened` и `text`.

    Returns:
        Словарь {index: shortened_text} только для сокращённых субтитров.
    """
    result: Dict[int, str] = {}
    for item in items:
        if item.get("shortened", False):
            idx = item.get("index")
            if idx is not None:
                text = join_text_lines(item.get("text", ""))
                result[idx] = text
    return result


def print_shortened_subtitles(items: List[Dict[str, Any]]) -> str:
    """Generate a formatted string of shortened subtitles for display.

    Выводит сокращённые субтитры в удобном для чтения формате.

    Args:
        items: Список субтитров.

    Returns:
        Строка JSON с сокращёнными субтитрами.
    """
    shortened = format_shortened_subtitles(items)
    return json.dumps(shortened, ensure_ascii=False, indent=2)


def get_segments_needing_further_shortening(
    items: List[Dict[str, Any]],
    target_ratio: float = TARGET_MISMATCH_RATIO,
) -> List[Dict[str, Any]]:
    """Find segments that still need further shortening after compression.

    После применения сокращения и пересчёта analysis, находит сегменты,
    у которых mismatch_ratio или extended_mismatch_ratio всё ещё >= target_ratio.

    Args:
        items: Список субтитров с пересчитанным analysis.
        target_ratio: Целевой максимальный ratio (по умолчанию 1.5).

    Returns:
        Список сегментов, требующих дополнительного сокращения.
    """
    still_need_shortening: List[Dict[str, Any]] = []
    
    for item in items:
        if not item.get("shortened", False):
            continue
            
        analysis = item.get("analysis") or {}
        mismatch_ratio = analysis.get("mismatch_ratio")
        extended_ratio = analysis.get("extended_mismatch_ratio")
        
        needs_more = False
        if mismatch_ratio is not None and mismatch_ratio >= target_ratio:
            needs_more = True
        if extended_ratio is not None and extended_ratio >= target_ratio:
            needs_more = True
            
        if needs_more:
            still_need_shortening.append(item)
    
    return still_need_shortening


def evaluate_compression_result(
    item: Dict[str, Any],
    original_text: str,
    shortened_text: str,
    attempt_number: int = 1,
    target_ratio: float = TARGET_MISMATCH_RATIO,
) -> ShorteningResult:
    """Evaluate the result of a single compression attempt.

    Анализирует результат сокращения текста. Успех определяется тем,
    достигнут ли целевой mismatch_ratio <= target_ratio.
    
    ВАЖНО: Нет ограничения на процент сокращения — модель может сокращать
    на любое значение, главное достичь целевого ratio.

    Args:
        item: Сегмент субтитра с пересчитанным analysis.
        original_text: Исходный текст до сокращения.
        shortened_text: Текст после сокращения.
        attempt_number: Номер текущей попытки (начиная с 1).
        target_ratio: Целевой максимальный mismatch_ratio.

    Returns:
        ShorteningResult с метриками результата.
    """
    shortening_percent = calculate_shortening_percent(original_text, shortened_text)
    
    # Получаем новый ratio после сокращения
    analysis = item.get("analysis") or {}
    new_combined = get_combined_mismatch_ratio(analysis)
    
    # Проверяем достигнут ли целевой ratio
    target_achieved = False
    if new_combined is not None and new_combined <= target_ratio:
        target_achieved = True
    
    return ShorteningResult(
        index=item.get("index", 0),
        original_text=original_text,
        shortened_text=shortened_text,
        shortening_percent=shortening_percent,
        new_mismatch_ratio=new_combined,
        target_achieved=target_achieved,
        attempt_number=attempt_number,
    )


def build_retry_prompt_context(
    items: List[Dict[str, Any]],
    segments_needing_more: List[Dict[str, Any]],
    window: int = 3,
    target_ratio: float = TARGET_MISMATCH_RATIO,
) -> str:
    """Build context for retry prompt when segments still need more shortening.

    Формирует информацию для повторного запроса к модели с указанием,
    какие сегменты всё ещё имеют слишком высокий ratio и нуждаются
    в дополнительном сокращении.

    Args:
        items: Полный список сегментов.
        segments_needing_more: Сегменты, требующие дополнительного сокращения.
        window: Размер контекстного окна.
        target_ratio: Целевой максимальный ratio.

    Returns:
        Строка с контекстом для повторного запроса.
    """
    segments_info: List[Dict[str, Any]] = []
    
    # Создаём индекс позиций
    index_to_pos: Dict[int, int] = {}
    for pos, item in enumerate(items):
        idx = item.get("index")
        if idx is not None:
            index_to_pos[idx] = pos
    
    for item in segments_needing_more:
        idx = item.get("index")
        if idx is None:
            continue
            
        pos = index_to_pos.get(idx)
        if pos is None:
            continue
        
        analysis = item.get("analysis") or {}
        current_ratio = get_combined_mismatch_ratio(analysis)
        current_text = join_text_lines(item.get("text", ""))
        context = build_context_window(items, pos, window=window)
        
        segments_info.append({
            "index": idx,
            "current_text": current_text,
            "current_mismatch_ratio": round(current_ratio, 2) if current_ratio else None,
            "target_ratio": target_ratio,
            "context": context,
            "instruction": f"Сократи ещё! Текущий ratio={current_ratio:.2f}, нужно <= {target_ratio}.",
        })
    
    return json.dumps({"retry_segments": segments_info}, ensure_ascii=False, separators=(",", ":"))


def apply_all_compressions(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Apply all compressed texts without filtering.

    Применяет все сокращения от модели без проверки уровня сокращения.
    Новая логика: нет ограничения на процент сокращения.

    Args:
        items: Исходный список сегментов.
        compressed_texts: Словарь index -> new_text от модели.

    Returns:
        Обновлённый список items.
    """
    return apply_compressed_texts(items, compressed_texts)


def get_compression_stats(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
    target_ratio: float = TARGET_MISMATCH_RATIO,
) -> Dict[str, Any]:
    """Get statistics about compression results.

    Собирает статистику по всем сокращениям: средний процент, мин/макс,
    количество успешных (ratio <= target) и требующих дополнительной работы.

    Args:
        items: Список сегментов после применения сокращений и пересчёта analysis.
        compressed_texts: Словарь index -> new_text от модели.
        target_ratio: Целевой максимальный ratio.

    Returns:
        Словарь со статистикой.
    """
    if not compressed_texts:
        return {
            "total_segments": 0,
            "target_achieved_count": 0,
            "still_need_shortening_count": 0,
            "avg_shortening_percent": 0.0,
            "min_shortening_percent": 0.0,
            "max_shortening_percent": 0.0,
        }
    
    # Собираем все проценты сокращения и статистику по ratio
    percentages: List[float] = []
    target_achieved = 0
    still_need = 0
    
    index_to_item: Dict[int, Dict[str, Any]] = {}
    for item in items:
        idx = item.get("index")
        if idx is not None:
            index_to_item[idx] = item
    
    for idx, new_text in compressed_texts.items():
        item = index_to_item.get(idx)
        if item is None:
            continue
        
        # Для корректного подсчёта нужен оригинальный текст
        # Если item уже изменён, то text уже новый — нужно хранить original отдельно
        # Здесь предполагаем что items ещё не изменены
        original_text = join_text_lines(item.get("text", ""))
        pct = calculate_shortening_percent(original_text, new_text)
        percentages.append(pct)
        
        # Проверяем достигнут ли target ratio
        analysis = item.get("analysis") or {}
        combined = get_combined_mismatch_ratio(analysis)
        if combined is not None and combined <= target_ratio:
            target_achieved += 1
        else:
            still_need += 1
    
    return {
        "total_segments": len(compressed_texts),
        "target_achieved_count": target_achieved,
        "still_need_shortening_count": still_need,
        "avg_shortening_percent": sum(percentages) / len(percentages) if percentages else 0.0,
        "min_shortening_percent": min(percentages) if percentages else 0.0,
        "max_shortening_percent": max(percentages) if percentages else 0.0,
    }


def recompute_full_analysis(
    items: List[Dict[str, Any]],
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> List[Dict[str, Any]]:
    """Recompute timing analysis for all segments.

    Запускает полный проход анализа по всему списку и возвращает тот же
    список с обновлённым полем ``analysis`` для каждого элемента.

    Args:
        items: Список сегментов для анализа.
        avg_chars_per_sec: Предполагаемая скорость чтения.
        max_mismatch_ratio: Порог критичности ratio.

    Returns:
        Обновлённый список ``items``.
    """

    result = analyze_subtitles(
        items,
        avg_chars_per_sec=avg_chars_per_sec,
        max_mismatch_ratio=max_mismatch_ratio,
    )
    return result["items"]


def main() -> None:
    """CLI для работы со скиллом сокращения субтитров.

    Режимы:
    - --mode payload:  читает input.json и печатает payload для агента;
    - --mode apply:    читает input.json и response.json (map index->text),
                       применяет изменения и пересчитывает analysis, пишет
                       результат в output.json;
    - --mode show:     показывает сокращённые субтитры в формате {index: text}.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Subtitle shortener helper CLI")
    parser.add_argument("input", help="Input JSON with analyzed subtitles")
    parser.add_argument("--mode", choices=["payload", "apply", "show"], default="payload")
    parser.add_argument("--response", help="Path to agent JSON response (for mode=apply)")
    parser.add_argument("--output", help="Path to save updated JSON (for mode=apply)")
    parser.add_argument("--min-ratio", type=float, default=MIN_MISMATCH_RATIO_FOR_SHORTENING,
                       help="Minimum mismatch ratio for segment selection (default: 1.5)")

    args = parser.parse_args()

    items = load_items(args.input)
    
    # Выводим статистику по загруженным субтитрам
    total_items = len(items)
    segments_for_shortening = collect_segments_for_agent(items, min_ratio=args.min_ratio)
    print(f"Загружено субтитров: {total_items}", file=__import__('sys').stderr)
    print(f"Субтитров для сокращения (ratio >= {args.min_ratio}): {len(segments_for_shortening)}", 
          file=__import__('sys').stderr)

    if args.mode == "payload":
        payload = build_agent_payload(items, min_ratio=args.min_ratio)
        print(payload)
        return
    
    if args.mode == "show":
        # Показываем уже сокращённые субтитры
        shortened = format_shortened_subtitles(items)
        if shortened:
            print(json.dumps(shortened, ensure_ascii=False, indent=2))
        else:
            print("{}")
        return

    # mode == "apply"
    if not args.response or not args.output:
        raise SystemExit("--response и --output обязательны для mode=apply")

    with open(args.response, "r", encoding="utf-8") as f:
        raw = f.read()

    compressed = parse_agent_response(raw)
    print(f"Получено сокращений от модели: {len(compressed)}", file=__import__('sys').stderr)
    
    # Применяем все сокращения (без ограничения на %)
    items = apply_all_compressions(items, compressed)
    items = recompute_full_analysis(items)
    
    # Проверяем какие сегменты всё ещё нуждаются в сокращении
    still_need = get_segments_needing_further_shortening(items)
    if still_need:
        print(f"Сегментов, требующих дополнительного сокращения: {len(still_need)}", 
              file=__import__('sys').stderr)
        for item in still_need:
            idx = item.get("index")
            analysis = item.get("analysis") or {}
            ratio = get_combined_mismatch_ratio(analysis)
            print(f"  - index={idx}, ratio={ratio:.2f}" if ratio else f"  - index={idx}", 
                  file=__import__('sys').stderr)
    
    save_items(args.output, items)
    print(f"Результат сохранён в: {args.output}", file=__import__('sys').stderr)
    
    # Выводим сокращённые субтитры
    shortened = format_shortened_subtitles(items)
    if shortened:
        print("\nСокращённые субтитры:")
        print(json.dumps(shortened, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
