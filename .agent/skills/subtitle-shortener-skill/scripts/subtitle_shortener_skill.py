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
MAX_SHORTENING_PERCENT: float = 30.0  # Максимально допустимое сокращение в %
TOLERANCE_PERCENT: float = 5.0  # Допуск ±5% от целевого уровня
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
    min_extended_ratio: float = 1.5,
) -> Dict[int, AgentSegment]:
    """Select critical segments that require compression and attach context.

    Учитываются только сегменты, которые по результатам анализа помечены
    как критичные и при этом сохраняют высокий ``extended_mismatch_ratio``.
    Для каждого такого сегмента строится контекст и собираются только те
    числовые метрики, которые реально нужны агенту.

    Args:
        items: Полный список сегментов с полем ``analysis``.
        window: Размер оконного контекста (количество соседей до/после).
        min_extended_ratio: Минимально допустимое значение
            ``extended_mismatch_ratio`` для попадания в выборку.

    Returns:
        Словарь ``index -> AgentSegment`` для всех выбранных сегментов.
    """

    segments: Dict[int, AgentSegment] = {}
    for pos, item in enumerate(items):
        analysis = item.get("analysis") or {}
        if not analysis.get("is_critical"):
            continue

        extended_ratio = analysis.get("extended_mismatch_ratio")
        if extended_ratio is None or extended_ratio < min_extended_ratio:
            continue

        idx = item.get("index")
        if idx is None:
            continue

        text = join_text_lines(item.get("text", ""))
        context = build_context_window(items, pos, window=window)

        duration_sec = analysis.get("duration_sec")
        estimated_sec = analysis.get("estimated_sec")
        combined = get_combined_mismatch_ratio(analysis)

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
    min_extended_ratio: float = 1.5,
) -> str:
    """Build compact JSON payload for the compression agent.

    Функция подготавливает минимальный по размеру JSON‑объект, который
    содержит только критичные сегменты, выбранные collect_segments_for_agent,
    а также служебные поля управления силой сжатия. JSON сериализуется без
    лишних пробелов для экономии контекста.

    Args:
        items: Список сегментов с заполненным ``analysis``.
        window: Количество соседей до/после для текстового контекста.
        min_extended_ratio: Порог по ``extended_mismatch_ratio`` для отбора.

    Returns:
        Строка JSON, готовая к вставке в промпт модели.
    """

    ensure_shortening_meta(items)
    segments = collect_segments_for_agent(items, window=window, min_extended_ratio=min_extended_ratio)
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
    """Result of text shortening with metrics."""
    
    index: int
    original_text: str
    shortened_text: str
    shortening_percent: float
    is_acceptable: bool
    is_too_aggressive: bool
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


def is_shortening_acceptable(
    shortening_percent: float,
    max_percent: float = MAX_SHORTENING_PERCENT,
    tolerance: float = TOLERANCE_PERCENT,
) -> bool:
    """Check if the shortening level is within acceptable bounds.

    Сокращение считается приемлемым, если:
    - Оно положительное (текст реально стал короче)
    - Не превышает максимально допустимый уровень с учётом допуска

    Args:
        shortening_percent: Текущий процент сокращения.
        max_percent: Максимально допустимый процент сокращения (по умолчанию 30%).
        tolerance: Допуск в процентах (по умолчанию ±5%).

    Returns:
        True, если сокращение в допустимых пределах.
    """
    # Сокращение должно быть положительным (текст стал короче)
    if shortening_percent < 0:
        return True  # Удлинение — это не проблема в контексте "слишком агрессивного"
    
    # Проверяем, не превышает ли сокращение максимальный порог с допуском
    upper_limit = max_percent + tolerance
    return shortening_percent <= upper_limit


def is_shortening_too_aggressive(
    shortening_percent: float,
    max_percent: float = MAX_SHORTENING_PERCENT,
    tolerance: float = TOLERANCE_PERCENT,
) -> bool:
    """Check if the text was shortened too aggressively.

    Сокращение считается слишком агрессивным, если процент сокращения
    превышает максимально допустимый уровень с учётом допуска.

    Args:
        shortening_percent: Текущий процент сокращения.
        max_percent: Максимально допустимый процент сокращения (по умолчанию 30%).
        tolerance: Допуск в процентах (по умолчанию ±5%).

    Returns:
        True, если сокращение слишком агрессивное.
    """
    upper_limit = max_percent + tolerance
    return shortening_percent > upper_limit


def evaluate_compression_result(
    index: int,
    original_text: str,
    shortened_text: str,
    attempt_number: int = 1,
    max_percent: float = MAX_SHORTENING_PERCENT,
    tolerance: float = TOLERANCE_PERCENT,
) -> ShorteningResult:
    """Evaluate the result of a single compression attempt.

    Анализирует результат сокращения текста и определяет, является ли он
    приемлемым или слишком агрессивным.

    Args:
        index: Индекс сегмента.
        original_text: Исходный текст до сокращения.
        shortened_text: Текст после сокращения.
        attempt_number: Номер текущей попытки (начиная с 1).
        max_percent: Максимально допустимый процент сокращения.
        tolerance: Допуск в процентах.

    Returns:
        ShorteningResult с метриками и флагами результата.
    """
    shortening_percent = calculate_shortening_percent(original_text, shortened_text)
    is_acceptable = is_shortening_acceptable(shortening_percent, max_percent, tolerance)
    is_aggressive = is_shortening_too_aggressive(shortening_percent, max_percent, tolerance)
    
    return ShorteningResult(
        index=index,
        original_text=original_text,
        shortened_text=shortened_text,
        shortening_percent=shortening_percent,
        is_acceptable=is_acceptable,
        is_too_aggressive=is_aggressive,
        attempt_number=attempt_number,
    )


def filter_aggressive_compressions(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
    max_percent: float = MAX_SHORTENING_PERCENT,
    tolerance: float = TOLERANCE_PERCENT,
) -> tuple[Dict[int, str], Dict[int, ShorteningResult]]:
    """Filter out segments that were compressed too aggressively.

    Проходит по всем сокращённым сегментам и разделяет их на:
    - Приемлемые (возвращаются в первом словаре)
    - Слишком агрессивные (возвращаются результаты во втором словаре)

    Args:
        items: Исходный список сегментов.
        compressed_texts: Словарь index -> new_text от модели.
        max_percent: Максимально допустимый процент сокращения.
        tolerance: Допуск в процентах.

    Returns:
        Tuple из:
        - Словаря приемлемых сокращений (index -> text)
        - Словаря результатов для слишком агрессивных (index -> ShorteningResult)
    """
    acceptable: Dict[int, str] = {}
    too_aggressive: Dict[int, ShorteningResult] = {}
    
    # Создаём индекс для быстрого поиска по index
    index_to_item: Dict[int, Dict[str, Any]] = {}
    for item in items:
        idx = item.get("index")
        if idx is not None:
            index_to_item[idx] = item
    
    for idx, new_text in compressed_texts.items():
        item = index_to_item.get(idx)
        if item is None:
            continue
        
        original_text = join_text_lines(item.get("text", ""))
        result = evaluate_compression_result(
            index=idx,
            original_text=original_text,
            shortened_text=new_text,
            max_percent=max_percent,
            tolerance=tolerance,
        )
        
        if result.is_too_aggressive:
            too_aggressive[idx] = result
        else:
            acceptable[idx] = new_text
    
    return acceptable, too_aggressive


def build_retry_prompt_context(
    items: List[Dict[str, Any]],
    aggressive_results: Dict[int, ShorteningResult],
    window: int = 3,
) -> str:
    """Build context for retry prompt when compressions were too aggressive.

    Формирует информацию для повторного запроса к модели с указанием,
    какие сегменты были сокращены слишком сильно и на сколько нужно
    смягчить сокращение.

    Args:
        items: Полный список сегментов.
        aggressive_results: Результаты слишком агрессивных сокращений.
        window: Размер контекстного окна.

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
    
    for idx, result in aggressive_results.items():
        pos = index_to_pos.get(idx)
        if pos is None:
            continue
        
        context = build_context_window(items, pos, window=window)
        
        segments_info.append({
            "index": idx,
            "original_text": result.original_text,
            "previous_attempt": result.shortened_text,
            "shortening_percent": round(result.shortening_percent, 1),
            "target_max_percent": MAX_SHORTENING_PERCENT,
            "context": context,
            "instruction": f"Сократи мягче! Предыдущая попытка убрала {result.shortening_percent:.1f}% текста, нужно не более {MAX_SHORTENING_PERCENT}%.",
        })
    
    return json.dumps({"retry_segments": segments_info}, ensure_ascii=False, separators=(",", ":"))


def apply_with_shortening_control(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
    max_percent: float = MAX_SHORTENING_PERCENT,
    tolerance: float = TOLERANCE_PERCENT,
) -> tuple[List[Dict[str, Any]], Dict[int, ShorteningResult]]:
    """Apply compressed texts with shortening level control.

    Применяет только те сокращения, которые не превышают допустимый уровень.
    Возвращает обновлённый список и информацию о сегментах, которые были
    сокращены слишком агрессивно и требуют повторной попытки.

    Args:
        items: Исходный список сегментов.
        compressed_texts: Словарь index -> new_text от модели.
        max_percent: Максимально допустимый процент сокращения.
        tolerance: Допуск в процентах.

    Returns:
        Tuple из:
        - Обновлённого списка items (с применёнными приемлемыми сокращениями)
        - Словаря слишком агрессивных результатов для повторной обработки
    """
    acceptable, too_aggressive = filter_aggressive_compressions(
        items, compressed_texts, max_percent, tolerance
    )
    
    # Применяем только приемлемые сокращения
    if acceptable:
        items = apply_compressed_texts(items, acceptable)
    
    return items, too_aggressive


def get_compression_stats(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
) -> Dict[str, Any]:
    """Get statistics about compression results.

    Собирает статистику по всем сокращениям: средний процент, мин/макс,
    количество приемлемых и агрессивных.

    Args:
        items: Исходный список сегментов.
        compressed_texts: Словарь index -> new_text от модели.

    Returns:
        Словарь со статистикой.
    """
    if not compressed_texts:
        return {
            "total_segments": 0,
            "acceptable_count": 0,
            "aggressive_count": 0,
            "avg_shortening_percent": 0.0,
            "min_shortening_percent": 0.0,
            "max_shortening_percent": 0.0,
        }
    
    acceptable, too_aggressive = filter_aggressive_compressions(items, compressed_texts)
    
    # Собираем все проценты сокращения
    percentages: List[float] = []
    index_to_item: Dict[int, Dict[str, Any]] = {}
    for item in items:
        idx = item.get("index")
        if idx is not None:
            index_to_item[idx] = item
    
    for idx, new_text in compressed_texts.items():
        item = index_to_item.get(idx)
        if item is None:
            continue
        original_text = join_text_lines(item.get("text", ""))
        pct = calculate_shortening_percent(original_text, new_text)
        percentages.append(pct)
    
    return {
        "total_segments": len(compressed_texts),
        "acceptable_count": len(acceptable),
        "aggressive_count": len(too_aggressive),
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
    """Простейший CLI для отладки.

    Режимы:
    - --mode payload:  читает input.json и печатает payload для агента;
    - --mode apply:    читает input.json и response.json (map index->text),
                       применяет изменения и пересчитывает analysis, пишет
                       результат в output.json.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Subtitle shortener helper CLI")
    parser.add_argument("input", help="Input JSON with analyzed subtitles")
    parser.add_argument("--mode", choices=["payload", "apply"], default="payload")
    parser.add_argument("--response", help="Path to agent JSON response (for mode=apply)")
    parser.add_argument("--output", help="Path to save updated JSON (for mode=apply)")

    args = parser.parse_args()

    items = load_items(args.input)

    if args.mode == "payload":
        payload = build_agent_payload(items)
        print(payload)
        return

    # mode == "apply"
    if not args.response or not args.output:
        raise SystemExit("--response и --output обязательны для mode=apply")

    with open(args.response, "r", encoding="utf-8") as f:
        raw = f.read()

    compressed = parse_agent_response(raw)
    items = apply_compressed_texts(items, compressed)
    items = recompute_full_analysis(items)
    save_items(args.output, items)


if __name__ == "__main__":  # pragma: no cover
    main()
