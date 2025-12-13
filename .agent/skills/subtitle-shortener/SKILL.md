---
name: subtitle-shortener
description: Скилл для сжатия критичных по длительности субтитров в JSON с полем analysis. Используй, когда нужно сократить только перегруженные реплики (analysis.is_critical=true и высокий extended_mismatch_ratio), сохранив смысл и пересчитав тайминговый анализ.
---

# Subtitle Shortener

## Overview

Скилл для мягкого сжатия перегруженных по длительности реплик в JSON‑сабах,
где уже есть поле `analysis` от `utils.analyze_text`. Работает только с
критичными сегментами (`analysis.is_critical=true`) и с высоким
`combined_mismatch_ratio` (см. ниже), сохраняя смысл и естественность речи.

## Основной сценарий

1. Пользователь даёт JSON с сабами и полем `analysis` (см. пример в
   `assets/test_subs_chunk_analyzed_v2.json`).
2. Скилл строит компактный представление для агента: для каждого
   критичного сегмента собирает локальный контекст (3 сегмента до и
   после), текст, длительность и метрики из `analysis`.
3. Это представление сериализуется в компактный JSON (`separators=(",", ":")`)
   и вставляется в промпт.
4. Агент возвращает **строго один JSON‑объект** формата

   ```jsonc
   {
     "1": "Из этой малазийской деревушки...",
     "5": "Это история о 'Последнем Герое'."
   }
   ```

   где ключ — `index` сегмента, значение — новый текст.
5. Скрипт применяет новые тексты к исходному JSON и (во внешнем пайплайне)
   запускается пересчёт `analysis` через `utils.analyze_text.analyze_subtitles`.

## Формат agent‑payload

Скрипт `scripts/compress_critical_segments.py` готовит данные для агентов.
Он собирает только нужные поля и убирает всё лишнее, чтобы минимизировать
нагрузку на контекст.

- Вход: список items того же формата, что в `assets/test_subs_chunk_analyzed_v2.json`.
- Отбор сегментов:
  - `analysis.is_critical == true`;
  - `combined_mismatch_ratio >= 1.5` (мягкое, неагрессивное сжатие);
    где `combined_mismatch_ratio = min(mismatch_ratio, extended_mismatch_ratio)`
    (если `extended_mismatch_ratio` отсутствует — используем `mismatch_ratio`).
- Контекст: 3 сегмента до и 3 после (если есть).

На выходе для агента получается JSON вида:

```jsonc
{
  "segments": [
    {
      "context": "текст вокруг сегмента (до/после)",
      "index": 1,
      "text": "Из этой небольшой малазийской рыбацкой деревушки...",
      "duration_sec": 2.28,
      "estimated_sec": 3.92,
      "mismatch_ratio": 1.72,
      "extended_mismatch_ratio": 1.35,
      "combined_mismatch_ratio": 1.35
    }
  ]
}
```

Сериализация для промпта:

```python
from .compress_critical_segments import collect_critical_segments, build_agent_payload

segments = collect_critical_segments(items, window=3, min_combined_ratio=1.5)
payload = build_agent_payload(segments)
# payload — компактная строка JSON без лишних пробелов
```

## Инструкции для агента

В промпте к модели:

- Объясни, что приходит `payload` c ключом `segments`.
- Для каждого элемента `segments` нужно **умеренно сократить** поле `text`:
  - ориентир — уменьшить оценочную длительность на ~15–20%,
    чтобы итоговый `combined_mismatch_ratio` вернулся в диапазон примерно 1.0–1.5;
  - использовать поле `context` только как подсказку для смысла;
  - не менять факты и не добавлять новых;
  - сохранять естественную русскую речь.
- Делать сжатие **плавно и итеративно** (несколько небольших шагов), используя
  промежуточные проверки метрик:
  - после каждой попытки сжатия пересчитывать `analysis` (или хотя бы метрики
    `mismatch_ratio`/`extended_mismatch_ratio`) и смотреть `combined_mismatch_ratio`;
  - если `combined_mismatch_ratio` всё ещё высокий (например, > 1.5), сжимать ещё;
  - если метрика уже в целевом диапазоне, остановиться;
  - избегать чрезмерного сокращения: не стремиться уводить метрику существенно ниже ~1.0
    и не выкидывать смысловые элементы.
  - считать сжатие **слишком агрессивным**, если:
    - новый `combined_mismatch_ratio` < 1.0, и
    - исходный `combined_mismatch_ratio` был не слишком высоким (≤ 2.1).
    В таком случае нужно сделать вторую попытку сжатия, сохранив больше деталей
    (меньше выбрасывать фраз/уточнений).
- Ответить **строго JSON‑объектом** без комментариев и текста вокруг.

Парсинг ответа и применение:

```python
from .compress_critical_segments import (
  parse_agent_response,
  apply_compressed_texts,
  recompute_analysis_for_position,
  should_continue_compression,
  is_too_aggressive_compression,
)

compressed_map = parse_agent_response(model_output)
updated_items = apply_compressed_texts(items, compressed_map)
# Далее во внешнем пайплайне запустить utils.analyze_text.analyze_subtitles
# чтобы пересчитать поле analysis.

# Для итеративного режима можно пересчитывать метрики и проверять агрессивность:
# pos = ...  # позиция элемента в items
# old_analysis = dict(items[pos].get("analysis") or {})
# recompute_analysis_for_position(items, pos)
# new_analysis = items[pos].get("analysis") or {}
# if is_too_aggressive_compression(old_analysis, new_analysis,
#                                  low_ratio_threshold=1.0,
#                                  strong_overflow_threshold=2.1):
#     # первая попытка получилась слишком жёсткой — сжимаем мягче
#     ...
# elif should_continue_compression(new_analysis, target_ratio=1.5):
#     # метрика всё ещё высока (>1.5) — можно сделать ещё один шаг сжатия
#     ...
```

## Resources


This skill includes example resource directories that demonstrate how to organize different types of bundled resources:

### scripts/
Executable code (Python/Bash/etc.) that can be run directly to perform specific operations.

**Examples from other skills:**
- PDF skill: `fill_fillable_fields.py`, `extract_form_field_info.py` - utilities for PDF manipulation
- DOCX skill: `document.py`, `utilities.py` - Python modules for document processing

**Appropriate for:** Python scripts, shell scripts, or any executable code that performs automation, data processing, or specific operations.

**Note:** Scripts may be executed without loading into context, but can still be read by Claude for patching or environment adjustments.

### references/
Documentation and reference material intended to be loaded into context to inform Claude's process and thinking.

**Examples from other skills:**
- Product management: `communication.md`, `context_building.md` - detailed workflow guides
- BigQuery: API reference documentation and query examples
- Finance: Schema documentation, company policies

**Appropriate for:** In-depth documentation, API references, database schemas, comprehensive guides, or any detailed information that Claude should reference while working.

### assets/

- `test_subs_chunk_analyzed_v2.json` — пример входного файла с тайминговым
  анализом. Используй его для отладки промптов и проверки пайплайна
  сжатия. В реальной работе вместо него подставляется любой JSON того же
  формата.
- `test_subs_chunk_analyzed_critical.json` — более крупный пример (подвыборка),
  содержащий много критичных сегментов для быстрой отладки промптов.

---

**Любые лишние example‑файлы (example_asset.txt и т.п.) можно удалить после
настройки скилла.**

