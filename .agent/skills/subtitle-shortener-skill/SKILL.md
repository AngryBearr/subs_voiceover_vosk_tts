---
name: subtitle-shortener-skill
description: Сжатие критичных субтитров (is_critical=true, extended_mismatch_ratio>=1.5) с контролем уровня сокращения (≤30%) и итеративными повторами при слишком агрессивном сжатии.
---

# Subtitle Shortener Skill

## Когда использовать

- JSON-сабы с полем `analysis` (см. `assets/subs_analyzed.json`)
- Нужно сократить реплики с `analysis.is_critical == true` и `extended_mismatch_ratio >= 1.5`
- Требуется контроль уровня сокращения (не более 30%)

---

## ⚠️ КРИТИЧЕСКИ ВАЖНО: Как запускать скрипт

**НЕ используй `uv run -m`** — скрипт находится вне структуры пакетов
репозитория и не может быть импортирован как модуль Python.

### Правильный способ (PowerShell, из корня d:\tts_projects\vosk)

Используй `uv run python` с **относительным путём** к скрипту:

```powershell
# Режим payload — генерация данных для сжатия моделью
uv run python .agent/skills/subtitle-shortener-skill/scripts/subtitle_shortener_skill.py skill_test/subs_analyzed.json --mode payload

# Режим apply — применение ответа модели и пересчёт analysis
uv run python .agent/skills/subtitle-shortener-skill/scripts/subtitle_shortener_skill.py skill_test/subs_analyzed.json --mode apply --response temp/response.json --output temp/output.json
```

### Аргументы CLI

| Аргумент     | Описание                                                  |
|--------------|-----------------------------------------------------------|
| `input`      | Путь к JSON с проанализированными субтитрами (позиционный)|
| `--mode`     | `payload` (по умолчанию) или `apply`                      |
| `--response` | Путь к JSON-ответу модели (обязателен для `mode=apply`)   |
| `--output`   | Путь для сохранения результата (обязателен для `mode=apply`) |

---

## Константы

```python
MAX_SHORTENING_PERCENT = 30.0  # макс. сокращение
TOLERANCE_PERCENT = 5.0        # допуск ±5%
MAX_COMPRESSION_ATTEMPTS = 5   # лимит попыток
```

## Пайплайн

1. `load_items(path)` — загрузка JSON
2. `build_agent_payload(items)` — формирование payload для модели
3. Модель возвращает `{"index": "новый текст", ...}`
4. `parse_agent_response(raw)` — парсинг ответа
5. `apply_with_shortening_control(items, compressed)` — применение с проверкой уровня
6. Если есть слишком агрессивные — `build_retry_prompt_context()` и повтор
7. `recompute_full_analysis(items)` — пересчёт таймингов
8. `save_items(path, items)` — сохранение

## Основные функции

| Функция | Назначение |
|---------|------------|
| `build_agent_payload(items)` | JSON-payload для модели |
| `parse_agent_response(raw)` | `dict[int, str]` из ответа модели |
| `apply_with_shortening_control(items, compressed)` | Применяет приемлемые, возвращает агрессивные |
| `filter_aggressive_compressions(items, compressed)` | Разделяет на приемлемые/агрессивные |
| `build_retry_prompt_context(items, aggressive)` | Контекст для повтора с мягким сжатием |
| `calculate_shortening_percent(orig, new)` | % сокращения |
| `recompute_full_analysis(items)` | Пересчёт `analysis` для всех |

## Формат ответа модели

```json
{"1": "Сокращённый текст...", "8": "Другой текст..."}
```

## Пример: итеративное сжатие

```python
from scripts.subtitle_shortener_skill import (
    load_items, save_items, build_agent_payload, parse_agent_response,
    apply_with_shortening_control, recompute_full_analysis,
    build_retry_prompt_context, MAX_COMPRESSION_ATTEMPTS,
)

items = load_items("input.json")

for attempt in range(1, MAX_COMPRESSION_ATTEMPTS + 1):
    payload = build_agent_payload(items)
    # ... отправка payload модели, получение raw_response ...
    compressed = parse_agent_response(raw_response)
    items, too_aggressive = apply_with_shortening_control(items, compressed)
    
    if not too_aggressive:
        break
    
    retry_context = build_retry_prompt_context(items, too_aggressive)
    # ... повторный запрос с retry_context ...

items = recompute_full_analysis(items)
save_items("output.json", items)
```

## Правила сжатия для промпта

- Сжимать только сегменты из `segments`
- Использовать `context` для связности, не трогать соседние реплики
- Цель: уменьшить текст на 15–25%, не более 30%
- Нельзя: менять смысл, добавлять факты, переводить

## Структура payload сегмента

```json
{
  "context": "текст 3 соседей до и после",
  "index": 1,
  "text": "Исходный текст...",
  "duration_sec": 2.28,
  "estimated_sec": 3.92,
  "mismatch_ratio": 1.35,
  "extended_mismatch_ratio": 1.72
}
```

## Resources

- `scripts/subtitle_shortener_skill.py` — основная логика + CLI (`--mode payload|apply`)
- `assets/subs_analyzed.json` — пример входных данных
