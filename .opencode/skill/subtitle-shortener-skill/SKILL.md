---
name: subtitle-shortener-skill
description: Сжатие субтитров с высоким mismatch_ratio (>= 1.5). Модель сокращает текст на любое значение, главное — достичь ratio <= 1.5 после пересчёта.
---

# Subtitle Shortener Skill

## Когда использовать

- JSON-сабы с полем `analysis` (см. `assets/subs_analyzed.json`)
- Нужно сократить реплики с `mismatch_ratio >= 1.5` ИЛИ `extended_mismatch_ratio >= 1.5`
- **Нет ограничения** на процент сокращения — модель сокращает сколько нужно
- Цель: достичь `mismatch_ratio <= 1.5` после пересчёта

---

## ⚠️ КРИТИЧЕСКИ ВАЖНО: Как запускать скрипт

**НЕ используй `uv run -m`** — скрипт находится вне структуры пакетов
репозитория и не может быть импортирован как модуль Python.

### Правильный способ (PowerShell, из корня d:\tts_projects\vosk)

Используй `uv run python` с **относительным путём** к скрипту:

```powershell
# Режим payload — генерация данных для сжатия моделью
uv run python .opencode/skill/subtitle-shortener-skill/scripts/subtitle_shortener_skill.py skill_test/subs_analyzed.json --mode payload

# Режим apply — применение ответа модели и пересчёт analysis
uv run python .opencode/skill/subtitle-shortener-skill/scripts/subtitle_shortener_skill.py skill_test/subs_analyzed.json --mode apply --response temp/response.json --output temp/output.json

# Режим show — показать уже сокращённые субтитры в формате {index: text}
uv run python .opencode/skill/subtitle-shortener-skill/scripts/subtitle_shortener_skill.py temp/output.json --mode show
```

### Аргументы CLI

| Аргумент       | Описание                                                  |
|----------------|-----------------------------------------------------------|
| `input`        | Путь к JSON с проанализированными субтитрами (позиционный)|
| `--mode`       | `payload` (по умолчанию), `apply` или `show`              |
| `--response`   | Путь к JSON-ответу модели (обязателен для `mode=apply`)   |
| `--output`     | Путь для сохранения результата (обязателен для `mode=apply`) |
| `--min-ratio`  | Минимальный ratio для отбора (по умолчанию 1.5)           |

---

## Константы

```python
TARGET_MISMATCH_RATIO = 1.5      # Целевой максимальный ratio после сокращения
MIN_MISMATCH_RATIO_FOR_SHORTENING = 1.5  # Порог для отбора сегментов
MAX_COMPRESSION_ATTEMPTS = 5     # Лимит попыток
```

**ВАЖНО:** Ограничение MAX_SHORTENING_PERCENT = 30% **УДАЛЕНО**. 
Модель теперь может сокращать текст на любой процент, главное — 
достичь `ratio <= 1.5` после пересчёта.

## Пайплайн

1. `load_items(path)` — загрузка JSON
2. `collect_segments_for_agent(items)` — сбор сегментов с `ratio >= 1.5`
3. `build_agent_payload(items)` — формирование payload для модели
4. Модель возвращает `{"index": "новый текст", ...}`
5. `parse_agent_response(raw)` — парсинг ответа
6. `apply_all_compressions(items, compressed)` — применение всех сокращений
7. `recompute_full_analysis(items)` — пересчёт таймингов
8. `get_segments_needing_further_shortening(items)` — проверка, нужны ли ещё итерации
9. Если есть — `build_retry_prompt_context()` и повтор
10. `save_items(path, items)` — сохранение
11. `format_shortened_subtitles(items)` — вывод результата в формате `{index: text}`

## Основные функции

| Функция | Назначение |
|---------|------------|
| `collect_segments_for_agent(items)` | Сбор сегментов с `ratio >= 1.5` (без проверки `is_critical`) |
| `build_agent_payload(items)` | JSON-payload для модели |
| `parse_agent_response(raw)` | `dict[int, str]` из ответа модели |
| `apply_all_compressions(items, compressed)` | Применяет все сокращения (без ограничений) |
| `get_segments_needing_further_shortening(items)` | Находит сегменты с `ratio > 1.5` после сокращения |
| `build_retry_prompt_context(items, segments)` | Контекст для повтора |
| `format_shortened_subtitles(items)` | `{index: text}` для сокращённых |
| `print_shortened_subtitles(items)` | Печать сокращённых в JSON |
| `recompute_full_analysis(items)` | Пересчёт `analysis` для всех |

## Формат ответа модели

```json
{"1": "Сокращённый текст...", "8": "Другой текст..."}
```

## Формат вывода сокращённых субтитров

После обработки можно вывести сокращённые субтитры:

```json
{
  "1": "Из этой малазийской деревушки...",
  "8": "Вы видите начало приключения!",
  "9": "Им дали две минуты захватить всё с лодки!"
}
```

## Пример: итеративное сжатие

```python
from scripts.subtitle_shortener_skill import (
    load_items, save_items, build_agent_payload, parse_agent_response,
    apply_all_compressions, recompute_full_analysis,
    get_segments_needing_further_shortening, build_retry_prompt_context,
    format_shortened_subtitles, MAX_COMPRESSION_ATTEMPTS,
)

items = load_items("input.json")

for attempt in range(1, MAX_COMPRESSION_ATTEMPTS + 1):
    payload = build_agent_payload(items)
    # ... отправка payload модели, получение raw_response ...
    compressed = parse_agent_response(raw_response)
    items = apply_all_compressions(items, compressed)
    items = recompute_full_analysis(items)
    
    # Проверяем, нужны ли ещё итерации
    still_need = get_segments_needing_further_shortening(items)
    if not still_need:
        break
    
    retry_context = build_retry_prompt_context(items, still_need)
    # ... повторный запрос с retry_context ...

save_items("output.json", items)

# Вывод сокращённых субтитров
shortened = format_shortened_subtitles(items)
print(shortened)  # {"1": "...", "8": "...", ...}
```

## Правила сжатия для промпта

- Сжимать только сегменты из `segments`
- Использовать `context` для связности, не трогать соседние реплики
- **Нет ограничения на процент сокращения** — сокращай сколько нужно
- Цель: достичь `mismatch_ratio <= 1.5` после пересчёта
- Нельзя: менять смысл, добавлять факты, переводить

---

## 🚫 СТРОГО ЗАПРЕЩЕНО

### Запрещённые методы сокращения

1. **НЕ создавать скрипты для сокращения** — никаких Python/bash скриптов,
   которые пытаются автоматически обрезать или модифицировать текст
2. **НЕ использовать обрезку с многоточием** — недопустимо:
   - `"Это очень длинная строка"` → `"Это очень длинная..."`
   - `"Привет, как дела?"` → `"Привет..."`
3. **НЕ применять механическое удаление слов** — нельзя просто убирать
   слова без понимания контекста и смысла
4. **НЕ использовать регулярные выражения** для сокращения текста
5. **НЕ заменять фразы на сокращения** типа `"и так далее"` → `"и т.д."`

### Единственный допустимый метод

**Только языковая модель (LLM)** может выполнять сокращение:
- Модель должна переформулировать текст короче, **сохраняя полный смысл**
- Сокращённый текст должен быть законченным предложением
- Нельзя терять важную информацию из оригинала
- Перефразирование должно звучать естественно на русском языке

### Примеры правильного и неправильного сокращения

| Оригинал | ❌ НЕПРАВИЛЬНО | ✅ ПРАВИЛЬНО |
|----------|----------------|--------------|
| "Им дали две минуты на то, чтобы захватить с собой всё, что они смогут" | "Им дали две минуты..." | "У них две минуты, чтобы взять всё возможное" |
| "Этот человек покинет этот остров с миллионом долларов" | "Этот человек покинет..." | "Победитель получит миллион долларов" |
| "Мы хотим хотя бы ближе подобраться к острову" | "Мы хотим ближе..." | "Хотим приблизиться к острову" |

---

## Проверка результатов

После сокращения **обязательно**:

1. Пересчитать `analysis` через `recompute_full_analysis(items)`
2. Проверить `get_segments_needing_further_shortening(items)`
3. Если есть сегменты с `ratio > 1.5` — повторить сокращение

### Цикл проверки

```
┌─────────────────────────────────────────────────────────┐
│  1. Получить сжатые тексты от модели                    │
│  2. Применить все сокращения (apply_all_compressions)   │
│  3. Пересчитать analysis (recompute_full_analysis)      │
│  4. Проверить ratio каждого сокращённого сегмента       │
│  5. Если есть с ratio > 1.5:                            │
│     → Сформировать retry-контекст                       │
│     → Запросить дополнительное сокращение               │
│     → Вернуться к шагу 2                                │
│  6. Сохранить результат                                 │
│  7. Вывести сокращённые субтитры в формате {idx: text}  │
└─────────────────────────────────────────────────────────┘
```

Максимум попыток: `MAX_COMPRESSION_ATTEMPTS = 5`

## Структура payload сегмента

```json
{
  "context": "текст 3 соседей до и после",
  "index": 1,
  "text": "Исходный текст...",
  "duration_sec": 2.28,
  "estimated_sec": 3.92,
  "mismatch_ratio": 1.72,
  "extended_mismatch_ratio": 1.35
}
```

**Важно:** 
- Сегмент попадает в payload если `mismatch_ratio >= 1.5` ИЛИ `extended_mismatch_ratio >= 1.5`
- Флаг `is_critical` **НЕ используется** для отбора
- Поле `extended_mismatch_ratio` может быть `null`

## Resources

- `scripts/subtitle_shortener_skill.py` — основная логика + CLI (`--mode payload|apply|show`)
- `assets/subs_analyzed.json` — пример входных данных
