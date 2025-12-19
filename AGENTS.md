# AGENT Instructions for `vosk` TTS repo

Этот файл имеет приоритет над другими инструкциями для агента.
Он описывает, как работать с проектом, какие паттерны
сохранять и как расширять пайплайн SRT → JSON → RUNorm + RUAccent →
Vosk TTS → микс.

## 1. Среда, запуск скриптов и установка зависимостей

- **Python и окружение**
    - Использовать Python 3.
    - Всегда активировать виртуальное окружение `vosk_env` перед
        запуском любых скриптов или установкой/удалением пакетов.
    - В PowerShell из корня репозитория:

        ```powershell
        cd d:\tts_projects\vosk
        .\vosk_env\Scripts\Activate.ps1
        ```

- **Запуск скриптов**
    - Скрипты запускать из корня репозитория через `uv` в виде модулей:

        ```powershell
        uv run -m utils.srt_to_json ...
        uv run -m utils.text_normalizer ...
        uv run -m synthesize.synthesize_cli ...
        uv run -m utils.audio_mixer ...
        ```

    - Примеры типичных команд:
        - SRT → JSON:

            ```powershell
            uv run -m utils.srt_to_json input.srt -o subs.json
            ```

        - Нормализация + акцентирование JSON сабов (RUNorm + RUAccent):

            ```powershell
            uv run -m utils.text_normalizer subs.json accentized.json 20 \
                --accent-device CPU --norm-device cpu
            ```

        - Синтез TTS (единый CLI):
            - Single-line:

                ```powershell
                uv run -m synthesize.synthesize_cli \
                    --text "Привет!" \
                    --model models/vosk-model-tts-ru-0.10-multi \
                    --voice 0 \
                    --speech-rate 1.0 \
                    --device cpu
                ```

            - Batch из JSON сабов:

                ```powershell
                uv run -m synthesize.synthesize_cli \
                    --json .\accentized.json \
                    --model models/vosk-model-tts-ru-0.10-multi \
                    --voice 0 \
                    --speech-rate 1.0 \
                    --device cpu \
                    --output-folder .\tts_out \
                    --file-prefix tts_
                ```

        - Быстрый микс сегментов:

            ```powershell
            uv run -m utils.audio_mixer subs.json tts_out output/mix \
                --sr 48000 --channels 1
            ```

- **Установка и удаление пакетов**
    - Любые `pip`-операции выполнять только после активации `vosk_env`.
    - Использовать `uv pip`:

        ```powershell
        # установка
        uv pip install <package>

        # удаление
        uv pip uninstall <package>
        ```

    - Не писать команды установки без явного указания, что окружение
        уже активировано.

## 2. Общий пайплайн данных

- **SRT → JSON**
    - Вход: SRT-файл сабов.
    - Инструмент: модуль `utils.srt_to_json`.
    - Выход: JSON-список записей вида:
        `{"index", "start", "end", "duration", "text", "gender", ...}`.
    - Поле `text` всегда список строк (даже если строка одна).

- **JSON → RUNorm + RUAccent**
    - Основная точка входа: `utils.text_normalizer`.
    - Пайплайн:
        1. `runorm_pass`: нормализация текста (RUNorm), добавляет временное
             поле `_normalized_text`.
        2. `ruaccent_pass`: акцентирование (RUAccent) по `_normalized_text`
             с восстановлением троеточий, результат сохраняется в `text` как
             список из одной строки.
        3. `normalize_and_accent_subtitles`: обёртка над двумя шагами, не
             меняет схему записей, кроме поля `text`.

- **RUAccent и троеточия**
    - Перед акцентированием троеточия обрабатываются через
        `replace_ellipsis` ("..." → ".."), после — восстанавливаются через
        `restore_ellipsis` (".." → "...").
    - Батч-акцентирование выполняется через `accentizer.process_all` над
        строковым представлением списка с использованием
        `skip_regex=r"[\[\],\"]"` и последующим `ast.literal_eval`.
    - При ошибке парсинга строковый результат разбивается fallback-ом на
        поэлементную обработку, длина списка сохраняется.

- **JSON → Vosk TTS (синтез)**
    - Точка входа: `synthesize.synthesize_cli`.
    - Режимы:
        - Single-line: аргумент `--text`.
        - Batch: аргумент `--json` (путь к JSON/JSONL); текст берётся из
            поля `text` (по умолчанию) или ключа `--text-key`.
    - Важные параметры: `--model`, `--device cpu|cuda`, `--voice`,
        `--speech-rate`, `--output-sample-rate`, `--output-folder`,
        `--file-prefix`.
    - Batch-синтез создаёт файлы `<file_prefix><n>.wav` (нумерация с 1)
        в указанной папке.

- **Микс сегментов в финальный звук**
    - Основной путь: `utils.audio_mixer.mix_segments` или CLI
        `uv run -m utils.audio_mixer ...`.
        - Читает JSON сабов (ожидает `index`, `start`, `end`).
        - Берёт сегменты `<index>.wav` из папки и миксует в единый трек с
            опциональным ресемплингом (через `soxr`, если установлен).
        - Может накладывать базовый трек (`base_audio_path`) и управлять
            `base_gain_db` / `speech_gain_db`.
    - Альтернатива для простых случаев: `utils.subs_utils`:
        - `create_silence_audio(video_path, silence.wav)` — создать тихий
            трек нужной длины по видео.
        - `overlay_audio(silence.wav, tts_out, subs.json, mixed.mp3)` —
            наложить TTS-сегменты на тишину.

## 3. RUNorm / RUAccent и устройства

- **Загрузка моделей**
    - RUNorm:
        - Функция `_load_runorm(device: str = "cpu")` в
            `utils.text_normalizer`.
        - Загружает модель c `model_size="big"` из `./runorm_cache`.
    - RUAccent:
        - Функция `_load_ruaccent(device: str = "cpu")` в
            `utils.text_normalizer`.
        - Использует `omograph_model_size="turbo3.1"`, `use_dictionary=True`
            и `custom_dict=cust_dict`.

- **Дефолтные устройства**
    - По умолчанию и RUNorm, и RUAccent загружаются на **CPU**.
    - Для GPU-окружений устройства нужно указывать явно (например,
        `device="CUDA"` при прямом вызове функций или
        `--accent-device CUDA --norm-device CUDA` в CLI).

- **CLI `utils.text_normalizer`**
    - Поддерживает флаги `--accent-device` и `--norm-device` с choices
        `{"CPU", "CUDA", "cuda", "cpu"}`.
    - Уважать эти флаги и не менять их смысл: они управляют устройствами
        RUAccent и RUNorm соответственно.

## 4. Vosk TTS и ONNXRuntime

- **Единый CLI**
    - Всегда использовать `synthesize.synthesize_cli` как точку входа для
        TTS (single и batch), а не вызывать низкоуровневые функции напрямую
        из внешних скриптов.

- **ONNXRuntime-провайдеры**
    - Не менять паттерн патчинга провайдеров в модулях `synthesize`.
    - Использовать существующий контекстный менеджер вида
        `_ForceCPUProviders` (или аналог) для принудительного CPU;
        **не** патчить `Model.__init__` и не создавать модель до входа в
        контекст.

## 5. Стиль кода, модульность и переиспользование

- **Общий стиль**
    - Импорты: стандартная библиотека → сторонние пакеты → локальные
        модули. Не использовать `from x import *`.
    - Типы: добавлять type hints (PEP 484) для публичных функций
        и ключевых внутренних утилит; предпочитать явные
        `List[Dict[str, Any]]` / `list[dict[str, Any]]` и понятные
        возвращаемые типы.
    - Форматирование: отступ 4 пробела, строки до 100–120 символов,
        короткие, информативные docstring-и.
    - Язык: внешние объяснения и документация — по-русски; код,
        идентификаторы и комментарии — по-английски.

- **JSON и структуры данных**
    - JSON читать/писать в UTF-8, при записи использовать
        `ensure_ascii=False` для корректной кириллицы.
    - Схему субтитров (`index`, `start`, `end`, `duration`, `text`,
        `gender`, ...) сохранять совместимой с существующими инструментами.
    - Поле `text` всегда должно быть списком строк, даже если строка одна.

- **Модульность и утилиты**
    - Код писать модульно: избегать дублирования логики по разным
        скриптам.
    - При обнаружении повторяющихся паттернов или функций в нескольких
        файлах выносить их в отдельные утилиты (в первую очередь в
        `utils/`) и переиспользовать.
    - Предпочитать расширение существующих утилит (`audio_mixer`,
        `subs_utils`, `text_normalizer`, `text_sanitize`, и т.д.) вместо
        копирования похожего кода в новые скрипты.

- **Стабильность интерфейсов**
    - Новые CLI-параметры должны быть опциональными с безопасными
        значениями по умолчанию и не ломать существующее поведение.
    - Любые новые флаги нужно протянуть как в CLI-слой, так и во
        внутренние функции.
    - Имена файлов, публичных функций и ключевых структур (полей JSON)
        по возможности сохранять стабильными.

## 6. Производительность и надёжность

- **Модели и кэширование**
    - Тяжёлые модели (RUNorm, RUAccent, Vosk TTS) загружать один раз и
        переиспользовать в батч-процессах.
    - В батч-синтезе TTS держать один `Model`/`Synth` на весь проход.

- **Обработка ошибок**
    - Рано валидировать пути к файлам, наличие моделей и корректность
        параметров.
    - Для логирования использовать модуль `logging`, а не `print`, за
        исключением осознанного CLI-вывода.
    - Для батч-процессинга RUAccent при ошибках парсинга результата
        падать в поэлементную обработку без изменения длины списка.

- **Минимальность изменений**
    - Изменения держать локальными и минимальными.
    - Не вводить крупные рефакторинги и новые фреймворки без явного
        запроса.

## 7. Приоритеты и связь с другими инструкциями

- Этот файл (`AGENTS.md`) имеет приоритет над другими наборами
    инструкций для агента.
- При расхождении с `.github/copilot-instructions.md` следовать
    правилам из `AGENTS.md`, сохраняя при этом общий пайплайн
    (SRT → JSON → RUNorm + RUAccent → Vosk TTS → mix/export).

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: Bash("openskills read <skill-name>")
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>algorithmic-art</name>
<description>Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use this when users request creating art using code, generative art, algorithmic art, flow fields, or particle systems. Create original algorithmic art rather than copying existing artists' work to avoid copyright violations.</description>
<location>project</location>
</skill>

<skill>
<name>brand-guidelines</name>
<description>Applies Anthropic's official brand colors and typography to any sort of artifact that may benefit from having Anthropic's look-and-feel. Use it when brand colors or style guidelines, visual formatting, or company design standards apply.</description>
<location>project</location>
</skill>

<skill>
<name>canvas-design</name>
<description>Create beautiful visual art in .png and .pdf documents using design philosophy. You should use this skill when the user asks to create a poster, piece of art, design, or other static piece. Create original visual designs, never copying existing artists' work to avoid copyright violations.</description>
<location>project</location>
</skill>

<skill>
<name>doc-coauthoring</name>
<description>Guide users through a structured workflow for co-authoring documentation. Use when user wants to write documentation, proposals, technical specs, decision docs, or similar structured content. This workflow helps users efficiently transfer context, refine content through iteration, and verify the doc works for readers. Trigger when user mentions writing docs, creating proposals, drafting specs, or similar documentation tasks.</description>
<location>project</location>
</skill>

<skill>
<name>docx</name>
<description>"Comprehensive document creation, editing, and analysis with support for tracked changes, comments, formatting preservation, and text extraction. When Claude needs to work with professional documents (.docx files) for: (1) Creating new documents, (2) Modifying or editing content, (3) Working with tracked changes, (4) Adding comments, or any other document tasks"</description>
<location>project</location>
</skill>

<skill>
<name>frontend-design</name>
<description>Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.</description>
<location>project</location>
</skill>

<skill>
<name>internal-comms</name>
<description>A set of resources to help me write all kinds of internal communications, using the formats that my company likes to use. Claude should use this skill whenever asked to write some sort of internal communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident reports, project updates, etc.).</description>
<location>project</location>
</skill>

<skill>
<name>mcp-builder</name>
<description>Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK).</description>
<location>project</location>
</skill>

<skill>
<name>pdf</name>
<description>Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, and handling forms. When Claude needs to fill in a PDF form or programmatically process, generate, or analyze PDF documents at scale.</description>
<location>project</location>
</skill>

<skill>
<name>pptx</name>
<description>"Presentation creation, editing, and analysis. When Claude needs to work with presentations (.pptx files) for: (1) Creating new presentations, (2) Modifying or editing content, (3) Working with layouts, (4) Adding comments or speaker notes, or any other presentation tasks"</description>
<location>project</location>
</skill>

<skill>
<name>skill-creator</name>
<description>Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Claude's capabilities with specialized knowledge, workflows, or tool integrations.</description>
<location>project</location>
</skill>

<skill>
<name>slack-gif-creator</name>
<description>Knowledge and utilities for creating animated GIFs optimized for Slack. Provides constraints, validation tools, and animation concepts. Use when users request animated GIFs for Slack like "make me a GIF of X doing Y for Slack."</description>
<location>project</location>
</skill>

<skill>
<name>subtitle-shortener-skill</name>
<description>Скилл для итеративного мягкого сжатия критичных субтитров в JSON с полем analysis. Работает по критичным сегментам (analysis.is_critical=true и extended_mismatch_ratio>=1.5), собирает локальный контекст, подаёт их агенту для сжатия, затем обновляет текст, тайминговый анализ и служебные поля shortened/max_shortening_ratio.</description>
<location>project</location>
</skill>

<skill>
<name>template</name>
<description>Replace with description of the skill and when Claude should use it.</description>
<location>project</location>
</skill>

<skill>
<name>theme-factory</name>
<description>Toolkit for styling artifacts with a theme. These artifacts can be slides, docs, reportings, HTML landing pages, etc. There are 10 pre-set themes with colors/fonts that you can apply to any artifact that has been creating, or can generate a new theme on-the-fly.</description>
<location>project</location>
</skill>

<skill>
<name>web-artifacts-builder</name>
<description>Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind CSS, shadcn/ui). Use for complex artifacts requiring state management, routing, or shadcn/ui components - not for simple single-file HTML/JSX artifacts.</description>
<location>project</location>
</skill>

<skill>
<name>webapp-testing</name>
<description>Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs.</description>
<location>project</location>
</skill>

<skill>
<name>xlsx</name>
<description>"Comprehensive spreadsheet creation, editing, and analysis with support for formulas, formatting, data analysis, and visualization. When Claude needs to work with spreadsheets (.xlsx, .xlsm, .csv, .tsv, etc) for: (1) Creating new spreadsheets with formulas and formatting, (2) Reading or analyzing data, (3) Modify existing spreadsheets while preserving formulas, (4) Data analysis and visualization in spreadsheets, or (5) Recalculating formulas"</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
