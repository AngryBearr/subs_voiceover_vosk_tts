# AGENTS.md

Этот файл имеет приоритет над другими инструкциями для агента (`CLAUDE.md` — ссылка на него).
Он описывает, как работать с проектом: английское видео → распознание и перевод (`stt/`) →
русская озвучка с подгонкой под время (`voiceover/`), а также классический пайплайн
SRT → JSON → RUNorm + RUAccent → Vosk TTS → микс.

## 1. Общие правила

- Репозиторий работает в Linux.
- Основное виртуальное окружение — `subs_env`; перед запуском скриптов всегда активируй его.
- Скрипты запускай из корня репозитория через `uv` в виде модулей, например:
    `uv run -m utils.srt_to_json ...` (или `subs_env/bin/python -m ...`).
- Пакеты ставь через `uv pip install <package>` / `uv pip uninstall <package>`, только после
    активации `subs_env`. Новые зависимости добавляй в `requirements.txt` с точной версией.
- Всегда типизируй код. Тестируй код после того, как написал (`uv run -m pytest tests`).
- В git попадает только код, конфиги и документация. Отчёты, результаты прогонов, аудио и
    модели лежат в `output/` и `models/` (они в `.gitignore`) и не коммитятся.

## 2. Язык общения

- Отвечать на русском языке, если не указано иное.
- Документацию и комментарии писать на английском языке, если не указано иное.
- Use plain, conversational language. Avoid unexplained jargon, acronyms, and unnecessary internal
    details; explain any necessary technical term immediately.
- For work summaries, lead with what was done, how it was tested, what the results mean, and the
    recommended next step. Use short sections only when helpful, not as a mandatory template for
    every reply.
- Clearly distinguish completed work from plans, real checks from simulations, and observations
    from guarantees. Mention limitations plainly; include technical details, logs, and identifiers
    only when needed or requested.

## 3. Окружения и основные пайплайны

- **Окружения** (всё сразу: `bash scripts/install.sh`, проверка: `subs_env/bin/python -m stt.selftest`;
    по отдельности: `bash scripts/setup_envs.sh`):
    - `subs_env` — всё: распознание (`stt/`), перевод, озвучка (`voiceover/`, Vosk, TeraTTS),
        проверка речи, Gradio, классические скрипты. Ставится из `requirements.txt` через uv с
        `overrides-onnxruntime.txt` (модуль `onnxruntime` даёт только `onnxruntime-gpu`).
    - `f5_env` — `f5-tts` для озвучки клонированными голосами (ESpeech, `voiceover/f5_worker.py`).
    - `diarizen_env` — DiariZen v2, разделение говорящих по умолчанию (код в `models/diarizen/repo`);
        без него `stt.run` берёт pyannote.
    - `stt_qwen_env` — необязательное, только для сравнений с Qwen3-ASR; удалено 29.10
        (`bash scripts/setup_envs.sh qwen` создаёт заново); пайплайн его не использует.
- **Настройки**: `stt_config.json` (общие, в git) и поверх него `stt_config.local.json` (личные, не в
    git: модели через подписку, папки); `voiceover_config.json` — детали озвучки. Интерфейс:
    `bash scripts/run_ui.sh` (вкладки «Обработка», «Банк голосов», «Проверка и правка реплик»,
    «Настройки», «Справка»).
- **Видео → русская озвучка целиком**:
    `subs_env/bin/python -m stt.run <видео или маска>` — английские субтитры со спикерами,
    глоссарий, имена, перевод под тайминг, сценарий озвучки, voiceover, проверка речи, микс.
    Настройки в `stt_config.json`, подробности в `docs/stt_local.md`.
- **Спикеры**: DiariZen v2 с эксклюзивным слоем (`diarizer_options`) → уточнение отрезков ResNet293
    (`speaker_refine`) → пословная перемаркировка Витерби (`stt/boundaries.py`, `word_speakers`;
    эмбеддинги слов кэшируются в `words.emb.<model>.npz`). Метрики: `stt.turn_eval` (метки `>>`
    субтитров CBS, S20) и `stt.sdh_eval --cues` (SDH-метки, S51); стенд в `output/exp_boundary/`.
- **Озвучка готовых русских субтитров**: `uv run -m voiceover.pipeline run <subs.json> --out <dir>`
    (настройки в `voiceover_config.json`).
- **Правка реплик серии**: `corrections.json` в папке серии (интерфейс или `stt.subtitle_edit`
    export/import для Subtitle Edit); повторный `stt.run` озвучивает заново только изменённые реплики.
- Ключи API (`OPENROUTER_API_KEY`, `ORCAROUTER_API_KEY`, `DEEPSEEK_API_KEY`, токен HF) лежат в `.env`;
    никогда не выводить их, как и токены opencode (`auth.json`).

## 4. Классический пайплайн SRT → JSON → RUNorm + RUAccent → Vosk → микс

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
    - Озвучка (`voiceover/engine.py`) так не делает: склеенная строка оставляла без ударения
        2-3% слов (первое слово реплики, слова после многоточия). Там каждая реплика идёт отдельно
        (`ruaccent_pass(per_cue=True)`, части между многоточиями отдельно), словам без ударения
        ставится словарное (`fill_dictionary_stress`), ударение, поставленное человеком (`зам+ок`),
        главнее всех (`keep_user_stress`). CLI `utils.text_normalizer` по умолчанию оставляет
        склейку; `--accent-per-cue` включает новый способ.
    - Числа в озвучке (`voiceover/normalize.py`) читает `ru-normalizr` (падеж по контексту),
        диапазоны «3-4 года» — своё правило, время, счёт и аббревиатуры — `rutextnorm`.

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

## 5. RUNorm / RUAccent и устройства

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

## 6. Vosk TTS и ONNXRuntime

- **Единый CLI**
    - Всегда использовать `synthesize.synthesize_cli` как точку входа для
        TTS (single и batch), а не вызывать низкоуровневые функции напрямую
        из внешних скриптов.

- **ONNXRuntime-провайдеры**
    - Не менять паттерн патчинга провайдеров в модулях `synthesize`.
    - Использовать существующий контекстный менеджер вида
        `_ForceCPUProviders` (или аналог) для принудительного CPU;
        **не** патчить `Model.__init__` и не создавать модель до входа в
        контекст. Число потоков ONNX Runtime ограничивает так же устроенный
        `_LimitThreads` (Vosk рядом с ESpeech на видеокарте: `VOICEOVER_CPU_THREADS`).

## 7. Стиль кода, модульность и переиспользование

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

## 8. Производительность и надёжность

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

## 9. Приоритеты и связь с другими инструкциями

- Этот файл (`AGENTS.md`) имеет приоритет над другими наборами
    инструкций для агента.
- При расхождении с `.github/copilot-instructions.md` следовать
    правилам из `AGENTS.md`, сохраняя при этом общий пайплайн
    (SRT → JSON → RUNorm + RUAccent → Vosk TTS → mix/export).

