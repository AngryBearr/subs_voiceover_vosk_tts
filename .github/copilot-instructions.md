# Copilot Project Instructions

Цель: Ускорить онбординг ИИ-агента в конвейер «SRT субтитры → нормализация + акцентирование → синтез речи Vosk TTS → микс/экспорт».
Пишите кратко. Следуйте текущим паттернам; не вносите фреймворки и крупные рефакторинги без запроса.

## 1. Big Picture Data Flow
1. Входной SRT (`*.srt`) → парсинг в структурированный JSON (`utils/srt_to_json.py`).
2. JSON сабы (список `{index,start,end,duration,text:[...],gender}`) → нормализация + акцентирование батчем (RUNorm + RUAccent) (`utils/text_normalizer.py`, при необходимости `utils/subtitle_processor.py`). Троеточие временно заменяем (`...`→`..`) до акцентирования и потом восстанавливаем.
3. Акцентированный текст (возможны маркеры ударения в словах `+`) → синтез Vosk TTS одиночный или батч через единый CLI (`synthesize/synthesize_cli.py`; под капотом `synthesize/synthesize.py` и `synthesize/synthesize_batch.py`).
4. Полученные WAV‑сегменты → опционально подгоняем темпом под слот (`utils/subs_utils.change_speed`) и/или собираем на таймлайн высокопроизводительным миксером (`utils/audio_mixer.mix_segments`). Лёгкая альтернатива: `create_silence_audio` + `overlay_audio` из `utils/subs_utils`.

## 2. Key Conventions & Patterns
- Текст как список: поле `text` всегда список строк. В батч‑процессорах строки обычно соединяются пробелом, а на выход кладётся одна строка в виде списка из одного элемента.
- Акцентирование: перед RUAccent обязательно `replace_ellipsis`/`restore_ellipsis`. Для безопасной пакетной обработки используем строковое представление списка (`str(list)`) и `skip_regex`, который пропускает синтаксис списка. Рекомендуемый шаблон: `r'[\[\],\"]'`.
	Примечание: в коде `utils/text_normalizer.py` `skip_regex` приведён к этому шаблону.
- RUAccent батч: превращаем список строк в строку‑лист, вызываем `process_all(..., skip_regex=...)`, затем парсим обратно `ast.literal_eval` с защитой и fallback‑ом на поэлементную обработку при ошибке.
- Выбор девайса/провайдеров (Vosk TTS): используется контекстный патч onnxruntime через менеджер `_ForceCPUProviders` внутри `synthesize/synthesize.py` (CPU принудительно) или нативные провайдеры ORT (CUDA при наличии). Не изменяйте подход (не патчить `Model.__init__`).
- RUNorm/RUAccent девайсы: в `utils/text_normalizer.py` по умолчанию загружаются на GPU (`device="CUDA"/"cuda"`). Для CPU‑окружений явно переключайте девайсы в коде или параметрах.
	В CLI доступны флаги `--accent-device` и `--norm-device` для выбора устройств (например, `--accent-device CPU --norm-device cpu`).
- Кастомный словарь для акцентирования: `utils/text_normalizer.py` (`cust_dict`) — сохраняйте и расширяйте аккуратно.
- Именование выходов:
	- single‑synth: если не указан `output_path`, автогенерация в `./out/` с префиксом и таймстампом (`filename_prefix`).
	- batch‑synth: `<file_prefix><n>.wav` (нумерация с 1) в `output_folder`.

## 3. External Dependencies / Requirements
- Core: `ruaccent`, `runorm`, `onnxruntime`/`onnxruntime-gpu`, `vosk-tts`, `tokenizers` (если есть BERT‑подмодель), PyTorch (GPU в зависимостях RUAccent/RUNorm).
- Resampling/IO: `scipy` (опц. для ресемплинга в TTS), `soundfile` (чтение/запись WAV), `soxr` (опц. высококачественный ресемплинг в миксере).
- Overlay/tempo: `pydub` (в `utils/subs_utils.change_speed`).
- FFmpeg + FFprobe должны быть в PATH (скорость, оверлей, транскод).
- RUNorm грузится с `model_size="big"` из кэша `runorm_cache`; RUAccent — `omograph_model_size='turbo3.1'`.

## 4. Typical Workflows
- SRT → JSON:
	- Python API: `srt_to_json(path, out_path)`
	- CLI: `python utils/srt_to_json.py input.srt -o subs.json`
- Нормализация + акцент: `python utils/text_normalizer.py subs.json accentized.json 20` (батч размер опционален). На выходе JSON с маркерами ударений.
	CPU пример: `python utils/text_normalizer.py subs.json accentized.json 20 --accent-device CPU --norm-device cpu`.
- Синтез (единый CLI):
	- Single‑line: `python -m synthesize.synthesize_cli --text "Привет!" --model models/vosk-model-tts-ru-0.10-multi --voice 0 --speech-rate 1.0 --device cpu`
	- Batch: `python -m synthesize.synthesize_cli --json .\accentized.json --model models/vosk-model-tts-ru-0.10-multi --voice 0 --speech-rate 1.0 --device cpu --output-folder .\tts_out --file-prefix tts_`
	- Опциональный ресемплинг: `--output-sample-rate 48000` (использует SciPy при наличии)
- Сборка/микс: быстрый миксер `python -m utils.audio_mixer subs.json tts_out output/mix --sr 48000 --channels 1` (даст `mix.wav` и по умолчанию `mix.mp3`).
- Альтернатива (простая): создать тишину `create_silence_audio`, затем `overlay_audio(silence.wav, tts_out, subs.json, mixed.mp3)` из `utils/subs_utils`.

## 5. Testing Notes
- Наличие примеров/тестов: `temp/test_srt_transformer.py` (парсинг), `temp/test_ruaccent.py` (структура/батч RUAccent), `utils/test_accent_array.py` (RUAccent.process_array демо).
- Запуск: `python -m unittest` (учтите, тесты RUAccent/RUNorm могут быть тяжелыми и требуют моделей/кеша).
- При добавлении новых процессоров — следуйте паттерну батч‑акцентирования (с сохранением длины списка) и fallback‑обработке ошибок.

## 6. Extension Guidelines
- Новые параметры — опциональны, с безопасными дефолтами; не ломайте существующие интерфейсы.
- Пер‑строчные настройки синтеза: брать по ключам из записи (например, `speaker_id`, `speech_rate`), падать на дефолты по образцу `synthesize_batch.py`/CLI.
- Вспомогательные утилиты — предпочитайте добавлять в `utils/`, а не разносить по синтез‑скриптам.

## 7. Performance / Reliability
- Тяжёлые модели грузим один раз и переиспользуем: RUAccent/RUNorm (в `text_normalizer.py`) и Vosk TTS.
	В батч‑синтезе (`synthesize_batch.py`) один `Model/Synth` создаётся и кэшируется на весь проход.
- Защищённые падения: при ошибке батч‑парсинга RUAccent — обрабатываем элементы по одному (`subtitle_processor.py` демонстрирует стиль), длина списка неизменна.
- Миксер `utils/audio_mixer` выполняет один проход по предвыделенному буферу; для очень длинных проектов допускается будущее расширение до потокового режима.

## 8. Pitfalls / Gotchas
- Обязательная обработка троеточий до RUAccent, иначе возможны ошибки/неверные ударения.
- JSON сохраняем в `utf-8` с `ensure_ascii=False` для кириллицы.
- `speaker_id` должен соответствовать выбранной модели; явной валидации нет.
- Патч провайдеров ORT должен происходить до создания `Model` (в `synthesize.py` это уже учтено контекст‑менеджером). Не переносите патч после инстанциирования.
- По умолчанию `text_normalizer` грузит модели на GPU; на CPU‑хостах переведите `device` на CPU, если нужно.

## 9. Safe Change Examples
- Добавить новый препроход нормализации: реализовать функцию в `text_normalizer.py`, вызвать до `replace_ellipsis`.
- Расширить CLI батч‑синтеза: используйте имеющийся `synthesize/synthesize_cli.py` (single/batch уже поддержаны), не меняя дефолтное поведение без аргументов.
- Микс: prefer `utils/audio_mixer.mix_segments` вместо петель `pydub.overlay` для больших проектов.

## 10. When Unsure
Держите изменения локальными, выделяйте утилиты вместо рефакторинга, и уточняйте только если требуется менять структуру данных сабов.

---
Дайте обратную связь, если фактические шаги расходятся с описанием или есть внутренние скрипты, которые стоит задокументировать.

# Copilot Chat Instructions

## Language
- Всегда отвечай на русском.
- Когда документируешь код, используй технический стиль, понятный разработчикам. Для документации выбирай формальный и точный язык. Используй Английский для кода, комментариев и технических терминов.
