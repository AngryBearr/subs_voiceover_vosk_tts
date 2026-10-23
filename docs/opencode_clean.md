# Чистый OpenCode для сокращения субтитров

Этот документ описывает только транспорт и изоляцию OpenCode. Он не меняет
глобальную конфигурацию Fusion. Подробные правила сокращения и измеренные
результаты находятся в [`shortening_memory.md`](shortening_memory.md).

## OpenCode 2.x: запросы пайплайна через подписку

С OpenCode 2.x (проверено на 2.1.3, 23 октября 2026) API сервера другое: пути под `/api/...`,
сессии 1.x (`/session/{id}/message`) больше нет. Поэтому `utils/opencode_transport.py` с
2.x не работает. Пайплайн `stt` использует `CleanOpenCodeServer.generate`: один запрос без
состояния через `POST /api/experimental/generate` с полями `prompt` и
`model: {providerID, id, variant}`. Системный промпт и текст запроса передаются одной строкой.
Количество токенов этот путь не возвращает. Проверка сервера при запуске для 2.x: `/api/info`,
`/api/agent`, `/api/plugin`, `/api/mcp`, `/api/model`. Какие модели идут этим путём,
задаёт `llm.opencode` в `stt_config.json` (см. `docs/stt_local.md`).

## Текущая конфигурация

Файл preset:

```text
config/opencode_clean/opencode/opencode.json
```

Его нужно выбирать через абсолютный `XDG_CONFIG_HOME`. Одного
`OPENCODE_CONFIG` недостаточно: это механизм слияния, а не полная изоляция.
`CleanOpenCodeServer` запускается из свежего нейтрального каталога, отключает
конфигурацию проекта, внешние skills и делегацию, запрещает все разрешения по
маске и не читает `AGENTS.md` репозитория. Это изоляция настроек и инструкций,
а не песочница ОС: процесс сохраняет права пользователя.

Активные модели и варианты:

- Luna `openai/gpt-5.6-luna`: `none`, `low`, `medium`, `high`;
- Sol `openai/gpt-5.6-sol`: только `medium`, используется как арбитр;
- DeepSeek вызывается отдельным прямым transport и в preset не добавляется.

Арбитр выбирается для каждого сообщения, а не как второй постоянно работающий
runtime. Данные субтитров трактуются как данные. Ответы требуют строгого JSON и
разбираются локально. Нативный structured output **не включён по умолчанию**:
его можно явно запросить через `native_structured_output` для отдельного вызова.
Это позволяет сохранить полезную ошибку провайдера и учёт usage, даже если
модель вернула JSON-текст вместо нативного объекта.

Инструмент `length_check_length` подключается только при явном
`enable_length_tool`/`--length-tools`. Он доступен для локальной проверки
символов, но его наличие не доказывает вызов и не заменяет вердикт Sol.
Инструментная переписка живёт только в памяти запроса; chain-of-thought вне неё
не сохраняется.

## Контракт окружения

В этом checkout правильная среда называется `subs_env`. Пример ниже — не
команда для запуска сейчас, а контракт дочернего процесса:

```bash
(
  REPO=/home/vladimir-barbuta/Projects/subs_voiceover_vosk_tts
  export XDG_CONFIG_HOME="$REPO/config/opencode_clean"
  export OPENCODE_DISABLE_PROJECT_CONFIG=1
  export OPENCODE_DISABLE_CLAUDE_CODE=1
  export OPENCODE_DISABLE_EXTERNAL_SKILLS=1
  export OPENCODE_PURE=1
  unset OPENCODE_CONFIG OPENCODE_CONFIG_CONTENT OPENCODE_CONFIG_DIR OPENCODE_PERMISSION
  unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE
  unset OPENCODE_DISABLE_DEFAULT_PLUGINS
  source "$REPO/subs_env/bin/activate"
)
```

Для Luna/Sol требуется уже существующая запись OpenAI типа `oauth`. Этот
процесс не выполняет login, logout или копирование `auth.json`; при отсутствии
OAuth следует остановиться. API-key fallback запрещён. Естественное обновление
OAuth-токена самим OpenCode возможно, поэтому документ не обещает, что файл
учётных данных никогда не изменится.

Пароль loopback-сервера передаётся только через `read -s` в переменную
окружения, не через командную строку, файл, историю или лог. Сервер должен
слушать только `127.0.0.1` и завершаться обычным сигналом процесса.

## Интерфейсы транспорта

`utils/opencode_clean.CleanOpenCodeServer` — используемый серверный класс;
старый `OpenCodeServer` для этого сценария не применять. Он создаёт свежий
нейтральный рабочий каталог и временную копию preset, проверяет OAuth до
запуска и отбрасывает вывод дочернего процесса. `utils/opencode_transport.py`
предоставляет:

- `create_session` и `delete_session` для отдельной сессии каждого логического
  запроса;
- `send_prompt_detailed`, который передаёт отдельные `model` и `variant`,
  собирает уникальные assistant steps и не считает POST дважды;
- необязательный `output_schema` и явный флаг нативного structured output.

Текущая совместимость проверена с установленным сервером OpenCode 1.18.31
15 сентября 2026 года: `opencode --version` вывел `1.18.31`, а локальный
clean-server `--check` вернул `status: ok`. Проверены health, варианты
провайдеров и OpenAPI checks; эти проверки не выполняли inference.
Это фиксирует фактическую совместимость, но не гарантирует семантическую
эквивалентность OpenCode 1.18.29 и 1.18.31. Исторические контракты и импорт
исторического source -> historical destination остаются без version bridge:
старые receipts нельзя переименовывать в 1.18.31.

Используемый сервер OpenCode 1.18.31 предоставляет:

- `GET /config` — разрешённые настройки;
- `GET /agent` — доступные агенты;
- `GET /provider` — подключённые модели и варианты;
- `GET /doc` — OpenAPI-контракт;
- `POST /session/{sessionID}/message` — сообщение с отдельными полями
  `model` и `variant`.

`send_prompt_detailed` возвращает типизированные записи с токенами, стадией,
ошибкой и отдельными полями output/reasoning. Нельзя объявлять
`usage.cost = 0` ценой платного OpenAI API: OAuth даёт usage и quota, но не
счёт-фактуру. DeepSeek считает стоимость своим прямым transport по тарифу из
его первичного отчёта.

## Что уже связано с экспериментом

Транспорт используется runner-ом для генерации Luna и проверки Sol medium,
включая свежие сессии, checkpoint и подробный ledger. `--length-tools` остаётся
опциональным. Проверенные числовые и аудио-материалы перечислены в
[`shortening_memory.md`](shortening_memory.md); там же отмечены ошибки native
structured output, отсутствие семантического ground truth и границы прогноза.

Проверка `/config`, `/agent`, `/provider` и `/doc` не должна делать inference
или печатать полный config/auth payload. Финальное число тестов итогового
набора здесь намеренно не фиксируется: его добавит основной агент после
финального запуска. Не копируйте секреты и не добавляйте ручные длинные
дублирующие инструкции в этот preset.
