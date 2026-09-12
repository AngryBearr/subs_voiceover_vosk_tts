# Задачи: надёжное сокращение полного файла

Статус на 12 сентября 2026: SH-00 и SH-01 реализованы и уже присутствовали в
HEAD `ec0cbaf` до этой задачи; SH-02 реализована и полностью проверена и
зафиксирована коммитом `c5bf383`; SH-03 реализована и проверена, но ещё
не закоммичена; SH-04 и все последующие задачи ожидают.

## Цель

Технически обработать полный JSON существующей утилитой можно уже сейчас. Но для обычного использования нужны понятная очередь нерешённых случаев, возобновление после сбоев и проверка на полных файлах.

Цель ближайшей работы — три полных файла с воспроизводимым результатом, а не новый большой перебор моделей. Последняя проверка baseline: 175 тестов пройдено за 3.38 секунды, exit 0; также прошли compileall и две чистые metadata-проверки. Новых платных вызовов моделей не было.

Уже есть сохранение промежуточных результатов, журналы запросов и расходов, контекстный аудит, построчная проверка вариантов в режиме C и проверка принадлежности профиля TTS. Возобновления запуска пока нет. Эти работающие части нужно использовать, а не переписывать.

## Приоритетные задачи

### [x] SH-00. Зафиксировать рабочую исходную точку — P0

Выполнено после явного разрешения пользователя: изменения и отсутствие секретов проверены, текущий код, тесты и краткие отчёты сохранены одним связным коммитом `6b973dae412a5b7fada49fd0ad46d413b75d34b0` (`6b973da`) с сообщением `feat(shorten): establish tested experiment baseline`. Коммит содержит 47 файлов кода, тестов и документации; `output`, модели, корпус и авторизация в него не вошли.

**Проверка:** 175 тестов пройдены за 3.38 секунды, exit 0; compileall и две чистые metadata-проверки завершились успешно. Платных вызовов моделей не было. Состояние можно восстановить по указанному SHA.

### [x] SH-01. Общий анализатор и короткие фразы — P0

**Implementation note (completed; historical verification):** The shared
planner/analyzer use 1.5× acceleration and a 200 ms reserve, classify 1–3-word
timing risks and exact-word repetitions, and provide opt-in original Vosk
preflight via `--preflight-model-path` with voice/text-mode options. Actual fits
need no LLM; failed or missing evidence remains unresolved. Fresh offline
verification recorded 215 tests passed and compileall exit 0, using fake
synthesis; no real Vosk or paid calls were made. The note that this verification
was not committed was historical; SH-01 is present in HEAD `ec0cbaf`.

Зависит от SH-00. Основные файлы: `utils/analyze_text.py`, `utils/shortening_plan.py`.

- Использовать один расчёт доступного времени и общие настройки: ускорение до 1.5×, резерв 200 мс. Сейчас старый анализатор использует 1.3 и 300 мс и пропускает часть коротких фраз; новый планировщик отдельно выделяет только фразы до двух слов.
- Для не помещающихся фраз из 1–3 слов добавить явную метку `short_timing_risk`. Короткая фраза, которая помещается, такой метки не получает.
- Считать слова одинаково во всех проверках. Старое поле `is_short_segment`, связанное с длительностью, не менять на другой смысл незаметно.
- «Пошли, пошли, пошли.» распознавать как кандидат на сокращение повторов: `repetition_candidate`. Такой текст можно отправить LLM с контекстом. Сам код слова не удаляет: повтор иногда передаёт усиление.
- «Да.» и «Я не знаю.» при превышении времени направлять в отдельный разбор. «Да, нет, да.» не считать точным повторением одного слова.
- Прогнозируемый риск не означает доказанную невозможность сокращения. Если доступна проверка Vosk, сначала измерить звук: фактическое попадание снимает риск без LLM. Непроверенные случаи не считать решёнными.
- Анализатор не меняет текст и таймкоды; он сохраняет причину, число слов, признаки повторения, доступное окно и источник оценки. Неверные тайминги помечаются отдельно.

**Готово, когда:** анализатор и планировщик согласованы; тесты покрывают перечисленные примеры, неверные тайминги и различие прогноза с фактическим измерением. Новые платные запросы не нужны.

### [x] SH-02. Не терять всю пачку из-за одной плохой строки — P0

Зависит от SH-01. Основной файл: `utils/shortening_runner.py`.

Перенести построчную проверку в обычное создание, проверку и исправление субтитров. Валидные соседние строки сохраняются; отсутствующие и неверные строки получают собственные ошибки. Чужие и неоднозначные идентификаторы не применяются. Повторно обрабатываются только нерешённые строки, в пределах заданного числа попыток.

**Готово, когда:** из шести строк при одной ошибочной сохраняются пять корректных; принятые результаты не генерируются заново; ошибки протокола не выдаются за смысловые отказы. Проверяется без платных запросов.

**Implementation note (SH-02, committed as `c5bf383` and fully verified):**
Ordinary generation, review, repair, and recheck now use bounded per-cue partial
parsing and shared extra recovery rounds. Valid JSON rows with identifiable IDs
are isolated, valid neighbours are checkpointed, and missing or invalid rows get
their own errors. A malformed envelope or repeated raw JSON keys fails the whole
response safely rather than allowing ambiguous application. Protocol failures
are not treated as semantic impossibility. The resilience matrix covers
six-row salvage, persistent malformed generation/review, mixed repair and
recheck rows, protocol misuse, transport failures, per-request usage
completeness, and source/export preservation.

The `max_repairs` value is a shared 0..2 extra-round budget (at most one
generation and one review request per cue in each extra round). Each repair
counts a corrective generation request, including one caused by a protocol
failure; review-only recovery does not spend a generation call. A malformed
judge response rechecks the same candidate. Returned transport failures are
recorded but are not automatically repeated.

Mode-C parsers and selection logic were not modified. Arm A is still evaluated
from the first generation attempt; it may remain unavailable if that attempt
failed while B recovered. The shared request ledger now also records returned
transport failures, so this change can affect failure reporting without changing
mode-C parsing or selection.

**Verification (12 September 2026):** `uv run --active --no-sync -m pytest -q`
reported `257 passed in 3.40s`, exit 0. `uv run --active --no-sync -m
compileall -q utils tests pipeline.py ui/gradio_app.py` exited 0. Both
`utils.opencode_clean --check` modes (with and without `--with-length-tool`)
exited 0 and returned `{"status": "ok"}`. All five CLI `--help` checks
(`shorten_subtitles`, `shortening_experiment`, `shortening_replay`,
`shortening_corpus`, and `vosk_calibration`) exited 0. `git diff HEAD --check`
exited 0. No paid or real model/TTS calls were made.

### [x] SH-03. Обычный запуск для одного файла и профиля — P0

Зависит от SH-01 и SH-02. Планируемая новая точка входа: `utils/shorten_file.py`; экспериментальные команды остаются совместимыми.

- Принимать произвольный полный SRT или JSON, включая SRT с BOM.
- Требовать явно выбранную модель; обычный запуск не включает всю матрицу моделей и два повтора.
- Явно выбирать профиль Vosk и проверять модель, голос, скорость и подготовку текста. Несовпадение останавливает работу до запросов к LLM.
- Использовать один способ подсчёта для планировщика, проверки ответа и инструмента модели. Нельзя незаметно смешивать исходные символы и символы после подготовки текста для TTS; в отчёте сохраняется `length_basis`.
- Использовать готовую калибровку, не обучая её заново на обрабатываемом файле. Старую оценку 13 символов/с разрешать только как явно обозначенный резервный вариант.
- Сохранять таймкоды, индексы JSON и неизвестные поля. Для SRT отдельно документировать его нумерацию.

**Готово, когда:** одна команда даёт новые JSON/SRT и отчёт; предварительный расчёт доступен без сети; несовместимый профиль отклоняется. Подключение в `pipeline.py` не обязательно для первой версии. Платный эксперимент пока не нужен.

**Implementation and verification note (SH-03, 12 September 2026):** The
single-model [`utils.shorten_file`](shorten_file.md) command processes one whole
SRT or JSON file, including UTF-8 BOM input, using one selected model for both
generation and review. Planning, local checks, and the optional length tool use
the normalized length basis, and the frozen Vosk profile must match the model,
voice, rate, and text normalizer. Offline `--dry-run` uses a lazy backend and
does not make LLM, network, Vosk, or synthesis calls; legacy experimental CLI
behavior remains compatible. Resume is not implemented yet.

Final verification: `uv run --active --no-sync -m pytest -q` reported `316
passed` and compileall for `utils tests pipeline.py ui/gradio_app.py` passed;
new and legacy CLI `--help` checks passed. A real existing Vosk model and frozen
profile passed the offline BOM SRT/JSON smoke check, while invalid voice and
rate values were rejected before output creation. No paid LLM or real synthesis
calls were made. The temporary local log, not a durable artifact, is
`/tmp/opencode/sh03-final-20260912/baseline.log`. SH-03 remains uncommitted; the
next task is SH-04.

### [ ] SH-04. Возобновление без повторной оплаты — P0

Зависит от SH-03. Использовать уже существующие промежуточные сохранения.

Продолжать запуск только при совпадении исходника, настроек, профиля и версии инструкций. Завершённые этапы не отправлять моделям заново. Журнал запросов дополнять, не переписывая историю.

Если запрос был отправлен, но ответ не сохранился, его состояние неизвестно. Такой запрос нельзя автоматически считать бесплатным, завершённым или безопасным для повтора. Повтор требует явного решения. Ровно одну обработку на стороне внешнего API гарантировать нельзя.

**Готово, когда:** тест прерывания и продолжения не меняет принятые строки и не повторяет завершённые запросы; изменённый исходник отклоняется; неизвестные запросы видны отдельно. Проверяется на имитации транспорта, без оплаты.

### [ ] SH-05. Удобная очередь ручной проверки — P0

Зависит от SH-02…SH-04. Планируемые результаты: редактируемый `review.json` и читаемый `review.md`.

Включать только нерешённые случаи: оригинал, соседний контекст, кандидаты, лимит и фактическую длину, причину, последнюю критику модели и предлагаемое действие. Отдельно показывать короткие фразы, ошибки таймингов и ошибки протокола.

SH-05 should consume SH-11's canonical outcome markers and queue only cues
whose outcome is unresolved or needs attention.

Команда применения правок принимает только явно одобренные известные `cue_id`, проверяет хеш исходника, формат и бюджет. Устаревшие и превышающие лимит правки не применяются молча. Остальные строки и таймкоды не меняются.

**Готово, когда:** любой нерешённый случай легко найти и исправить без поиска по большим журналам; правки применяются безопасно. Измеряем не только число правок, но и фактическое время пользователя, если оно записано. Новый интерфейс Gradio пока не требуется.

### [ ] SH-06. Проверить пользу арбитра при равных условиях — P1

Не блокирует первый технический пилот. Проверять одну гипотезу за раз:

- **H2:** текущая схема с критикой против исправления только по замечанию о длине — тот же первый кандидат и ровно одна дополнительная попытка в обеих схемах.
- **H3:** несколько вариантов только для сложных или отклонённых случаев, а не для всех реплик подряд.

До запуска фиксировать данные, контрольную схему, модели, инструкции, метрики и правило остановки. Смысл оценивают модели; главный агент не выбирает хорошие формулировки вручную. Разделять критические смысловые замечания и предпочтения стиля. Результат «данных недостаточно» допустим.

**Готово, когда:** польза и расход сравниваются на одинаковых случаях, разногласия видны, критерии не менялись после получения результатов. Платный опыт отдельно согласуется. Точность сохранения смысла нельзя гарантировать одними вердиктами LLM.

### [ ] SH-07. Архивирование и безопасная очистка — P1

До любого удаления подготовить раздельное хранение. Планируемые пути:

- `artifacts/results/<runid>` — постоянные результаты;
- `artifacts/corpus/<hash>` — одна сохранённая копия исходных данных;
- `output/runs/<runid>` — временные подробности запуска.

Эта структура одобрена пользователем, но SH-07 не реализована: каталоги не
созданы, миграция, архивирование и удаление не выполнялись. Очистка всегда
начинается с dry-run, затем требует отдельного подтверждения и проверенного
архива. Правило TTL в 30 дней пока только предложение и не является одобренной
политикой. Git хранит код и документацию; исходные субтитры, ответы моделей и большие результаты хранятся отдельно, откат Git не восстановит удалённые игнорируемые файлы.

При реализации большие артефакты исключить из Git. Сначала достаточно сжатия JSON/JSONL без потерь и устранения повторного хранения исходников; сложная база данных не нужна.

Сохранять итоговые субтитры, исходные ответы моделей, варианты, критику, инструкции, параметры, хеши, токены, тарифы и профили. Калибровочные WAV и важные проблемные примеры закреплять для хранения. Именно исходные ответы уже позволили восстановить 19 корректных строк без новой генерации.

Очистка сначала только показывает список файлов и освобождаемый объём. Удаление допускается после явного подтверждения и проверки восстановления архива. Не трогать исходники, модели, настройки и авторизацию OpenCode, цели символических ссылок, активные, незавершённые, неизвестные и закреплённые запуски. Не ломать ссылки других экспериментов.

**Предложение, ещё не включённое:** удалять промежуточные данные старше 30 дней только после проверенного архивирования. Постоянные результаты не удалять по сроку автоматически.

**Готово, когда:** из архива без сети восстанавливаются результаты и исходные ответы; повреждение архива блокирует очистку; повторная очистка безопасна. Старые каталоги сейчас не перемещаются и не удаляются.

### [ ] SH-08. Проверить три полных файла — P1

Начинается после SH-00…SH-05; SH-06 не является обязательным предварительным условием.

Заранее выбрать три файла разной длины и с разной долей коротких фраз, не использованные для настройки инструкций LLM. По возможности получить не менее 500 исходно проблемных реплик суммарно; если их меньше, указать фактическое число. Некоторые источники уже использовались для калибровки Vosk — такой пилот не является новой независимой проверкой TTS.

**Техническая приёмка:**
- один запуск создаёт JSON/SRT, отчёт и очередь проверки;
- нет потерянных или неверно назначенных реплик, изменённых таймкодов и ошибок инфраструктуры;
- завершённый запуск при продолжении не делает новых платных запросов;
- расходы и неизвестные состояния видны отдельно.

**Цель по ручной работе:** не более 15% нерешённых случаев в каждом файле отдельно. Знаменатель — все реплики, изначально требовавшие вмешательства, а не весь файл. В числитель входят короткие нерешённые фразы, ошибки и случаи, требующие альтернативного решения. Нельзя улучшать процент исключением неудобных случаев.

Провести заранее выбранную независимую модельную проверку; небольшая проверка пользователем возможна отдельно. Модельное согласие не является эталоном смысла. Если цель не достигнута, сохранить причины. Если по ошибкам пилота настраивались инструкции, эти файлы становятся набором разработки; для следующей независимой проверки нужны другие файлы.

**Готово, когда:** три файла реально обработаны, по каждому есть отдельные результаты, очередь и расход. Платный запуск отдельно согласуется. Решение коротких аудиопересечений пока не реализуется, но такие случаи не объявляются решёнными.

### [ ] SH-09. Отложенные задачи — P2

Объединение короткой реплики со следующей с допустимой задержкой, несколько движков TTS, профили TeraTTSv2/xVibePocketTTS и графический интерфейс остаются в [дорожной карте](tts_roadmap.md). Они не блокируют первый пилот. Edge-TTS не используется; установки и изменения требуют отдельного решения.

### [ ] SH-10. Финальный смысловой оценщик — P1 (будущая задача)

Это предложение, не реализация. Зависит от SH-03 и SH-05. Не блокирует первый технический пилот; может использоваться в SH-06 и SH-08 после выбора режима.

- Выбирается одна логическая роль сильного оценщика, но не обязательно один HTTP-запрос: большие файлы обрабатываются пачками с перекрытием.
- На входе: оригинал и сокращённая цель, исходный контекст соседей и финальный контекст соседей, чтобы проверять смысл через границы изменений.
- В полном режиме проверяются все сокращённые или обработанные цели и их стыки; неизменённые строки дают только контекст, а не повод заново переводить весь оригинал.
- В режиме выборки проверяется фиксированное подмножество, доступное текущей реализации.
- Для каждой цели выводится: `смысл сохранён`, `нужна правка` или `недостаточно контекста`, с обоснованием и указанием проблемного фрагмента; неоднозначное не принимается автоматически.
- Точная финальная модель и обычный режим по умолчанию пока не выбраны: решение зависит от стоимости и качества.
- Свежая сессия той же модели не является независимой проверкой семейства; где уместно, предпочтительно другое семейство, но конкретная модель ещё не выбрана.
- Независимый контроль пока выборочный, полного финального допуска сейчас нет. Вердикт модели не даёт гарантии 100%; бесконечных повторов и автоматических правок текста не будет.
- Приёмка будущего режима: все запрошенные ID покрыты в бюджете, границы контекста проверены, нет утечки идентичности или истории, неизвестные случаи попали в очередь, расходы отделены, ошибки пачек изолированы, а пробные тесты по возможности обходятся без модельных вызовов.

### [ ] SH-11. Per-subtitle outcome markers for review and safe reanalysis — P1

Depends on SH-03; coordinate with SH-05 and implement alongside it before SH-08.
This is a plan for a new capability, not a claim that an implementation already
exists.

Track two independent dimensions for every subtitle: whether its text changed,
and its outcome/evidence. The outcome vocabulary must distinguish `unchanged and
fits`, `accepted correction`, `needs attention/unresolved`, `invalid/error`, and
`not checked` (unverified). Never equate a modified cue with a solved cue, an LLM
acceptance with measured audio fit, or an untested cue with a resolved one.

Canonical metadata belongs in JSON or a sidecar, outside spoken text. For each
stable `cue_id`, retain the outcome and reason, the validation source
(`prediction`, `model`, `human`, or `measured`), measured timing as a separate
field, source-text, current-text, source-timing, current-timing, and profile
fingerprints, and relevant context needed to reuse a semantic decision. Expose
every cue's status in a
human-readable report. SH-05 must queue only unresolved cues.

An optional, separately named, review-only annotated SRT may show a clear
`NEEDS REVIEW` marker and map it by `cue_id`, but it must never alter the primary
subtitle, spoken text, or times. Annotations must never be fed to TTS or length
checks. Do not claim silent machine attributes in SRT; SRT has no such
non-standard metadata contract here.

The analyzer may skip repeated expensive processing only when recorded
approval/evidence matches the current input, text, times, configuration, profile,
and relevant context for that type of check. It must always retain lightweight
integrity checks and changed-neighbor-window checks. It must never skip an
unresolved cue merely because it was attempted or changed, or skip missing actual
measurement merely because a text-accept flag exists. Outdated evidence
invalidates the skip, and an explicit force-recheck option must be available.

**Готово, когда:** offline tests cover a mixed set of fits, changed-and-accepted,
unresolved, error, and unverified cues; reports clearly distinguish these states
while machine/TTS content remains unchanged; markers survive export through the
sidecar; changed text, timing, profile, or context invalidates cached evidence;
and analysis does not hide unresolved cues or remove them from the denominator.

### Known bugs — fixes deferred by user

**Audit note (12 September 2026; static code findings only):** the following
issues are recorded for later triage. No fixes or tests are claimed here, and no
automatic work has started.

- **BUG-01 — P0:** `runner._request` catches `CancelledError`, so subsequent batch
  calls may continue. A future fix must preserve unknown accounting and re-raise
  cancellation. Add a fake-cancellation test proving that no later calls occur
  and a checkpoint is not falsely marked complete.
- **BUG-02 — P1:** legacy `utils.shorten_subtitles._execute` can lose an
  incomplete child `run_state`, leaving the parent marked complete with exit 0.
  A future fix must propagate incomplete status and nonzero exit distinctly from
  an ordinary manual unresolved result. Test with a mocked child and no network.
- **BUG-03 — P1:** legacy `shorten_subtitles._execute` overwrites
  `runner.cost.projection_500` with `len(stages)` (four categories, not the
  request count) and drops `applicable`/`inapplicable_reasons`. A future fix must
  retain the runner projection and counters. Tests must compare the runner with
  the CLI report and preserve reasons and real counts.

The user will triage these bugs later; they do not take over the current order,
and no automatic work is requested. Recheck them before any paid runs. The
legacy diagnostic `timing_profile` identity limitation is tracked with SH-03;
do not reopen SH-03, which is now implemented with frozen calibration identity
checking in the new command. The legacy diagnostic interface is separate; the
new `shorten_file` command does validate identity.

## Порядок работы

SH-00 → SH-01 → SH-02 → SH-03 → SH-04 → SH-05 → SH-08.

SH-06 — по результатам первого пилота. SH-07 обязателен перед очисткой, но не должен оттеснять задачи качества. SH-09 отложен.

SH-11 accompanies SH-05 and must be implemented before SH-08; the main order and
the current next task, SH-04, remain unchanged.

## Коммиты и гипотезы

- Сначала один одобренный коммит рабочей исходной точки.
- Затем одна законченная задача — один проверенный рабочий коммит, например `fix(shorten): isolate malformed batch rows`.
- Перед платным экспериментом фиксировать коммит кода, версию инструкций и манифест. После эксперимента сохранять краткий результат отдельным коммитом документации при необходимости.
- Для каждой гипотезы записывать: что ожидаем изменить, с чем сравниваем, какие данные используем, как измеряем результат, когда останавливаемся, где код и результаты. Итог: подтверждена, не подтверждена или данных недостаточно.
- Коммиты создаются после явного разрешения пользователя. Откат кода через Git не возвращает удалённые игнорируемые результаты — для них нужен архив.

## Сколько места занято сейчас

Снимок `output` на 10 сентября 2026:

| Данные | Логический размер |
|---|---:|
| Всего | 264.80 МиБ |
| JSON | 173.68 МиБ |
| WAV | 62.52 МиБ |
| JSONL | 10.30 МиБ |
| SRT | 16.77 МиБ |
| MP3 | около 1.47 МиБ |
| Файлы `.log` | 23.1 КиБ |

Фактически на диске выделено около 283.65 МиБ. Основной объём дают многократные полные экспорты JSON, а не маленькие технические логи. Срочной нехватки места нет.

Есть 186 экспортов `subtitles.json`, из них 152 различных по содержимому, и 186 SRT, из них 151 различный. Точный объём, освобождаемый устранением дублей, отдельно не измерен. Нельзя считать все JSON ненужными копиями.

## Что было выполнено на момент исходного аудита

Проведён аудит без изменений данных и составлен этот план. На тот момент код не
менялся; модели и TTS не запускались; файлы не удалялись, не перемещались и не
сжимались; baseline-коммита ещё не было. Последний на тот момент тестовый
результат и измеренные объёмы не выдавались за новую проверку реализации.

Текущее решение на 12 сентября 2026: SH-00 и SH-01 реализованы и уже
присутствовали в HEAD `ec0cbaf` до этой задачи; SH-02 зафиксирована коммитом
`c5bf383`; SH-03 реализована, но ещё
не закоммичена; следующая — SH-04, а все последующие задачи ожидают.
Автоматический коммит не выполняется.
