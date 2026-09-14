# ПЕРЕДАЧА: продолжение SH08

## Current continuation block (14 September 2026; authoritative)

The current SH08 status is a completed first pass of the small matrix: 60
unique cases, two repetitions of the same 60 cases, and eight pairs. This is
not 120 independent observations. The 60-case denominator remains intact:
41 eligible cases, 18 valid zero-capacity cases, and 1 invalid source case;
the 18 valid zeros and the invalid case are not discarded. All 88 r3 requests
are complete and r3 has no unknown request. Repairs are only **PARTIAL** and
not uniform because the 300-unit conservative subscription cap was exhausted.

Detailed readable and machine results are
[`sh08_eightpair_results.md`](sh08_eightpair_results.md) and
[`sh08_eightpair_results.json`](sh08_eightpair_results.json). The first-pass
accept counts per repetition (each over 41 eligible cases) are:

| Pair | Rep 1 | Rep 2 |
|---|---:|---:|
| Luna → Luna | 23 | 21 |
| Luna → DeepSeek | 17 | 15 |
| Luna → Sol | 19 | 21 |
| DeepSeek → Luna | 8 | 8 |
| DeepSeek → DeepSeek | 8 | 8 |
| DeepSeek → Sol | 7 | 8 |
| Gemini → Sol | 19 | 16 |
| Muse → Sol | 21 | 23 |

The five invalid review rows are only the DeepSeek-arbiter-judging-Luna branch
(4 in repetition 1 and 1 in repetition 2). The local parser rejected them;
assume no semantic verdict. The known priced/actual total is `$0.1013308035`:
OpenRouter reported `$0.0344566035` and the conservative DeepSeek upper bound
is `$0.0668742`. Subscription usage is unpriced. Accounting is 39 paid HTTP
calls, 50 subscription logical calls, and 89 logical calls cumulatively,
with `50 × 6 = 300` subscription units plus 39 paid units = 339 total quota
units.

Main interpretation: Muse → Sol is a promising low-cost candidate for the next
comparison with Luna → Sol under the same arbiter: 44 versus 40 accepted out of
82 eligible observations (120 full-case observations including structural
skips), but a four-result difference is not a statistically established
winner. Luna → Luna also has 44 accepts, but a self-pair is not independent
validation. DeepSeek `off` as generator is overlength in 33/41 cases in each
repetition; this is a current length-compliance weakness, not global model
inferiority, and changing only the arbiter cannot fix it. A future targeted
low-thinking or local-length-repair experiment may help, but it is not
authorized and must not auto-expand the matrix. Gemini Flex is fast and cheap,
but it has more `cannot shorten` outcomes (13/20) than Muse (7/11) and Luna
(9/9); this records an observed pattern, not an assumption that Gemini is bad.

The confirmed Gemini configuration is the OpenRouter base model
`google/gemini-3.8-flash`, pinned to `google-ai-studio/flex` at the low
`$0.375` / `$1.875` per-million-token price (the same listed price as batch;
this was not an async batch test). Muse is
`meta/muse-spark-1.3-contributor` at `$0.10` / `$0.20`; its product-improvement
condition is user-approved. Muse and Luna are preferred next same-Sol
comparison candidates, not a final production selection.

No audio, full pilot, or independent audit was performed. Continue safely
offline with the 18 originals with zero predicted character capacity and local Vosk, then review meaning
before three full pilots; do not do additional TTS now. Further model calls
require new approval: the current 300-unit subscription subcap is spent, and
unused API cash does not authorize extra calls. Runtime/inference timings mix
stages and batch sizes and are not a universal benchmark per model.

Historical incidents: the r1 first Luna response was lost by a
visible/reasoning-counter bug; the user approved the retry and six prior units
were charged. The r2 cached Sol response was preserved and reconciled offline
as 55 cached requests into derivative r3, with no second paid replay. Raw final
status `blocked_unknown` was an exception-classification bug because
`BudgetLimited` inherits `MatrixError`, not unknown receipts. Corrected source
classification is true, archived raw status is unchanged, and the analysis
report derives `budget_exhausted`. Historical pre-fix checks (14 September
2026; superseded) recorded `561 passed, 5 skips`; targeted post-fix verification
was `92 passed`.

Latest final verification is recorded in
[`/tmp/opencode/sh08-final-verification.log`](/tmp/opencode/sh08-final-verification.log):
`pytest` reported `563 passed, 5 skipped in 48.64s`, exit 0; compileall exited
0; derived analysis validation matched existing outputs identically, exit 0;
10 JSON files validated and no secrets were found.

The latest relevant code commits are `4ca6997`, `c287848`, and `2b0c141`.
There is also an uncommitted terminal-status fix and final report; MAIN will
commit those, and no future commit ID should be guessed. Do not edit the
frozen [`sh08_live_protocol.json`](sh08_live_protocol.json), source/raw/hash or
approval files. Raw r3 and its analysis are authoritative; r1/r2 are preserved
historical records and remain ignored artifacts that still require transfer.
Selection, profile, and other frozen inputs are unchanged. Do not use the old
CLI resume path: the original r3 cannot normally resume after the code-hash
fix. Any future experiment requires a new approved contract and shared prior
limits unless a new budget is explicitly approved.

The continuation below is a **HISTORICAL snapshot**. Its earlier statements
that the matrix was planned, not implemented, or not run describe the prior
state and do not override this current block. SH08's full three-file pilots
remain incomplete, and SH10 remains unchecked.

## Historical prior continuation snapshot

Continuation update (14 September 2026): SH05/SH11 work was committed as
`344880b` (`feat(shorten): add review workflow and offline acceptance evidence`).
The offline r2 inventory is recorded in
[`sh08_matrix_preparation.md`](sh08_matrix_preparation.md); no model, network,
subscription, or paid-run permission was granted.

**Скопируй этот файл целиком и отправь принимающему агенту.** Текст самодостаточен:
ему не нужен текущий чат. Отвечай и веди работу простым русским языком. Это
передача состояния и план, а не сообщение о выполненном пилоте.

## 1. Цель и ближайшая работа

Продолжи проект сокращения субтитров. Ближайшая работа — подготовить честное
небольшое сравнение шести пар на одинаковых условиях. После выбора конфигурации
её нужно применить к трём полным пилотным файлам SH08.

Матрица и пилот пока **не реализованы и не запускались**. Не называй подготовку
матрицы, автономные проверки с имитацией или локальные пробы Vosk выполненной
матрицей или полным пилотом. Сначала подготовь протокол и кодовую основу, получи
необходимые решения и разрешения, затем запускай только согласованный объём.

## 2. Рабочее место и безопасные границы

- Корень проекта: `/home/vladimir-barbuta/Projects/subs_voiceover_vosk_tts`.
- Фактическая среда Linux.
- Сначала прочитай текущий `AGENTS.md`. Перед скриптами активируй `subs_env`.
- Запускай модули из корня проекта через `uv`.
- Базовая команда:
  `source subs_env/bin/activate`
- Проверки:
  `uv run --active --no-sync -m pytest -q`
- Проверка компиляции:
  `uv run --active --no-sync -m compileall -q utils tests pipeline.py ui/gradio_app.py`
- Осмотр состояния: `git status --short` и `git diff HEAD --stat`; новые файлы,
  ещё не добавленные в Git, осматривай отдельно.

Локальные тесты и при необходимости локальный синтез Vosk на уже имеющейся
модели разрешены. Не повторяй все технические прогоны без причины; после изменения
кода запускай подходящие контрольные проверки. Новые обращения к языковым моделям
запрещены до отдельного согласования, включая использование подписки. Установка
пакетов, изменение глобальных настроек и другие сетевые обращения тоже требуют
отдельного разрешения.
The earlier blanket warning in this handoff against staging, committing, and
pushing was superseded only for the user-approved SH05/SH11 commit, recorded as
`344880b`. Push remains unauthorized, and do not stage or commit without
separate user approval.

Preserve all uncommitted changes: do not use `reset`, `clean`, deletion,
overwrite, or `stash` in a way that could lose them.

Earlier references to commit `a958052`, uncommitted SH05/SH11 work, and the
prior working-tree state are historical. The SH05/SH11 work is recorded as
`344880b`; later documentation commits may follow. Most result files, models,
and source data remain Git-ignored. Check ignored artifacts by their exact
paths and ensure access to them; zero search hits under Git ignore rules do not
prove that a file is absent. The r2 raw artifacts and models must be transferred
separately, and their presence does not mean that they are committed.

## 3. Порядок чтения

Прочитай документы в таком порядке:

1. `docs/sh08_model_matrix_proposal.json` — запрошенные шесть пар;
2. `docs/shortening_model_pairs.md` — фактическая история;
3. `docs/sh08_pilot_plan.md` и `docs/sh08_pilot_manifest.json` — зафиксированная
   подготовка трёх файлов;
4. `docs/shortening_tasks.md` — SH06, SH08, SH11 и известные ошибки;
5. `docs/shortening_real_acceptance.md`;
6. `docs/vosk_timing_diagnostics.md`;
7. `docs/shortening_review.md`.

`sh08_model_matrix_proposal.json` — описание, а не исполняемая команда и не
текущая публичная возможность командной строки. Его статус `DRAFT_NOT_EXECUTED`,
авторизация везде `false`, а 24 случая — предложение, не зафиксированная выборка.

## 4. Что уже есть в коде

Основные файлы реализации: `utils/shorten_file.py` (командная строка для одной
модели), `utils/shortening_runner.py` (`run_trial`, отдельные роли генератора и
арбитра — модели, которая проверяет сокращённый текст, — внутри),
`utils/shortening_resume.py` (зафиксированный снимок настроек,
сохранённые ответы и сведения о запросах, запросы без надёжно сохранённого
результата и повторная попытка), `utils/shortening_review.py` (статусы реплик,
очередь проверки, привязка проверки к тексту и настройкам),
`utils/shortening_review_io.py` (наборы), `utils/apply_review.py` и
`utils/review_subtitles.py`.

Текущая командная строка не имеет `--arbiter`. В ней отсутствует ограничитель
расходов и обычный мост командной строки для аудита. Нельзя напрямую передавать
предложение в формате `JSON` как настройки запуска. Предложенный полный
идентификатор DeepSeek — форма командной строки; для поставщика `deepseek`
фактический `ModelSpec` должен использовать хвост `deepseek-flash` через
существующий разбор параметров. Нужен отдельный адаптер эксперимента для
смешанных ролей. Изменение вспомогательных функций запуска, по которым считается
`hash` — контрольная сумма для обнаружения изменённых файлов, — делает старые контракты
возобновления несовместимыми: не подделывай зафиксированный снимок настроек и не
меняй модели в старом запуске.

## 5. Намерение пользователя и история моделей

Пользователь хочет сравнить две модели сокращения — Luna и DeepSeek, а для
проверки результата — Luna, DeepSeek или Sol. Все шесть пар обозначай по-русски как
«сокращает -> проверяет»:

- Luna -> Luna;
- Luna -> DeepSeek;
- Luna -> Sol;
- DeepSeek -> Luna;
- DeepSeek -> DeepSeek;
- DeepSeek -> Sol.

Предлагаются настройки `Luna medium`, `DeepSeek Flash off` и `Sol medium`, но
пользователь ещё не утвердил их. В истории реально найдены только Luna -> Sol и
DeepSeek -> Sol. Среди прежних проверок была общая выборка из 48 реплик с двумя
повторами. Остальные четыре пары в проверенных результатах не найдены.

Старые запуски использовали другой расчёт допустимой длины: 13 символов в секунду,
а не нынешний калиброванный профиль. Для честного сравнения заново проверь все
шесть пар на одинаковых текущих условиях. Использование модели в отдельной
проверке качества не означает, что она уже работала арбитром в нужной паре.
Sol и Luna относятся к одному семейству моделей: новый чат сам по себе не делает
проверку независимой. Победитель пока неизвестен.

## 6. Честное небольшое сравнение шести пар

Отдельно заморозь набор для разработки и выбора (например, предложенные 24
случая); это **не** три SH08-пилотных файла. До финансирования реши с
пользователем размер, повторы, варианты и бюджет.

Для всех шести пар одинаковыми должны быть идентификаторы реплик, исходный текст,
окна, профиль, лимиты, запрос и его версия, контекст, инструменты, бюджеты и
лимит дополнительного исправления. Сначала каждый генератор один раз создаёт
первую версию сокращённого текста. Затем все три арбитра получают одну и ту же
сохранённую первую версию. Повторное использование допустимо только с записью о
происхождении результата; нельзя притворно использовать старый запуск с
возобновлением.

На каждую ветку разрешено максимум одно дополнительное исправление и только тем
же генератором. Сохранённые ответы и сведения о запросах, стоимость и статусы
запросов без надёжно сохранённого результата храни раздельно. Вариант с таким же
бюджетом и исправлением только длины для SH06 можно добавить, если вопрос именно
«помогает ли арбитр», а не «какой арбитр лучше». Это отдельный протокол, а не
скрытое расширение матрицы.

Сравнивай долю пригодных результатов, сохранение смысла, разногласия судей,
нерешённые случаи, аудио (если измерялось), стоимость и время выполнения. Сырой
счёт принятых результатов не является семантической истиной. При двух семействах
моделей смешанная пара не даёт автоматически независимую третью семью; протокол
независимой проверки ещё нужно явно решить.

Шесть пар на три полных файла означают 18 платных запусков. Это не первый шаг и
не разрешённый объём.

## 7. Разрешения и нулевая символьная ёмкость

Последний ответ пользователя запрашивал варианты и проверку Vosk, но не давал
согласия на бюджет. Исторические `$0.5` не являются новым бюджетом. Нельзя
говорить, что стоимость 0 означает бесплатную подписку OpenCode.

До первого обращения к модели получи явное подтверждение конкретных моделей,
вариантов, выборки, ограничения расходов и лимита обращений, а также одобрение
независимой проверки. Сначала реализуй и протестируй ограничитель расходов и
обработку запроса без надёжно сохранённого результата. Если разрешения нет,
закончи автономную подготовку и верни список оставшихся вопросов; не трактуй
отсутствие запрета как разрешение на запуск.

Можно читать и редактировать код, запускать локальные тесты и использовать уже
имеющиеся локальные ресурсы Vosk; не повторяй 138 уже валидных синтезов только
ради «воспроизведения» без причины.

Обработка реплик с нулевой допустимой длиной ещё не согласована и не реализована
отдельным правилом. В первых трёх файлах таких реплик 770: 646 уже отмечены как
рискованные по времени, ещё 124 — нет. Всего исходно проблемных случаев 2853,
включая два случая невалидных таймкодов.

Не исключай эти реплики из статистики и не трать запросы на заведомо невыполнимое
ограничение длины. При этом прогноз «ноль символов» не доказывает, что исходная
речь не поместится: локальный синтез иногда показывает обратное. Сначала учти
имеющиеся измерения. Если проблема не решена, оставь реплику в очереди и честно
укажи причину. Правило обработки нужно определить до платных запусков.

BUG02/BUG03 — отложенные требования в старом `shorten_subtitles._execute`;
обычный маршрут `shorten_file` их не вызывает. Не расширяй сейчас очистку и не
вызывай старый маршрут вслепую. H4 — будущий эксперимент; полный оцениватель SH10
не реализован и не должен быть помечен как завершённый.

## 8. Завершённые доказательства SH05/SH11

Техническая автономная проверка без сети завершена для четырёх полных файлов `SRT`:
1523 + 1007 + 995 + 894 = 4419. Выполнена конверсия в `JSON` с кодировкой `UTF-8`
и маркером `BOM`. Сохранены весь текст, время и индекс; исключение касается
только явно смоделированных изменений текста в тестовых заготовках. Случаи без
надёжно сохранённого результата и плохие результаты отклонялись; устаревшие
полные применения тоже отклонялись; все нерешённые случаи ставились в очередь.

Авторитетные файлы: `output/sh05_sh11_real_acceptance_20260913_r12/report.json`
и `output/sh05_sh11_real_acceptance_20260913_r12/verification.log`. Все поля
`product export_errors` равны `null`. Три исходных случая с непригодной нулевой
длительностью остались видимыми предупреждениями: два в S44E07 и одно в S44E11. Не
выдумывай ошибки экспорта.

Тестовый модуль содержит 7 тестов: 5 настоящих проверок, запускаемых по явному
включению, и 2 быстрые проверки вспомогательного кода и ограничителя, а не 7
файлов.
Дополнительный полный запуск для 995 реплик использовал имитационный транспорт:
12 обращений при свежем запуске и 0 новых при возобновлении. Это не настоящее
доказательство качества языковой модели. Правки тестовых заготовок человеком не
являются пользовательским одобрением смысла. Корни приёмочной проверки `r1-r11`
устарели; авторитетен `r12`.

Новая обычная проверка уже зафиксирована в
`/tmp/opencode/sh08-handoff-verification.log`: `471 passed, 5 skipped, compileall exit 0`.
Старые логи с `469/4 skipped` считать устаревшими. Настоящая приёмочная
проверка по умолчанию намеренно пропускается; команда для явного включения:
`SUBS_REAL_ACCEPTANCE=1 SUBS_REAL_ACCEPTANCE_OUTPUT="$PWD/output/sh05_sh11_real_acceptance_rerun" uv run --active --no-sync -m pytest -q tests/test_shortening_real_acceptance.py -s`

Вспомогательная программа сама выбирает свежую папку; проверка занимает около 9
минут и не должна повторяться без причины. `tests/shortening_real_acceptance_manifest.json`
фиксирует исходные данные, модель и профиль; на другой машине допустим
`SUBS_REAL_CORPUS_ROOT`. Не ослабляй ожидаемые контрольные суммы ради зелёного
теста.

## 9. Что установлено про Vosk

Использованы модель `models/vosk-model-tts-ru-0.10-multi`, голос 8, скорость речи
1.5, знаки препинания и зафиксированный профиль
`output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json` с
контрольной суммой `SHA-256`
`eab509922c511ed0dbd2b3948d5a0cf55de78dc11ab6127e9c59443f4cabcb59`. Не меняй
модель, нормализатор или профиль без новой явной задачи калибровки.

В начальной проверке 9 случаев риска: для старого ограничения 13 `cps` (символов
в секунду) `fits=9`,
для текущего ограничения с защитой `fits=0`, фактически `fits=5` — это не «4
случая, ошибочно признанных безопасными текущим ограничением». Отдельно до аудио
были заморожены 30 случаев, помещающихся при текущем ограничении: 15 с точным
ограничением числа символов и 15 с заданным запасом; фактически 30/30 помещаются,
несоответствие заголовка=0, сеть/платные вызовы=0. Это не глобальная гарантия;
пересечение калибровки не исключено. Одна фактическая длительность превысила
защищённый прогноз на 23 мс, но всё ещё уложилась в окно.
Смотри `output/sh08_preparation_20260913_r1/` и
`output/sh08_predicted_fit_check_20260913_r3/` (`report.md`, `report.json`,
`selection.json`, `verification.log`). В накопленных технических повторах 138
локальных реплик Vosk, но это не 138 независимых выборок: итоговая проверяемая
выборка — 30 реплик, которые расчёт считал помещающимися, и 9 отдельно выбранных
рискованных реплик. Не подбирай профиль и текст вручную ради хорошего результата.

## 10. Замороженные SH08-файлы и критерий

Исходные файлы находятся в `/home/vladimir-barbuta/Downloads/subs_found_for_testing`:

- `Australian Survivor S01E03_ru.srt` — 1523 реплики, отобрано и в начальной очереди 1219;
- `S01E02.srt` — 1007 реплик, отобрано и в начальной очереди 773;
- `Survivor.S44E07.srt` — 995 реплик, отобрано 859, начальная очередь 861.

Итого в первых трёх 3525 реплик, исходно требующих вмешательства 2853, а не
старое номинальное 698. Их контрольные суммы заморожены в
`docs/sh08_pilot_manifest.json`; не вставляй длинные контрольные суммы в отчёт,
если ссылка на манифест ясна. `Survivor.S44E11.srt` (894,
контрольная сумма `302a10...`) — четвёртый исторический файл для разработки и
опорной проверки, не пилот SH08.

В проверенных материалах не найдено подтверждения, что первые три файла
использовались для настройки инструкций языковым моделям. Однако единого полного
журнала такого использования нет, поэтому это ограниченный вывод, а не гарантия.
Часть данных уже использовалась при калибровке Vosk: новую независимую проверку
качества синтеза здесь заявлять нельзя. Не используй эти три файла для подбора
инструкций в малом сравнении — сохрани их для последующего полного пилота.

Цель — не более 15% нерешённых случаев **в каждом файле** от исходного числа
исходно проблемных реплик; это число включает непригодные случаи, короткие случаи
и ошибки, а не все исходные реплики. Провал цели сообщай честно, не выбирай
удобные случаи. Независимая проверка модели сейчас описана для 24 изменённых
реплик на файл, максимум 72, по отбору с фиксированным начальным состоянием; это
не полный SH10, и согласие моделей не является истиной.

## 11. Реализуй после решений пользователя

1. Проверь рабочее дерево, перечисленные документы и обычный набор проверок и
   компиляции.
2. Коротко вынеси пользователю неоднозначности протокола, модели и бюджета.
3. Сначала раскритикуй план, потом меняй код.
4. Реализуй путь сравнения смешанных ролей, запись о происхождении общей первой
   версии сокращённого текста, изоляцию веток, безопасность расходов и запросов
   без надёжно сохранённого результата, а также адаптер независимой выборки.
5. Прогони модульные тесты и реальные автономные копии случаев до любого
   финансируемого обращения.

Минимальные обязательные проверки новой реализации: несоответствие исходного
текста, контекста или профиля должно блокироваться до обращения к серверной
части; все три арбитра получают одинаковую первую версию текста; дополнительное
исправление выполняется не более одного раза; известные сохранённые ответы и
сведения о запросах повторно не тарифицируются; возобновление после запроса без
надёжно сохранённого результата блокируется; ограничения должны безопасно
отказывать при превышении лимитов модели, инструмента, повторных попыток и при
неизвестном расходе у поставщика; тестовые заготовки не обращаются к сети;
исходный `JSON` сохраняет время и индекс; все нерешённые случаи учитываются;
приёмка модели и фактические аудиоданные имеют разные статусы.

Не повторяй весь набор из 138 технических повторов синтеза без необходимости,
но после изменения кода запускай нужные контрольные случаи. Только после явного
разрешения на платные обращения выполни ровно одобренное небольшое сравнение,
проверь результаты без ручного смыслового отбора, выбери следующую конфигурацию
вместе с пользователем и затем переходи к трём полным SH08-файлам. Для полного
пилота сохраняй отдельные для каждого файла `JSON`, `SRT`, отчёт, очередь и
стоимость; возобновление должно показывать 0 новых платных обращений для уже
завершённых случаев. SH08 не завершён, пока нет трёх настоящих полных запусков и
проверок: автономная проверка и проверка с имитацией не считаются запуском в
рабочей среде.

## 12. Как отчитываться

Остановись и честно сообщи об отсутствующем внешнем ресурсе или разрешении; не
подменяй доказательство ненастоящей запасной подменой. В финале верни ссылки или
пути к результатам, фактический протокол, список решений, которые нужны
пользователю, и ограничения. Разделяй: что сделано, как проверено, что это
означает и что рекомендуется дальше. Не заявляй полный пилот на основании одной
подготовки.
