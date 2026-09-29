# Задачи: надёжное сокращение полного файла

## Resume here — 30 September 2026

**Objective:** finish and assess the Australian Survivor S01E03 Russian episode
before the other two SH-08 full-file pilots. The single-file Muse-low generator /
Sol-6-high arbiter path, private HTML, ledger, guarded budget and resume, and
review queue are implemented and covered by offline tests. The *episode* is not
finished or live-verified with the newer windowed scheduler; neither semantic
accuracy nor audio fit has been established. SH-10 final-context semantic
evaluation, separate audio verification, and the three full pilots remain
incomplete.

**Preserve this checkpoint:**
[`output/openrouter_episode_20260929_sol6_fresh/`](../output/openrouter_episode_20260929_sol6_fresh/)
contains the real [partial HTML](../output/openrouter_episode_20260929_sol6_fresh/shortening.html),
manifest, requests, calls, exports and **500 generated proposals**. Of 1,523
source cues, 1,219 were selected, 124 measured originals fit; 1,095 are in the
review queue, with **zero arbiter-accepted corrections** in this snapshot.
The journal has 121 starts, 120 completions and one unresolved start
`c966047a268147149ab9976bd511589a`. Accounting from the read-only journal
is $0.073219707 known reported USD and $0.319488 retained unknown-cost
reservation. The older HTML checkpoint shows only 120 attempts and $0.212992
unknown reservation; it is **stale**, not the current accounting. These are
reservations, not confirmed charges. The older recovery run has separate,
potentially overlapping receipts: never sum its spend into this run.

**Next engineering step (not done):** implement an *explicit* migration/reuse
path from this frozen stage-major snapshot into the new windowed processing,
with an auditable request ledger. Reuse the 500 saved proposals without
duplicating completed requests; reconcile the unresolved start without dropping
its unknown-cost reservation or sending a duplicate request. Fail closed on
the old three-unknown limit and frozen runtime module-hash mismatch; do not
edit the old manifest or silently reinterpret it as a new contract. Verify
offline against saved state first. Do **not** auto-resume, auto-retry, or start
a replacement live run; any future live continuation requires a separate
decision and authorization. The [command and offline status guide](shorten_file.md#inspect-saved-status-offline)
describes inspection, not permission to execute.

**Implemented for future fresh runs only:** up-to-12-cue generation/review
windows, graceful signal/deadline stop before the next request, at most one
explicit unknown-start replacement per logical request, retained unknown
reservation, and a finite physical attempt cap (default 1,000) with the same
$10 monetary guard. These changes have not been tested by another live episode.
Offline saved-state review building changed from 9.01 s to 0.67 s with an
identical canonical state; that does not prove a live speedup. Existing offline
verification logs record 798 passed / 5 skipped (full suite), 110 passed /
5 skipped (review performance group), and 143 passed (cancellation guards).
No paid model, network, or TTS calls are needed to resume engineering work.

**Keep this the one active plan:** after meaningful work, update this checkpoint
and append one short dated history entry with what changed, what was actually
checked, what remains blocked, and links to the relevant command guide or real
artifact. Leave older dated evidence intact; do not create a parallel active
plan. The historical experiment memory is [separate](shortening_memory.md).

### Compact dated history

- **15 Sep:** bounded Muse-low/high comparison and zero-capacity Vosk checks
  recorded in [SH-08 evidence](sh08_muse_reasoning_20260915.md) and
  [timing notes](sh08_zero_timing_20260915.md); three full pilots not closed.
- **20 Sep recovery:** older real [partial report](../output/openrouter_episode_20260920_recovery/shortening.html)
  has 397 accepted and 698 unresolved of 1,219 selected; costs and frozen
  contract are separate from the fresh Sol-6 run.
- **29 Sep:** fresh episode interrupted during generation; saved 500 proposals,
  no review verdicts, one outstanding request start. Offline work added
  windowing, bounded stop/retry policy and faster identical review-state build;
  no second live run verified these changes. See [the run guide](shorten_file.md)
  and the real [partial report](../output/openrouter_episode_20260929_sol6_fresh/shortening.html).
- **30 Sep:** refreshed this checkpoint for a safe handoff and corrected the
  [experiment memory](shortening_memory.md) pointer to Sol 6 high. Documentation
  and commit-candidate inspection only; no new live run, TTS or inference.

## Current production direction

The primary objective is user-facing score stability and a simple CLI with
complete cost and result reporting. Optional experiments are parked in the
single [`future experiments backlog`](future_experiments.md), not kept in the
active task list. Existing code, real artifacts, history, and module paths
remain in place; this documentation change does not move or remove anything.

The single-file OpenRouter CLI is implemented with Muse
`meta/muse-spark-1.3-contributor` at low mode as generator and Sol
`openai/gpt-6-sol` at high mode as arbiter. It has a guarded budget, reported
USD and unknown reservations, guarded resume, private HTML, SRT, and a review
queue. This is implementation and offline test evidence, not proof of semantic
or audio quality. SH-10 remains unchecked and unimplemented.

**Fresh Sol 6 episode run (29 September 2026):** The single authorized $10
run was interrupted before generation finished. The launcher had a
6,720-second SIGINT watchdog. The runtime snapshot was recorded at 19:54:23
+03:00 and the log's `KeyboardInterrupt` at 21:46:24 +03:00 (about 6,720
seconds later). The process was still running after the parent tool had been
interrupted: this timing is consistent with the watchdog, not the earlier tool
interruption, although the traceback alone cannot identify the SIGINT sender.
Its [private partial HTML report](../output/openrouter_episode_20260929_sol6_fresh/shortening.html)
and [SRT](../output/openrouter_episode_20260929_sol6_fresh/subtitles.srt)
are saved. All 1,523 source cues retain their IDs and timestamps; of 1,219
selected cues, 124 measured originals fit, 500 have generated proposals, and
1,095 remain in the review queue. Snapshot statuses: 800 pending, 282
short-timing, 12 error, 1 manual.
The runner schedules the **entire generation pass before any review**: the
500 saved proposals therefore have no arbiter verdict or accepted correction.
There are 120 completed generation requests and one started request without a
completion. The read-only journal accounts for $0.073219707 reported USD,
$0.319488 reserved against three unknown-cost attempts (including the
interrupted start), and 121 physical attempts. These are **not** a known final
invoice. The HTML/report checkpoint predates the last start and shows 120
instead of 121 attempts; its budget is stale. The existing offline publisher
has no journal-accounting notice option, so saved artifacts were not rewritten
to disguise the distinction. Do not automatically retry the unknown request,
resume, alter its frozen contract, or start a replacement run. The user
authorized finite retries in general, but the frozen primary CLI cannot retry
an unknown journal start. Reconcile it before separately approved recovery.
The SH-08 three-pilot requirement and SH-10 remain open.

The cap for the **new fresh whole-episode run is $10**; no paid preflight has
been performed for it. Prior recovery spend is separate, and overlapping
predecessor/recovery receipts must not be counted twice. The earlier approved
cap was **$10 total** for preflight model calls plus one
`Australian Survivor S01E03_ru.srt` episode (1,523 cues, approximately
75 minutes); it is distinct from the Jev trial and old subscriptions. A real
episode run and a separate recovery directory now exist, but the pilot is
incomplete. The preparation source is the exact
[`sh08_pilot_manifest.json`](sh08_pilot_manifest.json) file, whose SHA-256 is
`dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`.
That manifest remains a preparation-only historical record, not a ledger of
the later live execution. Costs for the recovery directory must not be added
to those of its predecessor: receipts can overlap.

**Verified saved state (29 September 2026):**
[`output/openrouter_episode_20260920_recovery/shortening.html`](../output/openrouter_episode_20260920_recovery/shortening.html)
is the existing real partial report. The recovery snapshot has 397 accepted
corrections and 698 unresolved of 1,219 selected targets. Its frozen budget
reports $1.74572717418 known USD and $4.85576 unknown reservation after a
completed transport-error request; there are no outstanding journal starts.
It remains incomplete, not a completed SH-08 pilot or a project-wide spend
total. The offline read-only `utils.shortening_status` diagnostic confirms the
unknown cost and also flags a changed frozen module hash. Reconcile billing
and review the immutable resume contract before any explicitly authorized
continuation; do not rewrite saved manifests or auto-retry failed calls.

The updated policy applies to **future fresh** output directories, not the
historical run. Sol 6 high / Muse low now schedules windows of up to 12 cues:
generation, review, one bounded repair/recheck, then the next window. The
first SIGINT/SIGTERM or an optional run deadline stops before another request;
an in-flight request has its existing 300-second client timeout and 16,384
output-token cap. Explicit new-contract unknown-start retry is limited to one
replacement, and its old unknown reservation remains under the frozen budget.
The new default permits unknown attempts up to the finite physical request
limit (1,000), subject to the same $10 monetary guard; the old run retains its
three-unknown contract. No new live run or quality result has tested this
updated scheduling. The interrupted run above establishes neither live
semantic quality nor a final invoice.

Saved real request timing (120 completed generation requests): median 26.04 s,
nearest-rank p90 48.58 s, maximum 70.02 s, total 3,439.39 s; 118 successes
and two HTTP errors. Gaps between completed requests have median 23.45 s,
p90 23.96 s and total 2,789.3 s. The manifest-to-first-request interval was
411.27 s. The journal does not attribute these gaps to Vosk, metadata lookups,
or checkpointing. An offline profile of `build_review_state` on the saved
1,523-cue state found 3,046 per-cue `_context` calls taking 42.64 s of a
44.72 s profiled pass. Indexing chronological neighbors once per source and
current snapshot reduced an unprofiled state build from 9.01 s to 0.67 s
on this saved input; the complete canonical state (including fingerprints,
snapshot ID and queue) stayed identical. This local build improvement does
not measure a live request-gap or end-to-end speedup; the old episode and its
stale report remain unchanged.

The SH07 snapshot is partial and verified, with no data deletion; it remains
open rather than closed. The existing SH06 checks and SH08 artifacts remain
historical evidence and are not replaced by this plan. README is absent; no
new mass documentation is planned beyond this required backlog.

## Current dated update (15 September 2026)

The SH08 small comparison first pass is complete: eight pairs over the same 60
unique cases in two repetitions. This is 60×2, not 120 independent cases.
First-pass results and the machine record are in
[`sh08_eightpair_results.md`](sh08_eightpair_results.md) and
[`sh08_eightpair_results.json`](sh08_eightpair_results.json). The 18 valid
zero-capacity cases and 1 invalid source case remain in the full 60-case
denominator. Repairs are partial and non-uniform because the 300-unit
subscription cap was exhausted. This does not complete the SH08 full-pilot
checkbox or SH10.

The bounded actual Muse low/high comparison is now complete over the same 60
cases in two repetitions. Its independent per-case accepted-and-fit summary,
costs, receipt latency, and current duplicate denominators are in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md) and
[`comparison summary`](../output/sh08_muse_reasoning_comparison_20260915/summary.json).
This does not close SH08: three full-file pilots, the full-file Muse adapter, and
a hard budget guard remain blocked. No further model, network, or TTS calls are
authorized by this record.

The offline local-Vosk check of the 18 originals with zero predicted character
capacity is complete: 1 measured fit and 17 overruns. The next safe work is the
offline quality decision before any three full pilots. Do not make additional TTS
or model calls now. New model calls require fresh approval; unused API cash is
not permission to continue.

**Prior final verification (14 September 2026):** The
[`verification log`](/tmp/opencode/sh08-final-verification.log) records
`pytest` with `563 passed, 5 skipped in 48.64s` and exit 0; compileall exited 0;
derived analysis validation matched existing outputs identically, exit 0; 10
JSON files validated and no secrets were found.

The status prose below is retained as a historical snapshot where it describes
the pre-execution plan.

Статус на 14 сентября 2026: SH-00 и SH-01 реализованы и уже присутствовали в
HEAD `ec0cbaf` до этой задачи; SH-02 реализована и полностью проверена и
зафиксирована коммитом `c5bf383`; SH-03 реализована, проверена и зафиксирована
коммитом `9a1ddd9`; по запросу пользователя SH-04 зафиксирована коммитом
`a958052`; SH-05 и SH-11 реализованы, покрыты модульными тестами и прошли
завершённую техническую real offline acceptance в commit `344880b`
([документальная запись](shortening_real_acceptance.md)); continuation inventory
is linked in [SH08 preparation](sh08_matrix_preparation.md).
Следующая задача — SH-08, для неё требуется отдельное разрешение на платный
технический пилот; её чекбокс здесь не отмечен.
Историческая матрица фактических generator→arbiter запусков и неисполненное
предложение новой матрицы вынесены в [короткую сводку](shortening_model_pairs.md)
и [JSON-предложение](sh08_model_matrix_proposal.json); диагностика Vosk — в
[отдельной записи](vosk_timing_diagnostics.md). Это не разрешение на платные
запуски.

## Цель

Технически обработать полный JSON существующей утилитой можно уже сейчас. Но для обычного использования нужны понятная очередь нерешённых случаев, возобновление после сбоев и проверка на полных файлах.

Цель ближайшей работы — три полных файла с воспроизводимым результатом, а не новый большой перебор моделей. Последняя проверка baseline: 175 тестов пройдено за 3.38 секунды, exit 0; также прошли compileall и две чистые metadata-проверки. Новых платных вызовов моделей не было.

Уже есть сохранение промежуточных результатов, журналы запросов и расходов, контекстный аудит, построчная проверка вариантов в режиме C и проверка принадлежности профиля TTS. SH-04 добавляет безопасное возобновление запуска к этим работающим частям, а не переписывает их.

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
behavior remains compatible.

Final verification: `uv run --active --no-sync -m pytest -q` reported `316
passed` and compileall for `utils tests pipeline.py ui/gradio_app.py` passed;
new and legacy CLI `--help` checks passed. A real existing Vosk model and frozen
profile passed the offline BOM SRT/JSON smoke check, while invalid voice and
rate values were rejected before output creation. No paid LLM or real synthesis
calls were made. The temporary local log, not a durable artifact, is
`/tmp/opencode/sh03-final-20260912/baseline.log`. SH-03 was committed as
`9a1ddd9`; the next task was SH-04.

### [x] SH-04. Возобновление без повторной оплаты — P0

Зависит от SH-03. Использует существующие промежуточные сохранения.

Продолжать запуск только при совпадении исходника, настроек, профиля и версии инструкций. Завершённые этапы не отправлять моделям заново. Журнал запросов дополнять, не переписывая историю.

Если запрос был отправлен, но ответ не сохранился, его состояние неизвестно. Такой запрос нельзя автоматически считать бесплатным, завершённым или безопасным для повтора. Повтор требует явного решения. Ровно одну обработку на стороне внешнего API гарантировать нельзя.

**Готово, когда:** тест прерывания и продолжения не меняет принятые строки и не повторяет завершённые запросы; изменённый исходник отклоняется; неизвестные запросы видны отдельно. Проверяется на имитации транспорта, без оплаты.

**Implementation and verification note (SH-04):** Resume now validates the
immutable source/configuration/profile/model/prompt contract and actual module
hashes, replays saved receipts locally, and appends request/call history
without resending completed work. Full-file and non-contiguous target plans,
durable complete-before-checkpoint ordering for generation, review, repair, and
recheck, saved calls before completion (unknown costs), explicit retry chains,
source/config/profile/model/prompt mismatches, the nonblocking `filelock`, and
preflight plus zero-target behavior are covered by tests. `BUG-01`
`CancelledError` swallowing is fixed as part of SH-04; cancellation remains an
unknown request requiring explicit retry authorization. Tests are in
`tests/test_shortening_resume.py` and
`tests/test_shortening_resume_integration.py`, including
`test_cancelled_attempt_requires_explicit_retry_and_then_completes`.

Independent final verification reported 407 tests passed. Compileall, the
`shorten_file` CLI help, all three legacy CLI help checks, and `git diff HEAD
--check` also passed. No live API, paid LLM, or TTS calls were made. The
complete captured log is `/tmp/opencode/sh04-final-verification/checks.log`.

The default unknown-request block returns exit 2 and requires an explicit
`--retry-unknown REQUEST_ID` decision; completed receipt errors are replayed as
errors, not resent. Exactly-once external API or billing behavior is not
guaranteed, and known costs remain separate from unresolved billing unknowns.

### [x] SH-05. Удобная очередь ручной проверки — P0

Зависит от SH-02…SH-04. Планируемые результаты: редактируемый `review.json` и читаемый `review.md`.

Включать только нерешённые случаи: оригинал, соседний контекст, кандидаты, лимит и фактическую длину, причину, последнюю критику модели и предлагаемое действие. Отдельно показывать короткие фразы, ошибки таймингов и ошибки протокола.

SH-05 should consume SH-11's canonical outcome markers and queue only cues
whose outcome is unresolved or needs attention.

Команда применения правок принимает только явно одобренные известные `cue_id`, проверяет хеш исходника, формат и бюджет. Устаревшие и превышающие лимит правки не применяются молча. Остальные строки и таймкоды не меняются.

**Готово, когда:** любой нерешённый случай легко найти и исправить без поиска по большим журналам; правки применяются безопасно. Измеряем не только число правок, но и фактическое время пользователя, если оно записано. Новый интерфейс Gradio пока не требуется.

**Implementation note (SH-05, completed alongside SH-11):** Every new run now
writes the all-cue `outcomes.json`/`outcomes.md` snapshot and the unresolved-only
`review.json`/`review.md` queue, bound to checksummed `review_state.json`.
The queue carries original/current text, neighbor context, candidates, budget,
actual length, reason, latest critique, and proposed action. Applying a copied
and explicitly edited queue accepts only known, unique, current-context-bound
`approved=true` `keep`/`replace` entries and writes a new derived export; stale,
invalid, over-budget, immutable, or malformed input fails closed. Apply and
reanalysis are offline and hold a nonblocking run lock. See the concise
[`review and reanalysis guide`](shortening_review.md) and the
[`shorten_file` user guide](shorten_file.md).

Outcome status and `text_changed` are independent. Model or human acceptance is
not measured audio fit; invalid or missing current measurement stays null. A
fresh model verdict refers to original neighbor context and has
`final_context_checked: false`; SH-10 is not implemented. Changed neighbor
windows invalidate reuse and keep attention in the queue. Prediction fits are
estimates only. Legacy budget fields remain compatibility data, while
`report.outcome_summary` is authoritative over legacy stage/verdict counts; the
initial intervention denominator is fixed and later regressions are reported
separately. This records implementation and verification only, not a semantic
guarantee or live pilot.

**Real offline technical acceptance (13 September 2026):** The four-file,
full-source SH-05/SH-11 gate passed without LLM, TTS, network, or paid calls.
See [`shortening_real_acceptance.md`](shortening_real_acceptance.md) for the
durable r12 evidence, frozen hashes, and explicit quality boundaries. This is
technical acceptance, not SH-08 or a semantic-quality result.

### [ ] SH-06. Проверить пользу арбитра при равных условиях — P1

SH-06 remains unchecked and deferred; it does not block the first technical
pilot. The complete H2–H6 branch proposals, historical checks, and evidence
links are consolidated in the [future experiments backlog](future_experiments.md).
Known findings are limited: Jev is not promoted to the main arbiter, Muse high
is not the default, and no full best-of-five run is the default. The recorded
Jev replay/follow-up and the two H6 examples (`cue-618` and `cue-210`) remain
historical or proposal-only evidence; no automatic new tests or calls are
authorized. This short active entry does not discard the unchecked SH-06
status or historical checks.

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

Частичный снимок и проверка отдельных артефактов записаны в
[`sh08_archive_20260915.md`](sh08_archive_20260915.md); это не миграция и не
разрешение на удаление. Плановые миграция, очистка и перенос не выполнялись.

При реализации большие артефакты исключить из Git. Сначала достаточно сжатия JSON/JSONL без потерь и устранения повторного хранения исходников; сложная база данных не нужна.

Сохранять итоговые субтитры, исходные ответы моделей, варианты, критику, инструкции, параметры, хеши, токены, тарифы и профили. Калибровочные WAV и важные проблемные примеры закреплять для хранения. Именно исходные ответы уже позволили восстановить 19 корректных строк без новой генерации.

Очистка сначала только показывает список файлов и освобождаемый объём. Удаление допускается после явного подтверждения и проверки восстановления архива. Не трогать исходники, модели, настройки и авторизацию OpenCode, цели символических ссылок, активные, незавершённые, неизвестные и закреплённые запуски. Не ломать ссылки других экспериментов.

**Предложение, ещё не включённое:** удалять промежуточные данные старше 30 дней только после проверенного архивирования. Постоянные результаты не удалять по сроку автоматически.

**Готово, когда:** из архива без сети восстанавливаются результаты и исходные ответы; повреждение архива блокирует очистку; повторная очистка безопасна. Старые каталоги сейчас не перемещаются и не удаляются.

### [ ] SH-08. Проверить три полных файла — P1

Начинается после SH-00…SH-05; SH-06 не является обязательным предварительным условием.
SH-05/SH-11 уже имеют отдельную real offline technical acceptance, но этот
чекбокс остаётся незакрытым; платный пилот требует отдельного разрешения.
Текущая подготовка также имеет отдельные [модельные факты и proposal]
(shortening_model_pairs.md) и [Vosk timing diagnostics]
(vosk_timing_diagnostics.md); ни одна из этих записей не закрывает SH-08.

#### Current first-pass note (14 September 2026)

The eight-pair first pass is now executed and complete for its planned
60-case/2-repetition small matrix. It is evidence for comparison, not the
three full SH08 pilots. The SH08 checkbox remains unchecked until those full
pilots and their per-file outputs, queues, and costs are completed. Repairs are
reported separately and are partial/non-uniform because the subscription cap
was exhausted. That note predates the bounded low/high comparison recorded
below. Remaining work is offline zero-budget analysis and quality decision-making;
do not check SH10 or authorize more model/TTS calls from this note.

The subsequent actual Muse low/high comparison is recorded in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md), with its
machine-readable summary in
[`output/.../summary.json`](../output/sh08_muse_reasoning_comparison_20260915/summary.json).
It is a completed bounded comparison, not three full pilots and not a semantic
or audio-quality result. The current saved-response analysis is detailed in
[`sh08_duplicate_analysis_20260915.md`](sh08_duplicate_analysis_20260915.md),
and the local Vosk timing follow-up in
[`sh08_zero_timing_20260915.md`](sh08_zero_timing_20260915.md) records 18
measured originals, 1 fit, and 17 overruns. Those measurements provide no sound
or meaning guarantee.

The three full-file dry-runs counted 2,081 positive-budget plans, including
`short_timing` cases that are not all sent to the runner. After filtering to
1,751 `needs_shortening` review candidates, the planner derives 147 review
batches / 882 conservative subscription units. This is the **planned initial
review estimate before repairs, without further measured-audio pruning**, not a
guaranteed actual minimum: errors, `cannot_shorten` outcomes, and local timing
may change the actual calls. The current comparison has 168 overall units
remaining, with 150 units protected for the pilot allocation.

#### Numeric representation prompt rule and recorded follow-ups (15 September 2026)

The shortening prompts now preserve numeric value and source numeric representation:
spelled-out numbers must remain words, so converting `двадцать пять` to `25` is
prohibited as a shortening shortcut. Numbers already written in digits may remain
digits. The guard is recorded in both actual comparison ledgers. No explicit
arbiter-reported representation violation was found, but a zero-violation rate was
not independently verified; historical runs were not rescored.

- [x] Analyze duplicate proposals on saved real outputs first. The scoped
  read-only analysis ran the existing `utils.sh08_duplicate_analysis` CLI once
  against each actual low/high run and wrote fresh reports under
  `output/sh08_muse_reasoning_comparison_20260915/{low_duplicates,high_duplicates}`.
  All 24 low-arm and 20 high-arm requests completed successfully; the high-arm
  model-unresolved result is not infrastructure partiality.
  Current parsed-ok intersection denominators and exact/whitespace duplicate
  counts are low 32/5/5 and high 31/8/8; duplicate repair targets and proven
  avoidable rechecks are zero in both. The historical 59/121 duplicate result is
  retained as historical independent-repetition evidence, not substituted for
  these current counts. This completes saved-output analysis only; a general
  runtime detector remains outside this follow-up. Preserve the `cannot_shorten`
  and unresolved fallbacks; never add endless retries. Existing best-of-5 requests
  1–5 distinct proposals per cue in one request and then performs selection, not
  five requests. Compare against sequential repair only in a separately
  authorized, controlled experiment, not as a default change.
- [x] Verify and compare Muse reasoning modes on the same 60 cases with the Sol
  arbiter and updated prompt version. The manifests requested `low` and `high`,
  and receipt `requested_variant` fields corroborate those modes; this is request
  evidence, not proof of hidden provider reasoning behavior. Independently derived
  accepted-and-fit counts are low 20/41, 22/41 initially and 28/41, 28/41 finally;
  high 27/41, 25/41 initially and 29/41, 28/41 finally. High is +10/82 initial
  and +1/82 final versus low, with higher known cost and serialized receipt time.
  Repairs attempted/rechecked are operational counts, not semantic resolution;
  the high rep1 `cannot_shorten` case is model-unresolved, not infrastructure
  partiality. No separate blinded semantic audit or old-vs-new control was run.
  Muse low remains the default because this sample is insufficient for a universal
  ranking. Per-provider token counters are reported separately; reasoning is not
  summed into output tokens blindly. Both actual runs have standalone HTML reports
  linked in [the comparison record](sh08_muse_reasoning_20260915.md); no offline
  comparison HTML was generated.

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

SH-09 remains deferred and does not block the core pilot. Its alternative TTS,
short-cue audio, and UI backlog is consolidated in the [future experiments
backlog](future_experiments.md), with the existing [roadmap](tts_roadmap.md)
retained as evidence. Edge-TTS, installations, and changes still require
separate authorization.

### [ ] SH-10. Финальный смысловой оценщик — P1 (будущая задача)

Это предложение, не реализация. Зависит от SH-03 и SH-05. Не блокирует первый технический пилот; может использоваться в SH-06 и SH-08 после выбора режима.

**Current status (15 September 2026):** SH-10 remains unchecked and
unimplemented. A full final-context audit was not performed; it is blocked until
full-pilot input is available and a separate evaluator is implemented. The Muse
low/high comparison does not close this task, and no false closure is recorded.

- Выбирается одна логическая роль сильного оценщика, но не обязательно один HTTP-запрос: большие файлы обрабатываются пачками с перекрытием.
- На входе: оригинал и сокращённая цель, исходный контекст соседей и финальный контекст соседей, чтобы проверять смысл через границы изменений.
- В полном режиме проверяются все сокращённые или обработанные цели и их стыки; неизменённые строки дают только контекст, а не повод заново переводить весь оригинал.
- В режиме выборки проверяется фиксированное подмножество, доступное текущей реализации.
- Для каждой цели выводится: `смысл сохранён`, `нужна правка` или `недостаточно контекста`, с обоснованием и указанием проблемного фрагмента; неоднозначное не принимается автоматически.
- Точная финальная модель и обычный режим по умолчанию пока не выбраны: решение зависит от стоимости и качества.
- Свежая сессия той же модели не является независимой проверкой семейства; где уместно, предпочтительно другое семейство, но конкретная модель ещё не выбрана.
- Независимый контроль пока выборочный, полного финального допуска сейчас нет. Вердикт модели не даёт гарантии 100%; бесконечных повторов и автоматических правок текста не будет.
- Приёмка будущего режима: все запрошенные ID покрыты в бюджете, границы контекста проверены, нет утечки идентичности или истории, неизвестные случаи попали в очередь, расходы отделены, ошибки пачек изолированы, а пробные тесты по возможности обходятся без модельных вызовов.

### [x] SH-11. Per-subtitle outcome markers for review and safe reanalysis — P1

Depends on SH-03; coordinate with SH-05 and implement alongside it before SH-08.
The original requirements are retained below; implementation and verification are
recorded after the acceptance criteria.

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

**Implementation note (SH-11, completed alongside SH-05):** The canonical
per-cue outcome state is checksummed and published as `review_state.json`, with
all-cue JSON/Markdown outcomes and an unresolved-only review queue. It records
independent text-change and evidence dimensions, source/current text and
timings, profile/configuration/context fingerprints, reusable checks, and
separate measured duration/fit fields. Forced offline reanalysis invalidates
cached model, human, and measured evidence while retaining lightweight
predictions; external edits never auto-approve. The generated sidecar is kept
outside spoken subtitle text, so SRT/JSON export remains machine/TTS-safe.
[`shortening_review.md`](shortening_review.md) documents application, recovery,
and resume boundaries. SH-10's final-context evaluator remains unimplemented.

**Historical verification (13 September 2026; superseded, past result retained):**
From the activated `subs_env`, `uv run --active --no-sync -m pytest -q`
reported `469 passed`, exit 0. Compileall, all six requested CLI `--help`
checks, and `git diff HEAD --check` each exited 0. The complete captured
stdout/stderr log is `/tmp/opencode/sh05-final-verification/checks.log`. No
network, TTS, LLM, or installation action was performed. The current
continuation baseline is `471 passed, 5 skipped`; see [SH08 preparation](sh08_matrix_preparation.md).

### Known bugs — remaining fixes deferred by user

**Audit note (12 September 2026):** BUG-01 was fixed and tested as part of
SH-04. The following remaining issue(s) are recorded for later triage; no
automatic work has started on them.

- **BUG-01 — P0, fixed in SH-04:** Previously, runner._request caught CancelledError and could continue later batches. Cancellation now propagates, leaves any sent request without a durable response unknown, and does not falsely complete the checkpoint. Covered by fake-cancellation and explicit-retry tests.
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

## Active order of work

1. Preserve the fresh Muse-low/Sol-6-high partial episode, its frozen runtime
   and journal. Inspect the unknown start and accounting offline with
   [`shortening_status`](shorten_file.md#inspect-saved-status-offline). Keep the
   older Sol-medium recovery run as separate historical evidence.
2. Future fresh runs have bounded generation/review windows, graceful stops,
   and at most one explicit unknown-start replacement. **The old episode is
   not migrated.** Reusing its 500 proposals and reconciling its interrupted
   request against the old frozen contract remains a separate engineering and
   authorization step; do not run or bypass its old CLI now.
3. After reconciliation and separate execution authorization, finish and
   assess this episode before the other two full pilots and separately
   planned semantic/audio verification.

SH-06, SH-07, SH-09, and other optional work stay deferred in their linked
backlogs. SH-07's partial verified snapshot remains open; no data deletion,
migration, or mass documentation is implied. SH-10 remains unchecked and
unimplemented until final-context semantic work is actually completed.
This documentation update itself makes no inference calls.

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

Историческое решение на 13 сентября 2026: SH-00 и SH-01 реализованы и уже
присутствовали в HEAD `ec0cbaf` до этой задачи; SH-02 зафиксирована коммитом
`c5bf383`; SH-03 зафиксирована коммитом `9a1ddd9`; по запросу пользователя SH-04
зафиксирована коммитом `a958052`; реализация SH-05 и SH-11 зафиксирована
коммитом `344880b`, а SH-08 по-прежнему ожидает отдельного разрешения на
платный технический пилот; все последующие задачи ожидают.
