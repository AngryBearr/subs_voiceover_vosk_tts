# Settings guide for /dub-run

The questions of the skill, in the order they are asked, with the Russian texts to show. Every question
maps its answers to keys of `ui.settings.SETTINGS` (the "Настройки" tab of the UI), so they can be saved for the
project or the season with `ui.run_setup save`. Launch-only choices (preset, transport, test piece) are passed to
`ui.run_setup launch` and are never saved.

Rules for every question:

- Take the current value from `show` → `settings[].effective` and put the option that matches it **first**, with
  ` · сейчас` at the end of its label. If another option is clearly better for the user's goal, say so in its
  description ("рекомендую, потому что ..."), do not reorder silently.
- Put the source of the current value into the question text when it is not the standard one:
  «(сейчас: 7 шагов — из настроек сезона)». Sources: `standard` → «стандарт», `global` → «общие настройки»,
  `season` → «настройки сезона», `show_override` present → «профиль шоу <label>» (the show profile wins over
  everything at run time; say so if the user tries to change such a key).
- Money: say "бесплатно" only when the model goes through the subscription (see "Cost" below). Time: give the
  numbers below as rough estimates ("примерно", "по замерам на S48E07").
- Never offer to voice the on-screen captions (`on_screen.voice` stays as it is) and never offer a "neutral voice
  when unsure" option: the user rejected both. Never offer the show-profile features (`voice_bank.identities`,
  `stitch`, stand-in pools) as a question: they belong to show profiles in `stt_config.json`.
- Skip a question that does not apply (ESpeech steps when ESpeech is not selected, transport when there is no
  subscription route, parallel options for a single short test piece).

## Reference numbers

Measured on a 64-minute Survivor episode (S48E07) on an RTX 4070 Ti SUPER 16 GB, all three voice-overs:
**65.7 min** in total (docs/roadmap.md, "after speed-ups"). Main parts: recognition ~8 min, translation with
on-screen text ~4 min, stress ~2.5 min, ESpeech voice-over ~29 min (32 steps), TeraTTS ~10 min, Vosk on the CPU
at the same time. Scale by episode length (minutes of video / 64).

| Run | Per 64-min episode (rough) |
|---|---|
| ESpeech + TeraTTS + Vosk, ESpeech 32 steps | 65-75 min |
| the same, ESpeech 7 steps | ~50 min |
| TeraTTS + Vosk (no ESpeech) | ~35-40 min |
| Vosk only, preset «Быстрый» | ~30-40 min |
| episode already prepared (`--prepare` before) | 15 min less |
| several episodes in one run (`overlap_episodes`) | ~4 min less per episode |
| test piece of 5 minutes | ~10-15 min (models still load) |
| «Подготовить весь сезон» (`--prepare`) | 8-10 min per episode, no translation or voice-over |

### Cost

Only text goes to the language models; the sound never leaves the computer. Per 64-min episode, all stages on
luna through OpenRouter: about **$0.21** (two transcripts merged $0.025, glossary $0.004, names $0.003,
translation $0.036, stress check $0.14). Sol everywhere through OpenRouter: about **$1 per episode**. Cast lookup:
$0.01-0.03 per season. Stress votes of Gemini Argon through OpenRouter (only when the OpenCode Go plan is out of its
limit): ~$0.03-0.05 per episode.

What is free: a model whose name matches a pattern in `launch_defaults.llm.opencode` (e.g. `openai/*`) goes through
the ChatGPT subscription (opencode) and costs $0, as long as `env.opencode` is found and the run uses the
subscription transport; `opencode-go/*` models go through the OpenCode Go plan the same way. Everything else goes
through `llm.router` (OpenRouter, the only paid router) and is paid; a subscription model falls back to it when the
subscription fails or its limit is used up. With the user's usual settings (Sol through the ChatGPT subscription, the
stress vote of Gemini Argon through OpenCode Go and gpt-6-astra through ChatGPT) the run is free except the cast
lookup per season and, while the Go limit is used up, Argon's votes through OpenRouter (~$0.03-0.05 per episode). Compute this from
`launch_defaults.llm` of the season instead of repeating these words blindly.

---

## Step «Что озвучиваем»

Header `Что делать`. Question: «Что сделать с сезоном <name>? Готово <finished> из <episodes> серий.»

| Label | Description | Launch |
|---|---|---|
| Все неготовые серии (N) | Озвучить все серии без готового видео одним запуском: так программа лучше узнаёт голоса героев по всему сезону и переводит следующую серию, пока озвучивается текущая. Примерно <time> и <cost>. | default selection |
| Одну серию | Выбрать одну серию (следующим вопросом). | `--episodes SxxEyy` |
| Пробный кусок 5 минут | Кусок одной серии, чтобы быстро послушать настройки (~10-15 мин). Результат в `<results>/_проба`, сезонные файлы не трогает. | `--preview-from 10 --preview-minutes 5 --episodes SxxEyy` |
| Подготовить весь сезон | Только распознание, надписи, кто говорит, голоса сезона; без перевода и озвучки. 8-10 мин на серию, потом каждая серия озвучивается на ~15 мин быстрее, а во вкладке «Кто это?» можно назвать безымянные голоса. | `--prepare --all` |

For «Одну серию» ask once more (header `Серия`) with up to 4 unfinished episodes as options (label = `label` of
the row, description = state and duration); the user types another code through "Other".

## Step «Как настроить»

Header `Настройки`. Show the settings table first (see SKILL.md), then ask «С какими настройками запускать?»

| Label | Description |
|---|---|
| Как сохранено | Ничего не менять: только подтвердить запуск. |
| Важное: качество и деньги | 2 коротких экрана: движки, модель перевода, распознавание, субтитры, проверки текста. |
| Пройтись по всему | Важное, потом что влияет на время, потом мелкие улучшения (4-5 экранов). |
| Готовый набор | Основной / Для облака / Быстрый / Прежний (следующим вопросом). Действует только на этот запуск. |

Presets (header `Набор`, labels from `show.presets[].label`): `main` best quality, three voice-overs; `upload` as
main plus video compressed to ~2 GB (+~5 min per episode, usually in parallel); `fast` Whisper only, season voices
off, no internet subtitles, Vosk only, no stress or speech checks (cheapest and fastest, clearly worse speakers and
stress); `previous` the old behaviour before 2.11 (for comparison only). A preset is applied on top of the saved
settings for this run only (`launch --preset <name>`); after picking one, go straight to the launch step.

---

## A. Качество озвучки и деньги

### A1 `ui.engines` — header `Движки`, multiSelect

«Какие озвучки сделать? Каждая — отдельная звуковая дорожка в одном готовом видео.»

| Label | Description |
|---|---|
| ESpeech — голоса героев | Каждый участник говорит своим клонированным голосом. Самая живая озвучка и самая долгая: ~29 мин на 64-минутную серию при 32 шагах. Видеокарта, окружение f5_env. |
| TeraTTS | Мужской и женский голос плюс свой голос ведущего. ~10 мин на серию на видеокарте. |
| Vosk | Мужской и женский голос, работает на процессоре одновременно с остальными: общее время почти не растёт. |

Warn and drop ESpeech when `env.f5_env` is false. Value: list of `f5`, `teratts`, `vosk`.

### A2 `llm.model` — header `Перевод`

«Какая языковая модель переводит, сокращает длинные реплики и подбирает имена? От неё больше всего зависит, как
звучит русский текст.»

| Label | Value | Description (pick the cost words from "Cost") |
|---|---|---|
| Sol | `openai/gpt-6.1-sol` | Лучше перевод и имена. Через подписку — бесплатно; через OpenRouter — около $1 за серию. |
| luna | `openai/gpt-6-luna` | Дешевле (через OpenRouter около $0.2 за серию со всеми этапами), качество хорошее. |

### A3 `mode` — header `Распознание`

«Как распознавать английскую речь? Это основа: ошибки здесь переходят в перевод.»

| Label | Value | Description |
|---|---|---|
| Точный: Cohere + Whisper | `accurate` | Две расшифровки, языковая модель сводит их в одну: меньше пропущенных и перепутанных слов. ~$0.025 за серию на luna (через подписку бесплатно). |
| Быстрый: только Whisper | `fast` | На несколько минут быстрее и без затрат на сведение, но ошибок распознавания больше. |

### A4 `subtitles.enabled` + `subtitles.online` — header `Субтитры`

«Брать готовые английские субтитры? По именам и знакам смены говорящего в них программа узнаёт, кто говорит, а
значит — чьим голосом озвучить реплику.»

| Label | Keys | Description |
|---|---|---|
| Да, и искать в интернете | enabled true, online true | Своя папка сезона, дорожки в видео, база, архив, затем OpenSubtitles и Gestdown. Только текст, бесплатно. |
| Только свои и из базы | enabled true, online false | То же без запросов в интернет. |
| Не брать | enabled false | Кто говорит — только по голосу; обычно больше реплик уходит не тому герою. |

### A5 Text checks — header `Проверки`, multiSelect

«Какие проверки текста делать до озвучки? Они исправляют ударения и ошибки перевода.»

| Label | Key | Description |
|---|---|---|
| Голосование за ударения | `voiceover.stress_check` | Спорные ударения решают модели голосованием (Gemini Argon через OpenCode Go и GPT-6 Astra через подписку ChatGPT, исправление — только если согласны обе): ~28 исправлений на серию. Бесплатно; если лимит OpenCode Go исчерпан, Argon идёт через OpenRouter, около $0.03-0.05 за серию. |
| Вычитка ударений (Sol) | `voiceover.stress_proofread` | Sol дважды перечитывает все реплики: ещё ~10 исправлений на серию, +3 мин. |
| Вычитка перевода (Sol) | `translate.proofread` | Sol ищет ошибки в русском переводе перед озвучкой. |
| Одно ударение имени на сезон | `voiceover.season_stress` | Имя героя звучит одинаково во всех сериях (файл `<сезон>.stress.json`). Бесплатно. |

Unticked = false. Mention the Sol cost through OpenRouter when Sol is not on the subscription.

### A6 Transport (launch only) — header `Оплата`

Ask only when `launch_defaults.llm.opencode` is not empty. «Как отправлять запросы к моделям OpenAI в этом
запуске?»

| Label | Launch | Description |
|---|---|---|
| Через подписку (opencode) | — | Бесплатно; если подписка недоступна, программа сама переходит на платный OpenRouter. |
| Через OpenRouter API | `--transport openrouter` | Платно (см. цены выше), зато не зависит от лимитов подписки. |

---

## B. Почти не меняет звук, но меняет время

### B1 `voiceover.engines.f5.nfe` — header `ESpeech` (only when ESpeech is selected)

| Label | Value | Description |
|---|---|---|
| 32 шага | 32 | Лучшее качество: ~1,2 с на реплику, ~29 мин на 64-минутную серию. |
| 7 шагов | 7 | В 2 раза быстрее (~13 мин на серию, экономия ~15 мин); на слух почти так же, ошибок в словах столько же (0,77% против 0,83%). |

### B2 Local checks — header `Проверки звука`, multiSelect

«Какие проверки на компьютере делать? Бесплатно, но добавляют время.»

| Label | Key | Description |
|---|---|---|
| Проверять речь распознаванием | `voiceover.speech_check` | GigaAM слушает каждую реплику и переозвучивает, если пропали слова. ~0,2 с на реплику в каждой дорожке (около 2 мин). |
| Слушать ударения в дублях | `voiceover.listen_stress` | Переозвучивает реплики, где имя или омограф прозвучали с неверным ударением (1-7 на дорожку). ~0,3 с на реплику (около 3 мин на дорожку). |
| Читать надписи на экране | `ocr.enabled` | Имена из титров помогают узнать говорящего, надписи переводятся отдельной дорожкой субтитров. ~7 мин на 90-минутную серию на видеокарте, обычно параллельно с другим. Без неё пропадает перевод надписей. |

### B3 Parallel work — header `Параллельно`

«Как распределять работу? На результат не влияет, только на время.»

| Label | Keys | Description |
|---|---|---|
| Авто | `gpu.parallel` auto, `overlap_episodes` true, `voiceover.parallel_cpu` true | Следующая серия переводится, пока озвучивается текущая (−4 мин на серию), Vosk на процессоре рядом, тяжёлые задачи видеокарты рядом друг с другом, только если хватает памяти. |
| Всегда параллельно | `gpu.parallel` on, the two others true | Ещё быстрее; нужна видеокарта от 16 ГБ (`env.gpu`). |
| Всё по очереди | `gpu.parallel` off, `overlap_episodes` false, `voiceover.parallel_cpu` false | Медленнее, но надёжнее, если были сбои из-за памяти. |

### B4 `mix.video.compress` (+ `target_gb`) — header `Размер видео`

| Label | Keys | Description |
|---|---|---|
| Не сжимать | compress false | Видео как исходное (часто 4-6 ГБ). |
| Сжать до ~2 ГБ | compress true, target_gb 2 | Разрешение то же, меньше битрейт; +~5 мин на серию, обычно параллельно. Удобно для облака. |

The user may type another size through "Other" (target_gb).

---

## C. Мелкие улучшения

### C1 `dub.interjections` + `crowd.enabled` — header `Выкрики`

| Label | Keys | Description |
|---|---|---|
| Как сейчас: обычные выкрики озвучиваются, поверх чужой речи — отдельный слой | overlap, crowd true | Рекомендуется (по умолчанию с 10.11). |
| Без слоя выкриков | overlap, crowd false | Выкрики поверх чужой речи остаются только в оригинале. |
| Не озвучивать выкрики в быстром обмене репликами | dense | Тише и спокойнее, но часть коротких реплик пропадёт. |
| Озвучивать все | keep | Даже поверх чужой речи в основной дорожке: бывает тесно. |

### C2 `mix.tracks` — header `Готовые видео`

| Label | Value | Description |
|---|---|---|
| Одно видео со всеми озвучками | combined | Одна mkv, озвучки — дорожки, переключаются в плеере. |
| Отдельное видео на каждый движок | separate | Несколько файлов: удобно, если плеер плохо переключает дорожки, но больше места. |

### C3 Extras — header `Дополнительно`, multiSelect

| Label | Key | Description |
|---|---|---|
| Русские субтитры всех реплик | `mix.subtitles` | Дорожка субтитров в видео (по умолчанию выключена в плеере). |
| Перевод надписей на экране | `on_screen.track` | Отдельная дорожка: русский текст закрывает английскую надпись. |
| Второй проход распознавания | `fill.enabled` | Whisper ещё раз слушает места, где речь есть, а слов нет (крики под музыку). +10-16 с на серию. |
| Список подозрительных реплик | `review` | Для вкладки «Проверка и правка реплик». |

### C4 Speakers, small — header `Говорящие`, multiSelect

| Label | Key | Description |
|---|---|---|
| Менять род реплики по голосу | `line_gender.override` | Когда обе модели уверены, что голос другого пола; такие реплики попадают в список для проверки. |
| Искать состав участников | `cast.enabled` | Википедия и поиск: одно написание имени на весь сезон. $0.01-0.03 на сезон. |
| Банк голосов сезона | `voice_bank.enabled` | Узнавать героев между сериями. |
| Хранить несжатый звук | `mix.keep_wav` | ~0,5 ГБ на серию; нужно только для ручного пересведения. |

### C5 Loudness — header `Громкость`

Show the current `mix.voice_gain_db`, `mix.rest_duck_db`, `mix.loudness_lufs`, then:

| Label | Description |
|---|---|
| Оставить как сейчас | Озвучка +3 дБ, музыка и шум под голосом −6 дБ, общая громкость −16 LUFS (стандарт). |
| Подобрать на слух в интерфейсе | Вкладка «Звук и размер видео»: самое громкое место серии, ползунки, «Послушать». Пересведение готовой серии — минуты, без переозвучки. |

Numbers typed through "Other" are saved as given (`mix.voice_gain_db`, `mix.rest_duck_db`, `mix.loudness_lufs`).

---

## What only the UI has (for the «Ещё улучшения» step)

| Tab | What it gives the dub |
|---|---|
| «Голоса сезона» | Послушать образец голоса каждого героя для ESpeech, найти другой автоматически или вырезать отрывок вручную; потом «Переозвучить ESpeech» — только реплики этих людей. Самое заметное улучшение дорожки ESpeech. |
| «Кто это?» | Назвать голоса, которые программа не узнала (лучше после «Подготовить весь сезон»): меньше реплик общим голосом. |
| «Настройки» | Голоса Vosk, TeraTTS и голос ведущего с кнопкой «▶ Послушать»; общий файл настроек. |
| «Звук и размер видео» | Громкость озвучки и музыки на слух, пробное сжатие 60 с. |
| «Проверка и правка реплик» | После озвучки: говорящий, пол, текст, ударение (`зам+ок`), «не озвучивать»; переозвучиваются только изменённые реплики. Также выгрузка в Subtitle Edit. |
| «Обработка» | Ход работы, контрольные точки, «Остановить» и «▶ Продолжить» (видит и запуск из этого скилла). |
