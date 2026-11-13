---
name: dub-run
description: Set up and start a Russian voice-over run (stt.run) for a season, one episode or a test piece. Shows the saved settings, asks about the important ones with explanations (quality and money first, then time, then small improvements), saves the answers for the project or the season, can open the UI for further improvements, then launches the run as a background job.
disable-model-invocation: true
argument-hint: "[season | folder | video | SxxEyy ...]"
allowed-tools: Bash(subs_env/bin/python -m ui.run_setup *)
---

# /dub-run — set up and start a voice-over

The user runs this by hand to voice a season or an episode. Your job is to make the launch **easy and
informed**: start from the settings already saved, ask only what matters, explain every choice in plain Russian
(what changes in the dub, what it costs, how long it takes), save the answers where the user wants, and start
the run. The questions, option texts and reference numbers are in [settings-guide.md](settings-guide.md) — read
it before the first question.

Arguments: `$ARGUMENTS` (may be empty).

## Ground rules

- Talk to the user in Russian, plainly; explain any term the first time. No JSON, keys or file dumps in the chat
  unless the user asks; name settings by their Russian labels.
- Ask with the harness's question tool (`AskUserQuestion` in Claude Code): at most 4 questions per call, 2-4 options
  each, every option with a description, headers of at most 12 characters. The option matching the current value
  goes first and ends with ` · сейчас`. The user can always type their own answer ("Other"). Without such a tool,
  ask the same as a short numbered list in one message.
- All data comes from the helper `subs_env/bin/python -m ui.run_setup <command>` (run from the repository root; it
  prints JSON). It uses the same settings layers as the UI: `stt_config.json` (standard) → global file
  (`stt_config.local.json`) → the season's `season_settings.json` → the show profile (applied by `stt.run`).
  Never edit these files by hand; use `save`.
- Never print API keys or tokens (`env.keys` only says whether a key exists). Never act as a model in a pipeline
  step: you only choose settings and start the program.
- The user approved the current Survivor pipeline by ear: do not push changes to its defaults. Offer, explain,
  let the user decide. Never offer to voice the on-screen captions or a "neutral voice when unsure".
- Only one long job at a time (one GPU). If `running.processes` is not empty, say what runs (and offer
  `status`), and do not start another job unless the user explicitly says so after hearing that.

## Step 1. Which season and what to do

1. Parse `$ARGUMENTS`: a season name (`Season_48`), a folder, a results folder `STT_*`, or a video file → the
   TARGET; tokens like `S48E03` → episodes; `проба` / `test` → a test piece.
2. No TARGET: run `seasons`. One season or a clear current one → use it and say which. Otherwise ask (header
   `Сезон`): up to 4 seasons, the current first; description = «готово X из Y, результаты в <results>».
3. Run `show <TARGET>`. Check and tell the user briefly, only when relevant:
   - `running` not empty → see the ground rules;
   - `env.f5_env` false → ESpeech cannot run; `env.diarizen_env` false → speakers are found by pyannote (less
     accurate); `env.gpu` null → no video card found, everything will be very slow;
   - `env.disk_free_gb` below ~5 GB per episode to voice → warn (rough figure);
   - `episodes[].duration_min` and `season.total_unfinished_min` give the minutes for the time estimate
     (null when the length could not be read: then assume ~64 min per episode and say so);
   - `season.show` set → «профиль шоу <show_label>: свои настройки говорящих, они главнее сохранённых».
4. Ask «Что озвучиваем» (settings-guide.md, step «Что озвучиваем») unless the arguments already said it. For one
   episode ask which one. Remember the selection for `launch` (`--episodes`, `--all`, `--prepare`, test piece).

## Step 2. Start from the saved settings

Show one compact table of the settings the questions cover, grouped A/B/C, from `show.settings` (`display` and
`source`), for example:

```
Сезон Season_48 · действуют настройки сезона (season_settings.json)
A. Качество и деньги
  Движки ............. ESpeech, TeraTTS, Vosk      общие настройки
  Модель перевода .... Sol (через подписку, $0)    общие настройки
  ...
B. Время
  ESpeech ............ 32 шага                     стандарт
C. Мелочи
  ...
Оценка: 4 серии × ~70 мин ≈ 4,5-5 ч; платно $0 (если лимит OpenCode Go исчерпан — до ~$0.2 за голоса Argon через OpenRouter).
```

Mark values that come from the show profile. Then ask «С какими настройками запускать?» (step «Как настроить» of
the guide): «Как сохранено» → Step 5; «Важное» → group A; «Пройтись по всему» → A, B, C; «Готовый набор» → ask the
preset, then Step 5.

## Step 3. The questions

Ask group by group, in the guide's order, batching up to 4 questions per call:

- **A. Качество озвучки и деньги** — A1-A4 in one call, A5 (+A6 when it applies) in the next.
- **B. Почти не меняет звук, но меняет время** — B1-B4 in one call.
- **C. Мелкие улучшения** — C1-C4 in one call, C5 alone (it may lead to the UI).

After each group, say in one line what changed and how it moves the estimate («ESpeech на 7 шагах: ~15 мин
быстрее на серию»). Keep a list of the changed keys with their new values; launch-only choices (preset, transport,
test piece) go to a separate list.

## Step 4. Where to save the changes

Only when some saved setting changed. Ask (header `Сохранить`) «Где сохранить изменения?»:

| Label | Effect |
|---|---|
| Для всего проекта | `save --scope global <TARGET> key=value ...` → the global file; every season without its own settings gets them. |
| Только для сезона <name> | `save --scope season <TARGET> key=value ...` → `<results>/season_settings.json`. If the season had no file yet, say plainly: the season gets its own copy of **all** settings and stops following later changes of the global ones (the UI works the same way; «Настройки» → «Убрать настройки сезона» or `forget-season` undoes it). |
| Только для этого запуска | nothing is saved; the values go to `launch --set key=value`. |

Recommend «Только для сезона» when the season already has its own settings (`season.has_own_settings`),
otherwise «Для всего проекта». Values are config values (JSON), one shell argument per pair: `mode=fast`,
`voiceover.engines.f5.nfe=7`, `voiceover.stress_check=false`, `'ui.engines=["f5","vosk"]'`,
`llm.model=openai/gpt-6.1-sol`. Report the `save` result in one line (which file, how many settings) and any `warnings`.
If `save` refuses a value (e.g. a number of steps the UI does not offer), use it for this run only via `--set`
and say so.

## Step 5. More improvements in the UI (offer once)

Ask (header `Интерфейс`) «Хотите улучшить что-то ещё? В интерфейсе есть то, чего нет в этих вопросах» with
the short list from the guide's last table in the question or descriptions (voices of the heroes, unnamed voices,
voices with «Послушать», loudness by ear, line fixes after the dub). Options: «Нет, к запуску», «Открыть
интерфейс», and, for a season not yet prepared with ESpeech selected, «Сначала подготовить сезон» (prepare →
then «Кто это?» and «Голоса сезона» in the UI → then voice it).

«Открыть интерфейс»: run `ui --open`, give the link (`url`), say which tabs to look at for their goal and that
the season is picked at the top of the page. Ask the user to come back and say «готово». Then run `show` again
and tell what changed in the settings (the UI saves them itself). If the user started the run from the UI, stop
here and say how to follow it (Step 7).

## Step 6. Confirm and launch

1. Run `launch <TARGET> <selection> --engines <A1 answer, e.g. f5,teratts,vosk> [--preset ...]
   [--transport openrouter] [--set k=v ...] [--preview-from M --preview-minutes N] [--prepare] --dry-run`
   (TARGET first; `--engines` always, in any order: the helper runs ESpeech first, then TeraTTS, Vosk on the CPU
   alongside). Without `--engines` it takes the saved default engines, after the preset.
2. Show a short card: season and episodes (codes), voice-overs, what is changed against the standard
   (`changed_from_standard`, in Russian labels), where the results go (`out_root`), the time estimate (episode
   minutes from `show.episodes[].duration_min`, the guide's table) and the paid part of the cost (the guide's
   "Cost"), and one sentence on who does what: «речь, голоса, озвучка и проверка — на этом компьютере; перевод,
   имена и ударения — языковые модели (только текст)».
3. Ask (header `Запуск`): «Запустить», «Сначала пробный кусок 5 мин» (same settings, `--preview-from 10
   --preview-minutes 5` on the first selected episode), «Изменить настройки» (back to Step 2), «Отмена».
4. Run the same `launch` without `--dry-run`. On `busy` say what runs. On success say: started, unit name, the
   log path, that it runs in the background and survives closing this chat, how to stop it (`stop`), and that
   the UI tab «Обработка» shows its progress with «Остановить» / «▶ Продолжить». If `detached` is `setsid`, warn
   that it may stop if the agent session restarts.

## Step 7. Following the run

When the user asks how it goes, run `status <TARGET>` and answer in one line per episode (stage, done or not) plus
anything alarming from `log_tail`. Do not poll in a loop or sleep. A failed run continues from where it stopped with
the same `launch` (finished stages are skipped) or the UI's «▶ Продолжить»; fix nothing in the code from this
skill — report the error and suggest the user asks for a fix.

## Helper commands (reference)

```
subs_env/bin/python -m ui.run_setup seasons
subs_env/bin/python -m ui.run_setup show [TARGET]
subs_env/bin/python -m ui.run_setup save --scope global|season [TARGET] key=value ...
subs_env/bin/python -m ui.run_setup forget-season [TARGET]
subs_env/bin/python -m ui.run_setup launch [TARGET] [--episodes S48E03 ...|--all] [--engines f5,teratts,vosk]
    [--preset main|upload|fast|previous] [--transport subscription|openrouter] [--set key=value ...]
    [--preview-from MIN --preview-minutes MIN] [--no-season-voices] [--prepare] [--dry-run]
subs_env/bin/python -m ui.run_setup status [TARGET]
subs_env/bin/python -m ui.run_setup ui [--port 7861] [--open]
```

TARGET: a season name, its source folder, its `STT_*` results folder or a video file; empty = the UI's current
season. Without `--episodes`/`--all`, `launch` takes every unfinished episode (or the video given as TARGET).
