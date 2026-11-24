"""Browser UI for the English video -> Russian voice-over pipeline (``stt.run``).

    subs_env/bin/python -m ui.dub_app [--port 7861]

The season bar above the tabs: one season (the folder with the original episodes + its results folder, remembered in
``ui.local.json`` by ``ui.state``) drives every tab; there are no other folder fields. Folders are picked with the
desktop dialog or an in-page browser (``ui.folder_pick``). The first line of each tab says what it writes where
(``ui.seasons.tab_notes``). The settings in effect for a season are layered by ``ui.season_settings``: standard,
global settings file, the season's ``season_settings.json``, then the form of the "Обработка" tab.

Tab "Обработка": pick episodes of the season, engines and options, start, watch the stages of every
episode (and a table of their states), find the finished mkv files. A job runs ``stt.run`` in a background process (one
job at a time, engines one after the other); its config (``ui.seasons.season_config``: everything in the results
folder) and log are kept in ``<results>/.ui/``. The last job is kept in
``output/ui_last_job.json``: after a restart of the page (or of the computer) it is found again, still running
(shown and stoppable), or "прервано" with the "▶ Продолжить" button, which runs the same command with the same
settings file; ``ui.job_view`` turns the stage marks, progress files and log into the words on the tab.

Tab "Голоса сезона" (``ui.voices``): the people of the season and the voice clips ESpeech clones: listen, find another
automatically, give a clip by hand from time ranges of an episode, re-voice ESpeech after a change, rebuild all clips,
count the season's voices again, check names across the season.

Every long action is a task of one background runner (``Job`` with a Russian title; one at a time); the status strip
under the season bar shows the running task, and how the last one ended, on every tab.

Tab "Проверка и правка реплик": every Russian line of an episode (filters: the lines ``stt.review``
found suspicious, edited ones, ones too long for their time), the original and the voice-over of
each, edits of text / speaker (with a voice sample) / gender / "not voiced" into ``corrections.json``
(``stt.subtitle_edit``) with a length check, the Subtitle Edit round trip (ASS / SRT out and back
in), and re-voicing only the changed lines.

Tab "Звук и размер видео": hear a 45-second piece of an episode with other voice levels (``stt.mix_preview``),
save the mix levels, mix the episode again without voicing it, shrink a finished video (``stt.video_compress``).

Tab "Справка": what the program does and how to work with it (``ui/help.md``).

Tab "Настройки": the defaults people change (recognition, models, voices, checks, mix), saved into the global
settings file (``stt_config.local.json`` unless another is chosen) or the season's file (``ui.settings``); the
"Обработка" form starts from the settings in effect for the season. Each voice has a
"▶ Послушать" button: a sample phrase in the voice chosen now (``ui.voice_preview``, made in a subprocess).
"""
