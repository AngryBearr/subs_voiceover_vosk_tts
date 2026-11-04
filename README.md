# Russian voice-over for English video

**English** | [Русский](docs/readme.md)

This repository turns an English TV episode (it was built and tested on reality shows such as
Survivor and The Amazing Race) into a video with a Russian voice-over. You give it a video file
or a whole season; you get one `.mkv` per episode with:

- several Russian audio tracks (the same dub made by up to three different voice engines),
- the original audio,
- Russian subtitles.

The Russian voices are laid over the original sound, which is turned down while a Russian line
is spoken. Where possible every person speaks in their own voice (a cloned one) or at least in a
male or female voice that fits.

Everything runs from the command line or from a browser page on your own computer.

## Contents

1. [What is local and what goes out to the network](#1-what-is-local-and-what-goes-out-to-the-network)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Keys and personal settings](#4-keys-and-personal-settings)
5. [Quick start](#5-quick-start)
6. [Presets](#6-presets)
7. [How the pipeline works, stage by stage](#7-how-the-pipeline-works-stage-by-stage)
8. [How speakers are found](#8-how-speakers-are-found)
9. [Existing subtitles](#9-existing-subtitles)
10. [Season voices](#10-season-voices)
11. [Voice engines](#11-voice-engines)
12. [Sound level](#12-sound-level)
13. [Video compression](#13-video-compression)
14. [The browser UI](#14-the-browser-ui)
15. [Output files and folders](#15-output-files-and-folders)
16. [Correcting lines](#16-correcting-lines)
17. [Configuration](#17-configuration)
18. [The older "classic" pipeline](#18-the-older-classic-pipeline)
19. [Training the speaker model](#19-training-the-speaker-model)
20. [Tests](#20-tests)
21. [Repository layout](#21-repository-layout)
22. [Further reading](#22-further-reading)

## 1. What is local and what goes out to the network

| Step | Where it happens | What leaves your computer |
|---|---|---|
| Decoding audio, speech recognition, who speaks when, gender, on-screen text, voice synthesis, speech check, mixing, video encoding | your GPU / CPU | nothing |
| Fusing two transcripts, season glossary, speaker names, translation, shortening long lines, stress check | a language model (an AI service that works on text) | text only: English lines, names, Russian lines |
| Cast of the season | Wikipedia article fetched by the program, or a web-searching language model | the name of the show and season, taken from the folder and file names |
| Existing subtitles (optional, on by default) | looked up in the video file, next to it, in a local archive, then online | show id, season and episode number; the subtitle file is downloaded |
| Model downloads | Hugging Face, first run only | nothing from your files |

Audio and video never leave the machine. Online subtitle search can be switched off
(`--no-subtitles`, or the preset `fast`).

Language models are reached through OpenRouter by default. You can instead use a ChatGPT
subscription through the OpenCode program, or the OrcaRouter service; see
[section 4](#4-keys-and-personal-settings) and "Language models" in `docs/stt_local.md`.

## 2. Requirements

- Linux. The scripts check for it and stop on anything else.
- An NVIDIA graphics card with a driver that supports CUDA 12.8 or newer. `scripts/install.sh`
  warns if there is less than 12 GB of video memory. The author's card is an RTX 4070 Ti SUPER
  with 16 GB; nothing was measured on other cards.
- Disk: the installer warns if less than 40 GB is free in the project folder or in the model
  cache (`~/.cache/huggingface`, or `$HF_HOME`). Roughly: Python environments 15 GB, downloaded
  models about 30 GB the first time, working files about 2 GB per episode.
- `git`, `curl` and `uv` (a fast Python package installer; the installer offers to install it).
  No system `ffmpeg` is needed.
- Accounts and keys:
  - a free Hugging Face account and a read token; you must also accept the terms of two models
    on their pages (`pyannote/speaker-diarization-community-1` and
    `CohereLabs/cohere-transcribe-03-2026`);
  - an OpenRouter key (paid per request) for the language-model steps, unless you use another
    route (section 4).
- Some of the models have non-commercial licences (for example DiariZen weights, CC BY-NC 4.0;
  the age and gender model, CC-BY-NC-SA-4.0). Check them before any commercial use. The
  repository itself has no licence file yet.

## 3. Installation

```bash
bash scripts/install.sh            # everything: also ESpeech (the heroes' own voices) and DiariZen
bash scripts/install.sh --basic    # Vosk and TeraTTS voices only, pyannote for speakers
subs_env/bin/python -m stt.selftest          # check GPU, packages, models, keys
subs_env/bin/python -m stt.selftest --quick  # the same without downloads
```

The installer goes through six steps: it checks the computer, creates the Python environments,
downloads the Vosk voice model, logs in to Hugging Face, asks for the OpenRouter key (it is
written to `.env` and never shown), runs the self-test (this is where the ~30 GB of models are
downloaded) and can add a menu shortcut. Run it again after `git pull`: finished steps are
quick. Environments can also be created one by one with `bash scripts/setup_envs.sh [main|f5|diarizen|qwen]`.

The Python environments:

| Folder | What it is for |
|---|---|
| `subs_env` | almost everything: recognition, translation, Vosk and TeraTTS, speech check, mixing, the UI |
| `f5_env` | ESpeech (F5-TTS), the voices cloned from the heroes. Needs `subs_env` first |
| `diarizen_env` | DiariZen v2, the default "who speaks when" model. Without it the program uses pyannote |
| `stt_qwen_env` | optional, only for comparisons with Qwen3-ASR; the pipeline does not use it |

All commands below are run from the repository root. `uv run -m ...` and
`subs_env/bin/python -m ...` are equivalent.

**Trained speaker weights are not in git.** The file `models/speaker/resnet293-ft/avg_model.pt`
(115 MB) is not part of the repository; it is on
[Hugging Face](https://huggingface.co/AngryBear9019/wespeaker-resnet293-survivor-ft). The tick on the processing tab
downloads it once before the work starts (`stt.run --download-voice-model` from the command line). Without it the
pipeline logs a warning and uses the plain `resnet293` model, which is slightly less accurate (section 10). See
[section 19](#19-training-the-speaker-model) for how it was made.

## 4. Keys and personal settings

Keys go into `.env` in the repository root (it is in `.gitignore`). The names the code looks for:

- `OPENROUTER_API_KEY` - language models through OpenRouter (the default route);
- `ORCAROUTER_API_KEY` - the same through OrcaRouter, optional;
- `DEEPSEEK_API_KEY` - used by the older subtitle-shortening scripts (`.env.example` shows the format);
- the Hugging Face token is stored by `hf auth login`, not in `.env`.

Personal settings go into `stt_config.local.json` next to `stt_config.json`. It is not in git
and is laid on top of the shared file by `stt.run` and the UI. A typical use is changing the
language model or the folders:

```json
{"paths": {"work": "/path/to/work", "finished": "/path/to/videos"},
 "llm": {"model": "openai/gpt-6.1-sol", "opencode": ["openai/*"], "router": ["orcarouter", "openrouter"]}}
```

With `llm.opencode`, models matching the pattern go through OpenCode logged in with a ChatGPT
subscription (`opencode auth login`), and the paid router takes over if that fails (set
`"fallback": false` to stop instead). The UI settings tab writes only your differences from the
shared config into this file. Do not paste keys or tokens anywhere else.

## 5. Quick start

### Through the browser

```bash
bash scripts/run_ui.sh        # opens http://127.0.0.1:7861 (PORT=... for another port)
```

At the top of the page pick a season ("＋ Новый сезон": the folder with the original episodes; the results folder
`STT_<season>` next to it is suggested and holds everything). On the tab "Обработка" tick the episodes, pick a preset
and press "Начать". The browser can be closed; the job goes on. If you stop it, the next run
continues where it stopped. To pick a folder, "📁 Выбрать…" opens the desktop's own folder dialog (`zenity`) when the
page runs on a computer with a desktop; otherwise a small folder browser opens on the page. Every tab has a line at
the top saying what it writes where.

### Through the command line

```bash
# one episode, everything: voice-over in all three engines, checks, mix
subs_env/bin/python -m stt.run /path/to/season/ep01.mkv

# a whole season by a mask (quote it so the shell does not expand it)
subs_env/bin/python -m stt.run "/path/to/season/*.mkv"

# stop after the Russian dubbing script, no voice synthesis
subs_env/bin/python -m stt.run "/path/to/season/*.mkv" --until dub
```

Useful options (`subs_env/bin/python -m stt.run --help` lists all):

| Option | Meaning |
|---|---|
| `--preset NAME` | a named set of settings, see [section 6](#6-presets) |
| `--engine f5,teratts,vosk` | which voice engines to use (default: all three) |
| `--until STAGE` | last stage to run: `transcribe`, `glossary`, `speakers`, `translate`, `dub`, `stress`, `voiceover`, `check`, `mix` |
| `--force STAGE` | recompute this stage and every stage after it |
| `--mode fast\|accurate` | Whisper only (fast) or two transcripts fused by a language model (accurate, the default) |
| `--start SEC --duration SEC` | process only a piece of the episode (for tests) |
| `--no-subtitles` | do not look for existing subtitles |
| `--no-season-voices` | skip the season voices stage |
| `--out-root DIR` | where the episode work folders go (default `paths.work`, i.e. `output/stt`) |
| `--config FILE` | another base config |

Finished stages are skipped, so a second run of the same command costs nothing and never repeats
language-model requests. A run can be stopped at any time (Ctrl+C, or "Остановить" in the UI)
and started again with the same command: finished stages are skipped, recognition keeps its
finished parts, and the voice-over keeps every line already synthesised.

One measured example: a ~86 minute episode on an RTX 4070 Ti SUPER took about 50 minutes for the
three voice-overs and the mix when the transcript and translation already existed; S20E01 end to
end took 1 h 15 min. Expect different numbers on other hardware and episodes.

## 6. Presets

A preset is a named set of settings laid over the config. Order, the last one wins: base
`stt_config.json` < `stt_config.local.json` < preset < UI form or command-line flags. Presets
cover `stt.run` only, not the classic pipeline.

| Preset | What it does |
|---|---|
| `main` (default) | best quality: season voices, existing subtitles, three voice-overs, stress and speech checks |
| `upload` | `main` plus video compression to about 2 GB per episode |
| `fast` | Whisper only, Vosk only, no stress or speech checks, no season voices, no online subtitle search |
| `previous` | no season voices, no subtitles, the old sound level |

Use `--preset upload` on the command line, or the radio "Набор настроек" on the tab "Обработка".

## 7. How the pipeline works, stage by stage

The stages (the names you give to `--until` and `--force`) are `transcribe`, `glossary`,
`speakers`, `translate`, `dub`, `stress`, `voiceover`, `check`, `mix`. Each episode gets its own
work folder, and episodes of one folder share a season glossary and a voice bank.

1. **transcribe** (local). Audio is cut into speech pieces and recognised by Whisper (`--mode
   fast`) or by Cohere Transcribe plus Whisper, whose two texts a language model fuses into one
   (`--mode accurate`). Word times come from forced alignment (matching known text to the audio).
   The video is also scanned twice per second for burned-in captions and name cards (OCR); a
   caption the speech recognition missed or garbled is used instead of it. Who speaks is worked
   out here (section 8), and existing subtitles are looked up first (section 9).
2. **glossary**. A language model builds a list of names, places and terms of the season with
   their Russian spelling. Once per season it also gets the cast (people, nicknames, relations)
   from the season's Wikipedia article or the web.
3. **speakers**. A language model names the speakers and gives a Russian label and a gender,
   using the glossary, the cast, the on-screen name cards, the season voice bank and subtitle
   evidence. The season voices stage runs here (section 10).
4. **translate**. English to Russian, line by line, each line fitted to the time the speaker
   took. Lines that are too long are shortened; lines with a wrong first-person gender are fixed.
5. **dub**. The dubbing script: which lines are voiced and with which voice. Short interjections
   ("Yeah", "Wow") over someone else's speech stay in the original audio.
6. **stress**. Russian stress marks. RUAccent places them; a rule removes marks from short words
   like "на" and "не" (except real cases such as "не было"); words RUAccent left bare get the
   dictionary stress; names and disputed words are re-checked by language models (three of
   them vote). Names get one stress for the whole season.
7. **voiceover** (local). Each line is synthesised by the chosen engines and placed on a timeline
   (section 11). A solver shifts and speeds up lines so they do not collide, and asks for shorter
   text only when needed.
8. **check** (local). Every voiced clip is recognised again in Russian (GigaAM-v3) and compared
   with the text; `speech_check.md` lists mispronounced or swallowed words. `review.json` and
   `review.srt` list lines where the voice may be wrong, for a person to look at.
9. **mix** (local). The Russian track is laid over the original (the original is turned down
   under the voice, section 12), encoded as AAC, and put into one `.mkv` with the original audio
   and the Russian subtitles. Each engine's run adds its own audio track.

## 8. How speakers are found

A reality show has many voices, background talk, shouts and music, so "who says this line" is
the weakest part of the pipeline. The chain:

1. DiariZen v2 splits the audio into turns by speaker (pyannote if `diarizen_env` is missing).
   It also gives an "exclusive" layer: one speaker per instant, the strongest of the overlapping
   ones.
2. Each turn is compared with the voice prints of the other speakers using the ResNet293 model
   (`speaker_refine`); a turn that sounds clearly closer to another speaker moves there.
3. Every word gets a short voice print and a Viterbi pass (a method that picks the most likely
   sequence of speakers) re-labels the words: it prefers the same speaker as the previous word
   unless the voice clearly changes (`word_speakers`, `stt/boundaries.py`).
4. Existing subtitles with speaker names or `>>` turn marks help (section 9).
5. The season voices stage re-assigns words over the people named in the whole season (section 10).
6. A language model names the speakers; the voice bank of the season, the cast and on-screen
   name cards help it. For speakers whose gender stays unknown, gender is decided per line from
   the voice.

How well it works, honestly: on the five S51 episodes E01-E05, with lines labelled by hearing-
impaired subtitles, this chain (before season voices) puts about 63.5% of labelled lines on their person's
speaker. Against the `>>` turn marks of S20 (four episodes), the F1 score of speaker-change
detection went from 0.841 to 0.878 with the exclusive layer and the word-level pass. With
season voices the numbers are in section 10. Several speakers that sound alike or speak only in
shouts are still merged. Plan to check the lines the program flags (section 16).

## 9. Existing subtitles

Many episodes already have English subtitles with speaker information: files for the
hearing-impaired name the speaker ("SANDRA: ..."), and broadcast captions mark a new voice with
`>>`. `stt.subtitles` looks for them before transcription, in this order:

1. subtitle streams inside the video (English text formats; a hearing-impaired stream first);
2. `.srt` / `.vtt` files next to the video and in its `subs/` folder, with the same episode code
   (`S05E02`, `10x05`) or file name (Russian files and the program's own products are skipped);
3. a local archive: `subtitles.archive` is a folder with a `manifest.jsonl`;
4. online (`subtitles.online`, keyless sources: the OpenSubtitles Stremio addon, then Gestdown),
   using `subtitles.imdb`.

`subtitles.imdb` is either one IMDb id or a map `{"part of the video path": id}` (the first part
found in the path wins; `null` means no online search). The shared config maps Survivor and The
Amazing Race; Australian Survivor has `null`. For another show, add your own entry in
`stt_config.local.json`.

What they change: each subtitle line is pinned to the words really heard. Named lines fix their
words to that person; `>>` marks make a speaker change cheaper at that place. The answer,
including "nothing found", is cached in `subs.source.json` in the episode folder. Measured on
S51 E01-E05 (half of the named lines given as the hint, the other half scored): the given lines
went from 64.9% to 84.2% on their person's speaker (S51E01), but lines not given barely changed
(59.7% to 59.8%). Switch off with `--no-subtitles`.

## 10. Season voices

A cluster from the diarizer is often a mix of people, and the word pass compares each word only
with the clusters of its own episode. The season voices stage (`stt.season_voices`, on by
default, model `resnet293-ft`) instead builds a voice print per person from everything the
pipeline has named in the whole season (a person needs 20 words), then assigns every word of
every episode again over those people plus the episode's unnamed clusters. No language model
is involved.

- It needs the episodes to have reached `speakers`. With a mask, `stt.run` does that first for
  all given episodes, and other episodes of the same folder that already have a transcript also
  feed the voices. It works with a single episode too.
- An episode that already has `subs.ru.json` is left alone, so a re-run never repeats the
  translation and the voice-over. `--force speakers` assigns it again.
- The first-pass files are kept as `*.before_voices.json`; `season_voices.json` records what moved.
- The button "Уточнить, кто говорит" on the "Голоса сезона" tab counts them again for the chosen season without
  processing anything else (`stt.run <videos> --season-voices-only`). Translated episodes stay as they are.
- By hand: `subs_env/bin/python -m stt.season_voices /path/to/work/season [--episodes "*S51E0[23]*"] [--dry-run]`.

Measured benefit, through the real pipeline with the trained model (`docs/experiments.md`):

| Test | Before | With season voices |
|---|---|---|
| S51 lines on their person (E01-E05, labelled lines) | 68.4% | 71.6% |
| S20 turn-change F1 (four episodes) | 0.879 | 0.897 |
| S29 turn-change F1 (E01, E03, E07) | 0.860 | 0.871 |
| S51 named words: right / wrong name / no name | 63.7 / 29.5 / 6.8% | 69.4 / 30.6 / 0% |

This is a gain, not a cure: about 30% of words still go to the wrong person. Also, the stage may
give a few words to a person who is not in the episode (a wrong name instead of a neutral voice).
A single episode is enough to build the voices: from one, two, three and five episodes of S51 the
right share was 69.2, 70.0, 69.6 and 69.4%. The owner listened to episodes voiced this way and
made it the default.

## 11. Voice engines

`voiceover.engine` is `f5,teratts,vosk` by default: all three, one after the other, each adding
its own audio track to the video.

| Engine (`--engine`) | Runs on | Voices |
|---|---|---|
| `f5` ESpeech | GPU, in `f5_env` | each person in their own voice, cloned from a 5-9 s reference clip cut from the episodes (the voice bank). People without a clip are spoken by TeraTTS |
| `teratts` TeraTTS | GPU | one female and one male voice (a blend of `ru_f1` and `ru_f2`, and `ru_m5`); the season's host gets a separate voice (`host_voice`) |
| `vosk` | CPU | Vosk's built-in voices (49 female, 56 male); the host is spoken by TeraTTS |

Every engine retries a line with other random seeds when the speech check hears a word missing or
garbled, and keeps the best take. Every synthesised line is stored in a cache
(`output/tts_cache/`), so a re-run, or a re-voice after a correction, only synthesises lines that
changed. With `voiceover.parallel_cpu` (default) Vosk runs beside the GPU engines. A 10-minute
test piece of S29E05 in all three engines took 25.4 minutes one after another and 16.4 minutes in
two lanes with the CPU cap.

The voice bank (`stt.voice_bank`, UI tab "Голоса сезона") keeps a voice print and reference clips
for each named person of a season.

## 12. Sound level

Defaults: the voice-over is raised by 3 dB; music and effects of the original go 6 dB down under
the voice, the original voices 12 dB down; the whole mix is brought to -16 LUFS (a loudness
measure) with a peak limiter. Separating voices from music is done with Hybrid Demucs. The
original audio track in the video is not changed. These are `mix.voice_gain_db`,
`mix.rest_duck_db`, `mix.duck_db`, `mix.loudness_lufs`, `mix.duck` in `stt_config.json`
(`"duck": "all"` turns the whole original down).

To hear other levels before changing anything: UI tab "Звук и размер видео", or
`subs_env/bin/python -m stt.mix_preview <episode folder> <video> --engine f5 --voice-gain-db 6 --out piece.wav`
(a 45 s piece; without `--start` it picks the place where speech meets the loudest original
sound). To apply new levels to a finished episode without voicing it again:
`subs_env/bin/python -m stt.run <video> --engine f5,teratts,vosk --force mix`.

## 13. Video compression

Off by default. `mix.video` is `{"compress": false, "target_gb": 2.0, "codec": "hevc_nvenc",
"original_audio_kbps": 160}`. It re-encodes the picture at the same resolution to a bitrate that
makes the finished file about `target_gb` (1 GB = 10^9 bytes). It uses the graphics card encoder
(`hevc_nvenc`, `h264_nvenc`) through PyAV and falls back to `libx265`/`libx264` on the CPU. The
preset `upload` turns it on.

One measured result: S51E01 (86 min) went from 7.07 GB to 2.07 GB in 15 minutes on an RTX 4070
Ti SUPER; SSIM (a similarity score, 1 is identical) was 0.959 on a 60 s sample, and fine skin
detail in close-ups is softer.

For a finished video:

```bash
subs_env/bin/python -m stt.video_compress FILE [--target-gb 2] [--codec hevc_nvenc] [--out path]
subs_env/bin/python -m stt.video_compress FILE --sample-minute 10 --sample-seconds 60   # a piece to judge quality
```

It never changes the input; the result is `<stem>.small.mkv` next to it (a sample is
`<stem>.sample.mkv`). All audio and subtitle streams are kept.

## 14. The browser UI

`bash scripts/run_ui.sh`, address http://127.0.0.1:7861. Tabs:

- **Season bar** (above the tabs) - the season picker and "＋ Новый сезон" (both folders have a "📁 Выбрать…" button:
  the desktop dialog when there is a desktop and `zenity`, else a folder browser on the page; the results folder
  `STT_<source folder>` is suggested and made on "Сохранить сезон"); one season drives every tab, remembered in the UI
  state file `ui.local.json` (not in git; older versions kept it in `stt_config.local.json` under `ui.seasons`, which
  is still read once). Seasons next to the configured folders are found on the first start. When a season is
  picked that has no settings of its own while the previous one had some that differ from the global ones, the page
  asks which to take. The first line of every tab says what the tab writes where.
- **Обработка** - pick episodes (with a table of their states), a preset, engines and options; start and stop; a
  status line in words (what runs now, how far, time left), a progress table (shared stages and
  one column per engine), "Контрольные точки" (what is saved and whether stopping now is safe),
  a readable log "Ход работы" (the raw log is under "Подробный лог") and the finished files.
  The last job is kept in `output/ui_last_job.json`: after a restart of the page or the computer
  it shows up again (still running: it can be watched and stopped; gone: "прервано"), and
  "▶ Продолжить" runs the same command with the same settings file, so finished stages, voiced
  lines and paid answers are not redone. "Пробный отрывок" voices a few minutes
  from a chosen minute into a `_проба` folder, to try settings fast.
- **Голоса сезона** - the people of the season and the voice clips ESpeech clones (✅ found automatically,
  ✋ given by hand, ❌ none, with the reason). Listen to a clip, find another automatically, or give a stretch by
  hand: pick an episode, write ranges like `12:30-12:41`, press "Вырезать и послушать" (the piece is cut, cleaned of
  music, the text recognised and editable), then save it as the person's clip; an automatic search never replaces it.
  After a clip changes a banner offers "Переозвучить ESpeech (N серий)": only that person's lines are voiced again.
  Also "Пересобрать все образцы", "Уточнить, кто говорит" (count the season's voices again) and the names check.
- **Status strip** (under the season bar, on every tab) - the one long task that runs now (processing, clips, re-voice,
  compression, re-mix) with its time and last progress line and an "Остановить" button; after the end "✅ Готово" or
  "❌ Не получилось" with the reason. One long task at a time; a second button press says which one runs. Every
  button answers at once: long actions start a task, medium ones show "⏳ ..." on the button and a line of what
  happens, quick ones show a message.
- **Проверка и правка реплик** - every Russian line, filters for the lines the program flagged,
  the original and the voice-over of a line, editing, the Subtitle Edit round trip, and
  re-voicing only what changed.
- **Звук и размер видео** - hear a 45 s piece with chosen levels beside the current mix, find
  the loudest place, save settings, re-mix an episode without voicing again, compression
  settings, shrink a finished video, a 60 s sample.
- **Настройки** - engines, models, voices, checks, mix. Global settings: `stt_config.local.json` unless another
  file is chosen on the tab (created as a copy when new; kept in `ui.local.json`); saved as differences only. Settings
  of a season: `<results>/season_settings.json`, a snapshot of all the tab's values that applies to that season only;
  the radio "Сохранять изменения в" picks where "Сохранить" writes, "Убрать настройки сезона" deletes the file.
  "Сбросить" puts the standard values into the chosen file. Also the TTS cache cleanup.
  Fields are laid out so every label is readable. Vosk voices are picked by name (number, name,
  gender), and "▶ Послушать" next to every voice plays a sample phrase in the voice chosen now
  (made in a separate process on the CPU, kept in `output/voice_previews/`; the first Vosk
  sample takes about 45 s, TeraTTS about 12 s, later listens are instant).
- **Справка** - the help text in Russian (`ui/help.md`).

One job runs at a time. The UI text is Russian.

## 15. Output files and folders

Folders (`paths` in the config; personal ones in `stt_config.local.json`):

- `work`: one folder per episode (default `output/stt/<episode>/`);
- `season`: the season's glossary, cast, voice bank and voice clips (default: `work`);
- `finished`: finished videos in a folder named like the season's video folder, `STT_<season>`
  by default (default: each episode's work folder).

Inside an episode folder: `subs.en.json`, `subs.ru.json` (and `.srt` copies), `dub.ru.<engine>.json`,
`stress.json`, `speakers.json`, `glossary.json`, `ocr.json`, `words.json`, `subs.source.json`,
`subtitle_hints.json`, `season_voices.json`, `voiceover.<engine>/` (`export/track.wav`,
`report.md`, `speech_check.md`), `review.json`, `review.srt`, `logs/`.

The finished video is `<name>.ru.mkv` with every voice-over made so far as its own audio track
(order in `mix.track_order`, the first is the default track), then the original audio, then the
Russian subtitles. With `"mix": {"tracks": "separate"}` you get `<name>.ru.<engine>.mkv` per
engine instead. The intermediate mixed WAV (~0.5 GB) is deleted after encoding unless
`keep_wav` is true. `output/`, `models/` and personal settings are in `.gitignore`; only code,
configs and documentation are committed.

## 16. Correcting lines

The program sometimes assigns a line to the wrong person, mixes up a male and female voice,
translates badly or stresses a word wrongly. Fix it without voicing the whole episode again:

- in the UI, tab "Проверка и правка реплик", or
- in Subtitle Edit: `subs_env/bin/python -m stt.subtitle_edit export <episode folder> [--review] [--srt]`
  writes the lines into `<episode>/edit/` (ASS: speaker in the "Actor" column, gender as the
  style "Мужской" / "Женский", "Не озвучивать" to skip a line; or SRT as `[Speaker] text`);
  edit the file, then `... import <episode folder> <file> [--dry-run]` lists the changes and
  writes them into `corrections.json`. Do not change timings or the "Effect" column (it holds
  the line number).

`corrections.json` in the episode folder is keyed by line number:

```json
{"123": {"gender": "female"}, "130": {"speaker": "Julie McGee"}, "140": {"text": "Новый текст"}, "141": {"skip": true}}
```

The next `stt.run` of the episode redoes the dubbing script, stress check, voice-over (only the
changed lines are synthesised), checks and mix. A stress typed as `+` before the vowel
(`зам+ок`) wins over everything and does not appear in the subtitles. Another manual file is
`speakers.override.json` (then `--force speakers`).

## 17. Configuration

- `stt_config.json` - the shared settings: mode, models, diarizer, `season_voices`,
  `subtitles`, language models (`llm`), `voice_bank`, `cast`, `translate`, `dub`,
  `voiceover`, `mix`, `crowd` (an experimental layer of crowd shouts, off), `presets`, `paths`.
- `stt_config.local.json` - your differences, not in git, laid on top.
- `voiceover_config.json` - details of voice synthesis and timeline fitting (`voiceover.config`
  points to it).

Order: `stt_config.json` < `stt_config.local.json` < preset < UI form or command-line flags. In the UI the
global settings file (`stt_config.local.json` by default) is followed by the settings of the season
(`<results>/season_settings.json`, if it exists), then the preset and the form; `stt.run` on the command line reads
`stt_config.local.json` only.
Language models: one switch, `llm.model` (default `openai/gpt-6-luna`), sets every step;
`llm.stages.<step>` overrides one step. Per-step prices measured on one 64-minute episode are in
`docs/stt_local.md`.

## 18. The older "classic" pipeline

Before the main pipeline there was a simpler one for ready Russian subtitles:
SRT -> JSON (`utils.srt_to_json`) -> RUNorm + RUAccent (`utils.text_normalizer`: number
normalisation and stress) -> Vosk TTS (`synthesize.synthesize_cli`) -> mix
(`utils.audio_mixer`). It is still there, and `uv run -m voiceover.pipeline run <subs.json> --out <dir>`
voices ready Russian subtitles with the engines of the main pipeline. Presets do not apply to
it. Details: `AGENTS.md` section 4 and `docs/pipeline.md`.

## 19. Training the speaker model

The weights `models/speaker/resnet293-ft/avg_model.pt` are the ResNet293 voice model
fine-tuned on Survivor speech: 1-second windows around a word, cut as the pipeline cuts them,
labelled by subtitles that name the speaker, 334 people of 20 seasons. On S51E01-E05 the
same-voice/other-voice error went from 25.0% to 20.6%. The scripts are in
`scripts/speaker_training/`, run in this order from the repository root:

1. `build_words.py` - Whisper word times of the training episodes;
2. `align_lines.py` - pins each subtitle line to the words really spoken;
3. `build_dataset.py` - writes the training clips into one file;
4. `train.py` - the training itself.

Their docstrings say what each takes and writes; the data folder is `output/subs_train`. These
are research scripts kept so the model can be reproduced, not a polished tool, and they need
episodes with subtitles that name speakers. Results of the experiments around them are in
`docs/experiments.md`.

## 20. Tests

```bash
subs_env/bin/python -m pytest tests -q
```

## 21. Repository layout

| Path | What is there |
|---|---|
| `stt/` | the main pipeline: `run.py` (entry point), recognition, speakers, subtitles, season voices, translation, mixing, video compression |
| `voiceover/` | voice engines (Vosk, TeraTTS, F5), timeline solver, stress check, speech check, TTS cache |
| `ui/` | the browser UI (`dub_app.py`), settings page, help text `help.md` |
| `utils/` | shared helpers: text normalisation, audio mixing, LLM transports, ffmpeg wrappers |
| `synthesize/` | the Vosk TTS command line used by the classic pipeline |
| `scripts/` | `install.sh`, `setup_envs.sh`, `run_ui.sh`, `status.py`, `speaker_training/` |
| `tests/` | unit tests |
| `docs/` | documentation, experiments and the roadmap |
| `config/opencode_clean/` | a bare OpenCode configuration used for language-model requests |
| `stt_config.json`, `voiceover_config.json` | shared settings |
| `models/`, `output/` | downloaded and trained models, results (not in git) |

## 22. Further reading

- `docs/stt_local.md` - the main pipeline in detail: models, language models and their costs,
  measurements.
- `docs/experiments.md` - what was tried, what helped and what did not.
- `docs/roadmap.md` - what is done and what is next.
- `docs/pipeline.md` and `AGENTS.md` - the classic pipeline and the working rules of the repository.
- `ui/help.md` - the in-app help (Russian).
