# English video → Russian voice-over

Package `stt/` turns an English episode into speaker-attributed English subtitles, names the
speakers, determines their gender (Russian grammar, voice choice), translates to Russian fitted
to the timing and prepares the dubbing script. The `voiceover` package synthesises it (Vosk or
TeraTTS, a voice per speaker gender), fits the lines to the timeline and exports the voice
track; `stt.mix` puts it over the ducked original and muxes a video copy. What comes next
(speaker bank, voice cloning, tips to test): `docs/roadmap.md`; state and pitfalls for a new
agent: `docs/handoff.md`.

```
video ─► audio 16 kHz ─┬─► VAD chunks ─► Whisper  (fast)
                       │               └► Cohere + Whisper → LLM fusion → CTC forced alignment (accurate)
                       └─► pyannote diarization (who speaks when) + gender/age per speaker
video frames (2 fps) ─► OCR: burned-in captions, name cards           ocr.json
        words + captions → speakers (sentence-level smoothing) → cues subs.en.json
        LLM: season glossary, speaker names, gender check             glossary.json, speakers.json
        LLM: EN → RU per cue time window, shortening, gender fix      subs.ru.json
        dubbing script: voiced lines, a voice per line                dub.ru.<engine>.json
        stress: guessed words / homographs re-checked by 3 LLMs       stress.json
        voiceover: TTS + timeline solver + shortening                 voiceover.<engine>/export/track.wav
        check: Russian ASR (GigaAM-v3) of every spoken clip           voiceover.<engine>/speech_check.md
        mix: voice over ducked original (AAC), one video with every   <name>.ru.<engine>.m4a
             voice-over as its own track + original + subtitles   <name>.ru.mkv
```

## Quick start

```bash
bash scripts/install.sh                         # once (and after `git pull`): everything below in one go
bash scripts/run_ui.sh                          # the browser UI, opened in the browser

# or by hand:
bash scripts/setup_envs.sh                      # subs_env (once)
subs_env/bin/hf auth login                      # once; accept the terms of pyannote community-1 and Cohere Transcribe
# models/vosk-model-tts-ru-0.10-multi for Vosk; OPENROUTER_API_KEY and DEEPSEEK_API_KEY in .env
subs_env/bin/python -m stt.selftest             # GPU, models (downloads them), keys; --quick: no downloads

# whole season, up to the Russian dubbing script (no TTS):
subs_env/bin/python -m stt.run "/media/.../Australia S10 1080p/*.mkv" --until dub
# prepare a whole season (every episode up to speakers: transcript, on-screen text, who speaks, names, season voices; no translation or voice-over):
subs_env/bin/python -m stt.run "/media/.../Australia S10 1080p/*.mkv" --out-root R --prepare
# one episode end to end (voice-over in all three engines by default, checks, mix); --engine teratts for one engine:
subs_env/bin/python -m stt.run ep01.mkv
```

Browser UI: `subs_env/bin/python -m ui.dub_app` (http://127.0.0.1:7861). Tab "Обработка":
episodes of a folder, engines, recognition mode, LLM, "embed subtitles", "duck only voices",
stress check and speaker refinement; "test piece" voices N minutes from minute M into
`<out>/_проба` (with a short video) to try settings quickly. One job at a time, engines one after
the other; progress per episode and stage, the log, the finished files. Tab "Настройки": the
defaults people change (folders, engines ticked at start, recognition, models, voices, checks,
mix), saved into `stt_config.local.json` as differences from `stt_config.json` only (other
personal keys there, such as the subscription, are kept; the previous file stays as `.bak`);
"Сбросить" removes the page's keys. The "Обработка" form starts from these values. Tab "Проверка и
правка реплик" (since 29.10; it replaced "Проверка реплик"): every Russian line of an episode
(filters: the review list, edited lines, lines too long for their time; search), the original and
the voice-over of a line, and its text, speaker (with a voice sample: the season clip, else the
person's longest line), gender and "not voiced", saved into `corrections.json` with a length check
(`stt.subtitle_edit.fit`: syllables per second of the line's time against the voice-over's base
5, comfort 6.3 and ceiling 6.7); the Subtitle Edit round trip (below); "Переозвучить изменённое".
Tab "Справка": how to work with all of it, in plain Russian (`ui/help.md`).

Editing lines outside the UI (`stt.subtitle_edit`): `export <episode folder> [--review] [--srt]`
writes the lines as they will be voiced (corrections applied) into `<episode>/edit/`: ASS for
Subtitle Edit (speaker in "Actor", gender as the style "Мужской" / "Женский", "Не озвучивать",
the line number in "Effect"; `--review` only the review list, with what was heard as a hidden
`{...}` note) or SRT (`[Speaker] text`). `import <episode folder> <file> [--dry-run]` matches the
edited lines by number or start time (±0.5 s), lists the changes and writes them into
`corrections.json` as differences from `subs.ru.json` (a line edited back to its translation
loses its correction; unknown speakers and new lines are reported, not applied). The next
`stt.run` of the episode redoes the dubbing script, the stress check, the voice-over (only the
changed lines are synthesised: every other take is in the TTS cache), the checks and the mix.
A stress typed as `+` before the vowel ("зам+ок") wins over RUAccent and the stress check and
never reaches the video's subtitles (`utils.text_normalizer.keep_user_stress`).

Quotes and "все/всё": the text sent to every voice engine (`utils.vosk_text.normalize_vosk_text`, the last
step of the text preparation, after ru-normalizr) has no quote marks or apostrophes and brackets turned into
commas, because ESpeech says a stray "ж/з/с" sound before a word after a straight quote or "(". The subtitles and
the review list keep the quotes. The stress check also sends every "все"/"всё" (and RUAccent's other е/ё pairs)
to its screening with the whole line: the models choose the spelling that fits the sentence, and the fix
writes it into the voice-over text only.

`scripts/install.sh` (Linux, NVIDIA driver with CUDA 12.8+): checks the GPU, disk and `uv`
(offers to install it), creates subs_env, f5_env and diarizen_env (`--basic`: subs_env only;
existing environments are kept and only updated), downloads the Vosk voice model, runs `hf auth
login` if needed (and lists the two gated models whose terms to accept), asks for the OpenRouter
key without showing it and appends it to `.env`, runs `stt.selftest` (downloads ~30 GB of models
the first time) and can add a menu shortcut to `scripts/run_ui.sh`.

Defaults live in `stt_config.json` (mode, models, OCR, LLM, TTS engine and voices, checks,
ducking). Stages already done are skipped; `--force STAGE` recomputes it and the following
ones; `--mode fast` uses Whisper only.

Folders (`"paths"`, personal ones in `stt_config.local.json`): `work` — episode work folders
(default `output/stt`; `--out-root` overrides it); `season` — the season's glossary, cast, voice
bank and voice clips (default: `work`); `finished` — finished videos, in a sub-folder named
like the season's video folder (default: each episode's work folder). Finished videos
(`"mix": {"tracks": ...}`): `"combined"` (default) is one `<name>.ru.mkv` per episode with every
voice-over voiced so far as its own Russian audio track (order and default track:
`track_order`, ESpeech first; titles: `track_titles`), then the original audio and the Russian
subtitles; each engine's run adds its track. `"separate"` writes `<name>.ru.<engine>.mkv` per
engine, each with a full copy of the video. The mixed WAV (~0.5 GB per episode and engine) is
deleted once encoded to AAC unless `"keep_wav": true`.

Before the episodes `stt.run` asks the OpenCode subscription once (`llm.opencode` models): an
expired login is logged as a warning (the LLM steps then go through the paid router; with
`"fallback": false` the run stops instead) and shown in the UI status, as is a fallback in the
middle of a run. A stage that runs out of GPU memory (another program on the card) is started
again after a minute, up to twice; ASR with half the batch size.

Several engines in one run: `stt.run ... --engine f5,teratts,vosk` (the UI does this) goes
episode by episode. With `"parallel_cpu": true` in `voiceover` (default) Vosk, which synthesises
on the CPU, runs in a second lane beside ESpeech and TeraTTS: the GPU lane makes an episode's
shared stages (recognition, glossary, names, translation), then both lanes voice it, Vosk with half
the CPU threads (ONNX Runtime sessions capped: on all cores it slowed ESpeech's own CPU work from
7.5 to 12.3 min on a 10-min piece); the stress check, the review list and the finished video take
turns. A 10-min test piece of S29E05 in all three engines, cold caches: 25.4 min one after the
other, 19.6 min in two lanes without the CPU cap, 16.4 min with it (a third less; the speech check
of the clips the same). The stress check is no longer redone for each engine (the lines are the
same: `stress.json` records them).

Translating ahead (`"overlap_episodes": true`, top level of the config, default; UI: "Переводить следующую серию,
пока озвучивается текущая"). While the GPU lane voices episode N, a lookahead thread runs episode N+1 up to and
including `translate` (LLM only, about 4 min), so the GPU does not wait for it. One episode ahead at most, and only
when N+1 needs no GPU stage first (recognition, glossary, names: with season voices those are done before the lanes
start); otherwise it is logged and left to the lane. The lane waits for the lookahead before it touches that
episode. It helps only when several episodes run in one command or one UI job; it is off for one file, for
`--until` before `translate` and for `--force` at or before `translate` (that stage is forced once per episode).
An error in the lookahead or in a lane stops the whole run.

Parallel work on the graphics card (`"gpu": {"parallel": "auto"}`, UI: "Параллельная работа на видеокарте"). Three
speed-ups overlap GPU work: the on-screen text scan beside diarization and recognition, the mix of one engine
(voice separation) beside the voicing of the next, and the video compression beside the voicing. They were
measured on a 16 GB card. `auto` (default) first asks `nvidia-smi` how much video memory the card has and runs an
overlap only if the card holds the heaviest job it meets, the job itself and a 1.5 GB margin
(`stt/gpu_budget.py`: about 10.4 GB for the scan, 9.8 GB for the mix, 8.4 GB for the compression) and that much is free
at the moment; otherwise the jobs go one after another (the scan after the alignment, the mix before the next engine,
the compression in the final mux): slower, same result. An unknown card counts as too small. `on` always overlaps,
`off` never. If the voice separation itself runs out of memory it waits 30 s, tries again, and then runs on the
processor. The peak video memory of the jobs we had no number for (ESpeech, separation, compression, each stage's
process) is written to `logs/vram_peaks.json` of the episode and to `f5_timing.json`.

## Stopping and continuing

A run can be stopped at any time (Ctrl+C, "Стоп" in the UI, or the computer restarting) and started
again with the same command. Finished episodes and stages are skipped (`[<episode>] <stage>: done earlier`);
what a stop in the middle of a stage keeps:

| stage | kept | redone |
|---|---|---|
| transcribe | every finished worker step (diarization, each ASR model, alignment, OCR, voices) | the step that was running |
| glossary, speakers, translate, stress | every LLM answer already received | only the requests without an answer |
| dub | nothing | the whole stage (seconds) |
| voiceover | every synthesised line (`output/tts_cache`; ESpeech writes each line as it is done) | the unfinished lines |
| check, mix | nothing | the whole stage (minutes) |

LLM answers of an unfinished stage are kept in `<episode>/.partial/llm/<stage>/`, one file per request (named by a
hash of the model, the prompt and the settings). They are used only to continue the same stage with the same
inputs, and the folder is deleted when the stage finishes, so `--force translate` on a finished episode asks the
model again. Files other stages read as "exists means done" (worker results, subtitles, per-line cache records,
WAVs of ESpeech) are written to a temporary file, flushed and renamed, so a power cut leaves the old file or the
new one, never half of one; a per-line cache record that still cannot be read counts as missing and the line is
synthesised again.

Progress is written for the UI (and for you) to `<episode>/.progress/<stage>[.<engine>].json`:

    {"stage": "voiceover", "engine": "f5", "state": "running", "done": 120, "total": 640, "unit": "lines",
     "started": 1793650000.0, "updated": 1793650300.0, "finished": null, "pid": 12345, "boot": "<boot id>"}

`state` is `running`, `done` or `failed`; `unit` is `lines` (translation, voice-over), `requests` (glossary, speakers,
stress check), `steps` (recognition), `clips` (speech check) and null for stages without a count (dub, mix). A
`running` file whose process is gone (or whose `boot` is not the current boot) means the stage was interrupted
there. `stt.progress.RESUME_NOTES` has the one-line text per stage shown in the UI.

The log has, besides `[<episode>] <stage> ...` at the start and `<stage>: done earlier` for skipped stages:

    [<episode>] <stage>: finished in <n> s              (shared stages)
    [<episode>] <stage> (<engine>): finished in <n> s   (dub, voiceover, check, mix)
    [<episode>] voiceover (f5): 25/640 lines            (about every 25 lines or clips of a stage run as a separate program)

The ESpeech worker's own messages go to `output/tts_cache/<identity>/f5_worker.log` (the end of it is in the error
if the worker fails).

## Presets

`presets` in `stt_config.json` are named sets of settings laid over the config: `stt.run --preset <name>`, or the radio
"Набор настроек" on the UI tab "Обработка" (it fills the form, which can still be edited). Order, last wins: `stt_config.json`
< `stt_config.local.json` < preset < form / command-line flags. They cover `stt.run` only, not the classic pipeline.

| Preset | What it is |
|---|---|
| `main` (default) | best quality: season voices, existing subtitles, three voice-overs, stress and speech checks |
| `upload` | `main` plus `mix.video.compress` to about 2 GB per episode |
| `fast` | `mode: fast` (Whisper only), Vosk only, no stress or speech checks, no season voices, no online subtitle search |
| `previous` | no season voices, no subtitles, the old sound level (`rest_duck_db` 3, `voice_gain_db` 0, no loudness target) |

## What runs where

| Step | Where | What is sent out |
|---|---|---|
| Audio decoding, VAD, ASR (Whisper, Cohere), alignment, diarization, gender/age | local GPU | nothing |
| On-screen text (RapidOCR: detection on GPU, recognition on CPU) | local | nothing |
| ASR fusion, glossary, speaker names, translation, stress re-check, voice-over shortening | OpenRouter LLMs | text only, see below |
| TTS (Vosk on CPU, TeraTTS on GPU), timeline fitting, Russian ASR check, mixing, muxing | local | nothing |

No audio ever leaves the machine. No manual corrections are applied anywhere; the only human
inputs are optional files: `speakers.override.json`, an edited `glossary.txt`, and the
`voiceover` overrides. Reference subtitles are used only by the benchmarks, never by the pipeline.

### Language models

One switch, `llm.model` in `stt_config.json` (default `openai/gpt-6-luna`), sets the model of
every step; `llm.stages.<step>` overrides one step (`fuse`, `glossary`, `speakers`, `translate`,
`voiceover`, `stress`, `stress_screen`, `translate_proofread`). By default only the stress re-check and the
translation proofreading (`translate.proofread`, sol) use other models. The proofreading reads the Russian
lines once more for real errors only (non-words such as "познаться", broken agreement, wrong meaning,
untranslated English, names off the glossary); a fix is taken if it fits the time budget and the speaker
gender, and all proposals are listed in `subs.ru.proofread.json`.

The shared default is OpenRouter with gpt-6-luna (good results, much cheaper than sol; the UI
offers luna or sol). Personal settings go into `stt_config.local.json` next to `stt_config.json`
(not in git, merged on top by `stt.run` and the UI), e.g. requests through a ChatGPT subscription:

```json
{"llm": {"model": "openai/gpt-6.1-sol", "opencode": ["openai/*"], "router": ["orcarouter", "openrouter"],
         "stages": {"stress": ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5",
                               "openrouter:google/gemini-4-argon"]}},
 "ui": {"llm_transport_choice": true}}
```

This is the user's own setup since 29.10: OpenAI models (sol for every single-model step) through
the subscription, other models through OrcaRouter, OpenRouter only where OrcaRouter does not work
or lacks the model (gemini-4-argon; web search `:online` of the cast lookup is OpenRouter's only).
A router that answers 404 (no such model) hands over to the next one at once, without retries.
The season's cast (`llm.stages.cast`, once per season): a model id ending in `:online` searches
the web (OpenRouter only; the shared default luna, then Claude Haiku, ~$0.02); any other model
reads the season's English Wikipedia article, which the program finds and fetches for it, so the
user's `"cast": ["openai/gpt-6.1-sol", "openrouter:openai/gpt-6-luna:online", ...]` costs nothing
and falls back to the web search only when Wikipedia has no article (`stt/cast.py`: S51, S29,
S20, Australian S10 found that way).

With an OpenCode Go subscription as well (`opencode auth login`, provider `opencode-go`), its
models are named `opencode:opencode-go/<model>` (no paid fallback: the router has no such ids;
a voter that does not answer is left out of the vote). The stress vote is then free: on S29E03's
164 checked sentences sol + gemini-4-argon (OpenCode Go) + any third gave exactly the 27
corrections of the paid trio (sol, claude-sonnet-5.5 by OrcaRouter, gemini-4-argon by
OpenRouter), because sol and gemini agree on 162 of 164; gemini through OpenCode Go agreed with
the paid trio wherever the three agreed (149 of 149). Third voters there: deepseek-v4-pro and
grok-4.7 147/149 but over 20 min for 55 sentences (timeouts), kimi-k3.1-pro 140, glm-5.3 139,
claude-haiku-5-5 118 in ~1 min: haiku, as it only breaks the rare sol / gemini ties. The user
chose not to use OpenCode Go for the votes (29.10): OrcaRouter / OpenRouter, as above.

`ui.llm_transport_choice` shows the UI switch "Подписка (opencode) / API OpenRouter".

Where requests go (`utils/llm_route.py`): transports in front of a model id, tried in order.
`stt.run` adds them from the config:

- models matching a pattern of `llm.opencode` (default `["openai/*"]`) go through OpenCode
  logged in with a ChatGPT subscription (`opencode auth login`, OpenAI), so they cost nothing
  beyond the subscription; with `llm.fallback` (default true) the paid router takes over when
  OpenCode fails for good or 3 times in a row (`opencode+orcarouter:openai/gpt-6.1-sol`);
- all other models go to `llm.router`: `orcarouter` (key `ORCAROUTER_API_KEY`) or `openrouter`
  (`OPENROUTER_API_KEY`);
- an id that already has a route keeps it: gemini-4-argon is not on OrcaRouter, so the stress
  voter is `openrouter:google/gemini-4-argon`.

OpenCode runs as a private `opencode serve` (OpenCode 2.x) per process, on 127.0.0.1 with a
random password and the clean preset `config/opencode_clean` (no tools, project config, plugins
or MCP), through its stateless generation route. It reports no token counts, and OrcaRouter no
cost, so `usage` shows $0 for both. OrcaRouter rejects two OpenRouter-only fields for OpenAI
models (`usage.include`, DeepSeek's `thinking`); the clients drop them there. S29E01 with
gpt-6.1-sol for every step through OpenCode: all LLM steps in 15 min, fusion WER 18.03%
(17.93% through OpenRouter), translation as through OpenRouter (2 of 918 lines over budget).

OpenRouter prices of S29E01 (64 min) per step, luna by default vs gpt-6.1-sol everywhere:

| Step | What is sent | What comes back | luna | sol |
|---|---|---|---|---|
| `fuse` (accurate mode) | per ~30 s chunk: the English text of Cohere and Whisper, neighbouring chunks, glossary | one corrected English text per chunk | $0.025 | $0.24 |
| `glossary` | the English transcript, the season glossary | names, places, terms with their Russian form | $0.004 | $0.08 |
| `speakers` | transcript with speaker ids, voice gender estimates, on-screen names | name, role, gender per speaker | $0.003 | $0.085 |
| `translate` | English cues with time budget, speaker cards, glossary; then over-long lines and lines with a wrong first-person gender | Russian lines (translation, shorter versions, gender fixes) | $0.036 | $0.47 |
| `stress_screen` + `stress` | Russian sentences with a guessed word or homograph | the stress of that word in that sentence; luna proposes, sol, claude-sonnet-5.5 and gemini-4-argon vote | $0.14 | — |
| `voiceover` | a Russian line that still does not fit its slot, its scene | shorter versions and a meaning check (0 calls on S29E01/E02: the translation already fits) | $0 | $0 |

Sol everywhere: $1.0 per episode instead of $0.21 and ~3 min longer. It recognised slightly
better (WER against the closed captions 17.93% vs 18.17%, within the LLM's run-to-run spread),
translated more completely and naturally and fitted the time budget better (2 of 921 lines over
it before shortening vs 53 of 911 with luna); speaker names agreed on 89% of the speech
(the rest: Вентворт/Уэнтворт and one voice cluster that holds two women).

## On-screen text

The show burns in English captions for quiet or hard-to-hear speech (whispers, distant or
overlapping voices, medics, hidden-camera moments) and shows name cards when people first
appear. `stt.ocr` scans the video at `ocr.fps` (2 by default, ~7 min per 90-min episode on an RTX 4070 Ti
SUPER), groups identical text into events with start/end times and classifies them by position
and size: `caption` (centred, bottom, sentence-like), `name_card` (+ descriptor line such as
"SEASON 6 WINNER"), `title`, `overlay` (recurring logos/ads), `scene` (props, signs, shirts).

- Captions are merged into the word stream before speaker assignment. Captions appear ~0.25 s
  before the speech and stay ~0.6 s after it, so back-to-back captions are aligned together
  against the nearby ASR words. If ASR already has ≥70% of a caption's words they stay (exact
  timings); otherwise the garbled ASR sentence is replaced by the caption text (`source: "ocr"`
  on the words, `ocr: true` on the cue), and a caption the ASR missed entirely is inserted at the
  nearest sentence boundary. A caption under another transcribed sentence is a quieter voice
  and stays only in the original audio (and in the Russian on-screen track below).
  S10E01: 61 captions — 32 already right in ASR, 17 replaced, 11 added, 1 under other speech.
- Since 4.11 the caption wins for its own utterance: a caption ASR already had is written with the
  caption's text over the same word times (`caption: true` on its words), and a captioned run of
  words is its own cue (`on_screen: true`). `on_screen.voice` decides what happens to them:
  `text` (default) does not voice them, the Russian on-screen text shows them and the original audio
  stays audible (the mix ducks only under voiced lines); `neutral` voices them with the track's neutral
  voice (`on_screen.neutral_gender`), not the hero's; `hero` voices them like any line (dub:
  `--on-screen-voice text|neutral|hero`). A manual speaker/gender fix in `corrections.json` wins and the
  line is voiced. With `on_screen.track` off, `text` falls back to `neutral` (logged warning), so the
  captioned lines are not lost. S51E01: 138 captions, 29 of them under other speech (mostly whispers
  under a confessional) only ever shown as text.
- In `text` mode a line with at least 30% of its time under a caption on screen (±0.4 s) is not voiced either
  (`stt.dub.under_captions`): when the ASR and the caption disagree, the ASR line is usually the caption's own
  speech misheard (S48E01 1:21: "I don't like being in the snow bathroom" under "I really needed this so badly").
  A long line that only touches a caption (a confessional over a whispered exchange) stays voiced. S48E01: 75 such
  lines silenced, 64 touching a caption kept.
- `onscreen.json` (written by the transcribe stage): captions, name cards (+ descriptor) and lower
  thirds with their screen box; `python -m stt.onscreen translate <folder>` (run in the translate
  stage) writes `onscreen.ru.json` with the cast's Russian names and the glossary, cached per text and
  speaker gender (each caption gets the gender of whoever says it, taken from the overlapping cue of
  `subs.en.json`, so "I was waiting" becomes "Я ждала" for a woman; name cards get the named person's).
  OCR text is repaired deterministically first (`stt.ocr_repair`: stray dots, glued words such as
  "yourname", damaged job lines), truncated animation frames are folded into the full text, and card
  lines read as one ("DEVIN ACTOR") are split into name and descriptor.
  `stt.onscreen_render` turns it into an ASS track muxed as "Надписи на экране" (default, forced,
  DejaVu fonts attached): every Russian item sits on a dark box (about 85% opaque, a little larger than the
  English text it translates) that covers the English: a caption, a card (name in bold, job and tribe
  below) and a lower third alike. The box may grow slightly for a longer Russian text, overlapping items are stacked. The track is never burned into the picture by default, so a viewer can turn it off;
  `on_screen.burn_in_when_compressing: true` burns the same text in when the video is compressed anyway
  (`mix.video.compress`, preset `upload`), for players without subtitle tracks. Off:
  `"on_screen": {"track": false}`.
- Name cards, titles and caption labels ("Shonee: …") become `on_screen` evidence for the
  speaker that talks most while they are shown (a card shown in a pause goes to the speaker of most
  words in the next 10 s, since 31.10); the speaker-naming LLM sees it. On S51E01-E05 this rule put
  140 of 191 cards with an SDH-known person on that person's main speaker, the best of four rules tried.
- `--no-ocr` (or `"ocr": {"enabled": false}`) skips the stage; `--force ocr` rescans.

## Existing subtitles as a hint for who speaks

Many episodes already have English subtitles with speaker information: SDH files name the speaker
("SANDRA: ...") and broadcast captions mark a new voice with ">>". `stt.run` looks for them before the
transcription (`stt.subtitles.find_subtitles`, in this order):

1. the season's own subtitle folder (`subtitles.folder`, best set in the season's settings): `.srt`/`.vtt` files with
   the episode code,
2. subtitle streams inside the video (English text codecs only; a hearing-impaired/SDH stream first),
3. `.srt`/`.vtt` files next to the video and in its `subs/` folder, with the same episode code
   (`S05E02`, `10x05`) or file name; Russian files and our own products (`_ru`, `translated`,
   `whisperx`, `_edit`, `_synced`) are skipped,
4. the subtitle base (`subtitles.base`): one chosen file per episode, made by `stt.subtitle_base` (below),
5. a local archive: `subtitles.archive` is a folder with a `manifest.jsonl` (season, episode, path, ...); it holds
   Survivor US only, so it is not asked for a video the `imdb` map gives to another show,
6. online (`subtitles.online`, `subtitles.imdb`): the OpenSubtitles Stremio addon, then Gestdown
   (`imdb` is one id, or a map `{"part of the path": id}`: the first part found in the video's path wins, `null` means no online search; the shared config maps Survivor and The Amazing Race, and Australian Survivor has `null`)
   (a few requests per episode, text only, a network failure is a warning and means "none").

The best file wins (named lines > turn marks > plain text; more named lines or marks first) and the
search stops at the first source that has named lines. The answer is cached in
`subs.source.json` (+ `subs.source.en.srt`) in the episode folder, "nothing found" included.

What it changes (`stt.transcribe --subtitles`, `stt.subtitle_hints`): every subtitle line is pinned to the
recognized words (the subtitles are ~0.3 s off, and a line the audio does not confirm is dropped).
Named lines fix their words to the person: a person who alone holds 80% of a cluster keeps that cluster, one
with at least 10 words that does not gets a speaker of their own (`P:NAME` in `diarizer_label`, gender
"unknown" until the naming step decides), the rest take their main cluster. Smoothing does not undo them.
Turn marks make a speaker change at the mark cheaper in the word-level pass (`subtitles.turn_bonus`).
The person goes to the naming step as evidence ("caption label: SANDRA x7", the host as "host per subtitles");
when the LLM leaves the name open, `speakers.json` gets the name from the subtitles. No extra LLM call.
`subtitle_hints.json` reports what was used (lines pinned and not confirmed, fixed words, persons).

Names in any case: a label is "NAME:" at the line start, also after "-" or ">>". ALL-CAPS names count as always;
"Jeff:", ">> jeff:" or "Jeff Probst:" count when the same file uses that name as a label at least 3 times (so
"Their destination: Tarutao Island," is a sentence) and it is not a generic word. Seasons 21-39 have broadcast
captions that name only the host (">> jeff:") and mark the others with ">>": these were read as turns before and now
give the host's lines as well (the turn marks still count). Over the hunt archive (742 episodes) the episodes with
named lines went from 438 to 707.

The subtitle base: `python -m stt.subtitle_base build --archive <hunt> --out <base> [--dry-run]` reads every file of
the archive with these rules, ranks them per episode (named lines > turn marks > plain, then more named lines, more
names, more marks, more cues) and writes the winner to `<base>/survivor/Sxx/SxxEyy.en.srt` with an `index.json`
(the facts of the winner, its original file and the next two candidates). Set `subtitles.base` to that folder.

Another cut of the episode: when the file came from the base, has named lines, and fewer than `subtitles.min_pinned`
(default 0.3) of them were pinned to the recognized words (`pinned_lines / (pinned_lines + not_confirmed)` in
`subtitle_hints.json`), it probably belongs to another cut of the episode. `stt.run` then tries the base's alternates
(only those with named lines or turn marks, at most 2) and assembles again each time (recognition and diarization
are cached). The first alternate that reaches `min_pinned` ends the search; if none does, the file with the best share
is kept and a warning is logged. `subs.source.json` then holds the file in use and a `rejected` list (path and pinned
share of every file tried and not kept), and the subtitle key follows it, so the next run does not repeat this.

A translated episode keeps the subtitles it was assembled with: when a better file turns up later (the base, a
season folder), only episodes without `subs.ru.json` are assembled again; `--force transcribe` takes the new file
for a translated one (and translates and voices it again).

Switching off: `"subtitles": {"enabled": false}` or `stt.run --no-subtitles`. Episodes without such subtitles
run exactly as before; transcripts made before this rule stay untouched until the transcribe stage is forced; if the subtitle source of an episode changes later, only the speaker assignment is redone
(cached models are not run again), and the stages after it follow as after any change of the transcript.

Numbers (S51 E01-E05 for named lines, two folds: half of the lines are the hint, half is scored; S20 for ">>"):
named lines are fixed directly (S51E01: 64.9% -> 84.2% of the labelled lines on their person's speaker, measured
on the lines that were given), but they barely help the lines not given: hidden lines 59.7% -> 59.8%. Turn marks
on hidden marks: recall 0.873 -> 0.894 at the bonus 0.3, precision the same.

## Timing of the Russian lines

A Russian line must not run ahead of the English speech. What keeps it in sync:

- Word times come from CTC forced alignment (see Models); cue starts match hand-synced
  subtitles within a few hundred ms (S10E01: median +0.19 s, 15 of 610 cues >1 s early).
- A pause of 1 s inside a sentence, or 0.8 s after one, starts a new cue, so the Russian of the
  second half is not said during the pause (S10E01 had 79 cues with such pauses). Except a pause
  of up to 4 s between words without punctuation that would cut off one or two words: voiced
  alone they sounded smeared ("Google | it." became "Проверь" ... "в Гугле" 3 s later; "Colby
  ready to make | a | move."), so the phrase stays one cue and is said at once (S20: ~9 per
  episode, `--fragment-words`, `--max-join`). A pause after a comma still splits ("First vote, |
  Sugar." keeps the host's pause before the name). The pauses are real: Whisper decoded no words
  in 92 of 99 such gaps where it had put the words side by side.
- The timeline solver prefers to start a line 250 ms after the English (voice-over style) and
  prices an early start 6x a late one (`policy.target_lag_ms`, `policy.weight_early`); base
  tempo 5.0 syllables/s, faster only where a line does not fit (`stt_config.json`).
- Continuation cues are not merged into one spoken unit for dubs (`units.enabled: false`): a
  merged unit placed at its first cue ran its later parts 1-4 s ahead of the speaker.

## Voices, stress and the speech check

- **The host's voice (since 29.10, `"host_voice"` in `voiceover`: `"eng_m3+eng_m4"`, picked by the user
  by ear from 8 blends).** In the
  Vosk and TeraTTS tracks the season cast's host (`role` "Host" in `<season>.cast.json`, else a
  speaker card whose role says host) speaks with this TeraTTS voice, so he does not sound like the
  male contestants; in the Vosk track his lines are TeraTTS clips (`teratts:<voice>`), as people
  without a clip are in the ESpeech track. The ESpeech track keeps his own cloned voice. Off with
  `null`; a change applies to episodes dubbed after it (`--force dub` for earlier ones).
- **Engines.** Vosk (CPU, 4 workers; voices 49 female / 56 male) or TeraTTSv2 (GPU, ~0.35 s per
  line; ru_f1+ru_f2 / ru_m5: the female voice is an equal blend of ru_f1 and ru_f2 since 28.10,
  picked by ear: ru_f1 alone sounded cartoonish; any built-in voice or blend `a+b` works). Every line gets the voice of its speaker's gender;
  continuation cues are never merged across voices. TeraTTS predicts only the total length of a
  phrase and stretches the speech to fill it (S10E01 scene: median 4.3 syllables/s, slowest
  2.1); the engine re-synthesises lines off the target tempo by more than 15% (median 5.4,
  slowest 4.1). The starting noise of its sampler is scaled to 0.667 (`temperature`, tip from the
  teratts-android port): on S29E01 first takes went from 1.51% to 0.74% character errors and from
  10 to 6 words not heard. Each TeraTTS clip is also transcribed right away (GigaAM); a clip with a word not
  heard, garbled or drawn out is synthesised again with other seeds (up to 3 takes) and the best
  one is kept. Vosk is not faster on the GPU, so it stays on the CPU.
- **ESpeech F5 (`--engine f5`, UI "ESpeech (голоса героев)").** Every line in its speaker's own
  voice, cloned from the voice bank's reference clips (`<out>/<season>.references/`, cut
  automatically for people who have none yet; a person keeps one clip over the season). Speakers
  without a clip get a fallback voice per gender: a Vosk sentence that F5 clones. F5 runs in
  `f5_env` (`bash scripts/setup_envs.sh f5`, reuses torch of subs_env) as `voiceover.f5_worker`;
  each line is synthesised at a natural length (5 syllables/s) and retried with other seeds when
  GigaAM misses or mishears a word. 32 sampling steps by default, 7 in the UI's fast mode
  (`voiceover.engines.f5.nfe`). Reference clips are 5-9 s (F5 cuts longer ones, and a cut
  clip no longer matches its transcript) with Whisper's transcript of the clip itself; when F5
  still speaks the end of the reference before the line, GigaAM's word times show it and the
  start is cut off. S29E03 (640 lines, 15 voices): 19 min; character errors 5.1% (12 s clips)
  -> 2.5% (5-9 s) -> 1.5% (with the cut), Vosk 0.6%, TeraTTS 0.5%.
- **Stress (before synthesis).** RUAccent is right for dictionary words; errors come from
  homographs and words it guesses. Stress is fixed in the text: from the audio it is heard only
  partly (TeraAnaly's stress head, given the text, on 159 phrases synthesised with a deliberately
  wrong stress: heard the wrong stress in 73% of Vosk and 62% of TeraTTS clips, reported a wrong
  stress on 1-2% of words said correctly), so `voiceover.stress_check` re-checks each guessed
  word and each homograph in its sentence:
  gpt-6-luna (high reasoning) proposes changes and three strong LLMs (gpt-6.1-sol,
  claude-sonnet-5.5, gemini-4-argon) vote on them; a majority correction is applied
  (`stress.json`: words, or single sentences for homographs). S29E01: 244 sentences checked,
  28 corrections (пора́, по́сле, О́строва, учи́тесь, начался́...), ~$0.14 per episode; the strong
  models on every sentence (`--no-screen`, ~$0.3) found 31. Cheap models alone are worse than
  RUAccent (luna: 36 extra changes, mostly wrong; deepseek-v4.1-flash: 75). Off with
  `"stress_check": false` in `stt_config.json`. Since 28.10 also checked, past the screener:
  the show's names in every case form (`--names`, the episode glossary: RUAccent read Jaclyn as
  Жакл+ин, Keith's "Кита" as кит+а; ~20% of name uses were wrong) and the sentences where a second
  stress placer, silero-stress, disagrees with RUAccent; homograph answers may leave RUAccent's
  variants (понял+а). With a ChatGPT subscription, gpt-6.1-sol as the screener
  (`llm.stages.stress_screen` in `stt_config.local.json`) costs nothing and matched the strong
  models on every sentence: S29E03 28 corrections (luna as the screener 24, everything 27; before
  28.10: 15) with 110 paid answers (luna 125, everything 325). The engines say the marked stress
  in 98.4-99.3% of words (TeraAnaly on S29E03), so the marks are what matters.
- **A stress mark on every word (fixed 29.10).** RUAccent got the cues ten at a time as one string
  (the list's repr, split at `[ ] , "`), the classic pipeline's way: it then read the pieces
  between commas and quotes apart, glued the first word of a cue to the repr's quote ("'Это") and
  marked almost nothing after an ellipsis, so 1.8-2.9% of the words with 2+ vowels reached the
  engines with no stress (S51E01: 143 of 4950, among them Эрик and Алексис at a cue's start, which
  the stress check had corrected) and each engine guessed. Now every cue goes alone
  (`normalization.accent_per_cue`, default true), the pieces between ellipses apart, and words
  RUAccent's stress-usage predictor leaves bare ("это", "или", "сразу", "слишком") get their
  dictionary stress (`normalization.fill_stress`; not homographs, not after a stressed
  one-syllable word: "н+е было"): S51E01 0.3%, S29E01 0.4%, S20E03 0.1% left, no slower. Episodes
  voiced before keep their takes until re-voiced (`--force stress`).
- **One stress per name for the season (since 29.10, `"season_stress": true`).** Names and guessed
  words that a majority of the models decided (a correction or RUAccent's stress confirmed) go to
  `<season>.stress.json` next to the season glossary, with the votes; later episodes take them from
  there without asking, so a name sounds the same in every episode. Words already in the file are
  never replaced, so changing a word's `"stressed"` there by hand fixes it for the whole season
  (the episodes voiced before need `--force stress`). Homographs stay per sentence. S29E01-E10
  checked one by one (sol + OpenCode Go voters, ~1 min per episode, free): 73 names and guessed
  words, 40 of them in several episodes, every one already stressed the same in all episodes (the
  names check with the English original is stable), so the file changes no stress there; it saves
  the ~180 later questions, holds if a vote wobbles and takes a season-wide hand fix. Fewer items
  change how the rest are batched: 2 of 86 homograph votes of S29E02 came out differently (LLM
  noise; one better: "за обе ст+ороны").
- **Whole-text proofreading (`"stress_proofread": true`, `llm.stages.stress_proofread`, default
  gpt-6.1-sol).** The checks above only look at words RUAccent guessed, at homographs, names and
  disputed words; a word taken from its dictionary was never asked about, even when wrong in its
  context. Australian S10E01, first 20 minutes, 269 lines: sol flagged 10 stresses, 8 of them real
  errors ("Само+а" for "Сам+оа" x5, "дом+а" for "д+ома", "кт+о-ниб+удь" x2) and 2 right future forms
  ("Подс+адим!"); luna got none right and made 4 harmful changes. So sol reads every line twice
  (`voiceover.stress_check --proofread MODEL --proofread-passes 2`; the second reading has its batch
  borders shifted, so it is not the cached first), its proposals are checked (the line and the word
  exist, the same letters, one "+") and go to the vote of the strong models like names (no
  screener), together with the other sentences of a proposed word; a correction needs a majority.
  A word that is no homograph and that all three models corrected the same way goes to
  `dictionary` and to the season file (`"source": "proofread"`), so it holds in all episodes; with
  two of three it is fixed in its sentences only. Season words are put on the text before the
  proofreader reads it. Each sentence is voted against the stress the word has in that sentence
  (before 3.11 against its first sentence's, so "Дом+а, в моём царстве" after an earlier "я д+ома"
  was never fixed). Whole S10E01: 10 more fixes than the old check, 9 right, 1 wrong by two votes
  of three ("в+ыигрывает"); 2.7 min per episode. A rule also removes the stress
  from "-нибудь", "-либо", "-то", "-ка", "-таки" and "кое-", "кой-". Finished episodes keep their
  `stress.json`: the stage reruns only when the lines change.
- **Stress heard in the takes (since 28.10, `"listen_stress": true` in `voiceover`).** The words
  the stress check looked at (`stress.json`: names, homographs, guessed and disputed words) are
  heard in every take by TeraAnaly (`voiceover.stress_listen`); a take whose voice said another
  stress is retried like a garbled one (ESpeech and TeraTTS with other seeds, Vosk once more: its
  sampling noise differs on every run) and the best take is kept. Cached takes are heard once and
  redone only when wrong. S29E03: 1-7 of ~260 such lines per engine heard wrong (11 in the three
  engines), 4 of them fixed by another take; TeraAnaly notices 37-67% of wrong stresses at 1-2% false alarms
  (`docs/roadmap.md` §6). Off in the UI settings or with `"listen_stress": false`.
- **Speech check (after synthesis).** `voiceover.speech_check` transcribes every clip with
  GigaAM-v3 and lists clips whose character error rate is over 15% (mispronounced
  abbreviations, swallowed words), drawn-out words and too slow/fast lines in `speech_check.md`.
  ESpeech swallows short words (и, в, да) more than the other engines; since 28.10 its takes start
  with "— ", get 0.3 s more length and are synthesised again while a word of the text, short ones
  included, is not heard (lines losing a word 40% -> 19% on 300 S29 lines; `docs/roadmap.md` §7).
  People without a voice-bank clip (unnamed voices, crowds) are spoken by TeraTTS in the ESpeech
  track (`"fallback": "teratts"` in the f5 engine: the TeraTTS voices of the config, ru_f1+ru_f2 /
  ru_m5) instead of a Vosk sentence cloned by F5; the F5 calibration still probes the Vosk clip.
  S29E06 voiced so (with DiariZen and the short-word fix): character errors 1.65% -> 1.03%, lines
  over the flag 12 -> 5, lost first words 62 -> 37 of ~600 lines.
  **Stand-in pool.** A season person (named, or a voice identity `V03`) without a clip gets a stand-in voice of
  its own instead of the one generic voice of the gender: `voiceover.engines.f5.standin_pool` is
  `{"female": [...], "male": [...]}` with entries `teratts:<voice>`, `vosk:<id>` (the Vosk fallback sentence cloned by
  F5) or a path to a WAV clip (transcript in the `.txt` beside it). The voice with the fewest people so far is taken
  (ties: pool order) and kept in `<season>.references/standins.json`, so a person sounds the same in every episode
  of the season; a person who later gets a clip switches to it. No pool: the generic voice, as before. Clusters
  with neither name nor identity stay on the generic voice.
  **Stitched clips.** When no single 5-9 s solo stretch of a person passes the checks, the export tries one clip
  stitched from several shorter solo cues (at least two, 1.5 s each or longer, 0.4 s of silence between them) with
  the same checks; the index entry has `"stitched": true` and `"pieces"`. `stt.voice_bank --export ... --no-stitch`
  turns this off.

## Season voice bank

Episodes of one folder share `<season dir>/<folder>.voices.json` (`stt.voice_bank`, on with
`"voice_bank": {"enabled": true}`). Every named speaker cluster with 20+ s of speech becomes a
voice print (its pyannote/WeSpeaker embedding with name, Russian label and gender). In the next
episode, before the LLM names the speakers:

- each cluster gets the closest names of the other episodes (similarity 0.6+) as `voice_bank`
  evidence in the naming prompt; the LLM still decides, since a print can carry an earlier
  naming mistake and relatives sound alike (S29 twins Nadiya/Natalie: 0.67);
- a cluster the LLM leaves unnamed takes the name of a strong (0.75+) and clear (0.1 ahead of
  the next name) match;
- a strong match the LLM contradicts is a conflict, logged and kept in the bank. Spellings of one
  name ("Jon"/"John Misch", "Baylor"/"Baylor Wilson") count as one person.

S29E02 with the E01 bank: the voice luna had named Missy Payne became Julie McGee (right: in
E01 she talks about her boyfriend John Rocker); the one conflict left is real (E01's "Jaclyn"
cluster holds Baylor's voice). Same person across episodes: similarity 0.67-0.98; different
people mostly under 0.5. Per person the bank also lists the longest solo stretches (6-12 s, no
overlap, no burned-in caption) as reference clips for voice cloning.

`subs_env/bin/python -m stt.voice_bank output/stt/<folder>.voices.json` shows people, episodes,
the best sample and the conflicts; `--export output/voices/<season>` writes one reference clip
per person for voice cloning. Candidates are the person's solo stretches (6-12 s) in every
episode; a stretch with more than 0.15 s of another diarized speaker is dropped; the ones whose
WeSpeaker embedding is closest to the person's prints go through vocal separation (Hybrid
Demucs from torchaudio); every 3 s window of the vocals must still sound like the person
(0.45+), which drops stretches with a second voice. The clip is the separated vocals, 24 kHz
mono, with 1 s of silence at the end and its transcript by Whisper (`<Name>.wav`,
`<Name>.txt`, `references.json`). S29E01-E04: 19 people; the music bed is 6-16 dB under the
voice in every candidate (reality TV), and 22-46 dB under it in the separated clips.

The first episode of a season is named without a bank, so its mistakes surface later:
`--recheck` lists prints whose voice at least two other episodes know under another name (and
more than under its own, similarity 0.85+, never between relatives with one surname: the S29
twins Nadiya and Natalie Anderson match at 0.80), `--write-overrides` puts the rename into that episode's
`speakers.override.json`; `stt.run <video> --force speakers` then redoes its naming and the
stages after it. Per-line prints show what such a cluster holds: S29E01's "Jaclyn" cluster
matches E02's Baylor at 0.72-0.86 in 10 of its 14 longer lines, Jaclyn's own line only 0.45.

### Voice clips by hand

When the automatic export finds no clean clip for a person, `<refs>/missing.json` says why (per
person: how many solo stretches there were, how many dropped for other voices inside, for a voice
too unlike the prints, or for a second voice/noise left after cleaning; the export log prints it
as one Russian sentence). `stt.voice_bank.clip_states(bank)` gives a table row per person
(`auto`/`manual`/`missing` plus the reason).

A clip can be made by hand: watch the episode, find where the person speaks alone (an interview)
and pass the time ranges (`12:30-12:41`, `1:02:05-1:02:14`, seconds also work; each at most 60 s):

```
subs_env/bin/python -m stt.voice_bank "<season>.voices.json" --manual-clip --person "Saiounia Hughley" \
    --media <episode>.mkv --range 12:30-12:41 --range 1:02:05-1:02:14 [--text "..."] [--gender female] \
    [--preview clip.wav]
subs_env/bin/python -m stt.voice_bank "<season>.voices.json" --remove-manual --person "Saiounia Hughley"
```

Each range is cut from the video, cleaned by the same vocal separator as the automatic clips,
trimmed of silence and joined with 0.3 s pauses (at most 10 s of speech, 6-10 s is best; the last
piece is cut at a pause); the end is faded and gets 1 s of silence; Whisper writes the transcript.
Warnings (too short, loud music, a voice unlike the person's prints) are shown, never blocking.
`--preview OUT.wav` only writes that clip and prints the text and warnings; nothing in the season
changes. Without it the clip is saved as `<refs>/<Name>.manual.wav` + `.txt`, an entry in
`<refs>/manual.json` (`"source": "manual"`, `episode`, `ranges`) and in `references.json`. Exports
(full, `--missing`, `--person`, `--reject`) never replace or drop a manual clip; `--remove-manual`
brings back the automatic clip it replaced (or leaves the person to the next `--missing` export).
`stt.run` sees the changed `references.json`, runs `dub` again and voices only that person's lines
(the F5 voice id hashes the clip and its text). A person outside the bank needs `--gender`.

### Cast of the season from the web

Once per season (before the first glossary) a cheap LLM with web search,
`openrouter:openai/gpt-6-luna:online` (`llm.stages.cast`; if it fails, refuses or finds fewer
than 3 people, the next model of the list: `anthropic/claude-haiku-5.5:online`, same price,
S29 and S10 as complete), reads the show and season off the folder and file names and writes `<season dir>/<folder>.cast.json`: every person who speaks, with
spellings, nicknames, gender, role and relations (twins, couples, parent and child). The
glossary uses its spellings, the speaker naming gets it as evidence, and after naming
`stt.cast.check_speakers` turns other spellings of a cast name into the cast's ("John Misch" ->
"Jon Misch"), and logs names not in the cast and genders that contradict it (`cast_check` in
`speakers.json`). The gender is never changed from the cast: the web is wrong too (S10 Stevie
Khouw came back "female" from luna, S29 Reed Kelly "female" from Haiku; the voice and the
transcript said male, correctly).

Measured: 15-20 s and $0.01-0.03 per season; S29: 19 people, S10: 26. Names on the S10 SDH
speaker labels (E01, E12, E16) were as right with the cast as without it (gpt-6.1-sol: 65.5% of
labelled lines both ways; gpt-6-luna: 63.9-65.9% both ways over three runs, the spread is one
cluster of E01). The S29E01-E04 names were already right. What it does give: one spelling per
person across the season, a name for a small cluster the LLM left unnamed (S10E01, 11 lines),
and a list to check names against for seasons the LLM knows less well. Off with
`"cast": {"enabled": false}`; a failed lookup only logs a warning. Run by hand:
`subs_env/bin/python -m stt.cast output/stt/<episode> --cast output/stt/<folder>.cast.json`.

## Season voices (on by default since 2.11.2026)

A cluster of the diarizer is often a mix of people, and the word-level speaker pass compares each word only with the
clusters of its own episode. This stage uses the people the pipeline has already named over the whole season instead.
No subtitles required and no LLM. It works with the stock model and better with the fine-tuned `resnet293-ft` (weights in `models/speaker/`, not in git; `season_voices.model`); another model than the first pass needs its word embeddings once (GPU, 30-45 s per episode), the stock one needs no GPU.

What it does, after every episode has its transcript and names (`speakers` stage):

1. Per person (the speaker `name`; spellings of one name are one person, as in the voice bank) the embeddings of the
   words under that name are summed over the season, OCR captions left out. Words of named subtitle lines
   (`subs.source.en.srt`) count for the person too (matched to a name by first name). A person needs
   `min_words` (20) words. The result is `<season>.season_voices.npz` beside the voice bank; it is reused while the
   episodes' first-pass files stay the same.
2. Every word of every episode is assigned again by the Viterbi pass of `stt.boundaries` over the season's people plus
   the episode's clusters that carry no name. A cluster change of the diarizer is a cheap switch (switch penalty 0.3,
   turn bonus 0.15); words of named subtitle lines stay with their person; on-screen captions keep their speaker.
   Then the same smoothing as in the first pass.
3. `words.json`, `subs.en.json`, `subs.en.srt`, `subs.en.speakers.srt` and `speakers.json` are rewritten with one
   speaker per person (`diarizer_label` is `V:<name>`, `diarizer_labels` lists the clusters the person came from; gender,
   role and Russian label come from the old card of that name, or from the card of another episode), and
   `season_voices.json` records what moved. The first-pass files are kept once as `*.before_voices.json`; the stage always
   starts from them, so a rerun (also with more episodes in the season) gives the same result as a first run.

Since 7.11.2026 (old seasons, `docs/roadmap.md` plan of 7.11): a person only the subtitles or the season's cast name
(S01E01 "DIRK:" lines) gets a voice too; a person with fewer than `min_words` words gets one from the words spoken under
their name cards (`ocr.json` and the multi-line cards of `onscreen.json`; title rolls and tribe / day / place cards do not
count); a cluster the naming calls mixed ("merged contestant voices") gives no words to the voice of the person it is
named after. Options in `season_voices` (numbers in `docs/experiments.md`): `dissolve` (on, "mixed" since 7.11, approved by
ear: 
the unnamed clusters the naming calls mixed are no state, their words go to the season's people; "unnamed": every
unnamed cluster; "none": off), and off by default: `voice_source` ("clean": voices from card and named-line words only), `tribe_prior` (people of other
tribes less likely at a tribe's camp, `stt.tribes`), `caption_bonus` (the switch bonus at ">>" marks, 0.15).

Voice identities: a cluster without a name, with at least 20 s of speech and not a mix of voices, gets a season identity
(`V01`, `V02`, ...) in the naming stage. The voice bank keeps its print (`"name": null, "identity": "V03"`,
`"source": "identity"`) and a later episode with the same voice (strong, clear match) gets the same id; the speaker card
carries `"identity": "V03"` (and `identity_evidence`) while `name` stays null. This stage treats an identity like a
named person: it gets a season voice (`diarizer_label` `V:V03`), and its words are assigned over the season. A name
the pipeline finds later stays the primary key, and nothing changes for named people. Switch off with
`"voice_bank": {"identities": false}` in `stt_config.json` (`stt.speakers --no-identities`).

**Show profiles.** The base config keeps the Survivor behaviour: `voice_bank.identities` and `voice_bank.stitch` are
off and `voiceover.engines.f5` has no `standin_pool`. `shows` holds per-show settings that are merged on top when the
video's path contains the key's words (the same matching as `subtitles.imdb`; `--show <name>` forces a profile,
`--show none` turns profiles off). The `amazing race` profile turns on voice identities, stitched clips and the
stand-in pool. A profile may also carry `paths` (e.g. `paths.finished` for the show): put it in
`stt_config.local.json` under `shows.<key>.settings`.

`speakers.llm.json` and the naming are not touched. The stage touches the `speakers` mark of an episode it changed, so the
stages after it (translation, dubbing, ...) run again and the earlier ones do not. If `corrections.json` or
`speakers.override.json` exists, the stage warns: line numbers and speaker ids of the first pass no longer apply.
Lines of speakers whose gender stays unknown keep the gender `stt.line_gender` heard for the old line at that time.

It is on in `stt_config.json` (`"season_voices": {"enabled": true, "model": "resnet293-ft", "min_words": 20}`); `--no-season-voices` or
the presets `fast` and `previous` switch it off for a run. Without the `resnet293-ft` weights (not in git; on Hugging Face, `AngryBear9019/wespeaker-resnet293-survivor-ft`, downloaded once by the tick in the UI or `stt.run --download-voice-model`) the stage logs a warning and uses the plain `resnet293`. `stt.run --season-voices-only` (UI button "Уточнить, кто говорит" on the "Голоса сезона" tab) only counts the voices again from the episodes that have a transcript.
An episode that already has `subs.ru.json` is left alone, so a re-run never repeats the translation and the voice-over; `--force speakers` assigns it again. With several videos, `stt.run` first takes every episode up to `speakers`, applies the stage, then runs the
other stages. With one video of a season it uses the voices of whatever episodes of that folder already have a first
pass. Without named people nothing changes. By hand:

```bash
subs_env/bin/python -m stt.season_voices output/stt [--episodes "*S51E0[23]*"] [--dry-run]
```

Measured offline on the finished runs (no listening yet): S51E01-E05 words under SDH labels: the right person 63.7% ->
69.0%, wrong name 29.5% -> 31.0%, no name 6.8% -> 0%; lines under 1.5 s 56.6% -> 64.3%. S20 E01/E05/E10/E14 against the CBS `>>`
marks: F1 0.879 -> 0.889 (3 of 4 episodes better). Through the real assembly with `resnet293-ft`: S51 lines on their person
68.4% -> 71.6% (stock 70.9%), S20 F1 0.879 -> 0.897 (stock 0.891), S29 E01/E03/E07 0.860 -> 0.871 (stock 0.860). It became the default on 2.11.2026 after the user listened to season-voiced episodes and approved them. Its limit: the stage gives words to named people who may not be in the episode at all (a wrong name instead of a
neutral voice is worse than no name), and a cluster that was named after its main speaker feeds the others' voices into
that person.

### Preparing a season

`stt.run ... --prepare` (UI: "Подготовить весь сезон" on the "Обработка" tab, which takes all videos of the season folder) is
the same as `--until speakers` and conflicts with another `--until`. Every episode gets its transcript, on-screen text,
speakers and names, and the season voices are counted over the whole season at once (about 8-10 minutes per episode, GPU;
the glossary and naming use the LLM). Episodes that are already translated stay as they are. At the end the command prints,
per source folder, how many episodes have `speakers.json`, which are translated, and the unnamed speaker groups per episode
(`S48E02: unnamed S02 41 lines`). Then optionally name them in the UI tab "Кто это?" (`speakers.override.json`; the speakers
stage reruns by itself) and voice each episode with the normal command: it only translates and voices, about 15 minutes faster
per episode. In the UI table such an episode shows "подготовлена (ждёт перевода)".

## Checking by ear and fixing lines

After the speech check, `stt.review` lists voiced lines that may have the wrong voice: the
age/gender model hears the other gender with 0.9+ on the line's voice separated from music (Hybrid
Demucs) while the line does not clearly sound like its own speaker (print similarity under 0.35),
or the line's voice print is clearly (0.15+) closer to another speaker than to its own. On
S29E01-E04 this brought the list from 100 to 65 lines (9-26 per episode) and the host's
narration and shouts "heard as female" from 30 to 6. Shouts and crowd noise still fool the
gender model, so this is a list for a person, not an automatic fix.

- `review.json`: line number (as in `subs.ru.srt`), time, speaker, voice used, what was heard,
  English and Russian text, and a ready `fix`; lines whose gender `stt.line_gender` changed by voice are
  always here, with a Russian `note`;
- `review.srt`: the same lines, for Subtitle Edit next to the original video (or open
  `subs.ru.srt` and jump to the numbers).

Fixes go into `corrections.json` in the episode folder, keyed by line number:

```json
{"123": {"gender": "female"}, "130": {"speaker": "Julie McGee"}, "140": {"text": "Новый текст"},
 "141": {"skip": true}}
```

`speaker` takes a name, Russian label or id from `speakers.json` (and its gender). The next
`stt.run` of the episode sees the newer file and redoes the dubbing script and everything after
it (for `--engine teratts` too); `speakers.override.json` works the same way for the speakers
stage.

## Simultaneous speech

1. ASR yields one stream of words: the dominant voice is transcribed, quieter voices under it
   mostly are not; cues are built from that stream, so they follow each other (S10E01: 2 of
   1180 cues overlap in time). `overlap: true` marks cues where two diarized speakers talk at the
   same time for 30%+ of the cue (3-6%; until 30.10 a cue whose speaker disagreed with the
   diarizer's turn under it counted too, ~9%, which made the dub drop shouts the word-level pass
   had just put right).
2. Speaker labels flicker at turn edges; after the word-level Viterbi pass (Models) short sentences
   get one (majority) speaker and other speaker changes are moved to sentence boundaries (cues
   ending mid-sentence 31% → 12%).
3. `stt.dub` drops interjections/backchannels ("Yeah", "Wow", "Go, go!") that sit on another
   speaker or inside a rapid exchange (S10E01: 36 of 1180): they stay audible in the original.
4. The voice-over timeline solver never lets two Russian lines collide: it shifts lines within
   −360/+500 ms, speeds them up to a comfort limit and asks for shorter text only when needed.
5. The original track stays under the voice. By default (`"mix": {"duck": "voices"}`) Hybrid
   Demucs (`stt.separate`) splits it into voices and everything else, and only the voices go
   12 dB down while the Russian voice speaks; music and effects only 6 dB (`rest_duck_db`), so
   the scene keeps its sound (43-min episode: 40 s on the GPU). `"duck": "all"` lowers the whole
   original by 12 dB as before. `"subtitles": false` leaves the Russian subtitles out of the mkv.
   **Voice level.** The voice-over is raised by `voice_gain_db` (3), and the finished mix gets one static gain to
   `loudness_lufs` (-16 LUFS integrated, measured with ffmpeg `ebur128`; `null` = off), so seasons that differ in
   loudness (S51 sat about 1 dB below S20) end up equal. Peaks are caught by a 5 ms look-ahead limiter at 0.97
   (`stt.mix.limit_peaks`); the old scaling of the whole track by its highest peak is gone, so a louder voice no
   longer makes the music quieter. The original audio track in the video is untouched. Hear a 45-second piece with
   other levels before re-mixing: UI tab "Звук и размер видео" or
   `subs_env/bin/python -m stt.mix_preview <episode folder> <video> --engine f5 --voice-gain-db 6 --out piece.wav`
   (without `--start` it takes the place where speech meets the loudest original sound, `loudest_spot`). To apply new
   levels to a finished episode without voicing again: `stt.run <video> --engine f5,teratts,vosk --force mix`
   (stages before `mix` that finished earlier are skipped; only the mix and the video are rebuilt).
   **Old seasons' band** (since 8.11.2026). ESpeech copies the acoustics of its reference, so the clones of a season recorded
   with a narrow band (S1: AAC 48 kbit/s, nothing above ~5.2 kHz) end there too. When the median band edge of the episode's
   reference clips (`speaker_voices.f5.json`) is under `mix.widen.max_band_hz` (7600), the mix takes
   `voiceover.f5/export/track.wide.wav`: the voice track through LavaSR v2 (`stt.widen`; the band above the edge drawn,
   48 kHz, no denoiser; a 45-min track in ~25 s on the CPU), made once and reused. It needs `lava_env`
   (`bash scripts/setup_envs.sh lava`; without it the track is mixed as before, with a warning). Blind listening: every
   one of 12 S01 picks was a widened version; S48 (9.7-11.6 kHz references) is left alone. Off: `"widen": {"enabled": false}`.
   **Smaller video.** `"mix": {"video": {"compress": true, "target_gb": 2.0, "codec": "hevc_nvenc",
   "original_audio_kbps": 160}}` (off by default) re-encodes the picture with the same resolution and the source
   timestamps (PyAV; NVENC on the graphics card, libx265/libx264 if it fails to open) to the bitrate that makes the
   finished file `target_gb` (1 GB = 10^9 bytes, all voice-over tracks at 192 kbps counted); the original audio becomes
   stereo AAC. The video is made once per episode (`video.<codec>.<kbps>k.mkv` in the work folder) and shared by the
   engines' videos; it is left as it is when the budget is not 15% under its own bitrate. An already finished video:
   `subs_env/bin/python -m stt.video_compress <video.ru.mkv> --target-gb 2` writes `<stem>.small.mkv` next to it (all
   audio and subtitle streams kept, AAC copied, others to stereo AAC 160k; the input is never changed);
   `--sample-minute 10 --sample-seconds 60` writes a `.sample.mkv` piece to judge the quality.
6. Experimental crowd layer (`"crowd": {"enabled": true}`, `stt.crowd`): skipped interjections of
   two or more speakers within 2 s, or chants ("Go, go, go!"), are said by three TTS voices at
   once with 70 ms offsets, 4 dB under the voice-over, and added to the mix without ducking the
   original. One person's "Oh my God!" stays in the original only.

`dub.ru.<engine>.json` carries a `voice` per line (by speaker gender); the voice-over uses it.

## Environments

| venv | Python / torch | Used for |
|---|---|---|
| `subs_env` | 3.12 / 2.11 cu128 | everything: orchestration, Whisper, Cohere, pyannote 4, gender/age, OCR, RUAccent/RUNorm, Vosk, TeraTTS, GigaAM, mixing, Gradio |
| `stt_qwen_env` (optional) | 3.12 / 2.11 cu128 | benchmarks with Qwen3-ASR / Qwen3-ForcedAligner (`qwen-asr` pins transformers 4.57) |

Model stages run as subprocesses in their venv automatically. `onnxruntime-gpu` is the only
onnxruntime (`overrides-onnxruntime.txt` drops the CPU wheel that vosk-tts/ruaccent ask for). No system ffmpeg is needed
(PyAV decodes; `imageio-ffmpeg` provides ffmpeg for muxing).

## Models

ASR (`stt/asr.py`): `whisper-large-v3` (fast mode), `whisper-large-v3-turbo` (half the VRAM),
`cohere-transcribe` fused with `whisper-large-v3` by an LLM (accurate mode). `qwen3-asr-1.7b` stays
available as a third fusion input but adds nothing measurable (whole episodes: S10E01 13.37% with
it vs 13.65% without, S29E01 18.26% vs 18.17%, within the run-to-run spread of the LLM).
Word times of the fused text: character CTC forced alignment with the English wav2vec2
`WAV2VEC2_ASR_LARGE_LV60K_960H` (11 s per hour of audio). Qwen3-ForcedAligner (`--aligner qwen`)
gave 7-11% zero-length words on these shows; with CTC, cues starting >1 s before the speech in
S10E01 (against hand-synced subtitles) went from 25 to 15.
Speakers whose gender stays unknown after the naming (mostly mixed clusters) get it per line from
their lines' own voices (`stt.line_gender`, after the naming, before the translation; S10 SDH lines:
right gender 78.2% -> 84.8%; `"line_gender": {"enabled": false}` (or `false`) turns it off).
For speakers of known gender the same stage can change a single line's gender (never the speaker): the
average of audeering and a logistic regression on the resnet293-ft voice print of the whole line
(`models/gender/resnet293-ft-gender.npz`, trained by `scripts/speaker_training/train_gender.py`) must be
sure of the other gender (p >= 0.9; `"line_gender": {"override": true, "voice_override": 0.9}`). The host
and caption-only lines are skipped. On 899 voiced S51 contestant lines it changed 26 (about 3%): 22 fixed,
4 broken (right gender 86.2% -> 88.2%); audeering alone fixed 37 and broke 36. A changed cue carries
`gender_by_voice`, is voiced by the voice of the new gender (not the speaker's own), and every such cue is
listed in `review.json` / `review.srt` with a note and a `fix` that undoes it. A gender in `corrections.json`
wins. Missing model files: a warning, the override is skipped.
Diarization: `diarizen-v2` (default since 28.10; `pyannote-community-1` when `diarizen_env` is missing; more speaker changes found: F1 vs the captions' turn marks 0.73 -> 0.83, who is who the same, see `docs/roadmap.md` §6; its own venv: `bash scripts/setup_envs.sh diarizen`), then speaker refinement (`stt.refine_speakers`,
`"speaker_refine"` in `stt_config.json`): every diarized turn gets an embedding of the stronger
WeSpeaker ResNet293 (pyannote's PyTorch port of the checkpoint, ~17 s per episode), every speaker a
centroid of its turns of 1.5+ s, and a turn that sounds clearly (0.1) closer to another speaker
moves there before the words get their speakers. Measured against SDH speaker labels of
Australian Survivor S10 (mostly off-screen or unclear voices): voiced cues on their person's main
speaker 63.5% -> 71.1% on E01/E12/E16 (249 lines), diarization turns 63.6% -> 68.8% on all 24
episodes (1300 lines); 10-28 cues per episode change the voice gender. Splitting speakers into
sub-voices and the CAM++ model did worse; ReDimNet2-B6 (`"model": "redimnet2-b6"` or `"redimnet2-b6-vb2"`,
29.10) was better on S51 (886 lines, 62.0% -> 63.1%, fewer people merged) and worse on S10 (334
lines, 61.1% -> 60.5%): not a gain that holds, ResNet293 stays. An LLM check of the speaker of each line (gpt-6.1-sol reading the
conversation with named speakers, fixing lines like "Get him, George!" on George) proposed 22-33
fixes per episode but changed nothing on the SDH-labelled lines, and the line's own voice was
closer to the new speaker in only 26 of 62 fixes it could judge: not used. Gender/age: `audeering/wav2vec2-large-robust-24-ft-age-gender`
(CC-BY-NC-SA-4.0, abstains below 0.7).

`"diarizer": "diarizen-local"` runs a DiariZen checkpoint from the directory in `$DIARIZEN_LOCAL_DIR` (the layout of the v2 snapshot: `config.toml`, `pytorch_model.bin`, `plda/`; since 31.10, for fine-tuned models; a different directory invalidates the cached turns). Since 30.10 DiariZen also returns an exclusive layer (`"diarizer_options": {"exclusive": true}`): one
speaker per 20 ms frame, the strongest of the simultaneously active ones by the segmentation model's
soft scores, which the words are labelled by (pyannote's pipeline gives such a layer itself). Then
the words get a second look (`stt.boundaries`, `"word_speakers"` in `stt_config.json`): every word an
embedding of a 1 s window around it (ResNet293 in batches, 30-45 s per episode, cached in
`words.emb.<model>.npz`), every speaker a centroid of its words far from any speaker change, and a
Viterbi pass picks the label sequence: cosine to the centroids, a penalty of 0.3 for changing speaker
between consecutive words (fading with the pause between them) and a bonus of 0.15 for a change where
the diarizer has one. This splits turns the diarizer merged (a shout inside the host's speech) and
moves boundaries by a word. Measured on S20 E01/E05/E10/E14 against the CBS captions' `>>` turn
marks (`stt.turn_eval`) and on the S51E01-E05 SDH speaker labels (`stt.sdh_eval --cues`, 886 lines):

| Speakers chain | S20 speaker changes: recall / precision / F1 | S51 labelled lines on their person / people merged |
|---|---|---|
| DiariZen + ResNet293 refinement (29.10) | 0.811 / 0.874 / 0.841 | 62.0% / 44 of 103 |
| + exclusive layer | 0.828 / 0.885 / 0.856 | 61.7% / 41 |
| + only filling the words with no diarizer label (`--word-prior 9`) | 0.828 / 0.900 / 0.863 | 62.9% / 41 |
| + word-level Viterbi (the default since 30.10) | 0.873 / 0.884 / 0.878 | 63.5% / 39 |

Better on each of the 9 episodes; on S29E01/E03/E07, not used for tuning, F1 0.824 -> 0.840 -> 0.861.
The exclusive layer adds no time (one segmentation pass gives the hard and the soft output), the word
embeddings 30-45 s per episode. The median filter DiariZen applies to its segmentation (11 frames,
`"median_filter": false` drops it) changed nothing (F1 0.843 / 0.856, S51 62.1% / 61.4%). The Viterbi
without the change bonus removed as many real changes as it added (a one-word "yep" sounds like its
neighbour in a 1 s window); a bonus for keeping the diarizer's own label of a word (0.15) costs recall
(S20 F1 0.873, S51 63.1%). DiariZen's AHC threshold 0.4 (S20 F1 0.868, S51 64.6%) and voice bank people
as Viterbi states (no change), voting with other diarizers and a two-voice separator were not taken:
`docs/experiments.md`. The sweep's scripts and logs (not in git, its data was deleted): `output/exp_boundary/` (`prepare.py`,
`run.sh`, `sweep_words.py`, `score.py`).

Benchmarked and dropped (summary below): Parakeet v2/v3/Unified/TDT 1.1B, Canary 1B v2/180M/Qwen
2.5B, Granite Speech 4.1 (+plus), Voxtral Mini 3B/Realtime 4B, Kyutai STT 2.6B, Moonshine v2,
Whisper medium.en, Distil-Whisper large-v3.5, Qwen3-ASR 0.6B, DiariZen, TeraAnaly.

## Benchmarks

Past experiments, what came of them and which outputs were deleted: `docs/experiments.md`.

```bash
# ASR vs English subtitles (clip snapped to subtitle gaps, SDH markup removed)
subs_env/bin/python -m stt.bench asr --media ep.mkv --srt ep.en.srt --start 600 --duration 600 --out output/stt_bench/ep_10m
subs_env/bin/python -m stt.bench summary output/stt_bench/ep_10m ...
# diarization: VoxConverse DER, or SDH speaker labels of a real episode
subs_env/bin/python -m stt.bench_diar --parquet voxconverse-test.parquet --out output/stt_bench/diar
subs_env/bin/python -m stt.sdh_eval --srt ep.en.srt output/stt/ep/diarization.*.json
# speaker changes vs the '>>' turn marks of CBS captions (S20, S29 WEB-DL; roll-up lines de-duplicated)
subs_env/bin/python -m stt.turn_eval --srt ep.en.srt output/stt/ep/words.json
# translations side by side
subs_env/bin/python -m stt.compare_translations output/stt/ep --variants a b --ass ep.ru.ass
```

Why these models (Australian Survivor S10, three 10-minute clips, WER against English SDH
subtitles): LLM fusion of Cohere + Whisper (+ Qwen3) ~17%, Cohere 19.6%, Whisper large-v3 20.2%,
Qwen3-ASR 20.5%, the other 15 models 21–27% (Canary 180M loops). pyannote vs DiariZen on a real
episode: equal speaker purity, pyannote 2.5x faster. Full tables: `docs/stt_benchmarks.md`.

## Outputs (`<paths.work>/<episode>/`, default `output/stt/<episode>/`)

- `subs.en.json`, `subs.ru.json` — project subtitle schema (`index`, `start`, `end`, `duration`
  in ms, `text: list[str]`, `gender`) plus `speaker`, `speaker_name`, `overlap`; the Russian
  file adds `text_en`, `budget_chars`, `window_ms`. `.srt` copies; `subs.en.speakers.srt` shows
  `[S03] text` for review.
- `dub.ru.<engine>.json` / `dub.report.json` — voiced lines with `voice`, skipped lines with reasons.
- `stress.json` — checked words, the votes of each LLM, applied corrections.
- `speakers.json` — ID, talk time, voice gender/confidence, age, LLM name/label/role/evidence,
  `gender_source` (`voice`, `voice+text`, `text`, `manual`), speaker embedding.
- `glossary.json/.txt`, `hotwords.txt`; the season glossary sits next to the episode folders.
- `voiceover.<engine>/` (voice-over run: `export/track.wav`, `report.md`, `speech_check.md`),
  `<name>.ru.<engine>.m4a` (the engine's mix), `<name>.ru.mkv` (all voice-overs; in
  `paths.finished/<season folder>/` when set) or `<name>.ru.<engine>.mkv` (`"tracks": "separate"`).
- `ocr.json` — captions, name cards and all text events with times and boxes.
- `subs.source.en.srt`, `subs.source.json`, `subtitle_hints.json` — existing subtitles used for speakers (see above).
- `words.json`, `diarization.*.json`, `asr.*.json`, `voice.*.json`, `timings.json`, `logs/`.
- With the season voices stage: `season_voices.json`, `*.before_voices.json` (the first pass), and `<season>.season_voices.npz` beside the voice bank.

Manual fixes: `speakers.override.json` (`{"S04": {"gender": "female", "label_ru": "Анна"}}`),
then `stt.run ... --force speakers`; edit `glossary.txt` and use `--force translate`.

TTS cache (`output/tts_cache/<engine set>/<key>.wav`, shared by all episodes): a re-run takes
every unchanged line from it. `subs_env/bin/python -m voiceover.cache_clean` reports and with
`--delete` removes the takes no episode folder names (in `voiceover.<engine>/*.json`) and that
are older than 3 days (`--keep-days`); the UI settings tab has the same as two buttons. After a
whole season is re-voiced, the old takes become unused and go with the next cleanup.

## Tests

```bash
subs_env/bin/python -m unittest tests.test_stt_local -v
```
