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
# one episode end to end (voice-over, checks, mix); --engine teratts for TeraTTS voices:
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
"Сбросить" removes the page's keys. The "Обработка" form starts from these values.

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

A run can be stopped at any time (Ctrl+C, or "Стоп" in the UI) and started again with the same
command: finished episodes and stages are skipped, the recognition keeps its finished parts
(each ASR model, alignment, OCR, diarization), and the voice-over keeps every line already
synthesised (Vosk, TeraTTS and ESpeech write each line to the cache as it is done). An LLM step
that was cut off (glossary, naming, translation, stress check) starts over; they take minutes.

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
`voiceover`, `stress`, `stress_screen`). By default only the stress re-check uses other models.

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
  and stays only in the original audio. The cues are then translated and voiced as usual.
  S10E01: 61 captions — 32 already right in ASR, 17 replaced, 11 added, 1 under other speech.
- Name cards, titles and caption labels ("Shonee: …") become `on_screen` evidence for the
  speaker that talks most while they are shown; the speaker-naming LLM sees it.
- `--no-ocr` (or `"ocr": {"enabled": false}`) skips the stage; `--force ocr` rescans.

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

## Checking by ear and fixing lines

After the speech check, `stt.review` lists voiced lines that may have the wrong voice: the
age/gender model hears the other gender with 0.9+ on the line's voice separated from music (Hybrid
Demucs) while the line does not clearly sound like its own speaker (print similarity under 0.35),
or the line's voice print is clearly (0.15+) closer to another speaker than to its own. On
S29E01-E04 this brought the list from 100 to 65 lines (9-26 per episode) and the host's
narration and shouts "heard as female" from 30 to 6. Shouts and crowd noise still fool the
gender model, so this is a list for a person, not an automatic fix.

- `review.json`: line number (as in `subs.ru.srt`), time, speaker, voice used, what was heard,
  English and Russian text, and a ready `fix`;
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
   1180 cues overlap in time). `overlap: true` marks cues with another speaker audible under
   them (~9%).
2. Speaker labels flicker at turn edges; short sentences get one (majority) speaker and other
   speaker changes are moved to sentence boundaries (cues ending mid-sentence 31% → 12%).
3. `stt.dub` drops interjections/backchannels ("Yeah", "Wow", "Go, go!") that sit on another
   speaker or inside a rapid exchange (S10E01: 36 of 1180): they stay audible in the original.
4. The voice-over timeline solver never lets two Russian lines collide: it shifts lines within
   −360/+500 ms, speeds them up to a comfort limit and asks for shorter text only when needed.
5. The original track stays under the voice. By default (`"mix": {"duck": "voices"}`) Hybrid
   Demucs (`stt.separate`) splits it into voices and everything else, and only the voices go
   12 dB down while the Russian voice speaks; music and effects only 3 dB (`rest_duck_db`), so
   the scene keeps its sound (43-min episode: 40 s on the GPU). `"duck": "all"` lowers the whole
   original by 12 dB as before. `"subtitles": false` leaves the Russian subtitles out of the mkv.
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
right gender 78.2% -> 84.8%; `"line_gender": false` turns it off).
Diarization: `diarizen-v2` (default since 28.10; `pyannote-community-1` when `diarizen_env` is missing; more speaker changes found: F1 vs the captions' turn marks 0.73 -> 0.83, who is who the same, see `docs/roadmap.md` §6; its own venv: `bash scripts/setup_envs.sh diarizen`), then speaker refinement (`stt.refine_speakers`,
`"speaker_refine"` in `stt_config.json`): every diarized turn gets an embedding of the stronger
WeSpeaker ResNet293 (pyannote's PyTorch port of the checkpoint, ~17 s per episode), every speaker a
centroid of its turns of 1.5+ s, and a turn that sounds clearly (0.1) closer to another speaker
moves there before the words get their speakers. Measured against SDH speaker labels of
Australian Survivor S10 (mostly off-screen or unclear voices): voiced cues on their person's main
speaker 63.5% -> 71.1% on E01/E12/E16 (249 lines), diarization turns 63.6% -> 68.8% on all 24
episodes (1300 lines); 10-28 cues per episode change the voice gender. Splitting speakers into
sub-voices and the CAM++ model did worse. An LLM check of the speaker of each line (gpt-6.1-sol reading the
conversation with named speakers, fixing lines like "Get him, George!" on George) proposed 22-33
fixes per episode but changed nothing on the SDH-labelled lines, and the line's own voice was
closer to the new speaker in only 26 of 62 fixes it could judge: not used. Gender/age: `audeering/wav2vec2-large-robust-24-ft-age-gender`
(CC-BY-NC-SA-4.0, abstains below 0.7).

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
- `words.json`, `diarization.*.json`, `asr.*.json`, `voice.*.json`, `timings.json`, `logs/`.

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
