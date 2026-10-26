# English video → Russian voice-over

Package `stt/` turns an English episode into speaker-attributed English subtitles, names the
speakers, determines their gender (Russian grammar, voice choice), translates to Russian fitted
to the timing and prepares the dubbing script. The `voiceover` package synthesises it (Vosk or
TeraTTS, a voice per speaker gender), fits the lines to the timeline and exports the voice
track; `stt.mix` puts it over the ducked original and muxes a video copy. What comes next
(speaker bank, voice cloning, tips to test): `docs/roadmap.md`.

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
        mix: voice over ducked original, mkv with Russian + original  <name>.ru.<engine>.wav / .mkv
```

## Quick start

```bash
bash scripts/setup_envs.sh                      # subs_env (once)
subs_env/bin/hf auth login                      # once; accept the terms of pyannote community-1 and Cohere Transcribe
# models/vosk-model-tts-ru-0.10-multi for Vosk; OPENROUTER_API_KEY and DEEPSEEK_API_KEY in .env

# whole season, up to the Russian dubbing script (no TTS):
subs_env/bin/python -m stt.run "/media/.../Australia S10 1080p/*.mkv" --until dub
# one episode end to end (voice-over, checks, mix); --engine teratts for TeraTTS voices:
subs_env/bin/python -m stt.run ep01.mkv
```

Browser UI: `subs_env/bin/python -m ui.dub_app` (http://127.0.0.1:7861). Tab "Обработка":
episodes of a folder, engines, recognition mode, LLM, "embed subtitles", "duck only voices",
stress check and speaker refinement; "test piece" voices N minutes from minute M into
`<out>/_проба` (with a short video) to try settings quickly. One job at a time, engines one after
the other; progress per episode and stage, the log, the finished files.

Defaults live in `stt_config.json` (mode, models, OCR, LLM, TTS engine and voices, checks,
ducking). Stages already done are skipped; `--force STAGE` recomputes it and the following
ones; `--mode fast` uses Whisper only.

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
{"llm": {"model": "openai/gpt-6.1-sol", "opencode": ["openai/*"], "router": "orcarouter",
         "stages": {"stress": ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "openrouter:google/gemini-4-argon"]}},
 "ui": {"llm_transport_choice": true}}
```

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
  second half is not said during the pause (S10E01 had 79 cues with such pauses).
- The timeline solver prefers to start a line 250 ms after the English (voice-over style) and
  prices an early start 6x a late one (`policy.target_lag_ms`, `policy.weight_early`); base
  tempo 5.0 syllables/s, faster only where a line does not fit (`stt_config.json`).
- Continuation cues are not merged into one spoken unit for dubs (`units.enabled: false`): a
  merged unit placed at its first cue ran its later parts 1-4 s ahead of the speaker.

## Voices, stress and the speech check

- **Engines.** Vosk (CPU, 4 workers; voices 49 female / 56 male) or TeraTTSv2 (GPU, ~0.35 s per
  line; ru_f1 / ru_m5, also ru_f2, ru_m1). Every line gets the voice of its speaker's gender;
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
  `"stress_check": false` in `stt_config.json`.
- **Speech check (after synthesis).** `voiceover.speech_check` transcribes every clip with
  GigaAM-v3 and lists clips whose character error rate is over 15% (mispronounced
  abbreviations, swallowed words), drawn-out words and too slow/fast lines in `speech_check.md`.

## Season voice bank

Episodes of one folder share `<out-root>/<folder>.voices.json` (`stt.voice_bank`, on with
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
`openrouter:openai/gpt-6-luna:online` (`llm.stages.cast`), reads the show and season off the
folder and file names and writes `<out-root>/<folder>.cast.json`: every person who speaks, with
spellings, nicknames, gender, role and relations (twins, couples, parent and child). The
glossary uses its spellings, the speaker naming gets it as evidence, and after naming
`stt.cast.check_speakers` turns other spellings of a cast name into the cast's ("John Misch" ->
"Jon Misch"), and logs names not in the cast and genders that contradict it (`cast_check` in
`speakers.json`). The gender is never changed from the cast: the web is wrong too (S10 Stevie
Khouw came back "female"; the voice and the transcript said male, correctly).

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
Diarization: `pyannote-community-1`, then speaker refinement (`stt.refine_speakers`,
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

```bash
# ASR vs English subtitles (clip snapped to subtitle gaps, SDH markup removed)
subs_env/bin/python -m stt.bench asr --media ep.mkv --srt ep.en.srt --start 600 --duration 600 --out output/stt_bench/ep_10m
subs_env/bin/python -m stt.bench summary output/stt_bench/ep_10m ...
# diarization: VoxConverse DER, or SDH speaker labels of a real episode
subs_env/bin/python -m stt.bench_diar --parquet voxconverse-test.parquet --out output/stt_bench/diar
subs_env/bin/python -m stt.sdh_eval --srt ep.en.srt output/stt/ep/diarization.*.json
# translations side by side
subs_env/bin/python -m stt.compare_translations output/stt/ep --variants a b --ass ep.ru.ass
```

Why these models (Australian Survivor S10, three 10-minute clips, WER against English SDH
subtitles): LLM fusion of Cohere + Whisper (+ Qwen3) ~17%, Cohere 19.6%, Whisper large-v3 20.2%,
Qwen3-ASR 20.5%, the other 15 models 21–27% (Canary 180M loops). pyannote vs DiariZen on a real
episode: equal speaker purity, pyannote 2.5x faster. Full tables: `output/stt_bench/RESULTS.md`
(local report, not committed).

## Outputs (`output/stt/<episode>/`)

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
  `<name>.ru.<engine>.wav`, `<name>.ru.<engine>.mkv`.
- `ocr.json` — captions, name cards and all text events with times and boxes.
- `words.json`, `diarization.*.json`, `asr.*.json`, `voice.*.json`, `timings.json`, `logs/`.

Manual fixes: `speakers.override.json` (`{"S04": {"gender": "female", "label_ru": "Анна"}}`),
then `stt.run ... --force speakers`; edit `glossary.txt` and use `--force translate`.

## Tests

```bash
subs_env/bin/python -m unittest tests.test_stt_local -v
```
