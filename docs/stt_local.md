# English video → Russian voice-over

Package `stt/` turns an English episode into speaker-attributed English subtitles, names the
speakers, determines their gender (Russian grammar, voice choice), translates to Russian fitted
to the timing and prepares the dubbing script. The `voiceover` package synthesises it (Vosk or
TeraTTS, a voice per speaker gender), fits the lines to the timeline and exports the voice
track; `stt.mix` puts it over the ducked original and muxes a video copy.

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

Defaults live in `stt_config.json` (mode, models, OCR, LLM, TTS engine and voices, checks,
ducking). Stages already done are skipped; `--force STAGE` recomputes it and the following
ones; `--mode fast` uses Whisper only.

## What runs where

| Step | Where | What is sent out |
|---|---|---|
| Audio decoding, VAD, ASR (Whisper, Cohere), alignment, diarization, gender/age | local GPU | nothing |
| On-screen text (RapidOCR: detection on GPU, recognition on CPU) | local | nothing |
| ASR fusion (accurate mode only) | OpenRouter LLM | English ASR text of the three models per chunk + glossary |
| Glossary, speaker names, translation, shortening, gender check | OpenRouter LLM | English transcript / cues, Russian drafts |
| Stress re-check of guessed words / homographs | OpenRouter (3 LLMs) | Russian lines with those words |
| Voice-over shortening + meaning check | DeepSeek / OpenRouter | Russian lines |
| TTS (Vosk on CPU, TeraTTS on GPU), timeline fitting, Russian ASR check, mixing, muxing | local | nothing |

No audio ever leaves the machine. No manual corrections are applied anywhere; the only human
inputs are optional files: `speakers.override.json`, an edited `glossary.txt`, and the
`voiceover` overrides. Reference subtitles are used only by the benchmarks, never by the pipeline.

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
- **Stress (before synthesis).** RUAccent is right for dictionary words; errors come from
  homographs and words it guesses. Judging stress from the audio does not work (three detectors
  on 160 phrases with deliberately wrong stress scored ~40%, chance level), so
  `voiceover.stress_check` re-checks each guessed word and each homograph in its sentence:
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
5. The original track stays under the voice, lowered by 12 dB only while the voice speaks, so
   crowd noise and overlapping voices remain audible.
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
Diarization: `pyannote-community-1`. Gender/age: `audeering/wav2vec2-large-robust-24-ft-age-gender`
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
