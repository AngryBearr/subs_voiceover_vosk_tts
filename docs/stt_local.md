# Local GPU STT → Russian voice-over

Package `stt/` turns an English episode into speaker-attributed English subtitles, names the
speakers, determines their gender (Russian grammar, voice choice), translates to Russian fitted
to the timing and prepares the dubbing script for the `voiceover` package of the
subtitle-shortening branch, which synthesises (Vosk), fits the lines to the timeline and
exports the voice track. `stt.mix` puts it over the ducked original and muxes a video copy.

```
video ─► audio 16 kHz ─┬─► VAD chunks ─► Whisper  (fast)
                       │               └► Cohere + Whisper + Qwen3 → LLM fusion → forced alignment (accurate)
                       └─► pyannote diarization (who speaks when) + gender/age per speaker
video frames (2 fps) ─► OCR: burned-in captions, name cards           ocr.json
        words + captions → speakers (sentence-level smoothing) → cues subs.en.json
        LLM: season glossary, speaker names, gender check             glossary.json, speakers.json
        LLM: EN → RU per cue time window, shortening, gender fix      subs.ru.json
        dubbing script: voiced lines, voices                          dub.ru.json
        voiceover (other branch): Vosk + timeline solver + shortening voiceover/export/track.wav
        mix: voice over ducked original, mkv with Russian + original  <name>.ru.wav / .ru.mkv
```

## Quick start

```bash
bash scripts/setup_stt_envs.sh                  # stt_env + stt_qwen_env (once)
stt_env/bin/hf auth login                       # once; accept the terms of pyannote community-1 and Cohere Transcribe
# OPENROUTER_API_KEY (and DEEPSEEK_API_KEY for voiceover shortening) in .env

# whole season, up to the Russian dubbing script (no TTS needed):
stt_env/bin/python -m stt.run "/media/.../Australia S10 1080p/*.mkv" --until dub
# with the voice-over and the final mix (path to a checkout of the shortening branch with its
# subs_env and models/vosk-model-tts-ru-0.10-multi):
stt_env/bin/python -m stt.run ep01.mkv --voiceover /path/to/shortening-checkout
```

Defaults live in `stt_config.json` (mode, models, OCR, LLM, voices, ducking). Stages already done
are skipped; `--force STAGE` recomputes it and the following ones; `--mode fast` uses Whisper only.

## What runs where

| Step | Where | What is sent out |
|---|---|---|
| Audio decoding, VAD, ASR (Whisper, Cohere, Qwen3), alignment, diarization, gender/age | local GPU | nothing |
| On-screen text (RapidOCR: detection on GPU, recognition on CPU) | local | nothing |
| ASR fusion (accurate mode only) | OpenRouter LLM | English ASR text of the three models per chunk + glossary |
| Glossary, speaker names, translation, shortening, gender check | OpenRouter LLM | English transcript / cues, Russian drafts |
| Voice-over shortening + meaning check (other branch) | DeepSeek / OpenRouter | Russian lines |
| TTS (Vosk), timeline fitting, mixing, muxing | local CPU/GPU | nothing |

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

`dub.ru.json` carries a `voice` per line (female 49, male 56 by default) for a two-voice dub;
the current `voiceover` engine uses one voice per run (`voiceover.voice` in `stt_config.json`).

## Environments

| venv | Python / torch | Used for |
|---|---|---|
| `stt_env` | 3.12 / 2.11 cu128 | orchestration, Whisper (CTranslate2), Cohere, pyannote 4, gender/age, mixing |
| `stt_qwen_env` | 3.12 / 2.11 cu128 | Qwen3-ASR + Qwen3-ForcedAligner (`qwen-asr` pins transformers 4.57) |

Model stages run as subprocesses in their venv automatically. No system ffmpeg is needed
(PyAV decodes; `imageio-ffmpeg` provides ffmpeg for muxing).

## Models

ASR (`stt/asr.py`): `whisper-large-v3` (fast mode), `whisper-large-v3-turbo` (half the VRAM),
`cohere-transcribe` + `qwen3-asr-1.7b` (accurate mode, fused with Whisper by an LLM).
Diarization: `pyannote-community-1`. Gender/age: `audeering/wav2vec2-large-robust-24-ft-age-gender`
(CC-BY-NC-SA-4.0, abstains below 0.7). Alignment: `Qwen/Qwen3-ForcedAligner-0.6B`.

Benchmarked and dropped (summary below): Parakeet v2/v3/Unified/TDT 1.1B, Canary 1B v2/180M/Qwen
2.5B, Granite Speech 4.1 (+plus), Voxtral Mini 3B/Realtime 4B, Kyutai STT 2.6B, Moonshine v2,
Whisper medium.en, Distil-Whisper large-v3.5, Qwen3-ASR 0.6B, DiariZen, TeraAnaly.

## Benchmarks

```bash
# ASR vs English subtitles (clip snapped to subtitle gaps, SDH markup removed)
stt_env/bin/python -m stt.bench asr --media ep.mkv --srt ep.en.srt --start 600 --duration 600 --out output/stt_bench/ep_10m
stt_env/bin/python -m stt.bench summary output/stt_bench/ep_10m ...
# diarization: VoxConverse DER, or SDH speaker labels of a real episode
stt_env/bin/python -m stt.bench_diar --parquet voxconverse-test.parquet --out output/stt_bench/diar
stt_env/bin/python -m stt.sdh_eval --srt ep.en.srt output/stt/ep/diarization.*.json
# translations side by side
stt_env/bin/python -m stt.compare_translations output/stt/ep --variants a b --ass ep.ru.ass
```

Why these models (Australian Survivor S10, three 10-minute clips, WER against English SDH
subtitles): LLM fusion of Cohere + Whisper + Qwen3 ~17%, Cohere 19.6%, Whisper large-v3 20.2%,
Qwen3-ASR 20.5%, the other 15 models 21–27% (Canary 180M loops). pyannote vs DiariZen on a real
episode: equal speaker purity, pyannote 2.5x faster. Full tables: `output/stt_bench/RESULTS.md`
(local report, not committed).

## Outputs (`output/stt/<episode>/`)

- `subs.en.json`, `subs.ru.json` — project subtitle schema (`index`, `start`, `end`, `duration`
  in ms, `text: list[str]`, `gender`) plus `speaker`, `speaker_name`, `overlap`; the Russian
  file adds `text_en`, `budget_chars`, `window_ms`. `.srt` copies; `subs.en.speakers.srt` shows
  `[S03] text` for review.
- `dub.ru.json` / `dub.report.json` — voiced lines with `voice`, skipped lines with reasons.
- `speakers.json` — ID, talk time, voice gender/confidence, age, LLM name/label/role/evidence,
  `gender_source` (`voice`, `voice+text`, `text`, `manual`), speaker embedding.
- `glossary.json/.txt`, `hotwords.txt`; the season glossary sits next to the episode folders.
- `voiceover/` (other branch's run), `<name>.ru.wav`, `<name>.ru.mkv`.
- `ocr.json` — captions, name cards and all text events with times and boxes.
- `words.json`, `diarization.*.json`, `asr.*.json`, `voice.*.json`, `timings.json`, `logs/`.

Manual fixes: `speakers.override.json` (`{"S04": {"gender": "female", "label_ru": "Анна"}}`),
then `stt.run ... --force speakers`; edit `glossary.txt` and use `--force translate`.

## Tests

```bash
stt_env/bin/python -m unittest tests.test_stt_local -v
```
