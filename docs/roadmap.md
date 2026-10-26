# Roadmap

What comes after the current pipeline (`docs/stt_local.md`). Numbers are from S29E01/E02
(Survivor: San Juan del Sur, 64 and 43 min) unless noted.

## 1. Now: stable whole episodes on Vosk and TeraTTS

The first goal: any episode of a season, from the video to the mkv, without manual steps.

- S29E02 from scratch with Vosk: 28 min for a 43-min episode (voice-over on the CPU 11 min of
  it), no failures, 0.79% character errors in the Russian ASR check, 7 of 608 lines over 15%.
  The same script with TeraTTS: 6 min more (voice-over 4 min), 0.78%, 4 lines over 15%.
- S29E01-E04 in one run (23.10, gpt-6.1-sol through the ChatGPT subscription, voice bank on):
  Vosk 1 h 50 min for the four (23-31 min per episode; fresh ASR ~6 min, LLM steps ~10 min,
  Vosk voice-over 10-14 min on the CPU), then TeraTTS 29 min for the four (voice-over 4-5 min
  each). No failures, no fallback to OrcaRouter needed. Character errors: Vosk 0.57-1.06%,
  TeraTTS 0.50-0.97%; the voice bank found no naming conflicts across the four episodes.
- Done since (26.10): speaker refinement by WeSpeaker ResNet293 (voiced cues on their person's
  main speaker 63.5% -> 71.1% on S10 SDH labels), ducking only the original voices, the browser
  UI (processing, voice bank, review), the review list with corrections, ESpeech F5 engine.
- Known weak spot: diarization merges similar voices. S29E01 has one voice cluster with two
  women (Jaclyn and Baylor), so one of them gets the other's name and voice. The season
  speaker bank (below) is the planned fix.

## 2. Season speaker bank

Done (`stt.voice_bank`, see `docs/stt_local.md`): prints, matches as LLM evidence, naming of
unnamed clusters, conflicts, season re-check, clean reference clips (5-9 s, one voice, music
removed, Whisper transcript), UI tab with "another clip". Splitting a cluster into sub-voices
was tried on S10 and lost accuracy (one person's narration and on-location voice split too).

Done (26.10, idea of the user): **the season's cast list from the web** (`stt.cast`) as
evidence for the glossary and the speaker naming, and a check of the names after it. No
measurable gain in names on S10/S29 (they were mostly right already), one spelling per person,
$0.01-0.03 per season; the web lookup got one gender wrong of 45 people, so gender conflicts are
only logged. Worth watching on less known seasons.

Keep a voice print per named person across the episodes of a season:

- clusters of a new episode are matched to the bank, so names and gender stay the same in every
  episode and do not depend on the LLM finding them in the transcript;
- a cluster that matches two people is split, two clusters of one person are joined;
- candidates for the voice prints: the ones Shmyrev ranks highest in his Telegram comparison
  (voxblink2 ResNet-293) and the pyannote embedding the pipeline already computes;
- the bank also keeps, per person, the best clean samples: solo speech (no overlap, no music),
  8-12 s, with the exact transcript from the aligned words. Confessionals are the natural
  source. These samples are the references for voice cloning (section 3).

## 3. A voice per main character (voice cloning)

Done (26.10): engine "f5" (UI "ESpeech (голоса героев)"), clips from the voice bank, fallback
voices for people without a clip, leaked reference speech cut off. S29E03: 19 min per episode,
1.5% character errors (Vosk 0.6%, TeraTTS 0.5%); liked by ear. Open: several clips per person
(calm / excited) only if they can be picked reliably; licences of ESpeech and the clips.

History of the measurements:

Measured: ESpeech-TTS-1_RL-V2 (F5-TTS trained for Russian, accepts `+` stress marks),
reference = 8 s of an English confessional of Jon Misch + 1 s of silence and its transcript,
60 of his lines from S29E01, 32 sampling steps, on the RTX 4070 Ti SUPER:

- 32 sampling steps: 1.16 s per line including the GigaAM check and up to 3 takes (63 takes for
  60 lines), ~18 min for a 900-line episode; 3.1 GB VRAM; 1.0% character errors on first
  takes, 0.83% after retries (Vosk 0.72%, TeraTTS 0.71% on the whole episode);
- 7 steps (EPSS schedule, a 4pda tip; F5 applies it by itself for 5-7, 10, 12, 16 steps):
  0.50 s per line with the check (64 takes), ~7.5 min per episode, 0.77% after retries;
  16 steps: 0.71 s, 0.76%. By ear almost the same as 32 steps (maybe a little less emotion):
  32 steps for the normal voice-over, 7 as a fast mode (`output/demo/f5_espeech/nfe7/`);
- each line must get its exact length (`fix_duration` = reference + slot length). F5's default
  length estimate counts UTF-8 bytes, so Cyrillic text came out about twice too long and the end
  of the English reference leaked into the speech (31% character errors);
- voice similarity judged by ear: good. Den4ik (Telegram): ESpeech-F5 keeps the acoustics but
  not hoarseness or breathiness, and has no accent cross-lingually. Samples:
  `output/demo/f5_espeech/`.

First listening demo (23.10): three 75 s scenes of S29E02-E04 (23-30 lines, 6-7 people each)
voiced line by line in each speaker's cloned voice, in the TeraTTS slots and at their
loudness: 1.1 s per line with up to 3 takes, 2.0-2.5% character errors (short exclamations
and names are the hard part). References: the voice bank's cleaned clips (vocals only).
Prototype scripts and videos (F5 / TeraTTS / Vosk / original tracks): `output/demo/s29_f5/`.
F5 runs in its own venv (f5-tts pins differ from subs_env); an engine would call it there.

Other engines discussed in the Telegram chat (Shmyrev's table of 20.10.2026, 1270 Russian
phrases; xRT = synthesis time / audio length, stress errors as he counts them):

| Engine | Size | CER % | stress errors | xRT | cloning | verdict for dubbing |
|---|---|---|---|---|---|---|
| Vosk TTS 0.10 (ours) | ~40M | 0.9 | 2.29 | 0.4 CPU, 0.07 GPU (torch) | no | in use |
| Vosk TTS 0.11 | ~40M | 0.8 | 1.78 | same | no | drop-in when released (not public on 23.10) |
| ESpeech F5 RL-V2 | 336M | 1.4 | 2.04 | 0.45 (4090) | yes | the clone engine to build on (above) |
| OmniVoice 0.6B | 600M | 1.2 | 3.13 | 1.5 (1080), ~0.3 (4090, official) | yes | livelier reading, more stress errors; stress fine-tunes only on Telegram |
| Pocket-TTS RU (xVibe, Storm) | ~90M | 1.9-2.1 | 2.44 | 0.17 (1080), 0.41 CPU | yes | fastest, but weak phrase starts, 7.6% lost endings (its own card), low similarity |
| Higgs TTS v3 | 4B | 0.8 | 1.81 | 0.55 (4090) | yes | best stress, but large |
| Qwen3-TTS 0.6B / 1.7B | | 1.1 / 0.9 | 5.36 / 3.53 | 0.91 / 2.25 (1080) | yes | too many stress errors |

TeraTTS's own cloning is weak (its author: not worth the effort); Supertonic-3 voices converted
to TeraTTS are unstable (azimutspb on 4pda); Fish S2 ignores RUAccent marks; Chatterbox is poor
in Russian.

Open problems: picking clean references automatically (reality TV is loud and emotional),
music under the reference (ClearerVoice separation?), licences of the models.

## 3b. Translation check before voicing

A pass of a strong model over the finished translation of an episode: names and terms as in the
glossary, gender agreement of every line with its speaker, lines that lost their meaning when
shortened. Only worth it if it measurably reduces errors heard in the voice-over (the LLM check
of line speakers did not, see `docs/stt_local.md`); measure on a few episodes first.

## 4. Vosk TTS 0.11

Shmyrev's table of 20.10.2026 lists Vosk TTS 0.11: same speed as 0.10, fewer stress errors
(1.78 vs 2.29) and 0.8% CER. Not published yet on 23.10 (no model file on alphacephei.com).
When it is out: download it, set `voiceover.engines.vosk.model_path` in `stt_config.json` to it, re-voice S29E01 and
compare the speech check and the stress errors with 0.10.

## 5. One-command setup and a settings page (after the core is done)

So that a new user only clones the branch, runs one script and then works in the browser:

- `setup` script: checks the NVIDIA driver and `uv`, creates the environments (`subs_env`, and
  `f5_env` if wanted), downloads the models (Vosk TTS, TeraTTS, ESpeech F5, pyannote with the
  Hugging Face login and its terms, GigaAM, WeSpeaker ResNet293, Hybrid Demucs), asks for the
  OpenRouter key and writes it into `.env` (never printed), runs a short self-test, then opens the
  UI;
- `run` script (or the setup script run again): only starts the UI and opens the browser;
- a "Настройки" page in the UI built from `stt_config.json`: the options people actually change
  (engines and voices, LLM model, recognition mode, subtitles in the mkv, duck only voices,
  stress check, speaker refinement, the review list on/off, ESpeech fast mode, default folders),
  saved into `stt_config.local.json` so the shared config stays untouched and "reset" is just
  deleting the local file. For example: no review step at all, just the finished voice-over file.

## 6. To test (tips from Telegram and 4pda)

- **Spoken stress check.** TeraAnaly's stress head, given the text: on 159 phrases synthesised
  with a deliberately wrong stress it heard the wrong stress in 73% (Vosk) / 62% (TeraTTS) of
  clips and reported a wrong stress on 1-2% of words said correctly. Useful only on the
  ~50 homographs and guessed words per episode the stress check already handles (on all words
  it would raise ~100 false alarms per episode): a second TeraTTS take when the voice ignored
  the stress mark.
- **TeraTTS teacher sampler** (`diffusion_model: "teacher"`): recommended by the author for
  quality on a GPU; compare dropped words, character errors and speed with the distilled one.
- **RuVoice stress placer** (azimutspb's PC port of the Android RuVoice TTS, used by others as
  a stress placer): compare with RUAccent on the homographs and guessed words of an episode.
- **Shmyrev's TTS metrics** (intelligibility, stress, speaker similarity) to compare engines
  and cloned voices with one yardstick.
- **DiariZen Large MD v2**: best open model in Shmyrev's Russian telephony benchmark (13.6% vs
  21.6% for pyannote community-1). The older DiariZen was as accurate as pyannote on Survivor
  and 2.5x slower; v2 is worth one run on S29E01 against the SDH speaker labels.

- **Look again on GitHub / Hugging Face** for diarization and speaker-embedding models newer
  than pyannote community-1 and WeSpeaker ResNet293 (the S10 SDH benchmark,
  `stt.refine_speakers sdh`, measures any of them in minutes).

## Not planned

- **MOSS-Transcribe-Diarize as a third input of the ASR fusion.** It would be one more full run
  (~1.5 min per 10 min of audio, 3.1 GB VRAM) and a third input did not help before (Qwen3-ASR:
  within the LLM's run-to-run spread). Alone it is close to the fusion of Cohere + Whisper
  (9.6% vs 9.1% WER on three 10-min clips of S10E01), not better, so it cannot replace them
  either. Its speaker separation was better than pyannote on one clip and worse on two.
- **Nemotron-3-Diarization**: at most 8 speakers; episodes have 16-21.
