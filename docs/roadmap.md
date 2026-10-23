# Roadmap

What comes after the current pipeline (`docs/stt_local.md`). Numbers are from S29E01/E02
(Survivor: San Juan del Sur, 64 and 43 min) unless noted.

## 1. Now: stable whole episodes on Vosk and TeraTTS

The first goal: any episode of a season, from the video to the mkv, without manual steps.

- S29E02 from scratch with Vosk: 28 min for a 43-min episode (voice-over on the CPU 11 min of
  it), no failures, 0.79% character errors in the Russian ASR check, 7 of 608 lines over 15%.
  The same script with TeraTTS: 6 min more (voice-over 4 min), 0.78%, 4 lines over 15%.
- Known weak spot: diarization merges similar voices. S29E01 has one voice cluster with two
  women (Jaclyn and Baylor), so one of them gets the other's name and voice. The season
  speaker bank (below) is the planned fix.

## 2. Season speaker bank

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

Only after section 1: when whole episodes are stable on Vosk and TeraTTS.

Measured: ESpeech-TTS-1_RL-V2 (F5-TTS trained for Russian, accepts `+` stress marks),
reference = 8 s of an English confessional of Jon Misch + 1 s of silence and its transcript,
60 of his lines from S29E01, 32 sampling steps, on the RTX 4070 Ti SUPER:

- 1.16 s per line including the GigaAM check and up to 3 takes (63 takes for 60 lines), so
  ~18 min for a 900-line episode; 3.1 GB VRAM;
- 1.0% character errors on first takes, 0.83% after retries (Vosk 0.72%, TeraTTS 0.71% on the
  whole episode);
- each line must get its exact length (`fix_duration` = reference + slot length). F5's default
  length estimate counts UTF-8 bytes, so Cyrillic text came out about twice too long and the end
  of the English reference leaked into the speech (31% character errors);
- not yet judged by ear: how much of the voice is kept and how strong the English accent is.
  Den4ik (Telegram): ESpeech-F5 keeps the acoustics but not hoarseness or breathiness, and has
  no accent cross-lingually. Samples: `output/demo/f5_espeech/`.

Other candidates from the Telegram chat and 4pda: OmniVoice and Qwen3-TTS with Russian stress
fine-tunes, Higgs TTS v3. TeraTTS's own cloning is weak (its author: not worth the effort);
Supertonic-3 voices converted to TeraTTS are unstable (azimutspb on 4pda).

Open problems: picking clean references automatically (reality TV is loud and emotional),
music under the reference (ClearerVoice separation?), licences of the models.

## 4. To test (tips from Telegram and 4pda)

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

## Not planned

- **MOSS-Transcribe-Diarize as a third input of the ASR fusion.** It would be one more full run
  (~1.5 min per 10 min of audio, 3.1 GB VRAM) and a third input did not help before (Qwen3-ASR:
  within the LLM's run-to-run spread). Alone it is close to the fusion of Cohere + Whisper
  (9.6% vs 9.1% WER on three 10-min clips of S10E01), not better, so it cannot replace them
  either. Its speaker separation was better than pyannote on one clip and worse on two.
- **Nemotron-3-Diarization**: at most 8 speakers; episodes have 16-21.
