# Roadmap

Open items only. The full chronological log of finished studies, plans and stages (with every number, date
and verdict) is `docs/history.md`; measurements are in `docs/experiments.md`; how the pipeline works today is
`docs/stt_local.md`. Superseded plans live in `docs/archive/`.

## Priority 1: lines still on the generic voice (the user's priority)

After own-voice clones, `fill.voice: speaker` and `season_voices.guests: home` (all on by default since 21.11,
`docs/history.md`, study of 20.11), what is left per season: mixed clusters (S20 34, S49 65, S50 36 lines),
speakers under 1 s of clean speech (1-22 lines), fill lines whose speaker has no voice (1-21), speakers of
unknown gender (S50 84: the tribe "Kalo" named as a person). Planned:

- (a) voice identities across episodes for Survivor (`voice_bank.identities` / `stitch`, today on only for
  The Amazing Race), so a small unnamed voice gathers speech from several episodes and a mixed cluster splits
  into its voices; measured on S51 (SDH labels) before any default;
- (b) a tribe name is never a speaker name ("Kalo");
- (c) subtitle labels as hints where they exist.

## Priority 2: awaiting the user's ear

- Russian clone samples: the Selectel author's taste ranking (melody, loudness, tempo against the original) is
  still open (`docs/history.md`, experiment of 19.11).
- Voiced seasons awaiting listening (see the run-state notes): S51 re-voiced 23.11, S48, S01E01, S20 rebuild.

## Priority 3: other shows (The Amazing Race)

Not done (`docs/history.md`, proposal of 10.11 and trial of 8.11): the card parser for team cards, the generic
cast prompt (step 4), the self-clone fallback, a whole season.

## Smaller open items

- Shouts: an OpenRouter speech-to-text backend in `stt.asr` (MAI-Transcribe-2, Scribe v2) for the fill /
  fusion, if the user wants it (`docs/history.md`, plan of 10.11).
- Re-grouping the turns with a bonus for DiariZen's clusters (B-cubed on S51 0.569 -> 0.613, not run through
  the pipeline); remaining stress errors of other kinds (`docs/history.md`, plan of 1.11).
- Old-season speakers: the card persons as extra Viterbi states (about one point; `docs/history.md`, plan of
  31.10); a LavaSR fine-tune for narrowband seasons (plan of 7.11).
- S20E02 / E03 in cloned voices: re-voice with `--engine f5` so Courtney and Danielle get their clips.
- Tips from Telegram and 4pda still untested: `docs/history.md` §6.

## Not planned

- MOSS-Transcribe-Diarize as a third ASR fusion input (not better than Cohere + Whisper: 9.6% vs 9.1% WER).
- Nemotron-3-Diarization: at most 8 speakers; episodes have 16-21.
