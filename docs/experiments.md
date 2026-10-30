# Experiments and what came of them

## Where things stand (29.10.2026)

What helps (in use), with the measurement that decided it:

| What | Measured gain | Where |
|---|---|---|
| LLM fusion of Cohere + Whisper (accurate mode) | S10 clips WER ~17% vs 19.6% / 20.2% alone; S20 episodes 6.5-12.9% of words differ from the CBS captions, better than either model in every episode | `docs/stt_benchmarks.md` |
| DiariZen v2 as the diarizer (28.10) | speaker-change F1 vs the captions' `>>` marks 0.73 -> 0.82-0.85; S10 SDH lines on their person 69.4% -> 71.3%; +2.5 min per episode | roadmap §6 |
| WeSpeaker ResNet293 speaker refinement | S10 SDH cues on the right speaker 63.5% -> 71.1% | `docs/stt_local.md` |
| Gender per line for speakers of unknown gender (28.10) | S10 SDH-labelled lines with the right gender 78.2% -> 84.8% | `stt/line_gender.py`, roadmap |
| Stress check of names, silero-stress disagreements, gpt-6.1-sol as the free screener (28.10) | ~20% of name uses were stressed wrong; S29E03 28 corrections (before: 15) with 110 paid answers instead of 325 | `docs/stt_local.md` |
| Spoken stress check of the checked words (28.10) | small: S29E03 11 of ~780 such lines heard wrong in the three engines, 4 fixed by another take | roadmap §6 |
| ESpeech: "— " before the line, +0.3 s, retake while a word is not heard (28.10) | lines losing a word 40% -> 19%, character errors 2.3% -> 0.85% | roadmap §7 |
| TeraTTS: retake while a word is not heard; sampler temperature 0.667 | words not heard 93 -> 64 (S29E03); first takes 1.51% -> 0.74% character errors (S29E01) | roadmap §7, `docs/stt_local.md` |
| TeraTTS voices for people without a clip in the ESpeech track (28.10) | S29E06 character errors 1.65% -> 1.03%, flagged lines 12 -> 5 | `docs/stt_local.md` |
| TeraTTS female voice ru_f1+ru_f2 (28.10) | the user's choice by ear (ru_f1 alone sounded cartoonish) | roadmap §6 |
| Stability: GPU out-of-memory retries, the subscription check before a run, UI warning | runs no longer die on a busy GPU or fall back to paid routers unnoticed | `docs/stt_local.md` |
| One-command install, self-test, settings page, TTS cache cleanup (28-29.10) | not yet tried by a person on a clean machine | roadmap §5 |
| Stress vote through the subscriptions: sol + gemini-4-argon + claude-haiku-5.5 of OpenCode Go (29.10) | S29E03, 164 sentences: the same 27 corrections as the paid trio; free, ~1 min per episode | `docs/stt_local.md`, Language models |
| One stress per name for the season (29.10) | S29E01-E10: no name was stressed differently between episodes even without it; ~180 questions saved per season, hand fixes season-wide | `docs/stt_local.md` |
| Vosk in a CPU lane beside the GPU engines, half the CPU threads (29.10) | 10-min piece, three engines: 25.4 -> 16.4 min (all cores for Vosk: 19.6 min) | `docs/stt_local.md` |
| Stress mark on every word: RUAccent per cue, apart at ellipses, dictionary stress for bare words (29.10) | words reaching the engines with no stress 1.8-2.9% -> 0.1-0.4% (S51E01, S29E01, S20E03); corrected names at a cue's start now stressed | `docs/stt_local.md` |
| Numbers by ru-normalizr + a range rule (29.10) | 64 subtitle-like phrases: ~15 wrong -> 0 ("Мне 22 года", "около 3-4 лет"); the dubbing translations write numbers in words (4 of 18003 lines had digits) | `voiceover/normalize.py` |
| Season cast from Wikipedia through the subscription (29.10) | free; S51, S29, S20, Australian S10 as complete as the paid web search | `stt/cast.py` |
| DiariZen exclusive layer (30.10) | S20 E01/E05/E10/E14 speaker changes vs the `>>` marks: F1 0.841 -> 0.856; S51 SDH lines 62.0% -> 61.7% (noise), people merged 44 -> 41 | `docs/stt_local.md`, Models |
| Word-level Viterbi relabelling (30.10) | on top of it: S20 F1 0.856 -> 0.878 (recall 0.828 -> 0.873); S51 lines 61.7% -> 63.5%, people merged 41 -> 39; 9 of 9 episodes better. Only filling the words without a diarizer label gives S20 0.863, S51 62.9%; moving labelled words gives the rest | `stt/boundaries.py`, roadmap 30.10 |

Tried and dropped:

| What | Why |
|---|---|
| FireRedTTS3 for the cloned voices | reads Russian with English sounds from English clips (75% character errors), no length control |
| Seed-VC: TeraTTS converted into the heroes' voices | a strong English accent by ear (the user) |
| Accent-free cloning in general | dropped by the user: stress and pronunciation matter more than a slight accent |
| DiariZen on the Demucs vocal stem | no gain over the plain audio |
| ASR on the Demucs vocal stem (29.10) | S51E01-E05 against the AMZN subtitles, WER: fusion 13.84% plain vs 14.04% vocals, Cohere 17.94 / 17.61, Whisper 15.78 / 16.01; no gain where the music is loud either |
| ReDimNet2-B6 (PalabraAI, VoxCeleb2 or + VoxBlink2) instead of ResNet293 for the refinement (29.10) | S51E01-E05 (886 SDH-labelled lines), the whole naming chain: lines on their person 62.0% -> 63.1%, people merged 44 -> 34 of 103, under the right name 65.8% -> 66.5%; but Australian S10 (5 episodes, 334 lines): 61.1% -> 59.9-60.5%, merged 37 -> 39. Not a gain that holds; kept as `--refine-speakers redimnet2-b6[-vb2]` |
| A speaker-count hint from the season cast for DiariZen (29.10) | nothing to gain: DiariZen's VBx clustering ignores the speaker count (`min/max/num_clusters` "not used" in its code); only the instantaneous count is capped |
| TeraTTS teacher sampler | no change (0.49% vs 0.52%, same lost words and speed) |
| Moving lines by the season voice bank | S10 names 63.6% before and after; 2 of 4 labelled moves wrong |
| Trusting a line's own voice over its speaker's known gender | the gender model is often the one that is wrong (shouting men heard as women) |
| Splitting a cluster into sub-voices | lost accuracy on S10 (narration and on-location voice split) |
| The season cast from the web as naming evidence | no gain in names (kept for one spelling per person) |
| Qwen3-ASR or MOSS-Transcribe-Diarize as a third fusion input | within the LLM's run-to-run spread |
| deepseek-v4-pro, grok-4.7, kimi-k3.1-pro, glm-5.3 (OpenCode Go) as stress voters | as good or better than haiku (147/149 where the paid trio agreed) but 20+ min for 55 sentences (timeouts) |
| Stress marks on one-syllable names | no change: short names like «Вэл» are misheard by the checking ASR, not misspoken |
| Active speaker detection + faces | dropped by the user before testing (a dead end) |
| Several clips per person by the line's mood (ESpeech) | dropped by the user: one voice per person is enough while some clusters still mix people |
| DiariZen without its 11-frame median filter (30.10) | S20 F1 0.843 / 0.856 (with the exclusive layer), S51 62.1% / 61.4%: no gain |
| Audio LLMs on the 45 joined places of S51 (29-30.10, `output/exp_joined/`) | gemini-4-argon 45.5% of the labelled lines right vs our 45.5%, gemini-3.8-flash 44.0% vs 48.8%, gpt-audio-mini 41.3%, qwen3.8-omni-flash 36.4%; $0.04-0.31 per run |
| DiCoW v3.3 and SE-DiCoW as the judge of who says which word (30.10, `output/exp_boundary/dicow/`, `dicow_env`; 0.12-0.2 s per target and 30 s window) | They follow the mask. A shout of another person under the target's mask lands in the target's text ("we got a lead" in Jeff Probst's), with SE-DiCoW's voice sample too. Masks widened by 0.5 s, boundaries moved by which candidate wrote the words, S20 E01/E05/E10/E14 exact on the `>>` marks: F1 0.795 -> 0.785 (SE-DiCoW), 0.785 at best (v3.3); of SE-DiCoW's moves 12 reached a marked word and 31 left one. S51 lines 63.4% -> 64.2% (noise). No mask, the voice sample alone (S51E01): 10% of labelled lines right vs our 65%. Weak hint only: words no candidate wrote are caption turns 5.2% of the time vs 0.8% |
| A text LLM on sentence boundaries we give to one speaker (30.10, gpt-6.1-sol, `output/exp_boundary/text/report.md`) | S20: 2180 boundaries, 12.5% are caption turns; its claims right 65-74%, it finds 29-39% of them. Upper bound if the acoustics always found the person: F1 0.878 -> 0.890. Measured, not plugged in |
| Levelling loudness (30.10, a measurement, no levelling was run) | Not the cause of wrong speakers. S51 885 SDH lines: right 61.7 / 64.0 / 64.4 / 65.8% from the quietest quarter to the loudest, median level -22.0 dBFS right vs -22.1 wrong, correlation 0.007 (with the line's length 0.164: short lines are the hard ones). Reality TV is already compressed: half the lines sit within 2.4 dB. The speaker model does not see gain at all: ResNet293 on a 1 s window at gain 0.1 and 3 gives cosine 1.000 |
| The centre channel of 5.1 sources instead of the mono downmix (30.10, not run) | Dropped by the user: old seasons are not 5.1 (S51E02/E05 and S20 are stereo). Measured for the record: on S51E01/E03/E04 the centre holds the voices (-23 dB, left and right -31 to -35), but it is not clean: -24 to -27 dB where we have no words. The Demucs vocal stem, a similar cleaning, gave no gain either |
| Word-level Viterbi without the change bonus, or with a bonus for the diarizer's label (30.10) | without: removed as many real changes as it added (S20E01: 33 vs 30); label bonus 0.15: S20 F1 0.873 vs 0.878, S51 63.1% vs 63.5%. It does cut what the mixed cluster gains (S51E01 95 words vs 259), but costs both metrics; on S51E05 alone it looked better (59.8% vs 58.3%), one episode misleads |
| DiariZen AHC threshold 0.4 with the whole chain (30.10) | S51 64.6% vs 63.5% (9 lines of 886, better on 3 of 5 episodes), people merged 39 -> 43 (not understood: a lower threshold should merge less); S20 F1 0.868 vs 0.878, precision 0.857 vs 0.884, E14 0.856 vs 0.887. Not taken; the S10 check was not run. The option stays (`"ahc_threshold"`, 0.5) |
| People of the season voice bank as extra states of the word-level Viterbi (30.10, code in commit 4cfd261, removed after it) | S51 63.4% vs 63.5%, S20 F1 0.878 vs 0.878, with the AHC threshold 0.4 no better than that threshold alone. A bank state gets almost nothing: 95 words on S51, of the SDH lines one landed on a bank label and it was another person's; 56 words on S20. Margins 0-0.1 and a minimum run of 1 s change nothing. Limitation: a clip can come from the scored episode, which only flatters it |

Not done yet (see `docs/roadmap.md`): the user's pick of the host's voice (`output/host_voices/`), re-voicing
S20 and S29 with all of the above, finishing S29 (E11, E13-E15, missing tracks of E01-E08), a
clean-machine test of `scripts/install.sh`, Vosk TTS 0.11 when it is out, RuVoice stress placer,
Shmyrev's TTS metrics, newer diarization and speaker-embedding models, S20E02/E03 in Courtney's and Danielle's own voices.

## Deleted on 30.10.2026

The data of the speaker boundary experiments (the conclusions are in the tables above and in
`docs/roadmap.md`, plan of 30.10). Scripts, logs and the text LLM's report stay, not in git:
`output/exp_boundary/` (`prepare.py` + `run.sh` make the variant folders again, `sweep_words.py`,
`score.py`; `dicow/boundary.py` scores a boundary exactly on the `>>` marks, `stt.turn_eval` allows one
word either side), `output/exp_ahc/`, `output/exp_joined/` (with the paid answers of the audio LLMs).

| Deleted | What it was | Made again by |
|---|---|---|
| `output/exp_boundary/s20`, `s51`, `smoke`, `runcheck`, `dicow/out` | copies of 9 episodes' inputs, 4 diarization variants, the Viterbi sweep, DiCoW's transcripts | the scripts above, ~1.5 h on the GPU |
| `output/exp_ahc/ahc*`, `s10`, `s10_audio` | DiariZen with other AHC thresholds (S51), the unfinished S10 check | `output/exp_ahc/run_s51.sh`, `run_s10.sh` |
| `dicow_env`, `models/dicow`, Hugging Face cache: DiCoW_v3_3, SE-DiCoW | python 3.11, torch 2.5.1 cu124, transformers 4.55, pandas; `BUTSpeechFIT/DiCoW`; the models with `trust_remote_code` | only if DiCoW is ever tried again |
| `~/.cache/uv` | downloaded packages (the environments keep their own copies) | the next install downloads again |

## Deleted on 29.10.2026

The outputs of the 27-28.10 experiments, their models and environments (the conclusions are in
the tables above and in the docs named there):

| Deleted | What was tried | Written in |
|---|---|---|
| `output/stt_dz_test` | DiariZen v2 on S29E01/E05-E08 end to end; S29E06 voiced both ways (`AB.mkv`) | roadmap §6 |
| `output/stt_dz_sweep` | DiariZen clustering sweep (AHC threshold, VBx Fb), the line-gender check | roadmap §6 (AHC 0.5; lower Fb splits people into 25-150 clusters) |
| `output/stt_dz_vocals` | DiariZen on the vocal stem (S10E01/E12/E16, S20E05/E10) | "Tried and dropped" above |
| `output/stt_s10_lc`, `output/stt_linecheck` | per-line checks on S10 (SDH labels) and S29: gender per line, the voice bank | roadmap "Next steps" |
| `output/tts_eval/firered`, `models/fireredtts3` | FireRedTTS3 (own environment, 19 GB) | roadmap §6 |
| `output/tts_eval/vc`, `models/seedvc` | Seed-VC voice conversion, Whisper P(ru) as an accent measure | "Tried and dropped" above |
| `output/tts_eval/f5_words`, `tera_teacher`, `tera_female` | ESpeech short words, TeraTTS teacher sampler and retakes, female voices | roadmap §6, §7 |
| `output/tts_eval/stress_scan`, `stress_retake`, `stress_heard.py` | heard stress of all words, names check, silero-stress, the sol screener; the spoken stress check | `docs/stt_local.md`, roadmap §6 (`voiceover/stress_listen.py` is the tool now) |
| `output/stt_s29_v2/eval`, `output/stt_s20/eval` | turn-change metric, recognition vs captions, ESpeech first-word probes | `stt/turn_eval.py`; the S20 section below |
| `output/stt_bench`, `output/stt_data` | the ASR and diarization benchmarks (S10 SDH, VoxConverse) | `docs/stt_benchmarks.md` |
| `models/diarizen` except `repo/` | the first DiariZen environment and its run scripts | `diarizen_env` (`bash scripts/setup_envs.sh diarizen`) uses `repo/` |
| `stt_qwen_env`; Hugging Face cache: Qwen3-ASR-1.7B, Qwen3-ForcedAligner, MOSS-Transcribe-Diarize, faster-whisper-large-v3-turbo, whisper-small, wav2vec2-xlsr-53-espeak, BigVGAN x2, CAM++ | comparison-only models and the FireRedTTS3 / Seed-VC models | `bash scripts/setup_envs.sh qwen` makes the environment again; models download on first use |
| `voiceover.<engine>/export/*.wav` in the S20 / S29 episode folders | voice tracks of finished voice-overs (the `.m4a` files and the videos stay) | an episode's next voice-over run renders them again from the TTS cache |
| `output/tts_cache`: takes no episode uses | old engine versions, calibration probes | `voiceover.cache_clean` (also in the UI settings) |
| `~/.cache/uv` | downloaded packages (the environments keep their own copies) | the next install downloads again |

Kept: the season runs `output/stt_s20` and `output/stt_s29_v2` (recognition, translations,
voice banks, run scripts), the reference captions `output/s20_subs` and `output/s29_subs`, the
LLM cache, the TTS cache takes the episodes use (for fast re-voicing), every model the pipeline
uses (`stt.selftest` lists them).

## Deleted on 27.10.2026

The outputs of the experiments below were deleted from `output/` on 27.10.2026 (disk space;
the S20 / S29 season runs are the working ones). Their conclusions live in the docs named in
each line; re-running an experiment means running its command again on the same video.

| Folder (deleted) | What was tried | What came of it, and where it is written |
|---|---|---|
| `stt_bench/s10e*_10m*` | 23 ASR models on three 10-min clips of Australian Survivor S10 against the SDH subtitles | LLM fusion of Cohere + Whisper best (~17% vs 19.6% / 20.2%); `docs/stt_benchmarks.md` |
| `stt_bench/sdh` | Speaker refinement and cluster splitting on S10 SDH speaker labels | WeSpeaker ResNet293 refinement 63.5% → 71.1% of cues on the right speaker; splitting a cluster into sub-voices lost accuracy; `docs/stt_local.md`, roadmap §2 |
| `stt_s10_eval` | The season cast from the web as naming evidence (S10E01/E12/E16, SDH labels; gpt-6.1-sol and gpt-6-luna, three runs each) | No gain in names (sol 65.5% with and without, luna 63.9-65.9%), one spelling per person, gender from the web only logged; `stt/cast.py`, roadmap §2 |
| `stt_full` | First end-to-end run of a whole episode (S10E01, Vosk) | The pipeline works on a full episode; superseded by everything after it |
| `stt` (`s10e01*`, `_проба`) | Early S10E01 runs and test pieces from the UI | — |
| `stt_s29` | First S29E01-E04 runs: Vosk / TeraTTS, mixing variants (v2-v6) | 23-31 min per episode, 0.5-1.1% character errors; roadmap §1. Replaced by `stt_s29_v2` |
| `stt_s29_sol`, `stt_s29_oc` | S29E01 with gpt-6.1-sol through OpenRouter and through the opencode subscription | The subscription works as a transport (OpenRouter / OrcaRouter as fallback); `docs/stt_local.md`, Language models |
| `stt_s29_tr`, `stt_s29_bank`, `voices` | Translation check on S29E02; the first voice bank and reference clip exports | Became `stt.voice_bank` (prints, clean clips, `--recheck`); `docs/stt_local.md`, Season voice bank |
| `demo` | Listening demos: ESpeech F5 at 7/16/32 steps, three S29 scenes in cloned voices next to TeraTTS / Vosk | 32 steps for the voice-over, 7 as a fast mode; ESpeech chosen for the heroes' voices; roadmap §3 |
| `voiceover_branch` | Outputs of the older voice-over branch (shortening v3 on AU S01E04) | `docs/shortening_v3_plan.md` |
| `stt_tar` | The Amazing Race S20E01 voiced with Vosk (22.10): another show end to end | Works; the video was kept: `/media/angrybear/new_large_storage/The Amazing Race/STT_Season 20/` |
| `tts_cache` (40 of 45 sets) | Synthesised lines of older engine versions and settings | Only the sets used by the S20 / S29 runs were kept |

Kept in `output/`: the season runs `stt_s20` and `stt_s29_v2`, the S20 reference subtitles
`s20_subs` (CBS captions with `>>` speaker changes, AMZN as `.alt`; `report.md`), the
VoxConverse test data `stt_data` and its diarization outputs `stt_bench/diar_voxconverse`, the
LLM cache and the TTS cache sets in use.

## S20 checks against the reference subtitles (27.10.2026)

- English recognition of whole episodes against the CBS captions (`output/s20_subs/srt/`), all
  15 episodes: fusion 6.5-12.9% of words differ (better than either model in every episode),
  Whisper alone 8.2-15.6%, Cohere alone 8.8-15.8%. Part of it is the captions' own shortening
  and uncaptioned background talk: the final transcript scores a little worse than the fusion
  only because it has more of that talk right ("damn holding these sons of bitches sure makes me
  hungry", not in the captions).
- Speech check of the voice-overs, S20E01-E15 (9933 lines each): ESpeech 1.45% character errors
  (195 lines flagged), Vosk 0.66% (77), TeraTTS 0.55% (54). LLM steps through the ChatGPT
  subscription; the Claude stress votes through OrcaRouter cost ~$0.05 per episode.
- Pauses inside a sentence that split it into cues are real: Whisper decoded no words of the
  phrase in 92 of 99 such gaps where its own word times had put the words side by side.
- ESpeech F5 loses the first word of a line in 5-7% of lines (Vosk / TeraTTS 2.5-3.5%, partly the
  checking ASR); causes and the probe in roadmap §3.

Scripts of these checks (not in git): `output/stt_s20/eval/` (`asr_wer.py`, `split_gaps.py`,
`gap_truth.py`, `first_word.py`, `f5_cut_probe.py`, `f5_prefix_probe.py`).
