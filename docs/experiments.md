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

Tried and dropped:

| What | Why |
|---|---|
| FireRedTTS3 for the cloned voices | reads Russian with English sounds from English clips (75% character errors), no length control |
| Seed-VC: TeraTTS converted into the heroes' voices | a strong English accent by ear (the user) |
| Accent-free cloning in general | dropped by the user: stress and pronunciation matter more than a slight accent |
| DiariZen on the Demucs vocal stem | no gain over the plain audio |
| TeraTTS teacher sampler | no change (0.49% vs 0.52%, same lost words and speed) |
| Moving lines by the season voice bank | S10 names 63.6% before and after; 2 of 4 labelled moves wrong |
| Trusting a line's own voice over its speaker's known gender | the gender model is often the one that is wrong (shouting men heard as women) |
| Splitting a cluster into sub-voices | lost accuracy on S10 (narration and on-location voice split) |
| The season cast from the web as naming evidence | no gain in names (kept for one spelling per person) |
| Qwen3-ASR or MOSS-Transcribe-Diarize as a third fusion input | within the LLM's run-to-run spread |
| Stress marks on one-syllable names | no change: short names like «Вэл» are misheard by the checking ASR, not misspoken |
| Active speaker detection + faces | dropped by the user before testing (a dead end) |
| Several clips per person by the line's mood (ESpeech) | dropped by the user: one voice per person is enough while some clusters still mix people |

Not done yet (see `docs/roadmap.md`): one stress per name for the whole season, Vosk in parallel
with the GPU engines, a separate voice for the host in the Vosk and TeraTTS tracks, re-voicing
S20 and S29 with all of the above, finishing S29 (E11, E13-E15, missing tracks of E01-E08), a
clean-machine test of `scripts/install.sh`, Vosk TTS 0.11 when it is out, RuVoice stress placer,
Shmyrev's TTS metrics, words at speaker changes without a pause, newer diarization and
speaker-embedding models, S20E02/E03 in Courtney's and Danielle's own voices.

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
