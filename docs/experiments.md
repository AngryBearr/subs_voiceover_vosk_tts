# Experiments and what came of them

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

- English recognition of whole episodes against the CBS captions (`output/s20_subs/srt/`):
  fusion 11.6-12.1% of words differ, Whisper alone 13.0-13.4%, Cohere alone 13.9-15.0%
  (E01-E03). Part of it is the captions' own shortening and uncaptioned background talk: the
  final transcript scores a little worse than the fusion only because it has more of that talk
  right ("damn holding these sons of bitches sure makes me hungry", not in the captions).
- Pauses inside a sentence that split it into cues are real: Whisper decoded no words of the
  phrase in 92 of 99 such gaps where its own word times had put the words side by side.
- ESpeech F5 loses the first word of a line in 5-7% of lines (Vosk / TeraTTS 2.5-3.5%, partly the
  checking ASR); causes and the probe in roadmap §3.

Scripts of these checks (not in git): `output/stt_s20/eval/` (`asr_wer.py`, `split_gaps.py`,
`gap_truth.py`, `first_word.py`, `f5_cut_probe.py`, `f5_prefix_probe.py`).
