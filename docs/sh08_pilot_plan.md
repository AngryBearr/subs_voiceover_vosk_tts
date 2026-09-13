# SH08 pilot preparation

**State:** `SH08_IN_PROGRESS/PREPARATION_ONLY`. The paid pilot is not completed and no LLM/backend/server was called. Historical model-pair facts and the unexecuted six-pair proposal are in [`shortening_model_pairs.md`](shortening_model_pairs.md) and [`sh08_model_matrix_proposal.json`](sh08_model_matrix_proposal.json); the Vosk discrepancy is documented in [`vosk_timing_diagnostics.md`](vosk_timing_diagnostics.md).

## Frozen candidate and configuration

- Output root: `/home/vladimir-barbuta/Projects/subs_voiceover_vosk_tts/output/sh08_preparation_20260913_r1`
- Three source files, their SHA-256 hashes, and cue counts are frozen in `sh08_pilot_manifest.json`.
- Vosk model: `models/vosk-model-tts-ru-0.10-multi`, voice 8, rate 1.5, punctuation mode.
- Frozen profile: `output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json`; no calibration or prompts were changed.
- Run parameters: generation batch 6, review batch 12, repair budget 1, context maximum 20,000; output-token bound is not chosen.

## Claim boundary

SH05 acceptance covered 4,419 cases without synthesis. This SH08 preparation instead contains exactly nine real local Vosk WAV syntheses (three selected cues per source), with 9/9 synthesis success, 5/9 original-audio fits, and 4/9 overflows. It made no LLM calls and no funded calls. This is a small smoke sample, not three whole processed files and not a 15% goal result.

## Offline preflight and local smoke

Each source has a complete `offlinepreflight.json` containing prepared plans, stable cue IDs, selected/queue IDs, timing-risk, zero-budget, repetition, eligibility, and text-flag counts. The local smoke selects at most three source-order risk cues: zero-budget risk, non-duplicate repetition risk, then positive-budget risk. Missing categories are recorded as `N/A`; no case is invented. Nine WAV headers are present at the expected 22,050 Hz sample rate; this is not a re-synthesis claim.

The smoke calls the existing `utils.shortening_audio.verify_original_timing` API with the exact frozen profile/model parameters. It records selected IDs, original-text hashes, timing windows, profile identity before synthesis, full returned evidence, WAVs, soundfile metadata, and worker stdout/stderr. It is technical readiness evidence only, not TTS accuracy or semantic validation.

## Guard and authorization boundary

The parent patches `LiveBackend.__call__`, `_LazyRequest._make_backend`, and `CleanOpenCodeServer.start`; the child interpreter loads a temporary `sitecustomize.py` that blocks and logs socket operations. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set. Any unexpected model/server callback fails closed. Paid requests, spend/quota caps, and audit authorization remain zero/pending.

The model label `deepseek/deepseek-flash` is configuration provenance only. Generator and arbiter are recorded as the same chosen model by SH03, while an independent-family checker remains a separate pending choice. No paid model selection can pass authorization in this phase.

## Eligibility, audit, and limitations

Eligibility evidence is recorded from the first three sample case IDs in the expanded pilot manifest, the 48-case common validation artifact, and prior corpus manifests used only for measurement/calibration. The correct bounded claim is: **no selected LLM instruction-tuning cases found in audited artifacts; global exclusion ledger absent**. This is not absolute proof. Survivor.S44E11 is excluded as the fourth historical source. TTS/profile-fit overlap is acknowledged.

The predeclared audit ranks original intervention cue IDs using `SHA256('sh08-independent-audit-v1|' + sourceSHA + '|' + cueID)`, then takes the first 24 text-changed cues per file after an unchanged full final output, auditing all when fewer exist and recording insufficiency. Unchanged cues do not count as checked; no prior verdict, confidence, provider identity, or history is exposed to the checker; this is not a full SH10 evaluator claim. The ordinary-run CLI bridge is missing and must be implemented and verified before any paid execution; legacy `shortening_replay` is not silently substituted.

Historical `$0.5` is not reused as a budget. Pricing is reference-only. Nominal batch counts are not strict bounds because pruning, context splitting, and recovery affect requests. Zero-budget non-short-risk cases are reported separately as potentially futile cost; policy remains pending and unresolved denominator is retained.

A zero forecast character cap does not make actual audio fitting impossible: one of the three zero-budget smoke samples fits. Preserve the actual preflight fit result, then retain unresolved zero-budget cases for policy review rather than assuming that their audio cannot fit.

Deferred BUG02/BUG03 remain unchanged: ordinary `shorten_file` does not call legacy execute, so they are not reachable on that route. Recheck before legacy use. No 15% goal claim or three-full-real-runs claim is made.

## Questions after preparation

1. Confirm same-model generator/arbiter versus the separate independent-family checker.
2. Authorize concrete spend and quota caps, with a hard guard (currently unavailable in the CLI).
3. Decide the safe policy for genuinely nonempty zero-character-budget cases.

The model-matrix choice remains a proposal only: no three-file six-pair matrix
or paid run is authorized. The timing investigation is a separate offline
diagnostic and does not alter the frozen manifest or profile.
