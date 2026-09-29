# SH08 zero-capacity original timing check

This is a real local Vosk timing measurement of the frozen SH08 selection. The
adapter selected all 19 rows with `max_chars == 0`, excluded the one row whose
status was `invalid`, and measured the remaining 18 original texts with voice
8 at speech rate 1.5 using punctuation preprocessing.

## Results

- Measured originals: **18/18**
- Actual fits: **1**
- Actual overruns: **17**
- Unverified measurements: **0**
- Invalid zero-capacity rows excluded: **1**

The evidence is in [`summary.json`](../output/sh08_zero_timing_20260915/summary.json).
The existing verifier checkpoint and WAV files are in
[`preflight_audio/`](../output/sh08_zero_timing_20260915/preflight_audio/).

These are duration measurements only. This run did not include listening,
pronunciation review, or semantic audit. A predicted zero character budget is
not treated as proof that an original is impossible; the recommendation here
is based on the measured counts: keep all 17 overruns in timing follow-up and
retain the one measured fit as evidence that the frozen predictor is not an
impossibility test. Do not change the predictor automatically from this run.
