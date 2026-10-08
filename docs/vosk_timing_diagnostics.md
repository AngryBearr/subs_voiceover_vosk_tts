# Vosk timing diagnostics

**Status:** offline read-only investigation; no new paid request, network
request, or TTS invocation was made while writing this document. The copied
source artifacts are preserved under
`output/sh08_timing_investigation_20260913/`.

## Nine-case discrepancy

The existing risk-selected nine-case smoke set contains nine real local Vosk
WAVs. Five fit their original subtitle windows and four overflowed. The current
calibrated planner classified **0/9** as status `fits`; its unguarded character
estimate fit **3/9**, guarded estimate fit **0/9**, and current character budget
fit **0/9**. The legacy 13-CPS comparison classified **9/9** as fits, but that
is a historical comparison, not a current guarantee.

There were **zero** overflows among cases whose current planner status was
`fits`. Thus the result is not “four false-safe errors from the current
planner”: the four measured overflows were already outside its current status
fit set. Conversely, the guarded planner is visibly conservative: several
measured WAVs fit even though their guarded estimate or character budget did
not.

The exact nine-case table and all per-case values are retained in
`source_tmp/timing_diagnostics.md` and `source_tmp/timing_diagnostics.json`;
large arrays are intentionally not repeated here.

## Why the forecast differs from audio

The current voice-8/rate-1.5 calibrated profile uses normalized TTS text and
the character-linear model:

```text
raw_estimate_seconds = 0.5066993855 + 0.05804439976 × normalized_characters
guarded_estimate_seconds = raw_estimate_seconds + 0.6197317843
```

The guard is additive and rate/profile-specific; there is no second “double
rate” operation. The minimum guarded baseline is therefore about 1.12643 s,
so many short windows receive a zero character budget. That forecast is not a
mathematical bound on a short utterance: an actual TTS utterance can still fit,
as the nine measured samples demonstrate. The protected estimate uses
normalized text, not display text.

For a concrete boundary check, source `1a8bee086a0a`, `cue-487` has a 3.254 s
window. Its actual WAV duration is `2.8908843537414968` s, while the guarded
estimate is `2.8677631625818063` s: the actual is 23.121 ms longer and still
fits. This is why the guard is conservative rather than a proof of impossible
audio.

The old 13-CPS approximation used display-text length and a different
intercept/reserve model. The current profile uses normalized text,
calibration-derived intercept and slope, and an explicit guard. Comparing the
old 9/9 status with current guarded 0/9 status as a model-quality result would
mix protocols.

## Frozen 30-case follow-up

Before any audio, the next selection froze 10 cues per source: five nearest the
positive character-budget boundary and five seeded comfortable cases. All 15
boundary cases had zero remaining characters. Fresh local Vosk synthesis then
fit **30/30**, with `false_safe = 0`, zero WAV-header mismatches, and zero
production-evidence disagreements. Source/profile/model identities and
selection remained unchanged; network and paid requests were zero.

This is a known-corpus calibration-overlap check, not an independent global
accuracy claim and not a measurement of all 672 current-fit-pool cues. The
30-case denominator is the primary final measurement; the nine-case result is a
separate risk diagnostic.

Accounting is explicit: four complete 30-case technical runs (unsuffixed,
`r1`, `r2`, `r3`) plus two nine-case preparation runs give **138 local TTS
invocations** in the recorded outputs. They are not 138 independent samples:
the frozen 30-case selectors and durations were repeated. No outcome-based
resampling was used, and the guard/profile were not tuned to meet the 15% goal.
Statistical conservatism still warrants a broader study.

## Operational recommendation

Run the actual original-audio timing preflight before an LLM request, especially
for zero-budget and timing-risk cues. Preserve the actual fit evidence and keep
unresolved counts rather than treating a zero forecast as impossible audio or
silently removing the case from a denominator.

## Durable evidence and hashes

The nine copied diagnostic files retain the exact bytes from the temporary and
`r3` sources. `source_tmp` contains the five timing-diagnostic outputs; its
`verification.log` records exit status 0, 5/9 actual fits, and zero new TTS or
paid calls. `source_r3` contains the four selection/report outputs for the
30-case check; its `verification.log` records 30/30 synthesis success and zero
network attempts.

| Durable file | SHA-256 |
|---|---|
| `source_tmp/command.stderr.txt` | `5c5b88f77e7307dc04a95fe2931380040250c758953365e419f4425fda4bbc61` |
| `source_tmp/command.stdout.txt` | `2744d86e98dfce6c14113328cbc856f0fcb1514c00114fcce3bef4cfdbcfc067` |
| `source_tmp/timing_diagnostics.json` | `9bc571dd4bb23d1220c4f0a6a4ffe3d926b10c851c330d3b45a179f6c7e02eca` |
| `source_tmp/timing_diagnostics.md` | `c4186d717e246054d12dd4aac6eb11cf0512df9c3f9a7d44416a778d218ba481` |
| `source_tmp/verification.log` | `eec12414e7bf996865aad64d3cea07c799983f419b05950daf741eae0fe1bac3` |
| `source_r3/selection.json` | `9766ee60822274cc63d3175dcb6bf88f50d5d3c14517ceb9e142d14c08ef74b7` |
| `source_r3/report.json` | `d67f827d0d898978d7658913f71ac7931c5d473b03a507d9c492efe87065820c` |
| `source_r3/report.md` | `e8e99977af1d44f91a46539a9845320d72b2bc30d80aa868e1f66a7f96b0ba84` |
| `source_r3/verification.log` | `82d3e50e54337e85dbda5606ffbbe9751a2fd49ab410653cf1495818b61c4a76` |

The unchanged SH08 selection manifest remains referenced by the reports at
`docs/sh08_pilot_manifest.json`, SHA-256
`dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`.
