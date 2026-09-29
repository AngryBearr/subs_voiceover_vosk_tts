# SH08 Muse reasoning comparison — 2026-09-15

## Result and claim boundary

This record compares the two actual, completed low/high Muse runs over the same
60 selected cases, with two repetitions per arm. The repetitions are matched
reuses of 60 cases, not 120 independent cases. The reproducible machine summary
is [`summary.json`](../output/sh08_muse_reasoning_comparison_20260915/summary.json).
The comparison itself is complete; SH08 is not complete because this is not the
three-full-file pilot.

The duplicate analyses only read saved `case_results.json`, `ledger.json`, and
`selection.json`; they made no model, network, Vosk, or TTS calls.

The full denominator is 60 cases in every repetition: 41 eligible positive-
denominator cases, 18 zero-capacity cases, and 1 invalid case. The initial and
final values below are independently derived from the per-case rows in each
`case_results.json`: accepted **and** locally fitting results, divided by 41.
They are not the aggregate legacy `accepted` labels in `report.json`.

| Arm | Rep 1 initial → final | Rep 2 initial → final | Aggregate initial → final |
|---|---:|---:|---:|
| Muse low → Sol medium | 20/41 → 28/41 | 22/41 → 28/41 | 42/82 → 56/82 |
| Muse high → Sol medium | 27/41 → 29/41 | 25/41 → 28/41 | 52/82 → 57/82 |

Relative to low, high has **+10/82 initial** accepted-and-fit results and
**+1/82 final** result. The sample is insufficient for a universal ranking.
Muse low remains the default: high costs more and takes longer by the saved
serialized receipt measure.

## Execution identity

- Matrix/protocol: `sh08-matrix-v1`, `sh08_followup_protocol.v1`.
- Prompt payload: `shortening-v4`; the updated prompt and anti-repeat rule were
  used in both arms.
- Generator: `meta/muse-spark-1.3-contributor` through OpenRouter.
- Arbiter: `openai/gpt-5.6-sol`, `medium`, through the OpenCode subscription.
- Requested Muse modes were `low` and `high`, as recorded in the manifests.
  Receipt call records contain `requested_variant=low` or `high` respectively;
  this is evidence of the requested transport mode, not proof of hidden
  provider reasoning behavior. Sol receipt records contain `medium`.
- OpenCode compatibility was checked at version `1.18.31`.
- The low arm's first local preflight was blocked by the OpenCode version pin;
  it made no model calls, preserved the originals, and was refrozen for the
  validated OpenCode `1.18.31`. This is separate from the three full-file
  preflight dry-runs below, which were also not pilots.

## Repairs and duplicate analysis

Repair counts are operational only. They mean attempted and rechecked requests,
not semantic resolution; the report's `resolved` label is not used as a
semantic claim.

| Arm | Rep | Repairs attempted | Rechecked |
|---|---:|---:|---:|
| low | 1 | 12 | 12 |
| low | 2 | 11 | 11 |
| high | 1 | 6 | 5 |
| high | 2 | 6 | 6 |

The one high rep1 `cannot_shorten` result is a model unresolved result, not
infrastructure partiality. The saved-response duplicate CLI was run afresh on
each actual run, producing new reports in
`output/sh08_muse_reasoning_comparison_20260915/{low_duplicates,high_duplicates}`.
All 24 low-arm and 20 high-arm ledger requests completed successfully; the
high-arm partiality is the one model-unresolved result noted above, not a
transport failure.

| Arm | Source requests | Current parsed-ok intersection | Exact duplicates | Whitespace-normalized duplicates | Duplicate repair targets | Proven avoidable rechecks |
|---|---:|---:|---:|---:|---:|---:|
| low | 24 | 32 | 5 | 5 | 0 | 0 |
| high | 20 | 31 | 8 | 8 | 0 | 0 |

These current denominators are derived from the actual low/high receipts. The
historical eight-pair artifact's 59 duplicate pairs out of 121 independent-ok
intersections is **not** the current comparison and is not reused as a current
count. No avoidable-call saving is claimed.

The numeric representation guard is present in the saved prompt: spelled-out
numbers remain words, while numbers already written as digits may remain digits.
No explicit arbiter-reported representation violation was found in these saved
verdicts, but a zero-violation rate was not independently verified. Historical
runs were not rescored.

## Cost, quota, tokens, and serialized latency

Known API costs are reported separately from the unpriced subscription:

| Arm | Known API cost | Subscription units | Serialized receipt sum |
|---|---:|---:|---:|
| low | `$0.01298938014` | 72 | 672.478167530011 s |
| high | `$0.02216955114` | 60 | 925.859101473005 s |
| **total** | **`$0.03515893128`** | **132 / 300** | — |

The remaining known dollar budget is `$4.96484106872`, and 168 subscription
units remain. The `$1.50` / 75-unit per-arm figures are split reservations,
not targets spent. Subscription use is unpriced.

Serialized latency is the sum of saved `elapsed_sec` values for generation,
review, repair, and recheck receipts. It is not wall-clock elapsed time and does
not promise end-to-end latency. High exceeds low by 253.380933942994 seconds on
this measure.

Provider token counters remain separate because provider semantics may differ and
reasoning may be included or reported separately. They must not be summed into
output tokens blindly:

| Arm / provider | Prompt | Completion | Total | Reported reasoning |
|---|---:|---:|---:|---:|
| low / OpenRouter Muse | 74,398 | 37,539 | 111,937 | 33,758 |
| high / OpenRouter Muse | 69,025 | 86,590 | 155,615 | 82,632 |
| lowSol / OpenCode Sol | 76,810 | 4,065 | 89,686 | 8,811 |
| highSol / OpenCode Sol | 71,423 | 3,173 | 80,603 | 6,007 |

The counters above use the canonical `input_tokens`, `output_tokens`,
`total_tokens`, and `reasoning_tokens` fields from one successful receipt call
per request; duplicated response snapshots are not counted. OpenRouter's
reported reasoning is included in its `output_tokens`/`total_tokens`, while the
OpenCode receipts report reasoning separately and include it in their reported
`total_tokens`; reasoning is therefore not added again where it is already
included. All 12 lowSol and 10 highSol requests had complete usage coverage.

## Quality and follow-up boundaries

There was no audio measurement or separate blinded semantic audit/control of old
versus new. Both arms used the same new rules and anti-repeat prompt. No semantic
quality guarantee follows from these verdicts or duplicate counts.

The zero-capacity local Vosk follow-up remains separate: **18 measured, 1 fit,
17 overruns**. It provides duration evidence only, with no sound or meaning
guarantee; see [`sh08_zero_timing_20260915.md`](sh08_zero_timing_20260915.md).
The saved-response scan details are in
[`sh08_duplicate_analysis_20260915.md`](sh08_duplicate_analysis_20260915.md).

The actual standalone reports are available here; no offline comparison HTML was
generated:

- [Muse low standalone report](../output/sh08_muse_reasoning_20260915_low/shortening.html)
- [Muse high standalone report](../output/sh08_muse_reasoning_20260915_high/shortening.html)

## Active-plan status

- **SH08 remains unchecked.** The three preflights were not pilots. Their 2,081
  positive-budget cases include `short_timing` cases that are not all sent to
  the runner; after filtering, 1,751 `needs_shortening` targets derive 147
  planned review batches / 882 conservative subscription units. This is the
  **planned initial review estimate before repairs, without further measured-audio
  pruning**, not a guaranteed actual minimum: errors, `cannot_shorten` outcomes,
  and local timing may change the actual calls. It is already beyond the
  remaining 168 units of the 300-unit comparison budget; the protected pilot
  allocation remains 150 units. The full-file adapter still lacks a Muse bridge
  and a hard spend/quota guard. This comparison does not close that blocker or
  authorize more calls.
- **SH10 remains unchecked and unimplemented.** No full final-context audit was
  performed; it awaits full-pilot input and a separate evaluator implementation.
  No false closure is recorded.
- **SH07 remains partial.** The snapshot/verification record is linked in
  [`sh08_archive_20260915.md`](sh08_archive_20260915.md). Planned migration,
  deletion, cleanup, and retention policy were not performed.
