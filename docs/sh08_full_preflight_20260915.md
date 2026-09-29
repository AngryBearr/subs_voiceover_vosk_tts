# SH08 full-file preflight — 2026-09-15

**Status:** `SH08_IN_PROGRESS/PREFLIGHT_ONLY`. These are three offline dry-run
plans, not pilots. No model, network, credentials, dependency installation, or
Vosk synthesis was used. No HTML or request receipts were generated. SH10 is
not implemented.

The machine-readable report is
[`output/sh08_full_preflight_20260915/summary.json`](../output/sh08_full_preflight_20260915/summary.json).
Each output directory contains only the CLI's `preflight.json`.

## Frozen inputs

All three source SHA-256 values matched `docs/sh08_pilot_manifest.json`; the
frozen profile SHA-256 also matched:
`eab509922c511ed0dbd2b3948d5a0cf55de78dc11ab6127e9c59443f4cabcb59`.
The source files and profile were not changed.

| Output | Source | Total | Problematic | Positive budget | Zero budget | Invalid |
|---|---|---:|---:|---:|---:|---:|
| `pilot-1` | Australian Survivor S01E03_ru.srt | 1523 | 1219 | 875 | 344 | 0 |
| `pilot-2` | S01E02.srt | 1007 | 773 | 525 | 248 | 0 |
| `pilot-3` | Survivor.S44E07.srt | 995 | 861 | 681 | 178 | 2 |
| **Total** | 3 frozen files | **3525** | **2853** | **2081** | **770** | **2** |

Problematic includes all non-fitting plans and invalid plans. Positive-budget
means `max_chars > 0`; zero-budget excludes the two invalid plans.
The CLI stdout selected counts were 1219, 773, and 859 respectively; the third
file's problematic count is 861 because its two invalid plans are retained in
the denominator but are not selected for shortening.

## Request estimates

Counts below are from the existing `make_batches` planner with batch sizes 6
and 12 and a 20,000-character context limit. The current runner's initial
generation and review queues contain only `needs_shortening` plans, so they are
shown separately from the full problematic denominator. No context-limit split
occurred beyond ordinary batch-size splitting.

| File | Runner generation targets / batches | Runner review targets / batches | Positive-budget runner targets / review batches |
|---|---:|---:|---:|
| `pilot-1` | 814 / 136 | 814 / 68 | 743 / 62 |
| `pilot-2` | 477 / 80 | 477 / 40 | 433 / 37 |
| `pilot-3` | 607 / 102 | 607 / 51 | 575 / 48 |
| **Total** | **1898 / 318** | **1898 / 159** | **1751 / 147** |

The 2,081 positive-budget cases include `short_timing` cases that are not all
sent to the runner. After filtering to 1,751 `needs_shortening` review
candidates, the table's planner derives 147 planned review batches / 147
one-unit floor units. This is the **planned initial review estimate before
repairs, without further measured-audio pruning**, not a guaranteed actual
minimum: errors, `cannot_shorten` outcomes, and local timing may change the
actual calls. The existing protocol's conservative accounting is six
subscription attempt units per logical prompt, which would be **882 units** for
those reviews alone. This is already beyond the remaining overall 168 units and
protected pilot allocation of 150 units under conservative accounting;
generation and recovery would require more.
No dollar estimate is asserted for subscription use. The current comparison
record remains `$0.03515893128` and 132 subscription units, leaving
`$4.96484106872` and 168 overall units; the protected pilot allocation remains
`$2` / 150 units without reassignment.

## Blockers and claim boundary

The CLI invocation used `openai/gpt-5.6-sol` / `medium` only as the required
offline-compatible placeholder label. It does **not** establish Sol as the
chosen generator or arbiter, and no generator ran. The full CLI currently has
no Muse adapter or mixed-role generator/arbiter bridge and has no hard spend or
quota guard for this pilot. Therefore no live run was launched.

The plans are feasibility evidence only. Dry-run mode does not synthesize
audio, prune by actual Vosk output, call an LLM, or produce semantic judgments.
SH08 remains incomplete; SH10 remains unimplemented.
