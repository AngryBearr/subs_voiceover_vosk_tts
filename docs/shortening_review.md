# Shortening review and reanalysis guide

This guide documents the SH-05/SH-11 offline review artifacts in a completed
SH-04-compatible run. It does not run an LLM, TTS, paid pilot, or actual timing
measurement, and it makes no final semantic-quality guarantee.

The durable real-corpus technical acceptance is recorded in
[`shortening_real_acceptance.md`](shortening_real_acceptance.md); it is not a
semantic-quality result or SH-08 pilot.

## What a run publishes

Each new live run checkpoints:

- `outcomes.json` and `outcomes.md`: all cues, with `text_changed` and the
  independent outcome/evidence status. Statuses distinguish unchanged-and-fits,
  accepted correction, needs attention/unresolved, invalid/error, and not
  checked/unverified.
- `review.json` and `review.md`: only unresolved or attention-needed cues, with
  original/current text and current neighbor context, candidates, budget and
  actual length, reason, latest model critique, and proposed action. Original
  validation context is tracked separately in the outcomes artifacts.
- `review_state.json`: a checksummed full-source/current/config/plans/records
  snapshot with text, timing, profile, configuration, context, and evidence
  fingerprints. It is the binding state for the generated queue.

Legacy budget and forecast fields are retained for compatibility and are not
approval markers. Legacy `report.summary` stage/verdict counts remain separate
from `report.outcome_summary`; the latter is authoritative for the current
per-cue queue. The initial intervention denominator is fixed. Regressions that
appear later are counted separately and never hidden by changing that
denominator.

Semantic model or human acceptance is not measured audio fit. A changed cue is
not thereby solved. `measured_duration_sec` and `measured_fit` remain null when
the measurement is not valid for the current source/text/timing/profile/config.
A fresh model verdict concerns the ORIGINAL neighbor context and records
`final_context_checked: false`; SH-10's final-context evaluator is not
implemented. A later neighbor-window change invalidates semantic skip/reuse, so
no final-context validation is claimed. Prediction fits are estimates only.

## Applying an approved queue

Start from the generated queue and preserve it for comparison:

```bash
cp RUN_DIR/review.json edited-review.json
```

Edit only `approved`, `decision`, and `replacement_text`. To apply an edit,
use `approved=true` with `decision=replace` or `decision=keep`. `defer` and
`approved=false` are never applied. A `keep` entry must have replacement text
equal to the current text and preserves list fragments exactly. Do not hand-edit
immutable IDs, source/current hashes, contexts, budgets, timings, profile, or
configuration.

Run the application into a new directory outside the input run:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.apply_review RUN_DIR \
  --review edited-review.json \
  --output-dir output/review-applied
```

An unknown or duplicate ID, source/current/profile/config mismatch, immutable
context or budget change, stale snapshot, over-budget or newline replacement,
or invalid timing rejects the whole queue before output creation. Invalid text
and invalid-timing cues cannot be approved through this text queue. The output
directory must be new and outside the input directory. A nonblocking
`.run.lock` protects the input run.

The frozen source, profile, and model file hashes are checked offline, and no
profile override is accepted. If the source was moved, pass an optional
`--source MOVED_INPUT.srt` only when it has the same bytes as the frozen source.

The primary export changes only explicitly approved text; string/list field
representation, source indexes, timings, unknown fields, and all other cues
are preserved. Review metadata stays outside spoken text, so no marker strings
enter TTS. The derived `application_report.json` records a UTC action timestamp
but does not measure human work time. It has no SH-04 run manifest and is not a
resumable LLM run. Further applications and reanalysis are supported from
derived bundles. If a derived write fails partway, retain the destination,
inspect it, and choose another fresh output directory; there is no automatic
deletion or overwrite.

Adjacent batch edits can invalidate context previously displayed in a queue.
Such a cue remains attention-needed until a later explicit confirmation on the
updated queue; it is never silently marked solved. Correct invalid source,
text, or timings externally and start a new run. Manual application is not a
timing editor, and there is no over-budget revert bypass.

## Offline reanalysis

Reanalyze an existing bundle without LLM, TTS, or actual audio review:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.review_subtitles RUN_DIR \
  --output-dir output/review-rechecked \
  --force-recheck
```

`--current-json`, when supplied, must be a complete JSON array with the same
cue count and positional `index` values. External changes never auto-approve a
cue. Timing or configuration changes recompute budgets and current context
windows. `--force-recheck` invalidates cached model, human, and measured
evidence, while always retaining the lightweight prediction. Reanalysis only
marks required work in the generated queue: it does not call an LLM or TTS and
does not actually measure or perform semantic review. Its derived
`reanalysis_report.json` is not a copy of a paid-request journal.

The frozen source bytes are still required and checked. `--force-recheck` is
not a migration override.

## Resume and recovery boundaries

Resume compatibility hashes the implementation intentionally. Upgrading hashed
implementation code invalidates older SH-04-only in-flight runs; finish a run
using its original revision before upgrading. Preserve existing directories and
never claim automatic migration. Legacy directories without `review_state.json`
are refused.

An edited generated `review.json` blocks resume before it can be clobbered.
Copy or move those edits to an external file first. An interrupted generated
sidecar publication is recoverable only when the runner's trusted reconstructed
queue matches it exactly; user edits are never discarded. Invalid SRT export is
reported as `export_error` with exit 1 while JSON and audit artifacts are
preserved; rows are never silently dropped.

## Next authorized work

SH-08 is the next technical pilot and requires separate paid authorization. No
live pilot is recorded here. The real offline acceptance record does not change
that status. BUG-02 and BUG-03 remain deferred and untouched; recheck them
before any paid run.
