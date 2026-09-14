# SH08 model test notes

**Status: approved; r1 failed smoke, r2 retry not executed.** This note records
the user's approvals and corrects earlier model-availability claims. It records
no paid request; r1's one OpenCode subscription attempt is retained as unknown.

## Corrections and consent

- Gemini and Muse were previously described as unfound; they are now confirmed
  by the user's cards, URLs, endpoint links, and actual public HTTP endpoint
  lookups in this session.
- The user approved the addition of Muse as a generator and possible product
  improvement use. This does **not** claim confirmed training on every prompt.
- Subtitle excerpts may be transferred under provider terms. The approved
  source file is text context only: no keys and no code.
- The user confirmed ordinary OpenRouter balance without BYOK.
- No independent audit model is approved. Muse is a generator, not an
  annotator, and no generator agreement is independent truth.

## Planned pair matrix

| Pair | Rationale | Status |
|---|---|---|
| Luna → Luna | Core same-family comparison | planned |
| Luna → DeepSeek | Core cross-family arbiter | planned |
| Luna → Sol | Core OpenCode-family arbiter | planned |
| DeepSeek → Luna | Core cross-family arbiter | planned |
| DeepSeek → DeepSeek | Core same-family comparison | planned |
| DeepSeek → Sol | Core OpenCode-family arbiter | planned |
| Gemini → Sol | Cheap external generator against Sol | planned |
| Muse → Sol | Cheap fourth generator suggestion against Sol | planned |

Self-pairs are useful for consistency checks, not independent truth. Gemini is
third generator; Muse is treated as cheap fourth generator, not as an
independent annotator.

## Models and prices

The authoritative pure specification is `utils/sh08_matrix_models.py`:

| Model | Variant | Planned price / endpoint |
|---|---|---|
| `openai/gpt-5.6-luna` | medium | unpriced subscription |
| `deepseek-flash` | off | DeepSeek peak: `$0.30` / `$1.20` per million input/output tokens |
| `openai/gpt-5.6-sol` | medium | unpriced subscription |
| `google/gemini-3.8-flash` | low | sync Flex `google-ai-studio/flex`: `$0.375` / `$1.875` |
| `meta/muse-spark-1.3-contributor` | low | mandatory reasoning: `$0.10` / `$0.20` |

Gemini standard pricing is `$0.75` / `$3.75` per million input/output tokens;
batch is `$0.375` / `$1.875`. The chosen sync Flex price is exactly the batch
price, so there is no day wait. Fallbacks are disabled and no async batch was
executed.

For the Gemini Flex example with 100k input and 20k output tokens, the
total is `$0.075`; the same example at standard pricing totals `$0.15`. The
Muse example is `$0.014` (`$0.01 + $0.004`). These are price calculations, not
invoices or executed costs.

Known pricing source URLs as of this session are preserved in the JSON record;
the user cards were the main verification source. No online pricing fetch is
required.

## Protocol boundaries

- Scope is 60 unique development cases, two repetitions, and eight pairs; no
  full pilots.
- The first six frozen cases in repetition 1 are smoke only. No extra cases are
  added there.
- Complete both repetitions' first-pass coverage before at most one branch
  repair and one recheck. Fixed batches are `6 + 27 + 27`, chosen to fit quota,
  not to randomize model changes.
- Select from seven test files, excluding pilot raw/canonical files and S44E11.
  Rank with SHA-256 salt `sh08-matrix-v1:20260914:`. Include all intervention
  cases, including zero-capacity cases; do not send LLM requests for zeros.
  Unknowns remain unresolved. Selection is not frozen until preparation.
- Use voice 8 at rate 1.5 and the current profile referenced, with its SHA,
  through `docs/sh08_pilot_manifest.json`.
- Tools and extra automatic retries are disabled. OpenCode's internal five
  retries mean six permanent quota units per prompt.
- Limits for the entire experiment, including restarts, are `$5` direct API,
  300 conservative subscription attempt units, 600 total attempt units, and
  16,384 maximum output tokens. Limits do not auto-increase.
- Subscription accounting counts logical prompts against the six-attempt
  upper bound; it does not claim exact physical visibility. Primary first-pass
  coverage is reported separately from repairs, which may be budget-limited.
- Reserve full context plus max output times `1.10`, compared with actual
  OpenRouter-reported cost and conservative no-cache DeepSeek cost. This is
  neither an expectation nor an invoice.

## Execution record

The r1 result is **unknown** and the r2 retry is **not yet executed**. Local
preparation is implemented and requires verification before execution. The
planned r2 output is `output/sh08_eightpair_20260914_r2`; it has not been
created. The user's approval covers these limits for the entire experiment,
including restarts.
Latest local checks recorded 531 passed and 5 skipped before the zero-count
regression, plus 61 targeted checks after it. The real offline record is 60
cases: 41 eligible, 18 valid, 0 zero, and 1 invalid. One real r1 model call
occurred; fresh r2 live output has not been created. These test
simulations do not measure model quality. The user previously explicitly
approved commits and follow-up commits, but push is not approved.

## Recovery revision

- The r1 smoke call was an actual failed smoke: one Luna call returned
  `usage_reasoning_exceeds_output`. The underlying call succeeded, but the
  response text was not retained; the lost text is not recoverable. Local
  recovery matched zero sessions.
- The first model call consumed six subscription quota units. This was a
  local OpenCode token-accounting failure, not a paid API request and not a
  free-cost claim. All other model spending remains part of the approved
  matrix pricing notes above; no other model calls occurred in r1.
- The user explicitly approved one Luna retry in the question. The retry is
  `new_revision_not_resume`, with the same source, case selection, variant,
  profile, and `$5`/`300`/`600` limits. Carryover is six subscription units,
  six total units, one logical prompt, and `$0` direct API cost. The new
  revision is planned at `output/sh08_eightpair_20260914_r2` and has not been
  created. The immutable r1 output remains at
  `output/sh08_eightpair_20260914_r1`.
- Preliminary quota forecast is base `252` plus six carried units = `258`,
  leaving `42` total units and seven subscription prompts after the first
  pass. The `42` figure uses the actual `542 passed, 5 skipped` smoke check;
  the first model call itself consumed six units.
- Latest carryover verification: `549 passed, 5 skipped`.

## Durable r2 receipt import

- The first repeat was done in the saved r2 artifact: 55 logical requests are
  present, with 54 complete rows and one locally reconciled Sol row. The second
  repeat was blocked only by local total/cache accounting; its response text and
  raw response snapshot are preserved exactly.
- No server retry or paid repeat was made. The planned r3 revision will import
  the 55 saved responses as durable receipts; all r1 and r2 files remain
  unchanged.
- The definitive latest r2 accounting is 56 cumulative logical prompts, 55
  revision prompts, 216 total quota units, and 192 subscription units. The
  known monetary sum is `$0.06299181375`: OpenRouter `$0.02588961375` plus the
  conservative DeepSeek upper bound `$0.0371022`; the reconciled Sol response
  contributes `$0` because subscription use is not priced.
- This record makes no final-winner claim and does not describe the imported
  response as a second repetition complete result.
