# SH08 eight-pair r3 results

Offline evidence analysis of two first-pass repetitions across eight pairs. No model judgment was added by this analysis.

## Scope and status

- 60 frozen cases, two repetitions, and 16 pair/repetition tables (eight pairs in each repetition); each table uses denominator 60, not a combined denominator of 120.
- Each table has 41 eligible cases. `Not executed=19` is structural: 18 valid zero-capacity cases plus 1 invalid source case; unexecuted eligible cases are 0.
- The raw execution status remains `blocked_unknown`. Offline evidence derives termination as `budget_exhausted` from the exact 300 subscription units and recorded `BudgetLimited` evidence; this does not rewrite raw status.
- `firstpass_execution_finished=true`: all planned generation/review receipts and explicit terminal skips are represented. The five invalid model rows are review parse failures, not unfinished model calls.

## First pass

| Pair/rep | Accept | Decline | Cannot shorten | Invalid review rows | Not executed |
|---|---:|---:|---:|---:|---:|
| luna->luna / 1 | 23 | 9 | 9 | 0 | 19 |
| luna->luna / 2 | 21 | 11 | 9 | 0 | 19 |
| luna->deepseek / 1 | 17 | 11 | 9 | 4 | 19 |
| luna->deepseek / 2 | 15 | 16 | 9 | 1 | 19 |
| luna->sol / 1 | 19 | 13 | 9 | 0 | 19 |
| luna->sol / 2 | 21 | 11 | 9 | 0 | 19 |
| deepseek->luna / 1 | 8 | 33 | 0 | 0 | 19 |
| deepseek->luna / 2 | 8 | 33 | 0 | 0 | 19 |
| deepseek->deepseek / 1 | 8 | 33 | 0 | 0 | 19 |
| deepseek->deepseek / 2 | 8 | 33 | 0 | 0 | 19 |
| deepseek->sol / 1 | 7 | 34 | 0 | 0 | 19 |
| deepseek->sol / 2 | 8 | 33 | 0 | 0 | 19 |
| gemini->sol / 1 | 19 | 9 | 13 | 0 | 19 |
| gemini->sol / 2 | 16 | 5 | 20 | 0 | 19 |
| muse->sol / 1 | 21 | 13 | 7 | 0 | 19 |
| muse->sol / 2 | 23 | 7 | 11 | 0 | 19 |

Accept means arbiter verdict `accept` and current local normalized-length fit; a raw verdict alone is not counted as acceptance. Decline combines complete `revise` and `impossible`. The five review parse errors are separately mapped to Luna generator -> DeepSeek arbiter (4 in repetition 1, 1 in repetition 2).

## Repairs and diagnostics

Repair eligibility is the valid first-pass candidate with a complete `accept` or `revise` verdict and `accepted=false`. Ledger prompt targets, not branch receipt references, determine actual repair and recheck attempts. Across all 16 tables: eligible 284 = attempted 51 + not-attempted 233; not-attempted is budget 233 + not-run 0; actual rechecks are 13 = successful 6 + failed 7; pending 0. Raw branch counters are retained only as metadata, not receipt counts.

The remaining seven subscription prompts were consumed by: deepseek->luna, deepseek->sol, luna->deepseek, luna->luna, luna->sol, muse->sol. The subscription cap reached exactly 300 units; paid repair requests after that cap are legitimate recorded calls, not a request to make new subscription calls.

Generator normalized-length facts from the frozen source: DeepSeek is overlength in 33/41 cases in each repetition; Luna in 1/41 in each; Muse in 1/41 in each; Gemini in 2/41 in repetition 1 and 1/41 in repetition 2. Generation schema errors are counted only on generation branches and are zero for every generator/repetition; the five review parse errors remain under first-pass review errors.

Same-first-candidate three-judge comparisons apply only to Luna and DeepSeek, which have three judges. Gemini and Muse are not applicable, rather than being reported as 41 missing-judge cases. No winner is declared.

## Costs and time

Known priced/actual total: `$0.1013308035`. Logical total is 89 including the prior r1 prompt; subscription quota is 300/300; paid calls are 39. Receipt elapsed metrics exclude offline analysis/debug gaps; the lost r1 unknown receipt is listed separately at 21.98998145799851 seconds.

Gemini and Muse provider/model metadata were validated from receipts. Async batch was not executed. Subscription entries remain unpriced; zero is not an invoice or a cache-saving claim.

## Interpretation boundary

Accept is an arbiter/fit result, not semantic truth. No audio, full-pilot, independent-human, or independent-audit claim is made.

Protocol and model notes: [sh08_live_protocol.json](sh08_live_protocol.json), [sh08_model_test_notes.md](sh08_model_test_notes.md). Raw artifacts and hashes are listed in `analysis/report.json`.

The examples JSON is a deterministic read-only inspection sample with no manual semantic filter and no independent audit label.