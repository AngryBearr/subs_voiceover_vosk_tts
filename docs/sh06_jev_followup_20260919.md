# SH-06 Jev follow-up: offline comparison, 19 September 2026

This is a private, offline analysis of the saved follow-up artifacts. It made no
model calls and makes no accuracy, gold, or deployment claim. The machine-readable
record is [`analysis.json`](../output/jev_followup_20260919/analysis.json). The
real-run HTML report is
[`shortening.html`](../output/jev_followup_20260919/shortening.html).

## Scope and execution

- Original replay: 24 cases, 48 complete Jev responses. The follow-up slice is
  six already-known cases, with 12 valid original responses.
- Follow-up execution: 58 attempts = 12 repeats + 23 isolated Jev safety calls +
  23 DeepSeek reviews; 56 complete, 2 errors, 0 pending.
- Provider accounting: Jev 35 attempted / 33 complete / 2 errors; DeepSeek 23 /
  23 / 0.
- The report is `partial`; its campaign status is
  `completed_with_invalid_response`, `clean_complete=false`, and remaining work
  is zero. The three requested parts executed 58 attempts: 56 valid, 2 invalid,
  and 0 pending. The partial status reflects retained invalid responses, not
  unfinished network work.
- The two invalid rows are both `bec0de4b2abc:cue-679`. They are retained as
  attempt 1, were never replayed, and are excluded as whole responses from all
  valid-repeat safety statistics. In order 1, the raw best choice was valid:
  `c5` had probability `0.31`; the offending question was `safety_c3`, whose
  choice was `reject` with probabilities `reject 0.49`, `accept 0.50`, and
  `insufficient_context 0.01`. In order 2, the raw best choice was also valid:
  `c4` had probability `0.39`; the offending question was `safety_c5`, with the
  same `reject 0.49`, `accept 0.50`, `insufficient_context 0.01` probabilities.
  The strict parser tolerance was `1e-6`, so the visible safety probability gap
  was `0.01`; the best-choice argmax was not the cause. Hidden unrounded
  probabilities were unavailable, so these receipts do not establish whether
  the gap came from rounding or from the model/provider. This is separate from
  three direct valid-repeat best/safety-reject contradictions. No acceptance
  weakening was applied.

The continuation record shows one acknowledged invalid request, followed by the
quarantine policy. Strict parsing and the original caps were unchanged; no retry,
new generation, or new Sol call occurred. The original payload was not changed:
all 12 same-case/same-order canonical body hashes match the original replay.
Unmatched cases were not compared.

## Recomputed repeat metrics

### Original response versus valid follow-up response

Only the ten valid repeat rows are included. The metrics are deliberately
separate:

| comparison | denominator | changed | unchanged |
|---|---:|---:|---:|
| best-choice ID | 10 rows | 2 | 8 |
| combined accepted selection/status | 10 rows | 2 | 8 |
| candidate safety verdict | 36 candidate-order observations | 4 | 32 |

The safety transitions were accept→accept 12, accept→reject 2, reject→accept 2,
and reject→reject 20. “Candidate-order observations” are not independent
samples.

### New order-1 versus order-2 comparison

After excluding the entire invalid `cue-679` pair, there are **5 complete case
pairs, not 6**:

| comparison | denominator | changed | unchanged |
|---|---:|---:|---:|
| best-choice ID | 5 cases | 3 | 2 |
| combined accepted selection/status | 5 cases | 2 | 3 |
| candidate safety verdict | 18 candidates compared across paired orders = 18 observations | 2 | 16 |

The safety transitions were accept→accept 6, accept→reject 1, reject→accept 1,
and reject→reject 10. The safety denominator is 18 candidate-order comparisons
because each candidate is compared once across the two orders; it is not a
sample-size claim.

The three direct valid-repeat contradictions were:

- `16e2689c67a5:cue-618`, order 1: best `c2`, safety `reject`;
- `bec0de4b2abc:cue-504`, orders 1 and 2: best `c3`, safety `reject`.

The two separate structural invalids were both `cue-679`: order 1 provider
choice `c5` and order 2 provider choice `c4`. In each row the raw best choice
was valid; the parser error “Jev choice is inconsistent with probabilities”
came from the named safety question (`safety_c3` in order 1 and `safety_c5` in
order 2), where the visible `reject` probability was `0.49` versus `accept`
`0.50`. The hidden unrounded probabilities are unavailable, so the receipts do
not prove rounding versus model/provider origin.

## Isolated safety and cross-tabs

All 23 isolated payloads exactly matched the frozen original candidate ID, text,
context window, and candidate-specific safety question: **23/23**. Isolated Jev
verdicts were **accept 10, reject 13, insufficient_context 0**.

Comparing each complete valid repeat case’s two safety vectors with its isolated
vector gives:

- both orders the same: `cue-353`, `cue-618` (2 cases);
- both orders different: `cue-94`, `cue-504` (2 cases);
- one-sided: `cue-210` (1 case).

This is a combined isolation effect: both the multi-candidate state and the
questions pack were removed. It is not proof of a context-only effect.

The historic grouped safety table contains 46 candidate-order observations
(23×2): accept 15, reject 31. The new valid-repeat table contains 36
candidate-order observations (18×2 after excluding the complete invalid cue-679
pair): accept 14, reject 22. Their cross-tab is:

| historic safety | new accept | new reject |
|---|---:|---:|
| accept | 12 | 2 |
| reject | 2 | 20 |

All of these are observations, not independent samples. The isolated Jev and
DeepSeek verdicts agree on accept versus not accept in **19/23** rows (8 + 11);
this is an agreement count, **not accuracy**.

## Independent DeepSeek review

DeepSeek returned 23 raw verdicts: **accept 10, revise 13, impossible 0**.
No `impossible` response was coerced into a truth label. The complete isolated
Jev-safety × DeepSeek table is:

| isolated Jev safety | DeepSeek accept | DeepSeek revise |
|---|---:|---:|
| accept | 8 | 2 |
| reject | 2 | 11 |

Full row table is also in `analysis.json` under
`deepseek_independent_review.full_table`.

Selected-repeat disagreements (only valid repeat rows with a selected
candidate) and the source-verbatim DeepSeek reasons are:

- `16e2689c67a5:cue-353`, orders 1 and 2, selected `c2` “Питалась четыре с
  половиной мес.”, DeepSeek `revise`: “Сокращение 'мес.' является письменной
  аббревиатурой, а не разговорным сокращением; для озвучивания (TTS) это
  неестественно и может быть прочитано неверно.” It also recorded: “Числительное
  'четыре с половиной' сохранено словами — это верно, но сокращение единицы
  измерения нарушает требования к произносимой речи.”
- `16e2689c67a5:cue-94`, order 2, selected `c4` “но знаю, так, чувствую кожей,
  сердцем.”, DeepSeek `revise`: “Candidate drops the pronoun 'я' and the
  demonstrative 'это', weakening the personal assertion 'я знаю, что это так' to
  a bare 'знаю, так'.” It also recorded: “The connective 'что' is removed,
  making 'знаю, так' read as two disconnected fragments rather than a coherent
  clause.”
- `2c58c71203c3:cue-210`, order 2, selected `c2` “Готов.”, DeepSeek `revise`:
  “Candidate drops the affirmative 'Да' and the prepositional phrase 'к работе',
  reducing the meaning from 'Yes, ready for work' to merely 'Ready.'” It also
  recorded: “The context shows the speaker is responding affirmatively to a
  question about readiness for work; omitting 'Да' removes the explicit
  confirmation.”

These are attributed model reasons, not manual language adjudication. The
numeric/grammar observations above are not proof of accuracy.

## Screenshot cases: raw provenance and model outputs

Historical Sol labels below are **provenance only, not matched truth**. “New
chosen” is the raw repeat selection; isolated and DeepSeek values are the raw
candidate-level verdicts. Empty chosen-order cells mean that candidate was not
selected in that order.

### `16e2689c67a5:cue-353`

Original: `И так я питалась четыре с половиной месяца.`

| ID | candidate text | historical Sol | historical chosen | new chosen | isolated Jev | DeepSeek |
|---|---|---|---|---|---|---|
| c1 | Ела четыре с половиной месяца | revise | — | — | accept | accept |
| c2 | Питалась четыре с половиной мес. | accept | 1, 2 | 1, 2 | accept | revise |
| c3 | Ела четыре с половиной месяца. | revise | — | — | accept | accept |
| c4 | Зонд четыре с половиной месяца. | revise | — | — | reject | revise |
| c5 | Я ела четыре с половиной месяца | revise | — | — | accept | accept |

Context excerpt: “с моими травмами / и поэтому у меня был такой назогастральный
зонд… / Да. / И так я питалась четыре с половиной месяца. / Всё хорошо. Я просто
каждый день радуюсь жизни.”

### `16e2689c67a5:cue-94`

Original: `но я знаю, что это так, чувствую это кожей и сердцем.`

| ID | candidate text | historical Sol | historical chosen | new chosen | isolated Jev | DeepSeek |
|---|---|---|---|---|---|---|
| c1 | но я знаю, чувствую это сердцем. | accept | — | — | accept | accept |
| c2 | но я чувствую это сердцем. | revise | — | — | reject | revise |
| c3 | но я знаю, что так, чувствую сердцем. | accept | — | — | accept | accept |
| c4 | но знаю, так, чувствую кожей, сердцем. | revise | 1, 2 | 2 | accept | revise |
| c5 | но знаю что так чувствую кожей сердцем | revise | — | 1 | accept | accept |

Context excerpt: “Я думаю, они не хотят признавать, что я становлюсь обузой, / но
я знаю, что это так, чувствую это кожей и сердцем. / Да.”

## Usage, latency, and integrity

- Jev: 95,738 input and 3,876 output tokens; estimated usage cost
  **$0.004020996**.
- DeepSeek: 17,199 input and 2,960 output tokens; estimated usage cost
  **$0.0087117**.
- Total estimated usage cost: **$0.012732696**. Reserves were $0, Sol
  subscription cost was $0, and new generation cost was $0. These are not
  provider-billed balances.
- Primary timing uses the persisted backend/response elapsed field: Jev
  `response.elapsed_seconds`, including failed rows where elapsed time was
  available, median **0.778544 s**, nearest-rank p95 **1.169836 s**, N=35;
  DeepSeek `response.elapsed_sec`, median **1.401180135 s**, nearest-rank p95
  **1.915586930 s**, N=23. DeepSeek's separate model-call metric,
  `response.calls[0].elapsed_sec`, is median **1.400898538 s**, nearest-rank
  p95 **1.915269292 s**, N=23. Backend response elapsed includes overhead; call
  elapsed is the model-call interval. These are observed values, not an SLA,
  speed guarantee, or public benchmark. Jev is a mixed workload (12 repeats +
  23 isolated safety calls), while DeepSeek is 23 single-candidate reviews;
  no cross-workload fairness claim is made.

Recomputed source bindings are recorded in `analysis.json`. In particular:

- follow-up manifest SHA-256:
  `34e7260bc3aa84771b98188b07bde1b4a82d6b957cbecaae378936b5f4613263`;
- follow-up ledger SHA-256:
  `a92a81b75e3702d5eb76402c559e855d5cadfa9672c1a423b247eb517843a524`;
- original manifest SHA-256:
  `e8b2f5dc16cf6f01d377417eb35bfc66d7c3ecd4fcb90920773ac40332558a7f`;
- original ledger SHA-256:
  `ff5b10db11f19ad1794508eb02eb2e5bdb40a180017a2d8ac3429f07b3f4df3d`.

## Limits and next steps

The selected six cases are not representative. There is no human reference,
ground truth, or independent language proof. Inputs were unchanged, so this
supports a descriptive stochastic-repeat comparison, not a causal order-position
claim. Historical Sol labels and model agreement are not correctness. H5 remains
unchecked pending a controlled matched comparison and held-out stage; this
follow-up does not validate Russian-language quality.

The main conclusion is **do not replace Sol or automatically adopt Jev's
single-evaluation judgments** on this record. Same-input changes and
choice/safety contradictions remain, and independent agreement is not truth.
Keep Jev experimental and shadow-only. This report does not make an automatic
selection or deployment decision.

## Specific H6 follow-up proposal — planned only, not executed

This is a two-case diagnostic of the failure-only Sol suggestion/rewrite idea,
not proof of a general benefit. The sets are cases with no DeepSeek `accept`,
not a claim that every model rejected every candidate:

- `16e2689c67a5:cue-618`: 3 candidates, all `revise` from DeepSeek;
- `2c58c71203c3:cue-210`: 2 candidates, all `revise` from DeepSeek.

For each case, compare one new Sol rewrite with one additional Muse repair. Both
use the same original `3 + target + 3` window, target character limit, rejected
candidate pool, and the same frozen DeepSeek feedback. Allow exactly one bounded
attempt for each generator, and allow `cannot_shorten`. Proposals get new
provenance; Sol does not self-approve its rewrite. Use a separate blinded
DeepSeek-off review under the frozen rubric, from a model family different from
both generators; it is an independent review, not gold. If references disagree
or remain unresolved, keep the original. Do not manually rank style
improvements. No H6 call was made, Sol subscription cost was $0 for this
follow-up, and this plan does not change the global default or prompts. See also
the [preserved broad H6 plan](sh06_jev_typesafeai_arbiter_plan.md#7-next-experiment-failure-only-sol-suggestionrewrite-h6).

## Offline verification

The corrected counts and timings were recomputed from the saved receipts only;
the local/private [HTML report](../output/jev_followup_20260919/shortening.html)
remains the report artifact. No API call or unittest is needed for this
documentation-only correction. The broader in-progress suite checkpoint is
`695 passed, 5 skipped` after stable fixtures; it was not rerun for this
documentation-only change.
