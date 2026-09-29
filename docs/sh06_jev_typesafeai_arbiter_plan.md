# SH-06 Jev / TypeSafe AI arbiter research and proposed experiment

**Research date:** 19 September 2026

**Status:** historical research snapshot with a preliminary real replay. A
separate bounded direct replay used `jev-1.13.0` for 24 cases in two recorded
orders (48 Jev requests); it is linked below and is not a full H5 matched-Sol
or held-out evaluation. Direct access was verified with an authorized
user-provided key; the manifest records user-reported `$5` trial credit, and no
actual balance query was made. The HTML is local/private. No further calls,
public benchmark, or upload are authorized by this note. Public documentation
was fetched for the historical research snapshot.

This note is linked from the unchecked [SH-06 task](shortening_tasks.md).
It proposes a bounded comparison of Jev with the current Sol arbiter. It does not select Jev for
production, close SH-06, close SH-08, or close SH-10.

The preliminary replay is reported in [the private replay result](sh06_jev_replay_20260919.md),
with the [local HTML report](../output/jev_replay_20260919/shortening.html) and
[offline analysis](../output/jev_replay_20260919/analysis.json). It is a
bounded real replay, not production integration, a production best-of-5 Muse
bridge, a matched new-Sol comparison, or a held-out evaluation.

The later three-part follow-up is recorded in the [private offline follow-up
analysis](sh06_jev_followup_20260919.md), with its [local HTML
report](../output/jev_followup_20260919/shortening.html) and
[machine-readable analysis](../output/jev_followup_20260919/analysis.json). It
executed 58 attempts: 56 valid, 2 structurally invalid, and 0 pending. Its
partial status reflects retained invalid responses, not unfinished network work;
the invalid responses were retained as attempt 1 and never retried. The
follow-up does not close H5, validate Russian quality, establish a benchmark or
speed guarantee, or authorize a default change.

## 1. Source-backed research

### What Jev is

- TypeSafe AI Inc. presents Jev as a specialized structured-decision model, not a
  general chat model, rewriter, or embedding model. See [Introducing System One,
  models, and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  and the [model documentation](https://docs.typesafe.ai/models).
- The documented stable pin is `jev-1.13.0`. `jev-latest` and `jev-preview` are
  movable; they are not stable experiment identities. The [model list](https://docs.typesafe.ai/models)
  must be checked at execution time.
- TypeSafe markets RLCD and parallel, non-autoregressive outputs. Parameters,
  backbone, training-language list, weights, and hardware are not disclosed in
  the reviewed sources. The SDKs are open under MIT licenses, but the model is
  not thereby an open model; see the [TypeSafe GitHub organization](https://github.com/typesafe-ai).
- English is the primary or best-supported language in the available material.
  Other languages, including CJK languages, are described as unevenly handled.
  No Russian-specific benchmark was found. This plan therefore makes no claim
  that Jev is good at Russian.

### HTTP interface and decision semantics

- The official HTTP API is `POST https://api.typesafe.ai/v1/systemone` with
  `{model,state,questions}`. The [API reference](https://docs.typesafe.ai/api)
  documents text or JSON state and structured primitive questions.
- [Choice](https://docs.typesafe.ai/primitives/choice) returns a chosen option,
  probabilities, and confidence. A Choice criterion map is limited to 255
  options. [Score](https://docs.typesafe.ai/primitives/score) has between 2 and 10 described levels;
  [Noul](https://docs.typesafe.ai/primitives/noul) returns a yes-probability.
  There is no free-form explanation output in these primitives.
- Independent questions use the common state and do not condition on earlier
  answers. A response therefore cannot be treated as an answer-aware chain of
  critique questions.
- Jev has no inherent rejection, tie, or abstention behavior. The experiment
  must provide explicit options such as `none_acceptable` and
  `insufficient_context`. A probability distribution's shape is not a
  probability of correctness. The [confidence documentation](https://docs.typesafe.ai/confidence)
  does not turn confidence into correctness; any threshold must be calibrated
  separately for each primitive and version.

### Limits, route, and cost

- The TypeSafe model page lists $0.042 per 1M input tokens, free output, a 64k
  total-token limit, and a 32k state-plus-longest-question limit. It also lists
  1,200 RPM and 250k tokens/s as dynamic limits, not a promise for this account
  or route: [models](https://docs.typesafe.ai/models).
- An illustrative 1,000 requests × 2,000 input tokens would be $0.084 in Jev
  input charges only. The 2,000-token figure is assumed total input per request,
  including state, repeated windows, instructions, and criteria; the actual
  payload may be larger. Generator/Sol/audit calls, retries, and taxes are
  excluded, but repeated windows are not excluded from billed input tokens.
  This is not an authorized budget or latency benchmark; actual usage is needed
  for a real estimate.
- The two verified route options are direct early access/waitlist and Cloudflare
  [Jev integration](https://developers.cloudflare.com/ai/models/typesafe/jev/)
  using model `typesafe/jev` on the Cloudflare route. Price on that route was
  not publicly verified here. The exact returned model/version and schema must
  be checked; direct-route pin or schema acceptance must not be assumed.
- No local weights or Hugging Face/inference package requirement was found.
  Permanent impossibility of local inference must not be asserted. No
  OpenRouter-supported-inference claim was found or made.

### Known model limits and data handling

- The [Jev jaggedness report](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
  reports limits around literal and multihop reasoning, numeracy, counting,
  context distraction, adversarial state, and structured invariants across
  questions; structured invariants across questions are not guaranteed. A local
  program can count lengths; Jev's semantic judgment remains unproven.
- Subtitle state is untrusted data. IDs, delimiters, and a strict parser do not
  magically cure instruction-looking subtitle content or prompt injection; the
  experiment must not give Jev tool actions.
- The [privacy policy](https://typesafe.ai/legal/privacy-policy) and [legal
  documentation](https://docs.typesafe.ai/legal) say customer data is not used
  to train models without consent, but this is not automatically zero retention;
  enterprise options, US/service-provider processing, and telemetry terms still
  need review.
- The [MCA](https://typesafe.ai/legal/mca), including section 2.3(f), restricts
  publishing benchmarks or performance. Before live calls, applicable provider
  terms must be clarified. The intended report is access-controlled private
  HTML unless publication permission is confirmed; this is not legal advice and
  does not claim that public publication is allowed.
- New-provider data consent and budget authorization are separate decisions.
  Existing Muse consent, budget, or artifacts are not automatically reusable.

## 2. Proposed H5 question and boundaries

**H5 (Jev selection and acceptability):** on the same generated candidate sets,
can a blinded Jev selection plus per-candidate acceptability decision provide a
safe, useful outcome for Russian subtitle shortening compared with a new Sol
judgment, without silently selecting an unsafe runner-up?

This is separate from **H3** (whether generation of multiple alternatives is
useful), **H4** (how original context is presented to the generator), and
**SH-10** (evaluation of original versus stitched final context). H5 does not
replace any of those tasks. The first arm uses each target's exact original
window: up to three preceding cues, the target, and up to three following cues;
that is the required `3 + target + 3` window link in this plan.

The existing default remains unchanged: Sol is not replaced, and this is a
limited/shadow experiment rather than deployment.

## 3. Proposed executable experiment

### A. Candidate set and generation

1. For each target, retain the exact source seven-cue window when available and
   the full original target. Use identical ORIGINAL text for up to three cues
   before and after the marked center. Substitute only the marked center in a
   candidate window; do not translate, rewrite, truncate context, or fabricate
   missing edge neighbors.
2. Preserve a stable source hash, cue IDs, chronological order, and timing.
   Clone windows per user request initially. Score the candidate center only;
   unedited neighbors are context, not scored candidates.
3. Reuse the existing 1–5 candidate protocol as the candidate-set contract.
   The initial plan retained a future adapter/bridge for production best-of-5
   generation here. That planned path remains future work: the separate
   bounded replay pooled saved low/high Muse artifacts, generated no new
   candidates, and is not a production Jev/Muse best-of-5 bridge. Once a
   bridge exists, use the proposed Muse low generator configuration to produce
   1–5 genuine alternatives per target; it is not a mandatory final selection.
4. Locally count the character budget with a frozen `length_basis`, validate
   IDs and format, and remove duplicate or over-budget options while preserving
   failure counts. Do not silently turn separate one-off outputs into one
   five-variant generation group.
5. Numeric words-to-digits behavior remains a prompt/model criterion, not a
   universal local numeric parser. Preserve source values and representation and
   measure violations; do not claim deterministic numeric enforcement.

### Candidate safety and semantic rubric

A candidate must preserve the full essential meaning: actors, entities, numeric
value and source words-versus-digits representation, negation, modality,
temporal and causal relations, and references. It must not inject facts from
neighboring cues. Semantic preservation is mandatory; fluency and naturalness
are tie-breakers only among safe candidates, and shorter is not automatically
better. Character length is checked by the exact local program, not by Jev.
Numeric compliance is a prompt-and-diagnostic criterion, not a deterministic
validator.

### B. Blinded Jev decision

1. Build one blinded state containing the original window and candidate windows,
   with shuffled opaque IDs. Omit generator/provider identities, prior critique,
   and prior verdicts. The original context and full candidate windows remain
   available to the judge.
2. Send one baseline Choice question with candidate IDs plus explicit
   `none_acceptable` and `insufficient_context` options. In the same request,
   send one safety Choice question per candidate with `accept`, `reject`, and
   `insufficient_context`. The maximum is six questions for five candidates.
3. Questions are independent against the common state. Accept a candidate only
   when the selected ID is valid, that candidate's safety result is `accept`,
   and local fit passes.
4. Any none, conflicting, missing, invalid, or uncertain result falls back to
   the original and an unresolved/manual queue. Never silently choose a
   runner-up and never regenerate indefinitely.
5. An optional confidence gate may be calibrated on development data only. It
   must not be an arbitrary threshold copied from another primitive or version.
6. All variants may be unacceptable, and several may be valid. A tie must not
   be forced into a false single-candidate fault. If diagnostics show mapping or
   consistency problems, run isolated original-plus-one-candidate safety calls
   as a separate follow-up while retaining the same five-way selection arm; do
   not launch a large experiment matrix up front.

### C. Matched Sol comparison

- Obtain new Sol judgments on the exact same candidate sets, original/full
  candidate windows, and operational vocabulary: `selected_candidate_id|null`
  plus per-candidate `accept`, `reject`, or `insufficient`.
- Record two identical permutations in both arms. Do not reuse old Sol verdicts
  with different prompts or state. Do not equate existing Sol `revise` or
  `impossible` with true rejection automatically.
- Keep generator, Jev, Sol, audit, and retry costs and token usage separate.
  Jev has no natural-language critique, so this first experiment replaces only
  selection/acceptability; it does not replace or test the repair-feedback
  pipeline.

## 4. Gates and data stages

All prerequisites below remain pending for any live call:

- Confirm direct or Cloudflare access, key handling, current model/version,
  schema, terms, new-provider data consent, explicit budget, control/audit/
  limit/retry allowance, and private-reporting permission. No spend is
  authorized by this document.
- First validate payloads, response schemas, IDs, window construction, strict
  parsing, fail-closed fallbacks, and unit checks offline. Do not call a model
  during this stage.

The preliminary replay did not complete these gates, the matched-Sol arm, or
the held-out stage. In particular, it made 48 Jev requests (one canary and 47
continuation requests), recorded 43 accepted records and an estimated
`$0.009278304` cost from 220,912 input tokens, but made no new generator, Sol,
or independent-judge calls. Its 24-case selection is not representative of all
60 cases or all subtitles: historical low/high candidate pools were deduplicated
and capped at five, with the actual distribution 4×2, 8×3, 4×4, and 8×5;
truncation is recorded in the manifest. The two orders confound stochastic
variation with presentation, so they do not establish causal position bias.
Numeric preservation was not independently validated, and Jev `accept` is not
ground truth.

The following proposed stages are ordered and remain pending; they are not
results:

1. **Pending — 24-case Russian development diagnostics.** Prepare diagnostics
   with predeclared contrasts: value and words/digits (a rule), negation,
   modality, actor/pronoun, causality/time, split cue, all-bad/multiple-valid,
   insufficient context, duplicate/position, and instruction-looking subtitle
   text treated only as content. Do no external security probing.
   Synthetic perturbations are clearly marked diagnostic controls, not genuine
   generated outputs. Labels are independently checked; the main agent does not
   hand-pick winners. Cyrillic output alone is not a pass.
   An optional Russian-versus-English-instructions development comparison may
   use identical Russian content without translation. Freeze the winning
   formulation before later sets; this is not a live-language claim.

2. **Pending — matched 60-case development stage.** Run the matched stage:

- Replay the same sources as SH08 where provenance is proven. Reuse saved
  best-of-5 groups when available. Reconstruct missing context only from exact
  source hash plus cue index/ID; otherwise exclude and count it. Never invent a
  window.
- If fresh genuine sets are needed, generate them once after authorization and
  use the identical sets for both judges. Do not assemble independent r1/r2
  single outputs and relabel them as one five-variant generation.
- `60 x 2 orders` means two recorded orders of the same 60 grouped cases, not
  120 independent cases. Group reporting by source and cue.

3. **Pending — final heldout evaluation.** Keep three pilot files for the final
   evaluation. The final heldout set must have disjoint raw and parsed source
   hashes. If any pilot file is tuned on, replace it in the heldout set. No
   heldout result is reported by this plan.

## 5. Metrics and promotion rule

Primary metrics are unsafe/critical false acceptance against an independent
blinded assessment, not agreement with Sol as truth; acceptable-and-fit
coverage; rejected-all; incorrect rejections; unresolved cases; order
stability; Russian error categories; structural consistency; and per-role cost,
p50/p95 latency, call counts, actual usage, and errors.

Before live work, choose and freeze a separate model family for reference. User
review of disagreements is optional and is not promised as autonomous gold
labeling. An unresolved reference is not ground truth. Any detected critical
acceptance blocks promotion pending diagnosis; zero observed failures is not a
guarantee. Freeze numerical tolerance and sample requirements before heldout
evaluation. Quantitative decision tolerances and sample size must be frozen
prospectively, not chosen after outcomes. Insufficient evidence is a valid
outcome.

The default Sol path remains unchanged unless the evidence supports a later
decision. This first stage is limited and shadow-only, not deployment.

## 6. Context and final-output boundary

Adjacent shortened cues can change the stitched final context. The initial
experiment uses original three-before/three-after windows. After stitching,
run a separate SH-10-style original-context versus final-context evaluation;
the current `final_context_checked` state remains false. H5 cannot claim full
final safety or measured audio fit from semantic selection.

Future work may add a small provider adapter and combination tests, using
[`shortening_candidate_prompts.py`](../utils/shortening_candidate_prompts.py),
[`shortening_experiment.py`](../utils/shortening_experiment.py),
[`shortening_plan.py`](../utils/shortening_plan.py), and
[`shortening_html.py`](../utils/shortening_html.py). No source changes are
authorized by this research note.

Any eventual real or partial run must produce a standalone private HTML report
with proposals, exact model mode, arbiter probabilities and selection, safety,
final text, status, and cost. It must not fabricate explanations. Dry-run,
fake, local-only, and Vosk-only artifacts are not live reports. Resolve the
provider's benchmark-publication restriction before any live report is shared.

## 7. Next experiment: failure-only Sol suggestion/rewrite (H6)

H6 records a user question, not an approved implementation or Sol call. Existing
Sol arbiter suggestions are already persisted and shared with the generator on
repair; old comments need no duplicate record. The path starts only when all
locally valid candidates are rejected or bounded repair is exhausted.

- Permit one extra Sol generation request with the exact original `3 + target +
  3` context, target, limit, `length_basis`, every rejected candidate ID/text,
  and the real issues/advice. Request one alternative without manual wording;
  preserve meaning, numeric representation, naturalness, and relations, or
  return `cannot_shorten`.
- Check length, format, and deduplication locally; record `generatorSol`
  provenance and a separate attempt/cost counter. Never auto-approve its rewrite
  and preserve the original on failure.
- Use an independently frozen arbiter/other family or shadow-only review. Same-
  Sol review is non-independent and cannot establish gold quality; Jev is
  eligible only after Russian validation.
- Match one extra Muse repair on the same rejected cases and allowance, separate
  from H5's Jev initial-judge comparison. Freeze metrics: critical meaning
  errors, accepted-and-fit coverage, additional cost/latency, and unresolved
  cases. Do not retry endlessly or auto-expand the budget.
- Suggestions for already accepted text remain optional future work, not a
  default per-cue cost. No Sol calls or implementation are authorized now. A
  separately authorized small live Jev sample does not authorize H6; the new $5
  Jev trial is Jev credit, not Sol credit.

### Concrete bounded H6 diagnostic (planned only)

The follow-up analysis proposes two failure-only sets with no DeepSeek
`accept`, rather than claiming that every model rejects every candidate:

- `16e2689c67a5:cue-618`: three candidates, all DeepSeek `revise`;
- `2c58c71203c3:cue-210`: two candidates, all DeepSeek `revise`.

For each case, compare one new Sol rewrite with one additional Muse repair using
the same original `3 + target + 3` window, target limit, rejected candidate
pool, and the same frozen DeepSeek feedback. Bound each generator to one
attempt; `cannot_shorten` is allowed. Review proposals in a separate blinded
DeepSeek-off pass under the frozen rubric, using a family different from both
generators; this is not gold. Give proposals new provenance and never allow Sol
to self-approve. If references disagree or remain unresolved, keep the
original; style improvement is not manually ranked. This plan has not been
called, Sol subscription cost was $0 for the follow-up, and it does not change
the global default or prompts. Two cases are diagnostic only and cannot prove a
general benefit.

Existing Muse evidence includes the actual low/high [comparison report](sh08_muse_reasoning_20260915.md).
The user-provided screenshot case `16e2689c67a5` (cues `cue-353` and `cue-94`)
is illustrative, not ground truth: Sol's advice to preserve special feeding
context and its length-only suggestion such as `ела так...` are not automatic
good rewrites.
