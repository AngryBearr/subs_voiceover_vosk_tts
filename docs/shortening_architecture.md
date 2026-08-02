# Subtitle shortening architecture

> **Normative architecture:** see the [shortening experiments log](shortening_experiments.md)
> for empirical decisions, evidence, reproduction notes, and pending work.

## Product goal

Shorten subtitles when their spoken content does not fit the available timing,
while preserving meaning, unresolved facts, and compatibility with the current
CLI and artifact contracts. The first step is a small, safe domain boundary,
not a pipeline rewrite.

## Current problems

Timing derivation and the character budget formula currently live in a legacy
helper beside orchestration, prompts, validation, and backend adapters. This
makes the timing rules mutable, difficult to characterize, and easy to couple
to provider behavior. The existing analysis fields also mix source timing,
estimated timing, and extended timing.

## Target flow

```text
analyze -> original-only semantic planner -> Flash generate
-> deterministic validate -> Pro critic/editor/verifier -> final fit decision
```

Phase 3 is implemented as an optional, fail-closed quality-v3 planner stage.
The planner receives only indexed original text, source context, budgets, and
deterministic requirements. Its immutable JSON plan contains atomic
propositions and anchored requirements and is passed unchanged to every later
quality stage. It has no function tools for character counting or semantic
diffing; those remain local deterministic checks. The endpoint response is
strict JSON and the local parser rejects unknown fields, wrong source text,
partial batches, and invalid anchors. A planner API, size, or parse failure
leaves the candidate unchanged and unresolved.

`max_chars` is an approximate generation hint, not a duration guarantee. A
future `Duration/FitEvaluator` must measure or validate the actual TTS duration
and make the final fit decision. The character-budget estimator must therefore
remain distinct from that evaluator.

Three threshold semantics must remain separate:

1. Generic analysis criticality - whether timing analysis marks a cue as critical.
2. Shortening selection - whether a checked cue is selected for shortening.
3. Generation target fit - the target ratio used to derive a generation hint.

Future domain boundaries should use immutable cue, timing, candidate, and
decision values. A narrow backend adapter belongs only at the batch-generation
boundary; provider details must not enter the domain core.

## Migration plan

### Phase 1 - current change

Introduce immutable `TimingWindow` and `CharacterRateBudgetEstimator`, and
connect them to `_stable_durations` and `calculate_budget`. Preserve legacy
analysis-field writes and all public signatures. Add characterization tests.

### Phase 2 - implemented

Add immutable concrete semantic requirements with source anchors, including a
`spatial` gate, and pass them through the critic/editor/compaction/verifier/
repair stages. Reports retain the legacy `risk_hints` projection and expose the
auditable requirement dictionaries. A semantically restored repair is retained
as unresolved when a final reverify fails only for naturalness or grammar.

### Phase 3 - implemented

Production combined Flash + Pro runs enable the original-only planner before
critic. Standalone review CLI enables it by default and supports
`--no-semantic-planner`; direct programmatic review remains planner-off by
default for backward compatibility. Sparse context timing is hydrated before
planning so source-only plans use stable indexed timing. Multi-candidate
generation and reranking remain future work.

### Future phases

- Extract immutable candidates and decisions around deterministic validation.
- Add multi-candidate generation and reranking.
- Add an actual TTS duration evaluator.
- Introduce `Duration/FitEvaluator` for actual TTS duration and final fit decisions.
- Add the narrow batch-generation backend adapter without changing domain APIs.
- Migrate orchestration incrementally while retaining compatibility adapters.

## Backward-compatibility invariants

- CLI arguments, JSON schema, reports, backends, Pro lineage, and pipeline
  entrypoints remain unchanged.
- `_stable_durations` continues to write the two legacy analysis fields and
  returns the same tuple values.
- `calculate_budget` retains its side effect, `ShorteningBudget` shape, exact
  rounding behavior, and max-character formula.
- Prompts, thresholds, target selection, and iterative orchestration retain
  their current semantics.

When a shortening candidate cannot be safely resolved, prefer unresolved text
over semantic damage. This unresolved-over-semantic-damage principle is a
product invariant, not merely a provider fallback.

## Optional independent final barrier

After candidate generation, the combined pipeline may run `utils.verify_subtitles_opencode` as an explicit independent semantic barrier. The order is Flash -> stable timing -> Pro -> stable timing -> unit-level barrier -> refresh analysis -> shortening status. When enabled, unit-level verification is the default: units are grouped conservatively at no more than three cues, with timing gaps no greater than 0.3 seconds and textual continuation; unknown timing remains single-cue. Requests use a four-unit batch by default. Unchanged neighbors are evidence only, and atomic fallback restores only changed unit members. `--no-semantic-barrier-units` opts back into cue-level verification. It accepts original and candidate arrays, and restores exact original text for every non-pass, uncertain, transport, or schema outcome. The Pro report is not passed as evidence. It uses local OpenCode HTTP plus OAuth subscription accounting only; direct APIs and public servers are out of scope. The stage remains disabled by default.

The honest scope is that Pro editing/review remains cue-level; only independent
final barrier acceptance is unit-level. There is no intra-unit error attribution
beyond shared issue codes.

Unit-level final-barrier implementation is validated by the tracked
`skill_test/semantic_benchmark60` control (with benchmark24 retained as the
historical subset). Pro editing/review remains cue-level;
only final-barrier acceptance is unit-level. The qualified default is Sol strict
text (`openai/gpt-5.6-sol`); StructuredOutput remains an explicit option for
compatible routes. The barrier remains an explicit combined-pipeline opt-in.
Startup/configuration failures are fatal; ordinary per-item failures are fail-closed
and continue with unresolved accounting.
OpenCode provider cost is informational and not billing-authoritative.

Malformed provider usage metadata is treated as a transport failure. The verifier
therefore restores text fail-closed even when the response text might otherwise
contain a valid verdict.

Semantic-unit requests use OpenCode 1.18.9 schema-validated synthetic
StructuredOutput tooling through the message `format` field. This is not a
provider-native `response_format`: tool-call reliability remains measured and
provider support can still fail. OpenCode retries are disabled with
`retryCount: 0`; verifier caller retries remain authoritative.
# Verification backends

Subtitle verification has two distinct direct transports. OpenCode uses its
local server and synthetic tool/structured-output behavior. OpenRouter uses the
native Chat Completions `json_schema` response format, with strict schema
validation and fail-closed handling. It sends temperature 0 by default for
repeatable verification; endpoints must support this parameter because
`provider.require_parameters` is true. The OpenRouter command always verifies
semantic units and records metered provider usage; it does not alter or enter
the combined shortening pipeline. Its qualified exact default model is
`xiaomi/mimo-v2.5-pro`, with one schema retry by default. The adapter's native
smoke was technically validated after installing the already-pinned `aiohttp`
dependency; this is transport validation, not a live model-quality claim.
# Ollama Cloud boundary

The direct Ollama Cloud semantic verifier is an isolated backend. Its direct
`/api/chat` transport works, but Cloud has no native structured-output contract
on this route: no `format`, tools, or schema is sent. The existing strict unit
parser is the client-side schema authority and fails closed on malformed
content. The `--model` argument is required; there is no stable
qualified/default Ollama Cloud model. This standalone verifier does not alter
the combined pipeline or OpenRouter behavior. Subscription use has no
authoritative per-call cost.
