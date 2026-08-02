# Subtitle shortening experiments log

**Snapshot date: 2026-08-01.** This is the empirical, decision-oriented record for
subtitle shortening. The architecture document is normative; this log records
what was measured, rejected, accepted, or still needs evidence.

## Purpose, scope, and update rules

Use this log to prevent repeated failed experiments and to make every shortening
decision reproducible. Record the goal, fixture and provenance, backend/model,
exact command or settings, metrics, failures, cost/accounting, decision, and next
step. Update this file when an experiment changes a decision, adds independent
evidence, or establishes a release-relevant limitation. Do not silently turn an
opt-in experiment into a default.

### State at this snapshot

- **Committed baseline:** Phase 1-3 implementation and tests at commit `54a4e62`.
- **Current uncommitted experimental work:** duration profiles, direct Flash
  selection, standalone Pro duration gate, and combined-profile wiring are
  experimental opt-ins; this log describes their measured state, not a claim that
  they are production defaults.
- **Ignored/temp artifacts:** generated reports, audio, caches, and smoke outputs
  under `output/` and `/tmp/opencode` are non-durable unless explicitly copied
  into a tracked fixture or otherwise preserved. A path in a command is not
  evidence that the artifact is committed.
- **Future work:** multi-cue semantic units, actual-TTS final measurement and
  bounded compaction/reverification, and independent model-family verification
  remain pending.

## Definition of success and release criteria

A release candidate must satisfy all of these working targets. They are revisable
targets, not achieved results, and require evidence across multiple episodes,
genres, and voices:

- Zero severe silent semantic errors: dropped negation, modality, time,
  causality, comparison, alternatives, references, spatial relations, or other
  material propositions are not silently accepted.
- Mean human semantic score >=9/10, with no accepted item below 7/10.
  Automated verifier scores are not a substitute for human quality.
- >=95% actual post-processed TTS fit among resolved items, measured at the
  selected production settings rather than inferred from `max_chars` or a proxy.
- <=10% unresolved on the release set, with every unresolved result explicit and
  fail-closed.
- <=1% parse/API failures, and zero failures counted as verified. Bounded retries,
  stable failure reasons, and retained diagnostics are required.
- At least five independent episodes, two genres, and two voices. Accepted-subset
  quality must not be confused with overall coverage: a clean score on accepted
  targets does not prove that all critical targets can be resolved.

## Architecture and invariant principles

The current flow is:

```text
  analyze -> Flash generate -> original-only semantic planner
  -> Pro critic/editor/verifier/repair -> fit decision
```

The invariants are:

- Timing is immutable for planning and decisions; generated text does not rewrite
  the source timing contract.
- The planner receives original text only. Flash candidates and editor lineage
  cannot contaminate the semantic plan.
- Anchors are deterministic, indexed, and auditable.
- Retries are bounded and stage-specific; malformed or unavailable responses do
  not become verification.
- Unresolved is fail-closed: retain the safest locally valid text rather than
  silently accept semantic damage.
- Quality stages are non-thinking. Character counting, schema validation, and
  semantic checks remain local deterministic checks where applicable.
- Semantic verification is separate from duration measurement and fit selection.
  `max_chars`, CPS, and exploratory duration profiles are not actual TTS proof.

## Fixtures and data provenance

- Tracked repository fixtures include the E01 data and the `hard10` smoke set.
  The full E01 fixture remains the release regression reference; `hard10` is a
  fast comparison subset selected from known hard cases.
- Independent public E03 and E04 data were evaluated from a Drive file. Preserve
  the authoritative file IDs, SRT byte sizes, SHA-256 values, and parsed cue
  counts with each copied experiment report; these values are provenance checks,
  not interchangeable fixture names. Durable provenance is: **E03: Drive file ID
  `1brbJN4K9Sl2fg9Qvjk-pFrs0VGl-3rJo`, 156031 bytes, SHA-256
  `1a8bee086a0afebb288095b618cf2d152ff96721bc26c77241baf606c9d984fa`, parsed
  count 1523; E04: Drive file ID `1a0tQHjUVLg9GPJ74ysV0f9_8fIqPToou`, 104447
  bytes, SHA-256
  `16e2689c67a5ca6c44cb7c2598b15600ba8abff52bc91ba6dacb1899632eb6e5`, parsed
  count 1056**.
- `/tmp/opencode` and `output/` artifacts are non-durable/ignored unless
  explicitly preserved. Reports without fixture identity, model, command, and
  hash are not release evidence.

## Backend and model matrix

| Backend/model | Role | Guarantees and limits |
|---|---|---|
| DeepSeek Flash (`deepseek-v4-flash`) | Generation and direct Flash target-selection experiment | Direct API path supports explicit non-thinking and JSON requests; usage/cost metadata is available on that path. |
| DeepSeek Pro (`deepseek-v4-pro`) | Planner, critic, editor, verifier, repair, and standalone duration gate | Non-thinking quality stages, strict local parsing, bounded retries, and fail-closed routing. |
| Existing OpenCode HTTP path | Alternative provider/server orchestration | Its server path has weaker guarantees for JSON controls and thinking controls. AssistantMessage token/cache/reasoning/cost metadata is captured when reported, but `info.cost` has unspecified billing semantics and is not an invoice, subscription charge, or guaranteed actual billed cost. |
| Official ChatGPT Pro Codex path | Recommended subscription-backed test harness for future independent model-family verification | Test through the official local Codex CLI, not by copying browser state. |
| OpenCode ChatGPT OAuth | Third-party subscription-backed experiment path | Supported by OpenCode, but not explicitly endorsed by OpenAI; it is not equivalent to the official Codex path. |

Authoritative setup and reference links:

- Codex: [`codex login`](https://developers.openai.com/codex/auth),
  [`codex exec --json` and `--output-schema`](https://developers.openai.com/codex/non-interactive-mode),
  [Codex pricing](https://developers.openai.com/codex/pricing).
- OpenCode: `opencode auth login --provider openai`, then
  `opencode models openai --refresh`; use the repository's existing command with
  `--model openai/<model-from-discovery>`. See [OpenCode providers](https://opencode.ai/docs/providers/)
  and [OpenCode server](https://opencode.ai/docs/server/).

The official Codex path is the recommended subscription-backed test harness.
OpenCode OAuth is supported by OpenCode but is not explicitly endorsed by
OpenAI. The ordinary OpenAI SDK/Responses API still requires separately billed
API credentials. Subscription automation is local, trusted, and rate-limited; it
is not a public gateway. Do not use browser-cookie hacks, copied cookies, or
local auth tokens in this repository or in experiment reports.

## Decision table

### 2026-08-01 OpenCode OAuth independent barrier benchmark

**Accepted experimental opt-in, not default.** OpenCode 1.18.9 with the Luna
model was exercised through the OpenCode CLI and OAuth on the full-context E01
and E04 runs. E01 covered indices 16, 81, 121, and 77: 4 cases, with 4/4
expected failures restored and one successful request. E04 covered indices 517,
875, 222, and 641: 4 cases, with 2 expected passes kept and 2 expected failures
restored; 4/4 outcomes were correct with one successful request. In aggregate,
the runs were 8/8 overall and 6/6 on the clear subset, with 0 false positives,
0 false negatives, 2 total requests, and 0 retries or transport/schema
failures. The temporary artifacts are under
`/tmp/opencode/openai_subscription_smoke/barrier_benchmark8` and are
non-durable. This validates the standalone barrier behavior, not release
evidence, and it is not a 10/10 result.

OpenCode OAuth is a third-party OAuth path supported by OpenCode, not OpenAI-
endorsed. The official Codex verifier remains a separate Pending item. Failed
preflight attempts were environment-only: the wrong interpreter lacked `aiohttp`
and did not reach the provider, so they are not model failures.

### 2026-08-01 E04 prior-report barrier run

With the prior review report, only verified indices 517 and 875 were sent; both
passed. Unresolved indices 222 and 641 were not sent and were restored. This was
one successful request with 0 failures and approximately 10.19 seconds elapsed.
The result is temporary/non-durable experiment evidence and remains an opt-in
OpenCode OAuth run.

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | Committed Phase 1-3 baseline | Accepted | Commit `54a4e62`; implemented immutable timing/planning boundaries, semantic stages, and compatibility-preserving behavior. |
| 2 | Explicit semantic checks, bounded retries, no reasoning | Accepted | Deterministic required checks, bounded stage retries, fail-closed handling, and non-thinking quality stages are implemented and tested. |
| 3 | Multi-candidate editor | Rejected | Verified 2 -> 1; compaction 5 -> 2; silent omission at index 16; estimated cost rose about $0.01494 -> $0.01634, without justifying the added candidates. |
| 4 | Proposition-level verifier | Rejected | 3/5 positive result, with false-pass indices 81 and 121; parse-failure indices 20 and 77. |
| 5 | Scalar CPS | Rejected | First-30 p25 was 9.477 versus 6.930 for the disjoint stratified-30 p25; scalar CPS is length-sensitive and unstable. |
| 6 | Raw additive regression | Accepted as research only | Chars-only combined-60: `duration=1.877565+0.051104*chars`, R2 0.872356, MAE 0.378399, RMSE 0.545698; 5-fold MAE 0.390463, RMSE 0.556578. Raw chars+punctuation combined coefficients were `1.553941846/0.049656856/0.176788939`; new60 held-out MAE 0.389846, RMSE 0.544727, R2 0.932680. Held-out p75 margin coverage was only 66.7-76.7% in the prior two-sample transfer. |
| 7 | Post-silence additive model | Accepted opt-in | 120 records across 3 episodes; coefficients `(0.6234919816, 0.0517855126, 0.1424302413)`; LOEO MAE 0.208, RMSE 0.288, R2 0.972; p90/p95 margins 0.311711/0.431555 sec; held-out selector counts were 99/1/19/1. |
| 8 | Direct Flash duration selection | Accepted opt-in, not default | Real paid cost was $0.00076104: four actionable additional targets, two safely unchanged, and two actual fits. One of the fits had semantic loss until Pro repair; therefore this remains opt-in, with proxy measurement and nondeterminism caveats. |
| 9 | Standalone Pro duration gate | Accepted opt-in | Deterministic decisions were 517/875; no-gate was 1/4 versus gate 2/4 on semantic + actual-fit outcomes. Costs were $0.01514621 without the gate and $0.00797013 with it, but request and nondeterminism differences prevent attributing the difference to the gate. |
| 10 | Combined profile wiring | Accepted experimental, not default | Smoke cost was $0.01189873 with 0 reasoning tokens, 0 API failures, and 2 parse failures; one target was verified in that stochastic run. Conservative unresolved behavior was retained; the index-42 status bug was fixed offline. Final actual Edge measurement remains pending because of service/cache state. |
| 11 | Multi-cue semantic units | Accepted for final barrier | The tracked benchmark24 control validates unit-level final-barrier behavior; Pro remains cue-level. Grouping is conservative (maximum three cues, timing gap at most 0.3 seconds, textual continuation, unknown timing single-cue), with atomic fallback. |
| 12 | Actual-TTS final measurement plus one bounded compaction/reverify | Pending | Final production-equivalent TTS measurement and the single bounded recovery path have not yet been completed as release evidence. |
| 13 | Official Codex independent verifier | Pending | Official Codex/ChatGPT Pro path has not yet supplied an independent verifier evaluation. |
| 14 | OpenCode OAuth independent semantic barrier | Accepted experimental opt-in, not default | OpenCode 1.18.9/Luna CLI runs on E01 and E04 passed the recorded full-context controls with fail-closed restoration; artifacts are temporary and non-durable. This is third-party OpenCode OAuth evidence, not OpenAI endorsement or release evidence. |
| 15 | E04 prior-report OpenCode barrier | Accepted experimental opt-in, not default | Only prior-verified 517/875 were sent and both passed; unresolved 222/641 were not sent and restored; one request, 0 failures, approximately 10.19s. |

## 2026-08-02 benchmark60 control calibration

The reproducible tracked fixture is `skill_test/semantic_benchmark60/`, with
benchmark24 retained as its historical subset. The Sol control run dated
2026-08-02 qualified all 57 clear cases and all 3 diagnostic cases (57/57 and
3/3), with zero schema or transport failures in 17 requests. Usage was 172395
input, 2926 output, 2792 reasoning, 459264 cache-read, and 637377 total tokens;
provider-reported cost was 0 on the subscription route. Raw results are at
`/tmp/opencode/model_screening/benchmark60_sol_control_run1_2026-08-02/`.

The Luna control run scored 53/57 clear cases and was technically clean. The
exact false rejects were E01/114 and E01/531; uncertain outcomes were E03/220
and E04/57. Usage was 278413 input, 3079 output, 4284 reasoning, 353280
cache-read, and 639056 total tokens in 17 requests; provider-reported cost was
0 on the subscription route. Raw results are at
`/tmp/opencode/model_screening/benchmark60_luna_control_run1_2026-08-02/`.
Luna is no longer the qualified benchmark60 default.

Benchmark60 is now structurally validated and Sol control-model calibrated;
benchmark24 is historical evidence only.

## 2026-08-02 benchmark24 model screening

The reproducible fixture is `skill_test/semantic_benchmark24/`; raw result files
remain dated, local artifacts under `/tmp/opencode/model_screening/`. It contains
24 manually adjudicated cases: 21 clear gate cases and 3 diagnostic debatable
cases. Unit grouping needs full episode context. Results below are screening
evidence, not a release claim; provider costs are informational.

The current tracked fixture is now `skill_test/semantic_benchmark60/`, which
retains benchmark24 as its historical subset and adds 36 manually reviewed clear
cases (18 pass / 18 fail). Benchmark60 calibration is recorded above; this
section remains historical benchmark24 screening evidence.

* **Luna strict text (`openai/gpt-5.6-luna`)**: historical benchmark24
  `skill_test/semantic_benchmark24/` (21 clear / 3 diagnostic) control, clear
  21/21 and overall 23/24,
  with 0 schema/transport failures and 8 requests. Usage was 190028 input,
  1308 output, 2186 reasoning, 105984 cache-read, 299506 total tokens; provider
  cost 0 on the subscription route. It is not the qualified benchmark60
  default.
* **Terra20**: clear 20/21; rejected because of E01/114.
* **GLM5.2/Ollama**: E01 clear 7/7 after 128 diagnostic, but E03/E04 transport
  failures prevented qualification. StructuredOutput is an explicit compatible
  option, not the default.
* **DeepSeek V4 Pro/Ollama**: clear 20/21; overstrict E04/306.
* **Nemotron Super free**: revised clear 19/21 because of E04/641 and transport
  875.
* **Gemini3 Flash**: revised clear 20/21, with schema/unresolved outcomes and
  high request/cost overhead; not qualified.
* **MiMo base, Gemini2.5, Qwen, Gemma, GPT-OSS, Mistral, and multiple Orca
  routes**: rejected or blocked by combinations of semantic false rejects,
  schema/unresolved outcomes, and provider transport/tool incompatibility.
  Semantic failures are separate from route failures: a blocked route is not
  evidence that its model is semantically weak. The pre-confirmation direct
  screen total was `$0.27371521485`; later confirmation costs are separate.

The verifier's StructuredOutput is an OpenCode synthetic schema-validated tool,
not provider-native `response_format`. Text mode uses the strict local parser and
fails closed. The benchmark60 Sol decision therefore defaults to strict text
while keeping StructuredOutput available as an explicit route option. Provider
costs above are informational for the OpenCode route, not billing-authoritative.

### Direct OpenRouter live validation

The OpenRouter adapter's native smoke was technically validated after installing
the already-pinned `aiohttp` dependency. The following are standalone direct
OpenRouter semantic-unit runs on the 24-case benchmark (21 clear cases and 3
diagnostic cases), not combined-pipeline results:

* **Gemini2.5 Flash-Lite**: E01 clear 3/7 and overall 5/10 in one request,
  cost `$0.000621225`; rejected for four clear false passes, with diagnostic
  behavior also recorded. There were no technical failures.
* **Qwen3.5 Flash**: the first run was 21/21, but two independent default,
  temperature-0 runs were each 20/21 because of false rejects E01/114. It is
  not stable or qualified.
* **MiMo2.5 Pro production run**: 21/21; one schema batch was recovered
  atomically through singleton units, 12 requests, cost `$0.1005397668`.
* **MiMo2.5 Pro confirmation**: 21/21, zero technical failures, 8 requests,
  cost `$0.0776182572`; this is the qualified direct default.

The listed costs are billing-authoritative OpenRouter credit amounts. Raw local
result directories remain outside the repository and contain no key or account
data.

## Do not repeat without new evidence

Do not repeat the rejected multi-candidate editor, proposition-label hard checks,
or scalar CPS approach without a changed design and new evidence addressing the
failure mode. Do not make in-sample p75 margin safety claims, and do not treat a
same-model automated score as human semantic quality. A new experiment must state
what evidence would overturn the prior decision before spending API or
subscription budget.

## Honest readiness assessment

The current evidence supports roughly **9/10 for timing/selection on tested
same-show data**, **7/10 for semantic protection**, and **5/10 for
completion/coverage**. Overall readiness is approximately **6-7/10**, not close
enough to claim 10/10. The estimate is limited by small A/B samples, same-show
sampling, limited independent episodes/genres/voices, proxy duration work, and
pending actual-TTS measurement. Small OpenCode independent evidence exists, but
broader independent-model validation and official Codex evaluation remain
pending. It is a readiness signal, not a release metric.

## Reproduction commands

Use repository-root commands and the project convention shown below. Paths in
angle brackets are placeholders; in particular, `<EXPERIMENTAL_PROFILE_PATH>`
does not imply that a production profile is committed.

### Calibration: raw and post-silence

```bash
uv run --python subs_env/bin/python -m utils.calibrate_tts_budget <INPUT_JSON> \
  --output-dir output/tts_calibration
uv run --python subs_env/bin/python -m utils.calibrate_tts_budget <INPUT_JSON> \
  --output-dir output/tts_calibration --duration-mode post_silence \
  --silence-dbfs -35 --silence-threshold-sec 0.5 \
  --silence-target-sec 0.2 --silence-frame-ms 10
```

### Direct Flash profile selection

```bash
uv run --python subs_env/bin/python -m utils.shorten_subtitles_deepseek <INPUT_JSON> \
  --duration-profile <EXPERIMENTAL_PROFILE_PATH> --duration-fit-ratio 1.0
```

### Standalone Pro gate

```bash
uv run --python subs_env/bin/python -m utils.review_shortened_subtitles_deepseek \
  <ORIGINAL_JSON> <SHORTENED_JSON> \
  --duration-profile <EXPERIMENTAL_PROFILE_PATH> --duration-fit-ratio 1.0
```

### Combined experimental profile

```bash
uv run --python subs_env/bin/python -m utils.shorten_review_pipeline <INPUT_JSON> \
  --context-source <FULL_EPISODE_ANALYZED_JSON> \
  --duration-profile <EXPERIMENTAL_PROFILE_PATH> --duration-fit-ratio 1.0 \
  --output-dir output/experimental_combined
```

Relevant local verification commands are:

```bash
uv run --python subs_env/bin/python -m pytest -q
uv run --python subs_env/bin/python -m pytest -q tests/test_calibrate_tts_budget.py
uv run --python subs_env/bin/python -m pytest -q tests/test_review_shortened_subtitles_deepseek.py
uv run --python subs_env/bin/python -m pytest -q tests/test_shorten_review_pipeline.py
uv run --python subs_env/bin/python -m pytest -q tests/test_shorten_subtitles.py
uv run --python subs_env/bin/python -m pytest -q tests/test_shortening_domain.py
```

Paid API runs are nondeterministic. Preserve the exact flags, model discovery
result, fixture hashes, usage reports, and failure reports when a run is used as
evidence.

## Next experiment order and commit boundary

1. Commit the current experimental opt-in foundation after review; it is ready
   to commit as experimental work and must not become a default by implication.
2. Keep the unit foundation accepted for the opt-in final barrier based on the
   Luna 21/21 control. A full E03 live rerun remains blocked by provider
   transport and is additional operational validation, not an acceptance blocker.
3. Calibrate the integrated unit behavior, then screen models; model screening
   comes after calibration.
4. Add the actual-TTS final gate and one bounded compaction/reverify path, with
   production-equivalent measurement and explicit unresolved accounting.
5. Broaden model-family, episode, genre, and voice evaluation, including the
   independent Codex/ChatGPT Pro verifier path.

## Cost and accounting

### Independent barrier benchmark notes

The validated benchmark recorded these informational OpenCode results:

| Run | Result | Requests/tokens | Provider cost |
|---|---|---|---|
| Historical cue-level benchmark24 before E01/128 confidence revision - DeepSeek V4 Pro (`deepseek-v4-pro`) | historical clear 22/22, overall 23/24; pass 10/10, fail 13/14; only debatable E01/121 disagreement | 5 requests; input 9407, output 1221, reasoning 13125, cache-read 280960, total 304713; zero schema/transport | `$0.006120548` |
| MiniMax M3 | clear 20/22, overall 21/24; false rejects E01/114 and 128; rejected as worse and costlier | zero schema/transport reported | `$0.05551218` |
| E03 full-context comparison | legacy restored 502/1054; calibrated restored 502 and kept 1054 | 2 requests; input 1527, output 282, reasoning 1115, cache 112384, total 115308; zero schema/transport | `$0.0009196152` |

All OpenCode costs above are informational provider-reported metadata, not
billing-authoritative amounts. They must not be added to the combined pipeline's
`estimated_cost_usd`, which remains Flash API plus Pro API cost.

Direct DeepSeek API calls have token-based costs and emit usage/cost estimates;
record the model, cache state, token counts, pricing label, and timestamp with
each paid run. Subscription-backed calls through local Codex or OpenCode are
accounted as subscription usage rather than stable per-token API pricing. OpenCode
`info.cost`, when present, is provider-reported metadata with unspecified billing
semantics; never treat it as a guaranteed invoice, subscription charge, or actual
billed cost. Record
the small A/B costs as observed run totals, but do not claim stable pricing or
extrapolate them to production without a current provider price source. Keep
direct API costs separate from subscription calls and from local CPU/TTS/cache
costs.
# OpenRouter verifier experiment

The direct OpenRouter verifier sends semantic-unit prompts with the native strict
JSON schema and fails closed on schema incompatibility, refusal, transport
errors, or malformed usage. Configure the key with
`OPENROUTER_API_KEY` or `--api-key`; optional attribution can use
`OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE`.
# Ollama Cloud experiment status

The direct `/api/chat` transport works, but Ollama Cloud has no native
structured output for this route. The client sends text only and validates it
with the local strict parser; `schema_json` below is an experiment label, not
a provider-native schema contract. The adapter remains available for explicit
experiments, but `--model` is required and no stable qualified/default model
exists.

The live findings below are screening evidence, not a release claim. They are
subscription usage, not authoritative per-call billing:

* **GLM5.2**: the batch-10 `schema_json` run was not enough to qualify it. The
  production-like batch-4/schema-1 run left 8 technical outcomes unresolved
  after 13 requests and 22,105 tokens, so it is unusable for a strict
  verifier. Results:
  `/tmp/opencode/model_screening/direct_ollama_glm52_smoke_2026-08-02/` and
  `/tmp/opencode/model_screening/direct_ollama_glm52_schema_recovery_2026-08-02/`.
* **Nemotron Super**: the first run cleared 21/21 in 8 requests and 29,092
  tokens, but independent confirmation failed with clear semantic errors
  E01/106 and E01/114 plus transport/schema failures. It is unstable and not
  qualified. Results:
  `/tmp/opencode/model_screening/direct_ollama_text_json_screening_2026-08-02/nemotron-3-super/`
  and `/tmp/opencode/model_screening/direct_ollama_nemotron3super_confirmation_2026-08-02/`.
* **Qwen3.5:397B**: full run 20/21, with clear false reject E01/114, zero
  technical failures, 8 requests, and 53,804 tokens. Reject. Results:
  `/tmp/opencode/model_screening/direct_ollama_qwen35_397b_full_2026-08-02/`.
* **GPT-OSS 120B**: 6/7 with clear false reject E01/106. **DeepSeek V4 Pro**:
  6/7 with clear false pass E01/77. Both are rejected. **Gemma** and
  **Mistral** were schema-blocked. Results for this screen are under
  `/tmp/opencode/model_screening/direct_ollama_text_json_screening_2026-08-02/`.

There is no stable qualified/default Ollama Cloud model. A first-run success
must not be confused with qualification. Subscription status also does not
provide an authoritative per-call cost.
