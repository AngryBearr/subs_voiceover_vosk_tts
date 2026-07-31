# Shorten Subtitles — Subtitle Shortening Tools

Two approaches to shorten subtitles that are too long for their duration:

1. **Direct DeepSeek API** — `utils.shorten_subtitles_deepseek`
2. **OpenCode HTTP API** — `utils.shorten_subtitles_opencode`

Both use the same shortening logic and iterative loop from `utils.shorten_helpers`.

The recommended orchestration command runs Flash shortening and then uses direct
DeepSeek V4 Pro to inspect every Flash-changed subtitle and every unchanged subtitle
that remains critical.

---

## Prerequisites

- Python 3.11+
- Virtual environment: `subs_venv`
- For DeepSeek API: API key in `.env` file or `DEEPSEEK_API_KEY` environment variable
- For OpenCode: `opencode` CLI installed (`opencode --version`)

---

## Input Format

JSON file with analyzed subtitles (output of `utils.analyze_text`):

```json
[
  {
    "text": ["Subtitle text here"],
    "index": 1,
    "start": 0,
    "end": 2000,
    "analysis": {
      "duration_sec": 2.0,
      "estimated_sec": 3.5,
      "mismatch_ratio": 1.75,
      "extended_mismatch_ratio": 1.75,
      "words_count": 8,
      "is_short_segment": false,
      "is_checked": true,
      "is_critical": true
    }
  }
]
```

Checked subtitles use `extended_mismatch_ratio` when present, otherwise
`mismatch_ratio`; values above the threshold require shortening.

---

## Common CLI Arguments

Both scripts accept these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input JSON file with analyzed subtitles |
| `--model` | `deepseek-v4-flash` | Model name |
| `--threshold` | `1.5` | Mismatch ratio threshold for shortening |
| `--context-window` | `3` | Number of surrounding subtitles for context |
| `--context-source` | (disabled) | Full-episode analyzed JSON used only for real neighboring context |
| `--max-iterations` | `3` | Maximum number of iterations |
| `--min-words` | `3` | Minimum words in shortened text |
| `--output-dir`, `-o` | `output/shortened` | Output directory |
| `--avg-chars-per-sec` | `13` | TTS speed used to calculate character budgets |
| `--batch-size` | `5` | Number of subtitles per API call (max: 25) |
| `--stuck-threshold` | `5` | Iterations without change before skipping subtitle |
| `--early-stop-patience` | (disabled) | Stop after N consecutive iterations with < 2 changes |

---

## Method 1: Direct DeepSeek API

Uses DeepSeek API directly via OpenAI-compatible client.

By default it explicitly disables thinking and requests JSON mode. `--thinking-effort high|max`
is an explicit, more expensive opt-in (DeepSeek does not provide savings for low/medium effort).
Usage and the V4 Flash direct-API estimate (cache miss $0.14/M, hit $0.0028/M,
output $0.28/M) are saved as `{stem}_final.usage.json`.
The estimate is explicitly labeled `pricing_model: deepseek-v4-flash`; custom models
still require their own price interpretation.

### Setup

Add to `.env`:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### Usage

```bash
# Basic usage
uv run -m utils.shorten_subtitles_deepseek input.json

# With options
uv run -m utils.shorten_subtitles_deepseek input.json \
    --model deepseek-v4-flash \
    --concurrency 10 \
    --max-iterations 3 \
    --threshold 1.3 \
    --batch-size 10 \
    -o output/my_shortened

# Override API key
uv run -m utils.shorten_subtitles_deepseek input.json --api-key sk-xxx

# Custom base URL
uv run -m utils.shorten_subtitles_deepseek input.json --base-url https://api.deepseek.com
```

### Arguments (additional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--concurrency` | `5` | Max parallel API calls |
| `--api-key` | (from .env) | DeepSeek API key |
| `--base-url` | `https://api.deepseek.com` | API base URL |
| `--thinking-effort` | disabled | Explicit opt-in: `high` or `max` |

### Available Models

```
deepseek-v4-flash
deepseek-v4-pro
```

---

## Method 2: OpenCode HTTP API

Starts an `opencode serve` instance and sends prompts via HTTP API. Avoids cold start on multiple requests.
The documented server API does not guarantee provider thinking/JSON controls or token usage;
the non-thinking and cost guarantees above apply only to the direct DeepSeek path.

### Usage

```bash
# Basic usage (starts server automatically)
uv run -m utils.shorten_subtitles_opencode input.json

# Specify model with provider
uv run -m utils.shorten_subtitles_opencode input.json \
    --model opencode-go/deepseek-v4-flash

# Use orcarouter provider
uv run -m utils.shorten_subtitles_opencode input.json \
    --model orcarouter/deepseek/deepseek-v4-flash

# Use existing server
uv run -m utils.shorten_subtitles_opencode input.json \
    --server-url http://localhost:4096

# Custom server port
uv run -m utils.shorten_subtitles_opencode input.json \
    --port 5000 \
    --hostname 127.0.0.1
```

### Arguments (additional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--concurrency` | `5` | Max parallel API calls |
| `--port` | (random) | Port for opencode server |
| `--hostname` | `127.0.0.1` | Hostname for opencode server |
| `--server-url` | (auto-start) | Use existing server instead of starting new one |

### Model Format

```
model_name                              -> provider: opencode-go
provider/model_name                     -> provider: provider
provider/vendor/model_name              -> provider: provider, model: vendor/model_name
```

Examples:

```
deepseek-v4-flash                       -> opencode-go/deepseek-v4-flash
opencode-go/deepseek-v4-flash           -> opencode-go/deepseek-v4-flash
orcarouter/deepseek/deepseek-v4-flash   -> orcarouter/deepseek/deepseek-v4-flash
openrouter/deepseek/deepseek-v4-flash   -> openrouter/deepseek/deepseek-v4-flash
```

### Available DeepSeek Models in OpenCode

```
opencode/deepseek-v4-flash
opencode/deepseek-v4-flash-free
opencode-go/deepseek-v4-flash
opencode-go/deepseek-v4-pro
openrouter/deepseek/deepseek-v4-flash
orcarouter/deepseek/deepseek-v4-flash
```

---

## Output Files

Each iteration produces:

- `{stem}_analyzed.json` — Initial analysis result
- `{stem}_iter{NN}.json` — Result after each iteration
- `{stem}_final.json` — Final result
- `{stem}_final.usage.json` — Direct DeepSeek token usage and cost estimate

## Direct DeepSeek V4 Pro review

### Semantic quality pass

Production Pro review first runs the quality-v3 original-only semantic planner.
It receives source text, indexed original-only context, budgets, and local
deterministic requirements - never Flash candidates, lineage, or editor data.
The strict endpoint JSON is parsed locally into an immutable plan of atomic
propositions and anchored requirements. Character counts and semantic
comparison use no function tools; they remain local checks. Planner batches are
capped at two targets. If a batch is malformed or unavailable, each target gets
exactly one original-only individual retry; successful retries continue through
the quality stages, while failed retries remain unresolved. Oversized single
targets are not retried. Planner failure is fail-closed: the unchanged candidate
is reported unresolved. Standalone review
enables this stage by default; use `--no-semantic-planner` to opt out. Direct
programmatic `review_subtitles` keeps planner-off as its backward-compatible
default.

The Pro pass retains the complete immutable planner guidance alongside the
deterministic candidate-specific requirements. Requirements carry an exact
source anchor and an English obligation, and are sent with the immutable plan
to critic, editor, compaction, verifier, and bounded repair. Explicit verifier
required checks remain the unique deterministic risk hints plus unique critic
issues; planner-only guidance is still visible and participates in the general
two-way entailment verdict, but is not promoted into a required-check label.
In addition to time, modality, causality, comparison, alternatives, and
references, `spatial` is a required-check gate for relations such as `вокруг
лагеря` and `рядом с нами`. A spatial or reference anchor may be paraphrased,
but cannot be replaced by ambiguous `там`, `туда`, or `это` without an
unambiguous equivalent in full context.

If a repaired candidate passes every semantic required check and formal
validation but re-verification reports only grammar/naturalness, the repair is
retained and reported as unresolved. Semantic damage, missing checks,
uncertainty, and formal errors still use the normal rollback policy.

### Recommended Flash + Pro pipeline

```bash
uv run -m utils.shorten_review_pipeline skill_test/subs_analyzed.json \
  --context-source skill_test/S01_E01_ru_analyzed.json \
  --api-key sk-xxx --output-dir output/skill_test_shortened_reviewed
```

The API key is shared by both stages (and may instead come from
`DEEPSEEK_API_KEY`). Combined-pipeline defaults are one non-thinking Flash iteration,
Flash batch 6/concurrency 3, then Pro context window 3, concurrency 3, and
a conservative 50,000-token input estimate. Stage artifacts are under `flash/`
and `pro/`; the refreshed final is `{input_stem}_reviewed.json` at the output
root, alongside combined pipeline report and usage files. If Pro fails entirely,
the command exits nonzero while retaining the Flash final, usage, and failure
report.

Use pipeline `--pro-concurrency N` or standalone reviewer `--concurrency N` to
change the Pro request limit (default: `3` for both). Independent requests within
each Pro stage run in bounded parallel, while the critic → editor → targeted
compaction → verifier → repair → reverify stage barriers remain in place.

The reviewer selects the unique-index union of Flash-changed and remaining-critical
targets. Pro uses three separated roles: a non-thinking critic evaluates every target;
a non-thinking editor receives only critic failures, formal failures, and
remaining-critical targets (batches of at most two); a fresh non-thinking blind
verifier sees only original/full context/candidate. If a non-empty verifier,
compaction, or repair response is malformed, planner-enabled production review
makes at most one non-thinking, same-stage schema retry per target; a target that
remains malformed keeps the existing parse-failure path. A verifier failure gets at
most one single-target non-thinking repair and one fresh blind verification. API or malformed
responses never count as verified, and the safest locally valid candidate is retained
but reported unresolved.

Experiment 1 applies a semantic hard gate to known risks. The final verifier must return
one strict, explicit check for every deterministic risk hint and relevant critic issue
selected in stable order, with at most six bounded diagnostic labels; the complete
semantic plan and merged requirements are still evaluated holistically by two-way
entailment, and omitted labels are not assumed passed;
missing, duplicate, malformed, or failed required checks cannot verify a target. A critic
failure or known semantic risk also requires a locally valid editor or repair candidate,
so a generic pass over an unchanged lossy Flash candidate is insufficient. If an editor
candidate is valid except for exceeding `max_chars`, it is retained diagnostically and
gets at most one single-target non-thinking compaction request, with one schema-only retry
when planner mode is enabled and the response is non-empty but malformed. The compacted result is
formally validated before blind verification. A hard-gated target that still lacks valid
recovery may use the existing single bounded repair attempt, but there are no additional
retries. Reports expose targeted-compaction request counts, decisions, stage errors, and
rejected candidates with reasons in per-target lineage. When any gate remains unmet, the
safest prior locally valid candidate is retained and the target is conservatively reported
unresolved. Repair candidates remain provisional until their fresh reverification passes
all required checks; otherwise the reviewer restores the prior candidate and its verifier
fields while retaining both attempts and the rejected repair in diagnostics.

With `--context-source`, windows are based on each unique subtitle `index` position in
the full episode. Non-target neighbors come from that source while current target text
is overlaid by index. The source is rejected for duplicate indices, missing targets, or
target/source text mismatch. Reports label this `context_mode: full` and otherwise use
`context_mode: sparse`.

```bash
# Review the current skill test result (recommended default)
uv run -m utils.review_shortened_subtitles_deepseek \
  skill_test/subs_analyzed.json \
  output/shorten_skill_test_last_retry_paid/subs_analyzed_final_final_final_final.json \
  --context-source skill_test/S01_E01_ru_analyzed.json \
  --batch-size 4 --concurrency 3 -o output/reviewed_skill_test
```

The legacy `--thinking-mode` and pipeline risk-threshold options remain accepted for
command compatibility but do not override quality routing: critic/editor/verifier/
compaction/repair are always non-thinking. Planner-enabled malformed responses get at
most one same-stage schema retry; API-empty responses do not retry.

Pro receives `original_text` and `current_text` for context. Reports contain stage
usage, per-target Flash/editor/compaction/repair/final lineage (including rejected
candidates), critic and verifier verdicts/issues and required checks,
and `automated_verified_count` (not a human quality rate). Combined usage sums each Pro
stage once. A top-level `shortening.status` is resolved only when the target is locally
within budget and passes final blind verification; otherwise the pipeline uses
`completed_with_unresolved` and lists unresolved indices.

The default model is `deepseek-v4-pro`. The default batch size is 4, and batches use an
estimated 50,000 input-token limit; use `--batch-size`, `--max-input-tokens`, and
`--context-window` to tune them. A full-file one-request design is intentionally
avoided for reliability even though the official model context is 1M tokens.
The tokenizer-free estimate deliberately treats every input character as a token and
adds fixed message overhead, making it conservative for Cyrillic. The default
`--target-ratio 1.5` must match the threshold/target ratio used by the Flash shortening
run; override it when Flash used another value. Refreshed timing metadata from the
shortened file is preferred, with original timing as fallback.

For shortened input `name.json`, outputs are predictable:

- `name_reviewed.json` — final subtitles;
- `name_review.report.json` — selection, routing, acceptance, fallback, and errors;
- `name_review.usage.json` — cache hit/miss, completion/reasoning tokens and Pro cost.

The Pro estimate uses direct prices: cache miss $0.435/M, cache hit $0.003625/M,
and output $0.87/M, explicitly labeled `pricing_model: deepseek-v4-pro`.

### Real API hard-case smoke set

`skill_test/subs_analyzed_hard_cases.json` is a 10/30-case smoke set selected from
the real run report. It covers unresolved repair/reverify (342), editor-too-long
and compaction (394), semantic critic failures (358/380), time/naturalness (391),
modality/reference (346/333), context restoration (379), comparison (352), and
spatial/time anchors (381).

Run the paid, nondeterministic comparison from the repository root with the same
explicit production flags as the full run:

```bash
uv run --python subs_env/bin/python -m utils.shorten_review_pipeline \
  skill_test/subs_analyzed_hard_cases.json \
  --context-source skill_test/S01_E01_ru_analyzed.json \
  --flash-model deepseek-v4-flash --pro-model deepseek-v4-pro \
  --threshold 1.5 --avg-chars-per-sec 13 \
  --flash-max-iterations 1 --flash-batch-size 6 --flash-concurrency 3 \
  --pro-batch-size 4 --pro-concurrency 3 --pro-thinking-mode auto \
  --pro-risk-threshold 35 --pro-context-window 3 \
  --pro-max-input-tokens 50000 \
  --output-dir output/skill_test_hard_cases_real
```

The full 30-item set remains the release regression suite; this smaller set is
intended for fast comparisons. The API call is paid and nondeterministic.

---

## How It Works

1. **Analyze** — Run `analyze_subtitles` to find critical subtitles
2. **Select** — Use `extended_mismatch_ratio` when present, otherwise `mismatch_ratio` (excluding stuck)
3. **Batch** — Group subtitles into batches of `--batch-size`
4. **Shorten** — Send batched prompts to LLM with context (surrounding subtitles)
5. **Validate** — Require changed, strictly shorter text within `max_chars`, preserving numbers and explicit negations
6. **Apply** — Replace original text with shortened version
7. **Re-analyze** — Run analysis again to check progress
8. **Repeat** — Until no subtitles need shortening or max iterations reached

### Adaptive Strategy

The system uses two strategies depending on the iteration number:

- **Iterations 1-2**: "Shorten" — asks the model to shorten the text while preserving meaning
- **Iterations 3+**: "Rephrase" — asks the model to restructure the sentence for a more compact form

When the strategy changes, all tracking counters (stuck detection, early stopping) are reset, giving subtitles a fresh chance with the new approach.

### Stuck Detection

If a subtitle's text hasn't changed for `--stuck-threshold` consecutive iterations (within the same strategy), it's marked as "stuck" and excluded from further processing. This prevents wasting API calls on subtitles that can't be shortened.

The stuck counter resets when the strategy changes (e.g., from "shorten" to "rephrase").

### Early Stopping

If `--early-stop-patience` is set, the loop stops after N consecutive iterations where fewer than 2 subtitles were actually changed. This prevents running unnecessary iterations when the system has converged.

The patience counter resets when the strategy changes or when an iteration produces 2+ changes.

### Batching

Multiple subtitles are sent in a single API call (controlled by `--batch-size`). This reduces:
- Number of API calls (5-25x fewer)
- System prompt duplication (sent once per batch instead of per subtitle)
- Overall cost

The model returns a JSON array with results for each subtitle in the batch.
Overlapping context is represented once in compact JSON rather than repeated for every target.

---

## Comparison

| Feature | DeepSeek API | OpenCode API |
|---------|--------------|--------------|
| Cold start | None | ~3s (server startup) |
| Multiple requests | Each request independent | Reuses server session |
| API key required | Yes | No (uses opencode config) |
| Model flexibility | DeepSeek only | Any opencode provider |
| Cost | Pay per token | Depends on provider |

**Recommendation:**

- **DeepSeek API** — Simple scripts, one-off tasks
- **OpenCode API** — Pipelines with many requests, access to multiple providers

---

## Examples

### Shorten with DeepSeek, aggressive threshold

```bash
uv run -m utils.shorten_subtitles_deepseek subs_analyzed.json \
    --threshold 1.3 \
    --max-iterations 3 \
    --concurrency 8 \
    --batch-size 10
```

### Shorten with OpenCode, orcarouter provider

```bash
uv run -m utils.shorten_subtitles_opencode subs_analyzed.json \
    --model orcarouter/deepseek/deepseek-v4-flash \
    --threshold 1.5 \
    --max-iterations 3 \
    --batch-size 8
```

### Use existing opencode server

```bash
# Terminal 1: start server
opencode serve --port 4096

# Terminal 2: run shortening
uv run -m utils.shorten_subtitles_opencode subs_analyzed.json \
    --server-url http://localhost:4096
```

### Enable early stopping

```bash
uv run -m utils.shorten_subtitles_deepseek subs_analyzed.json \
    --early-stop-patience 3 \
    --batch-size 10
```

### Cost-optimized run

```bash
uv run -m utils.shorten_subtitles_deepseek subs_analyzed.json \
    --batch-size 8 \
    --stuck-threshold 3 \
    --early-stop-patience 3 \
    --max-iterations 3
```

Recommended cheap command for the repository fixture (this command performs paid API calls):

```bash
uv run -m utils.shorten_subtitles_deepseek skill_test/subs_analyzed.json \
  --threshold 1.5 --avg-chars-per-sec 13 --batch-size 8 --max-iterations 3
```

Batch sizes 5–10 are the recommended starting point. Larger batches are not always
cheaper because they increase context and retry scope.
