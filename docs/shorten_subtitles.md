# Shorten Subtitles — Subtitle Shortening Tools

Two approaches to shorten subtitles that are too long for their duration:

1. **Direct DeepSeek API** — `utils.shorten_subtitles_deepseek`
2. **OpenCode HTTP API** — `utils.shorten_subtitles_opencode`

Both use the same shortening logic and iterative loop from `utils.shorten_helpers`.

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

Subtitles with `analysis.is_checked = true` and `mismatch_ratio > threshold` will be shortened.

---

## Common CLI Arguments

Both scripts accept these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input JSON file with analyzed subtitles |
| `--model` | `deepseek-v4-flash` | Model name |
| `--threshold` | `1.5` | Mismatch ratio threshold for shortening |
| `--context-window` | `3` | Number of surrounding subtitles for context |
| `--max-iterations` | `15` | Maximum number of iterations |
| `--min-words` | `3` | Minimum words in shortened text |
| `--output-dir`, `-o` | `output/shortened` | Output directory |
| `--reasoning-effort` | `max` | Reasoning effort: `minimal`, `low`, `medium`, `high`, `max` |

---

## Method 1: Direct DeepSeek API

Uses DeepSeek API directly via OpenAI-compatible client.

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
    --max-iterations 5 \
    --threshold 1.3 \
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

### Available Models

```
deepseek-v4-flash
deepseek-v4-pro
```

---

## Method 2: OpenCode HTTP API

Starts an `opencode serve` instance and sends prompts via HTTP API. Avoids cold start on multiple requests.

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

---

## How It Works

1. **Analyze** — Run `analyze_subtitles` to find critical subtitles
2. **Select** — Pick subtitles with `mismatch_ratio > threshold`
3. **Shorten** — Send prompts to LLM with context (surrounding subtitles)
4. **Validate** — Check shortened text meets requirements (min words, not just "...")
5. **Apply** — Replace original text with shortened version
6. **Re-analyze** — Run analysis again to check progress
7. **Repeat** — Until no subtitles need shortening or max iterations reached

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
    --max-iterations 10 \
    --concurrency 8
```

### Shorten with OpenCode, orcarouter provider

```bash
uv run -m utils.shorten_subtitles_opencode subs_analyzed.json \
    --model orcarouter/deepseek/deepseek-v4-flash \
    --threshold 1.5 \
    --max-iterations 5
```

### Use existing opencode server

```bash
# Terminal 1: start server
opencode serve --port 4096

# Terminal 2: run shortening
uv run -m utils.shorten_subtitles_opencode subs_analyzed.json \
    --server-url http://localhost:4096
```
