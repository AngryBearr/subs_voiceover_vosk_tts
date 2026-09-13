# `shorten_file`: ordinary full-file shortening

SH-03 adds a single-model command for a complete `.json` or `.srt` file. SH-04
adds fail-closed resume for this command. It uses a frozen Vosk calibration
profile and the resilient generation/review runner; it does not run an
experiment matrix or repetitions.

Activate the Linux project environment before running commands:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.srt \
  --output-dir output/short-file-run \
  --model openai/gpt-5.6-luna \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json
```

Resume the same run with the same original input, output directory, model,
profile, and every other option from the original command:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.srt \
  --output-dir output/short-file-run \
  --model openai/gpt-5.6-luna \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --resume
```

The profile is required, as are `--model` and `--output-dir`. Supported model
IDs are `openai/gpt-5.6-luna`, `openai/gpt-5.6-sol`,
`deepseek/deepseek-flash`, and `deepseek/deepseek-v4-flash`. The selected one
model is used for both the generator and arbiter stages (one model does not
mean one HTTP request). OpenCode uses the default `medium` variant and
DeepSeek uses `off`; `--variant` may select another supported value.

## Calibration and length basis

Planning, local candidate checks, and the optional length tool all use
`normalized_tts_text`, not the display text. The frozen profile is checked
against the actual model files, voice, rate, and normalizer identity. The
profile's `char_linear` estimate is a calibrated character estimate, not a
guarantee that the final audio will fit. Short timing-risk cues receive an
actual original-audio preflight; a measured fit avoids an LLM request, while
an unresolved short cue remains unresolved.

There is no hidden 13-character-per-second fallback in this command. The
requested rate is finite and restricted to `[1.0, 1.5]`, and the exact
matching profile must be present.

## Input and outputs

JSON is read as an array of objects with UTF-8-SIG support. Unknown keys,
JSON indexes, and timing fields are retained. SRT is parsed with the existing
fail-closed parser. Its export uses canonical chronological numbering, while
JSON source indexes remain unchanged. A live run writes the runner's
`report.json`, `subtitles.json`, `subtitles.srt`, and checkpoints; incomplete
runs return a non-zero exit code and retain their checkpoint metadata.

## Resume safety and limitations

Resume replays saved generation, review, repair, and recheck receipts locally.
Completed receipts, including completed receipt errors, are replayed as their
recorded responses or errors and are never resent. Accepted checkpoint exports
are never regressed during replay. The replay cursor covers the complete file,
including non-contiguous target IDs, and only after it is exhausted may live
work append new checkpoints.

`requests.jsonl`, `calls.jsonl`, and `resume_events.jsonl` are append-only
journals. A `resume_events.jsonl` row records each explicit authorization to
replace an unknown request. `run_manifest.json` is an immutable compatibility contract;
`resume_baseline.json` stores the post-preflight plans and records and is
reused on resume. `resume_status.json` separately records unknown-request
details and known accounting; `status: processed` means that workflow
processing is exhausted, not that expense reconciliation is complete.

Compatibility is strict: the source bytes and canonical source data, all
configuration, profile bytes and canonical profile data, model and model
path, voice, rate, text normalizer, `PROMPT_VERSION`, and actual source-code
hashes (including costing code) must match. Relocating the source or profile
path, changing the subtitle source even by whitespace, or changing
implementation comments in hashed source code invalidates resume intentionally.
The command fails closed rather than guessing that old receipts are safe.

There is no automatic migration for legacy pre-SH04 directories or dry-run
directories. Malformed or truncated manifests, JSON, or JSONL logs are
rejected without modification; preserve the directory for inspection. This
does not promise a free restart. A nonblocking per-run lock prevents concurrent
processes from using one run directory.

If preflight is interrupted before the baseline is written and no request
journal exists, local preflight may restart in a new
`preflight_attempts/<attempt-id>` directory; partial audio from the old
attempt is preserved. Once a valid baseline exists, it is reused and preflight
is not repeated.

An unknown request is blocked by default for the entire run. Exit 2 prints its
request ID(s); inspect `resume_status.json`, make the risk decision, then
explicitly add `--retry-unknown REQUEST_ID` (repeat the flag for each ID that
is approved). There is no blanket retry or `--retry-all`. The old ID remains
unmatched and its known costs remain in accounting; an authorized retry gets a
new UUID, and if that retry is unknown it requires a new approval. Historical
IDs cannot reauthorize later attempts. Exactly-once external API delivery or
billing cannot be guaranteed. `--retry-unknown` applies only to unknown sent
requests, never to completed receipt errors.

Even when an authorized retry succeeds, any remaining historical billing
unknown keeps `report.json` at `run_state: "incomplete"` and
`accounting_complete: false`, with exit 1. `resume_status.json` may then say
`status: "processed"`: that means the workflow is exhausted, not that costs
are reconciled. Exit 0 means normally completed; exit 1 means failed or
incomplete; exit 2 means blocked by unknown requests.

## Offline dry-run

Use a separate fresh directory for each new run or dry-run; use the existing run directory only with --resume. `--dry-run` performs all
offline validation and writes only `preflight.json` with all plans, selected
IDs, calibration metadata, and an explicit statement that no LLM or acoustic
evidence exists yet:

```bash
uv run --active --no-sync -m utils.shorten_file INPUT.json \
  --output-dir output/short-file-preflight \
  --model deepseek/deepseek-flash \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --dry-run
```

The dry-run directory is a fresh offline planning directory and is not
resumable as a live run; `--dry-run` cannot be combined with `--resume`. Use a
separate fresh directory for the live command. The command performs no API
credential lookup, server or backend construction, network call, calibration
training, or Vosk inference in dry-run mode.
