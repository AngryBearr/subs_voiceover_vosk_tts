# `shorten_file`

`shorten_file` runs one complete `.json` or `.srt` subtitle file through the
existing shortening runner. The normal production path has two fixed
OpenRouter roles:

* generator: `openrouter/meta/muse-spark-1.3-contributor`, reasoning `low`;
* arbiter: `openrouter/openai/gpt-6-sol`, reasoning `high`.

Both roles use OpenRouter's native structured-output request with a fixed
`max_tokens=16384`. The command performs no experiment matrix and makes no
claim that every cue has been quality-verified. Accepted final SRT rows are
exported; failed or unresolved rows keep their original text and are queued
for review rather than auto-accepted.

## Production run

Activate the project environment, then run the primary command:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.srt \
  --output-dir output/short-file-run \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --budget-usd 10 --run-time-limit 6600
```

`--profile` and `--output-dir` are required. A finite positive
`--budget-usd` is required for a live primary run; `--max-requests` defaults to
1000 and is a physical request cap. Credentials are loaded from the existing
dotenv path only when the first model request is needed. Do not put credentials
in prompts, logs, or report metadata.

For **new** primary runs, generation/review happen in ordered windows of at
most `--review-batch-size` cues (default 12), with at most one queued
repair/recheck round inside each window. Accepted earlier windows are
checkpointed before later generation. The optional `--run-time-limit` (seconds)
and the first SIGINT/SIGTERM ask the runner to stop **before its next live
request**, after the current request reaches its 300-second client timeout or
completes and its receipt is saved. The stop produces an incomplete report,
exports, and nonzero exit. A second signal requests cancellation at the next
async suspension; synchronous Vosk preflight or atomic checkpoint work may
finish first. If a sent request has no durable completion, its billing remains
unknown and requires explicit recovery. A failed cancellation snapshot update
does not replace the cancellation, but the saved journal then needs inspection.
Do not wrap the command in an external timer that sends SIGINT while a request
is active.

The following exact command was run once for the approved Australian Survivor
episode. **Do not rerun it:** it was interrupted with an unknown journal start.
The source/profile came from the saved historical manifest:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file \
  '/home/vladimir-barbuta/Downloads/subs_found_for_testing/Australian Survivor S01E03_ru.srt' \
  --output-dir output/openrouter_episode_20260929_sol6_fresh \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --model-path models/vosk-model-tts-ru-0.10-multi --voice 8 --rate 1.5 \
  --budget-usd 10
```

This run has its own $10 cap; historical recovery receipts are not billed a
second time into its policy ledger. Its
[partial HTML report](../output/openrouter_episode_20260929_sol6_fresh/shortening.html)
and [SRT](../output/openrouter_episode_20260929_sol6_fresh/subtitles.srt)
are saved. The processing checkpoint is incomplete; the last request start is
unresolved, and no arbiter verdict was produced. Generation runs across every
selected cue before review begins, so 500 proposals were saved but none was
accepted. The HTML/report budget is the prior 120-attempt checkpoint; the
read-only journal includes a 121st unknown start and must be used for current
accounting. The user authorized bounded retries, but this frozen primary CLI
cannot retry an unknown start. Do not use `--resume` or retry that request
without reconciliation and separately authorized recovery.

The output directory contains `subtitles.srt`, `subtitles.json`, `report.json`,
`calls.jsonl`, `requests.jsonl`, `run_manifest.json`, resume checkpoints, the
cost summary, and the review queue. The report records both roles, actual
OpenRouter usage costs, unknown-cost reservations, and any stop reason.

For each new request, admission reserves the published maximum context and
16,384 output tokens (about $4.86 for a Sol request). Previous missing-cost
receipts reserve their *observed* input/output tokens at the frozen maximum
rates; a timeout without counters reserves the full published maximum. At most
`--max-requests` physical attempts (default 1,000) may have unknown USD on
new runs; the old frozen three-unknown policy is unchanged. Admission requires
known USD + unknown reservations + the next request reservation <= the frozen
budget, and no more than `--max-requests` physical attempts. An unknown amount
is never recorded as zero or
combined with reported USD as actual spend. The upper reservation assumes the
catalog's listed rates and no unlisted fees: it cannot guarantee the eventual
invoice, and work stops at the guard when uncertainty prevents further admission.

Primary calls have a 300-second client timeout and fixed 16,384-token output
cap (including reasoning). A client timeout does not guarantee the provider
stopped processing. One repair/recheck round may retry a failed request for
affected cues; failed neighbors retain their original text and review entry.
Authentication, invalid identity, foreign route and pricing errors stop the
run. Completed responses are never resent solely to obtain missing USD.

## Resume

Resume the same run with the same source, profile, budget, roles, and options:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.srt \
  --output-dir output/short-file-run \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --budget-usd 10 --run-time-limit 6600 --resume
```

The manifest freezes the source, configuration, runtime hashes, complete
pricing snapshot, budget cap, request cap, and role contract. Resume uses that
frozen snapshot and performs a fresh public pricing compatibility check; it does
not replace the frozen snapshot or pricing ID. A changed source, profile,
budget, role, pricing contract, module hash, or request reservation fails closed
before a paid request. Completed runs make zero additional paid requests.

The old journal is replayed locally and receipts are never charged twice.
Primary runs do not automatically retry a request whose journal has a start
without a completion: its billing is unknown and requires a separate explicit
decision. For **new windowed contracts only**, after inspecting the request ID
and monetary risk, an explicitly authorized recovery uses the same frozen
source/profile/options and adds `--resume --retry-unknown REQUEST_ID` to the
command above. It permits at most **one replacement per logical request**,
keeps the old unknown reservation, counts the replacement against the same
budget and request cap, and refuses inadmissible replacement before saving a
retry event. The older 2026-09-29 episode has a stage-major/three-unknown
contract and **cannot** be resumed by this code: reuse/migration of its 500
proposals requires separate engineering and authorization, not manifest edits.
Completed recoverable failures use the bounded cue recovery queue;
completed successes are replayed locally without another POST. Processing may
be `completed` even while cost accounting is incomplete: read both `run_state`
and `summary.accounting_complete`. Budget stops remain `incomplete`.

### Inspect saved status offline

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shortening_status --run-dir output/short-file-run
```

The command prints JSON to stdout and exits 0 only for a validated completed
snapshot without known blockers; blocked or invalid snapshots exit 2. It reads
the frozen manifest, baseline, journal, report, and frozen pricing snapshot
without credentials, network, TTS, inference, HTML generation, or writes to the
run directory. It distinguishes a request start without a completion from a
completed transport failure whose cost remains unknown. It reports request,
stage, model, and receipt identifiers, redacted reason codes, recomputed known
USD and unknown reservations, and differences from frozen runtime module hashes.
It never prints subtitle or prompt text or raw provider errors.

This is **not** resume authorization: it cannot clear an unknown cost, make a
retry decision, or verify the original input, profile, or current public pricing.
An active lock or an observed file change blocks inspection. Missing or corrupt
artifacts fail closed; keep them for investigation. A genuine run can have a
`running` report and no status sidecar after interruption, or a `blocked_unknown`
sidecar without `run_state`; outstanding requests remain visible and blocked.
A completed run with no model requests can legitimately have no request/call
journals. Editing code also changes
frozen module hashes and can block normal resume. Do not modify a saved
manifest to make its hashes match.

## Offline dry-run

Use a separate fresh directory. Dry-run has no API key lookup, public GET, POST,
server, Vosk inference, model call, or HTML artifact:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.json \
  --output-dir output/short-file-preflight \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --dry-run
```

It writes only `preflight.json`; it does not create a fake live report. There
is no `--execute` switch: omit `--dry-run` for the production command above.

## Calibration, timing, and limits

Input fields, timing, indexes, and unknown JSON keys are preserved. Planning
uses the frozen `normalized_tts_text` calibration profile. Local original-timing
preflight is evidence about the original cue only; it is not a final-audio or
SH-10 quality guarantee. There is no final-audio guarantee in this command.

The primary CLI intentionally hides legacy `--model`, `--variant`,
`--length-tools`, `--max-tokens`, and `--retry-unknown` options from normal
help. Explicit `--model` remains available for the historical OpenCode and
DeepSeek paths and keeps their old mapping and resume behavior. Those paths are
experimental compatibility interfaces, not the main instruction. Legacy paths
are not cost-guarded: `--budget-usd` and non-default `--max-requests` are
rejected with explicit `--model` instead of being silently ignored, so no CLI
budget promise applies to a legacy run. See
[`future_experiments.md`](future_experiments.md) for legacy research notes.

## Artifacts and review

Live output includes SRT and JSON exports, HTML/report artifacts supplied by
the existing publisher, actual cost records, and a review queue. Model
acceptance is not a claim of complete semantic or timing quality. Failed,
malformed, transport-error, semantic-impossible, and unresolved cues retain
their original text; no automatic acceptance or silent data loss is performed.
