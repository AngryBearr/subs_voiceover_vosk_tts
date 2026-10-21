# Explicit OpenRouter STT upload mode

**Authorized full-episode follow-up completed:** Australian Survivor e04 audio-only
MAI direct, 57 fixed <=60s chunks, 660 native cues. [Text/SRT/JSON and evidence](../temp/australian_survivor_e04_mai/README.md).
Episode actual **$0.09344444444444445**; shared historical+episode total
**$0.1011395548888889 / 72 reservations**, all costs known. The explicit programmatic
cap was 72; default constructor/CLI cap remains 18. No GPU setup or local model run.

**Historical bounded comparison:** two 30s episodes, a validated 7.855s three-source
mixture, a real S46 production hybrid run, and six profile-bound stem calls from a
separate **annotation-assisted SepFormer3 diagnostic**. MAI returned native timings;
Muse returned text/usage only. Both APIs recovered the full third-source phrase on
diagnostic stems, not on the raw mixture. This does not establish automatic triple
acceptance: default PixIT missed a voice and never triggered SepFormer on that fixture.
The fifth interrupted request's actual cost was reconciled from the user-provided
generation ID, without recovering or retrying its response. **Historical actual total
$0.007695110444444445 / 15 reservations**, all costs known, then within the shared $1/18 cap.
Details:
[stt_api_comparison.md](stt_api_comparison.md), with a
[durable ignored workspace snapshot](../temp/stt_api_results/README.md).
That bundle's ledger is read-only historical evidence, not an active API output budget.
The current user-provided workspace
`.env` was authorized/loaded safely; the historically denied external source was
not accessed and is not the current blocker.
Self-key aggregates were not used to assign the lost response a cost. The older
published-input fixture is archived separately because its first filename conflicts
with exact HF-row content. The new full-utterance gold fixture uses the validated
HF-row inputs. The diagnostic reused the same context and original clean profiles,
with unchanged matching thresholds; no production force flag or gating change was added.

`fast` and `accurate` remain local and never call this client. Only `--preset api`
uploads audio to OpenRouter/providers and may incur charges. No native separated
audio, acoustic gender, three-overlap recovery, ZDR/no retention or no-training
guarantee is assumed. No privacy filter is silently weakened.

## Usage (generic examples; explicit upload/charges, not instructions to repeat tests)

Activate `vosk_env`, then run from the repository root:

```bash
source vosk_env/bin/activate
uv run --active -m utils.stt_pipeline INPUT -o /tmp/opencode/stt_api_compare/mai_direct \
  --preset api --api-model microsoft/mai-transcribe-2 --api-mode direct \
  --start 60 --duration 30 --api-budget 1
uv run --active -m utils.stt_pipeline INPUT -o /tmp/opencode/stt_api_compare/muse_hybrid \
  --preset api --api-model meta/muse-voice-transcribe-1.0 --api-mode hybrid \
  --start 60 --duration 30 --api-budget 1
```

`--duration` is required and capped at 180 seconds per request; comparison excerpts
remain <=30s. Full episodes require explicit full-audio authorization and bounded
chunking, not a single whole-video or oversized-audio upload. Existing collisions
are rejected before calls. Missing keys fail before decoding or model loading.

Set `OPENROUTER_API_KEY` locally in the environment or workspace `.env`, never in
chat/logs/command arguments. Environment values take precedence, even if invalid or
empty. Only that local dotenv assignment is parsed as literal data; no shell
evaluation/substitution or duplicate assignments. No other provider key is used.
Ignore rules cover `.env`/`.env.*` except a non-secret `.env.example`; actual secrets
should have mode 0600. The blocked external copy is **not performed by this code**.

## Contracts and truthful output

Stdlib HTTP sends JSON to `https://openrouter.ai/api/v1/audio/transcriptions` with
`input_audio={data: BASE64_COMPLETE_WAV, format: wav}` (no data URI/chat completions).
Audio is clipped/rounded deterministically to mono PCM16/16kHz. WAV and encoded JSON
each have a 25MiB guard. HTTP timeout is 70s; the upstream STT **processing deadline
is 60s**, not an audio-duration limit, and cannot be raised by the client timeout.
The authorized episode used <=60s audio chunks, without assuming linear latency.
There is **no automatic paid POST retry**.
The documented STT endpoint returns synchronous final JSON, not an async job or
server processing percentage. `X-Generation-Id` is billing/debug tracking; OpenRouter
batches do not support audio transcriptions. Whole-file compression can reduce upload
bytes but not the upstream deadline; the service guide supports MP3 even though this
client intentionally sends WAV. Existing episode chunk logs show completed-input
progress, not provider-internal percent. No progress UI/new upload was added.
See the [research report](../reports/Диаризация%20и%20разделение%20голосов.md) and
[primary STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt).
No input base64/authentication headers are persisted; secret echoes/auth/audio fields
are redacted from saved replies. Sensitive field names are normalized for case and
separators (including secret/access-token/API-key variants); exact key-string echoes
are redacted case-insensitively, without speculative decoding.

- MAI requests `verbose_json`, segment/word granularities and
  `provider.options.azure.diarization.enabled=true`: documented, accepted on both
  bounded live episodes; returned seconds-based segments/words with integer speakers.
- Muse requests candidate `provider.options.meta={mode: DIARIZATION,
  languageBias: [English]}` only, not word timestamps. Actual passthrough/normalization
  is **unverified**: both live episode replies contained only `text`/`usage`, no
  timestamps/labels. The parser also supports synthetic normalized-segment fixtures
  and native `turns[{startMs,endMs,transcript,speaker}]`; native Meta turns have not
  been observed through these OpenRouter calls.

### Bounded four-model experiment (partial live results; billing-blocked)

The shared `Client` additionally supports these exact experimental IDs for **<=30s
clips only**. They are intentionally not added to the ordinary local pipeline's CLI
choices or the MAI-only whole-episode launcher. All four send `verbose_json` and
`timestamp_granularities=[segment, word]`, plus:

| Model | Requested diarization option |
| --- | --- |
| `elevenlabs/scribe-v2` | top-level `diarize: true` |
| `deepgram/nova-3` | `provider.options.deepgram.diarize: true` |
| `x-ai/grok-stt-1.0` | top-level `diarize: true` |
| `google/gemini-3.5-transcribe` | top-level `diarize: true` |

October 13 research confirmed catalog/provider availability and the generic STT
schema; Deepgram's option is an explicitly allowed passthrough. At phase2 the other
three integrations were **live-unverified**; phase3 observations below narrow this.
Both requested collections/labels
are not guaranteed; timestamps may affect recognition. A 400 is an experiment result,
not permission for a format fallback or another paid POST. Native xAI 2.0/Google/ElevenLabs
knobs are not imported into these OpenRouter contracts. References:
[STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt) and
[create transcription](https://openrouter.ai/docs/api/api-reference/stt/create-transcription).

No verified hourly tariffs were invented for these four. Their ledger entries use a
**fixed $0.05 conservative reservation** and
`preflight_cost_kind=fixed_experiment_reservation_not_provider_rate`; MAI/Muse rates
and existing 10x guard/default cap18 remain unchanged. Experimental requests require
a shared budget <=$1. Gemini's audio-input/output token pricing is not a fixed hourly
tariff. Actual usage cost (or existing bounded generation lookup) is authoritative;
unknown billing stops all subsequent POSTs. The reserve is not a server billing cap.

The ignored [experiment plan](../temp/stt_new_models_compare/plan.json) has exactly
eight sequential slots on two unchanged blind-mixture WAVs, with explicit total cap80
and genuine 72-entry history. Default command is **read-only, no key/network**:

```bash
source vosk_env/bin/activate
uv run --active -m temp.stt_new_models_run
```

`--allow-upload` was used only after phase3 approval; unknown billing now blocks any
further uploads. Raw replies/costs are saved before parsing; known-cost text-only,
bad-timing or rejected outcomes remain evidence without fabricated captions/retries.
Unresolved attempts, lost replies, corrupt bindings or unknown costs block the run.
The missing active ledger was recovered byte-for-byte from the complete episode's
backup, matching the earlier active SHA256 and all receipts—not reset or replaced by
the old 15-entry snapshot. [Recovery proof](../temp/stt_new_models_compare/ledger_recovery.json).

**October 13 authorized execution:** Scribe/Nova each returned two known-cost replies
with segments and words. Scribe's segments omit labels while words contain three
labels on each fixture; primary segment exports and companion word diagnostics remain
distinct. Nova returned one gold/two TV labels. Neither recovered all three gold
phrases. Grok gold then rejected top-level `diarize` (HTTP400), without cost/generation
ID; billing stays unknown, so Grok TV and both Gemini slots were not sent. Five POST
attempts, no retries/billing GETs. [Actual comparison](stt_new_models_comparison.md).
These observations supersede the earlier candidate generic-contract assumption;
Grok plain ASR without that flag was not tested. Shared actual total is now unknown:
known subtotal $0.10500917711111217 / 77 reservations. Atomic ledger replacements
retain 0600 permissions and refuse read-only snapshots.

Export selection is deterministic: the first nonempty advertised collection in
**turns → segments → words** order wins; duplicate representations are never combined.
Only that collection is validated, so malformed unused words cannot invalidate good
segments. Whitespace-only item text is skipped before timing/label validation;
non-string text and malformed selected nonempty items fail, without fallback repair.
The saved request's `timing_normalization` records selection, ignored collections,
blank-item counts and any clipping. Integer-valued finite nonnegative float speaker
labels canonicalize to integers; other floats/bools fail. String labels are stripped
consistently; their original typed values remain in the sanitized raw reply.
Documented string `speaker_label` is used when numeric/native `speaker` is absent
or null; numeric speaker zero remains valid and takes precedence.
Words may use native `text` when `word` is absent and `speaker_id` when both existing
label fields are absent. Explicit word types `spacing`/`audio_event` are skipped
before time/label validation with a diagnostic count; unknown types fail rather than
discarding possible speech. Raw root text/event tags remain unchanged in evidence.

Direct never loads PixIT. Native labels are scoped to a request and assigned public
IDs chronologically, not invented global identities across requests. Missing labels
stay `SPEAKER_UNKNOWN`, direct gender stays unknown, native overlapping segments and
identical phrases stay separate. Direct manual overrides require externally reviewed
IDs and are rejected here. Hybrid retains existing local PixIT/SepFormer numeric
source binding and clean-original-speech gender/overrides, and transcribes each local
source with API chunks. Cloud labels cannot replace local source identity. This is
**local separation + remote ASR**, not native API separation.
Hybrid sources with arbitrary PixIT peaks above 1 are attenuated by exactly `1/peak`
before PCM conversion; source peak/gain are recorded, rather than hard-clipping the
waveform. This transport scaling is not a separation repair or quality claim. Use
the same resulting PCM samples for any matched local-ASR comparison.

`PREFIX.json/.srt/.speakers.json` use real native timestamps only: integer ms,
`text: list[str]`, crop offset once. Segment-only bounds remain native; words-only
replies conservatively keep individual native word bounds, not evenly-spaced guessed
alignment. A separate **API policy** permits only an end endpoint overshoot of at
most 250ms, measured from the returned end minus actual clip duration; it is not the
local 80ms ASR-grid invariant or a live accuracy claim. Only an intersecting cue's
end is clamped; negative/nonfinite/reversed/gross or wholly outside cues fail.
Clipping is marked by `timestamp_clipped` and diagnostics retain original bounds;
the sanitized raw response is not modified by timing normalization. Overlap is
marked only across different speaker IDs, consistently with local/hybrid exports.
All API cues require review. Plain text produces `.transcript.txt` and
`.api.json` plus an actionable JSON/SRT-unavailable error; if any hybrid chunk lacks
timing, the full run has no fabricated/partial subtitle export. `.api.json` holds
sanitized replies, request scopes, usage/generation/cost evidence, POST-only and
full-pipeline wall times, including failure evidence after decoding.
Returned root text/evidence are captured **before** timing parsing. Invalid selected
timings therefore retain `.transcript.txt`, sanitized raw reply, known billing and
`timing_error`, but no JSON/SRT. Hybrid keeps labeled completed-chunk text without
pretending a partial SRT is a full export. When root text is absent, successful
parsing derives the canonical transcript only from actual selected item text; no
duplicate transcript lines or extra POST retry is added.

## Client budget, not a server billing cap

Default guard is $0.10; task limit is **$1 shared across all calls**. The output
directory's `.stt_api_budget.json` is shared across prefixes/models/chunks; do not reset
or delete it to evade the limit. Default at most 18 POSTs; reserve a conservative 10× duration
estimate (minimum $0.001) before each. Supplied October 2026 rates are **MAI $0.10/hour**
(promotion through December 31, 2026), **Muse $0.18/hour**. MAI catalog
`pricing.prompt="0.1"` is not treated as USD/token. Verify pricing again before live
use; this is not a provider/server hard billing cap and cannot undo unexpected charges.

Canonical numeric `usage.cost` (documented whole-request USD) wins; `costUSD` is only
a compatibility fallback. Otherwise use header `X-Generation-Id` read-only generation
lookup `data.total_cost` (at most two GET attempts, never a POST retry). Unknown costs
are **null, not $0**, and stop further uploads, including potentially billed HTTP
errors/timeouts. Atomic ledger updates and Linux `fcntl` reservation locking prevent
stale callers bypassing pending billing; unsupported platforms fail before POST.
A leftover pending file needs review, not overwrite. POST-only timing excludes cost
lookup; pipeline timing includes decode/local processing/lookups.
Sanitized structured HTTP error evidence is saved when available. If an error has
a generation header and lookup resolves actual cost, that cost is retained once;
otherwise billing remains unknown. A process/outer-command kill can leave a pending
reservation without its response or generation ID: reconcile, never reset/retry.

### Explicit programmatic request cap for the authorized episode

`BudgetLedger(budget, path=None, *, max_requests=18)` accepts only a positive,
non-boolean integer. `Client` uses that supplied ledger; the public CLI was not
changed and has no new cap flag. The dedicated ignored episode helper explicitly
used the existing active ledger, not a new output-directory budget:

```python
ledger = BudgetLedger(1, Path("/tmp/opencode/stt_api_compare/.stt_api_budget.json"),
                      max_requests=72)  # 15 historical + 57 planned; no retry slots
```

Historical entries, the $1/unknown-cost guards, atomic writes and locking remain.
All 57 costs came from response usage; no new billing GET/retry occurred. Sequential
complete PCM chunks were persisted before upload and native replies before parsing.
Cached replies are matched to model/interval/WAV hash/reservation and reparsed only;
pending/unknown/error stops instead of resubmitting. Final IDs are request-scoped
`CHUNK_0001_SPEAKER_00`, gender unknown; minute boundaries and native cues need review.
This establishes complete audio-input coverage/export, not every-word accuracy.

Results: [English text](../temp/australian_survivor_e04_mai/episode.txt),
[SRT](../temp/australian_survivor_e04_mai/episode.srt),
[JSON](../temp/australian_survivor_e04_mai/episode.json),
[machine checks](../temp/australian_survivor_e04_mai/acceptance_checks.json).
The old `temp/stt_api_results/` 15-reservation snapshot remains unchanged. The new
episode's `ledger_snapshot.json` is read-only historical evidence, not an active ledger.
The earlier 90s preflight was planning only; the actual plan is `run_plan.json` (60s).

## Whole-input MAI launcher (plan first; completed AU is never uploaded again)

`utils.stt_episode` handles **MAI direct only**. It is one client-side whole-input task,
not one upstream POST or an async provider job. The default prints a read-only plan:
first audio stream, source sample rate/channels, <=60s chunk count, approximate rounded-
second price forecast and required total request cap. It never loads credentials,
decodes full media or uploads without `--allow-upload`. Existing plans use their
recorded stream/source and decoded sample count instead of probing or re-decoding.
New media with a reported nonzero audio stream start is rejected for reviewed timeline
handling rather than silently placing that speech at timestamp zero.

The current episode has already completed; this command verifies genuine caches,
ledger bindings and text/JSON/SRT exports, then returns **complete 57/57 / 660 cues /
new_POSTs 0** without changing any existing output or budget:

```bash
source vosk_env/bin/activate
SOURCE='/home/vladimir-barbuta/Downloads/Australian_Survivor/SurvAU (2016) e04 [9193582_456239200].mp4'
uv run --active -m utils.stt_episode "$SOURCE" \
  --output-dir temp/australian_survivor_e04_mai \
  --budget-ledger /tmp/opencode/stt_api_compare/.stt_api_budget.json
```

Adding `--allow-upload` to this **same completed invocation** also verifies and returns
the cached result without POST/key/decode; it is unnecessary here. Do not point the
already completed AU source at a new output folder to repeat its paid transcription.

For a **new, separately authorized** input/output, inspect the plan first. Actual
preparation/resume requires the existing shared ledger and explicit `--allow-upload`:

```bash
# NEW_INPUT and NEW_OUTPUT must refer to a separately authorized job, not a duplicate AU run.
uv run --active -m utils.stt_episode "$NEW_INPUT" --output-dir "$NEW_OUTPUT" \
  --budget-ledger /tmp/opencode/stt_api_compare/.stt_api_budget.json
# Only after reviewing that plan and authorizing uploads/charges:
uv run --active -m utils.stt_episode "$NEW_INPUT" --output-dir "$NEW_OUTPUT" \
  --budget-ledger /tmp/opencode/stt_api_compare/.stt_api_budget.json \
  --max-requests 129 --allow-upload
```

**129 is an example**, 72 current historical reservations + another 57 planned calls;
use the reviewed job's actual required total cap. It is never raised automatically.
Launcher default is 72; the existing API constructor/CLI still default to 18. Budget
defaults to $1, cannot exceed $1 (`--budget` can lower it), and actual/unknown-cost/
conservative per-request guards remain. Pricing is a forecast, not a server billing
cap. No output-directory ledger/reset or historical snapshot is accepted for uploads.
Run one caller against the shared ledger at a time; this launcher is sequential and
locks its output, while existing ledger reservations enforce the shared cap.

New preparation decodes once to mono PCM16/16kHz WAV and reads bounded sample-aligned
chunks; source stats/hash, per-chunk WAV hashes, scopes and native replies are durable.
Resume matches raw cached replies with the actual ledger entry before parsing; a
genuine returned ledger response can recover a missing cache **without POST/key**.
An attempt without returned evidence, failed/lost reply, unknown cost, corrupt cache,
source change, output collision or missing native timings stops, never retries a paid
POST. Interruptions during preparation/pending atomic writes require manual review,
not implicit re-decode. Complete legacy evidence uses its saved source path/size/mtime
binding (the old plan has no original-media SHA256); new plans also bind source SHA256.

Progress reports completed unique input seconds/total and completed chunks/total,
with shared actual dollars. “Waiting” is not provider percent-complete. Exports are
`episode.txt/.json/.srt/.speakers.json`, with native offsets applied once and IDs
`CHUNK_0001_SPEAKER_00` scoped to that input minute. Gender remains unknown; native
overlap and repeated phrases remain. Empty native chunks are recorded, not certified
silent. Failed runs preserve native completed-prefix `episode.partial.*` and raw text;
no untimed text is given invented captions. Boundaries/words require human review.

### Audio formats are pipeline choices, not a universal STT requirement

The AU original stream is **AAC, stereo, 44100Hz**. This launcher downmixes/resamples
it to **mono PCM16 WAV at 16000Hz** for the current MAI client. That removes stereo
channel distinction/high-frequency content; it is not MP3 compression, denoising,
music removal or human-speaker separation. Other API providers/formats can have
different requirements; OpenRouter's guide includes MP3, not just WAV.

The existing local pipeline decodes to **mono float32 / 16000Hz**. Its pinned
three-source SepFormer refiner resamples **16kHz → 8kHz → 16kHz**; upsampling cannot
restore discarded bandwidth. No original media is modified, and no local full-AU
ASR/diarization run or GPU setup is performed by the new cloud launcher. It adds no
other-model integration or GUI; those would need separate verified contracts.

## Offline verification

```bash
source vosk_env/bin/activate
uv run --active -m unittest discover -s tests -p test_stt_openrouter.py -v
```

HTTP is mocked and dotenv fixtures are synthetic temporary files. Live comparison:
[stt_api_comparison.md](stt_api_comparison.md). References supplied for this task:
https://openrouter.ai/docs/guides/overview/multimodal/stt.md and
https://openrouter.ai/docs/api/api-reference/stt/create-transcription.md
