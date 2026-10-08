# `shorten_file`

## Resumable continuous batch (native text, measured Vosk audio)

### Latest independent validation — final stop, 5 October 2026

[Real final report](../output/pipeline_batch_validation_20261005/report.html) ·
[counts/checks](../output/pipeline_batch_validation_20261005/final_completion.json) ·
[exception texts/timings](../output/pipeline_batch_validation_20261005/final_exceptions.tsv).
The predetermined disjoint positions 500–530 contain 31 cues / 65.449 s, with
unchanged full-source windows, voice 8, base 1.25 and total cap 1.5. Yield was
**8/31 (25.8%) → 16/31 (51.6%)**: seven direct and nine accelerated final fits;
ten original texts and 21 rewrites. Actual TTS: **31 + 19 = 50**, with all eight
first-round fits reused byte-exact. Fifteen exceptions block export: eleven
accepted repairs require tempo above 1.2; four null/refusal fallbacks retain
earlier timing failures. No normalization or bounded-render residual failures.

Measured round-2 advice: seven met (four fit, three fail), twelve exceeded
(four fit, eight fail), four null (not budget passes). There were 32 proposal
events, including 28 nonnull and four null, across 31 source cues: 27 accepts,
one revise and four refusals. Four native Sol 6.1 medium invocations accepted
both exact full assemblies; this is model-only, not human/audio approval.
Audio wall times **123.547159 + 92.327364 = 215.874523 s** exclude unknown native
time/cost; Python paid API calls were zero. The completed audio rerun added zero
inference/TTS and preserved state and all 150 WAV hashes, including 93 round-1
WAVs. Prior 240 artifacts and 323 historical hashes are unchanged. Offline
Chrome verified the real report; unchanged production tests were not rerun.

The earlier 105.821 s scene reached 13/31 → 20/31. Different dialogue density,
text and duration make this a descriptive comparison, not causal evidence for
or against the advisory clarification. Practical full-batch readiness remains
unproven; native orchestration is still needed. **Final stop:** no third repair,
manual edits, mixing or video work. Defer predictor changes to a separate
systemic timing/calibration decision; these results promise no general gain.

### Latest offline checkpoint — advisory contract clarified, 5 October 2026

[Saved-measurement diagnosis](../output/pipeline_batch_calibration_20261005/analysis.json) ·
[implementation replay/checks](../output/pipeline_batch_calibration_20261005/implementation/checks.json).
The ten timing failures comprise seven above measured advice, two within advice
but still too long, and one 5.766 ms tempo-render overrun; normalization is a
separate guard failure. Only 16 true timing pairs exist. Vowel ratios did not
justify a robust predictor replacement; the **budget algorithm/formula is unchanged**.

Requests now expose `current_normalized_chars` for the exact selected text;
counts include canonical spaces/punctuation before RUNorm/RUAccent stress.
Measured repair advice takes precedence over historical nominal `max_chars/status`,
with an explicit required reduction only when timing exists. Exact-candidate
review inputs carry nonblocking `advisory_check` feedback, not a semantic or audio
verdict. Over-budget proposals/refusals remain allowed; under budget guarantees nothing.
Offline replay preserves all 17 existing measured budgets across 18 targets:
nine warnings, eight not exceeded, one unavailable. **94 focused batch/native
tests passed** with mocked expensive work; no new model/normalization/TTS/render
calls or live pilot occurred. Prior analysis and every pilot file/WAV are intact.
The old runtime now correctly fails `load_state`; do not rewrite its binding or
regenerate its report. The observed result remains **20/31**, with no demonstrated
yield improvement. Native handoffs are clearer, not fully unattended. Any later
yield claim needs separate authorization for an independent contiguous sample,
fixed voice/base/cap/windows/pause policy, and no manual text edits. Not run here.

### Latest live checkpoint — final round exhausted, 5 October 2026

[Real standalone report](../output/pipeline_batch_pilot_20261005/report.html) ·
[final status](../output/pipeline_batch_pilot_20261005/status_final.json) ·
[exceptions with chosen text/original timing](../output/pipeline_batch_pilot_20261005/final_exceptions.tsv).
The unchanged original positions 1250–1280 contain 31 cues / 105.821 seconds:
**13/31 (41.9%) fit after round 1; 20/31 (64.5%) after the single final repair**.
Eight final texts retain the source; 23 are native rewrites. There were 34
proposal events (16 + 18), not 34 source cues, with exact-hash Sol 6.1 **medium**
candidate and assembled-text accepts in four native writer/reviewer invocations.
Model acceptance is not human listening or a timing guarantee.

Actual TTS calls were **30 + 17 = 47**; two normalization failures stopped before
TTS, with one still unresolved (`cue-1260`, RUNorm `крут` → `крутой`). Ten final
timing failures remain: nine exceed the remaining 1.2 tempo cap; `cue-1266`
still exceeds its 2.237 s window by 5.766 ms after reaching total speed 1.5.
All 13 first-round fits remain byte-exact. The completed audio rerun added zero
normalization/TTS calls and left state and all 141 saved WAV hashes unchanged.
Audio wall times were **131.426 s + 89.842 s = 221.268 s**; native elapsed/cost
and physical provider request counts are unknown. Python made no paid API calls.

The exact final assembly is model-accepted, but **mix/export remains blocked**
by eleven exceptions. No third repair, manual cue editing, mixing, retiming,
video extraction, package change or episode-wide run occurred. The reusable
controller and safe resume work; this yield does **not** pass practical fully
automatic/mix-ready or ten-episode readiness. Native orchestration is still
required between Python commands. Source/plan/config, 323 historical hashes and
the 14-file earlier preparation are preserved. Local Chrome loaded the final
text report offline; media metadata was inspected, not listened to. Production
code was unchanged, so its passing 168 tests were not rerun. Stop here.

Current frozen preparation (5 October 2026):
[`pipeline_batch_pilot_20261005`](../output/pipeline_batch_pilot_20261005/README.md).
The 4 October preparation is retained unchanged with zero inference calls; it
is not another live experiment and its old runtime binding is intentionally stale.

`utils.pipeline_batch` freezes a **full original** SRT/JSON and an inclusive,
zero-based continuous position range. Global `cue-N` IDs, source indexes and
times stay unchanged. Windows come from the full source, including the immediate
next cue outside the range; a sparse target list never expands a window.

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.pipeline_batch prepare FULL_ORIGINAL.json \
  --start 1250 --end 1280 --config pipeline_config_vosk.json \
  --output-dir output/pipeline_batch_pilot_20261005
uv run --active --no-sync -m utils.pipeline_batch status \
  --output-dir output/pipeline_batch_pilot_20261005
```

Preparation/status make **no model, normalization or TTS calls**. The native
assistant orchestrates writer/reviewer calls between commands; Python does not
call Sol, OpenRouter or OmniRoute. This is not an unattended one-command LLM run.

1. Give the native writer `request_round1.json`: only nominal nonfits are targets;
   full source/plan paths, each original ±3 context and the entire assembled run
   remain available. Save actual proposals as `writer_round1.json`.
2. Run `review-input writer_round1.json --output-dir RUN`. Give the independent
   reviewer the resulting `reviewer_input_round1.json`. It contains exact
   candidate and proposed assembly hashes, not Python semantic approvals.
   If any per-cue verdict rejects a proposal, run the same command with
   `--per-cue-reviewed per_cue_round1.json` to assemble only accepted proposals
   with original fallbacks. Review that **exact final assembly** in the same
   independent reviewer invocation; the hash command makes no model call.
   Per-cue and assembled verdicts may share `invocation_id`, provided it differs
   from the writer's. The controller does not require a third model call.
3. Run `ingest-reviewed reviewed_round1.json --output-dir RUN`, then
   `audio --output-dir RUN`. Audio reuses process-local models, normalizes each
   cue with existing lexical guards, and makes at most one engine attempt per
   cue in the round. Voice 8, base 1.25 and total cap 1.5 come from the frozen
   config. Raw, silence-reduced/resampled pretempo and final WAVs are retained;
   engine inputs, frame counts, hashes and tempo factors are recorded.
4. Audio automatically writes **one** `request_round2.json`, selecting only
   measured timing nonfits and normalization failures, including nominal fits
   that proved too long. Its measured character target has 10% headroom and is
   advisory, not permission to lose meaning. Repeat steps 1–3 for round 2 only.
   Successfully fitted cues never return to TTS. Canonically identical repairs
   are `repair_unchanged` and never resynthesize the same text. Timing repairs
   must be strictly shorter than the measured current text (`repair_not_shorter`
   otherwise); normalization repairs need a changed proposal but may be longer
   than the current wording if still eligible against the original.
   Rejected repair proposals keep the prior text/WAV evidence; no third automatic
   loop exists.
5. `fits.json` keeps actual fits and original timing; `exceptions.json` lists
   remaining failures. Only `ready_for_separate_mix=true` permits a separate
   mixing/export step: the latest assembled review must accept the exact current
   assembly hash, and measured WAV end times must not overlap the next original
   cue start. An assembled rejection cannot be bypassed by fitted original
   fallback audio; `assembly_rejected_or_unreviewed` remains a separate blocker
   even with 100% audio fit. The explicit no-rewrite exception is a batch with
   zero initial writer targets and unchanged trusted originals, labelled
   `retained_originals_no_model_review_needed`; it needs no new grammar review
   and never overrides an actual rejected review.
   This controller never mixes missing cues or inserts replacement silence.
   Judge the automatic fit fraction, not a manually edited
   claim of 100% success. Scene alignment/video extraction is separate.

All commands use `uv run --active --no-sync -m utils.pipeline_batch` from the
repository root. Reruns do not repeat completed attempts or compound tempo.
The CLI rejects a different working directory before reading runtime/cache state.
Interrupted `started` attempts become `needs_reconciliation` in status and are
**never automatically retried**; there is deliberately no retry/reset switch.
Changed source, frozen config/plan, runtime/model files, request, candidate,
review or saved WAV hashes fail closed. One local writer lock and atomic state
replacement prevent overlapping mutations. Runtime hashes bind actual resolved
weights, tokenizers, model configs and code, excluding mutable Hugging Face
coordination metadata (`.locks`, `.no_exist`, `refs`, download metadata and
incomplete files). Benign cache bookkeeping does not invalidate resume; changed
real model content still does. Keep the frozen runtime unchanged
through the pilot; investigate unknown attempts before choosing a new batch.

### Native JSON contract

Writer response: `schema="pipeline-batch-proposals-v1"`, `execution="native_live"`;
copy `round`, `request_sha256`, `binding_sha256`, `source_sha256`, `plan_sha256`
and `policy` exactly from the request. Return every target once in request order:

```json
{
  "cue_id": "cue-1250",
  "original": {"copy": "the complete original object from the request"},
  "candidate": "One proposed text, or null when no safe proposal exists",
  "writer": {"model": "provider/model", "mode": "native_writer", "invocation_id": "actual-writer-invocation"}
}
```

Reviewed response: same binding fields and entries, but
`schema="pipeline-batch-reviewed-v1"`. Each entry adds `review` with
`verdict` (`accept`, `revise`, `refusal`), `text_sha256` (null for null candidate),
`request_sha256`, `model`, `mode`, and independent `invocation_id`.
The top-level `assembly_review` requires `verdict`, `assembly_sha256`,
`request_sha256`, `model`, `mode`, and independent `invocation_id`. Assembly is
the ordered list of `{cue_id, text}` for **all** run cues, including originals
where a proposal was rejected/ineligible. `review-input` computes its exact
SHA-256 using `json_sha256`; candidate hashes use UTF-8 `text_sha256`.
The reviewer checks meaning/naturalness in this assembled context, not just
isolated rewritten lines. Independent invocations can use the same model;
candidate and assembled reviewers may be the **same** invocation, independent
only from the writer.

Accepted mechanically eligible texts may exceed nominal targets. A candidate
that is not shorter than the original or introduces a new numeric representation
is not applied, even with model acceptance. A rejected assembled review blocks
changed texts, not processing the remaining originals. No Python semantic rewrite
or special per-pilot word whitelist is used. A writer batch is immutable after
binding; importing the identical final response again is a no-op.

Real native proposal/result imports publish a standalone escaped `report.html`,
including partial runs, proposals, models/modes, arbiter, exact chosen texts,
stats and exceptions. Preparation/Vosk-only runs publish no HTML.
The report opens with a Russian fit/attempt summary and a chosen-text, status,
duration/window table; raw proposals, model identities, reviews and audio
evidence remain available in collapsed details.
Cost and physical model request counts remain unknown without receipts; native phases
are logical writer/reviewer phases, not assumed billable requests. Optional
`elapsed_sec` in the final response must be measured, not estimated.

## Practical Vosk pipeline: native text handoff (4 October 2026)

### Latest scene checkpoint: final attempt exhausted, no dubbed preview

[Current standalone report](../output/pipeline_scene_20261004/view9/report.html) contains
13 actual proposal/review events (six initial, seven final), all model-only accepts,
and the nine current audio receipts. Final synthesis ran each of seven approved
texts once; cue-1262 and cue-1265 were carried byte-exact without processing.
[Measured results](../output/pipeline_scene_20261004/view9/repair/audio/final_results.json):
6/9 fit; cue-1263, cue-1264 and cue-1267 exceed the remaining 1.2 tempo cap at base
1.25. The single final batch is exhausted: no further repair/resynthesis, assembled
voiceover or dubbed preview. A separate [original 33-second reference](../output/pipeline_scene_20261004/view9/repair/audio/reference.mp4)
has unmuted original audio, 825 decoded frames at 25 fps, and reset PTS. Its re-encoding
and 20 ms video-grid crop difference are explicit; the -7900 ms inspection anchor
is approximate (visual residuals up to ~0.4 s), not verified speech synchronization.
The earlier shared ellipsis fix preserves punctuation outside learned abbreviation
expansion and leaves the name/word guards active; 105 prior focused tests were not
rerun. New helper compilation, exact text/audio/source hashes and full reference
decode passed. Raw reviewer JSON is unchanged; final medium/session identify the
parent-confirmed native request, not worker self-attestation or known billing.
Next is user inspection and a decision on remaining blockers. The historical
checkpoints below are preserved, not current scene/full-episode readiness claims.

### Current checkpoint: final selected targets fit, not scene/full-file readiness

The latest bounded continuation is saved in
[the standalone report](../output/pipeline_finish3_20261004/report.html) and
[final measured results](../output/pipeline_finish3_20261004/repair/final_results.json).
All **10/10 selected primary targets** fit their unchanged windows at base 1.25
and total cap 1.5. Eight successful earlier target WAVs were preserved; the last
two exact accepted candidates were each synthesized once as their original pair.
Five proposals/five Sol 6.1 medium accepts span the two phases; this is model-only
review, not user approval of the new audio or an episode success-rate guarantee.
Neighbors are separate: prior **1/3 fit**, final repair **1/2 fit**; the unchanged
cue-1266 neighbor still needs more than the allowed remaining tempo. Therefore
scene/episode end-to-end readiness is **not passed**. Source video and compatible
scene timing are missing, and no video/full-episode output or further repair runs
were made. Ten successful new WAVs exist across these phases; the interrupted
earlier attempt count and actual billing remain unknown. The 120 passing tests
were unchanged/not rerun; saved final checks verify exact audio/source hashes.

**Narrow name protection:** `_load_runorm` returns `ProtectedNameRUNorm`, a local
subclass that protects only the whole source word `Кэт` (case-insensitive) from
RUNorm's abbreviation expansion. It clones fresh tagger records, verifies their
source offsets/words, and treats only tags wholly inside that word and adjacent
punctuation as plain text. Foreign, invalid or crossing spans fail closed. Other
normalization stages, numbers/time/abbreviation handling and the lexical guard
remain active. This is not a general pronunciation dictionary or automatic name
protection: `Катя`, `Кати`, `Кэтрин` and other names are not whitelisted. No package
files or tagger methods are patched, and no generated `кати` is blindly replaced.
Inherited capitalization may lowercase `Кэт`; preserving its letters/referent
and RUAccent stress is verified, but exact original-case restoration is not claimed.

### Earlier checkpoint: completed small live pilot, not full-file readiness

The authorized ten-cue pilot is complete: [standalone report with ten embedded
current WAVs](../output/pipeline_live10_20261004/report.html),
[final presentation data](../output/pipeline_live10_20261004/final_results.json).
Strict measured timing progressed **1/10 after phase one → 2/10 after the
single measured-overrun text round → 7/10 after five technical WAV corrections**.
The remaining `cue-6`, `cue-809`, and `cue-1265` are unresolved; no further
automatic repair, speed increase, episode export or benchmark is authorized.
These selected hard cases are not a representative episode success rate.

RUNorm big / RUAccent turbo3.1 resources were downloaded with user approval and
actually loaded; package versions stayed unchanged. The shared Vosk punctuation
boundary now handles the original closing-quote failure and records exact TTS
input. Five 6–15 ms ffmpeg overruns were corrected on preserved phase-two WAVs,
with one real additional render per clip and no new TTS, trimming, window
tolerance or rate-cap relaxation. The new bounded second-render branch was
covered by mocked checks; these five real renders fitted on their first attempt.
Saved technical checks report 81 passing focused tests; this presentation update
does not rerun them or claim a full normalization/TTS rerun after the final guards.

The normalization guard detects word changes in already-written Russian RUNorm
input without Latin letters/digits, and requires RUAccent to preserve word/numeric
tokens for all input. It **detects, not repairs**, `Кэт` → `кати`; the old bad
`cue-1265` clip remains inspection-only. This is not a general semantic or
pronunciation guarantee. The pipeline is implemented and partly tested live,
not fully ready for unknown cases. Next: listen to the seven timing-fitting clips
with their complete original utterance/neighbors before considering an episode.
If needed after listening, protect Kat from normalization or specify an explicit
pronunciation-dictionary spelling, and define handling of the three hard cases,
not a higher 1.5 cap. Direct
translation remains deferred.

All 13 proposals and exact-text model reviews are retained. The two reused
Sol 6.1 medium sessions supplied 15 new native assistant responses (two new user
turns per role), excluding 54 earlier responses. Conditional API-equivalent
writer/reviewer cost is **$0.2367976–$0.2498576**, not an invoice or full-system
price; cache/output assumptions and actual billing are unverified, and local
normalization/TTS compute, downloads, preparation and orchestration are excluded.
No clip has human/scene audio approval; no listening claim is made.

This is a **separate working `pipeline` path**, not another benchmark or an
automatic OpenRouter route. Existing `shorten_file` model defaults and historical
experiments remain unchanged. Python never launches Sol: text generation and a
separate exact-candidate review belong to the current native assistant workflow.
Native reviewed-text import is CLI-only; the UI shares the Vosk speed policy,
but does not import reviewed text or launch a model.

From the repository root:

```bash
source subs_env/bin/activate
# Offline: full original context, stable position IDs, advisory targets; stop here.
uv run --active --no-sync -m pipeline input.srt --config pipeline_config_vosk.json \
  --prepare-text --output-dir output/my_text --base-speech-rate 1.25 --max-total-speed 1.5
# Later, after native writing and separate review, explicitly select the artifact.
# This RUNS normalization/TTS: explicit audio authorization and fresh outputs required.
uv run --active --no-sync -m pipeline input.srt --config pipeline_config_vosk.json \
  --reviewed-text output/my_text/json/reviewed_native_text.json \
  --output-dir output/my_audio --base-speech-rate 1.25 --max-total-speed 1.5
```

The assistant reads `json/native_text.json` directly: no manual asset copying or
recursive OpenCode process is needed. Save the reviewed copy under a new name.
Retain `schema`, `source_sha256`, `plan_sha256`, `policy`, and each selected row's
**complete `original` object** unchanged. `entries` may contain a unique subset
of source IDs; missing rows keep the original. Each row uses this contract
(the hash is computed from the exact UTF-8 candidate, not a claimed length):

```text
candidate: string or null
review: null, or {
  verdict: "accept" | "revise" | "refusal",
  text_sha256: SHA256(candidate UTF-8), or null for a null candidate,
  model: reviewer model identifier,
  mode: reviewer mode (for example native_separate_reviewer)
}
```

Use a separate native reviewer for meaning, natural wording, word-form numbers,
and full source context. Verdicts are **model-only evidence**, not human/scene
approval or proof of audio fit. A real later model run must publish its standalone
HTML report; offline preparation/checks do not produce HTML.
Only accepted, nonempty candidates strictly shorter after the existing lightweight
Vosk punctuation/whitespace normalization can replace originals with valid timing.
Nominal `fits` is advisory: leave such originals alone by default, but an explicitly
selected reviewed candidate may replace them (for example after a measured overrun),
without changing the speed policy. A shorter accepted candidate may exceed its
advisory character target. Invalid source timing/text, new numeric digit tokens,
unchanged/nonshorter candidates, refusals, and revise/null rows keep the original
with reasons in `json/reviewed_text_report.json`. Unknown or
duplicate IDs, source/policy/original mismatches, and stale candidate-review hashes
fail **before normalization/TTS**. Old experimental `final_results.json` files are
not supported handoffs. Changing source, CPS, base, or cap requires fresh preparation.
RUNorm/accents happen only in the later audio stage; the offline length check does
not predict their output or validate pronunciation.

**One speed budget:** the old nominal Vosk analyzer already allowed 1.5; Edge's
separate 1.3 config is unchanged. The familiar Vosk synthesis base is fixed **1.25**,
not unaccelerated 1.0. `tts.max_total_speed` is the authoritative cap (1–1.5);
the analyzer derives its `max_mismatch_ratio` from that cap, ignoring a divergent
old analysis setting. `--max-total-speed 1.3` aligns both stages exactly.
Legacy `use_analysis_speech_rate=true` is explicitly disabled/logged in this
bounded pipeline; the UI hides the obsolete predicted-rate/extra-percent controls.
Standalone synthesis and legacy audio analysis retain their old behavior.

Final WAV durations are measured after silence reduction and sample-rate conversion.
Each cue uses the shared planner window, including only eligible following gap
minus the **single 200 ms gap reserve**; cue timestamps are unchanged. Already
fitting audio gets no extra tempo. Required tempo is `max(1, actual/window)` and
must satisfy `synthesis_rate × tempo <= max_total_speed`: at base 1.25/cap 1.5 the
remaining factor is at most **1.20**, not another unconditional 1.10. A saved
experimental rate-1.5 profile already includes rate 1.5: at that same total cap
there is **no** further acceleration budget. This multiplication accounts for
settings, not a mathematical guarantee of speech speed relative to human 1.0.

If fitting would need more than the remaining cap, the cue stays **unresolved**
and unchanged (no futile acceleration or truncation). Bounded tempo requires
ffmpeg `atempo`; missing/failed ffmpeg never falls back to pitch-changing resampling.
Output is remeasured. A first-render overrun of at most 20 ms may receive at most
one additional render from the same original input, within the remaining factor
cap; this is an internal render target, not timing tolerance. No compounded
corrective input is used. A remaining overrun is honestly unresolved.
`json/audio_timing.json` records before/after durations, windows, synthesis rate,
required/applied factors, total speed, statuses and failure count. CLI/UI display
that count. The Vosk UI **stops before mixing/export** if any cue is unresolved,
invalid or missing; it exposes the timing report for download and retains segments
for inspection or a separately reviewed retry. It does not export an overlapping
unresolved mix. Disabling post tempo still measures and applies this export gate.
The CLI `pipeline` returns **segments plus their timing report**, not a final mixed
track; producing segments is not an all-cues-fit or scene/quality approval.

Use fresh segment directories: synthesis can skip/fail rows, so the pipeline rejects
nonempty segment outputs rather than measuring stale WAVs. A bounded post-stage
repeat with the same complete report, settings and WAV hashes returns saved results
without compounding tempo. Changed WAVs/settings or interrupted pending reports are
rejected; preserve them and synthesize afresh in a new output directory. Arbitrary
external pre-accelerated WAVs without rate provenance are not supported: the bounded
API requires the caller's known actual synthesis rate and its own retained report.
Deleting that report outside the pipeline removes its repeat-processing evidence;
the direct utility cannot prove a caller-supplied rate or detect undocumented prior tempo.

## Current fresh episode run

The independent full-episode run launched at **2026-09-30 11:00:08 +03:00**
and ended **15:19:23 +03:00**, exit 0 (4 h 19 min). Its real
[final HTML report](../output/openrouter_episode_20260930_clean/shortening.html)
and [machine report](../output/openrouter_episode_20260930_clean/report.json)
record 343 completed calls, $4.34939095206 reported cost, $0 unknown reserve,
1,523 source cues, 1,219 selected, 551 accepted changed cues, 124 measured
original fits, and 544 unresolved (282 short-timing, 262 manual). Source IDs
and timestamps remain intact in SRT/JSON. This completes one technical
full-file run; it does **not** establish semantic quality or audio fit for all
551 changed cues. A bounded [local TTS timing sample](../output/accepted_tts_check_20261001/accepted_timing.json)
(12 accepted finals and 12 originals; [24 WAVs](../output/accepted_tts_check_20261001/wav/))
found 12/12 finals fitting the frozen windows and 2/12 originals fitting at
voice 8/rate 1.5 with the configured 200 ms gap. The smallest final margin
was 41 ms. This nonrepresentative sample was not listened to or semantically
reviewed; the other 539 accepted finals remain unmeasured. The 304 unselected
cues fit by prediction only; two more full-file pilots and SH-10 remain open.
The old v2 recovery is a separate lineage and separate budget. The following
is the historical launch command, **not** an instruction to run it again:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shortening_terminal \
  '/home/vladimir-barbuta/Downloads/subs_found_for_testing/Australian Survivor S01E03_ru.srt' \
  --output-dir output/openrouter_episode_20260930_clean \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --model-path models/vosk-model-tts-ru-0.10-multi --voice 8 --rate 1.5 \
  --budget-usd 10 --max-requests 1000 --chunk-seconds 21600
```

It used Muse low / Sol 6 high in windows of 12, with the default one bounded
repair/recheck round. The wrapper passed the 21,600-second soft-stop deadline
to the CLI, not 600 seconds. Actual execution was detached with `setsid`
and `nohup`, input from `/dev/null` and output redirected to
`/tmp/opencode/episode_clean_20260930_launcher.log`; its internal log is
`output/logs/terminal-20260930-110008-2758959-1790755208999476798.log`.
The terminal log ends `child_exit=0` at 15:19:23. Saved journals and report
contain the completed accounting; startup-only observations no longer describe
the run's status. Do not spawn a duplicate or treat the prior lineage's
unknown request IDs as retries for this run. Follow the
[current verdict and remaining plan](shortening_tasks.md#resume-here--30-september-2026)
before any new paid work.

Only the previous v2 presentation files were moved to
[`archived_reports/20260930T105949364412504+0300/`](../output/openrouter_episode_20260930_windowed_recovery_v2/archived_reports/20260930T105949364412504+0300/).
Its journals, manifest, receipts, proposals, known reported $1.882997127 and
unknown reservations $5.175248 remain historical evidence, **not** this
new run's $10 accounting and not proof of erased provider charges. The old
v2 recovery command below is blocked under its own frozen budget and is
not an active gate or command for this fresh run.

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

## Experimental MiMo arbiter check

This **separate** single-candidate comparison does not change the production
Muse-low/Sol-high roles or rerun the episode. Eight saved candidate/Sol-review
pairs are already frozen in
`output/mimo_arbiter_check_20261001/manifest.json`; do **not** repeat
`--prepare`. The configured OmniRoute endpoint is
`http://127.0.0.1:20128/v1`, remote ID `opencode-go/mimo-v2.6-pro`.
The workspace `.env` key restored authenticated `/models` access (200, exact
ID listed). The first reply used backend name `mimo-v2.6-pro`; live model
metadata (`owned_by: opencode-go`, `root/name: mimo-v2.6-pro`) and pinned
OmniRoute routing/response source establish this **specific** prefix mapping,
recorded in `identity_verification.json`. They do not establish upstream
account selection, fallback behavior or pricing. The first raw response was
validated offline in a new journal event, preserving its original error
receipt. The following **historical** first command must not be rerun:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.mimo_arbiter_check --run
```

The separate `--resume` command subsequently sent **only seven** untouched
candidates; it also must not be rerun. The append-only ledger now has eight
starts, eight completions and one offline validation. The
[real final HTML](../output/mimo_arbiter_check_20261001/report.html) records
every proposal, the saved Sol decision, MiMo response, model/mode, historical
final text and status. Across eight requests: 6,075 input / 3,143 output
tokens (2,289 reasoning tokens are a subset of output), median 12.576 seconds;
all USD costs are **unknown**, not zero. Exact verdict agreement was 3/8 on
these purposively selected cases. Sol's historical batches versus MiMo's
single-candidate prompts confound the comparison; agreement is not a quality
score and does not justify a production replacement. The episode SRT/JSON and
production Muse-low/Sol-high path were not changed.

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
contract and **cannot** be resumed by this code. Its explicit offline migration
is saved separately at `output/openrouter_episode_20260930_windowed_recovery_v2`;
the source is not changed. The first offline descendant without inherited-byte
binding (`output/openrouter_episode_20260930_windowed_recovery`) is superseded:
do not resume or alter it. The v2 descendant replays its old generation requests
in their original order, then reviews saved proposals in windows. At migration
time its unresolved last generation start blocked continuation until an explicit
once-only retry (since used; see the current status below).
It retains the original unknown reservation and prior physical attempts against
the same $10/1,000-attempt limits; it does not redo preflight audio. At
migration time the partial report contained only inherited results; its
[archived HTML](../output/openrouter_episode_20260930_windowed_recovery_v2/archived_reports/20260930T105949364412504+0300/shortening.html)
was subsequently updated by live continuation.

The v2 descendant has since made live progress: the current read-only journal
has 227 starts and 225 completions; 264 accepted corrections are in the saved
partial report. The old generation unknown was explicitly superseded; the
**current** unresolved start is review request
`262e18e40b1941fd856fae408d2dec3e`. Known reported USD is $1.882997127,
unknown reservations $5.175248, and the guard's remaining room $2.941754873
of the **original** $10 cap. This is not an invoice. The last terminal log
`/tmp/opencode/episode_v2_live_20260930.log` records interruption during a
request; it does not establish why the process stopped or its exit code.
No live requests are made by this documentation or wrapper work.

**Budget blocker, checked offline against the frozen policy and journal:** the
next Sol review replacement reserves $4.85576, exceeding the $2.941754873
remaining under the $10 guard. The projected guarded total is $11.914005127.
`BudgetPolicy.admit` returns `blocked_by_budget` /
`reservation_exceeds_limit`. The saved `budget.status: ok` only describes the
current ledger *before admitting the next request*; it does not authorize that
replacement. The wrapper is ready, but this episode's next paid request is
**not admissible** with the current frozen cap. Do not infer that running the
command below will make progress. No retry event or request was created by
this offline check. Keep the manifest, unknown reservations and $10 cap intact;
decide separately how to proceed after reviewing actual accounting evidence.

## One terminal chunk (historical blocked v2 recovery; do not run)

From the repository root, **keep the terminal open**. The wrapper runs the
unchanged production CLI as one child, then exits; it never automatically
resumes or retries. Its default 600-second soft stop sends the CLI's first
SIGINT once, asking it to stop **before the next request**. The active request
can take up to its 300-second client timeout, followed by local checkpointing;
600 seconds is not a hard wall-clock limit. The v2 manifest freezes
`run_time_limit_sec` as `null`: adding `--run-time-limit 600` to that resume
would fail the contract. The wrapper therefore does not add that flag on
resume and uses the existing CLI signal path; new runs receive the CLI's
`--run-time-limit 600` as well. The lineage's $10 guard still applies.

**Blocked example, do not run expecting continuation under the current $10
contract:** this first command contains the *current* unknown ID but its
replacement fails monetary admission before sending a request. It may incur a
charge for both old and replacement attempts only if a future separately
approved, admissible continuation exists. Do not use the superseded
`c966047a268147149ab9976bd511589a` ID again:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shortening_terminal \
  '/home/vladimir-barbuta/Downloads/subs_found_for_testing/Australian Survivor S01E03_ru.srt' \
  --output-dir output/openrouter_episode_20260930_windowed_recovery_v2 \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --model-path models/vosk-model-tts-ru-0.10-multi --voice 8 --rate 1.5 \
  --budget-usd 10 --resume --retry-unknown 262e18e40b1941fd856fae408d2dec3e
```

The following is a **future** separate-chunk template, *not* a way around this
blocker. Use it only after an admissible authorized replacement has completed;
omit the previously used retry flag. If a *new* unknown appears, stop and
decide separately about its exact ID. Otherwise:

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shortening_terminal \
  '/home/vladimir-barbuta/Downloads/subs_found_for_testing/Australian Survivor S01E03_ru.srt' \
  --output-dir output/openrouter_episode_20260930_windowed_recovery_v2 \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --model-path models/vosk-model-tts-ru-0.10-multi --voice 8 --rate 1.5 \
  --budget-usd 10 --resume
```

Console and a new append-safe `output/.../logs/terminal-*.log` file receive
timestamped child stdout/stderr and 15-second heartbeats: child PID/liveness,
current journal stage/model/request prefix/duration, last durable completion,
active unresolved IDs, and accepted/unresolved counts from the **last saved
checkpoint** (stale during a request). After 360 seconds without journal or
child-output activity the wrapper warns; heartbeat itself is not progress and
the cause remains unknown. It prints the child exit code, log path and active
unknown IDs at the end. A first Ctrl+C is forwarded once for graceful stop;
a second Ctrl+C forwards a cancellation signal and may create unknown billing.
The 600-second automatic soft stop already counts as that first signal, so a
subsequent Ctrl+C may cancel rather than request another graceful stop.
Neither signal guarantees immediate exit in synchronous local work; the
wrapper does not kill the child. Closing the terminal may orphan or interrupt
it, so wait for the `END child_exit=...` line. On disk-log failure the console
continues with a warning. `--help` only prints wrapper help, without launching
the production child; other production CLI flags are passed through.

The offline migration entrypoint is
`uv run --active --no-sync -m utils.shortening_migrate --source SOURCE_DIR --destination NEW_DIR`.
It requires a nonexistent destination, verifies saved input, profile, preflight
report, journal and receipts under the source lock, and leaves an incomplete
marker until the separate contract and inherited HTML are ready. It has already
been run for this episode; do not rerun it into an existing directory. Resume
checks the bound inherited byte prefixes of requests/receipts and the complete
baseline/preflight evidence before any retry event or transport; newly appended
journal entries remain permitted. The accepted source snapshot is narrowly
identified by its saved report hash; its summary and unprocessed cues cannot be
silently fabricated. Offline regression tests use independent generated fixtures;
real v2 accounting is verified separately against the saved episode.
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

Zero-cap fix (4 Oct 2026): ordinary `analyze_text` uses nominal CPS without a
profile; the calibrated planner's intercept and guard previously made 344 valid
short windows receive zero targets. Only a calibrated zero now falls back to
`floor(cps * max_speed * available_ms / 1000)` when positive, labelled
`nominal_cps_fallback_zero_calibration`; `calibrated_max_chars=0` and the timing
warning remain. Positive calibrated caps, source timing and the frozen profile
are unchanged; a 1 ms nominal zero remains zero, and invalid timing stays invalid.
This is a text-only target, not measured audio fit or semantic approval; changed
text still requires separate audio verification. Offline evidence:
`output/zero_character_target_fix_20261004/before_after.json` (344 targets: 0 to
3–23; 1,179 positive caps unchanged; 44 historical actual fits preserved).
No new TTS/model calls or historical artifact rewrites were used. Saved runs
remain viewable; changed module hashes still block resume without explicit migration.

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

### Future voiceover checkpoint — 5 Oct 2026

The user listened to three saved Silero before/after pairs (1262, 1259, 526)
and preferred AFTER for **Vosk multi 0.10 voice 8**. This approves retaining the
fixed pause stage for future runs, not an episode/TTS rerun or other voices.
The CLI, Vosk UI and native batch now use: fixed-rate synthesis → legacy
500→200 ms silence reduction at −35 dBFS → resample → Silero/energy-agreed
internal cuts → actual PCM measurement → only needed bounded tempo. Base 1.25,
total cap 1.5 (remaining tempo ≤1.2), and cap 1.3 remain supported.

`utils.silero_pauses` uses installed `silero-vad==6.2.3`, bundled hash-checked
ONNX on CPU, and existing torch/onnxruntime/soxr; it never downloads assets.
Detection is mono 16 kHz in memory (threshold .5, speech 250 ms, silence
100 ms, pad 30 ms). RMS −50 dBFS/10 ms, any-channel activity, 40/20 ms guards
and interior padded-speech gaps must agree; cores ≥200 ms retain 150 ms by
central deletion. The internal stage makes no edge, text, gain or speed edits; stereo detector input
is rejected rather than downmixed. VAD does not prove word/consonant safety.
Missing local assets fail clearly when enabled; disable explicitly to use legacy.

For a **separately authorized future run**, from the activated environment:

```bash
uv run --offline --no-project -m pipeline INPUT.srt \
  --config pipeline_config_vosk.json --output-dir output/NEW_RUN
```

That chosen config enables pause processing and complete WAV assembly into
`mix/voiceover.wav`. Other/old configs without these enabled keys keep their
legacy stage meaning. `--mix` explicitly requests assembly;
`--no-silero-pauses` disables the new stage. UI has a voice-8-only checkbox,
ignored for Edge. Complete mixing uses original subtitle starts, never
back-to-back concatenation. Every source cue must have its expected filename,
current hash-bound measured fit and current processing provenance; missing,
unresolved, stale or overlapping segments block export. Vosk UI also uses
`missing_policy="error"`. Reviewed-text semantic gates remain unchanged.

Native batches still process only the requested source-position slice:
`ready_for_separate_mix` is **not full-episode readiness**. Complete assembly
requires the complete verified source. Batch RAW and legacy PRETEMPO stay
immutable; new `pauses.wav` precedes FINAL, and measured repair advice uses
that actual pause-processed duration. CLI/UI retain both copies in `pause_stage`
plus `json/silero_pauses.json` receipts. Policy/model/helper identity is bound
into new runtime contracts. Old frozen bindings now fail explicitly on changed
code/runtime; do not rebind, resume or rewrite saved successful pilots.

Evidence: [local integration checkpoint](../output/pipeline_vad_integration_20261005/README.md).
Exactly three offline VAD reproductions matched approved WAV/PCM, including
1259's correct **177759** frames; no new TTS/text-model calls or local HTML.

### Conditional edge approval — 5 Oct 2026

The user subsequently listened to **all eight conservative edge previews** and
approved retaining the fixed policy for selected Vosk voice 8. The extra stage
now runs after internal cuts **only if** actual current frames × 1000 exceed
the unchanged canonical available milliseconds × sample rate, before bounded
tempo. Already-fitting or exactly-fitting clips are byte-copied without edge
cuts. `plan_subtitles` supplies full-source canonical windows, including its
existing gap allowance; cue end−start is not substituted and no new allocation
or source-start shift is introduced.

The unchanged preview policy protects the union of primary RMS −60 dBFS/10 ms
any-channel activity and padded VAD speech from the **same internal receipt**,
mapped through internal cuts exactly. Normal head/tail guards are 40/80 ms;
short guards are 100/150 ms for ≤2 punctuation-only chosen-text words **or**
original PRETEMPO ≤1.5 s. Text is bound before RUNorm/RUAccent; missing original
text conservatively uses short guards. Minimum side cut is 40 ms. Diagnostic
−50 dBFS is only an uncertainty gate: endpoint disagreement >40 ms keeps that
edge. No VAD speech, no RMS activity or unsupported stereo cannot enable cuts.
There is no second VAD, added silence, internal-word edit or relaxed threshold.
After edges, measured tempo is unchanged: base 1.25, total ≤1.5 (remaining ≤1.2),
with total cap 1.3 still supported. Guards remain even if the clip cannot fit.

Chosen `pipeline_config_vosk.json` enables `silero_edges.enabled`; old configs
without the key retain internal-only behavior. `--no-silero-edges` permits that
comparison. `--no-silero-pauses` disables both stages. UI has a separate checkbox;
enabled edges with disabled internal processing fail before TTS. Other voices
remain rejected when enabled, Edge remains unaffected. New immutable `edges.wav`
and receipts record current PCM savings, the conditional reason, original text
hash and exact window. Native measured repair advice uses edges first; new runtime
bindings include edge policy and enabled flag. Full mixing validates current
edge text/window/stage provenance; semantic gates and ranged-batch limits remain.

[Offline integration evidence](../output/pipeline_edge_integration_20261005/README.md):
all eight unconditional pure plans reproduce the listened-to AFTER exactly;
conditional production replay skips already-fitting 1262/1259 and applies 526
to 3476.961 ms / 3517 ms. No fresh VAD/TTS/LLM/ASR, install, episode or complete
real-track run occurred. Eight-example approval is not an all-word safety or
episode-quality guarantee. Grouping remains off. Speaker confirmation is relevant
only to separately authorized future neighbor joining/borrowing, not this
within-clip pause step; no speaker analysis or diarization was added.

Live output includes SRT and JSON exports, HTML/report artifacts supplied by
the existing publisher, actual cost records, and a review queue. Model
acceptance is not a claim of complete semantic or timing quality. Failed,
malformed, transport-error, semantic-impossible, and unresolved cues retain
their original text; no automatic acceptance or silent data loss is performed.

### Scoped source listening confirmation — 6 Oct 2026

[Human confirmation](../output/pipeline_source_confirmation_20261006/README.md):
the user confirmed one speaker **within each pair** 1263–1264 and 1270–1271;
older unknown-speaker fields are historical for these pairs, not their current
within-pair status. No diarization is needed for these two pairs. This does not
identify a voice or link speakers across pairs, verify exact speech boundaries,
resolve puzzle translation meaning, approve grouped text/overall quality, or
authorize retiming/TTS. Texts and frozen artifacts stay unchanged. Next suggestion
only: independently review one joint candidate per thought and actual timing;
no candidate or follow-up execution is approved here.
