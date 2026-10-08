# Задачи: надёжное сокращение полного файла

## Latest independent validation — two rounds exhausted, 5 October 2026

[Real report](../output/pipeline_batch_validation_20261005/report.html) ·
[completion evidence](../output/pipeline_batch_validation_20261005/final_completion.json) ·
[final exception queue](../output/pipeline_batch_validation_20261005/final_exceptions.tsv).
Fixed disjoint positions 500–530, 31 cues / 65.449 s: **8/31 → 16/31 fits
(25.8% → 51.6%)** under unchanged voice/base/cap/windows/pause settings. Final
texts: ten originals, 21 rewrites; fit classes: seven direct, nine accelerated.
Actual TTS was **31 + 19 = 50**. Eight initial fits were retained byte-exact;
the four null/refusal targets had no second TTS. Fifteen exceptions remain:
eleven accepted texts require tempo above 1.2, plus four retained first-round
timing failures. Zero normalization or bounded-tempo residual failures.

Round-2 advice: seven met → four fits/three failures; twelve over → four
fits/eight failures; four null are not under-budget texts. Events: 32 proposals
(28 nonnull + four null), 27 semantic accepts, one revise, four refusals—not
32 source cues. Four native Sol 6.1 medium invocations accepted two exact full
assemblies; no human listening approval. Audio wall time was **215.874523 s**
(123.547159 + 92.327364), not native end-to-end time. Native USD/physical request
counts remain unknown; Python paid API calls were zero.

Safe completed resume added zero inference/TTS; all 150 WAVs, 93 first-round
WAVs, eight fitted cue states, 240 prior artifacts and 323 historical hashes
were preserved. The real report passed offline Chrome checks. Production code,
config and tests were unchanged; the earlier 94 tests were not rerun. The older
105.821 s span reached 13/31 → 20/31, but different text/density makes comparison
descriptive only, not proof that the patch helps or harms. **Final stop: export
blocked; no third repair, manual polishing, mix/video or commit.** Defer predictor
choice and full-batch readiness claims to a separately scoped systemic timing
decision, not another per-cue repair. Native orchestration remains necessary.

## Latest offline checkpoint — measured advice presentation, 5 October 2026

[Diagnosis](../output/pipeline_batch_calibration_20261005/analysis.json) ·
[replay](../output/pipeline_batch_calibration_20261005/implementation/request_replay.json) ·
[checks](../output/pipeline_batch_calibration_20261005/implementation/checks.json).
Completed the minimal request/reviewer contract change, not a new predictor or
live pilot. Of ten timing failures, seven exceeded measured advice, two met it
but failed, and one missed by 5.766 ms after tempo. Normalization remains separate.
The 16 paired measurements do not justify changing the **existing budget formula**.
Canonical current character counts and measured required reductions are explicit;
spaces/punctuation count before RUNorm/RUAccent/model-added stress. Historical
nominal targets stay unchanged. Exact-candidate advisory warnings are nonblocking;
semantic acceptance and under-budget text are not measured audio fit.

Offline replay: **17/17 measured budgets identical**, 18 target warnings/checks
(nine exceeded, eight not exceeded, one unavailable). **94 focused batch/native
tests passed**, with expensive work mocked. Zero new inference/TTS/render calls;
all prior pilot files, 141 WAVs and prior calibration evidence remain unchanged.
The old controller runtime intentionally fails read-only loading; its saved
report remains directly viewable. Observed yield is still **20/31**, not improved
by this presentation change. Native orchestration is clearer, not unattended.
Next evidence criterion, only if separately authorized: an independent contiguous
prospective sample under unchanged voice/base/cap/windows/pause policy and no
manual edits. No such run, further repair, package change or commit occurred here.

## Latest continuous batch checkpoint — 5 October 2026

[Real standalone report](../output/pipeline_batch_pilot_20261005/report.html) ·
[final status](../output/pipeline_batch_pilot_20261005/status_final.json) ·
[final evidence](../output/pipeline_batch_pilot_20261005/final_completion.json) ·
[exception queue with chosen text and original times](../output/pipeline_batch_pilot_20261005/final_exceptions.tsv).

Fixed original positions 1250–1280: 31 cues, 105.821 s, full 1,523-cue source
context and canonical windows. Voice 8 / base 1.25 / total cap 1.5 are unchanged.
The automatic yield is **13/31 (41.9%) → 20/31 (64.5%)** after exactly one measured
repair batch; eight final source texts and 23 native rewrites. All 13 initial
fits were reused byte-exact. Four actual native Sol 6.1 medium writer/reviewer
invocations supplied 16 + 18 proposals and exact final-assembly review; 34
proposal events are not 34 source cues. Semantic acceptance remains model-only.

Actual TTS: **30 + 17 = 47**, no hidden retries. Two normalization attempts failed
before TTS; the final `cue-1260` guard rejects `крут` → `крутой`. Ten timing
exceptions remain: nine require tempo above 1.2, plus `cue-1266` at total cap 1.5
still 5.766 ms over its window. Final fit classes: eight direct, twelve tempo-fits.
Audio wall times: 131.426 s and 89.842 s (221.268 s combined, excluding native work).
Completed audio resume was checked: zero new inference/TTS, unchanged whole
state and 141 WAV hashes. Earlier 323 historical hashes and all 14 preparation
files are unchanged. Browser/metadata checks are technical, not listening approval.

**Final stop: eleven exceptions block mixing/export; no third repair or manual
cue fixes.** No video/episode-wide run, package/code change or new price estimate.
Native USD/physical provider request counts are unknown; Python paid API calls
are zero. The batch controller exists and handles partial failures/resume, but
this yield does not pass practical fully automatic/mix-ready or ten-episode
readiness. Python is not an unattended native-model orchestrator. The earlier
168 code tests remain valid and were not rerun for unchanged production code.
Audio completed before the delegation interruption; only offline checks/docs
needed resuming, with no unknown expensive intent or repeated TTS. Earlier
checkpoints below remain historical; do not automatically continue spending.

## Resume here — 4 October 2026

**Latest continuous-scene checkpoint — final attempt exhausted, 6/9 fit:**
[standalone 13-proposal/13-review report](../output/pipeline_scene_20261004/view9/report.html) ·
[current nine audio receipts](../output/pipeline_scene_20261004/view9/repair/audio/final_results.json) ·
[final checks](../output/pipeline_scene_20261004/view9/repair/audio/final_checks.json) ·
[33-second original reference](../output/pipeline_scene_20261004/view9/repair/audio/reference.mp4).
Seven final texts each received one new synthesis (voice 8, base 1.25, total cap
1.5); cue-1262's initial WAV and the historical cue-1265 WAV were carried byte-exact.
This scene has 15 new WAVs across initial/final phases, not 15 source cues.
Final blockers are cue-1263, cue-1264 and cue-1267; their required post-tempo factors
exceed 1.2. No dubbed preview or assembled voiceover was exported, and no further
repair/model/TTS round is authorized. The original reference retains unmuted audio,
33 seconds, 825 frames at 25 fps; crop encoding and the 20 ms video-grid difference
are recorded. The common -7900 ms anchor remains approximate, with visual residuals
up to about 0.4 s; speech synchronization and human listening are unverified.
All 13 proposal events have model-only accepts. Final reviewer medium/session are
parent-confirmed native-request provenance, not raw-worker self-verification or
billing proof. The shared ellipsis fix passed 105 focused tests earlier; those
unchanged tests were not rerun. Source, previous audio and raw reviews are preserved.
Next: user inspection of the source/current individual clips and a decision about
the remaining blockers, not an automatic continuation. Earlier checkpoints follow unchanged.

**Latest final bounded checkpoint — 10/10 selected targets fit; scene/video not passed:**
[standalone current report](../output/pipeline_finish3_20261004/report.html) ·
[final results and five exact reviews](../output/pipeline_finish3_20261004/repair/final_results.json) ·
[saved checks](../output/pipeline_finish3_20261004/repair/final_checks.json).
The final two accepted texts were each run once as their unchanged original pair,
voice 8, base **1.25**, total cap **1.5**, without a worker timeout. Cue-6 measured
**2.507755 → 2.410023 s / 2.429 s**; cue-1265 measured **3.691973 → 3.523832 s /
3.526 s**. Applied tempo factors were **1.041714 / 1.047071**; conservative measured
total bounds **1.302143 / 1.309644**. Exact frame timing, source windows and spoken
`к+эт` were verified; no source retiming, trimming or pitch-changing fallback.
All eight previously successful target texts/WAVs remain unchanged, including
cue-809; cue-414 still uses its original text/audio without a new Sol review.

Five real proposals have five exact-hash Sol 6.1 medium accepts across the two
phases (three initial, two final repairs), not five distinct source cues or human
approval. The user approved the earlier pilot sound, not the new audio or scene
synchronization. **Ten successful new WAVs** exist across these phases: six from
the first finish-three stage/recovery and four from the final repair. The earlier
interrupted internal TTS attempt count remains unknown. Original neighbors remain
separate: **1/3 fit** previously, **1/2 fit** in the final repair; cue-1266 still
overruns **2.856054 / 2.237 s**. Do not mask that with the 10/10 primary-target score.
The 373 historical files and eight retained target WAV hashes are unchanged;
normalizer/test hashes are unchanged and the passing 120-test suite was not rerun.
The final report embeds all ten current target WAVs byte-for-byte, with a 4 MB raw
audio allowance. No further repair/TTS rounds, whole-episode mix or video rendering
were performed. Source video covering the planned continuous 12-cue section and
its scene-local time mapping are still missing. Next is user listening and a
matching local source-video path, not an automatic new run. Actual billing is unknown.

**Earlier bounded finish-three checkpoint — real audio partial, no retry:**
[standalone report](../output/pipeline_finish3_20261004/report.html) ·
[audio evidence](../output/pipeline_finish3_20261004/audio_summary.json) ·
[approved runtime patch](../output/pipeline_finish3_20261004/runtime_patch.json).
The user listened to and approved the existing pilot sound, not all-ten timing,
the new audio, or scene synchronization. All three exact new Sol 6.1 medium texts
have separate exact-hash Sol acceptance. A narrow local RUNorm subclass now protects
only the whole source token `Кэт` at abbreviation expansion: cloned, validated
source offsets; crossing/foreign spans fail closed; other normalization and the
lexical guard remain active. New and historical shortened variants retain `кэт`
and receive `к+эт` stress. Inherited case lowering is not exact-case restoration.
Package source and the 373 historical files are unchanged; **120 focused tests
passed once** after the adapter change, and seven successful old texts/WAVs remain.

Three original pairs were invoked once, base **1.25**, total cap **1.5**, voice 8.
Four new WAVs are available: cue-809 fits **1.473220 / 1.491 s**; cue-6 remains
unresolved **2.948934 / 2.429 s**, requiring tempo 1.214053 above the remaining 1.20.
The worker's 120-second foreground tool deadline interrupted cue-1265 after
normalization, without a WAV or TTS receipt; its timing and internal TTS attempt
count are unknown. No retry was made. Current target timing: **8/10 fit**, one
unresolved and one unmeasured/interrupted. Original neighbors are counted separately:
one fit, one over cap, one interrupted. The finite same-source ffmpeg correction
branch is now demonstrated by two real renders each for cue-7 and cue-809, without
extra TTS. Do not automatically repair or rerun the bounded round.
Source video covering cue-809/1265 and scene time mapping are still missing; the
continuous **59:05.978–59:45.319**, 12-cue section is only a plan. No episode mix,
video export, further model calls, downloads, or new price estimate occurred.

**Current practical pipeline checkpoint — small live pilot completed:**
[standalone Russian report with ten current embedded WAVs](../output/pipeline_live10_20261004/report.html),
[final presentation data](../output/pipeline_live10_20261004/final_results.json).
Strict timing results are **phase one 1/10 → bounded text-repair phase 2/10 →
technical processing 7/10**. `cue-6`, `cue-809`, and `cue-1265` remain unresolved;
the selected hard ten are not an episode success-rate sample. Nine new texts
have exact-hash Sol acceptance; `cue-414` keeps its original and prior fitting WAV.
All 13 proposals/reviews and the first quote failure/retry are preserved.

User-authorized normalizer downloads/load succeeded without package-version
changes. Implemented root fixes: supported punctuation plus exact input logging
at the Vosk boundary; finite same-source ffmpeg correction without trimming or
changing physical windows/total cap 1.5; and fail-closed normalization word checks.
Five real 6–15 ms misses needed **five additional ffmpeg renders, no new TTS**;
each fitted on its first technical render. The bounded second-pass branch has
mocked coverage, not a demonstrated second real render. Saved focused checks:
**81 passed**, compile/diff checks passed; this final presentation does not rerun
them or imply a full end-to-end rerun after the new guards.

`cue-1265` selected `Кэт`, but RUNorm changed the captured TTS input to `кати`.
The new guard detects that change before future TTS; it does not correct the
unchanged historical bad clip. RUNorm word checks apply to already-written Russian
without Latin letters/digits; RUAccent preserves word/numeric tokens for all input.
Names, pronunciation and overall meaning still require listening/context review.
Actual pilot totals: **38 TTS attempts, 37 successful historical TTS WAVs, plus
five derivative WAVs**. Native accounting: **15 new assistant responses**, exactly
two new user turns per reused Sol 6.1 medium role, excluding 54 prior responses.
Conditional writer/reviewer API equivalent **$0.2367976–$0.2498576**, not an invoice
or full-system price; normalization/TTS compute, downloads and orchestration are
excluded, and native cost zero is not free.

**Next: user listening to the seven timing-fitting clips with whole-utterance
original context.** No human/audio approval or full-episode readiness is claimed.
Do not automatically schedule another benchmark, inference round, full-file run,
or raise the 1.5 ceiling. After listening, any further engineering needs separate
scope: protecting Kat from normalization or an explicit pronunciation-dictionary
spelling, and explicit handling of the three hard cases. Direct English→Russian
translation remains deferred.

**Initial practical pipeline implementation checkpoint (historical):** `pipeline` has an offline native-text
prepare/reviewed-import boundary and the CLI/UI share one bounded Vosk speed policy:
fixed base 1.25, total cap 1.5 (or an explicit aligned 1.3). Actual measured overrun
alone can request the remaining ffmpeg tempo; fitting audio gets none, over-cap
audio stays unresolved without truncation, and repeat post-processing is hash-bound.
Nominal source `fits` remains advisory: an explicitly bound accepted shorter text
can replace it without changing policy. Vosk UI unresolved/invalid/missing cues
stop mixing/export and return the timing report while retaining segments; CLI
returns segments/report rather than a final mixed track.
Old predicted dynamic-rate controls are disabled in this path; Edge and standalone
legacy functions remain unchanged. See [commands and handoff contract](shorten_file.md#practical-vosk-pipeline-native-text-handoff-4-october-2026).
Offline implementation checks are saved in
[`output/pipeline_native_handoff_20261004/`](../output/pipeline_native_handoff_20261004/).
**Initial implementation baseline: 104 focused tests passed**, including mocked full CLI/UI audio wiring; compile
and diff checks passed. Offline preparation preserved all **1,523 original cues**
from the frozen source snapshot; fresh nominal targets are not calibrated historical
caps, quality comparisons, or measured audio results.
**After parent review: 54 affected-module tests passed** (not added to the earlier
104): explicit nominal-fit replacement after mocked measured overrun, invalid/no-review
preservation, Vosk overrun/invalid/missing/all-missing export stops, and unchanged
Edge behavior. Compile/diff checks passed; saved preparation/evidence hashes remain
unchanged. The saved-episode test now writes only into its temporary directory.
No new Sol/TTS/API calls, historical-result promotion, HTML report or production
audio run occurred. Next: the current native assistant may write and separately
review a fresh source/policy-bound artifact when requested; audio execution remains
a separate explicit command, with listening/scene approval still outstanding.
The checkpoints below are preserved historical experiments, not current launch instructions.

**Earlier checkpoint — natural-wording ten-cue TEXT-ONLY test completed:** [standalone report](../output/sol61_natural10_20261004/report.html), [final candidates](../output/sol61_natural10_20261004/final_results.json), [completion checks](../output/sol61_natural10_20261004/completion_evidence.json).
All **13 proposals / 13 reviews** retained; ten current candidates have Sol quality `accept`, not user/scene approval, guaranteed naturalness or production readiness. **One is not a shortening: cue-1119 grows from 36 to 41 normalized characters; only nine are strictly shorter.** No pending writer/reviewer work; no automatic promotion.
Same positive matched baseline caps: **1 baseline fit, 2 only hypothetical external tempo 1.10, 7 above both**; **3/10 forecast coverage**, not measured audio success or seven audio failures. Original fallbacks remain separate; `audio_fit=null` throughout.
Reused writer/reviewer Sol 6.1 / medium sessions knew prior answers; this is not fresh blind or a controlled causal benchmark. New prompt and three quality requests changed the policy; original model accepts and two parent overrides remain distinct.
Incremental **14 native responses**, conditional API-equivalent **$0.0965608–$0.1163008**, excluding the previous 40 responses; actual invoice/cache normalization remain unknown. [Usage](../output/sol61_natural10_20261004/usage_evidence.json) · [cost](../output/sol61_natural10_20261004/cost_estimate.json).
**Next decision:** define a separately authorized bounded real-audio acceptance check, including user listening, rather than another text-model benchmark; no production pipeline, audio, global speed change or further experiment was implemented.
Future proposal only: one natural candidate + at most one quality repair, exact-text/settings cached measurements, only the needed tempo up to tentative total 1.10 over the raw profile (never compound stages), otherwise explicit unresolved. Direct English→Russian translation remains [deferred](future_experiments.md#direct-english--russian-translation-under-aligned-russian-timing); predictor-quality/1184–1185 ms boundary work remains future scope.

**Earlier offline checkpoint — user-authorized zero-cap fix completed:** all 344 zero work targets became positive (3–23); zero work targets in the saved 1,523-cue episode: **344 → 0**.
Raw `calibrated_max_chars=0` forecasts and audio-timing warnings remain; target source is explicitly `nominal_cps_fallback_zero_calibration`. Ordinary analyzer caps, 1,179 positive calibrated caps and historical results are unchanged; saved checks: 459 tests passed.
[Before/after evidence](../output/zero_character_target_fix_20261004/before_after.json) · [Check log](../output/zero_character_target_fix_20261004/checks.log). The fix applies to future plans only: no historical report rewrites, new model/TTS calls or automatic rerun; text-target fit is not measured audio fit.

**Historical completed authorized ten-refusal TEXT-ONLY diagnostic (4 Oct):** [Russian HTML with ten originals, alternatives and usage](../output/sol61_refusal10_20261004/report_with_usage.html), [final rows](../output/sol61_refusal10_20261004/final_results.json), [completion](../output/sol61_refusal10_20261004/completion_evidence.json), [zero-cap evidence](../output/sol61_refusal10_20261004/zero_limits.json).
Ten source-order-spaced entries among the 303 saved null/old-cap-fitting cases were fixed before text inspection: **9 accepted, one exhausted over-limit (`cue-414`, SRT 415: 21/12)**. Unresolved output retains the original; the old matched trial remains **294 accept**, not 303.
New policy requires a concrete faithful candidate before any refusal, even if over-limit; program excess feedback precedes bounded repairs, with at most three distinct proposals per ID. All ten first proposals exceeded caps; round 2 added **6 accepts**, round 3 **3**. No nulls, no pending review or writer retry remain.
**24 new proposals / 34 semantic verdict rows (10 historical + 24 new)**, not API-request counts. Six cases used two attempts, four used three; previously eight stopped on immediate null and two on a second-attempt null.
Blind historical alternatives: **6 accept / 4 revise**; final new texts: **10 semantic accept**, nine also pass frozen character limits. Historical approval is not ground truth; same-family separate-session Sol acceptance is not human/scene or audio-fit verification.
`cue-809`'s identical old/new “Упала!” was accepted both times; no contradictory verdicts on identical raw/normalized cue texts were found. `cue-414` remains a bounded search failure, not proof of irreducibility.
These nine successes rebut treating the sampled earlier nulls as impossibility proof, **not** a 90% episode success rate. Selection targets counterexamples; batch size, fresh session and prompt also changed, so a single-policy causal effect is not isolated. Full original context/blinding are parent attestations and intended input restrictions, not an independent tool-read audit.
Diagnostic-only conditional writer+reviewer **$0.3302400–$0.3540000**, complete usage of **40 native responses (17 writer, 23 reviewer), all Sol 6.1 / medium, standard**; includes old/new blind comparison and loading full context in both sessions. Not an invoice, per-cue or episode extrapolation; native cost 0 is not free. [Usage](../output/sol61_refusal10_20261004/usage_evidence.json), [cost/assumptions](../output/sol61_refusal10_20261004/cost_estimate.json).
Input **94,426**, output **2,830**, reasoning **2,376**, cache read **1,130,880**, write **0**; conditional cache share **92.2937%**. Native input/cache and output/reasoning overlap remain unverified; preparation, orchestration and mechanical workers excluded. This does not replace the separate matched-trial **$0.6316724–$0.6720324** below.
**Zero durations: none.** All 344 blocked source spans are positive (**158–1179 ms**; saved available windows **158–1182 ms**). Zero is a conservative character cap: intercept **0.506699 s** + guard **0.619732 s**, slope **0.0580444 s/char**; first positive cap at about **1184.476 ms**, not a mandatory silence/pause.
All 344 originals were present in model context but protocol `needs_rewrite=false` excluded them from rewriting, not a Sol refusal. **44 historically measured fitting originals** show predictor/actual-fit disagreement; SRT 1097 has 1182 ms available versus saved speech 1102.948 ms despite cap 0. Saved gap 200 ms controls borrowing between cues, not another subtraction; no new audio measurement occurred.
**Current priority:** do not promote the full pipeline. The zero-cap fallback is implemented and checked; broader predictor-quality validation against saved measurements is future work, not a blocker for this fix. Candidate-first bounded repair merits later implementation/verification. No automatic full rerun, global retiming/speed changes or audio continuation; SH-10 remains open.
This presentation update made no model/audio/API calls; frozen helpers, state, raw reports, final results, prior matched evidence and accounting files remain unchanged. The dated checkpoint below is preserved history, not the current next-run instruction.

## Earlier checkpoint — 4 October 2026 production-matched trial

**Completed authorized production-matched TEXT-ONLY trial (4 Oct):** [Standalone HTML with usage](../output/sol61_text_matched_20261004/report_with_usage.html), [completion](../output/sol61_text_matched_20261004/completion_evidence.json), [comparison and five examples](../output/sol61_text_matched_20261004/comparison_evidence.json), [usage](../output/sol61_text_matched_20261004/usage_evidence.json), [conditional cost](../output/sol61_text_matched_20261004/cost_estimate.json).
All **1,523 original single cues**, exact production selection **1,219 IDs** and frozen per-cue character caps are preserved: **875 positive-cap targets, 344 blocked zero caps, 304 unselected unchanged**. No groups, recomputed windows or extra reserve subtraction.
Separate writer → program `check_length`/ID/schema checks → separate Sol semantic review → bounded text repairs; at most three distinct candidates/nulls per target. **No audio generation/measurement, speed changes or two-hour cap anywhere in this trial.** Character fit remains predictive, not audio-fit proof.
Final **294 accept (33.6% of 875), 580 writer null, one final semantic revise (`cue-887`)**; no pending review or retry queue. Accepted means separate-session same-family Sol approval, not human/scene validation or production approval; unresolved output keeps the original.
**Critical limitation:** the first **527 nulls stopped after one attempt by protocol**; writer refusals were not separately reviewed. The 580 nulls are not proof of impossibility or model capacity limits.
**1,004 candidate-or-null attempts and 340 verdict rows**, not model-request counts; attempts per target: 757 used one, 107 used two, 11 used three. Five unaccepted cases reached the cap (four null + one revise).
Round 1: **230 accept**, conditional writer+reviewer **$0.4588896–$0.4817896**. Round 2: **+58**, **$0.1189036–$0.1286236**. Round 3: **+6**, **$0.0538792–$0.0616192**. Repairs added 64 accept and must not be automatically removed.
Writer conditional **$0.3843048–$0.4096548**, reviewer **$0.2473676–$0.2623776**; combined **$0.6316724–$0.6720324** for complete usage of all 38 responses. This is a conditional API-price equivalent, **not an invoice**; native cost 0 is not free.
Saved native metadata identifies **all 38 responses as `openai/gpt-6.1-sol#medium` (20 writer, 18 reviewer)**, not an independent upstream receipt. The parent confirms three resumed rounds in each session; round cost sums use their saved user-turn partitions.
Known input **130,566**, output **21,511**, reasoning **4,036**, cache read **1,554,304**, cache write **0**. Conditional cache-read share **92.2507%** assumes input is uncached; all responses use standard pricing, maximum prompts writer **73,530**, reviewer **60,651**, below 272,000.
Output/reasoning overlap and native/OpenRouter cache normalization/route equivalence remain unverified; actual billing/physical POST count unknown. Cost scope excludes preparation, orchestration, mechanical workers and the collector.
Saved timing: **1,522.479 s / 25.37465 min**, first writer response creation → final reviewer response completion, including tools and inter-stage waits, excluding pre-run preparation. Not pure model time or comparable to the historical 4 h 19 min end-to-end run.
On the **same 875 positive-cap IDs**, production had **551 accepted + 80 fitting originals, 244 unresolved**; this trial has **294 accept, 581 unresolved**. Same IDs/caps do not isolate model quality: prompts, decision/reviewer settings and the old actual-fit branch differ.
Accepted-set overlap **247**, old-only **304**, new-only **47**, identical text **31**; 33 new-only IDs already had fitting originals, only 14 were previously unresolved. Of 344 zero-cap IDs, 44 were historically measured fitting originals: blocking them now is a filter difference, not a newly measured regression or physical-impossibility claim.
**303 new nulls have historically accepted text passing the identical cap**; five chronological examples are `cue-6`, `468`, `902`, `1250`, `1522`. These are saved counterexamples to assuming no previously approved cap-compliant option, not fresh semantic approval or proof that all old wording is faithful.
Historical reported **$4.34939095206** is numeric context only, not equal-quality cost evidence or proven “6.5x cheaper”. Earlier prep-only artifacts are not live evidence: `sol61_text_only_20261004` (1,494 units including 25 groups) was not run and is not this matched trial.
**Next priority: do not promote.** A possible bounded TEXT-ONLY examination of early refusals/saved counterexamples requires new user approval; do not automatically resume either child, rerun the episode or add audio work.
Production exports, old **534 accepted finals unmeasured / 544 unresolved**, historical reports and scoped confirmations 19/1025 remain unchanged. This trial does not close the full-file pilots or SH-10 production final-context review.
This saved presentation/documentation update made no new model, reviewer, audio, network or native API calls; the original report matrix and all frozen evidence are preserved.

**Historical full-source native Sol checkpoint (3 Oct, completed trial, partial results):** [Standalone HTML with later usage](../output/sol61_episode_20261003/report_with_usage.html), [completion/failure evidence](../output/sol61_episode_20261003/completion_evidence.json), [native usage](../output/sol61_episode_20261003/usage_evidence_20261003.json), [conditional cost](../output/sol61_episode_20261003/cost_estimate_20261003.json).
All **1,523 original cues** are accounted for exactly once in **1,494 units: 25 saved groups + 1,469 singles**; every original unit was measured fresh or cached, not every unit successfully shortened.
**527 timing/self-assessment-ready (184 original + 343 changed); 967 unresolved (960 need shortening + 5 zero-budget + 2 failed).** The 83 self-classified irreducible units are within the 960, not an extra count or proof of impossibility.
**1,222 units proposed; 1,311 candidate attempts; 2,243 physical TTS attempts (1,034 original + 1,209 candidate), 2,241 successful.** The real measured-overrun → successful-repair branch has **39 cases**.
At most two candidate variants were used (three authorized); the assistant-chosen 7,200 s helper wall (not a user or Sol requirement) stopped third variants and many second variants. The **124.9643-minute** native session includes setup/read/completion, unlike the helper clock from first measurement.
Saved metadata identifies `openai/gpt-6.1-sol#medium` for all **92** responses in `ses_efd8bd4adffejZEg6OW298q0Po`; not an independent upstream receipt or physical POST count. Semantics remain same-model self-assessment, not production approval or independent review of the 343 changes.
Known usage: input **323,019**, cache read **13,790,336**, output **85,428**, reasoning **3,807**, cache write **0**. Conditional cache-read share is **97.7113%** if input is uncached; all 91 priced prompts use the standard tier (maximum 228,430 < 272,000).
**$2.8793516–$2.9174216 is an incomplete conditional API equivalent for 91 records**, with one error/retry record missing usage; native cost 0 is not free, and the actual invoice/full-bill upper bound is unknown. Output/reasoning overlap and native/OpenRouter normalization remain unverified.
This covers only the main session, excluding preparation, orchestration/review and local TTS compute. No-cache **$29.080990–$29.119060** is the same-token-trajectory counterfactual, not measured savings; the earlier three-group trial is separate, not summed by default.
`cue-2` (“24 австралийца…”) and `cue-22` (“Осталось 22…”) failed with `SynthesisError` / `KeyError '4'` and `'2'`: recorded digit/text-format failures, not missing files or permissions. The existing prompt preserves word-form numbers; punctuation normalization does not expand digits.
Remaining historical evidence: independent text/scene and timing-policy review, and retry/billing reconciliation; not prerequisites blocking the next text-only trial. Any later audio continuation for word-form replacements needs separate authorization and measurement, preserving failed attempts and the closed deadline; no automatic full rerun.
Original reports, pinned helper/protocol/state, production exports, scoped user confirmations 19/1025 and old production counts (**534 accepted finals unmeasured; 544 unresolved**) remain unchanged. The saved-report copy/accounting update made no inference, TTS or network calls and repeated no audio checks.

**Earlier three-group native Sol-only checkpoint (3 Oct, historical real model/TTS run):** [Standalone HTML](../output/sol61_tool_loop_20261003/report_01.html), [JSON](../output/sol61_tool_loop_20261003/report_01.json), [dispatch evidence and review notes](../output/sol61_tool_loop_20261003/dispatch_evidence.json).
The parent selected native `sidekick`; saved native API metadata confirms `openai/gpt-6.1-sol#medium` for all 14 assistant responses, not an independent upstream provider receipt or physical POST count.
The worker attests reading all 1,523 original cues across four file reads, then rewriting only groups `772–773`, `1192–1193`, `1249–1250`; not a full-episode rewrite or proof of one full-context API prompt.
All three first candidates fit their unchanged windows with the 200 ms diagnostic reserve at voice 8/rate 1.5: **2.008526 / 4.040272 / 3.157914 s**, reserved margins **458.474 / 782.728 / 599.086 ms**.
Three actual WAVs and three attempts of at most nine; no failures or further iterations: **the measured-overrun → repair branch was not exercised**.
Semantics are same-model self-assessment only; no independent arbiter or scene review. Status is `ready_inspection_only`, not production approval; reserve fit does not verify an inserted pause.
Saved [native usage](../output/sol61_tool_loop_20261003/usage_evidence_20261003.json) now confirms input **96,559**, output **2,228**, reasoning **311**, cache read **865,280**, cache write **0**; billed USD and physical POST count remain unknown, despite native reported cost 0.
The offline [conditional API equivalent](../output/sol61_tool_loop_20261003/cost_estimate_20261003.json) is **$0.301926–$0.305036** using public OpenRouter `openai` rates checked 3 Oct: assumes native input is uncached and output either includes reasoning or requires adding it; normalization and native/OpenRouter cache equivalence are unverified, not an invoice or a full-episode price.
Under that assumption cache read is **89.961%**; the largest of 14 per-request prompts is **94,801**, below the 272,000 long-prompt threshold (the 961,839 session sum is not a request). No-cache **$1.945958–$1.949068** is only a same-token-trajectory counterfactual, not a paid run, controlled A/B saving or forecast for 1,523 rewrites; local TTS CPU cost is separate and unmeasured.
This is a local prototype, not an integrated OpenRouter tool transport. Original reports/protocol snapshots, scoped user confirmations for groups 19/1025 and production counts (**534 accepted finals unmeasured; 544 unresolved**) remain unchanged.
At that checkpoint, next criteria were independent text/scene review, measured-overrun → repair evidence and billing/route reconciliation; the later full-source trial above supplies repair evidence, not independent semantic approval or an invoice. This saved-cost calculation made no new model, TTS or network calls; historical report_01 accounting remains unchanged.

**Production baseline verdict (30 Sep, 15:19:23 +03:00):** The independent Australian
Survivor episode run launched at 11:00:08 finished with exit 0 after 4 h 19 min.
Its [real final HTML](../output/openrouter_episode_20260930_clean/shortening.html),
[report](../output/openrouter_episode_20260930_clean/report.json), SRT and JSON
are saved under `output/openrouter_episode_20260930_clean/`. All 343 requests
completed; reported cost is **$4.34939095206**, unknown reserve $0 under its
separate $10 guard. All 1,523 source IDs and timings are preserved: 1,219 were
selected, 813 sent through generation/review, 551 have accepted changed text
(67.8% of 813), 124 original texts measured fitting, and 544/1,219 (44.6%)
remain unresolved: 282 `short_timing` and 262 `manual`. The 15% manual-work
target is **not met**. This is the first technically completed full-file run,
**not** three completed SH-08 pilots or a semantic/audio-quality pass: 12 of
551 accepted finals had the first bounded local Vosk TTS timing check (all 12 fit).
The alignment pilot below adds three distinct accepted-neighbor finals (all
three fit individually); the textual grouping pilot adds two more distinct
accepted finals with measured audio. **534 accepted finals remain unmeasured**.
The 304 unselected cues fit by prediction
only. Preserve the old recovery's independent receipts; its
unknown reservations do not enter this clean run's accounting.

**Bounded final-audio check (1 Oct):** [Measured JSON](../output/accepted_tts_check_20261001/accepted_timing.json)
and [24 WAVs](../output/accepted_tts_check_20261001/wav/) cover 12 matched
accepted corrections (six initial, four after repair, two extreme reductions;
IDs and per-cue durations are in the JSON). At voice 8, rate 1.5, normalized
Vosk text and the frozen available windows (including the configured 200 ms
gap), **12/12 finals fit**, versus **2/12 originals**; the smallest final
margin is 41 ms. Shortening removed the measured overrun in 10 cases. Two
originals already fit: conservative prediction can request an unnecessary
shortening, but this small nonrepresentative sample does not establish a
global calibration defect. WAV durations were measured, not listened to;
semantic quality was not checked. Do not generalize to all 551 finals:
full accepted-cue audio verification remains open. No Vosk-only HTML was
published.

**Quality and cost audit (30 Sep; read-only, no inference):** A purposive
18-cue sample (12 accepted, six unresolved) is not representative or a defect
rate. In particular, accepted `cue-400` (“Вместе?”) may omit the explicit
cooperation commitment, while `cue-1516` (“Желаю удачи с идолом”) may omit the
reported-past “I said before leaving”; inspect original **and final** neighbor
context before any approval. No subtitle was edited. Saved call costs:
generation $0.08369, review $2.85235, repair $0.03780, recheck $1.37555;
arbiter review/recheck account for 97.2% of spend. Repair/recheck cost
$1.41335 (32.5%) and added 162 accepted cues beyond the first 389. This
supports testing cheaper arbiter settings, not blindly removing the quality
gate; neither cost nor speed improvements are yet demonstrated.

**Historical confirmations and pending review (3 Oct, not the next text-only task):** unchanged joined
`cue-1025`–`cue-1027` and rewritten `cue-19`–`cue-20` now have scoped user confirmation.
These different local modes do not establish a representative quality rate or authorize integration.
Review the full-source trial's 527 timing/self-assessment-ready units and 967 unresolved before audio continuation or integration, not as a blocker for the next text-only trial. Real overrun repair and native usage are now evidenced above; independent text/scene approval, actual billing and native/OpenRouter cache equivalence remain open. The earlier three-group and 2 Oct audio-first results stay historical, not a cheaper/full-rollout conclusion.
No time borrowing, cue remapping or production integration was performed.
Do not map joined audio/text automatically to old cues. Internal alignment
and the prior 500 ms per-cue start-shift constraint are unknown for continuous
TTS. Production exports and 544-unresolved accounting stay frozen.

**Audio-first checkpoint (2 Oct, real model/TTS run):** [Standalone HTML](../output/group_audio_first_20261002/report.html), [JSON](../output/group_audio_first_20261002/report.json). Isolated prompts make old character thresholds advisory; Sol checks semantics only, with previous judgments supplied only to Muse.
All three proposals exceeded old predictors (Craig 46/23, immunity 100/63, Kat 59/45) and received Sol `accept` with empty issues; all three were physically synthesized at unchanged voice 8/rate 1.5.
Craig: 2658.685 ms, raw margin **+8.315 ms**, reserved margin **−191.685 ms**; experimental raw-fit accepted text needs reserve-policy and human scene/speaker/pause review, not production approval.
Immunity: 6188.118 ms, margins **−1165.118/−1365.118 ms**; Kat: 4899.410 ms, margins **−942.410/−1142.410 ms**. Both are semantically model-accepted but overlong, with no accepted final; saved WAVs remain inspection-only.
Six POSTs, three new WAVs, no retries, **$0.013315401** reported, no unknown charges; all-six monetary ceiling **$0.3532260** under $1, 120.688 s. Above-old-limit mocked TTS, strict parser/prompt/budget/escaping, compile, real-WAV/receipt and **747 protected-file hash** checks passed; logs `/tmp/opencode/group_audio_first_{checks_,live_}20261002.log`.
Craig has only 50/51 ms neighboring gaps; existing previous/next TTS already overflow by 101.018/728.247 ms, so no outside time is borrowed. The old immunity baseline is saved-current, not full-original audio; historical reports and confirmations for groups 19/1025 remain unchanged.

**Two-repair checkpoint (2 Oct, real model run):** [Standalone HTML](../output/group_repair2_20261002/report.html), [JSON](../output/group_repair2_20261002/report.json); exact previous proposals/feedback were supplied only to Muse, not the independent Sol review.
`1192–1193`: new 62/63-character proposal received `revise`: “вне голосования” is not immunity from elimination, and “все остаются” can expand the addressed group.
`1249–1250`: new 44/45-character proposal retained the personal best-choice framing but received `revise` because “ход против Кэт” does not explicitly preserve voting against her.
Four physical POSTs, zero retries, **$0.022470624** reported, no unknown charges, 63.480 s; 4096 output tokens were sent per call, with a separate 4916-token monetary reserve; all-four maximum **$0.2105008** under $1.
No accepted final or new TTS; new duration/margins remain unmeasured. Strict parsing, feedback isolation, budget/escaping, compile, receipt checks and **734 protected-file hashes** passed; logs `/tmp/opencode/group_repair2_{checks_,live_}20261002.log`.
Old reports/scenes and the user-confirmed groups 19/1025 remain unchanged; no scene listening, second repair or production mapping was performed.

**Remaining-four checkpoint (1 Oct, real model run):** [Standalone HTML](../output/group_batch4_shortening_20261001/report.html), [JSON evidence](../output/group_batch4_shortening_20261001/report.json).
`208–209`: Muse ended at the 2048-token cap (`length`, no content); `772–773`: `cannot_shorten` at 23 characters. Neither received a judge or TTS; the first failure is not proof of impossibility.
`1192–1193`: 63/63-character candidate lost explicit elimination/no-one-goes-home meaning; `1249–1250`: 44/45-character candidate lost the speaker-specific best-choice claim. Both Sol verdicts were `revise`; no accepted final or TTS.
Six physical POSTs, no retries/repair, **$0.031781871** reported, no unknown charges; all-eight preflight maximum **$0.3716069** under $1; actual run 112.420 s.
Muse reported 2051 and 2077 completion tokens for the latter two generations despite the sent 2048 cap; literal receipts are preserved and each reported dollar charge remained below its admission ceiling.
Strict parser/payload/budget/escaping, compile, saved-artifact/receipt checks and **545 protected-file hashes** passed; logs `/tmp/opencode/group_batch4_{checks_,live_}20261001.log`. No new WAV, so duration/margins are unmeasured.
All four full originals and original ±3-cue context were frozen before calls, with saved current text separate; the old `1192–1193` audio is saved-current baseline, not full-original audio. Groups 19 and 1025 keep their prior scoped human confirmations.

**Group 19–20 whole-unit checkpoint (1 Oct, real model run):**
[Standalone real report](../output/group19_shortening_20261001/report.html)
and [machine evidence](../output/group19_shortening_20261001/report.json) retain
the original 79-character normalized whole phrase, original ±3-cue context,
saved current context, exact payloads/raw receipts and source cue provenance.
One Muse-low generation proposed “Но главный альянс племени выгнал главную
угрозу Бьянку.” (55/60 normalized characters); one Sol-high review returned
`accept`, empty issues/advice. No repair or hand-edited wording was used.
One [real WAV](../output/group19_shortening_20261001/tts/wav/19_utterance-19-20_original.wav)
at voice 8/rate 1.5 is 80,128 frames / 22,050 Hz = **3.633923 s**, versus the
saved unchanged original's 5.398639 s. Margins are **1.224077 s** against the
4.858 s raw span and **1.024077 s** against the 4.658 s diagnostic budget;
this does not prove a restored 200 ms internal pause or per-cue alignment.
Two physical POSTs, zero retries, **$0.004706361 reported**, no unknown costs;
live-catalog/actual-token-bound maximums totaled $0.091599 under the $1 guard.
Temporary harness checks covered strict parsing, payload/rubric, escaping and
budget; compile, saved-artifact/full-WAV checks and all 411 protected-file
hashes passed. Logs: `/tmp/opencode/group19_shortening_{live_,checks_}20261001.log`.
Current human status: `user_confirmed_scene_and_meaning_check`; [append-only review](../output/group19_scene_20261001/human_review.json) records the exact questions/answers, including qualified “В принципе сохранился смысл.”, one narrator and timely end before the following line.
[Original-audio scene](../output/group19_scene_20261001/source_0030.000_0105.000_original_audio.mp4), [TTS-only comparison](../output/group19_scene_20261001/context_group19_tts_inspect_only_original_muted.mp4) and [comparison provenance/checks](../output/group19_scene_20261001/comparison_provenance.json) remain hash-bound to that review; model JSON/HTML and comparison provenance keep their historical pending status unchanged.
The existing 3.633923 s candidate is placed at local 13.5–17.133923 s (source 43.5–47.133923 s), anchored to the midpoint of the visual 13.4–13.6 s subtitle-start bracket, not verified speech onset; the entire original track is omitted from the TTS clip, with silence outside unchanged TTS.
The user confirmed those three scene criteria; no repeated review is pending for this candidate, but perfect translation, other scenes and general naturalness are not inferred.
The combined cue-20/21 subtitle still provides no separate measured speech boundary; the user's timely-end confirmation is perceptual evidence, not word alignment or proof of the hypothetical 48.756 s next-cue projection/global offset.
Both 35 s clips have 875 frames and identical compressed video; exact TTS PCM/silence, new-clip decoding and unchanged source/provenance/model-report/TTS hashes passed. No new download, rewrite, synthesis or provider call accompanied assembly.
No assistant listening, new execution, production approval, SRT replacement or automatic mapping; the completed episode, earlier pilots, 544-unresolved accounting and user-confirmed group 1025 stay intact.

**Scene-check checkpoint (1 Oct; user-confirmed for this local scene only):**
One further authorized public VK download produced [30 s original-audio context](../output/scene_source_wide_20261001/source_4743.000_4813.000_original_audio.mp4), 00:47:43–00:48:13, titled `SurvAU (2016) e03`; this is not the full episode.
[Coarse frames](../output/scene_source_wide_20261001/coarse_02.jpg), five boundary sheets and [provenance/checks](../output/scene_source_wide_20261001/provenance.json) locate the whole group visually: start 00:47:51.600–51.800, end 00:47:58.600–58.800; next cue starts 00:47:59.800–00:48:00.000.
Seven burned-subtitle boundary brackets share a source-minus-workspace shift interval of −7.168 to −7.112 s; −7.140 s was chosen only for this local inspection, not a global release/SRT correction.
The logged accurate-seek/reencode command uses `-ss 2863 -t 30`; zero local PTS and [overlap frames](../output/scene_source_wide_20261001/overlap_seek_audit.jpg) support the mapping, with 40 ms frame uncertainty and no independent absolute-time proof.
The unchanged 6.350658 s saved-current joined TTS starts near 00:47:51.749 and ends 00:47:58.100; the visual group span is 6.8–7.2 s, leaving at least 449 ms duration margin (369 ms allowing two 40 ms boundary uncertainties).
[TTS-only inspection video](../output/scene_source_wide_20261001/context_tts_inspect_only_subtitle_estimate_original_muted.mp4) omits original audio entirely and uses silence outside the existing TTS; no new synthesis, speedup, trim or internal-pause edits.
Both clips are 30 s with 750 H.264 640×360 frames; complete decoding, identical compressed-video hashes, exact TTS PCM/silence and prior-source/TTS hash checks passed. Log: `/tmp/opencode/scene_download_wide_20261001.log`.
Burned-text boundaries are not measured speech boundaries; the 224/51 ms workspace subtitle gaps are not actual speech pauses. Original audio language remains `und`.
The user watched/listened to both exact clips and confirmed one host speaker, no reported meaningful pause/interruption issue and overall scene fit; [append-only human review with exact quotes and hashes](../output/scene_source_wide_20261001/human_review.json) records `user_confirmed_scene_check`. This is not full translation/naturalness review, approval of other groups or automatic integration; the −7.140 s shift is local only. No new rewrite/synthesis, model/paid calls or production changes; 544-unresolved accounting and prior artifacts stay frozen.

**Textual continuation checkpoint (1 Oct):** the preserved arbitrary-neighbor
baseline was 1/6 timing fits. The new [candidate JSON](../output/grouping_pilot_20261001/candidates.json)
scans all 1,523 cues / 544 unresolved anchors and 1,813 bounded 2–3-cue windows:
25 disjoint source-closed heuristic candidates cover 28 anchors; 120 windows
are ambiguous/incomplete, 1,668 unsafe, and 516 anchors are outside candidates.
Unknown speakers always require review; ellipses, replies, internal sentence
ends and open/longer-than-three-cue chains are not automatic joins. `cue-1519`
remains ambiguous. Original and saved current joins/provenance are both retained.
One preselected six-group run measured 13 real WAVs: **0 timing-only fits,
six duration deficits (626–2665 ms)** under separate clips + 200 ms minimum
silence; no group WAV/SRT was forced in that separate-clip pilot.
Six [offline group-rewrite inputs](../output/grouping_pilot_20261001/group_shortening_inputs.json)
remain an immutable separate-clip snapshot, not generated text or proof that
all six need rewriting. One additional [joined check](../output/grouping_joined_check_20261001/joined_results.json)
synthesized the same exact measurement texts once as six continuous WAVs,
voice 8/rate 1.5: **1/6 raw-span fits and 1/6 diagnostic reserved-budget fits**.
`cue-1025`–`cue-1027`: 7762 ms separate clips + minimum pauses versus
6350.658 ms continuous audio in 7023 ms, margin 672.342 ms; a 400 ms time
reserve leaves 272.342 ms. No internal silence was inserted or identified,
and fitting that reserve does not prove pause preservation or internal cue
timing. The other five joined clips overrun by 329.200–2173.719 ms.
All 13 old individual clips and the candidate JSON stayed byte-identical;
the two phases total **19 physical synthesized clips**, with no text edits.
Forty-seven affected tests, compile and diff checks passed; real execution
checked all six WAV frame payloads and replayed the prior source/selection
guard before and after synthesis. Logs: `/tmp/opencode/grouping_joined_check_20261001.log`
and `/tmp/opencode/grouping_joined_checks_20261001.log`. Zero LLM/network calls,
provider cost or SRT export; no listening, speaker detection, semantic approval
or HTML. The earlier 75-test/source-preservation evidence remains valid.

**Bounded alignment checkpoint (1 Oct, local real TTS only):**
[`timing_proposal.json`](../output/alignment_pilot_20261001/timing_proposal.json)
retains every selected source timestamp, exact saved final/original text,
per-cue WAV measurement, baseline window and failed proposal. The selection
was fixed before synthesis: divide the 282 chronological `short_timing`
anchors into six equal-rank bands; choose the shortest source window per band
(position breaks ties), alternating anchor+next pairs and
previous+anchor+next triples. Triple bands first prefer an anchor with an
already accepted adjacent cue, to include saved shortened neighbor texts.
Requested neighbors must exist and positions must not overlap earlier groups;
boundary-invalid groups would
remain failures, not silently disappear. This purposive sample emphasizes
tiny windows and different neighbor loads, not a representative success rate.

The six disjoint groups contain **15 cues / nine short-timing cues** and use
voice 8, rate 1.5 and the existing normalized Vosk text/profile identity.
There were **12 fresh physical WAVs**, with exact normalized-text reuse for
four identical “Да.” cues inside this one fixed-settings run; no earlier cache
was used. Only four of 15 cues fit individually, both in their raw source
windows and the existing enlarged windows. Baseline comparison below caps
enlarged per-cue windows at the same raw group envelope: no next-external-gap
borrowing, no summing overlapping budgets and no extra 200 ms overhead.
Durations are rounded **up** to whole milliseconds for planning; JSON retains
the exact frame durations.

| Group / cue IDs | Measured cue durations, ceil ms | Individual fits | Raw envelope ms | Speech + required pauses ms | Result / failed constraint |
| --- | --- | --- | --- | --- | --- |
| 1 / 185–186 | 674, 2485 | 0/2 | 2214 | 3359 | Fail: 1145 ms envelope overrun; 643 ms start shift |
| 2 / 252–254 | 500, 500, 1417 | 1/3 | 2854 | 2817 | Fail: total fits, but third start requires 964 ms shift |
| 3 / 606–607 | 500, 2137 | 0/2 | 1563 | 2837 | Fail: 1274 ms envelope overrun; shift 459 ms is within cap |
| 4 / 795–797 | 802, 767, 1498 | 1/3 | 3220 | 3467 | Fail: 247 ms envelope overrun; 816 ms start shift |
| 5 / 1043–1044 | 778, 1487 | 0/2 | 1081 | 2465 | Fail: 1384 ms envelope overrun; 694 ms start shift |
| 6 / 1346–1348 | 1556, 500, 2764 | 2/3 | 5463 | 5254 | Inspect-only fit: max shift 160 ms; rescues cue-1347 |

The earliest-placement helper allows at most three cues, absolute start shift
at most 500 ms, no overlaps/order changes and no speech outside the raw outer
source envelope. It rejects source overlaps (including external neighbors),
gaps above 500 ms and known speaker/dialogue boundaries. There are no explicit
speaker identities in this selected source: gender is not identity, and no
speaker or scene detection was run. The experiment does not establish that
the grouped cues belong to one speaker or one scene.

Inter-cue silence is at least 200 ms. Longer source gaps are retained first;
only gaps at most 500 ms may explicitly reduce to 200 ms if otherwise needed.
No selected group needed that reduction. In group 6, the 37 ms source gap is
increased to 200 ms and the longer 234 ms gap is retained. Its start shifts
are 0, −160 and +17 ms. Planning conservation is exact:
**4820 ms speech reservation + 434 ms pauses + 209 ms unused tail = 5463 ms**.
The complete speech frames, including native TTS silence, are copied unchanged.
The [assembled WAV](../output/alignment_pilot_20261001/groups/group-6_inspect_only.wav)
contains 120,459 frames at 22,050 Hz (5.462993 s), including 106,240 unchanged
speech frames and 14,219 explicitly zero-valued frames. Physical inter-cue
silences are 4,416 and 5,177 frames (200.272 and 234.785 ms); the remaining
4,626 zero frames are trailing silence. No truncation, resampling, mixing or
forced speedup was used. Failed groups have no shifted exports; their
[individual WAVs](../output/alignment_pilot_20261001/individual/wav/) remain available.
The [shifted SRT](../output/alignment_pilot_20261001/feasible_inspect_only_subset.srt)
is explicitly a three-cue **inspect-only subset**, not a replacement episode.

Actual checks: 17 new focused tests and 33 existing audio/calibration tests
passed (**50 total**); module compile and diff whitespace checks passed.
`utils.alignment_pilot --verify-saved` replayed all six allocations without
inference, checked unchanged hashes for all **370 frozen source files**,
unchanged speech bytes, zero silence, exact physical frame conservation and
subset text/timing/SRT agreement. Logs:
`/tmp/opencode/alignment_pilot_20261001.log` and
`/tmp/opencode/alignment_pilot_checks_20261001.log`.
One real pilot was executed; **zero network/LLM calls, $0 provider cost**.
Audio was measured, not listened to; no semantic approval or Vosk-only HTML.
One of nine sampled short-timing cues has a timing-only proposal; five of six
groups remain unresolved. Do not extrapolate to all 282 short-timing cases.

**Related completed arbiter comparison (1 Oct):** review the bounded eight-case MiMo
arbiter comparison and the separate TTS sample, not another full-episode run.
The OmniRoute route `opencode-go/mimo-v2.6-pro` was listed by authenticated
`GET /models` at `http://127.0.0.1:20128/v1`. Live catalog metadata and
[pinned OmniRoute source](../output/mimo_arbiter_check_20261001/identity_verification.json)
establish that this specific route reports backend model `mimo-v2.6-pro`;
this does not prove upstream account selection, fallback policy or price.
The original first response was validated offline without another POST, then
only seven untouched cases were sent. The [real final HTML](../output/mimo_arbiter_check_20261001/report.html)
and append-only `calls.jsonl` contain eight physical requests, eight valid
structured judgments and **unknown USD cost**. MiMo returned six accepts, one
revise and one impossible; exact agreement with saved Sol is **3/8**, not an
accuracy measure. It accepted three Sol-rejected candidates (`cue-1519`,
`cue-432`, `cue-818`) and rejected Sol-accepted `cue-1516`. It also accepted
suspect `cue-400`; review the joined utterance, causality/wind and agreement
before considering any substitution. The frozen
[`manifest.json`](../output/mimo_arbiter_check_20261001/manifest.json) contains
four saved Sol accepts (`cue-295`, `cue-11`, `cue-400`, `cue-1516`) and four
rejections (`cue-531`, `cue-1519`, `cue-432`, `cue-818`). The [isolated run guide](shorten_file.md#experimental-mimo-arbiter-check)
records the historical commands; **do not rerun**. Sol is not ground truth,
and its historical batched review may differ from this one-candidate context.
The bounded TTS timing samples are complete, but 534 accepted finals remain
unmeasured. Do not promote MiMo or claim it is cheaper from eight purposively
selected cases with no USD receipt. Parallel windows and cheaper Sol modes
remain deferred; the [completed episode](shorten_file.md#current-fresh-episode-run)
and its subtitles/accounting remain unchanged.

**Historical fresh-run checkpoint (30 Sep, 13:21 +03:00; superseded by final result above):**
Measured snapshot `/tmp/opencode/episode_clean_speed_snapshot.json` from the
saved report and complete journal lines: 1,523 source cues, 1,219 selected;
305 accepted corrections plus 124 originals already fitting. Of 790 still
unresolved, 357 are pending with no model history, 282 are preflight
`short_timing` with no model calls, and 151 were attempted but remain manual
(145 semantic, 5 timing, 1 cannot-shorten). Generation and review each covered
456 cues, not all 1,219. The process was alive at the snapshot: 193 requests
started, 192 completed sequentially, one active and excluded from checkpoint
accounting. Elapsed time was 2 h 14 min since the first request, 2 h 21 min
since launch. Completed requests took 6,773.8 s (about 84% of request-to-request
time), with 1,284.6 s of gaps (about 16%); repairs plus rechecks took 2,148 s,
about 32% of completed request time. At this checkpoint, known reported spend
is $2.23263680706 and unknown reserve $0; the active request is not included.
At the observed pace the snapshot estimated another 1 h 45 min for 357
not-yet-attempted cues, **not** an estimate of eventual acceptance. This
dated projection is historical, not a current running-process instruction.

**Objective:** assess the completed Australian Survivor S01E03 Russian episode
before the other two SH-08 full-file pilots. The single-file Muse-low generator /
Sol-6-high arbiter path, private HTML, ledger, guarded budget and resume, and
review queue are implemented and covered by offline tests. Its technical run
is finished; semantic accuracy and final audio fit across all accepted cues
remain unestablished beyond the 17 measured finals. SH-10
final-context evaluation, full audio verification and the other two full
pilots remain open.

**Historical v2 partial (30 Sep, before fresh restart; not an active gate):** The v2 windowed descendant
now has 264 accepted corrections and 831 unresolved of 1,219 selected in its
saved report. Read-only journal inspection found 227 starts / 225 completions;
only `262e18e40b1941fd856fae408d2dec3e` (Sol review) is an *active*
unknown. The former generation unknown `c966047a268147149ab9976bd511589a`
was superseded by an authorized replacement, not cleared from billing risk.
Known reported USD $1.882997127 + unknown reservations $5.175248 leaves
$2.941754873 under the same $10 guard, not confirmed invoice room. Module
hash differences: none. **Offline next-request admission is blocked:** Sol review
replacement reservation $4.85576 would make a guarded total of $11.914005127;
`BudgetPolicy.admit` returned `blocked_by_budget` /
`reservation_exceeds_limit`, despite current summary status `ok`. The wrapper
is ready, but this *old lineage* cannot advance through this request under its frozen
$10 cap; do not run the documented v2 example expecting live progress. Preserve
the old cap, reservations and existing artifacts. The last log
`/tmp/opencode/episode_v2_live_20260930.log` says interrupted during a
request; the process was not running at inspection, its exit code and why it
stopped are unknown. The old [terminal command guide](shorten_file.md#one-terminal-chunk-historical-blocked-v2-recovery-do-not-run)
is historical, not an instruction for the new run. Never remigrate the old
episode to change logging. This v2 note predates the completed clean run above.
No live requests were made for this wrapper or admission check.

**Preserve this checkpoint:**
[`output/openrouter_episode_20260929_sol6_fresh/`](../output/openrouter_episode_20260929_sol6_fresh/)
contains the real [partial HTML](../output/openrouter_episode_20260929_sol6_fresh/shortening.html),
manifest, requests, calls, exports and **500 generated proposals**. Of 1,523
source cues, 1,219 were selected, 124 measured originals fit; 1,095 are in the
review queue, with **zero arbiter-accepted corrections** in this snapshot.
The journal has 121 starts, 120 completions and one unresolved start
`c966047a268147149ab9976bd511589a`. Accounting from the read-only journal
is $0.073219707 known reported USD and $0.319488 retained unknown-cost
reservation. The older HTML checkpoint shows only 120 attempts and $0.212992
unknown reservation; it is **stale**, not the current accounting. These are
reservations, not confirmed charges. The older recovery run has separate,
potentially overlapping receipts: never sum its spend into this run.

**Migration checkpoint (30 Sep, offline only):**
[`output/openrouter_episode_20260930_windowed_recovery_v2/`](../output/openrouter_episode_20260930_windowed_recovery_v2/)
is a separate windowed descendant of the frozen source. It retains the original
121 starts (120 completed, one unresolved), receipts, 500 proposals, source
and post-preflight baseline, and the small preflight timing report. No requests
or audio synthesis were made by the migration. At that point its partial HTML
showed no arbiter verdicts or accepted corrections and $0.073219707 reported
with $0.319488 reserved; its [archived later partial HTML](../output/openrouter_episode_20260930_windowed_recovery_v2/archived_reports/20260930T105949364412504+0300/shortening.html)
was updated by later live continuation. The original directory remains frozen and its
old manifest still cannot be resumed. The earlier offline descendant
`output/openrouter_episode_20260930_windowed_recovery` is superseded: its
manifest predates inherited-byte binding; do not resume or modify it.
At migration time, the v2 descendant could not continue without the explicit
one-time retry for `c966047a268147149ab9976bd511589a`. That retry was later
authorized and superseded; it is **not** the current unknown (see above).
Its reservation does not disappear. The $10 cap is for the entire lineage, not
a new $10. This migration phase made no paid calls; subsequent live progress
and the obsolete conditional command are recorded in the [run guide](shorten_file.md#one-terminal-chunk-historical-blocked-v2-recovery-do-not-run).

**Implemented for new windowed contracts, including migrated v2:** up-to-12-cue
generation/review windows, graceful signal/deadline stop before the next request, at most one
explicit unknown-start replacement per logical request, retained unknown
reservation, and a finite physical attempt cap (default 1,000) with the same
$10 monetary guard. The clean episode's final result is recorded above.
Offline saved-state review building changed from 9.01 s to 0.67 s with an
identical canonical state; that does not prove a live speedup. Existing offline
verification logs record 798 passed / 5 skipped (full suite), 110 passed /
5 skipped (review performance group), and 143 passed (cancellation guards).
No paid model, network, or TTS calls are needed to resume engineering work.

**Keep this the one active plan:** after meaningful work, update this checkpoint
and append one short dated history entry with what changed, what was actually
checked, what remains blocked, and links to the relevant command guide or real
artifact. Leave older dated evidence intact; do not create a parallel active
plan. The historical experiment memory is [separate](shortening_memory.md).

### Compact dated history

- **3 Oct, saved native usage/cost:** [14-response usage](../output/sol61_tool_loop_20261003/usage_evidence_20261003.json) supports the [conditional $0.301926–$0.305036 API equivalent](../output/sol61_tool_loop_20261003/cost_estimate_20261003.json); Decimal fixtures, column/step sums, per-request threshold and source-hash/JSON checks passed, with no new inference/TTS/network calls. Invoice, independent scene review and real overrun repair remain open; historical reports are untouched.
- **15 Sep:** bounded Muse-low/high comparison and zero-capacity Vosk checks
  recorded in [SH-08 evidence](sh08_muse_reasoning_20260915.md) and
  [timing notes](sh08_zero_timing_20260915.md); three full pilots not closed.
- **20 Sep recovery:** older real [partial report](../output/openrouter_episode_20260920_recovery/shortening.html)
  has 397 accepted and 698 unresolved of 1,219 selected; costs and frozen
  contract are separate from the fresh Sol-6 run.
- **29 Sep:** fresh episode interrupted during generation; saved 500 proposals,
  no review verdicts, one outstanding request start. Offline work added
  windowing, bounded stop/retry policy and faster identical review-state build;
  no second live run verified these changes. See [the run guide](shorten_file.md)
  and the real [partial report](../output/openrouter_episode_20260929_sol6_fresh/shortening.html).
- **30 Sep:** refreshed this checkpoint for a safe handoff and corrected the
  [experiment memory](shortening_memory.md) pointer to Sol 6 high. Documentation
  and commit-candidate inspection only; no new live run, TTS or inference.
- **30 Sep, offline migration:** verified the generation-only Sol6 source,
  reconciled the stale 120-attempt report against 121 immutable request starts,
  and published the [real inherited partial HTML](../output/openrouter_episode_20260930_windowed_recovery/shortening.html)
  in a separate run. The frozen source was byte-hash checked before/after;
  no live calls. Offline migration/replay and related checks passed (see the
  [run guide](shorten_file.md#resume)). This first descendant is superseded;
  its manifest lacks the inherited-byte guard. No live continuation happened.
- **30 Sep, guarded v2 (offline):** inherited request/receipt byte prefixes and
  baseline/preflight evidence are bound in the frozen contract and checked before
  retry intent or transport. The source report is pinned to the verified saved
  byte hash, including unprocessed rows and summaries. That offline checkpoint
  was generation-only; the linked
  [archived v2 partial HTML](../output/openrouter_episode_20260930_windowed_recovery_v2/archived_reports/20260930T105949364412504+0300/shortening.html)
  includes later live continuation with 264 accepted corrections. The earlier
  descendant is preserved, not rewritten.
  Eight self-contained migration checks and the broader offline group passed
  (306 total); `compileall` and diff whitespace checks passed. SHA-256 for all
  369 original output files matched before/after. The real v2 CLI preparation
  succeeded offline; a guarded runner invocation stopped on the expected unknown
  ID before transport. [The run guide](shorten_file.md#resume) has the conditional
  command. No paid, network, or TTS calls were made in this work.
- **30 Sep, terminal handoff:** inspected the later *real* v2 partial and
  frozen module hashes offline; added a standalone terminal tee/heartbeat and
  bounded soft-stop wrapper without changing production runtime files or run
  artifacts. Focused child-process tests cover output, exit, no-progress
  warnings, journal tail/supersession and first signal. The [old v2 run guide](shorten_file.md#one-terminal-chunk-historical-blocked-v2-recovery-do-not-run)
  has separate first-retry and later-resume examples. Offline `BudgetPolicy.admit`
  for the current Sol review replacement rejects the $4.85576 reservation
  against $2.941754873 remaining; projected guarded total $11.914005127.
  This blocks the examples under the frozen cap. No inference was launched.
- **30 Sep, fresh restart:** on explicit user approval, archived only the old
  v2 presentation pair (HTML and sidecar), leaving its manifest, journals,
  receipts and proposals untouched. Launched one independent, supervised
  [whole-episode run](shorten_file.md#current-fresh-episode-run), new $10 cap,
  1,000-attempt physical limit, one configured recovery round and 21,600-second
  soft-stop ceiling. Initial manifest and audio preflight appeared; process
  was alive at startup inspection. No result, final costs or new HTML were
  available yet. Old v2 recovery remains deferred, not a gate for this run.
- **30 Sep, 13:21 +03:00:** read-only fresh-run snapshot recorded above:
  305 accepted, 124 original fits, 357 untouched pending, 282 short-timing
  without model calls, 151 attempted manual. Measured request/gap timing and
  checkpoint budget; no inference initiated for this documentation, restart
  or runtime changes. Optimize only after this run completes and benchmark
  quality as well as speed.
- **30 Sep, 15:19 +03:00:** independent clean full-file run finished, exit 0;
  551 accepted, 544 unresolved, 343 completed calls and $4.34939095206
  reported. [Final HTML](../output/openrouter_episode_20260930_clean/shortening.html)
  exists. A subsequent read-only 18-cue audit identified possible context
  losses, not a measured error rate. Final-audio and SH-10 checks remain open;
  no inference or TTS was run for this documentation update.
- **1 Oct, isolated MiMo preparation (offline):** added
  `utils/mimo_arbiter_check.py` and self-contained focused tests (12 passed).
  Froze eight exact historical candidate/Sol-review pairs in
  `output/mimo_arbiter_check_20261001/manifest.json`; the source episode and
  primary Muse-low/Sol-high runtime are unchanged. The configured OmniRoute
  endpoint/model are known from local configuration, but the earlier
   authenticated `/models` request returned 401. At preparation time no later
   authentication probe, model call, MiMo verdict, reported cost or real MiMo
   HTML was produced.
  Wait for restored access and verify identity before any charged request;
  the separate bounded TTS timing check is recorded below. Parallelism and
  cheaper Sol modes remain deferred.
- **1 Oct, bounded accepted-final TTS:** synthesized and measured 12 accepted
  finals and their 12 originals locally (24 WAVs), voice 8/rate 1.5 with the
  frozen 200 ms gap policy. All 12 finals fit their windows, two originals fit;
  minimum final margin 41 ms. Sample IDs: `cue-3`, `cue-11`, `cue-293`,
  `cue-534`, `cue-636`, `cue-645`, `cue-1012`, `cue-1046`, `cue-1289`,
  `cue-1458`, `cue-1516`, `cue-1522`. [Per-cue measurements](../output/accepted_tts_check_20261001/accepted_timing.json)
  and [WAVs](../output/accepted_tts_check_20261001/wav/) preserve exact IDs and
  durations. This nonrepresentative 6-initial/4-repair/2-extreme sample has no
  listening or semantic review; the remaining 539 accepted finals are not
  measured. No model-provider call or Vosk-only HTML was produced.
- **1 Oct, renewed OmniRoute access check after OpenChamber restart:** a single authenticated
  `GET http://127.0.0.1:20128/v1/models` using the configured provider
  credential again returned **401**. The exact configured remote ID
  `opencode-go/mimo-v2.6-pro` was therefore not confirmed against the live
  model list. The eight-case manifest remains unchanged; zero charged MiMo
  requests, verdicts, costs or real MiMo HTML exist. Restore access locally
  before the first bounded review call; do not switch routes or infer that MiMo
  is cheaper or good enough from this blocked check.
- **1 Oct, workspace `.env` credential and partial MiMo trial:** a fresh
  authenticated `/models` request returned 200 and listed the exact configured
  `opencode-go/mimo-v2.6-pro` ID. Before a paid request, corrected an isolated
  prompt-order replay bug in `utils/mimo_arbiter_check.py` without changing the
  frozen manifest; all eight hashes matched and 12 focused tests passed. One
  POST for `cue-295` returned `model: mimo-v2.6-pro`, so the fail-closed runner
  saved the raw response and stopped with zero identity-validated judgments.
  The [real partial report](../output/mimo_arbiter_check_20261001/report.html)
  and receipt show 15.677 s, 832 input / 523 output tokens (444 reasoning
  within output) and **unknown USD cost**, not free. Its content parses offline
  as `accept` like saved Sol, but provider identity has not been independently
  reconciled; the other seven cues were not sent. Never resend `cue-295` to
  repair parsing or accounting; no conclusion about MiMo quality or savings.
- **1 Oct, verified route mapping and seven pending MiMo calls:** live catalog
  metadata for the exact prefixed ID identified `owned_by: opencode-go` and
  `root/name: mimo-v2.6-pro`; pinned OmniRoute source explains provider/model
  parsing and response-model rewriting. The narrow mapping and its limits are
  saved in `identity_verification.json`. Offline validation appended a new
  event for the first saved response without altering its error receipt or
  repeating its POST. `--resume` sent exactly seven pending requests: journal
  totals eight starts/eight completions/one validation, no unknown starts.
  [Final HTML](../output/mimo_arbiter_check_20261001/report.html) shows eight
  judgments, six accepts; 3/8 exact agreement with Sol on this selected set.
  The eight receipts report 6,075 input and 3,143 output tokens (2,289
  reasoning within output), median 12.576 s; all USD costs are unknown.
  Four Sol-rejected cases include three MiMo accepts requiring human review;
  no claim of quality or savings follows from this experiment.
- **1 Oct, bounded local alignment pilot:** added the isolated typed
  `utils/alignment_pilot.py` helper and 17 focused tests; preserved production
  shortening modules and all 370 completed-source files. One actual voice-8,
  rate-1.5 run measured 15 cues in six preselected disjoint groups (12 physical
  WAVs through exact-text reuse). One group rescues `cue-1347` under the unchanged
  500 ms start-shift cap, with physical zero silence and complete speech bytes;
  five fail, including a group with enough total time but an illegal 964 ms
  shift. [Machine evidence](../output/alignment_pilot_20261001/timing_proposal.json),
  [listen-only WAV](../output/alignment_pilot_20261001/groups/group-6_inspect_only.wav)
  and a clearly labeled subset SRT are saved. Fifty focused/reused-helper tests,
  compile, diff checks and no-inference saved-artifact verification passed.
  Zero provider/network calls, $0 provider cost; no HTML, listening or source
  scene/speaker approval. Next: compare this group to its original scene before
  any integration or wider run; do not weaken limits to increase successes.

## Current production direction

The primary objective is user-facing score stability and a simple CLI with
complete cost and result reporting. Optional experiments are parked in the
single [`future experiments backlog`](future_experiments.md), not kept in the
active task list. Existing code, real artifacts, history, and module paths
remain in place; this documentation change does not move or remove anything.

The single-file OpenRouter CLI is implemented with Muse
`meta/muse-spark-1.3-contributor` at low mode as generator and Sol
`openai/gpt-6-sol` at high mode as arbiter. It has a guarded budget, reported
USD and unknown reservations, guarded resume, private HTML, SRT, and a review
queue. This is implementation and offline test evidence, not proof of semantic
or audio quality. SH-10 remains unchecked and unimplemented.

**Historical Sol 6 episode run (29 September 2026):** That earlier $10
run was interrupted before generation finished. The launcher had a
6,720-second SIGINT watchdog. The runtime snapshot was recorded at 19:54:23
+03:00 and the log's `KeyboardInterrupt` at 21:46:24 +03:00 (about 6,720
seconds later). The process was still running after the parent tool had been
interrupted: this timing is consistent with the watchdog, not the earlier tool
interruption, although the traceback alone cannot identify the SIGINT sender.
Its [private partial HTML report](../output/openrouter_episode_20260929_sol6_fresh/shortening.html)
and [SRT](../output/openrouter_episode_20260929_sol6_fresh/subtitles.srt)
are saved. All 1,523 source cues retain their IDs and timestamps; of 1,219
selected cues, 124 measured originals fit, 500 have generated proposals, and
1,095 remain in the review queue. Snapshot statuses: 800 pending, 282
short-timing, 12 error, 1 manual.
The runner schedules the **entire generation pass before any review**: the
500 saved proposals therefore have no arbiter verdict or accepted correction.
There are 120 completed generation requests and one started request without a
completion. The read-only journal accounts for $0.073219707 reported USD,
$0.319488 reserved against three unknown-cost attempts (including the
interrupted start), and 121 physical attempts. These are **not** a known final
invoice. The HTML/report checkpoint predates the last start and shows 120
instead of 121 attempts; its budget is stale. The existing offline publisher
has no journal-accounting notice option, so saved artifacts were not rewritten
to disguise the distinction. Do not automatically retry the unknown request,
resume, alter its frozen contract, or start a replacement run. The user
authorized finite retries in general, but the frozen primary CLI cannot retry
an unknown journal start. Reconcile it before separately approved recovery.
The SH-08 three-pilot requirement and SH-10 remain open.

The **completed clean whole-episode run** used its own $10 cap; 343 paid calls
and $4.34939095206 reported spend are verified in its saved report. Prior
recovery spend is separate, and overlapping
predecessor/recovery receipts must not be counted twice. The earlier approved
cap was **$10 total** for preflight model calls plus one
`Australian Survivor S01E03_ru.srt` episode (1,523 cues, approximately
75 minutes); it is distinct from the Jev trial and old subscriptions. A real
episode run and a separate recovery directory now exist. One technical
full-file run is complete, but quality gates and two other pilots remain.
The preparation source is the exact
[`sh08_pilot_manifest.json`](sh08_pilot_manifest.json) file, whose SHA-256 is
`dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`.
That manifest remains a preparation-only historical record, not a ledger of
the later live execution. Costs for the recovery directory must not be added
to those of its predecessor: receipts can overlap.

**Verified saved state (29 September 2026):**
[`output/openrouter_episode_20260920_recovery/shortening.html`](../output/openrouter_episode_20260920_recovery/shortening.html)
is the existing real partial report. The recovery snapshot has 397 accepted
corrections and 698 unresolved of 1,219 selected targets. Its frozen budget
reports $1.74572717418 known USD and $4.85576 unknown reservation after a
completed transport-error request; there are no outstanding journal starts.
It remains incomplete, not a completed SH-08 pilot or a project-wide spend
total. The offline read-only `utils.shortening_status` diagnostic confirms the
unknown cost and also flags a changed frozen module hash. Reconcile billing
and review the immutable resume contract before any explicitly authorized
continuation; do not rewrite saved manifests or auto-retry failed calls.

The updated policy applies to new windowed contracts, including the separately
migrated v2, but not the frozen historical source. Sol 6 high / Muse low now
schedules windows of up to 12 cues:
generation, review, one bounded repair/recheck, then the next window. The
first SIGINT/SIGTERM or an optional run deadline stops before another request;
an in-flight request has its existing 300-second client timeout and 16,384
output-token cap. Explicit new-contract unknown-start retry is limited to one
replacement, and its old unknown reservation remains under the frozen budget.
The new default permits unknown attempts up to the finite physical request
limit (1,000), subject to the same $10 monetary guard; the old run retains its
three-unknown contract. The clean run finished with this scheduling, but no
finished semantic or final-audio quality result exists. The interrupted run above establishes neither live
semantic quality nor a final invoice.

Saved real request timing (120 completed generation requests): median 26.04 s,
nearest-rank p90 48.58 s, maximum 70.02 s, total 3,439.39 s; 118 successes
and two HTTP errors. Gaps between completed requests have median 23.45 s,
p90 23.96 s and total 2,789.3 s. The manifest-to-first-request interval was
411.27 s. The journal does not attribute these gaps to Vosk, metadata lookups,
or checkpointing. An offline profile of `build_review_state` on the saved
1,523-cue state found 3,046 per-cue `_context` calls taking 42.64 s of a
44.72 s profiled pass. Indexing chronological neighbors once per source and
current snapshot reduced an unprofiled state build from 9.01 s to 0.67 s
on this saved input; the complete canonical state (including fingerprints,
snapshot ID and queue) stayed identical. This local build improvement does
not measure a live request-gap or end-to-end speedup; the old episode and its
stale report remain unchanged.

The SH07 snapshot is partial and verified, with no data deletion; it remains
open rather than closed. The existing SH06 checks and SH08 artifacts remain
historical evidence and are not replaced by this plan. README is absent; no
new mass documentation is planned beyond this required backlog.

## Current dated update (15 September 2026)

The SH08 small comparison first pass is complete: eight pairs over the same 60
unique cases in two repetitions. This is 60×2, not 120 independent cases.
First-pass results and the machine record are in
[`sh08_eightpair_results.md`](sh08_eightpair_results.md) and
[`sh08_eightpair_results.json`](sh08_eightpair_results.json). The 18 valid
zero-capacity cases and 1 invalid source case remain in the full 60-case
denominator. Repairs are partial and non-uniform because the 300-unit
subscription cap was exhausted. This does not complete the SH08 full-pilot
checkbox or SH10.

The bounded actual Muse low/high comparison is now complete over the same 60
cases in two repetitions. Its independent per-case accepted-and-fit summary,
costs, receipt latency, and current duplicate denominators are in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md) and
[`comparison summary`](../output/sh08_muse_reasoning_comparison_20260915/summary.json).
This does not close SH08: three full-file pilots, the full-file Muse adapter, and
a hard budget guard remain blocked. No further model, network, or TTS calls are
authorized by this record.

The offline local-Vosk check of the 18 originals with zero predicted character
capacity is complete: 1 measured fit and 17 overruns. The next safe work is the
offline quality decision before any three full pilots. Do not make additional TTS
or model calls now. New model calls require fresh approval; unused API cash is
not permission to continue.

**Prior final verification (14 September 2026):** The
[`verification log`](/tmp/opencode/sh08-final-verification.log) records
`pytest` with `563 passed, 5 skipped in 48.64s` and exit 0; compileall exited 0;
derived analysis validation matched existing outputs identically, exit 0; 10
JSON files validated and no secrets were found.

The status prose below is retained as a historical snapshot where it describes
the pre-execution plan.

Статус на 14 сентября 2026: SH-00 и SH-01 реализованы и уже присутствовали в
HEAD `ec0cbaf` до этой задачи; SH-02 реализована и полностью проверена и
зафиксирована коммитом `c5bf383`; SH-03 реализована, проверена и зафиксирована
коммитом `9a1ddd9`; по запросу пользователя SH-04 зафиксирована коммитом
`a958052`; SH-05 и SH-11 реализованы, покрыты модульными тестами и прошли
завершённую техническую real offline acceptance в commit `344880b`
([документальная запись](shortening_real_acceptance.md)); continuation inventory
is linked in [SH08 preparation](sh08_matrix_preparation.md).
Следующая задача — SH-08, для неё требуется отдельное разрешение на платный
технический пилот; её чекбокс здесь не отмечен.
Историческая матрица фактических generator→arbiter запусков и неисполненное
предложение новой матрицы вынесены в [короткую сводку](shortening_model_pairs.md)
и [JSON-предложение](sh08_model_matrix_proposal.json); диагностика Vosk — в
[отдельной записи](vosk_timing_diagnostics.md). Это не разрешение на платные
запуски.

## Цель

Технически обработать полный JSON существующей утилитой можно уже сейчас. Но для обычного использования нужны понятная очередь нерешённых случаев, возобновление после сбоев и проверка на полных файлах.

Цель ближайшей работы — три полных файла с воспроизводимым результатом, а не новый большой перебор моделей. Последняя проверка baseline: 175 тестов пройдено за 3.38 секунды, exit 0; также прошли compileall и две чистые metadata-проверки. Новых платных вызовов моделей не было.

Уже есть сохранение промежуточных результатов, журналы запросов и расходов, контекстный аудит, построчная проверка вариантов в режиме C и проверка принадлежности профиля TTS. SH-04 добавляет безопасное возобновление запуска к этим работающим частям, а не переписывает их.

## Приоритетные задачи

### [x] SH-00. Зафиксировать рабочую исходную точку — P0

Выполнено после явного разрешения пользователя: изменения и отсутствие секретов проверены, текущий код, тесты и краткие отчёты сохранены одним связным коммитом `6b973dae412a5b7fada49fd0ad46d413b75d34b0` (`6b973da`) с сообщением `feat(shorten): establish tested experiment baseline`. Коммит содержит 47 файлов кода, тестов и документации; `output`, модели, корпус и авторизация в него не вошли.

**Проверка:** 175 тестов пройдены за 3.38 секунды, exit 0; compileall и две чистые metadata-проверки завершились успешно. Платных вызовов моделей не было. Состояние можно восстановить по указанному SHA.

### [x] SH-01. Общий анализатор и короткие фразы — P0

**Implementation note (completed; historical verification):** The shared
planner/analyzer use 1.5× acceleration and a 200 ms reserve, classify 1–3-word
timing risks and exact-word repetitions, and provide opt-in original Vosk
preflight via `--preflight-model-path` with voice/text-mode options. Actual fits
need no LLM; failed or missing evidence remains unresolved. Fresh offline
verification recorded 215 tests passed and compileall exit 0, using fake
synthesis; no real Vosk or paid calls were made. The note that this verification
was not committed was historical; SH-01 is present in HEAD `ec0cbaf`.

Зависит от SH-00. Основные файлы: `utils/analyze_text.py`, `utils/shortening_plan.py`.

- Использовать один расчёт доступного времени и общие настройки: ускорение до 1.5×, резерв 200 мс. Сейчас старый анализатор использует 1.3 и 300 мс и пропускает часть коротких фраз; новый планировщик отдельно выделяет только фразы до двух слов.
- Для не помещающихся фраз из 1–3 слов добавить явную метку `short_timing_risk`. Короткая фраза, которая помещается, такой метки не получает.
- Считать слова одинаково во всех проверках. Старое поле `is_short_segment`, связанное с длительностью, не менять на другой смысл незаметно.
- «Пошли, пошли, пошли.» распознавать как кандидат на сокращение повторов: `repetition_candidate`. Такой текст можно отправить LLM с контекстом. Сам код слова не удаляет: повтор иногда передаёт усиление.
- «Да.» и «Я не знаю.» при превышении времени направлять в отдельный разбор. «Да, нет, да.» не считать точным повторением одного слова.
- Прогнозируемый риск не означает доказанную невозможность сокращения. Если доступна проверка Vosk, сначала измерить звук: фактическое попадание снимает риск без LLM. Непроверенные случаи не считать решёнными.
- Анализатор не меняет текст и таймкоды; он сохраняет причину, число слов, признаки повторения, доступное окно и источник оценки. Неверные тайминги помечаются отдельно.

**Готово, когда:** анализатор и планировщик согласованы; тесты покрывают перечисленные примеры, неверные тайминги и различие прогноза с фактическим измерением. Новые платные запросы не нужны.

### [x] SH-02. Не терять всю пачку из-за одной плохой строки — P0

Зависит от SH-01. Основной файл: `utils/shortening_runner.py`.

Перенести построчную проверку в обычное создание, проверку и исправление субтитров. Валидные соседние строки сохраняются; отсутствующие и неверные строки получают собственные ошибки. Чужие и неоднозначные идентификаторы не применяются. Повторно обрабатываются только нерешённые строки, в пределах заданного числа попыток.

**Готово, когда:** из шести строк при одной ошибочной сохраняются пять корректных; принятые результаты не генерируются заново; ошибки протокола не выдаются за смысловые отказы. Проверяется без платных запросов.

**Implementation note (SH-02, committed as `c5bf383` and fully verified):**
Ordinary generation, review, repair, and recheck now use bounded per-cue partial
parsing and shared extra recovery rounds. Valid JSON rows with identifiable IDs
are isolated, valid neighbours are checkpointed, and missing or invalid rows get
their own errors. A malformed envelope or repeated raw JSON keys fails the whole
response safely rather than allowing ambiguous application. Protocol failures
are not treated as semantic impossibility. The resilience matrix covers
six-row salvage, persistent malformed generation/review, mixed repair and
recheck rows, protocol misuse, transport failures, per-request usage
completeness, and source/export preservation.

The `max_repairs` value is a shared 0..2 extra-round budget (at most one
generation and one review request per cue in each extra round). Each repair
counts a corrective generation request, including one caused by a protocol
failure; review-only recovery does not spend a generation call. A malformed
judge response rechecks the same candidate. Returned transport failures are
recorded but are not automatically repeated.

Mode-C parsers and selection logic were not modified. Arm A is still evaluated
from the first generation attempt; it may remain unavailable if that attempt
failed while B recovered. The shared request ledger now also records returned
transport failures, so this change can affect failure reporting without changing
mode-C parsing or selection.

**Verification (12 September 2026):** `uv run --active --no-sync -m pytest -q`
reported `257 passed in 3.40s`, exit 0. `uv run --active --no-sync -m
compileall -q utils tests pipeline.py ui/gradio_app.py` exited 0. Both
`utils.opencode_clean --check` modes (with and without `--with-length-tool`)
exited 0 and returned `{"status": "ok"}`. All five CLI `--help` checks
(`shorten_subtitles`, `shortening_experiment`, `shortening_replay`,
`shortening_corpus`, and `vosk_calibration`) exited 0. `git diff HEAD --check`
exited 0. No paid or real model/TTS calls were made.

### [x] SH-03. Обычный запуск для одного файла и профиля — P0

Зависит от SH-01 и SH-02. Планируемая новая точка входа: `utils/shorten_file.py`; экспериментальные команды остаются совместимыми.

- Принимать произвольный полный SRT или JSON, включая SRT с BOM.
- Требовать явно выбранную модель; обычный запуск не включает всю матрицу моделей и два повтора.
- Явно выбирать профиль Vosk и проверять модель, голос, скорость и подготовку текста. Несовпадение останавливает работу до запросов к LLM.
- Использовать один способ подсчёта для планировщика, проверки ответа и инструмента модели. Нельзя незаметно смешивать исходные символы и символы после подготовки текста для TTS; в отчёте сохраняется `length_basis`.
- Использовать готовую калибровку, не обучая её заново на обрабатываемом файле. Старую оценку 13 символов/с разрешать только как явно обозначенный резервный вариант.
- Сохранять таймкоды, индексы JSON и неизвестные поля. Для SRT отдельно документировать его нумерацию.

**Готово, когда:** одна команда даёт новые JSON/SRT и отчёт; предварительный расчёт доступен без сети; несовместимый профиль отклоняется. Подключение в `pipeline.py` не обязательно для первой версии. Платный эксперимент пока не нужен.

**Implementation and verification note (SH-03, 12 September 2026):** The
single-model [`utils.shorten_file`](shorten_file.md) command processes one whole
SRT or JSON file, including UTF-8 BOM input, using one selected model for both
generation and review. Planning, local checks, and the optional length tool use
the normalized length basis, and the frozen Vosk profile must match the model,
voice, rate, and text normalizer. Offline `--dry-run` uses a lazy backend and
does not make LLM, network, Vosk, or synthesis calls; legacy experimental CLI
behavior remains compatible.

Final verification: `uv run --active --no-sync -m pytest -q` reported `316
passed` and compileall for `utils tests pipeline.py ui/gradio_app.py` passed;
new and legacy CLI `--help` checks passed. A real existing Vosk model and frozen
profile passed the offline BOM SRT/JSON smoke check, while invalid voice and
rate values were rejected before output creation. No paid LLM or real synthesis
calls were made. The temporary local log, not a durable artifact, is
`/tmp/opencode/sh03-final-20260912/baseline.log`. SH-03 was committed as
`9a1ddd9`; the next task was SH-04.

### [x] SH-04. Возобновление без повторной оплаты — P0

Зависит от SH-03. Использует существующие промежуточные сохранения.

Продолжать запуск только при совпадении исходника, настроек, профиля и версии инструкций. Завершённые этапы не отправлять моделям заново. Журнал запросов дополнять, не переписывая историю.

Если запрос был отправлен, но ответ не сохранился, его состояние неизвестно. Такой запрос нельзя автоматически считать бесплатным, завершённым или безопасным для повтора. Повтор требует явного решения. Ровно одну обработку на стороне внешнего API гарантировать нельзя.

**Готово, когда:** тест прерывания и продолжения не меняет принятые строки и не повторяет завершённые запросы; изменённый исходник отклоняется; неизвестные запросы видны отдельно. Проверяется на имитации транспорта, без оплаты.

**Implementation and verification note (SH-04):** Resume now validates the
immutable source/configuration/profile/model/prompt contract and actual module
hashes, replays saved receipts locally, and appends request/call history
without resending completed work. Full-file and non-contiguous target plans,
durable complete-before-checkpoint ordering for generation, review, repair, and
recheck, saved calls before completion (unknown costs), explicit retry chains,
source/config/profile/model/prompt mismatches, the nonblocking `filelock`, and
preflight plus zero-target behavior are covered by tests. `BUG-01`
`CancelledError` swallowing is fixed as part of SH-04; cancellation remains an
unknown request requiring explicit retry authorization. Tests are in
`tests/test_shortening_resume.py` and
`tests/test_shortening_resume_integration.py`, including
`test_cancelled_attempt_requires_explicit_retry_and_then_completes`.

Independent final verification reported 407 tests passed. Compileall, the
`shorten_file` CLI help, all three legacy CLI help checks, and `git diff HEAD
--check` also passed. No live API, paid LLM, or TTS calls were made. The
complete captured log is `/tmp/opencode/sh04-final-verification/checks.log`.

The default unknown-request block returns exit 2 and requires an explicit
`--retry-unknown REQUEST_ID` decision; completed receipt errors are replayed as
errors, not resent. Exactly-once external API or billing behavior is not
guaranteed, and known costs remain separate from unresolved billing unknowns.

### [x] SH-05. Удобная очередь ручной проверки — P0

Зависит от SH-02…SH-04. Планируемые результаты: редактируемый `review.json` и читаемый `review.md`.

Включать только нерешённые случаи: оригинал, соседний контекст, кандидаты, лимит и фактическую длину, причину, последнюю критику модели и предлагаемое действие. Отдельно показывать короткие фразы, ошибки таймингов и ошибки протокола.

SH-05 should consume SH-11's canonical outcome markers and queue only cues
whose outcome is unresolved or needs attention.

Команда применения правок принимает только явно одобренные известные `cue_id`, проверяет хеш исходника, формат и бюджет. Устаревшие и превышающие лимит правки не применяются молча. Остальные строки и таймкоды не меняются.

**Готово, когда:** любой нерешённый случай легко найти и исправить без поиска по большим журналам; правки применяются безопасно. Измеряем не только число правок, но и фактическое время пользователя, если оно записано. Новый интерфейс Gradio пока не требуется.

**Implementation note (SH-05, completed alongside SH-11):** Every new run now
writes the all-cue `outcomes.json`/`outcomes.md` snapshot and the unresolved-only
`review.json`/`review.md` queue, bound to checksummed `review_state.json`.
The queue carries original/current text, neighbor context, candidates, budget,
actual length, reason, latest critique, and proposed action. Applying a copied
and explicitly edited queue accepts only known, unique, current-context-bound
`approved=true` `keep`/`replace` entries and writes a new derived export; stale,
invalid, over-budget, immutable, or malformed input fails closed. Apply and
reanalysis are offline and hold a nonblocking run lock. See the concise
[`review and reanalysis guide`](shortening_review.md) and the
[`shorten_file` user guide](shorten_file.md).

Outcome status and `text_changed` are independent. Model or human acceptance is
not measured audio fit; invalid or missing current measurement stays null. A
fresh model verdict refers to original neighbor context and has
`final_context_checked: false`; SH-10 is not implemented. Changed neighbor
windows invalidate reuse and keep attention in the queue. Prediction fits are
estimates only. Legacy budget fields remain compatibility data, while
`report.outcome_summary` is authoritative over legacy stage/verdict counts; the
initial intervention denominator is fixed and later regressions are reported
separately. This records implementation and verification only, not a semantic
guarantee or live pilot.

**Real offline technical acceptance (13 September 2026):** The four-file,
full-source SH-05/SH-11 gate passed without LLM, TTS, network, or paid calls.
See [`shortening_real_acceptance.md`](shortening_real_acceptance.md) for the
durable r12 evidence, frozen hashes, and explicit quality boundaries. This is
technical acceptance, not SH-08 or a semantic-quality result.

### [ ] SH-06. Проверить пользу арбитра при равных условиях — P1

SH-06 remains unchecked and deferred; it does not block the first technical
pilot. The complete H2–H6 branch proposals, historical checks, and evidence
links are consolidated in the [future experiments backlog](future_experiments.md).
Known findings are limited: Jev is not promoted to the main arbiter, Muse high
is not the default, and no full best-of-five run is the default. The recorded
Jev replay/follow-up and the two H6 examples (`cue-618` and `cue-210`) remain
historical or proposal-only evidence; no automatic new tests or calls are
authorized. This short active entry does not discard the unchecked SH-06
status or historical checks.

### [ ] SH-07. Архивирование и безопасная очистка — P1

До любого удаления подготовить раздельное хранение. Планируемые пути:

- `artifacts/results/<runid>` — постоянные результаты;
- `artifacts/corpus/<hash>` — одна сохранённая копия исходных данных;
- `output/runs/<runid>` — временные подробности запуска.

Эта структура одобрена пользователем, но SH-07 не реализована: каталоги не
созданы, миграция, архивирование и удаление не выполнялись. Очистка всегда
начинается с dry-run, затем требует отдельного подтверждения и проверенного
архива. Правило TTL в 30 дней пока только предложение и не является одобренной
политикой. Git хранит код и документацию; исходные субтитры, ответы моделей и большие результаты хранятся отдельно, откат Git не восстановит удалённые игнорируемые файлы.

Частичный снимок и проверка отдельных артефактов записаны в
[`sh08_archive_20260915.md`](sh08_archive_20260915.md); это не миграция и не
разрешение на удаление. Плановые миграция, очистка и перенос не выполнялись.

При реализации большие артефакты исключить из Git. Сначала достаточно сжатия JSON/JSONL без потерь и устранения повторного хранения исходников; сложная база данных не нужна.

Сохранять итоговые субтитры, исходные ответы моделей, варианты, критику, инструкции, параметры, хеши, токены, тарифы и профили. Калибровочные WAV и важные проблемные примеры закреплять для хранения. Именно исходные ответы уже позволили восстановить 19 корректных строк без новой генерации.

Очистка сначала только показывает список файлов и освобождаемый объём. Удаление допускается после явного подтверждения и проверки восстановления архива. Не трогать исходники, модели, настройки и авторизацию OpenCode, цели символических ссылок, активные, незавершённые, неизвестные и закреплённые запуски. Не ломать ссылки других экспериментов.

**Предложение, ещё не включённое:** удалять промежуточные данные старше 30 дней только после проверенного архивирования. Постоянные результаты не удалять по сроку автоматически.

**Готово, когда:** из архива без сети восстанавливаются результаты и исходные ответы; повреждение архива блокирует очистку; повторная очистка безопасна. Старые каталоги сейчас не перемещаются и не удаляются.

### [ ] SH-08. Проверить три полных файла — P1

**Current status (30 September 2026):** One of three technical full-file runs
completed with [final HTML](../output/openrouter_episode_20260930_clean/shortening.html),
but its 44.6% unresolved rate misses the 15% target; final changed-cue audio
and independent semantic checks are still missing. The other two full-file
pilots have not run. This checkbox remains open.

Начинается после SH-00…SH-05; SH-06 не является обязательным предварительным условием.
SH-05/SH-11 уже имеют отдельную real offline technical acceptance, но этот
чекбокс остаётся незакрытым; платный пилот требует отдельного разрешения.
Текущая подготовка также имеет отдельные [модельные факты и proposal]
(shortening_model_pairs.md) и [Vosk timing diagnostics]
(vosk_timing_diagnostics.md); ни одна из этих записей не закрывает SH-08.

#### Current first-pass note (14 September 2026)

The eight-pair first pass is now executed and complete for its planned
60-case/2-repetition small matrix. It is evidence for comparison, not the
three full SH08 pilots. The SH08 checkbox remains unchecked until those full
pilots and their per-file outputs, queues, and costs are completed. Repairs are
reported separately and are partial/non-uniform because the subscription cap
was exhausted. That note predates the bounded low/high comparison recorded
below. Remaining work is offline zero-budget analysis and quality decision-making;
do not check SH10 or authorize more model/TTS calls from this note.

The subsequent actual Muse low/high comparison is recorded in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md), with its
machine-readable summary in
[`output/.../summary.json`](../output/sh08_muse_reasoning_comparison_20260915/summary.json).
It is a completed bounded comparison, not three full pilots and not a semantic
or audio-quality result. The current saved-response analysis is detailed in
[`sh08_duplicate_analysis_20260915.md`](sh08_duplicate_analysis_20260915.md),
and the local Vosk timing follow-up in
[`sh08_zero_timing_20260915.md`](sh08_zero_timing_20260915.md) records 18
measured originals, 1 fit, and 17 overruns. Those measurements provide no sound
or meaning guarantee.

The three full-file dry-runs counted 2,081 positive-budget plans, including
`short_timing` cases that are not all sent to the runner. After filtering to
1,751 `needs_shortening` review candidates, the planner derives 147 review
batches / 882 conservative subscription units. This is the **planned initial
review estimate before repairs, without further measured-audio pruning**, not a
guaranteed actual minimum: errors, `cannot_shorten` outcomes, and local timing
may change the actual calls. The current comparison has 168 overall units
remaining, with 150 units protected for the pilot allocation.

#### Numeric representation prompt rule and recorded follow-ups (15 September 2026)

The shortening prompts now preserve numeric value and source numeric representation:
spelled-out numbers must remain words, so converting `двадцать пять` to `25` is
prohibited as a shortening shortcut. Numbers already written in digits may remain
digits. The guard is recorded in both actual comparison ledgers. No explicit
arbiter-reported representation violation was found, but a zero-violation rate was
not independently verified; historical runs were not rescored.

- [x] Analyze duplicate proposals on saved real outputs first. The scoped
  read-only analysis ran the existing `utils.sh08_duplicate_analysis` CLI once
  against each actual low/high run and wrote fresh reports under
  `output/sh08_muse_reasoning_comparison_20260915/{low_duplicates,high_duplicates}`.
  All 24 low-arm and 20 high-arm requests completed successfully; the high-arm
  model-unresolved result is not infrastructure partiality.
  Current parsed-ok intersection denominators and exact/whitespace duplicate
  counts are low 32/5/5 and high 31/8/8; duplicate repair targets and proven
  avoidable rechecks are zero in both. The historical 59/121 duplicate result is
  retained as historical independent-repetition evidence, not substituted for
  these current counts. This completes saved-output analysis only; a general
  runtime detector remains outside this follow-up. Preserve the `cannot_shorten`
  and unresolved fallbacks; never add endless retries. Existing best-of-5 requests
  1–5 distinct proposals per cue in one request and then performs selection, not
  five requests. Compare against sequential repair only in a separately
  authorized, controlled experiment, not as a default change.
- [x] Verify and compare Muse reasoning modes on the same 60 cases with the Sol
  arbiter and updated prompt version. The manifests requested `low` and `high`,
  and receipt `requested_variant` fields corroborate those modes; this is request
  evidence, not proof of hidden provider reasoning behavior. Independently derived
  accepted-and-fit counts are low 20/41, 22/41 initially and 28/41, 28/41 finally;
  high 27/41, 25/41 initially and 29/41, 28/41 finally. High is +10/82 initial
  and +1/82 final versus low, with higher known cost and serialized receipt time.
  Repairs attempted/rechecked are operational counts, not semantic resolution;
  the high rep1 `cannot_shorten` case is model-unresolved, not infrastructure
  partiality. No separate blinded semantic audit or old-vs-new control was run.
  Muse low remains the default because this sample is insufficient for a universal
  ranking. Per-provider token counters are reported separately; reasoning is not
  summed into output tokens blindly. Both actual runs have standalone HTML reports
  linked in [the comparison record](sh08_muse_reasoning_20260915.md); no offline
  comparison HTML was generated.

Заранее выбрать три файла разной длины и с разной долей коротких фраз, не использованные для настройки инструкций LLM. По возможности получить не менее 500 исходно проблемных реплик суммарно; если их меньше, указать фактическое число. Некоторые источники уже использовались для калибровки Vosk — такой пилот не является новой независимой проверкой TTS.

**Техническая приёмка:**
- один запуск создаёт JSON/SRT, отчёт и очередь проверки;
- нет потерянных или неверно назначенных реплик, изменённых таймкодов и ошибок инфраструктуры;
- завершённый запуск при продолжении не делает новых платных запросов;
- расходы и неизвестные состояния видны отдельно.

**Цель по ручной работе:** не более 15% нерешённых случаев в каждом файле отдельно. Знаменатель — все реплики, изначально требовавшие вмешательства, а не весь файл. В числитель входят короткие нерешённые фразы, ошибки и случаи, требующие альтернативного решения. Нельзя улучшать процент исключением неудобных случаев.

Провести заранее выбранную независимую модельную проверку; небольшая проверка пользователем возможна отдельно. Модельное согласие не является эталоном смысла. Если цель не достигнута, сохранить причины. Если по ошибкам пилота настраивались инструкции, эти файлы становятся набором разработки; для следующей независимой проверки нужны другие файлы.

**Готово, когда:** три файла реально обработаны, по каждому есть отдельные результаты, очередь и расход. Платный запуск отдельно согласуется. Решение коротких аудиопересечений пока не реализуется, но такие случаи не объявляются решёнными.

### [ ] SH-09. Отложенные задачи — P2

SH-09 remains deferred and does not block the core pilot. Its alternative TTS,
short-cue audio, and UI backlog is consolidated in the [future experiments
backlog](future_experiments.md), with the existing [roadmap](tts_roadmap.md)
retained as evidence. Edge-TTS, installations, and changes still require
separate authorization.

### [ ] SH-10. Финальный смысловой оценщик — P1 (будущая задача)

Это предложение, не реализация. Зависит от SH-03 и SH-05. Не блокирует первый технический пилот; может использоваться в SH-06 и SH-08 после выбора режима.

**Current status (15 September 2026):** SH-10 remains unchecked and
unimplemented. A full final-context audit was not performed; it is blocked until
full-pilot input is available and a separate evaluator is implemented. The Muse
low/high comparison does not close this task, and no false closure is recorded.

- Выбирается одна логическая роль сильного оценщика, но не обязательно один HTTP-запрос: большие файлы обрабатываются пачками с перекрытием.
- На входе: оригинал и сокращённая цель, исходный контекст соседей и финальный контекст соседей, чтобы проверять смысл через границы изменений.
- В полном режиме проверяются все сокращённые или обработанные цели и их стыки; неизменённые строки дают только контекст, а не повод заново переводить весь оригинал.
- В режиме выборки проверяется фиксированное подмножество, доступное текущей реализации.
- Для каждой цели выводится: `смысл сохранён`, `нужна правка` или `недостаточно контекста`, с обоснованием и указанием проблемного фрагмента; неоднозначное не принимается автоматически.
- Точная финальная модель и обычный режим по умолчанию пока не выбраны: решение зависит от стоимости и качества.
- Свежая сессия той же модели не является независимой проверкой семейства; где уместно, предпочтительно другое семейство, но конкретная модель ещё не выбрана.
- Независимый контроль пока выборочный, полного финального допуска сейчас нет. Вердикт модели не даёт гарантии 100%; бесконечных повторов и автоматических правок текста не будет.
- Приёмка будущего режима: все запрошенные ID покрыты в бюджете, границы контекста проверены, нет утечки идентичности или истории, неизвестные случаи попали в очередь, расходы отделены, ошибки пачек изолированы, а пробные тесты по возможности обходятся без модельных вызовов.

### [x] SH-11. Per-subtitle outcome markers for review and safe reanalysis — P1

Depends on SH-03; coordinate with SH-05 and implement alongside it before SH-08.
The original requirements are retained below; implementation and verification are
recorded after the acceptance criteria.

Track two independent dimensions for every subtitle: whether its text changed,
and its outcome/evidence. The outcome vocabulary must distinguish `unchanged and
fits`, `accepted correction`, `needs attention/unresolved`, `invalid/error`, and
`not checked` (unverified). Never equate a modified cue with a solved cue, an LLM
acceptance with measured audio fit, or an untested cue with a resolved one.

Canonical metadata belongs in JSON or a sidecar, outside spoken text. For each
stable `cue_id`, retain the outcome and reason, the validation source
(`prediction`, `model`, `human`, or `measured`), measured timing as a separate
field, source-text, current-text, source-timing, current-timing, and profile
fingerprints, and relevant context needed to reuse a semantic decision. Expose
every cue's status in a
human-readable report. SH-05 must queue only unresolved cues.

An optional, separately named, review-only annotated SRT may show a clear
`NEEDS REVIEW` marker and map it by `cue_id`, but it must never alter the primary
subtitle, spoken text, or times. Annotations must never be fed to TTS or length
checks. Do not claim silent machine attributes in SRT; SRT has no such
non-standard metadata contract here.

The analyzer may skip repeated expensive processing only when recorded
approval/evidence matches the current input, text, times, configuration, profile,
and relevant context for that type of check. It must always retain lightweight
integrity checks and changed-neighbor-window checks. It must never skip an
unresolved cue merely because it was attempted or changed, or skip missing actual
measurement merely because a text-accept flag exists. Outdated evidence
invalidates the skip, and an explicit force-recheck option must be available.

**Готово, когда:** offline tests cover a mixed set of fits, changed-and-accepted,
unresolved, error, and unverified cues; reports clearly distinguish these states
while machine/TTS content remains unchanged; markers survive export through the
sidecar; changed text, timing, profile, or context invalidates cached evidence;
and analysis does not hide unresolved cues or remove them from the denominator.

**Implementation note (SH-11, completed alongside SH-05):** The canonical
per-cue outcome state is checksummed and published as `review_state.json`, with
all-cue JSON/Markdown outcomes and an unresolved-only review queue. It records
independent text-change and evidence dimensions, source/current text and
timings, profile/configuration/context fingerprints, reusable checks, and
separate measured duration/fit fields. Forced offline reanalysis invalidates
cached model, human, and measured evidence while retaining lightweight
predictions; external edits never auto-approve. The generated sidecar is kept
outside spoken subtitle text, so SRT/JSON export remains machine/TTS-safe.
[`shortening_review.md`](shortening_review.md) documents application, recovery,
and resume boundaries. SH-10's final-context evaluator remains unimplemented.

**Historical verification (13 September 2026; superseded, past result retained):**
From the activated `subs_env`, `uv run --active --no-sync -m pytest -q`
reported `469 passed`, exit 0. Compileall, all six requested CLI `--help`
checks, and `git diff HEAD --check` each exited 0. The complete captured
stdout/stderr log is `/tmp/opencode/sh05-final-verification/checks.log`. No
network, TTS, LLM, or installation action was performed. The current
continuation baseline is `471 passed, 5 skipped`; see [SH08 preparation](sh08_matrix_preparation.md).

### Known bugs — remaining fixes deferred by user

**Audit note (12 September 2026):** BUG-01 was fixed and tested as part of
SH-04. The following remaining issue(s) are recorded for later triage; no
automatic work has started on them.

- **BUG-01 — P0, fixed in SH-04:** Previously, runner._request caught CancelledError and could continue later batches. Cancellation now propagates, leaves any sent request without a durable response unknown, and does not falsely complete the checkpoint. Covered by fake-cancellation and explicit-retry tests.
- **BUG-02 — P1:** legacy `utils.shorten_subtitles._execute` can lose an
  incomplete child `run_state`, leaving the parent marked complete with exit 0.
  A future fix must propagate incomplete status and nonzero exit distinctly from
  an ordinary manual unresolved result. Test with a mocked child and no network.
- **BUG-03 — P1:** legacy `shorten_subtitles._execute` overwrites
  `runner.cost.projection_500` with `len(stages)` (four categories, not the
  request count) and drops `applicable`/`inapplicable_reasons`. A future fix must
  retain the runner projection and counters. Tests must compare the runner with
  the CLI report and preserve reasons and real counts.

The user will triage these bugs later; they do not take over the current order,
and no automatic work is requested. Recheck them before any paid runs. The
legacy diagnostic `timing_profile` identity limitation is tracked with SH-03;
do not reopen SH-03, which is now implemented with frozen calibration identity
checking in the new command. The legacy diagnostic interface is separate; the
new `shorten_file` command does validate identity.

## Active order of work

**5 Oct voiceover checkpoint:** after listening to the three saved Silero pairs,
the user preferred AFTER for selected Vosk voice 8. Future CLI/UI/native-batch
processing now retains the fixed internal VAD+energy pause stage before measured
bounded tempo. CLI chosen config supports complete original-timeline WAV mixing,
blocked unless every source cue has a current verified fit; native ranged batch
readiness still concerns segments only. No episode rerun, new TTS/LLM call,
historical rebinding or audio overwrite was authorized/performed. Three allowed
offline PRETEMPO reproductions matched approved AFTER exactly; see
[integration evidence](../output/pipeline_vad_integration_20261005/README.md) and
[future entry flow](shorten_file.md#future-voiceover-checkpoint--5-oct-2026).
Listening approval is scoped to those examples/that voice, not a quality or
phoneme-safety guarantee for the whole episode. Existing semantic gates stand.

**Subsequent edge approval, 5 Oct:** the user listened to all eight conservative
edge pairs and approved the fixed policy for voice 8. CLI/UI/native batch now
add edges only when actual internal-processed PCM exceeds the unchanged canonical
window, before bounded tempo. Already-fitting clips remain byte-identical;
internal receipts are reused without a second VAD. Old configs without
`silero_edges.enabled` remain internal-only; the chosen profile defaults on,
with independent disable and early dependency checks. Text/window/runtime and
mix provenance are bound; RAW/PRETEMPO/PAUSES and historical runs stay immutable.
[Evidence](../output/pipeline_edge_integration_20261005/README.md): eight saved
pure-plan matches, conditional 1262/1259 skips and 526 at 3476.961/3517 ms,
207 tests passed and 808 historical/unrelated files unchanged. No fresh inference,
TTS/LLM/ASR, episode run, install, grouping or speaker analysis. Grouping stays
off; speaker-confirmed boundaries are required only for any separately approved
future neighbor joining/borrowing, which this task does not implement. Listening
approval does not guarantee every phoneme or full-episode readiness.

1. The authorized ten-case refusal diagnostic is complete: 9 accepted, one exhausted over-limit, versus six freshly accepted historical alternatives. Do not treat sampled terminal nulls as impossibility proof or promote the full pipeline; the separate matched trial remains 294 accept versus 551 historical accepted rewrites. Candidate-first bounded repairs merit later implementation/verification, not automatic episode-wide extrapolation or a rerun.
2. Zero-cap root-cause fix completed offline on 4 Oct: calibrated zero forecasts remain visible as `calibrated_max_chars=0`; positive nominal CPS targets replace only zero work targets, with an explicit fallback label and timing warning. Replay of all 1,523 original cues changed exactly 344 limits from zero to 3–23; 1,179 positive limits and frozen artifacts are unchanged, including 44 historical fitting originals. See `output/zero_character_target_fix_20261004/before_after.json`; ordinary unprofiled analyzer limits are unchanged, and true 1 ms zeros/invalid timing remain distinguishable. No new TTS/model calls, rerun, automatic continuation or audio approval; candidate-first repairs and SH-10 remain separate future work. Historical module-hash resume guards remain enforced.
3. Historical checks and pending review (not blockers for the text-only trial): arbitrary-neighbor baseline 1/6, textual
   separate-clip sample 0/6, unchanged continuous joins 1/6 outer-duration fits.
   The user confirmed joined `cue-1025`–`cue-1027` and rewritten `cue-19`–`cue-20` locally;
   the latter preserves meaning in the user's qualified judgment, has one narrator and ends in time.
    The completed 3 Oct full-source native Sol trial covers 1,523 cues / 1,494 units: 527 timing/self-assessment-ready, 967 unresolved; 39 successful measured-overrun repairs and conditional usage/cost evidence are saved above.
    Independent text/scene and timing-policy review and missing-retry/billing reconciliation remain pending before audio continuation or integration. Earlier three-group and 2 Oct audio-first outcomes remain historical evidence; the assistant-chosen two-hour cap does not carry into the text-only trial.
   No representative quality rate, further inference, automatic integration or per-cue mapping is implied.
   Preserve the frozen episode. Seventeen accepted finals have individual timing
   measurements; **534** remain unmeasured and semantic quality is unvalidated.
4. Review the completed eight-case [MiMo arbiter check](shorten_file.md#experimental-mimo-arbiter-check)
   against saved Sol reviews and original/final neighboring context; neither
   model is semantic ground truth and USD cost was not reported. Do not rerun.
5. Before production integration, not before the text-only trial, review these checks for short-window timing, SH-10 context
   work, unresolved cues or the other two full-file pilots. Concurrency,
   batching and cheaper Sol modes are explicitly deferred until later; retain
   repair/recheck unless quality evidence supports changing it. Older runs
   remain separate historical evidence, including their charges.

SH-06, SH-07, SH-09, and other optional work stay deferred in their linked
backlogs. SH-07's partial verified snapshot remains open; no data deletion,
migration, or mass documentation is implied. SH-10 remains unchecked and
unimplemented until final-context semantic work is actually completed.
This documentation update itself makes no inference calls.

## Коммиты и гипотезы

- Сначала один одобренный коммит рабочей исходной точки.
- Затем одна законченная задача — один проверенный рабочий коммит, например `fix(shorten): isolate malformed batch rows`.
- Перед платным экспериментом фиксировать коммит кода, версию инструкций и манифест. После эксперимента сохранять краткий результат отдельным коммитом документации при необходимости.
- Для каждой гипотезы записывать: что ожидаем изменить, с чем сравниваем, какие данные используем, как измеряем результат, когда останавливаемся, где код и результаты. Итог: подтверждена, не подтверждена или данных недостаточно.
- Коммиты создаются после явного разрешения пользователя. Откат кода через Git не возвращает удалённые игнорируемые результаты — для них нужен архив.

## Сколько места занято сейчас

Снимок `output` на 10 сентября 2026:

| Данные | Логический размер |
|---|---:|
| Всего | 264.80 МиБ |
| JSON | 173.68 МиБ |
| WAV | 62.52 МиБ |
| JSONL | 10.30 МиБ |
| SRT | 16.77 МиБ |
| MP3 | около 1.47 МиБ |
| Файлы `.log` | 23.1 КиБ |

Фактически на диске выделено около 283.65 МиБ. Основной объём дают многократные полные экспорты JSON, а не маленькие технические логи. Срочной нехватки места нет.

Есть 186 экспортов `subtitles.json`, из них 152 различных по содержимому, и 186 SRT, из них 151 различный. Точный объём, освобождаемый устранением дублей, отдельно не измерен. Нельзя считать все JSON ненужными копиями.

## Что было выполнено на момент исходного аудита

Проведён аудит без изменений данных и составлен этот план. На тот момент код не
менялся; модели и TTS не запускались; файлы не удалялись, не перемещались и не
сжимались; baseline-коммита ещё не было. Последний на тот момент тестовый
результат и измеренные объёмы не выдавались за новую проверку реализации.

Историческое решение на 13 сентября 2026: SH-00 и SH-01 реализованы и уже
присутствовали в HEAD `ec0cbaf` до этой задачи; SH-02 зафиксирована коммитом
`c5bf383`; SH-03 зафиксирована коммитом `9a1ddd9`; по запросу пользователя SH-04
зафиксирована коммитом `a958052`; реализация SH-05 и SH-11 зафиксирована
коммитом `344880b`, а SH-08 по-прежнему ожидает отдельного разрешения на
платный технический пилот; все последующие задачи ожидают.

### Scoped source listening confirmation — 6 Oct 2026

[Human confirmation](../output/pipeline_source_confirmation_20261006/README.md)
now establishes one speaker **within each** 1263–1264 and 1270–1271 pair; prior
unknown-speaker statuses for those pairs are historical. No diarization is needed
for these two pairs. Cross-pair identity, exact speech timing, puzzle meaning,
grouped-text approval and overall quality remain unconfirmed; production
retiming/synthesis is not authorized. No frozen evidence or text changed. A future
joint candidate per thought needs independent review and actual timing checks;
this entry does not execute or approve that next step.
