# Future experiments backlog

This is the single backlog for optional experiments. Every item below is
deferred; none is part of the current production path. Existing code, real
artifacts, history, and experiment records remain in place. Parking an item
here does not mean that code or results were removed.

No budget is authorized by this backlog. Any live model, translation, TTS, or
UI work needs separate approval and must report its actual USD usage, unknown
costs, and primary artifacts. Acceptance criteria below describe what must be
held fixed and what evidence is required; they do not invent a universal
quality threshold or spending limit.

## H2 — critique control

**Status:** Deferred.

**Evidence:** The original SH-06 control description and current runner
boundaries remain in [`shortening_tasks.md`](shortening_tasks.md). Historical
SH08 comparison evidence is recorded in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md).

**Hold fixed:** Use the same cases, first candidate, model settings, prompt
version, character budget, `length_basis`, batching, retry policy, and one
additional attempt. Change only whether the critique is used for the repair
decision. Keep model and transport failures separate from semantic outcomes.

**Accept when:** The comparison reports matched outcomes, meaning findings,
fit, unresolved cases, latency, tokens, and actual cost with uncertainty and
does not promote a control based on style preference alone.

## H3 — best-of-five on hard cases

**Status:** Deferred; not a default five-request pass.

**Evidence:** Existing SH08 notes record that a best-of-five request means up
to five proposals in one request, not five independent requests; see the
best-of-five note in [`shortening_tasks.md`](shortening_tasks.md) and the
historical model records in [`shortening_model_pairs.md`](shortening_model_pairs.md).

**Hold fixed:** Select and freeze only hard, rejected, or otherwise explicitly
justified cases before the run. Keep source context, target IDs, model,
prompt, limits, review policy, and comparison arm fixed. Distinguish one
request containing several proposals from sequential repairs, and retain all
candidate and rejection provenance.

**Accept when:** An independently reviewed matched comparison shows whether
extra candidate diversity changes meaning preservation, fit, unresolved
coverage, latency, and actual cost. No general benefit is claimed from a
small diagnostic sample.

## H4 — context presentation

**Status:** Deferred.

**Evidence:** The existing SH-06 H4 proposal is retained in
[`shortening_tasks.md`](shortening_tasks.md); related preparation and matrix
records are linked from [`sh08_matrix_preparation.md`](sh08_matrix_preparation.md).

**Hold fixed:** Compare the current structured cue list with chronological
continuous context and an explicit target boundary. Use the same target,
neighbors, order, IDs, model, settings, budget, `length_basis`, batching,
tools, and retry policy. Do not add summaries, rewritten context, or facts;
the output must be the target only. Freeze cases involving pronouns,
negation, contrast, causality, split sentences, repeated targets, and
ambiguous references before any live comparison.

**Accept when:** Blinded semantic checks report meaning loss, neighbor-fact
contamination, target/ID correctness, fit, unresolved cases, tool calls,
tokens, and actual cost. A format change is adopted only if the evidence
supports it; model agreement is not ground truth.

## H5 — Jev selection and acceptability

**Status:** Deferred; Jev is not the main arbiter and is not promoted to the
production default.

**Evidence:** The bounded replay and its limitations are recorded in
[`sh06_jev_replay_20260919.md`](sh06_jev_replay_20260919.md), with the
follow-up proposal in [`sh06_jev_followup_20260919.md`](sh06_jev_followup_20260919.md)
and the full plan in
[`sh06_jev_typesafeai_arbiter_plan.md`](sh06_jev_typesafeai_arbiter_plan.md).
These are historical checks, not authorization for more calls.

**Hold fixed:** Use a controlled matched comparison with a frozen rubric,
held-out cases, the same candidate inputs, and an independent Russian
semantic check. Keep Jev, Sol, generator, transport, and subscription costs
separate. Do not call a same-family or self-review result independent.

**Accept when:** The evidence distinguishes semantic errors, acceptable
rejections, infrastructure failures, unresolved cases, latency, and actual
cost, and is strong enough to justify a role decision without claiming a
winner from the existing replay.

## H6 — failure-only Sol rewrite

**Status:** Deferred; proposal only.

**Evidence:** The historical H6 proposal and recorded checks remain linked in
[`sh06_jev_followup_20260919.md`](sh06_jev_followup_20260919.md) and
[`shortening_tasks.md`](shortening_tasks.md).

**Hold fixed:** Enter only when every locally valid candidate is rejected or
bounded repair is exhausted. Compare one Sol rewrite with one extra Muse
repair on the same rejected cases, using the original context, target limit,
candidate pool, and frozen feedback. Check format, length, deduplication,
numeric representation, meaning, and provenance locally; never let Sol
self-approve. Keep the original when the result is unresolved.

The examples `16e2689c67a5:cue-618` and `2c58c71203c3:cue-210` remain
diagnostic proposals only. They do not authorize automatic new tests or
generalize to all hard cases.

**Accept when:** A separately authorized, blinded review reports critical
meaning errors, accepted-and-fit coverage, unresolved cases, latency, and
actual incremental cost against the matched Muse repair. Disagreement remains
visible rather than being resolved by style preference.

## Muse high reasoning mode

**Status:** Deferred; high is not the production default.

**Evidence:** The completed bounded low/high comparison and its standalone
reports are recorded in
[`sh08_muse_reasoning_20260915.md`](sh08_muse_reasoning_20260915.md), with
the machine-readable summary linked there. The comparison did not establish
a universal ranking or close SH08.

**Hold fixed:** Use the same cases, prompt, model, arbiter, context, limits,
retry policy, and accounting as the low-mode comparison. Report requested
mode, returned model information, actual USD usage where known, latency, and
unresolved outcomes separately.

**Accept when:** A separately authorized comparison supplies enough semantic,
fit, operational, and cost evidence to justify a production-mode change. A
small historical difference is not by itself a winner or a reason to make
high the default.

## Direct English → Russian translation under aligned Russian timing

**Status:** Deferred until the core pilots and production-path stability are
complete.

**Evidence:** Existing timing constraints and measured-profile limits are
documented in [`shortening_memory.md`](shortening_memory.md) and
[`vosk_timing_diagnostics.md`](vosk_timing_diagnostics.md). The SH08
preparation source is the exact
[`docs/sh08_pilot_manifest.json`](sh08_pilot_manifest.json) file (SHA-256
`dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`), not a
copy of its contents.

**Hold fixed:** Compare direct English→Russian translation from the English
subtitle/full context with shortening the existing Russian subtitle on the same
aligned IDs, frozen Russian cue windows, available gaps, voice/profile, review
criteria and bounded repair policy. Preserve the aligned Russian timing; do not
assume English cue boundaries or speech durations match it. Measure real Russian
TTS only with separate authorization, then make only bounded text or timing
adjustments for available gaps. Do not create overlap, cross speaker changes,
or unverified meaning changes. The
timing window differs from actual speech duration; the existing 1.4-second
speech example must not be treated as the timing window. Direct translation is
not guaranteed to be better than the existing Russian subtitle.

**Accept when:** Meaning, speaker attribution, timing, real TTS measurement,
and any bounded adjustment are all recorded per cue, with unavailable gaps
and unresolved cases preserved. This work needs separate authorization and
must not be inferred from text-only acceptance.

## SH09 — alternative TTS and UI

**Status:** Deferred.

**Evidence:** The existing roadmap is [`tts_roadmap.md`](tts_roadmap.md).
SH09 covers alternative TTS engines/profiles, short-cue audio policy, and a
graphical interface; it does not block the core pilot.

**Hold fixed:** Keep the current subtitle text, source/timing identity,
speaker changes, and baseline voice/profile available for comparison. Do not
introduce Edge-TTS, new installations, or UI behavior without separate
authorization.

**Accept when:** Each alternative has reproducible source/profile identity,
real audio measurements, failure and cost records, and a clear comparison to
the current baseline. The UI must expose unresolved and unknown states rather
than implying that a changed subtitle is audio-verified.
