# Shortening HTML reports

The report publisher reads saved shortening artifacts and does not call a model,
the network, Vosk, or any other inference service.

## Open a report

From the repository root:

```bash
uv run --active --no-sync -m utils.shortening_html OUTPUT_DIR --confirmed-real-run
```

Open `OUTPUT_DIR/shortening.html`. The confirmation flag is an explicit user
assertion that historical outputs came from a genuine real run; it is not a
permission to run inference. Without the flag, only an observed live entrypoint
may publish. The publisher writes `html_report_metadata.json` beside the HTML.

## Automatic live publication scope

These five CLI roots own the `live_html_report` scope:

- `utils.sh08_matrix run`
- `utils.shorten_file`
- `utils.shorten_subtitles --execute`
- `utils.shortening_experiment` in execute mode
- `utils.shortening_replay` in execute mode
- `utils.jev_replay run`
- `utils.jev_followup run`

Inside those scopes, only an actual model transport HTTP call records a request
with `note_model_request()`, immediately before the call. Preparation, local
work, preflight, and non-model HTTP calls do not authorize publication. A
failed or partial run is still reportable after a real model request attempt.

Programmatic fake helpers do not create HTML automatically. A programmatic real
caller should wrap its work explicitly with `live_html_report(path)`, or run the
offline confirmed builder with `publish_html_report(path, confirmed_real_run=True)`
for saved artifacts. The confirmed builder does not run inference.

Automatic publication is best effort and cannot replace the live run's primary
exception, including when warnings are treated as errors. Explicit confirmed
publication keeps renderer and file errors real and raises them to the caller.

## Supported artifacts and coverage

The loader supports SH08 matrix manifests and case results, ordinary
`report.json` trial runs, Jev replay `schema=jev_replay.v1` reports, A/B/C
experiment roots, and legacy roots containing standard child reports. Jev
rows retain every original, context item, candidate, saved Muse provenance,
Jev probabilities, confidence values, and historical Sol notes. Pending and
error calls are never presented as accepted; an accepted Jev row requires the
saved accepted status, a known candidate id, matching accepted text, and a
safe saved choice. It retains missing, zero, pending, declined, and
unpriced states instead of turning them into successes. Length badges use only
saved local measurements; absent measurements remain unknown rather than being
recomputed from displayed text. It reports full model identities and modes
saved in the artifact, never the current configuration.

The output is one combined parent-root report; it does not create an HTML file
for every child source report. The resulting HTML is a standalone file that can
be opened directly; it does not need a report server. Unknown schemas are
rejected explicitly.

Partial reports preserve the proposals, arbiter and model modes, accepted final
text, and status available in the saved artifacts. Publishing them performs no
fresh inference: it only loads and renders existing files.

## Ordinary primary reports

An ordinary `utils.shorten_file` primary report keeps the frozen generator and
arbiter identities from `report.json`, including their saved modes. When the
report contains the OpenRouter pricing snapshot and budget summary, the HTML
shows provider-reported API USD costs separately for generation/repair and
review/recheck, along with reported input/output/reasoning/cache token counts,
physical requests, the request limit, limit, remaining amount, unknown
reservation, next reservation, stop reason, and accounting state. A reservation
is a budget admission amount, not provider spend; missing usage is never
rendered as zero, and no subscription or free-cost claim is made.

The report also retains the pricing snapshot ID, pinned Meta/OpenAI route
names, request-fee assumption, and provider receipt IDs when those safe IDs are
persisted in the report. Budget-stopped runs are explicitly marked partial and
do not make public quality claims. Original-timing preflight is not final-audio
evidence, so ordinary primary reports note when final voiceover duration has
not yet been checked and unresolved lines retain their original text.

## Reasoning and privacy

Only the saved model mode/`variant` is displayed as a reasoning mode. Jev's
saved `Choice` selection and safety mode is displayed as a decision mode, not
as reasoning effort. Private reasoning, prompts, request headers, ledgers,
raw response bodies, and environment data are not exported. Analysis-derived status is shown only when
the four required saved-file hashes (manifest, ledger, case results, and
report) match; stale provenance is warned about and ignored. Selected error
labels redact common credential formats.

Jev replay publication is local report rendering, not a public benchmark or a
claim that the historical Sol output is true. It does not alter final
subtitles. Prepared, fixture, offline, and `backend_consumed=false` artifacts
are rejected; a saved actual partial or complete report can be rendered with
`--confirmed-real-run` without new inference. A live `utils.jev_replay run`
scope publishes after its actual model-call marker, including partial runs.

## Jev follow-up reports

The follow-up producer writes `report.json` with the exact
`jev_followup.v1` schema. To render a saved private run from the repository
root, use:

```bash
uv run --active --no-sync -m utils.shortening_html OUTPUT_DIR --confirmed-real-run
```

The live follow-up command is run from the repository root:

```bash
uv run --active --no-sync -m utils.jev_followup run --output-dir OUTPUT_DIR
```

The loader accepts only a report with an observed or explicitly confirmed real
model attempt. Prepared and zero-attempt follow-ups cannot publish HTML. The
58 rows contain 12 unchanged-input repeats, isolated safety evaluations, and
independent DeepSeek reviews. Evaluation rows are never selections: they show
`Оценён, без выбора` and cannot provide accepted final text. The HTML is a
private experimental diagnostic for six selection-biased cases, not a public
benchmark or ground truth; it does not change final subtitles. The report
shows saved Jev probabilities/confidence, baseline payload hashes, candidate
provenance, safe context details, per-backend usage and cost accounting, and
only model-provided independent-review issues/advice.

## Manual backfill

Manual backfill revisions r1/r2/r3 are authorized for later review, but this
command does not perform them automatically. Confirm only a saved historical
run that you know was real; do not confirm dry-run, fixture, simulation, or
offline artifacts.

Tests use temporary fixtures and do not create a permanent HTML artifact.
