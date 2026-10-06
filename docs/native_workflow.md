# Native author/reviewer → full-file editor → voiceover

Run commands from the repository root after `source subs_env/bin/activate`.
Nothing in this guide is an instruction to rerun a historical pilot. The editor
does not call language models, paid APIs, transcription or speaker detection.
Preparing, adopting, exporting, importing, approving and viewing status do not
run synthesis. Only an explicit `audio` action can load local normalization/TTS
models; `mix` assembles already validated audio.

## Two distinct workflows

**Active native workflow:** `utils.pipeline_batch` freezes the complete source,
exports requests for an assistant, accepts independently reviewed native text,
then measures local Vosk audio. The assistant coordinates author and reviewer
outside Python. One initial proposal batch and at most one measured repair
batch are allowed. After those two rounds, unresolved cues go to the human
editor; they do not cause a third automatic author/model round.

**Legacy API workflow:** `utils.shortening_terminal` / `utils.shorten_file` use
their own saved plans, calibration profiles, request budget and API author/arbiter
settings. See [the legacy CLI guide](shorten_file.md)
for exact options and historical run commands. These are not native editor
commands or an automatic fallback. Do not feed a legacy results file into
`adopt_batch`, and do not silently resume a paid run to fix editor exceptions.

The existing `ui.gradio_app` is a separate **Vosk/Edge audio-pipeline UI**, not
the API shortening author/arbiter UI. The new active editor UI is
`ui.native_editor`; it uses the native project editing/audio API below.

The native reviewer checks the exact proposed candidate and exact assembled
chosen text, independently of the author. Its acceptance is model-only semantic
review, not proof that the WAV fits or that a human listened. The legacy
author/arbiter and budget-based selection use a different protocol; their
“accepted” and “unresolved” totals are not interchangeable with measured native
audio statuses.

## 1. Prepare the full episode and ask the native assistant

Use bounded, disjoint native batches (for example **31 cues** at a time), or a
smaller range chosen to fit the assistant/model context. The source itself stays
complete. Example for the first 31 positions of a **1,523-cue** episode:

```bash
source subs_env/bin/activate
SOURCE='/path/to/episode.srt'
BATCH='output/episode_native_0000_0030'
PROJECT='output/episode_editor'
uv run -m utils.pipeline_batch prepare "$SOURCE" --start 0 --end 30 \
  --config pipeline_config_vosk.json --output-dir "$BATCH"
```

Continue with positions `31–61`, `62–92`, etc., ending at the actual last position
(`1522` in this example). Use a new batch directory for each range and adopt all
disjoint batches into **one** complete editor project. The CLI can prepare a
full range, but a single 1,523-cue author/reviewer request is not guaranteed to fit
the model context. Do not silently treat that large range as an unattended model
job. Assistant coordination must bound each request and its supplied context.
Provide file references and relevant context rather than blindly pasting a whole
episode into each request. A partial batch still freezes the full source for
context and timing, but is **not** a complete episode mix.
Original position IDs (`cue-0`, `cue-1`, …) differ from SRT indices; both survive
editing, including nonconsecutive or nonascending original SRT indices.

Give the native assistant `request_round1.json`, `full_source.json` and
`full_plan.json`. Ask it to author one proposal per requested cue, preserving
meaning, names, numbers and natural Russian. Return the request-bound
`pipeline-batch-proposals-v1` JSON, including actual writer model, mode and
invocation identity. Python does not obtain those proposals for you.
Keep the complete source available for meaning/context checks while bounding the
actual request size; exact candidate and selected-assembly review is still required.

```bash
uv run -m utils.pipeline_batch review-input /path/to/writer_round1.json \
  --output-dir "$BATCH"
```

Give `reviewer_input_round1.json` and the full source to an independent native
reviewer. It must check exact candidate hashes **and the exact chosen assembly**,
returning `pipeline-batch-reviewed-v1` with real review identity and evidence.
If some candidates are rejected, use `review-input --per-cue-reviewed
/path/to/per_cue_review.json` to export the accepted-subset assembly, then have
the reviewer approve that exact assembly before importing. Do not mark fabricated
or dry-run evidence as `native_live`.

```bash
uv run -m utils.pipeline_batch ingest-reviewed /path/to/reviewed_round1.json \
  --output-dir "$BATCH"
uv run -m utils.pipeline_batch audio --output-dir "$BATCH"
uv run -m utils.pipeline_batch status --output-dir "$BATCH"
```

The `audio` command is the explicit local synthesis authorization. If
`request_round2.json` has targets, repeat the author → `review-input` → independent
reviewer → `ingest-reviewed` → `audio` sequence using round-2 files and measured
advice. Character budgets are advice, not semantic acceptance or guaranteed fit.
There is no automatic third round. Real native runs publish their standalone
`report.html`, including partial runs; viewing saved artifacts needs no inference.

## 2. Adopt into a complete, immutable-source editor project

```bash
uv run -m utils.pipeline_editor init "$SOURCE" --project "$PROJECT" \
  --config pipeline_config_vosk.json
uv run -m utils.pipeline_editor adopt "$BATCH" --project "$PROJECT"
uv run -m utils.pipeline_editor export --project "$PROJECT"
uv run -m utils.pipeline_editor status --project "$PROJECT"
```

`init` requires a new directory. It accepts SRT or a batch's **full**
`full_source.json`, copies the original and freezes the selected effective config
and runtime/model identity. Sources and batch audio are never overwritten.
`adopt` validates the current batch runtime, full origin, chosen text, reviews and
WAV hashes, then copies its audio and provenance into the editor. Disjoint batches
can be adopted separately; an overlapping import cannot replace human edits,
approvals or existing audio. Adopting the identical batch again is a no-op.
Interrupted/failed copies are not adopted: the project state changes only after
the complete copy succeeds. A fresh unique import directory lets the same batch
be retried without deleting prior orphan copies or unrelated user files.

Export always writes **all source cues**, including unresolved/unprocessed ones,
to `exports/<revision>/subtitles.srt` and `project.json`. The SRT contains speech
text only: pending labels are not spoken text. The JSON has cue IDs, original
indices, current millisecond timings, source hash, revision, review/audio statuses,
`needs_human`, reasons and audio paths (null when unavailable). A full draft export
is not necessarily a ready voiceover.

Historical batches can intentionally fail strict runtime loading after code,
model weights or settings change. Do not bypass that guard or relabel their WAVs
as current. To reuse source context only, initialize a new project from their
`full_source.json` and the current selected config. That imports neither old audio
nor native approval. Historical saved HTML remains directly viewable.

## 3. Edit SRT, approve current text/timing, process only the queue

Copy the exported SRT to a separate editable file. Preserve **all** indices,
their order and cue count. Do not delete, split, merge or renumber cues. Text and
timestamps may be edited; negative/zero durations and overlapping timings are
rejected. Obtain the current `revision` from `status` before each changing command:

```bash
uv run -m utils.pipeline_editor import /path/to/edited.srt \
  --project "$PROJECT" --expected-revision CURRENT_REVISION
uv run -m utils.pipeline_editor approve --project "$PROJECT" \
  --cue-ids cue-12 cue-25 --expected-revision NEW_CURRENT_REVISION
# Alternatively, approve the complete current source in one explicit operation:
uv run -m utils.pipeline_editor approve --project "$PROJECT" \
  --all --expected-revision NEW_CURRENT_REVISION
uv run -m utils.pipeline_editor audio --project "$PROJECT"
uv run -m utils.pipeline_editor export --project "$PROJECT"
```

Use the revision returned by the previous operation, not a literal placeholder.
A stale revision refuses the edit rather than losing another editor's changes.
SRT import is a human edit, not native reviewer evidence. Approval is explicit
human **current text/timeline** approval, not a claim of measured fit, listening
or model review. No pairwise approval ritual is required: selected cues or the
whole current source can be approved together.

Unchanged successful audio is skipped byte-for-byte. A text edit clears current
approval and hides old audio until approved processing succeeds; old files remain
preserved. Timing edits also invalidate the previous cue when its available window
changes. After renewed approval, unchanged text can be remeasured from immutable
RAW in a fresh directory, without new TTS or applying tempo to an already accelerated
final. The base rate and total speed cap come from the frozen config.

Normalization rejects lexical changes (for example `крут` → `крутой`) instead of
silently speaking a different word. A human can rewrite the flagged cue, approve
that exact new text and explicitly process it. Failed/interrupted intents never
retry automatically. If the text itself should stay unchanged, explicitly select
the failed cues for one fresh attempt:

```bash
uv run -m utils.pipeline_editor retry --project "$PROJECT" \
  --cue-ids cue-12 --expected-revision CURRENT_REVISION
uv run -m utils.pipeline_editor audio --project "$PROJECT"
```

Check an interrupted attempt before choosing retry: the previous synthesis may
have occurred even if completion was not saved. Retry is a new user authorization,
not a guarantee that no prior work happened. Runtime changes block audio and mix;
there is no silent model/config rebinding.
Draft status and full SRT/JSON export remain available when the runtime cannot be
loaded; JSON records the actual runtime error and does not expose current audio.
An explicit retry can also repair damaged/missing final audio from intact immutable
RAW without new TTS. Valid completed fits cannot be retried; missing/changed RAW
blocks this recovery instead of silently resynthesizing a completed cue.

## 4. Full voiceover and optional simple UI

```bash
uv run -m utils.pipeline_editor mix --project "$PROJECT"
uv run -m ui.native_editor
```

The simple Gradio editor uses the same project API; it does not author/review text
through model APIs. CLI remains usable without launching the UI.

`mix` requires complete original coverage, approval bound to every current text
and timing window, intact current WAV hashes, valid decoded frames, measured fit,
bounded total speed and a nonoverlapping full timeline. It uses fresh indexed
copies and the existing mixer with missing-audio errors. A stale “fits” label,
changed/missing WAV, pending approval or batch slice cannot authorize a full mix.
The result is a new `mixes/.../voiceover.wav` using the original or explicitly
human-edited full timeline. This is not a listening/semantic quality guarantee.

Optional `speaker_id` is a human annotation only. It is not the configured TTS
voice and does not select voices, stretch groups, borrow time or run diarization.
Changing annotation alone preserves text/timing approval and PCM. Group metadata
can inform future work, but no automatic grouping is implemented here.

## What the current evidence means

The [saved 62-cue diagnosis](../output/pipeline_remaining_diagnosis_20261006/summary.json)
records **43/62 individually fitting WAVs**, **18 timing exceptions** and **one
normalization failure**. These are saved measurements, not a new episode run or
certified speaker/group placement. The [legacy full episode](shorten_file.md#current-fresh-episode-run)
recorded **544 unresolved** out of 1,219 selected targets in a 1,523-cue source.
That API selection/acceptance result and these native measured audio samples use
different protocols and denominators: neither is evidence that one workflow
will outperform the other on the complete episode.

The sampled coverage is `62 / 1523 ≈ 4.1%`. Simple proportional arithmetic gives
`19 × 1523 / 62 ≈ 467` potential **manual exceptions**, not 467 mandatory rewrites.
Timing exceptions may call for human source/timing checks rather than shortening;
normalization is a separate issue. The two 31-cue samples have 8 and 11 exceptions:
their separate extrapolations are roughly **393–540**. This is not a statistical
confidence interval: scene clustering and untested episode/season differences
are unknown. There is no whole-episode or ten-episode success guarantee.

For a season, use a separate source/project workspace per episode. There is no
batch scheduler or unattended native-model service: assistant coordination and
human exception decisions remain necessary. This implementation was checked
offline using synthetic WAVs and mocked model/TTS calls, not a new live full run.
