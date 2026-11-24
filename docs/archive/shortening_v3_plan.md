# Subtitle shortening v3 — a fresh design

Status: phase 1 implemented in `voiceover/` (see section 13); later phases are proposals. The design deliberately does
not inherit the previous shortening workflow or its code; only SRT parsing
and the raw TTS wrapper are considered reusable building blocks.

## 1. Reframing the problem

The usual framing is "cue N is too long, shorten cue N". That framing loses
twice: it ignores the slack that exists between neighbouring cues, and it
asks a language model to hit a length it cannot measure.

The voiceover is one continuous audio track. The real constraints are:

* speech for a cue should begin close to the cue's start (sync tolerance);
* speech for a cue must end before the next cue's speech begins (no
  collision), with a small breathing gap;
* speech must not be faster than a comfort limit.

The end of the subtitle on screen is **not** a constraint for the audio: a
voiceover line that runs 400 ms past its subtitle is normal, as long as it
does not collide with the next line. So the problem is a **scheduling
problem on an elastic timeline**, and text shortening is the last resort
the scheduler reaches for when shifting and tempo are exhausted.

Three ideas follow from this and form the backbone of v3:

1. **Elastic timeline solver** decides *how many seconds* each cue must
   lose, after shifting and tempo are used globally, not per cue.
2. **Semantic units** turn "shorten by 30%" into "drop these units, keep
   these", which is explainable, checkable and repeatable.
3. **Duration oracle** gives the model and the pipeline one engine-neutral
   function `seconds(text)`; it is self-calibrated on every run from the
   engine's own first synthesis pass.

## 2. Elastic timeline solver

Inputs per cue: `start`, `next_start`, measured speech duration `d0` at the
engine's base rate (from the first synthesis pass, which the pipeline does
anyway), and policy:

```text
lead_ms      how early speech may start before the cue (default 250)
lag_ms       how late speech may start after the cue (default 400)
gap_ms       minimal silence between two spoken lines (default 150)
tempo_max    comfort limit on speed-up, total (default 1.5; hard ceiling 1.6 per cue)
```

The solver places every line on the timeline, minimising a cost:

```text
cost = Σ |shift_i| / lag_ms  +  Σ (tempo_i − 1) · w_tempo  +  Σ removed_sec_i · w_text
```

subject to `end_i + gap ≤ start_{i+1}` for consecutive spoken lines.
Because the constraints are a chain, this is solvable exactly by dynamic
programming over cues in order (state = end time of the previous line,
discretised to 10 ms), or as a small linear program. Output per cue:

```text
start_shift_ms, tempo, deficit_sec    (deficit > 0 ⇒ text must lose that much)
```

Why this matters: a long cue followed by silence has no deficit at all; a
long cue followed by another long cue shares the problem with its neighbour,
and the solver chooses the cheaper place to cut. Shortening targets become
smaller and fewer than any per-cue analysis would produce, and they are
expressed in seconds, which is the only unit that is actually true.

The solver also defines the budget for **mergeable lines**: two cues that
are one sentence (no terminal punctuation, next starts lowercase, gap under
500 ms) are scheduled as one line with one start and one deficit, and the
engine synthesises them as one utterance. That is both more natural and
cheaper to fit.

## 3. Duration oracle

```python
class DurationOracle:
    def seconds(self, text: str) -> float          # fast estimate
    def measure(self, text: str) -> float          # real synthesis, exact
    def identity(self) -> dict                     # engine, voice, rate, version
```

The fast estimator is a small regression on features that drive speech
time in Russian: syllable count (vowels), word count, punctuation pauses,
digits. It is **fitted on the current run's own first pass**: every episode
synthesises all its cues once anyway, so there are hundreds of
(text, measured seconds) pairs for exactly this engine, voice and rate. No
separate calibration corpus, no stale profile. The fitted coefficients and
their residual spread are written into the run manifest; the spread becomes
the safety margin.

Engines plug in by implementing `measure`; the estimator is derived.

**Speech-rate canonicalisation.** Each engine's base rate parameter is
mapped to syllables per second measured on the first pass. The pipeline
config states one target, e.g. `5.5 syl/s`; the engine adapter chooses its
own rate parameter to hit it. The timeline solver and the shortener never
see engine-specific rates, only seconds and syllables. Comparing engines
becomes a one-line report: measured syl/s at base, residual spread of the
oracle, fraction of cues with deficit.

## 4. Semantic units instead of percentages

Before any compression, one cached model pass per scene (a run of cues
with gaps under ~2 s) annotates each deficit cue:

```json
{"cue": 17,
 "units": [
   {"id": "u1", "text": "Они дали нам две минуты", "role": "core"},
   {"id": "u2", "text": "чтобы собрать всё с лодки", "role": "support"},
   {"id": "u3", "text": "честно говоря", "role": "decor"},
   {"id": "u4", "text": "ребята", "role": "address"}],
 "refs": {"они": "команда продюсеров (cue 15)"},
 "act": "statement|question|exclamation|command"}
```

Roles: `core` (the proposition the scene depends on), `support` (detail
that can be compressed or implied), `decor` (fillers, hedges, repetition),
`address` (vocatives). Resolved references tell the compressor which
pronoun substitutions are safe.

Why: it turns compression into a ladder of explicit, auditable moves,
makes repeated runs converge on the same decisions (same units ⇒ same
drops), and gives the verifier a checklist instead of a vague "is the
meaning preserved?".

## 5. Compression with the oracle in the loop

The compressor receives the scene text, the annotated units of the target
cues, the deficit in seconds, and a tool `seconds(text)` backed by the
oracle. It is asked for the **smallest edit** that reaches the deficit,
following the ladder:

```text
1. drop decor and address units
2. compress support units (synonyms, pronouns from refs, active voice)
3. merge with the neighbour line and rebalance words across the boundary
4. drop a support unit entirely
5. core-only rewrite, flagged "lossy" (never silent)
```

Output is not just text; it is the text plus the list of unit ids kept,
dropped and compressed. The model is instructed to call `seconds()` on each
attempt and stop at the first rung that satisfies `seconds(text) ≤
target − margin`. If native tool calling is unavailable for a provider,
the same loop runs outside the model: ask for rungs 1–5 as five texts in
one response, score them with the oracle locally, take the first that fits.

Why the ladder beats "give me three variants": each rung is a bounded,
nameable change, the ledger records which rung was needed, and the next
episode's statistics tell exactly where the difficulty lives.

## 6. Verification as a checklist

1. **Hard local checks** (reject): single line; `seconds(text)` within
   target; every number word from the original present; no new
   capitalised tokens absent from the scene; negation particles not lost
   unless the unit containing them was explicitly dropped.
2. **Unit entailment** (a second, different-family model, cached): for each
   `core` unit the question "Does the new line still state this?" with a
   yes/no answer. One rejected core unit ⇒ candidate rejected and the
   compressor is re-run with that unit marked `locked`.
3. **Measured fit**: the chosen text is synthesised; the timeline solver is
   re-run with real durations. Only cues whose measured deficit is still
   positive go to a second compression pass; nothing else is touched.
4. **Normalisation guard**: the engine adapter checks that normalisation
   (numbers, abbreviations, stress) did not replace words.

## 7. Repeatability

* All model calls are pure functions of `(provider, model, prompt version,
  request JSON)`; responses are cached under that hash. A second run of the
  same input and config performs zero network calls.
* Decisions downstream of the model are pure functions of numbers
  (oracle estimate, measured seconds, solver cost). Even if a provider
  returns a different paraphrase one day, the *policy* that selected it is
  reproducible and the ledger shows the deviation.
* Temperature 0, provider seed when supported, scenes processed in source
  order, fixed scene sizes.
* The run manifest records: engine identity, oracle coefficients and
  residual spread, policy values, prompt versions, model ids actually
  returned, and a per-cue ledger `{rung, units kept/dropped, shift, tempo,
  measured seconds, verdicts}`.

## 8. Pipeline

```text
parse       srt → cues.json
synth-0     all cues at base rate → durations0.json        (engine)
fit-oracle  durations0 → oracle.json                        (local)
schedule    cues + durations0 + policy → schedule0.json     (solver; deficits)
scenes      group cues into scenes; mark mergeable lines    (local)
annotate    deficit cues → units.json                       (model, cached)
compress    units + deficits + oracle → candidates.json     (model + oracle)
verify      candidates → verified.json                      (local + model)
synth-1     verified texts → durations1.json                (engine)
schedule    re-solve with real durations → schedule1.json   (solver)
repeat compress/verify/synth once for residual deficits only
export      final timeline, audio, spoken-text SRT, display SRT, ledger, human queue
```

`pipeline.py run episode.srt --engine vosk --policy default.json --out DIR`
with `--until` / `--from` to stop at or resume from any stage. Every stage
reads and writes JSON only, so any stage can be replaced or inspected.

## 9. Fallbacks when nothing fits

In order: allow overrun into silence after the cue if the next spoken line
is far enough; allow tempo up to the hard ceiling (1.6) for this cue only,
recorded; keep the core-only rewrite (rung 5) flagged as lossy. A line is
never dropped entirely: the thought must survive even if only its core
unit does. What remains goes to the human queue with the original, the
best candidate, the deficit and the rungs tried.

## 10. Phases

| Phase | Deliverable | Model calls |
|---|---|---|
| 1 | Engine protocol, synth-0, oracle fitting, timeline solver, deficit report | none |
| 2 | Scenes, mergeable lines, re-scheduling with merges | none |
| 3 | Annotation and compression with the oracle loop, cache, ledger | yes |
| 4 | Unit entailment, measured re-solve, export, human queue | yes |
| 5 | Second engine adapter, rate canonicalisation report | none |

Phase 1 answers the first real question: after the solver uses shifting,
tempo and gaps globally, how many seconds actually have to come out of
the text, and in how many cues. Everything after that is sized by that
number.

## 11. Decisions to take

* Decided: speech may start 360 ms before and up to 500 ms after the subtitle start.
* Decided: comfort tempo 1.5, hard ceiling 1.6 per cue with a ledger entry.
* Decided: no line is dropped entirely; partial reduction down to the core
  thought is allowed and flagged.
* Engines will be tried and chosen by ear later; the user names the
  candidates. The shortening targets follow the chosen engine through the
  per-run oracle, so no engine-independent target rate is fixed now.

## 12. What is needed beyond the current environment

Nothing new. Checked on 9 October 2026:

* ffmpeg 6.1 is installed (tempo change with pitch preserved).
* numpy/scipy are installed (oracle regression is a least-squares fit).
* Vosk synthesis on CPU takes 1–3 s per cue; 22 cores allow parallel
  workers, so measuring five candidates per deficit cue is affordable.
* Syllables are counted as Russian vowels; scenes and mergeable lines are
  punctuation and timing rules. No speech recogniser, aligner or trained
  classifier is required.
* Semantic annotation, compression and the entailment check are ordinary
  LLM calls through the transports already in `utils/`. Tool calling is
  optional: the default mode asks for all ladder rungs in one response and
  scores them locally with the oracle.

## 13. Phase 1 status (9 October 2026, superseded by section 14 where it differs)

Implemented and unit-tested (`tests/test_voiceover_*.py`, 18 tests):

* `voiceover/text.py` — syllables (Russian vowels), words, pause marks.
* `voiceover/oracle.py` — non-negative least-squares fit of
  `seconds = a + b·syllables + c·pauses` on the run's own measurements;
  margin = 90th percentile of underestimation; syllable capacity for a
  target duration.
* `voiceover/timeline.py` — exact dynamic-programming solver on a 10 ms
  grid: start shift within [−lead, +lag], tempo factor up to the comfort
  limit (cheap) and up to the ceiling (expensive), text removal (more
  expensive), collision (fixed penalty + per second, last resort).
* `voiceover/engine.py` — `Engine` protocol and `VoskEngine`: RUNorm +
  RUAccent with per-cue fail-closed fallback, parallel synthesis workers,
  edge trim (−40 dBFS, 30 ms pad) and long-pause reduction. Identity hash
  covers model files, package version, voice, rate, normalisation and
  trim settings.
* `voiceover/pipeline.py` — `run SRT --out DIR [--until stage] [--limit N]`
  with hash-bound artifacts `cues.json`, `durations0.json`, `oracle.json`,
  `schedule0.json`, `summary.json`, `report.md`. Reruns skip synthesis
  when the source and engine identity match.

Checked on a 40-cue smoke run: synthesis is deterministic (identical frames
on repeated runs); edge trimming removes 330–450 ms of silence per cue;
the solver rejects tiny collisions once the fixed penalty exists; the
rerun skips synthesis.

Full-episode result (Australian Survivor S01E04, 1056 cues, Vosk voice 8,
base 1.25; run directory `output/v3_full_AU_S01E04`):

| Measure | Value |
|---|---:|
| Cues whose speech is longer than their own subtitle window | 686 |
| Cues that still need text removal after the solver | 250 |
| of which easy (≤20 %) / medium (≤40 %) / hard (≤60 %) | 73 / 87 / 36 |
| of which timing-limited (text cannot help) | 54 |
| Total seconds to remove | 98.4 |
| Oracle: seconds = 0.083 + 0.175·syllables + 0.223·pauses; MAE / margin | 0.33 s / 0.53 s |
| Median speech rate | 4.55 syl/s |
| Wall time: normalisation / synthesis (4 workers) / solver | 94 s / 40 min / 2 s |

Normalisation fell back to raw text on 20 cues where RUNorm changed a word
(hyphenated names, interjections); they were still measured.

## 14. Phase 2 status (9 October 2026)

### Text preparation: RUNorm replaced by rules

RUNorm (a neural normaliser) invented words on ordinary Russian text
("Ро" → "роскошь", "Сэм" → "сам", "у тебя" → "у меня", "НЕ" → "э") and
glued hyphenated names. The fail-closed guard caught 20 such cues in the
episode, but the guard had to be switched off for cues containing digits
or Latin, exactly where RUNorm's output mattered. In this episode no cue
contained digits or Latin at all, so RUNorm did no useful work.

Replaced by `rutextnorm` (rule-based, deterministic, no model, ~0.15 ms per
sentence) in `voiceover/normalize.py`, with a guard that now applies to
every cue: each Cyrillic word of the input that is not an abbreviation
candidate (followed by a dot) and not a unit after a number must reappear,
in order, in the output. Spans the rules are unsure about (bare numbers,
foreign words) are recorded per unit as `notes` for the later phases.
RUAccent stays for stress. A glued ellipsis between words
(`хотят…хотят`) is spaced before accentuation, which fixed the one
remaining accent failure. Result on the episode: 1056/1056 cues prepared
without fallback; engine texts are identical across two runs.

### Synthesis cache

`VoskEngine.synthesize_many` now caches by engine text under
`output/tts_cache/<synth-identity>/<sha(text)>.{wav,json}`. Engine identity
is split into the synthesis part (model, voice, rate, trim) and the text
preparation part; `durations0.json` is rebuilt when either changes, and
only texts never synthesised before cost time. Re-running the episode after
the normaliser change synthesised 362 texts (names are now capitalised)
instead of 1056; later reruns synthesised 1–10.

### Speech units (`voiceover/units.py`)

Consecutive cues that continue one sentence are spoken as one unit,
synthesised from the joined text and placed at the first cue's start.
Merge when the previous text ends with a comma/semicolon/colon
("comma"), a dash ("dash"), an ellipsis followed by a lowercase or
ellipsis start ("ellipsis"), or no punctuation followed by a lowercase
start ("dangling"); never across a sentence end, a dialogue dash, a gap
above 300 ms, a scene boundary (silence ≥ 1.5 s), more than 3 cues or
8 s per unit. Scenes are assigned for phase 3 context. For merged units
the schedule records an estimated start of every member cue
(syllable-proportional) and its drift from the subtitle.

### Classification fixes

* Removals of ≤ 50 ms are solver granularity, not text work: such units are
  reported as `tight` (band `none`).
* The text target is now derived from the removal amount with the unit's
  own measured seconds per syllable (rounded up), instead of an absolute
  syllable capacity that included the oracle margin. The margin made 10 ms
  deficits on short lines look "timing limited".

### Full-episode results (same episode, `output/v3_full_AU_S01E04_units`)

| Measure | Phase 1 | Phase 2, no merge | Phase 2, merge |
|---|---:|---:|---:|
| Units | 1056 | 1056 | 942 (100 merged, 114 cues absorbed) |
| Units needing text | 250 | 177 | 138 |
| easy / medium / hard | 73 / 87 / 36 | 93 / 63 / 9 | 72 / 51 / 8 |
| timing-limited | 54 | 12 | 7 |
| tight (≤ 50 ms, no text work) | – | 76 | 63 |
| Total seconds to remove | 98.4 | 95.2 | 77.9 |
| Collisions | 0 | 0 | 0 |
| Merge reasons | – | – | comma 57, dangling 48, ellipsis 6, dash 3 |
| Member drift inside merged units, p50 / p90 / max | – | – | 574 / 1331 / 2118 ms (30 members over 1 s) |
| Preparation fallbacks | 20 | 0 | 0 |

Merge candidates not merged: 16 blocked by a gap above 300 ms (ellipsis
pauses of 0.6–1.4 s), 5 by a scene gap, 4 by a dialogue dash.

Open items carried into phase 3: member drift up to 2.1 s inside long merged
units (a cap or a split rule may be needed once audio is listened to); the
remaining 7 timing-limited units are short standalone sentences in windows
under 1 s followed immediately by the next line.

### Listening draft (`voiceover/render.py`)

A draft track can be rendered from any schedule without text shortening:
cached clips are time-stretched with ffmpeg `atempo` (pitch preserved) by the
solver's `tempo_factor` and mixed at `start_ms`. Lines that still need text
simply overrun and overlap the next line, so the deficit is audible rather
than hidden. Options: episode windows (`--segments 45-120,450-540`), a soft
tick at every subtitle start (`--ticks`) to judge drift by ear, a Markdown
guide with per-line status, speed, shift, seconds to remove and member drift.
First excerpt: `output/v3_full_AU_S01E04_units/listen/draft.wav` (five
windows, 7 min, 143 units, 33 audible overlaps). Open question for the user:
whether base rate 1.25 sounds too slow (a higher base plus a higher ceiling
would be tested as a pair, since the deficit depends on the ceiling only).

### Tempo thresholds after the first listening (10 October 2026)

The user found base 1.25 dull and the 1.6 ceiling acceptable with at most
+0.1 headroom, and asked for a narrower band. New defaults: base 1.4,
comfort 1.6, ceiling 1.7 (same episode, `output/v3_full_AU_S01E04_r140`).
Units needing text 138 → 103 (easy 48 / medium 42 / hard 8 / timing-limited
5), seconds to remove 77.9 → 58.3, collisions 0, member drift p50/p90/max
517/1395/1905 ms. Observations still open: lines placed after their subtitle
(lag) are heard as "click, then speech" — 330 of 942 lines use lag, median
440 ms; the solver prices lead and lag equally, so a lag cap or an
asymmetric price is the lever. Overlaps in the draft are the unshortened
deficit lines and disappear with phase 3. Voice: sampler of all 57 speakers
in `output/v3_full_AU_S01E04_units/listen/voices.wav`.

### Voices chosen (10 October 2026)

The user picked speaker 49 (tarhova_yulia, female) and 56 (zlatoustov_ivan,
male). Vosk's `speech_rate` is voice-specific: on the same sentence at rate
1.4, speaker 8 takes 6.57 s, speaker 49 4.91 s, speaker 56 4.44 s. The tempo
thresholds the user approved (1.4 / 1.6 / 1.7) therefore describe speaker 8
only. Matched rates that reproduce speaker 8's duration: 49 at 1.05, 56 at
0.95 (thresholds scaled by the same factor). Listening excerpts with those
settings: `output/v3_excerpt_voice49/listen/` and
`output/v3_excerpt_voice56/listen/` (same five windows). On the windows the
measured rate is 5.25 syl/s for 49 and 4.86 for 56 versus 5.14 for speaker 8,
so the final rates may need a ±0.05 adjustment after listening. Policy
thresholds should eventually be expressed in syllables per second rather
than Vosk units so that they survive a voice change.

## 15. Phase 3 status (10 October 2026)

### What was built

* `voiceover/llm.py`: a minimal OpenAI-compatible chat client (DeepSeek by
  default, `deepseek-flash`), temperature 0, JSON answers, disk cache keyed by
  sha256 of (provider, model, prompt version, temperature, max_tokens,
  reasoning, messages) under `output/llm_cache/`. A rerun with unchanged inputs
  makes zero network calls. `llm.reasoning` = `off` by default: the model's
  thinking mode cost 2–3k tokens and ~35 s per short answer, hit the token cap
  on 8 of 95 units and cut more aggressively; with thinking off the whole
  stage takes ~40 s and the results are as good or better on a 12-unit
  side-by-side check.
* `voiceover/compress.py`: annotation (one call per scene with units to
  shorten: fragments with roles core/support/decor/address, refs, speech act),
  compression (one call per unit: the five cumulative rungs), local scoring
  with the oracle, hard checks and a ledger. Rung 3 of the plan ("merge with
  the neighbour") is replaced by "tighten core wording" because phase 2
  already merges units.
* Scoring: `estimated_sec = speech_sec × predict(candidate)/predict(original)`
  (the line's own measured rate scaled by the oracle ratio), target
  `speech_sec − removed_sec × tempo_factor`. First rung that fits and passes
  the checks wins; otherwise the shortest passing rung is kept as
  `unresolved` for the human queue; a unit with no passing rung is `rejected`.
* Hard checks: single line, shorter, numbers kept (digits, cardinals except
  "один", ordinal stems), no new capitalised word outside the scene, negation
  particles kept unless the dropped fragment held them. Negative pronouns
  ("ничего", "никогда") are not hard-checked.
* `lossy` = rung 5 chosen, or a rung whose reported drops include a core
  fragment.
* Pipeline stages `annotate` and `compress` (artifacts `annotations.json`,
  `candidates.json`), report section "Сокращение текста" with the ledger
  table and the human queue.
* A fix in phase 2 code: `removed_sec` is measured at the chosen tempo while
  the line's rate is measured at base, so the syllable target now scales the
  removal by `tempo_factor`.

### Full-episode result (`output/v3_full_AU_S01E04_r140`, speaker 8, 1.4/1.6/1.7)

| Measure | Value |
|---|---:|
| Units needing text | 103 (95 sent to the model, 8 timing-limited) |
| Resolved by the oracle estimate | 90 |
| Unresolved (best candidate kept, human queue) | 3 |
| Rejected (original kept, human queue) | 2 |
| Chosen rungs 1 / 2 / 3 / 4 / 5 | 20 / 34 / 24 / 9 / 6 |
| Lossy | 8 |
| Syllables removed | 479 |
| Estimated base-rate seconds saved | 97.3 (deficit was 58.3 s at tempo) |
| Model calls / tokens in / out | 95 / 70k / 30k (~40 s wall time) |

Not yet verified by measurement: the estimates are oracle ratios. Phase 4
synthesises the chosen texts, re-solves the timeline with measured durations
and sends only the residual deficit back for a second pass.

Known weak spots seen in the ledger: the model sometimes drops a repeated
cheer or an address that carries emotion ("Люблю тебя Пит" → "Люблю тебя"),
and short lines with a negation cannot lose one syllable without losing the
negation. Both are caught only by the human queue or the phase-4 entailment
check.

## 16. Tempo in syllables per second (10 October 2026)

Pulled forward from phase 5. The policy now reads `base_syl_per_sec` 5.5,
`comfort_syl_per_sec` 6.3, `ceiling_syl_per_sec` 6.7 (user-approved: "a bit
faster than speaker 8 at 1.4", same proportions as 1.4/1.6/1.7). A new
`calibrate` stage (`voiceover/calibrate.py`) synthesises a fixed 20-sentence
Russian corpus at probe rates 0.8/1.1/1.4, fits `syl/s = a·rate + b`, solves
the rate for the base target, verifies at that rate and re-solves if off by
more than 2%. The engine is then built at the calibrated rate and the
solver's comfort/ceiling totals are derived proportionally. `calibration.json`
is cached per run dir; the probe syntheses sit in the shared TTS cache. Voice
49: rate 1.036 (measured 5.50 syl/s); voice 56: rate 1.015 (5.53 syl/s).
Default voice is now 49. Cost: ~8 min per voice the first time (four engine
loads), free afterwards.

Open: calibration measures a corpus, the oracle measures the episode; the
two agree within ~5% on the excerpts (voice 56 showed more deficit than 49
at the same target, likely pause behaviour). The lag question stays as is
until the track is watched with video.

## 17. Phase 4 status (10 October 2026)

### What was built

* Four new stages after `compress`: `measure` → `durations1.json` (every rung
  that passed the hard checks is synthesised with the calibrated engine),
  `verify` → `verified.json` (one cached entailment call per measured rung:
  core fragments the rung claims to keep must still be stated; lost support
  fragments are only noted), `resolve` → `schedule1.json` (rungs are chosen by
  measured duration, lowest rung first; the timeline is re-solved; a second
  solve steps residual units down the same ladder — no re-prompting), and
  `export` → `export/` (`track.wav`, `spoken.srt`, `display.srt`,
  `ledger.json`, `human_queue.md`, `manifest.json`, `track_guide.md`).
* `voiceover/placement.py` holds the shared solve-and-enrich code used by
  `schedule` and `resolve`; `voiceover/verify.py`, `voiceover/resolve.py`,
  `voiceover/measure.py`, `voiceover/export.py` are the new modules.
* Deviations from section 6: entailment failure does not re-run the
  compressor with a locked fragment; the next rung is tried instead (the
  ladder already contains the alternatives). Rungs whose annotation has no
  core fragment are checked on their support fragments; a fragment listed as
  both kept and dropped counts as kept.
* Zero overlap is enforced at export: a unit whose deficit no text fixed is
  played at a forced tempo so its clip ends where the solver ended it; such
  units are in the human queue. `verify` may use a different model through
  `verify.llm` (merged over `llm`); default is the same DeepSeek model.
* Tests: 157 (33 new) with a fake engine and fake transport; the real engine
  and model were exercised by the full-episode run below.

### Full-episode result (after the no-core fix)

| Measure | Voice 49 | Voice 56 |
|---|---:|---:|
| Candidate clips synthesised | 267 (~10 min) | 249 |
| Entailment calls (cached on rerun) / rejected rungs | 262 / 51 | 241 / 55 |
| Units with replaced text | 79 (resolved 72, unresolved 7, rejected 7) | 71 (62 / 9 / 6) |
| Oracle estimate vs measurement, chosen rungs | 232.5 s vs 229.6 s (−1.2%) | 205.7 s vs 200.9 s (−2.3%) |
| Base-rate seconds saved by text | 83.5 | 80.0 |
| Residual after re-solve (forced tempo) | 5 units, fastest 1.75× base | 10 units, fastest 2.02× base |
| Collisions / render overlaps | 0 / 0 | 0 / 0 |
| Human queue | 18 units | 22 units |
| Track | 3365.8 s, peak 0.94 | 3365.8 s |

The first run rejected 16 rungs as "no core left"; these were annotation
artefacts (lines annotated with only support fragments, or a fragment
reported as both kept and dropped), so the rule was relaxed as described
above and the run repeated. The remaining `core_lost` verdicts read as
genuine (e.g. "думает иначе" → "не так", "чувствовал себя в безопасности" →
"был в безопасности"). Pass 2 never triggered on this episode: the residual
units had no shorter verified rung left, so they went to forced tempo and the
queue. Not yet done: listening to the exported tracks with the video; the
lag decision (section 14) is still pending that.

### Verify with a second model (11 October 2026)

The user chose voice 56 after watching the video with the track, and asked for the
entailment check to run on a model other than the compressor (OpenAI Luna or Meta
Muse Spark 1.3 Contributor). Both are reachable through the existing OpenAI-compatible
client on OpenRouter via the `verify.llm` config block (no code change); Luna is also
callable through `opencode run --format json` (ChatGPT subscription) but that would need a
subprocess transport in `voiceover/llm.py`.

Comparison run: `output/v3_full_AU_S01E04_v56_muse` (config `output/cfg56_muse.json`,
verify = `meta/muse-spark-1.3-contributor`, reasoning left to the server, ~1000 output
tokens per verdict, 7 upstream rate-limit retries over 243 calls).

| verify model | rungs rejected | agree with DeepSeek | resolved / unresolved / rejected | forced | queue |
|---|---:|---:|---|---:|---:|
| deepseek-flash | 55 / 249 | — | 62 / 9 / 6 | 10 | 22 |
| muse-spark-1.3-contributor | 92 / 249 | 202 / 249 | 58 / 12 / 7 | 12 | 27 |

Muse Spark rejects 42 rungs DeepSeek accepted; the sampled reasons are literal
("опущено 'небольшой'", "пропущено 'написано'", "это заменено на всё"), i.e. it treats
every dropped modifier as a lost core fragment. The current verify prompt does not say that
intensifiers and hedges inside a core fragment may go; that is the lever if a stricter
model is kept. Not decided yet which model is final.

Luna run (`output/v3_full_AU_S01E04_v56_luna`, config `output/cfg56_luna.json`,
verify = `openai/gpt-6-luna` on OpenRouter): with prompt v1 Luna rejected 105 / 249 rungs,
for the same literal reasons as Muse Spark. The verify prompt was rewritten as **v2**
(`voiceover/verify.py`, `VERIFY_SYSTEM`): a fragment is kept when its event and
participants survive; intensifiers, hedges, discourse words and manner details that do not
change the fact may go; a fragment is lost only when a fact changes (subject/object,
negation, condition, cause, time) or the line becomes so general the viewer reads it
differently; when in doubt, keep. The user confirmed Luna as the verify model on 11 Oct 2026.

| verify | rejected / 249 | agree with DeepSeek v1 | resolved / unresolved / rejected | forced | queue | verify cost |
|---|---:|---:|---|---:|---:|---:|
| deepseek-flash, prompt v1 | 55 | — | 62 / 9 / 6 | 10 | 22 | n/a |
| gpt-6-luna, prompt v1 | 105 | 197 | 54 / 13 / 10 | 15 | 30 | $0.024 |
| gpt-6-luna, prompt v2 | 72 | 212 | 59 / 10 / 8 | 11 | 25 | $0.027 |

Luna v2 rejections that DeepSeek accepted are mostly defensible ("двумя путями" →
"двояко", "Надеюсь, что так" → "Надеюсь"); a few are still pedantic ("нашего племени" →
"в племени"). `voiceover_config.json` now carries `verify.llm` = Luna and
`verify.prompt_version` = v2. Upstream rate limits on OpenRouter: 2–3 retries per run, harmless.

### Manual overrides (11 October 2026)

`human_queue.md` was read-only; decisions had nowhere to go. Added `voiceover/overrides.py`
and a `--overrides PATH` option on `run` (default `<out>/overrides.json` when present):

```json
{"schema": "voiceover-overrides-v1",
 "units": {"unit-138": {"text": "Она думает иначе.", "note": "optional"},
           "unit-40":  {"accept": true}}}
```

`text` is synthesised and measured by the engine (durations1 row with `rung: "manual"`,
engine key `unit#manual`), skips verify, and becomes the unit's only candidate (`status:
manual`); if it still does not fit, forced tempo applies and the unit is queued with the
reason "ручной текст не влезает" (residual > 0.05 s). `accept` keeps the pipeline's choice
and marks the unit `reviewed`, which removes it from the queue; it also works for forced
units the compressor never touched. Unknown unit ids fail loudly. The overrides hash is part
of the measure/resolve/export inputs; verify hashes only non-manual rows, so a changed
overrides file never triggers model calls. Checked on the real Luna run: measure rebuilt
(cached clips reused), verify skipped, queue 25 → 24, report line "Ручные правки
применены: ручных текстов 1, принятых единиц 1". Tests: 187.

### Human queue rewritten for people (12 October 2026)

The table in `human_queue.md` (rung marks like `3✗в`, reasons like `entailment`) was unreadable
without the code. `render_queue` now writes an intro (what the file is, the four steps, an
`overrides.json` example) and one card per queued unit: plain-Russian reason and a sentence
on what happened, original vs. what is spoken now, a time line (budget at base tempo, original
and current speech seconds, so the reviewer knows how short the own text must be), every tried
candidate with its text, length and why it was refused (the verifier's own `why` strings for
lost fragments), and ready-to-paste `accept` / `text` lines. The file ends with a complete
`overrides.json` skeleton that accepts every queued unit. Reason labels changed to
"не влезло", "отклонено проверкой смысла", "сократитель не дал вариантов", "потеря смысла",
"ускорено сверх нормы", "ручной текст не влезает". `ledger.json` `rungs_tried` items gained
`text` and `verify_notes`. Tests updated (187 pass); regenerated on the Luna run (queue 24).
