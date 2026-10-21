# OpenRouter STT: final bounded comparison

**Durable evidence:** [workspace snapshot](../temp/stt_api_results/README.md)
(`temp/stt_api_results/`, ignored by Git), with listening audio, replies and captions.
The [copy manifest](../temp/stt_api_results/bundle_manifest.json) maps original run
paths to byte-identical bundle files; the canonical `/tmp` evidence paths below are
retained as historical provenance, not a requirement for inspecting this snapshot.

This page records the **historical 15-reservation comparison**. A separately authorized
[full AU e04 MAI direct follow-up](../temp/australian_survivor_e04_mai/README.md) then
completed 57 new minute chunks / 660 native cues. Episode cost $0.09344444444444445;
the **same active ledger** now totals $0.1011395548888889 / 72 reservations, all known,
under an explicit episode cap 72 (default remains 18). The old snapshot/tables below
were not rewritten or recopied; diagnostic/automatic-separation limitations remain.

**MAI returned native caption timings; Muse returned text only.** Neither direct
model recovered all three voices from the controlled mixture. A separate,
**annotation-assisted SepFormer3 diagnostic** recovered three profile-matched stems:
both APIs then transcribed the complete Quilter/apostle/middle-classes phrase.
This does **not** repair the default production result: PixIT missed a voice and
never triggered SepFormer on that fixture. No universal model winner is established.

## Results first

| Test / route | Wall seconds | Actual USD | Observed result |
| --- | ---: | ---: | --- |
| S46 60–90s: MAI / Muse direct | POST 9.334 / 8.930 | 0.0008333333333333334 / 0.0015 | MAI 2 timed cues / 2 labels; Muse text only |
| AU e04 60–90s: MAI / Muse direct | POST 14.333 / 7.840 | 0.0008333333333333334 / 0.0015 | MAI 5 timed cues / 3 labels; Muse text only |
| New 7.855s gold mixture: MAI / Muse direct | POST 11.795 / 3.583 | 0.00022222222222222223 / 0.000396 | MAI one phrase/label; Muse interleaved fragments; neither full third phrase |
| Gold fixture: default native fast | Full 204.12 (log rounded) | 0 | 3 global sources but source 00 silent in overlap; no triple window / SepFormer invocation |
| New S46 production MAI hybrid | Full 120.056; POST sum 19.811 | 0.0003888888888888889 | 2 real-source chunks / 2 timed cues; loses “And I'm ready for it” |
| Known-span SepFormer3 diagnostic | Preparation 10.166; forward 5.037 | 0 | One blind forward, ECAPA association passed; annotation-assisted, not default production |
| Diagnostic stems × MAI/Muse | 6 POSTs, sum 28.144 | 0.0018546666666666666 | Complete source-specific texts; MAI 3 timed exports; Muse text only; some leakage remains |

**Historical total $0.007695110444444445; 15 reservations, 14 returned responses, 1 lost response
with known cost.** The $1/18 cap held at that checkpoint, with three slots then remaining. The
interrupted fifth attempt was reconciled, not retried. Only six new stem POSTs were
made in the diagnostic follow-up; no new GET, raw-mixture/episode replay or full
fixture model rerun. Actual cost is returned usage/generation evidence, not estimates.

## Three-source diagnostic: fixed identity and full-gold scores

The same saved 7.855s PCM mixture was blindly separated once with cached, pinned
`speechbrain/sepformer-wsj03mix` revision `21e78b44a378092c34fff32ba8e788a73584f474`.
The fixture's scripted annotation supplied source count 3 and known overlap span;
**no predicted turns were fabricated, production gating patched or thresholds tuned**.
Existing helpers performed 16→8→16kHz resampling to exactly 125680 samples and
joint least squares against the original mixture to restore gain/polarity. Raw-channel
gains were `[-0.05395175, 0.05714540, 0.05715570]`, rank 3. No isolated gold waveform
was used in separation or identity matching.

Pinned ECAPA matched separated-core embeddings to actual clean-original prefix/suffix
profiles. Prefix and suffix repeat the same utterance: profile totals below are not
independent new speech. Association was frozen **before ASR**, never chosen by gold
text, ASR hypotheses or overlap reference correlation.

| Global source / dataset speaker | Unique utterance / profile total seconds | SepFormer channel | Assigned cosine | Post-binding correlation / SI-SDR |
| --- | ---: | ---: | ---: | ---: |
| SPEAKER_00 / 1089 | 3.275 / 6.550 | 2 | 0.561521 | 0.931527 / 8.170dB |
| SPEAKER_01 / 1221 | 4.825 / 9.650 | 0 | 0.617978 | 0.870438 / 4.951dB |
| SPEAKER_02 / 1272 | 5.855 / 11.710 | 1 | 0.489540 | 0.911602 / 6.918dB |

Existing guards passed: assigned cosines ≥0.45; best-vs-runner-up **mean permutation
margin 0.307836 ≥0.08**; maximum off-diagonal source-embedding similarity
**0.259627 <0.90**. Correlation/SI-SDR are subsequent fixed-association diagnostics,
not identity evidence used to select the permutation. FLOAT peaks were
0.870300/0.566282/0.400926; all PCM transport gains were 1.0, without clipping.

Each model and matched local control received the same saved mono PCM16/16kHz bytes
for that source, containing its entire expected utterance once in the 7.855s crop.
Parakeet/Whisper each loaded once for **only these three new stems** (4.062s/18.913s
total per preset). No old clean/raw control was repeated or substituted for a stem.

### End-to-end diagnostic WER / CER

Cells show **word edits/reference words; character edits/reference characters**.
This scores separation omissions/leakage plus ASR against complete human dataset
text, not isolated ASR accuracy or arbitrary raw-mixture word ordering.

| Exact input | Parakeet fast | Whisper accurate | MAI | Muse |
| --- | ---: | ---: | ---: | ---: |
| 1089-134686-0001 | 6/8 (75%); 28/42 (66.67%) | 4/8 (50%); 17/42 (40.48%) | 1/8 (12.5%); 1/42 (2.38%) | 1/8 (12.5%); 7/42 (16.67%) |
| 1221-135766-0002 | 0/11; 0/74 | 0/11; 0/74 | 0/11; 0/74 | 0/11; 0/74 |
| 1272-128104-0000 | 1/17 (5.88%); 4/89 (4.49%) | 1/17 (5.88%); 4/89 (4.49%) | 1/17 (5.88%); 4/89 (4.49%) | 1/17 (5.88%); 4/89 (4.49%) |

- Source 00 gold: **STUFF IT INTO YOU HIS BELLY COUNSELLED HIM**. MAI returned
  `"Stuff it into you," his belly counseled him.` (one spelling substitution).
  Muse returned `stuff it into you, his belly counselled him. gospel` (one insertion).
  Parakeet added `Glad to welcome his gospel.`; Whisper added `Glad, it's gospel.`:
  residual third-source leakage is real, not concealed by scoring or hand repair.
- Source 01: all four routes returned the full Hester Prynne sentence exactly after
  normalization, with no word/character edits.
- Source 02: both APIs returned **“Mr. Quilter is the apostle of the middle classes
  … we are glad to welcome his gospel.”** with all phrase content, unlike the direct
  mixture replies. All four routes use “Mr.” versus dataset **MISTER**: counted as
  one word/four character edits, not custom-normalized away.

Standard Levenshtein uses Unicode casefold, punctuation-separated word tokens and
collapsed whitespace; CER includes normalized spaces. No Mr→Mister/name repair.
MAI's native bounds were exported with crop offset **15.955s exactly once** and
ECAPA-bound global IDs, not cloud labels (third-source cue: **17.555–22.435s**).
Muse returned no times/labels: text/provenance
were saved, **no fabricated Muse SRT/subtitle JSON**. Gender metadata only reuses
local clean-original acoustic estimates, not API truth. These tiny measurements
support a separation-vs-direct comparison here, not a global model ranking.

| New request / source | MAI POST seconds / USD | Muse POST seconds / USD |
| --- | ---: | ---: |
| 0010/0011 / SPEAKER_00 | 13.496 / 0.00022222222222222223 | 3.911 / 0.000396 |
| 0012/0013 / SPEAKER_01 | 1.668 / 0.00022222222222222223 | 2.654 / 0.000396 |
| 0014/0015 / SPEAKER_02 | 3.550 / 0.00022222222222222223 | 2.865 / 0.000396 |

Each upload is 7.855s; returned billed usage is 8s MAI / 7.92s Muse. Six costs were
recorded once in the original shared root ledger, with no unknown cost or paid retry.

## Exact fixture and the automatic negative case

Three existing HF-row FLAC SHA/sample counts and distinct dataset speaker annotations
were verified. Complete references are human dataset fields, not ASR; no independent
original OpenSLR transcript cross-check was performed.

| Dataset speaker / exact input | Duration | Complete human dataset reference |
| --- | ---: | --- |
| 1089 / 1089-134686-0001 | 3.275s | STUFF IT INTO YOU HIS BELLY COUNSELLED HIM |
| 1221 / 1221-135766-0002 | 4.825s | YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN APPREHENSION |
| 1272 / 1272-128104-0000 | 5.855s | MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL |

Existing gain-only mix: RMS 0.08 and global attenuation 0.8567566374. Full clean
prefixes start 0/4.275/10.100s; overlap utterances start together at 16.955s; suffixes
start 23.810/28.085/33.910s; full fixture 39.765s. Each complete utterance occurs
once in overlap; profile repetitions are deliberate, not stretched/padded overlap.
Upload/diagnostic context **15.955–23.810s**, true common three-source span
**16.955–20.230s**. The full fixture was not uploaded or rerun for this diagnostic.
Input PCM SHA256: `db1d501589b090f763b0b9afd3a4810b69307d27043cfd77fb2ee1907a500dd6`.

| Previously completed direct raw-mixture route | Actual text |
| --- | --- |
| Parakeet | Yet these thoughts affected Hester Prince less with hope to welcome his gospel. |
| Whisper / MAI | Yet these thoughts affected Hester Prynne less with hope than apprehension. |
| Muse | Yet these thoughts Stuff it in affected Hester Prynne less with hope than apprehension. glad welcome his gospel. |

MAI returned one label/segment at crop 1.44–5.52s, original **17.395–21.475s**.
It intersects the true triple span but does not prove three voices. Muse had no
timing, so its fragments cannot be temporally attributed. Neither full third phrase
was recovered directly; no mixture WER or best-text speaker assignment was computed.

The earlier real default native run had 12 cues / 3 global sources `[0,2,1]`, clean
prefix binding passed, but source 00's overlap waveform was **silent**. PixIT missed
triple activity: no automatic SepFormer invocation, other stems had altered text.
The six default-route cloud stems were initially skipped. The new six calls above
use a **separate explicitly annotated route**; default three-voice acceptance remains
**unmet**, with no parameter/model search, synthetic turns or production force flag.
Earlier isolated clean controls scored 0 for sources 00/01 and 1/17 WER, 4/89 CER
for source 02 on both presets. Those clean scores are separate from stem scores.

## Episode excerpts and real S46 production hybrid

Original excerpts are 60–90s. No reviewed human episode gold: these are transcript
disagreements, **not episode WER/accuracy percentages**. Original four calls and
matched raw-mixture local controls remain unchanged, with matching PCM inputs.

| S46 route | Actual text |
| --- | --- |
| Parakeet | Is about. But it is in that moment right there, makes the game so worth playing. And I'm ready for it. I've applied for Survivor the day I become a citizen of the United States. I came out of the courthouse |
| Whisper | is about but it is in that moment right there makes the game so worth playing and I'm ready for it I've applied for Survivor the day I've become a citizen of United I came out of the courthouse |
| MAI direct | Is about. But it is in that moment right there makes the game so worth playing. And I'm ready for it. I've applied for Survivor the day I become a citizen of the United States of America. I came out of the courthouse. |
| Muse direct | is about But it is in that moment right there makes the game so worth playing. And I'm ready for it. I applied for survivor the day I become a citizen of United States of America I came out of the courthouse |

MAI/Muse retained “States of America” beyond local controls; “I've applied” versus
“I applied” remains unverified. MAI original direct bounds: 60.28–70.6 / 84.36–90s.

| AU e04 route | Actual text |
| --- | --- |
| Parakeet | Working her alliances to get Evan voted out of the game. The tribe is spoken.21 are left. Who will be going home tonight? So tonight |
| Whisper | Working her alliances to get Evan voted out of the game. The tribe has spoken. 21 are left. Who will be going home tonight? So tonight |
| MAI direct | Amazing. Working her alliances to get Evan voted out of the game. The tribe has spoken. 21 are left. Who will be going home tonight? So tonight. |
| Muse direct | Anything. Working her alliances to get Evan voted out of the game. The tribe has spoken. 21 are left. Who will be going home tonight? Tonight |

Opening “Amazing”/“Anything”/omission and “is/has spoken” are disagreements. MAI's
three labels do not prove simultaneous three-voice recovery or persistent identity.

The previously completed public S46 `run_pipeline(preset="api", api_mode="hybrid")`
used real local PixIT/gender and two MAI chunks, no direct/AU reply substitution.
Raw columns `[1,0]`; two valid integer-ms/list-text/SRT-parity cues. Source 00:
“Is about. But it is in that moment right there makes the game so worth playing.”
Source 01: “I've applied for Survivor the day I become a citizen of United States
of America. I came out of the courthouse.” Local gender male/unknown came from clean
original speech. **“And I'm ready for it” was lost before cloud ASR**: production
execution/export acceptance passed, not better coverage or accuracy.

## Billing reconciliation and artifacts

User generation `gen-stt-1791790097-52pr3WCIRKPTNUmENWO4` resolved fifth cost
**$0.000166666**, MAI dated version `microsoft/mai-transcribe-2-20260903`, Azure/STT,
created **2026-10-12 07:28:17.020 UTC**, consistent with the original attempt.
Original 5.352s duration is reservation evidence, not an absent remote duration field.
Only the pending entry was reconciled under lock, preserving lost-response history;
**no transcript/SRT was recovered or retried**. The outer 120s interruption did not
prove provider timeout. Aggregate self-key usage was not used to infer cost.

### Durable workspace copies

The curated snapshot contains **133 copied files / 30,868,657 bytes**, plus its
README and manifest. Every copied SHA256/byte count matched; all 70 copied JSON files
parsed. It excludes original movies, model weights, credentials, request base64,
session traces/OpenCode schemas and the unverified published WAV fixture family.
Only exact HF-row FLACs/metadata are copied under the partial `triple_restore/`.

- Episode listening: [S46 PCM](../temp/stt_api_results/season46.pcm16.wav),
  [AU PCM](../temp/stt_api_results/australia.pcm16.wav).
- Diagnostic listening: [source 00](../temp/stt_api_results/triple_gold_sepformer_diagnostic/SPEAKER_00.pcm16.wav),
  [source 01](../temp/stt_api_results/triple_gold_sepformer_diagnostic/SPEAKER_01.pcm16.wav),
  [source 02](../temp/stt_api_results/triple_gold_sepformer_diagnostic/SPEAKER_02.pcm16.wav).
  Matching FLOAT stems and complete profile/matching/metric evidence are alongside.
- SRT examples: [S46 MAI direct](../temp/stt_api_results/season46_mai_direct.srt),
  [S46 production hybrid](../temp/stt_api_results/season46_mai_production_hybrid_new.srt),
  [third diagnostic source](../temp/stt_api_results/diagnostic_sepformer_SPEAKER_02_mai.srt).
- [Human-gold recipe](../temp/stt_api_results/triple_gold/recipe.json),
  [diagnostic metrics](../temp/stt_api_results/triple_gold_sepformer_diagnostic/evaluation.json),
  [acceptance summary](../temp/stt_api_results/acceptance_summary.json).

Evidence JSON retains its **original absolute paths**; consult the manifest instead
of interpreting them as new runs at the copied location. The ledger copy is a
**0444 read-only historical snapshot**, not a new/active budget: do not use this
bundle as an API output directory. The authoritative task ledger was unchanged by
copying; no HTTP/model/ASR call or benchmark rerun was made.

### Original canonical experiment paths

Original artifact root: `/tmp/opencode/stt_api_compare/`:
- `triple_gold_sepformer_diagnostic/`: `matching.json`, `evaluation.json`, local
  controls, raw channels and listening stems **`SPEAKER_00/01/02.float.wav`** and
  **`SPEAKER_00/01/02.pcm16.wav`**; full per-source PCM SHA256/profile matrices there.
- `diagnostic_sepformer_SPEAKER_{00,01,02}_{mai,muse}.*`: sanitized native replies,
  texts, three MAI JSON/SRT/speaker exports; no Muse subtitle fabrication.
- `triple_gold/`, `gold_mixture_{mai,muse}.*`, `season46_mai_production_hybrid_new.*`:
  preserved exact recipe/gold/native failure, direct replies and production exports.
- `.stt_api_budget.json`, `acceptance_summary.json`, `diagnostic_checks.json`,
  `finish_checks.json`, `request_0005.reconciliation.json`: shared actual accounting,
  current results/checks and explicitly scoped historical checks.

Older `triple_restore/` published-input artifacts remain archived: first icefall WAV
is 6.625s, conflicting with the 3.275s exact HF-row FLAC. It is not this new validated
fixture or evidence of canonical human identity; see `source_assets_manifest.json`.

Saved waveform/hash/usage/identity/schema/SRT/offset/security/original-media/budget
checks, metric self-check, compile and whitespace passed. Production/deps unchanged;
26 API mocks / 15 pure tests and completed model benchmarks were not repeated.
Only private workspace `.env` was used, no key/hash/headers/input base64 persisted;
denied external credentials, management endpoints, commits/push were not used.
API fields describe observed calls only: MAI structure is useful for caption exports;
Muse is useful text here, no unsupported claim about every possible Muse mode.
See [API contract](stt_api.md) for existing client behavior.
