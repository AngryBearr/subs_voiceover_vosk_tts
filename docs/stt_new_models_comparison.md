# Four-model STT experiment — billing-blocked partial result

**October 13, 2026: five POST attempts, four successful replies, no retries or billing
GETs.** Request 0077 (Grok gold) returned HTTP400: “The selected model does not support
diarize.” Cost/generation ID were absent; **unknown billing** stopped remaining slots.
Gemini is **untested**. [Summary/receipts](../temp/stt_new_models_compare/summary.json).

## Matched inputs and costs

Identical historical mono PCM16/16kHz WAVs:
7.855s blind gold mixture (`db1d5015…a500dd6`), and S46 60–90s TV/music sanity check
(`53717944…955c003`). The latter has no independently reviewed transcript or verified
simultaneous-human count. MAI/Muse baselines were **not rerun**.

Rounded USD; [raw usage](../temp/stt_new_models_compare/results/) is exact. Timing is
POST seconds, except Grok's error-call time.

| Model / gold,TV scopes | Gold USD / seconds | TV USD / seconds | Native output |
| --- | ---: | ---: | --- |
| MAI / 0006,0001 (historical) | .000222222 / 11.795 | .000833333 / 9.334 | gold 1 label; TV 2 |
| Muse / 0007,0002 (historical) | .000396 / 3.583 | .0015 / 8.930 | text only |
| Scribe-v2 / 0073,0074 | .000240014 / 3.405 | .000916667 / 1.809 | one unlabeled segment each; **3 word labels** each |
| Nova-3 / 0075,0076 | .000562942 / 1.725 | .00215 / 1.163 | gold 1 label; TV 2 labels/4 segments |
| Grok-STT-1.0 / 0077,not sent | **unknown** / .089 error call | not sent | HTTP400 unsupported `diarize` |
| Gemini-3.5-transcribe | not sent | not sent | blocked before request |

Scribe's two calls cost **$0.0011566805555553452**; Nova's **$0.0027129416666679287**.
Each received 37.855s; returned billed usage was 7.855/30s, with no token counts.
Their observed short-clip effective costs are approximately **$0.11/$0.258 per audio
hour**, not tariffs or episode forecasts. Different runs/provider loads preclude
model/GPU speed ranking. High-resolution full-job wall time is unavailable.

Known new subtotal: **$0.0038696222222232737**. Known shared subtotal:
**$0.10500917711111217**, **77 reservations**. Actual new/shared totals remain null.
The unknown request's $0.05 reservation is neither an actual charge nor an upper bound.

## Blind-three content, not label-count success

References are verified dataset-source transcripts: 1089 “STUFF IT INTO YOU HIS BELLY
COUNSELLED HIM”; 1221 “YET THESE THOUGHTS AFFECTED HESTER PRYNNE LESS WITH HOPE THAN
APPREHENSION”; 1272 “MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD
TO WELCOME HIS GOSPEL”. Common scripted utterance span is **1–4.275s**, not human
VAD/word alignment. No gold prompts/separated stems were uploaded.

[Metrics](../temp/stt_new_models_compare/metrics.json): casefold/punctuation tokens,
ordered LCS/reference length; multiset recall gives identical counts. “Full” means
contiguous normalized phrase; no Mr/Mister/spelling repair.

| Model | 1089 /8 | 1221 /11 | 1272 /17 | Complete contiguous sources |
| --- | ---: | ---: | ---: | --- |
| MAI | 0 | 11 | 0 | 1221 only |
| Muse | 3 | 11 | 4 | none; Hester sentence is interleaved |
| Scribe | 4 | 8 | 12 | none |
| Nova | 1 | 8 | 2 | none |

These are **mixed-text proxies**, not speaker accuracy: shared “his” inflates scores.
Scribe returned “Yet these thoughts left the apostle Paul … Stuff it into the septic
tank. Paul teaches the apostolical account in our classes …”. Neither “Quilter” nor
“middle classes” survived; “apostle” did. Nova returned “Yet these thoughts affected
history prem last with hope than apprehension. His gospel.” It missed the Stuff phrase
and Quilter/apostle/middle-classes content. **Neither recovered all three complete phrases.**

[Native word groups](../temp/stt_new_models_compare/native_words.json) retain Scribe's
three labels, **14 zero-duration gold words**, no cross-label native overlap: not
correct humans/separated waveforms. Segment-first exports stay unchanged. Nova TV's
30.085s endpoint was explicitly clipped by 85ms. Both new TV replies include “and I'm
ready for it”; Scribe adds “[cheering] That's it, Siga!”—unreviewed. No WER/cpWER/DER.

## Preserved evidence and whole-file MAI

Snapshot/[checks](../temp/stt_new_models_compare/checks.json): full72 prefix, 613 old
files and inputs unchanged; snapshot0444/active0600. Guarded resume: zero calls/writes.
Eight-result criterion **unmet**;
0077 needs authoritative billing reconciliation before anything further.

The October 13 [OpenRouter STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt)
describes synchronous JSON and a 60s **processing** deadline, not a categorical
60s audio limit. A single 56-minute MAI request is not guaranteed through that route;
compression does not remove the deadline. Direct Azure requires separate credentials
and verified diarization-duration constraints. One request scope alone cannot guarantee
speaker consistency. No new whole-episode/Azure upload was performed.

## Resume criterion

Manually check [OpenRouter Logs](https://openrouter.ai/docs/guides/features/logs) via https://openrouter.ai/logs for `x-ai/grok-stt-1.0`, October 13, 2026 **09:32:47–09:32:55 UTC** (request 0077).
Provide the matching actual generation ID and per-request cost, **never an API key**; only an actual ID enables [generation metadata retrieval](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation).
A missing log or validation HTTP400 does **not** prove zero cost; `/activity` aggregates and `/key` totals cannot reconcile this request. If evidence is unavailable, support@openrouter.ai can confirm billing using the safe error/time/model; no automatic ledger update or further experiment is permitted while billing is unknown.
