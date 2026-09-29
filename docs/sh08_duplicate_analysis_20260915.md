# SH08 saved-response duplicate analysis

Analysis time: `2026-09-15T17:07:06.087870+00:00` (UTC)

This is a read-only scan of 60 selected cases and 88 saved ledger requests. It
made no model, network, or TTS calls. The machine-readable evidence is in
`output/sh08_duplicate_analysis_20260915/report.json`.

## Source checks

| Source | SHA-256 |
| --- | --- |
| `case_results.json` | `6f326d705d858d8f80ced49a2529f2acac9accd05ff899b3c8f8e7baddba7900` |
| `ledger.json` | `9e8b19640ca2b484f7a8fb4740460e707925c4bb8fc3d23b0fc427f96d629a8a` |
| `selection.json` | `818a88a11946c54066c16258f78a6a9577ac3de634d30b715e8cd4980dfe5d0c2` |

The `case_results` and `ledger` envelopes were checksum-verified, and their
ledger checksums match. The selection hash is recorded in JSON. The declared
manifest checksum is recorded but not independently verified because this
analysis intentionally reads only the three authoritative input files.

## Independent generation repetitions

The denominator is the intersection of cases with an actual parsed `ok`
generation in both rep 1 and rep 2 for the same generator. Whitespace-only
normalization is exactly `' '.join(text.split())`; it does not case-fold.

| Generator | Requests / target attempts | Unavailable cases (rep1 / rep2) | Parsed `ok` (rep1 / rep2) | Intersection | Exact duplicates | Whitespace-normalized duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek | 6 / 82 | 19 / 19 | 41 / 41 | 41 | 34 | 34 |
| gemini | 6 / 82 | 19 / 19 | 28 / 21 | 21 | 8 | 8 |
| luna | 6 / 82 | 19 / 19 | 32 / 32 | 29 | 8 | 8 |
| muse | 6 / 82 | 19 / 19 | 34 / 30 | 30 | 9 | 9 |
| **total** | **24 / 328** | — | — | **121** | **59** | **59** |

The 59 equal pairs are benchmark evidence from independent repetitions, not
avoidable calls or proven savings. Cases without a usable parsed `ok` result
were excluded from the pair denominator; `cannot_shorten` counts are retained
explicitly in JSON (gemini 33, luna 18, muse 18, deepseek 0).

## Repairs

Repair counts use the completed ledger request target payloads and parsed saved
responses. They do not infer missing rows from nested `case_results` repair
objects. “Comparable” means a parsed `ok` repair row had a `prior_candidate` in
the repair request target payload.

| Generator / arbiter | Requests | Attempted targets | Parse errors | Cannot shorten | Comparable `ok` | Same prior exact | Same prior whitespace-normalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek / deepseek | 2 | 19 | 0 | 12 | 7 | 1 | 1 |
| deepseek / luna | 2 | 20 | 0 | 12 | 8 | 1 | 1 |
| deepseek / sol | 1 | 3 | 0 | 0 | 3 | 0 | 0 |
| gemini / sol | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| luna / deepseek | 1 | 2 | 0 | 0 | 2 | 0 | 0 |
| luna / luna | 1 | 2 | 0 | 0 | 2 | 0 | 0 |
| luna / sol | 1 | 2 | 0 | 1 | 1 | 0 | 0 |
| muse / sol | 1 | 2 | 0 | 0 | 2 | 0 | 0 |

Two exact same-as-prior repair outputs were found:

- `b2d1135c9143:cue-1038`, deepseek / deepseek, receipt
  `23044d65a3eb9dbeaea2c48ca5259c0be81759d9edb2c56329a4dbb794a6f1ed`;
- `2c58c71203c3:cue-210`, deepseek / luna, receipt
  `fe319ba6cfab5dde03230a17a201657aca020d64496424fbcdace1ae16e2000c`.

No recheck target was sent for either duplicate repair. Therefore no recheck
target or whole request is proven avoidable. No call-saving total is asserted.

## Limits and caution

Avoidability requires the same repetition, generator, arbiter, original,
context, limit, `length_basis`, model spec, system prompt, schema, and the same
other-target scope; only the candidate may differ. Different scope or arbiter
is not treated as zero or as proven saved. The historic v2 prompts and the new
duplicate-analysis rule were not tested by inference. No raw prompts,
responses, or secrets are copied into this document.
