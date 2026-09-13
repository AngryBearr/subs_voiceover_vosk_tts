# Historical shortening model pairs

**Status:** factual artifact review only. No model, TTS, network, or paid run was
started for this note. The proposed six-pair matrix is separate and remains
`DRAFT_NOT_EXECUTED` in [`sh08_model_matrix_proposal.json`](sh08_model_matrix_proposal.json).

## What actually ran

This note records role evidence found in the audited artifacts, not an
ambiguous quality ranking.

The audited artifacts show two executed generator-to-arbiter paths:

| Generator family | Found generator variants | Arbiter actually used | Historical scope and result convention |
|---|---|---|---|
| Luna | `none`, `low`, `medium`, `high` | OpenCode Sol, `medium` | `none` and `high`: six targets, two repeats each; `low` and `medium`: the common 48 targets, two repeats. |
| DeepSeek Flash | `off`, `low`, `high`, `max` | OpenCode Sol, `medium` | The six-target benchmark and the `deepseek_limits_v2` run have two repeats per variant. Expanded 48-target artifacts exist for `off` and `low`; the other two variants are six-target artifacts. |

The six-pair status below answers the historical question without turning
artifact presence into a quality ranking. “Not found” is scoped to the audited
artifacts; it is not a claim that the pair never existed.

| Generator family | Arbiter family | Audited artifact status |
|---|---|---|
| Luna | Luna | not found |
| Luna | DeepSeek | not found |
| Luna | Sol | actual |
| DeepSeek | Luna | not found |
| DeepSeek | DeepSeek | not found |
| DeepSeek | Sol | actual |

For expanded comparison, the common 48 cases are the same cases in both
repetitions, not 96 independent targets. Historical pass counts therefore must
not be used as a direct new-model ranking: protocol versions, calibration and
length basis changed, and the common cases were replayed. Luna and Sol are in
the same provider family; a fresh session does not make them independent
families.

## What is not evidence for an additional arbiter pair

- Luna and Flash appearing as separate auditors are auditor roles, not the
  generator/arbiter pair being measured here.
- The SH03 same-model CLI capability and offline fixtures are capability checks,
  not historical model runs.
- The v1 context audit is invalid for this comparison and is excluded.
- The four other combinations in the proposed 2×3 matrix were not found in the
  audited artifacts. That is a search result, not a claim that they never
  existed.
- The old protocol's 13-character-per-second cap and the new normalized
  calibrated profile are not interchangeable; a count from one cannot be
  promoted to a cross-protocol model rank.

## Primary artifact references

These are the compact summaries used for the matrix; raw responses are not
copied here.

| Artifact | SHA-256 |
|---|---|
| `output/shortening_verification_20260910/final_validation.json` | `ebead6060f9b7d60e5cb22cea6c80c2a1a1cfa21e710be504da1c82e85898f75` |
| `output/shortening_verification_20260910/expanded_comparison_final.json` | `ccdc655d82f6eef0e5e760f498bcf24b2644a4807df6290092972985a8ffd0c7` |
| `output/shortening_benchmark_luna_20260909/experiment_summary.json` | `f049609126e8f319f808df058a183d7a627653c4cde68da876825c2663ea9bea` |
| `output/shortening_benchmark_flash_20260909/experiment_summary.json` | `d0e8d83de0dcc8aaba8648a857658b7a8a78a6fadde9c4748231857dd98b098a` |
| `output/shortening_expanded_luna_low_20260910/experiment_summary.json` | `aa6da7f0ebe4bbe6d4b1cfc9e0ea1b56b50f8c61ad14342d419e5437e581b191` |
| `output/shortening_expanded_flash_low_20260910/experiment_summary.json` | `b2d89dab990c1fb26350907962fb4a78178db90d83861760c9df69519812cabe` |
| `output/shortening_expanded_luna_medium_20260910_run2/experiment_summary.json` | `efaac7421a6126b744a987507e5d44aa075757003cbdc194d8d5275d397cdc89` |
| `output/shortening_deepseek_limits_v2_20260910_run2/experiment_summary.json` | `053e0990c43cb5415563c5d3028c64934b693a589a93f0bc94cb09d3d1aed0d6` |

The source-selection contract remains bound to
`docs/sh08_pilot_manifest.json` (SHA-256
`dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`). This
document does not alter that manifest.
