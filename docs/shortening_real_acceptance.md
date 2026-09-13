# SH-05/SH-11 real offline acceptance

**Status: complete for the real offline technical gate (13 September 2026).**

This is durable evidence for the SH-05 review queue and SH-11 per-subtitle
outcome/reanalysis invariants on a frozen real corpus. It is a separate gate
from unit tests, synthetic fixtures, and any paid pilot. The authoritative
record is `output/sh05_sh11_real_acceptance_20260913_r12/report.json`; its
verification log is
`output/sh05_sh11_real_acceptance_20260913_r12/verification.log`.

This acceptance does **not** assess independent meaning, semantic quality, TTS,
or audio quality. It does not complete SH-08, and no paid pilot was authorized.
H4 remains a future SH-06 presentation experiment; BUG-02 and BUG-03 remain
deferred requirements and were not changed by this record.

## Reproduction

From the repository root, with the project environment activated:

```bash
source subs_env/bin/activate
SUBS_REAL_ACCEPTANCE=1 \
SUBS_REAL_ACCEPTANCE_OUTPUT="$PWD/output/sh05_sh11_real_acceptance_rerun" \
uv run --active --no-sync -m pytest -q tests/test_shortening_real_acceptance.py -s
```

The default corpus root and the optional override are defined by
`tests/shortening_real_acceptance_manifest.json`:

- default: `/home/vladimir-barbuta/Downloads/subs_found_for_testing`;
- override: `SUBS_REAL_CORPUS_ROOT`.

The manifest freezes the four source hashes, the Vosk profile, model metadata,
and historical-report hash. A missing source or any hash mismatch fails the
opt-in acceptance. Without `SUBS_REAL_ACCEPTANCE=1`, the standard suite skips
the real-corpus tests. The output directory must be fresh; the large output
artifacts are ignored and are not committed. An existing requested name is
suffixed by the harness; each actual output directory is fresh. The test source
and manifest are the tracked reproducibility inputs. No cleanup was performed.

The current harness SHA is
`a6c03e31f61b03c31ac0647424d806136408361b484c7d8f4abdae3361007fd6` and the
current manifest SHA is
`b440dc06ee706fab6e43579127ed22f7d797935b74b391dc05bd4eac2cc49aff`.
Both hashes are captured in `report.json` and are checked by the harness.

## Corpus and gate result

All four files passed all 15 required phases. Source totals are **4,419**
cues across all four files and **3,525** across the first three (not 3,535).
The baseline queue is the unresolved full-file taxonomy; it is not a manual or
LLM fraction.

| Source | Source cues | Selected | Queue | Valid zero-character budgets | Invalid zero-character budgets |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Australian Survivor S01E03_ru.srt` | 1,523 | 1,219 | 1,219 | 344 | 0 |
| `S01E02.srt` | 1,007 | 773 | 773 | 248 | 0 |
| `Survivor.S44E07.srt` | 995 | 859 | 861 | 178 | 2 (`cue-70`, `cue-356`) |
| `Survivor.S44E11.srt` | 894 | 710 | 711 | 136 | 1 (`cue-275`) |
| **Total** | **4,419** | 3,561 | 3,564 | **906** | **3** |

The first three files contain **770** valid zero-character budgets. SH-08 must
define an explicit policy for them while retaining them in the denominator;
they must not be excluded to improve a result. The three invalid cases are
source timing cases. Their invalid zero-length timings are retained by actual
SRT serialization with source-level warnings. All four product export-error
fields (`export_error` and `product_report_export_error`) are `null` for every
source; these warnings are not manufactured product export failures.

The 15 phases per file were:

`prepare_srt`, `build_baseline_state`, `write_baseline_bundle`,
`load_baseline_bundle`, `prepare_json`, `json_bundle`,
`apply_simulated_fixture`, `apply_invariants`, `reanalyze_cached`,
`reanalyze_forced`, `neighbor_text_fault`, `neighbor_timing_fault`,
`negative_apply`, `output_refusal_and_lock`, and `source_profile_mismatch`.

The full runner also processed the real 995-cue S44E07 file with mock
transport: 12 fixture requests on the fresh run and 0 fixture requests on
resume. It produced 995 outcomes, retained both invalid cues, and recorded 0
paid requests, 0 network requests, and cost 0. This is mock transport, not an
actual LLM or an SH-08 pilot.

## What was verified

- SRT preparation and conversion to UTF-8-BOM JSON were exercised for every
  source; SRT/JSON counts remained equal.
- Cue count, text, timing, source index, unknown fields, and list-valued text
  fields were preserved except for the one explicit mechanically generated
  fixture edit per file. Source files, model/profile files, and the historical
  report were unchanged.
- Cached and forced offline reanalysis, neighbor text/timing invalidation,
  negative apply cases, output refusal, locking, and source/profile mismatch
  were covered. No LLM, synthesis, network, or audio measurement ran.
- The opt-in module contains seven tests: five real-corpus tests (four files
  plus the full runner) and two helper/guard checks. The companion standard
  verification recorded 471 passed and 5 skipped; compileall passed. The
  authoritative opt-in log records `7 passed in 533.64s` and exit 0.
- Four historical candidates were exact source matches in the S44E11 anchor,
  but all four were rejected by the current cap. No historical semantic proof
  was inherited.

The apply edit is explicitly test-only mechanical provenance
(`SIMULATED_HUMAN_ACTION_FIXTURE`): it is not user approval and is not semantic
validation. Fixture outcomes and any manual-fraction or “100%” baseline must
not be presented as a quality result.

Reports from r1-r11 were harness-development/superseded records, not
acceptance authority; r12 is authoritative. An earlier fabricated
`export_error` harness label was removed from the final record and did not
mutate product reports. If SH-08 later receives separate authorization, any
local-TTS follow-up must use its own task and output and must not be conflated
with this acceptance.
