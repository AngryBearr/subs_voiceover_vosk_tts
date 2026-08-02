# Semantic benchmark24 fixture

This directory contains the six original/candidate episode arrays and the
manifest for 24 manually adjudicated semantic barrier cases. The benchmark has
21 clear cases and 3 diagnostic debatable cases. `clear` is the release-gate
subset; debatable cases are retained for diagnosis and are not silently treated
as failures.

The arrays are grouped by episode (E01: 10, E03: 5, E04: 9). Unit-level
screening requires the full episode context because adjacent cues and timing
continuation are evidence for conservative grouping. E01 context is tracked at
`skill_test/S01_E01_ru_analyzed.json`.

E03 and E04 context files are external provenance only and are intentionally not
required in this repository. Their exact source metadata is in `manifest.json`:
file name, Drive file ID, byte size, SHA-256, and parsed cue count. The source
files were evaluated from Drive; the dated raw screening outputs under
`/tmp/opencode/model_screening/` are ephemeral experiment artifacts and are not
part of the fixture.

The manifest retains the E01 index 128 revision history: it changed from clear
to debatable after disagreement between cue-level and unit-aware screening.
Run `uv run --python subs_env/bin/python -m pytest -q
tests/test_semantic_benchmark24_fixture.py` to validate the fixture.

## Offline evaluator

The tracked evaluator scores supplied report and usage JSON only; it makes no
network or model calls and does not inspect run directories. Each episode is
passed explicitly and assignments are `EPISODE=PATH`:

```bash
uv run --python subs_env/bin/python -m utils.evaluate_semantic_benchmark \
  --manifest skill_test/semantic_benchmark24/manifest.json \
  --episode-report E01=path/to/E01.report.json \
  --episode-report E03=path/to/E03.report.json \
  --episode-report E04=path/to/E04.report.json \
  --episode-usage E01=path/to/E01.usage.json \
  --episode-usage E03=path/to/E03.usage.json \
  --episode-usage E04=path/to/E04.usage.json \
  --output output/semantic_benchmark24.evaluation.json
```

Exit status is 0 for a qualified benchmark, 1 for a valid but unqualified
benchmark, and 2 for invalid configuration or artifacts. A valid evaluation
always writes one newline-delimited JSON object. `clear` cases are the release
gate; diagnostic cases are reported separately and never block qualification.
