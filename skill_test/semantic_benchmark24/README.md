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
