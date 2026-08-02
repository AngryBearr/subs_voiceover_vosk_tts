# Semantic benchmark60 fixture

This tracked fixture contains 60 manually reviewed semantic barrier cases: 57
clear gate cases and 3 diagnostic cases. Expected outcomes are 28 pass and 32
fail, with E01: 22, E03: 17, and E04: 21. The release gate scores clear cases;
diagnostic cases are reported separately.

The first 24 cases are the benchmark24 historical subset. The remaining 36
cases are the approved expansion: manually reviewed clear cases, balanced 18
pass and 18 fail, with reviewer audit and exact neighbor checks at E01/242,
E03/122, and E03/994. Unit construction uses full context, a maximum of three
cues, and a maximum timing gap of 0.3 seconds. The benchmark is structurally
validated and calibrated against the Sol control model. The qualified Sol
control run dated 2026-08-02 scored 57/57 clear cases and 3/3 diagnostic cases,
with 0 schema/transport failures across 17 requests. Usage was 172395 input,
2926 output, 2792 reasoning, 459264 cache-read, and 637377 total tokens. The
local result is `/tmp/opencode/model_screening/benchmark60_sol_control_run1_2026-08-02/`.

The Luna control run scored 53/57 clear cases. Its exact false rejects were
E01/114 and E01/531; uncertain outcomes were E03/220 and E04/57. It was
technically clean, but is no longer the qualified benchmark60 default. Luna
usage was 278413 input, 3079 output, 4284 reasoning, 353280 cache-read, and
639056 total tokens across 17 requests. Its local result is
`/tmp/opencode/model_screening/benchmark60_luna_control_run1_2026-08-02/`.

Benchmark60 is now structurally validated and Sol control-model calibrated;
benchmark24 is retained as historical evidence.

E01 is tracked in the repository. E03 and E04 use external context provenance;
their filenames, Drive IDs, SHA-256 values, byte sizes, and parsed cue counts
are recorded in `manifest.json`. Temporary screening paths are not fixture
provenance.

Validate the fixture with:

```bash
uv run --python subs_env/bin/python -m pytest -q tests/test_semantic_benchmark60_fixture.py
```

## Offline evaluator

The evaluator is dynamic and makes no network or model calls. Its default
manifest is benchmark60; benchmark24 remains available through an explicit
`--manifest skill_test/semantic_benchmark24/manifest.json`.

```bash
uv run --python subs_env/bin/python -m utils.evaluate_semantic_benchmark \
  --episode-report E01=path/to/E01.report.json \
  --episode-report E03=path/to/E03.report.json \
  --episode-report E04=path/to/E04.report.json \
  --episode-usage E01=path/to/E01.usage.json \
  --episode-usage E03=path/to/E03.usage.json \
  --episode-usage E04=path/to/E04.usage.json \
  --output output/semantic_benchmark60.evaluation.json
```

Exit status is 0 for a qualified benchmark, 1 for a valid but unqualified
benchmark, and 2 for invalid configuration or artifacts.
