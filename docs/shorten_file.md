# `shorten_file`: ordinary full-file shortening

SH-03 adds a single-model command for a complete `.json` or `.srt` file. It
uses a frozen Vosk calibration profile and the resilient generation/review
runner; it does not run an experiment matrix, repetitions, or resume logic.

Activate the project environment before running commands (on Windows, use the
equivalent `subs_env` activation command):

```bash
source subs_env/bin/activate
uv run --active --no-sync -m utils.shorten_file INPUT.srt \
  --output-dir output/short-file-run \
  --model openai/gpt-5.6-luna \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json
```

The profile is required, as are `--model` and `--output-dir`. Supported model
IDs are `openai/gpt-5.6-luna`, `openai/gpt-5.6-sol`,
`deepseek/deepseek-flash`, and `deepseek/deepseek-v4-flash`. The selected one
model is used for both the generator and arbiter stages (one model does not
mean one HTTP request). OpenCode uses the default `medium` variant and
DeepSeek uses `off`; `--variant` may select another supported value.

## Calibration and length basis

Planning, local candidate checks, and the optional length tool all use
`normalized_tts_text`, not the display text. The frozen profile is checked
against the actual model files, voice, rate, and normalizer identity. The
profile's `char_linear` estimate is a calibrated character estimate, not a
guarantee that the final audio will fit. Short timing-risk cues receive an
actual original-audio preflight; a measured fit avoids an LLM request, while
an unresolved short cue remains unresolved.

There is no hidden 13-character-per-second fallback in this command. The
requested rate is finite and restricted to `[1.0, 1.5]`, and the exact
matching profile must be present.

## Input and outputs

JSON is read as an array of objects with UTF-8-SIG support. Unknown keys,
JSON indexes, and timing fields are retained. SRT is parsed with the existing
fail-closed parser. Its export uses canonical chronological numbering, while
JSON source indexes remain unchanged. A live run writes the runner's
`report.json`, `subtitles.json`, `subtitles.srt`, and checkpoints; incomplete
runs return a non-zero exit code and retain their checkpoint metadata.

Use a separate fresh directory for every invocation. `--dry-run` performs all
offline validation and writes only `preflight.json` with all plans, selected
IDs, calibration metadata, and an explicit statement that no LLM or acoustic
evidence exists yet:

```bash
uv run --active --no-sync -m utils.shorten_file INPUT.json \
  --output-dir output/short-file-preflight \
  --model deepseek/deepseek-flash \
  --profile output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json \
  --dry-run
```

The dry-run directory cannot be resumed as a live run: SH-04 is not
implemented. The command performs no API credential lookup, server or backend
construction, network call, calibration training, or Vosk inference in
dry-run mode.
