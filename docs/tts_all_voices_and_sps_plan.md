# Plan: All-voices Vosk TTS run + Symbols-per-second metrics

This document describes two small additions to this repository:
- A script to synthesize a long, complex Russian text with every speaker in `models/vosk-model-tts-ru-0.10-multi` using CPU.
- A metrics utility that computes symbols-per-second (SPS) for the produced WAV files, counting all characters in the text.

 The design follows existing patterns (provider monkey-patch, output naming, utilities reuse) outlined in this repo.
 Counting policy for SPS and manifest (per request):
 - Count punctuation (commas, periods, exclamation and question marks, colons, semicolons, hyphens, ellipses, quotes, etc.).
 - Exclude spaces and newline characters from the count.
 - Ignore RUAccent stress marks (`+`) — they are not counted as separate symbols; they’re treated as part of the letter they modify.

## Goals and scope
- Produce one WAV per voice (57 speakers) for the model `vosk-model-tts-ru-0.10-multi`.
- Use a sufficiently long and challenging Russian text to get stable timing and exercise the grapheme/phoneme pipeline.
- Default device: CPU; no need to use CUDA.
- Output naming: incremental `{file_prefix}{n}.wav` (1-based) under `tts_out/`.
- Emit a compact `manifest.csv` mapping output index and speaker_id to filename and text length.
- Provide a reusable SPS calculator that reads durations from WAV without requiring FFmpeg.
- Do not alter existing public behavior; keep new utilities optional.

## Existing components to reuse
- `synthesize/synthesize_batch.py` — contains `patch_model_for_device(device="cpu"|"cuda")` for ONNX providers monkey‑patch; reuse this patch before model instantiation.
- `synthesize/synthesize.py` — reference for single-run naming and Synth API usage.
- `utils/subs_utils.py` — `ensure_folder_exists(path)` to create output folders.
- `utils/audio_mixer.py` — uses `soundfile`; we’ll mirror the same dependency to read WAV duration reliably without FFmpeg.
- `utils/text_normalizer.py` and `utils/subtitle_processor.py` — provide RUAccent integration patterns (ellipsis replacement/restore, batching, and a `cust_dict` for domain overrides). We will reuse this approach to add stress marks to the single long text before synthesis.
- `models/vosk-model-tts-ru-0.10-multi/config.json` — includes `num_speakers`, `speaker_id_map`, `audio.sample_rate`; we’ll assume numeric `speaker_id` in 0..56.

## Script 1: synthesize_all_voices.py

Responsibilities:
- Load Vosk TTS `vosk-model-tts-ru-0.10-multi` once on CPU.
- Preprocess the input text with RUAccent to add stress marks (`+`) and then synthesize the resulting accentized text for each `speaker_id` in range(num_speakers).
- Save outputs using `{output_folder}/{file_prefix}{n}.wav`, where `n` starts at 1.
- Write `manifest.csv` alongside outputs: `n,speaker_id,filename,text_length`.

CLI (proposed):
- `--model-name` (str, default: `vosk-model-tts-ru-0.10-multi`) — resolved under `models/`.
- `--model-path` (str, optional) — if provided, overrides model-name resolution.
- `--output-folder` (str, default: `tts_out`) — created if missing.
- `--file-prefix` (str, default: `tts_`) — prefix for output naming.
- `--text-file` (path, optional) — if not provided, use the embedded default text.
- `--speech-rate` (float, optional) — fallback to model default if None.
- `--noise-level` (float, optional) — fallback to model default if None.
- `--duration-noise-level` (float, optional) — fallback to model default if None.
- `--scale` (float, optional) — model-dependent additional control.
- `--no-accent` (flag, optional) — by default RUAccent preprocessing is enabled; this flag disables it.

 Defaults and policies:
- Device is forced to CPU (providers = ["CPUExecutionProvider"]).
- Language is set to `ru`.
- RUAccent preprocessing is ON by default to ensure correct pronunciation and more accurate timing; stress marks (`+`) are preserved in the text passed to synthesis.
  - Symbol counting for `manifest.csv` and SPS follows the policy above: remove spaces and newlines, drop `+` stress marks, but keep punctuation (including `...`).

 Implementation notes:
- Import and call `patch_model_for_device(device="cpu")` before creating the model.
- Instantiate `Model` and `Synth` once, reuse across all speakers.
- Determine `num_speakers` as `len(config["speaker_id_map"])` or `config["num_speakers"]` (whichever the patched model exposes); if unavailable, fallback to 57.
- For index `i` in 0..num_speakers-1, write `{file_prefix}{i+1}.wav`.
- Use `utils.subs_utils.ensure_folder_exists(output_folder)`.
  - Emit `manifest.csv` with header row and one line per output: `i+1,speaker_id,filename,symbol_count` where `symbol_count` uses the policy above.

 RUAccent preprocessing:
- We will follow the repository’s established pattern:
  1) Replace ellipses before accenting to avoid parsing issues (`replace_ellipsis`).
  2) Use RUAccent with model size `turbo3.1` on CPU to add stress marks. For a single long string we can either:
     - Use RUAccent’s `process` for a single string, or
     - Reuse the batch pattern: convert `[text]` to a Python list string and call `process_all(..., skip_regex=...)`, then `ast.literal_eval` back.
  3) Restore ellipses (`restore_ellipsis`).
  4) Preserve any `cust_dict` overrides defined in `utils/text_normalizer.py` if we extend or adapt its logic.
- If RUAccent fails to parse the full string (rare), we’ll fallback to a defensive path: split on sentences and accent each piece, then join; log a warning.

Embedded default Russian text (criteria):
- One multi-paragraph block ~700–1200 symbols to ensure measurable durations per voice.
- Rich morphology and orthography: loanwords, compound words, abbreviations, proper nouns, numbers, dates, rare or archaic lexemes, hyphenation, and punctuation variety.
- Example categories to include: «какафония», «инфраструктура», «экзистенциальный», «непротиворечивость», «реинжиниринг», «квазиисторический», «мультимодальность», «субстантивированный», «кросс‑валидация», «электрофизиология», «непреложный».
- The exact text lives as a module-level constant `DEFAULT_RU_TEXT` inside the script; users can override via `--text-file`.

Output artifacts:
- WAVs: `{output_folder}/{file_prefix}{n}.wav` for each speaker.
- `manifest.csv` saved into `{output_folder}`.

Error handling:
- If model path/name not found: clear error with hint to place model under `models/`.
- If a particular `speaker_id` fails (out of bounds): log and continue (or abort if strict mode is desired; default: continue).
- If text is empty or whitespace: abort with error.
- If RUAccent preprocessing is enabled but fails, attempt a per-sentence fallback; if still failing, either disable accenting with a warning (respecting `--no-accent`) or abort based on a `--strict-accent` flag (optional).

## Script 2: utils/audio_metrics.py (SPS)

Responsibilities:
- Provide a function to compute symbols-per-second (SPS) given a folder of WAVs and the text used for synthesis.
- Count all symbols in the provided text (including spaces/punctuation).
- Read WAV durations using `soundfile` to avoid external dependencies.
- Save per-file metrics to `metrics.csv` and return a summary.

 Public API (Python):
 - `compute_symbols_per_second(outputs_folder: str, text: str, expected_prefix: str = "tts_", sort=True) -> dict`
   - Iterates files matching `{expected_prefix}*.wav` under `outputs_folder`.
   - For each file: duration_sec = frames / samplerate (via `soundfile.info(path)`).
   - chars = `count_symbols(text)` where `count_symbols` applies the policy: remove spaces (`" ")` and newlines (`"\n"`, `"\r"`), drop `+` stress marks, keep punctuation and all other visible symbols.
   - cps = `chars / duration_sec` if `duration_sec > 0` else `None`.
   - Returns `{ "per_file": [ {"filename", "duration_sec", "chars", "cps"}... ], "aggregate": {"mean_cps", "min_cps", "max_cps"} }`.
   - Also writes `metrics.csv` with columns: `filename,duration_sec,chars,cps` into `outputs_folder`.
   - Provide `count_symbols` in the same module for reuse; future flags can toggle inclusion of spaces/newlines if needed.

CLI (optional small wrapper or flag in the synth script):
- Inputs: `--outputs-folder`, `--text-file` (or `--text` literal), `--file-prefix`.
- Produces `metrics.csv` under `outputs-folder` and prints summary.

Notes on durations:
- Use `soundfile.info(path)`; it returns samplerate and frames consistently for WAV.
- Avoid pydub/ffprobe to remove dependency on FFmpeg for this task.

## End-to-end flow
1) Run `synthesize_all_voices.py` with defaults. Internally it will RUAccent‑accentize the text first, then synthesize.
  - Outputs 57 WAVs and a `manifest.csv` in `tts_out/` (or user-specified folder).
2) Run the SPS utility with the same (accentized) text to produce `metrics.csv` where `chars` uses the policy above (punctuation included, spaces/newlines excluded, `+` removed).

## Try it (example commands)
- Synthesize all voices on CPU with defaults:

```powershell
# Uses embedded complex Russian text; outputs to .\tts_out\tts_1.wav ... tts_57.wav
python synthesize/synthesize_all_voices.py
```

- Synthesize with a text file and custom prefix:

```powershell
python synthesize/synthesize_all_voices.py --text-file .\temp\long_ru_text.txt --file-prefix vosk10_
```

- Compute SPS (counting all symbols) for the run above:

```powershell
python -c "from utils.audio_metrics import compute_symbols_per_second; import json; print(json.dumps(compute_symbols_per_second('tts_out', open('temp/long_ru_text.txt', 'r', encoding='utf-8').read(), expected_prefix='vosk10_'), ensure_ascii=False, indent=2))"
```

The SPS function also writes `tts_out/metrics.csv`.

## Edge cases and safeguards
- Missing or empty text: the synthesizer should raise a clear error.
- Zero-length or extremely short audio: CPS becomes undefined; record as empty and warn.
- Partial output set: the SPS utility will only include found files; aggregate computed from available files.
- File ordering: if `sort=True`, files are sorted lexicographically; numeric parsing of the trailing index can be added if needed.
- Sample rate: do not assume 48k; read actual WAV samplerate. Model 0.10 uses 22050 Hz per `config.json`.
 - Ellipses handling: text-level counting treats `...` as three characters; pronunciation-wise the model may map it to a single phoneme token, but counting follows the textual rule.

## Quality gates
- Lint/typecheck: both scripts are plain Python with stdlib + soundfile; ensure imports are added to `requirements.txt` if `soundfile` isn’t already present. If not desired, we can fall back to `wave` from stdlib, but `soundfile` is more robust.
- Tests (lightweight):
  - Unit test for `compute_symbols_per_second` using a tiny generated WAV with known duration (e.g., 1s of silence) to validate CPS math.
  - Smoke test for `synthesize_all_voices.py` with `--limit 2` (optional flag) to synthesize two speakers quickly.
  - RUAccent preprocessing test: given a short snippet with ellipses and known stress, verify ellipsis replacement/restore and presence of `+` markers in the output.

## Implementation checklist
- Add `synthesize/synthesize_all_voices.py`:
  - Import: `patch_model_for_device`, `Model`, `Synth`, `ensure_folder_exists`, `json`, `csv`, `argparse`, `pathlib`.
  - Embed `DEFAULT_RU_TEXT` and implement optional `--text-file`.
  - Add RUAccent preprocessing (CPU) with ellipsis handling and optional `cust_dict` integration.
  - CPU-only providers via patch; single Model/Synth instance reused.
  - Iterate 0..num_speakers-1; write WAVs as `{prefix}{i+1}.wav`.
  - Emit `manifest.csv` with index, speaker_id, filename, text_length.

- Add `utils/audio_metrics.py`:
  - Implement `compute_symbols_per_second(outputs_folder, text, expected_prefix='tts_', sort=True)` using `soundfile`.
  - Write `metrics.csv` with `filename,duration_sec,chars,cps` and return JSON-like dict.

- Optional: Add a `--compute-sps` flag to the synth script to compute metrics immediately after synthesis.

## Future extensions
- Add per-entry synthesis overrides (speech rate, noise) per speaker from a manifest.
- Allow speaker name mapping by reading `speaker_id_map` from model `config.json`.
- Permit filename pattern `{prefix}{n:02d}.wav` and/or include speaker name in filenames.
- Add resampling/transcoding switches (e.g., to 48kHz WAV or MP3) reusing `utils/audio_mixer.py` patterns.
