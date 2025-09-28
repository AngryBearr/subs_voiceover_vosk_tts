# Copilot Project Instructions

Purpose: Speed up AI agent onboarding to this subtitle → normalized/accented text → Vosk TTS synthesis pipeline.
Keep responses concise. Follow existing patterns; do not introduce frameworks or large refactors without request.

## 1. Big Picture Data Flow
1. Input SRT (`*.srt`) → parsed to structured JSON (`utils/srt_to_json.py`).
2. JSON subtitles (list of `{index,start,end,duration,text:[...],gender}`) → normalization + accent injection (RUAccent + RUNorm) in batch (`utils/text_normalizer.py`, `utils/subtitle_processor.py`). Ellipses are temporarily rewritten (`...`→`..`) before accenting, then restored.
3. Accentized text (may include `+` markers for stress) → Vosk TTS single or batch synthesis (`synthesize/synthesize.py`, `synthesize/synthesize_batch.py`).
4. Produced wav segments → optionally speed‑matched (`utils/subs_utils.change_speed`) and overlaid onto a silent bed or video duration (`overlay_audio`, `create_silence_audio`).

## 2. Key Conventions & Patterns
- Text lists: subtitle entries keep `text` as a list (even if single line). Batch processors often join lines with space, then restore to single‑element list containing processed string.
- Accent workflow: Replace ellipsis before RUAccent (`replace_ellipsis` / `restore_ellipsis`). Skip characters that would break list literal parsing by operating on `str(list)` plus a skip regex.
- RUAccent batching: Convert list of strings to a Python list string representation and call `process_all(..., skip_regex=...)`; then `ast.literal_eval` back.
- Device selection for ONNX / Vosk TTS done via monkey‑patching `Model.__init__` to inject providers (`CUDAExecutionProvider` vs `CPUExecutionProvider`). Replicate existing patch style; do not redesign.
- Custom dictionary for accenting defined in `utils/text_normalizer.py` (`cust_dict`). Preserve when extending.
- Output audio naming: single synth uses `<file_prefix>output.wav`; batch uses incremental `<file_prefix><n>.wav` (1‑based index). Maintain for consistency.

## 3. External Dependencies / Requirements
- Core libs: `ruaccent`, `runorm`, `onnxruntime`, `vosk-tts`, PyTorch (for potential GPU), `tokenizers` when BERT sub-model present.
- RUAccent loads with model size `turbo3.1`; RUNorm with `model_size="big"` from cached `runorm_cache` directory.
- FFmpeg + FFprobe must be present on PATH for audio speed and overlay utilities.

## 4. Typical Workflows
- Convert SRT → JSON: call `srt_to_json(path, out_path)` or run `python utils/srt_to_json.py input.srt -o subs.json`.
- Normalize + accent: `python utils/text_normalizer.py subs.json accentized.json 20` (batch size optional). Returns transformed JSON with accent markers.
- Batch synth: `python synthesize/synthesize_batch.py` after editing hard‑coded example OR expose a wrapper. Ensure `model_path` or `model_name` resolves to existing folder under `models/`.
- Single line synth for quick test: run `python synthesize/synthesize.py` (edit sample text & params).
- Overlay segments: ensure silent base via `create_silence_audio`, then `overlay_audio(silence.wav, tts_out, subs.json, mixed.mp3)`.

## 5. Testing Notes
- Existing tests: `test_srt_transformer.py` (parsing), `test_ruaccent.py` (structure & RUAccent batch). Use `python -m unittest` to run all.
- When adding new processors, mirror batch pattern from `subtitle_processor.process_subtitles_with_ruaccent` and keep list length stable.

## 6. Extension Guidelines
- Keep new params optional with safe defaults; do not break existing function signatures relied upon by scripts.
- If adding per‑entry synthesis controls, follow pattern in `synthesize_batch.py` (lookup keys, fall back to defaults).
- Prefer adding small utility in `utils/` rather than scattering logic into synth scripts.

## 7. Performance / Reliability
- Heavy model loads: load RUAccent / RUNorm / Vosk TTS once and reuse; avoid per‑subtitle instantiation.
- Graceful fallback: On RUAccent batch parse failure, process each line individually (see `subtitle_processor.py`). Maintain this defensive style.

## 8. Pitfalls / Gotchas
- Ellipsis handling mandatory before accenting to avoid mis-accenting sequences of dots.
- Ensure JSON encoding uses `utf-8` and retains `ensure_ascii=False` for Cyrillic output.
- `speaker_id` must be valid for the selected model variant; no runtime check currently—document in PRs if adding.
- Monkey patch must happen before instantiating `Model`; do not delay patch after creating the object.

## 9. Safe Change Examples
- Adding a new normalization pre-pass: implement function in `text_normalizer.py`, call it inside existing loop prior to `replace_ellipsis`.
- Adding CLI to batch synth: wrap argument parsing; keep existing default invocation behavior when script run without args.

## 10. When Unsure
Keep changes localized, prefer utility extraction over refactor, and ask for clarification if altering data structure of subtitle entries.

---
Provide feedback if any workflow differs in actual usage or if additional internal scripts should be documented.
