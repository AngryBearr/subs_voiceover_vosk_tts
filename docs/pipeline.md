# Subtitle → Speech Pipeline (Updated)

Status: DRAFT (updated to unified CLI + new synth)  
Goal: Convert an input `.srt` subtitle file into a single aligned spoken audio track (`.wav`) using normalization, stress accenting, and Vosk TTS synthesis.

---

## 1. High-Level Flow

Input: `input.srt`  
Output: `voiceover.wav`

Steps:
1. Parse SRT → structured JSON.
2. Extract and retain timing (ms) + text lines.
3. Normalize + accent (RUNorm + RUAccent).
4. Synthesize each subtitle to an audio segment (Vosk TTS via unified CLI).
5. Adjust segment playback speed to better match subtitle duration.
6. Place segments on a timeline (silence padding / overlay).
7. Concatenate / mix down → final WAV export.

---

## 2. Data Shapes

### 2.1 Parsed Subtitle Entry (JSON)
```jsonc
{
  "index": 12,
  "start": 68920,          // milliseconds from t=0
  "end": 70200,
  "duration": 1280,        // end - start
  "text": ["Привет мир!"], // always a list
  "gender": null           // optional / reserved
}
```

### 2.2 Post-Normalization / Accent
`text[0]` may contain stress markers (`+`) inserted by RUAccent:  
Example: `приве+т ми+р`

### 2.3 Segment Audio Files
Produced sequentially with a 1-based counter and configurable prefix (default `tts_`):
```
tts_out/tts_1.wav
tts_out/tts_2.wav
...
```
All share a consistent sample rate (model native or an optional resampled rate).

---

## 3. Components & Scripts (Current / Planned)

| Stage | Purpose | Script / Module | Notes |
|-------|---------|-----------------|-------|
| Parse | SRT → JSON | `utils/srt_to_json.py` | Ensures list-form `text` |
| Normalize | Orthographic normalization | `utils/text_normalizer.py` | RUNorm (`big`) |
| Accent | Stress marking | `utils/subtitle_processor.py` (internally) | RUAccent `turbo3.1` |
| Synthesize | TTS per entry | `synthesize/synthesize_cli.py` | Single-line mode (`--text`) and batch mode (`--json`); batch uses model in `models/` |
| Speed Adjust | Duration alignment | `utils/subs_utils.py` | `change_speed` helper |
| Mix / Overlay | Timeline assembly | `utils/audio_mixer.py` | Silence base + overlay |
| Export | Final WAV | (to add orchestrator) | Planned: `pipeline_run.py` |

---

## 4. Detailed Step Descriptions

### 4.1 Parse SRT
- Read sequential blocks.
- Convert timestamps (`HH:MM:SS,mmm`) → ms.
- Preserve ordering; warn if non-monotonic.
- Multi-line subtitle text collapsed into a list (`["line1", "line2"]`).

### 4.2 Normalize & Accent
Workflow:
1. Temporarily replace ellipses: `...` → `..` (prevents RUAccent errors).
2. Join each entry’s `text` list into a single string.
3. RUNorm normalization (case rules, spacing, standardization).
4. RUAccent batch processing (with skip regex for safe literal parsing).
5. Restore ellipses (`..` → `...`).
6. Store result back as single-element list per entry.

Failure Handling:
- On batch parse failure, fallback to per-entry RUAccent invocation.
- Length of subtitle list must remain invariant.

### 4.3 TTS Synthesis
- Use unified CLI entry point: `python -m synthesize.synthesize_cli`.
- Device/providers: follows onnxruntime defaults; on CUDA, will warn if `CUDAExecutionProvider` is unavailable and proceed on CPU.
- Iterate normalized entries (batch mode):
   - Lines are joined when `text` is a list.
   - Skip truly empty results; otherwise synthesize to WAV.
   - Name sequentially with 1-based index and the configured `--file-prefix`.
- (Future) Allow per-entry `speaker_id` overrides by extending JSON schema and CLI mapping.

### 4.4 Duration Alignment
Duration Alignment (policy)
- Purpose: when a synthesized audio segment is longer than its subtitle time slot, the system may shorten playback by increasing tempo. The system MUST NOT slow audio down to lengthen it, and MUST NOT insert silence padding to force a fit.

- Rules:
   - Only allow speeding up (tempo increase) to reduce segment duration. Do not slow down or add silence.
   - Cap the maximum allowed speed-up factor (e.g., 1.75x by default). If the required factor exceeds the cap, mark the segment as "unfit" and allow overlay behavior described below.
   - Minimum speed-up threshold: apply tempo adjustments only when the synthesized duration exceeds the target slot by more than a small tolerance (e.g., 5%).

- Logging & metrics:
   - For every adjusted segment, log: segment index, original_duration_ms, target_duration_ms, applied_speed_factor, method (time-stretch), and whether it hit the cap.
   - Accumulate statistics (count, mean overshoot, percent capped) for reporting after the run.

- Overlay behavior (when speed-up > cap):
   - If the required speed-up > cap, do not force-fit. Leave the segment at the adjusted speed and allow it to overlap the neighboring segment(s) on the timeline.
   - Mixing guidance for intelligibility:
     1. When two segments overlap, favor the later (overlapping) segment by raising its level slightly (e.g., +1–3 dB) so the subsequent speech remains clear. Alternatively, apply a short duck (3–6 dB) to the earlier segment during the overlap.
     2. Use a short crossfade (10–30 ms) at overlap boundaries to avoid clicks and to make transitions smoother.
     3. Avoid excessive gain that causes clipping; apply peak limiting or normalize peaks post-mix if necessary.
   - Logging & reporting:
     - Mark segments that could not be fit ("unfit") in the validation report and include overlap start/end info.
     - If a high proportion of segments are "unfit," recommend merging adjacent subtitles or revising SRT timing as remediation.

- Implementation notes:
   - Use a high-quality time-stretch algorithm that preserves pitch (e.g., Rubber Band, WSOLA, or SoX/FFmpeg with -filter:a atempo chains). Avoid resampling that changes pitch.
   - Perform tempo adjustments after any resampling to the target project sample rate.
   - Prefer in-memory streamed processing for large projects to avoid excessive disk IO.


### 4.5 Timeline Assembly / Mixing
Two patterns:

**A. Silence Base + Overlay**  
1. Create silent track of total video (or last subtitle end) length.  
2. Overlay each segment at its `start` offset.  

**B. Direct Concatenation with Gaps**  
1. Append each segment in order.  
2. Insert silence for gaps between `prev.end` and `next.start`.  
3. No true parallel mixing (slightly different if overlaps exist).

Overlaps:
- Option 1: Allow overlay (retain both voices).
- Option 2: Merge text earlier (pre-process stage).
(Current draft defaults to overlay if implemented.)

### 4.6 Export
- Write final `voiceover.wav` (PCM 16-bit or model-native).
- (Optional) Produce `voiceover_meta.json` with:
  - `model_name`, `speaker_id`, `segment_count`, `total_duration_ms`, `gen_timestamp`.

### 4.7 Validation Pass (Optional)
Checks:
- Segment count matches processed entries (minus skipped empties).
- Peak amplitude < 0 dBFS (no clipping).
- Average drift (|synth_duration - slot_duration| / slot_duration) aggregated.
- Presence of unaccented entries (sanity).

---

## 5. Proposed Orchestration Script (Planned)
File: `pipeline_run.py` (to be added)  
Responsibilities:
- Argument parsing (SRT path, model, speaker, flags).
- Temporary workspace or designated output folder.
- Sequential invocation of existing scripts.
- Hook for optional speed alignment.
- Call mixer / concatenator utility to finalize output.

---

### 6. CLI Examples (Unified)

```powershell
# 1. Parse SRT
python utils/srt_to_json.py .\input.srt -o .\subs.json

# 2. Normalize + Accent
python utils/text_normalizer.py .\subs.json .\accentized.json 20

# 3a. Single-line synthesis
python -m synthesize.synthesize_cli --text "Привет!" --model "models/vosk-model-tts-ru-0.10-multi" --voice 0 --speech-rate 1.0 --device cpu

# 3b. Batch synthesis
python -m synthesize.synthesize_cli --json .\accentized.json --model "models/vosk-model-tts-ru-0.10-multi" --voice 0 --speech-rate 1.0 --device cpu --output-folder .\tts_out --file-prefix tts_

# (Optional) Resample batch to 48 kHz
python -m synthesize.synthesize_cli --json .\accentized.json --model "models/vosk-model-tts-ru-0.10-multi" --output-sample-rate 48000

# 4. (Future) Orchestrator once added
python pipeline_run.py .\input.srt -o voiceover.wav --model_path models/vosk-model-tts-ru-0.10-multi --speaker_id 0
```

---

## 7. Error Handling Strategy (Draft)

| Stage | Potential Issue | Strategy |
|-------|-----------------|----------|
| Parse | Malformed timestamp | Fail fast with block index |
| Normalize | RUAccent batch literal eval fails | Fallback to per-entry |
| Accent | Unexpected Unicode | Skip accent, warn |
| TTS | Missing model folder | List available `models/` and exit |
| Duration Align | Stretch factor extreme | Cap & warn, or skip |
| Mix | Missing segment file | Insert silence placeholder |
| Export | Sample rate mismatch | Resample all to target before concatenation |

---

## 8. Edge Cases to Validate
1. Empty subtitle blocks → should not produce audio.
2. Overlapping subtitles → both should overlay unless policy changes.
3. Very short durations (<300 ms) → maybe skip or merge with next (policy).
4. Ellipsis-heavy dialogue → ensure restoration accurate.
5. Multi-line entries with punctuation at line breaks.

---

## 9. Performance Considerations
- Single load per model (RUNorm, RUAccent, TTS).
- Batch RUAccent to reduce Python overhead.
- Optional future parallel segment synthesis (only if model thread-safe).
- Avoid writing large intermediate PCM unless required (stream mixing later?).

---

## 10. Configuration Surface (Initial)
| Parameter | Default | Description |
|-----------|---------|-------------|
| model_path | `models/vosk-model-tts-ru-0.10-multi` | Absolute or workspace-relative path to model directory |
| speaker_id | 0 | Voice index |
| batch_size | 20 | RUAccent batch size |
| speed_align | off | Enable tempo/time stretch |
| file_prefix | `tts_` | Segment naming prefix in `tts_out` |
| output | `voiceover.wav` | Final mix path |

---

## 11. Open Decisions (Need Validation)
| Topic | Question |
|-------|----------|
| Overlaps | Overlay vs merge text pre-synthesis? |
| Empty blocks | Skip silently or log? |
| Duration policy | Hard fit or tolerant drift? |
| Clipping control | Normalize final mix to -1 dBFS? |
| Metadata | Always emit JSON sidecar? |
| Multi-speaker | Per-line `speaker_id` support now or later? |

---

## 12. Future Enhancements (Backlog)
- CLI entry is unified in `synthesize/synthesize_cli.py` (single and batch).
- Per-subtitle voice switching (extend JSON schema).
- Optional noise bed / room tone layering.
- Whisper-based re-alignment (phoneme timing refinement).
- Caching normalized/accentized text across runs.
- Real-time factor logging for performance tracking.

---

## 13. Validation Checklist (When Implemented)
- [ ] End-to-end run produces non-empty `voiceover.wav`.
- [ ] Number of generated segments equals subtitle count minus skipped empties.
- [ ] All stress markers removed or intentionally retained in TTS input (decide).
- [ ] No unhandled exceptions with malformed SRT sample.
- [ ] Peak amplitude below clipping threshold.
- [ ] Optional: Drift report printed.

---

## 14. Quick ASCII Flow

```
input.srt
   │
   ▼
[Parse] ──> subs.json
   │
   ▼
[Normalize + Accent] ──> accentized.json
   │
   ▼
[Synthesize] ──> seg_1.wav ... seg_N.wav
   │
   ▼
[Align (opt)] ──> adjusted segments
   │
   ▼
[Mix / Concatenate]
   │
   ▼
voiceover.wav (+ metadata)
```

---

## 15. Review Notes (Fill During Validation)
- Parsing correctness: ___________________________
- Accent accuracy: _______________________________
- Timing drift: __________________________________
- Audio artifacts: _______________________________
- Priority next step: ____________________________

---

*End of draft.*
