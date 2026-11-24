AngryBearr: <main_task>
I need you to plan a pipeline for the subtitles tool that I am building.
</main_task>
<detailed_task_description>
- entry point - python script
- input - subtitle file (srt)
- output - audio file (wav)
- steps:
  1. Read and parse the subtitle file (srt format).
  2. Extract text and timing information from the subtitles.
  3. Normalize the text with the help of runorm and ruaccent libraries.
  4. Use a text-to-speech (TTS) engine to convert the extracted text into audio segments.
  5. Align the audio segments with the timing information from the subtitles (adjust the timing as needed).
  6. Concatenate the audio segments into a single audio file.
  7. Export the final audio file in wav format.
</detailed_task_description>
<docs>
Document the pipeline steps for the subtitles tool in a markdown format.
</docs>

GitHub Copilot: Here's a clear, end‑to‑end pipeline plan plus ready-to-drop Markdown documentation you can add as `docs/pipeline.md` (or similar). Let me know if you’d like me to generate a combined runner script next.

## High-Level Plan
1. Parse SRT → structured JSON (`srt_to_json.py`).
2. Normalize + accent/stress → updated JSON (`text_normalizer.py` / `subtitle_processor.py`).
3. Batch synthesize TTS → per‑subtitle WAV segments (`synthesize_batch.py`).
4. Timing alignment / speed adjustments (optional) → `subs_utils.py` utilities.
5. Concatenate or overlay → one final WAV (`audio_mixer.py`).
6. Export result + optional logs/metrics.

---

```markdown
# Subtitle → TTS Voiceover Pipeline

## Overview
This pipeline converts a subtitle file (`.srt`) into a single spoken audio track (`.wav`) using:
- Parsing & structuring of subtitle timing and text.
- Text normalization (RUNorm) and stress/accent marking (RUAccent).
- Vosk TTS synthesis (batch).
- Optional duration alignment / speed correction.
- Concatenation or overlay onto a silent (or existing) media timeline.

## Entry Point
A Python orchestration script (future: `pipeline_run.py`) that chains existing utilities:
1. `utils/srt_to_json.py`
2. `utils/text_normalizer.py`
3. `synthesize/synthesize_batch.py`
4. `utils/audio_mixer.py` (or concatenation logic)

## Data Shapes

### Raw SRT
Typical block:
```
12
00:01:08,920 --> 00:01:10,200
Привет мир!
```

### Parsed JSON (list of objects)
Fields (current convention):
```jsonc
[
  {
    "index": 12,
    "start": 68920,          // ms
    "end": 70200,            // ms
    "duration": 1280,        // ms (derived)
    "text": ["Привет мир!"], // list always
    "gender": null           // optional / future metadata
  }
]
```

### Post‑Normalization / Accent
`text[0]` may include stress markers (e.g., `приве+т ми+р`). Ellipses handled via temporary replacement (`...` → `..` before accenting).

### Synth Output
- Directory of segment wavs: `tts_out/<prefix><n>.wav` (1-based index).
- Final merged wav: `output.wav` (or user-specified).

## Detailed Steps

### 1. Parse SRT
Script: `utils/srt_to_json.py`
- Reads SRT.
- Produces JSON with ms timing.
- Ensures `text` is a list (even single line).
- Validates chronological order; can warn on overlaps.

CLI (planned unified form):
```powershell
python srt_to_json.py .\input.srt -o .\subs.json
```

### 2. Normalize & Accent
Script: `utils/text_normalizer.py`
- Loads RUNorm (model_size="big") once.
- Loads RUAccent (`turbo3.1`).
- Pre-pass: replaces `...` with `..` to avoid mis-accenting.
- Batch processes joined text lines.
- Restores ellipses.
- Writes accentized JSON (e.g., `accentized.json`) with same list length.

CLI:
```powershell
python text_normalizer.py .\subs.json .\accentized.json 20
```
(Last arg = optional batch size)

Failure Handling:
- If batch RUAccent parse fails, falls back to per-line processing (defensive design in `subtitle_processor.py`).

### 3. Synthesize TTS (Batch)
Script: `synthesize/synthesize_batch.py`
- Loads Vosk TTS model from `models/<model_name>/`.
- Monkey-patches provider selection (GPU vs CPU) before model init (keep existing pattern).
- Iterates normalized lines; maps each to an audio segment.
- Output naming: `<file_prefix><n>.wav`.

Potential Parameters:
- `--model_name vosk-model-tts-ru-0.10-multi`
- `--speaker_id 0`
- `--sample_rate 48000`
- `--file_prefix tts_out/seg_`

(If not currently CLI-enabled, add optional arg parsing without breaking defaults.)

### 4. Duration Alignment (Optional)
Purpose: If spoken segment longer/shorter than subtitle slot.
Tools:
- `utils/subs_utils.change_speed(audio_path, factor)` to tempo-adjust.
Strategies:
- Stretch/compress to match `duration` (guard against > +/-25% drift to avoid artifacts).
- Or pad with silence (prepend/append) when audio shorter.

### 5. Concatenate / Overlay
Two approaches:

A. Simple Concatenation:
- Place each segment with silence gaps between according to `start` offsets.
- Build silent base of total length (`create_silence_audio` concept).
- Overlay each segment at `start`.

B. Overlay onto Existing Video:
- Create silent reference track equal to video duration.
- Mix segments in at computed positions.
- Export combined wav (then mux externally if needed).

Utility: `utils/audio_mixer.py` (extend if needed).
Naming: final `voiceover.wav`.

### 6. Export
- Ensure WAV PCM (e.g., 16-bit, 48 kHz).
- Provide metadata JSON (optional) with:
  - model
  - speaker
  - synthesis date
  - segment count
  - total real-time factor (if measured)

### 7. (Optional) Post Checks
- Validate no clipping (scan peak amplitude).
- Verify summed durations vs timeline coverage.
- Emit warning if >5% average drift from subtitle slots.

## Proposed Orchestration Script Skeleton

```python
#!/usr/bin/env python
# filepath: pipeline_run.py

import json, subprocess, tempfile, argparse, os, sys, shutil, uuid, time
from pathlib import Path

def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return r

def main():
    ap = argparse.ArgumentParser(description="SRT → TTS voiceover pipeline")
    ap.add_argument("srt", help="Input .srt file")
    ap.add_argument("-o", "--out", default="voiceover.wav", help="Final wav output")
    ap.add_argument("--workdir", default="pipeline_work", help="Temp workspace")
    ap.add_argument("--model_name", default="vosk-model-tts-ru-0.10-multi")
    ap.add_argument("--speaker_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--speed_align", action="store_true", help="Enable tempo adjustment to subtitle durations")
    ap.add_argument("--prefix", default="tts_out/seg_")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(exist_ok=True)

    subs_json = wd / "subs.json"
    accent_json = wd / "accent.json"

    # 1. Parse SRT
    run([sys.executable, "utils/srt_to_json.py", args.srt, "-o", str(subs_json)])

    # 2. Normalize + Accent
    run([sys.executable, "utils/text_normalizer.py", str(subs_json), str(accent_json), str(args.batch_size)])

    # 3. Synthesize batch (assumes script can be parameterized; add args if not)
    # (If synth script not yet CLI-ready, you would patch it similarly)
    run([
        sys.executable, "synthesize/synthesize_batch.py",
        "--input_json", str(accent_json),
        "--model_name", args.model_name,
        "--speaker_id", str(args.speaker_id),
        "--file_prefix", args.prefix
    ])

    # 4. (Optional) Speed alignment placeholder
    if args.speed_align:
        print("[INFO] Speed alignment step not yet implemented here—integrate subs_utils.change_speed per segment.")

    # 5. Concatenate
    # Implement concatenation: read accent_json, compute timeline, place wavs.
    # (Add a small mixer utility call or inline once defined.)
    print("[TODO] Concatenate segments into final wav:", args.out)

    print("[DONE] Pipeline completed (except TODO steps).")

if __name__ == "__main__":
    main()
```

(Adjust once `synthesize_batch.py` exposes CLI args—if not present, that's a next enhancement.)

## Error Handling Strategy
| Stage | Common Issue | Handling |
|-------|--------------|----------|
| Parse | Malformed timestamp | Skip or abort with line ref |
| Normalize | RUAccent batch parse fail | Fallback to per-line |
| TTS | Missing model dir | Abort with suggestion listing `models/` contents |
| Duration | Extreme mismatch | Log warning; cap speed change |
| Concatenate | Missing segment file | Insert silence placeholder |

## Edge Cases
1. Overlapping subtitles → choose first, merge, or extend; define policy (default: independent layering).
2. Empty text lines → skip synthesis, preserve gap.
3. Very long subtitle (>10s) → optional chunking or slower speech rate.
4. Punctuation-only lines → may skip unless needed for timing.
5. Ellipses sequences and unusual quotes → ensure normalization doesn’t remove intent.

## Performance Notes
- Load RUAccent and RUNorm once; reuse instances (already current pattern).
- Avoid re-instantiating TTS model per line; batch script already does this.
- Optionally parallelize synthesis if model allows (careful with GPU memory).
- Provide progress logging every N lines.

## Configuration Surface (Recommended)
| Parameter | Default | Description |
|-----------|---------|-------------|
| model_name | vosk-model-tts-ru-0.10-multi | TTS model directory |
| speaker_id | 0 | Speaker / voice index |
| batch_size | 20 | RUAccent normalization batch size |
| speed_align | off | Whether to time-stretch segments |
| sample_rate | model native | Force resample if needed |
| file_prefix | tts_out/seg_ | Segment naming |

## Testing Recommendations
- Unit: Confirm `srt_to_json` stable with multiline, overlapping, empty lines.
- Unit: RUAccent transformation preserves list length and indices.
- Integration: End-to-end small 3-line SRT produces deterministic number of wavs.
- Regression: Ellipsis cases: `...`, `....` preserved after accenting.
- Audio: Peak normalization not clipping (optional test scanning max amplitude).

## Future Enhancements
- CLI for `synthesize_batch.py` if missing.
- Optional silence trimming post-synthesis.
- Voice selection per subtitle via added `speaker_id` field.
- Alignment metrics JSON report.
- Caching normalized/accentized text to avoid rework.

## Quick Usage (Manual Current Flow)
```powershell
# 1. Parse
python srt_to_json.py .\input.srt -o .\subs.json

# 2. Normalize + Accent
python text_normalizer.py .\subs.json .\accentized.json 20

# 3. Batch Synthesize (adjust once CLI added)
python synthesize_batch.py  # edit inside for params

# 4. Concatenate / overlay (manual or future script)
# (Implement using audio_mixer utilities)
```

## Directory Expectations
- Models directory: `models/vosk-model-tts-ru-0.10-multi/`
- Working artifacts: `subs.json`, `accentized.json`, `tts_out/seg_*.wav`
- Final: `voiceover.wav`

---

Let me know if you want:
- A finished concatenation function.
- CLI upgrade for `synthesize_batch.py`.
- Automated speed alignment prototype.

Happy to implement the next piece. Just say the word.
```

AngryBearr: For now write this as a doc for further validation/corrections.

GitHub Copilot: Acknowledging your request—here’s a focused Markdown document you can drop in as `docs/pipeline.md` for review and iteration.

```markdown
# Subtitle → Speech Pipeline (Draft for Review)

Status: DRAFT (awaiting validation / corrections)  
Goal: Convert an input `.srt` subtitle file into a single aligned spoken audio track (`.wav`) using normalization, stress accenting, and Vosk TTS synthesis.

---

## 1. High-Level Flow

Input: `input.srt`  
Output: `voiceover.wav`

Steps:
1. Parse SRT → structured JSON.
2. Extract and retain timing (ms) + text lines.
3. Normalize + accent (RUNorm + RUAccent).
4. Synthesize each subtitle to an audio segment (Vosk TTS).
5. (Optional) Adjust segment playback speed to better match subtitle duration.
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
Produced sequentially:
```
tts_out/seg_1.wav
tts_out/seg_2.wav
...
```
All share a consistent sample rate (model native or resampled).

---

## 3. Components & Scripts (Current / Planned)

| Stage | Purpose | Script / Module | Notes |
|-------|---------|-----------------|-------|
| Parse | SRT → JSON | `utils/srt_to_json.py` | Ensures list-form `text` |
| Normalize | Orthographic normalization | `utils/text_normalizer.py` | RUNorm (`big`) |
| Accent | Stress marking | `utils/subtitle_processor.py` (internally) | RUAccent `turbo3.1` |
| Synthesize | TTS per entry | `synthesize/synthesize_batch.py` | Batch uses model in `models/` |
| Speed Adjust (opt) | Duration alignment | `utils/subs_utils.py` | `change_speed` helper |
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
- Load Vosk TTS model once (e.g., `vosk-model-tts-ru-0.10-multi`).
- Monkey-patch provider selection before model instantiation (CPU/GPU).
- Iterate normalized entries:
  - Skip empty or punctuation-only lines (configurable).
  - Generate WAV segment.
  - Name sequentially with 1-based index.
- (Future) Allow per-entry `speaker_id` overriding global default.

### 4.4 Duration Alignment (Optional)
Mismatch scenarios:
- Audio longer than slot → optionally tempo compress (limit e.g. ±25%).
- Audio shorter → optionally pad with silence (head/tail).
Policies (to decide):
- Hard enforce fit vs tolerant drift.
- Log real vs target duration; accumulate stats.

### 4.5 Timeline Assembly / Mixing
Two patterns:

A. Silence Base + Overlay  
1. Create silent track of total video (or last subtitle end) length.  
2. Overlay each segment at its `start` offset.  

B. Direct Concatenation with Gaps  
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

## 6. CLI Examples (Current Manual Flow)

```powershell
# 1. Parse SRT
python srt_to_json.py .\input.srt -o .\subs.json

# 2. Normalize + Accent
python text_normalizer.py .\subs.json .\accentized.json 20

# 3. Synthesize (currently requires editing synth script params if no CLI)
python synthesize_batch.py

# 4. (Future) Run orchestrator once added
python pipeline_run.py .\input.srt -o voiceover.wav --model_name vosk-model-tts-ru-0.10-multi --speaker_id 0
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
| model_name | `vosk-model-tts-ru-0.10-multi` | Directory under `models/` |
| speaker_id | 0 | Voice index |
| batch_size | 20 | RUAccent batch size |
| speed_align | off | Enable tempo/time stretch |
| file_prefix | `tts_out/seg_` | Segment naming prefix |
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
- CLI argument support inside `synthesize_batch.py`.
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

Feedback welcome—mark any section needing correction or deeper detail.
```

Let me know if you’d like me to:
- Add the orchestration script now.
- Turn `synthesize_batch.py` into a proper CLI.
- Implement the mixer/concatenation utility.

Just point me to the next priority.

AngryBearr: Add what you've described as a markdown document

GitHub Copilot: Created 2 todos

Creating the pipeline documentation file in the docs directory. 

Using "Apply Patch"


```

```


Starting (2/2) *Report addition to user*

## Added Documentation

Created pipeline.md with the full draft pipeline specification you requested (parsing → normalization/accent → synthesis → optional alignment → mixing → export), plus open decisions, edge cases, and a validation checklist.

### Next Possible Enhancements (pick any to proceed):
1. Add `pipeline_run.py` orchestrator script.
2. Add proper CLI argument parsing to synthesize_batch.py.
3. Implement concatenation / overlay utility (if `audio_mixer.py` needs expansion).
4. Add duration alignment (speed adjust) logic with configurable tolerance.
5. Emit metadata sidecar (`voiceover_meta.json`) after pipeline completion.
6. Add automated integration test (mini SRT → final wav presence check).

Let me know which one you’d like me to implement next (or suggest another direction), and I’ll proceed.

Made changes.
