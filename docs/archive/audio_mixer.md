# Audio Mixer Utility (`utils/audio_mixer.py`)

Efficiently combine a large number of short TTS-generated segment files (`<index>.wav`) into a single full-length track aligned to subtitle timing. This replaces iterative `pydub` overlay loops with a single-pass, preallocated NumPy mix for major performance gains.

## Key Features
- Single allocation linear-time mixing (no repeated whole-buffer copies)
- Optional existing base/bed track (silence, ambience, music)
- Forced padding to an explicit target duration (e.g. exact video length)
- Independent gain controls for base and speech layers
- Missing segment handling policy: `skip`, `warn`, or `error`
- Optional normalization + peak limiting with headroom margin
- Resampling (uses `python-soxr` if installed; fallback linear interpolation)
- Output to WAV (PCM) then optional transcode to `mp3`, `ogg`, or `opus`

## JSON Input Format
The mixer expects a list of entries with at least:
```jsonc
[
  {"index": 1, "start": 0, "end": 820, "text": ["..."]},
  {"index": 2, "start": 900, "end": 1600, "text": ["..."]}
]
```
Times are milliseconds. `index` ties to a file named `<index>.wav` in the segments folder.

## Basic Usage (CLI)
```powershell
python -m utils.audio_mixer subs_chunk.json tts_out output/mix
```
Produces `output/mix.wav` (intermediate) and `output/mix.mp3`.

## Important CLI Arguments
| Argument | Purpose | Default |
|----------|---------|---------|
| `--sr` | Target sample rate | 48000 |
| `--channels` | Channel count (1=mono, 2=stereo) | 1 |
| `--dtype` | Accum/output dtype (`int16` faster/smaller; `float32` for headroom) | int16 |
| `--no-normalize` | Disable peak limiting | (enabled) |
| `--limiter-margin` | dB headroom after limiting/normalizing | 0.5 |
| `--ext` | Segment extension | .wav |
| `--base-audio` | Optional existing bed file | None |
| `--base-gain` | Gain dB for base track | 0.0 |
| `--speech-gain` | Gain dB for speech segments | 0.0 |
| `--total-duration-ms` | Force minimum output duration | None |
| `--missing-policy` | `skip` / `warn` / `error` | skip |
| `--transcode` | mp3|ogg|opus|wav|none | mp3 |

Example adding a base ambience and forcing exact video length:
```powershell
python -m utils.audio_mixer subs_chunk.json tts_out output/mix `
  --base-audio video_silence_bed.wav `
  --total-duration-ms 1800000 `
  --speech-gain 1.5 `
  --limiter-margin 1.0
```

## Programmatic Usage
```python
from pathlib import Path
from utils.audio_mixer import mix_segments

mix_segments(
    subtitles_json=Path('subs_chunk.json'),
    segments_folder=Path('tts_out'),
    output_audio=Path('output/mix'),
    sample_rate=48000,
    channels=1,
    base_audio_path=Path('optional_bed.wav'),  # or None
    total_duration_ms=1_200_000,               # pad to 20 minutes if longer than last subtitle
    speech_gain_db=1.5,                        # add 1.5 dB to speech
    base_gain_db=-3.0,                         # lower bed a bit
    missing_policy='warn'
)
```

## Scenarios
### 1. Simple Narration Only
Just supply the subtitles JSON + segment folder. Output length = last subtitle `end`.

### 2. Ensure Output Matches Video Length
Provide `--total-duration-ms` retrieved via `ffprobe`. Trailing area is silent (unless base track longer).

### 3. Include Pre-Made Silence or Music Bed
Pass `--base-audio` path. Length becomes the max of (bed length, last subtitle end, total_duration_ms).

### 4. Adjust Loudness Balance
Use `--speech-gain` and `--base-gain` to quickly rebalance without remastering source files.

### 5. Strict Missing File Enforcement
Set `--missing-policy error` to abort if any `<index>.wav` is absent.

### 6. Large Project with Multiple Speakers
Option A: Merge all entries into one JSON (ensuring unique indices), place all files in one folder, run once.  
Option B: Mix per-speaker to full-length aligned tracks (using `--total-duration-ms` for consistency), then sum those tracks with a tiny follow-up script (future enhancement: multi-layer wrapper).

### 7. Avoid Normalization
If you already loudness-normalized segments externally, add `--no-normalize` to preserve exact levels. Mind clipping risk if segments overlap.

### 8. Choosing `float32` Path
If you expect heavy overlaps (e.g., multiple simultaneous voices) and want to apply external processing later, use `--dtype float32` for extra headroom before a final mastering stage.

## Performance Notes
- Complexity: O(total_samples + Σ segment_samples)
- Memory (int16 path): `duration_seconds * sr * channels * 2 bytes` (plus accumulator overhead ~8 bytes/sample until final cast). For 30 min mono @ 48 kHz ≈ ~165 MB accumulator + overhead.
- If memory becomes tight for multi-hour projects, implement a streaming/chunk mixer (future extension).

## Error Handling & Policies
- Invalid timing (start > end) raises immediately.
- Missing segments handled per `--missing-policy`.
- Buffer auto-extends if a segment runs past predicted end (logged via progress prints if interval hits).

## Resampling Details
- If `python-soxr` installed, high-quality band-limited resampling is used.
- Fallback: simple linear interpolation (adequate for most TTS speech when exact reproduction not critical).

## Output Formats
- Intermediate: WAV (PCM 16 or 32).
- Transcode: ffmpeg encodes once to chosen final format. Set `--transcode none` to skip.

## Suggested Validation Workflow
```powershell
# 1. Generate TTS segment WAVs into tts_out/
# 2. Inspect a few .wav files for sample rate/channels consistency
# 3. Mix
python -m utils.audio_mixer subs_chunk.json tts_out output/mix --sr 48000 --channels 1
# 4. Listen / loudness check
```

## Future Extensions (Ideas)
- Streaming (chunked) mixing for very long durations
- Loudness normalization to LUFS target (e.g., EBU R128)
- Optional dynamic range compressor
- Multi-layer JSON input in one invocation

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Output shorter than video | Last subtitle earlier & no total duration given | Use `--total-duration-ms` |
| Clipping/distortion | Overlapping segments exceed headroom & normalization disabled | Re-enable normalization or reduce gain |
| Missing file warnings | Gaps in generated indices | Regenerate or adjust policy |
| Different sample rates | Upstream TTS used varied SR | Provide `--sr` and let resample, or re-render uniformly |

---
**Enjoy faster mixing!** Report any issues or request features in future iterations.
