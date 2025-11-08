Here are some things I have been thinking of. Can you adjust the plan that we already have.
Like we can shorten the pauses in tts generated audio, also we should not have audio sped up more than 10-15%.

We can also add that we can analyze the subtitle overflows and also add that to the plan.



## The "Natural Speech" MVO Pipeline

### 0\. Preparation: Required Libraries

First, ensure you have the necessary tools for subtitles, audio processing, and high-quality time-stretching.

```bash
# For subtitles
pip install pysrt

# For TTS (assuming Silero)
pip install torch silero-models

# For audio manipulation (the workhorse)
pip install pydub

# For high-quality time-stretching
pip install librosa pyrubberband soundfile

# You ALSO need to install the command-line backends
# On Ubuntu/Debian:
# sudo apt-get update
# sudo apt-get install ffmpeg rubberband-cli
#
# On Windows/macOS:
# Install ffmpeg and rubberband-cli via their official sites or package managers (like Homebrew)
```

### 1\. The Gatekeeper: Text Pre-Analysis

**Goal:** Stop the pipeline if the source text is *so long* that no audio magic can save it. This is your check against 45-50% problems.

**Logic:** Estimate the raw speech duration from the text. If it's absurdly longer than the subtitle's allowed time, **fail** and demand a manual text edit (translation "trimming").

  * **Heuristic:** You must find *your* average characters per second. Start with a baseline, e.g., `AVG_CHARS_PER_SEC = 18.0`.
  * **Threshold:** Define a "failure" ratio, e.g., `MAX_TEXT_RATIO = 1.3` (meaning 30% longer is the absolute max we'll try to fix automatically).

<!-- end list -->

```python
import pysrt
import sys

# --- CONFIG ---
# (Tune this by testing your TTS)
AVG_CHARS_PER_SEC = 18.0 
# (Max text length vs. sub duration we'll try to fix)
MAX_TEXT_RATIO = 1.3 
# ---

subs = pysrt.open('your_project.srt')
problem_subs = []

for sub in subs:
    target_duration = sub.duration.to_seconds()
    if target_duration == 0:
        continue

    # Clean text for a more accurate length check
    clean_text = sub.text.strip().replace('\n', ' ')
    
    # Calculate estimated speech duration
    estimated_duration = len(clean_text) / AVG_CHARS_PER_SEC
    
    mismatch_ratio = estimated_duration / target_duration
    
    if mismatch_ratio > MAX_TEXT_RATIO:
        problem_subs.append({
            "index": sub.index,
            "text": clean_text,
            "ratio": mismatch_ratio
        })

if problem_subs:
    print("--- FATAL ERROR: TEXT ANALYSIS FAILED ---", file=sys.stderr)
    print(f"Found {len(problem_subs)} subtitle(s) that are too long to process.", file=sys.stderr)
    for item in problem_subs:
        print(f"  - #{item['index']} (Est. {item['ratio']:.1f}x too long): '{item['text']}'", file=sys.stderr)
    print("\nSOLUTION: Manually edit and shorten this text in the .srt file before re-running.", file=sys.stderr)
    sys.exit(1) # Exit the script

print("Text analysis passed. Starting audio generation.")
```

### 2\. Smart TTS Generation (with Silero)

**Goal:** Generate audio, but give the TTS a "hint" to speak slightly faster if needed.

**Logic:** Use the `rate` parameter in Silero. A `rate` of `1.15` (15% faster) at the *synthesis* stage sounds far more natural than a 15% *post-processing* time-stretch.

```python
import torch
import os

# --- Load Model (do this once) ---
device = torch.device('cpu') # 'cuda' if available
torch.set_num_threads(4)
local_file = 'model.pt'
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt', local_file)  
model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)
# ---

output_folder = "1_generated"
os.makedirs(output_folder, exist_ok=True)

for sub in subs:
    # Get values from Step 1
    target_duration = sub.duration.to_seconds()
    clean_text = sub.text.strip().replace('\n', ' ')
    estimated_duration = len(clean_text) / AVG_CHARS_PER_SEC

    # Decide on a pre-emptive speed-up
    # We aim for a 'gentle' speed-up, max 15%
    speech_rate = 1.0
    if estimated_duration > target_duration:
        # Calculate a rate, but cap it at 1.15
        speech_rate = min(1.15, estimated_duration / target_duration)

    # Use put_accent=True and put_yo=True for best quality
    audio_path = os.path.join(output_folder, f"sub_{sub.index}.wav")
    model.save_wav(text=clean_text,
                   speaker='xenia', # or any other speaker
                   sample_rate=48000,
                   put_accent=True,
                   put_yo=True,
                   rate=speech_rate, # Apply the smart rate
                   file_path=audio_path)
```

### 3\. Aggressive Silence Compression

**Goal:** Get "free" time savings by removing digital silence without distorting the voice.

**Logic:** TTS generates silence at the start/end and pauses between words that are often too long. We will *truncate* (shorten) all pauses to a minimal, natural-sounding value (e.g., 150ms). This step alone can save 10-25% of the total time.

```python
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- CONFIG ---
SILENCE_THRESH_DB = -40  # -40dBFS is a good starting point
MIN_PAUSE_LEN_MS = 250   # Look for pauses longer than 250ms
KEEP_PAUSE_MS = 150      # Truncate them to 150ms
# ---

processed_folder = "2_processed"
os.makedirs(processed_folder, exist_ok=True)

for sub in subs:
    in_path = os.path.join(output_folder, f"sub_{sub.index}.wav")
    out_path = os.path.join(processed_folder, f"sub_{sub.index}.wav")

    if not os.path.exists(in_path):
        continue

    audio = AudioSegment.from_wav(in_path)

    # Split audio on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_PAUSE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=KEEP_PAUSE_MS # <-- This is the magic!
    )

    if not chunks:
        # No silence found, maybe it's one short word.
        # Just trim the edges manually (not shown)
        final_audio = audio 
    else:
        # Re-combine the speech chunks with the short 150ms pauses
        final_audio = AudioSegment.empty()
        for chunk in chunks:
            final_audio += chunk

    final_audio.export(out_path, format="wav")
```

### 4\. The "Gap-Aware" Decision Engine

**Goal:** Implement your "overflow" logic to decide if a time-stretch is *actually* needed.

**Logic:** Compare the audio's `overflow` (how much it's *longer* than its subtitle) with the `gap` (the silent time *until the next* subtitle).

```python
import librosa

final_folder = "3_final"
os.makedirs(final_folder, exist_ok=True)

# We need the full sub list to look ahead
subs_list = list(subs) 

# This dictionary will store the *final* speed ratio needed.
# 1.0 = no stretch needed.
stretch_ratios = {}

for i, sub in enumerate(subs_list):
    processed_path = os.path.join(processed_folder, f"sub_{sub.index}.wav")
    
    if not os.path.exists(processed_path):
        stretch_ratios[sub.index] = 1.0
        continue

    # Get the *actual* duration of our cleaned-up audio
    current_audio_duration = librosa.get_duration(filename=processed_path)
    target_duration = sub.duration.to_seconds()

    overflow = current_audio_duration - target_duration

    # --- CASE 1: Audio fits perfectly or is short ---
    if overflow <= 0:
        stretch_ratios[sub.index] = 1.0 # No stretch
        continue

    # --- CASE 2: Audio overflows. Check the gap. ---
    
    # Get gap_duration
    gap_duration = 0.0
    if i < len(subs_list) - 1: # If not the last sub
        next_sub = subs_list[i+1]
        gap_duration = (next_sub.start.to_milliseconds() - sub.end.to_milliseconds()) / 1000.0
        if gap_duration < 0: # Handle overlapping subs
             gap_duration = 0.0

    # --- DECISION LOGIC ---
    
    # CASE 2a: Overflow fits in the gap. (Your brilliant idea)
    # We add a small 50ms buffer just to be safe.
    if overflow <= (gap_duration + 0.05): 
        stretch_ratios[sub.index] = 1.0 # No stretch! Let it overflow.
    
    # CASE 2b: Overflow is BIGGER than the gap.
    # We MUST stretch it to avoid a collision.
    else:
        # The new maximum time we have is target + gap
        max_allowed_duration = target_duration + gap_duration
        
        # Calculate the *minimum* speed-up required
        # e.g., 7s audio / 6s max_allowed = 1.16x speed-up
        speed_ratio_needed = current_audio_duration / max_allowed_duration

        # SAFETY CAP: Prevent extreme speed-ups.
        # If this happens, your MAX_TEXT_RATIO in Step 1 is too high.
        final_speed_ratio = min(speed_ratio_needed, 1.25) # Cap at 25%
        
        stretch_ratios[sub.index] = final_speed_ratio
```

### 5\. Final Processing & Assembly

**Goal:** Apply the (now minimal) time-stretch *only* where needed and build the final audio track.

**Logic:**

1.  Loop one last time, apply high-quality `pyrubberband` stretch *only if* `stretch_ratios[index] > 1.0`.
2.  Use `pydub` to create a final long track and `overlay` all the finished clips at their correct start times.

<!-- end list -->

```python
import pyrubberband as rb
import soundfile as sf
from pydub import AudioSegment

# --- Part 5a: Conditional Time-Stretching ---
print("Applying final time-stretching (where needed)...")
for sub in subs:
    in_path = os.path.join(processed_folder, f"sub_{sub.index}.wav")
    out_path = os.path.join(final_folder, f"sub_{sub.index}.wav")
    
    if not os.path.exists(in_path):
        continue

    ratio = stretch_ratios.get(sub.index, 1.0)
    
    # We add a small tolerance (e.g., 3%)
    if ratio > 1.03: 
        print(f"  - Stretching sub #{sub.index} by {ratio:.2f}x")
        y, sr = librosa.load(in_path, sr=None)
        
        # Use pyrubberband for high-quality stretching
        # Note: rb.time_stretch expects rate > 1.0 for speed-up
        y_fast = rb.pyrb.time_stretch(y, sr, ratio)
        
        sf.write(out_path, y_fast, sr)
    else:
        # No stretch needed, just copy the file
        import shutil
        shutil.copy(in_path, out_path)

# --- Part 5b: Final Assembly ---
print("Assembling final audio track...")
final_track = AudioSegment.silent(duration=0)

# Find the end time of the last subtitle to set track length
total_duration_ms = subs_list[-1].end.to_milliseconds() + 2000 # Add 2s padding

# Create a silent track of the total duration
final_track = AudioSegment.silent(duration=total_duration_ms)

for sub in subs:
    audio_file = os.path.join(final_folder, f"sub_{sub.index}.wav")
    
    if not os.path.exists(audio_file):
        continue
        
    clip = AudioSegment.from_wav(audio_file)
    
    # Overlay the clip at its precise start time
    start_time_ms = sub.start.to_milliseconds()
    final_track = final_track.overlay(clip, position=start_time_ms)

# Export the final MVO track
final_track.export(
    "FINAL_MVO_TRACK.mp3", 
    format="mp3", 
    bitrate="192k"
)

print("--- PIPELINE COMPLETE ---")
print(f"Final MVO track saved to FINAL_MVO_TRACK.mp3")
```

-----

## Conclusion: The New Strategy

By following this plan, you have transformed your pipeline from a "brute force" problem to an intelligent "triage" system:

1.  **Textual Errors (e.g., +50% length):** Are now **Rejected** by **Step 1**, forcing a manual text edit. This is the *correct* fix.
2.  **Audio Bloat (e.g., +15-25% length):** Is **Fixed** for free by **Step 3 (Silence Compression)** without sound distortion.
3.  **Minor Overflows (e.g., +10-20% length):** Are **Absorbed** by **Step 4 (Gap-Aware Logic)**, prioritizing natural speech over rigid timing.
4.  **True Collisions (e.g., +5-15% length):** Are **Resolved** by **Step 5 (Time-Stretching)**, but now the speed-up is minimal (e.g., 1.1x, not 1.5x) and uses a high-quality algorithm.

Your 45-50% speed-ups will disappear, replaced by a pipeline that either produces natural-sounding audio or tells you exactly *why* it can't (because the source text is the problem).
