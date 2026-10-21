# Local English STT with overlapping speakers

The default local presets are unchanged. Explicit remote upload mode and bounded
live direct verification are documented in [stt_api.md](stt_api.md).
The [bounded comparison](stt_api_comparison.md) includes actual episode/triple texts
and a successful production S46 hybrid export. A separate annotation-assisted
SepFormer3 diagnostic produced three ECAPA-bound stems and full third-source API
texts. Default PixIT still missed a voice on that fixture: this diagnostic does not
establish automatic three-voice production acceptance. All billing is reconciled.
Listening audio and captions are preserved in the
[durable ignored workspace snapshot](../temp/stt_api_results/README.md).
The separately authorized [full AU e04 MAI direct transcript](../temp/australian_survivor_e04_mai/README.md)
is complete (audio-only remote ASR), without local GPU/model execution.

Python 3.12, FFmpeg and ffprobe are required. This independent environment does not
install or modify the existing TTS requirements. CPU-only PyTorch wheels are pinned;
the CUDA driver is not needed. The legacy `pyannote.audio` 3.3.2 stack is intentional.
SpeechBrain 1.0.3 preserves the `from_hparams(use_auth_token=...)` API used by this
pyannote release; 1.1.1 does not. Transformers 4.45.0 correctly accounts for legacy
WavLM `weight_g/weight_v` checkpoint keys with PyTorch parametrized weight normalization.
Lightning and pytorch-lightning are aligned at 2.4.0.

```bash
uv venv vosk_env
source vosk_env/bin/activate
uv pip install -r requirements-stt.txt --index-strategy unsafe-best-match
uv run --active -m utils.stt_pipeline input.mkv -o output/english \
  --start 60 --duration 30 --preset fast
```

Asteroid's indirect dependency `pesq` needs a C compiler and Python development
headers when no wheel is available. On Linux, install the matching Python 3.12
development headers before installation. In the checked workspace the headers
were extracted locally under ignored `vosk_env/build-headers`, not installed system-wide.
The successful preparation command (after obtaining matching headers) was:

```bash
# vosk_env must already be activated; NumPy 1.26.4, Cython and wheel must be installed.
CFLAGS="-I$PWD/vosk_env/build-headers/usr/include/python3.12 -I$PWD/vosk_env/build-headers/usr/include" \
  uv pip install pesq==0.0.4 --no-build-isolation
```

## Authentication and local processing

### Current GPU preflight (2026-10-12)

PCI discovery sees NVIDIA **RTX 1000 Ada Generation Laptop GPU** and Intel Meteor
Lake-P graphics (`i915`, `/dev/dri/card1` and `renderD128`). NVIDIA had no bound
driver; `nvidia-smi` could not communicate with it (exit 9). Driver version/VRAM
were therefore unknown, not inferred from the product name. PyTorch 2.5.1+cpu has
no CUDA build/devices; ONNX Runtime 1.31.0 offers Azure/CPU providers, not CUDA;
CTranslate2 4.8.2 sees 0 CUDA devices. Hardware potential is not configured acceleration.

These STT functions explicitly select CPU (PixIT, SepFormer, Parakeet ONNX,
Whisper and gender ONNX); the CLI has no device flag. GPU acceleration would require
working backend/driver support and reviewed device plumbing, not just GPU hardware.
No driver/CUDA/dependency installation or speed benchmark was performed. Local GPU
cannot accelerate remote MAI provider inference; the full-episode direct run does
not load local separation/ASR models. [Saved preflight](../temp/australian_survivor_e04_mai/PREFLIGHT.md).

### Gated local models

Accept **both** gated MIT model conditions using your Hugging Face account:

- https://huggingface.co/pyannote/speech-separation-ami-1.0
- https://huggingface.co/pyannote/separation-ami-1.0

With the environment activated, run `huggingface-cli login` locally, or provide
`HF_TOKEN` through your local environment. Never put the token in this repository,
command arguments, output files, or chat. The pipeline detects missing credentials
before gated weight downloads and checks access to the underlying checkpoint first.
It does not upload audio. Model downloads are the only network processing; HF
telemetry and implicit account credentials for public ASR downloads are disabled.
The installed libraries execute the models; `trust_remote_code` is not enabled.

Model caches default to ignored `models/stt`; change with `--model-dir`.
HF downloads use its resumable/atomic cache. The gender model is downloaded through
a temporary file, verified by SHA256, then atomically renamed. No archives or remote
Python source are extracted or executed by this module.

## Processing and presets

FFmpeg decodes the selected original-media interval to mono float32 at 16 kHz.
One PixIT pipeline call performs joint diarization, separation and global clustering.
The pipeline's raw `discrete_diarization` hook artifact is held by reference (including
later leakage-removal mutations) and converted with the same `to_annotation` settings
after inference. Its integer cluster labels identify source columns directly. Returned
human-readable labels are **not** channel indices: upstream string-sorts integers
(`0,1,10,11,2,...`) before renaming, so label positions are unsafe with 10+ global speakers.
Public IDs are assigned independently by first occurrence. Inactive source columns and
sparse cluster IDs are allowed; an active cluster without a source channel fails with
counts and review guidance, never an invented voice. Retrying a known `--num-speakers`
count may help upstream overcounting but is not a guaranteed fix.
Each bound source is recognized on its own active intervals. ASR models are loaded once, with
disjoint chunks capped at 28 seconds. Channels, sample timing and output lengths
are validated; mismatches fail rather than fabricate a source-to-speaker match.

When a sweep of **predicted activity** finds three concurrent IDs, the original mix
in those spans is refined by pinned **SepFormer WSJ03mix**, not a two-source separator.
Global source count alone never triggers it. Adjacent same-ID spans separated by at
most 0.32 seconds share context; cores are at most 8 seconds with up to 1 second of
padding on each side (at most 10 seconds per inference). Context excludes additional
global speakers. The model is loaded once only when needed. Audio is explicitly
resampled 16→8→16 kHz and trimmed to the exact original sample length/origin.

New source order is **not** assumed: the already-loaded PixIT ECAPA produces clean
original-speech profiles (at least 3 seconds per identity; normalized excerpt vectors,
mean, L2 normalization), and embeddings of separated sources. All six one-to-one
permutations are compared. Each assigned cosine must be at least 0.45, the winning
mean-cosine margin over the runner-up at least 0.08, and pairwise source-embedding
cosines below 0.90. These are conservative heuristics, **not calibrated probabilities**.
Missing/short/silent profiles, nonfinite embeddings, duplicate sources or ambiguous
assignments fail with `needs review`, rather than silently collapse voices or invent
identities. Source evidence shorter than 2 seconds also requires review. Production
never uses fixture IDs, reference transcripts or known isolated test waveforms.

Confidently matched waveforms replace the failed PixIT audio, not sum with it. Joint
least squares against the original mixed waveform restores new-source gain/polarity;
old tracks are calibrated by a single least-squares gain on their own clean original
speech, avoiding huge PixIT/context gain swallowing quiet repaired speech. No hard
clipping or arbitrary peak boosting is used. Only 40ms boundary crossfades retain a
transition; adjacent bounded windows keep the same global identities and sample grid.
Matched context windows are additional, unioned ASR routes (no overlapping duplicate
chunks), split at repaired-context boundaries so repeated clean/overlap phrases are
not collapsed together by Whisper. Sub-0.32s edge fragments keep neighboring context
instead of becoming separate tiny ASR calls. These routes are **not fabricated
diarization or gender evidence**. Gender still uses original
non-overlapping speech. One/two-speaker processing does not load/download the refiner.

`fast` uses onnx-asr 0.12.0 and English Parakeet TDT v2 int8. `accurate` uses
faster-whisper 1.2.1, `large-v3-turbo`, CPU int8, English word timestamps and VAD.
These are practical presets, **not benchmark-proven universally fastest/best choices**.
`--whisper-model` accepts a different Whisper name or local model directory.
Parakeet token starts are model timestamps in seconds; word ends are approximated
from the next word/chunk boundary, capped 0.8 seconds after the last piece. This is
not forced alignment. Whisper word timestamps are also estimates. At most one 80ms
grid frame of positive boundary overshoot is clamped to the clip end, preserving the
word (possibly with zero duration); nonfinite, backward or larger out-of-clip times fail.

Options: `--start` (nonnegative seconds), `--duration` (positive seconds),
`--threads` (1–32, default 4 or available CPU count), `--num-speakers` (known positive
**global** count), `--gender-threshold` (default 0.8), and `--speaker-genders`.
All exported times include the crop offset **once**, relative to the original media.
Invalid paths/options, output/input collisions and malformed overrides fail before
model downloads. Unknown override IDs fail after diarization and before ASR.
Digital silence writes empty JSON, SRT and speaker metadata without loading models.

## Output and review

- `PREFIX.json`: chronological subtitle array, contiguous 1-based `index`, integer
  `start`, `end`, `duration` in milliseconds, `text: list[str]`, `speaker_id`,
  `gender`, nullable `gender_confidence`, `gender_source`, `overlap`,
  `separation_status` and `needs_review`.
- `PREFIX.srt`: plain transcription text, using the existing JSON-to-SRT utility.
- `PREFIX.speakers.json`: raw numeric cluster/source channel/upstream label mapping, acoustic scores and durations,
  manual override flags; matched refinement windows/cosines/margins, model revision and
  clean-source calibration gain when used. Window times are clip-relative seconds;
  subtitle times remain original-media milliseconds. Top-level source indices refer
  to the PixIT matrix; nested refinement source indices refer to each SepFormer window.

IDs are assigned by first occurrence within the analyzed clip and are not guaranteed
to match another crop or recording. Concurrent speakers keep separate cues with
overlapping timestamps; identical phrases from different sources are not deduplicated.
SRT players handle simultaneous cues differently: JSON is authoritative.
Gender unknown never becomes male in these outputs. Existing TTS consumers may
choose their own default voice for unknown labels; that is not an acoustic estimate.

The classifier uses **original non-overlapping speech**, never pitch alone or the
mixed overlap. It requires at least 3 seconds of clean evidence, selects several
1–5 second excerpts up to 15 seconds total, and abstains when confidence is low or
excerpt predictions conflict. Confidence is a normalized model score, **not calibrated
certainty**. Binary adult-English voice labels do not identify true gender, children,
nonbinary people, identity, or grammatical gender. Noise, music, acted voices, model
separation leakage and missed activity can invalidate estimates. Review and correct
Russian grammatical gender contextually before translation/TTS.

Supply a JSON object such as `{"SPEAKER_00":"female","SPEAKER_01":"unknown"}` with
`--speaker-genders corrections.json`. Manual labels take precedence and skip acoustic
classification for those speakers; manual confidence is null. Explicit unknown is honored.

PixIT has **three local simultaneous source slots**, with arbitrary global speakers.
Predicted triple-overlap cues get `needs_review=true`. Four or more simultaneous
voices are outside scope; the cap does not reliably detect hidden fourth voices.
Meeting-trained separation is not guaranteed to recover every voice in noisy TV.
Recognition on predicted activity can miss speech when segmentation misses it.
Refinement cannot discover hidden three-way overlap missed by PixIT. SepFormer was
trained on WSJ03mix speech mixtures, not TV/music; confident matching does not prove
perfect separation. Any triple overlap/refined boundary needs review. Bounded-window
tests do not establish long-recording quality or throughput.
This is not lightweight: CPU processing and memory can be substantial; the full
selected waveform, window outputs and separated global tracks reside in memory.
Use bounded excerpts first, not entire episodes. Source WAVs are not exported by default.

## Models and attribution

| Model/runtime | Pinned source | License / size |
|---|---|---|
| PixIT pipeline | `pyannote/speech-separation-ami-1.0@9486b106945ae0cc0784041a08bfcdba5edadfb9` | gated MIT |
| ToTaToNet checkpoint | `pyannote/separation-ami-1.0@4d38e95cfd067c894b8b60b00761831fb01e4a8c` | gated MIT; 1,277,290,690 bytes |
| WavLM / speaker embeddings | `microsoft/wavlm-large` / dependencies from pinned pipeline config | WavLM MIT, ~1.2 GB plus embeddings; library-managed dependencies |
| Three-source refinement | `speechbrain/sepformer-wsj03mix@21e78b44a378092c34fff32ba8e788a73584f474` | Apache-2.0; 113,407,611 bytes / ~108.2 MiB for four assets |
| Parakeet int8 | `istupakov/parakeet-tdt-0.6b-v2-onnx@0bbb45a3365852604aef28b538a8f066f4ccaa85` | NVIDIA weights CC-BY-4.0; 661,182,300 bytes for graphs; onnx-asr MIT |
| Whisper | faster-whisper `large-v3-turbo` | MIT; ~1.62 GB; library-managed model revision |
| ECAPA gender | `Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2@8aedeb1a399945e52e6f3f61714abe07c1c48742` | MIT; 16,057,697 bytes |

Gender file: `onnx/model_quantized.onnx`; SHA256
`fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252`.
Input is raw float32 `[batch,samples]`; STFT/log-mel preprocessing is built in.
Output logits index 0 = male, index 1 = female, followed by stable softmax.
The label contract is recorded in the pinned public config:
https://huggingface.co/Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2/blob/8aedeb1a399945e52e6f3f61714abe07c1c48742/config.json
This verified mapping does not provide ground-truth validation of individual voice estimates.

PixIT reference: Kalda, Pagés, Marxer, Alumäe and Bredin, *PixIT: Joint Training of
Speaker Diarization and Speech Separation from Real-world Multi-speaker Recordings*,
Odyssey 2024, https://arxiv.org/abs/2403.02288. Retain upstream model licenses and
CC-BY attribution when redistributing weights; weights are not part of this repository.

SepFormer allows exactly `hyperparams.yaml`, `encoder.ckpt`, `masknet.ckpt`,
`decoder.ckpt` from the pinned snapshot. All four SHA256 values are verified before
loading (checkpoint hashes also matched HF LFS metadata in the pilot); no remote
Python/custom.py is downloaded/executed. SpeechBrain loads from the verified local
snapshot and a revision-specific runtime cache, not a floating direct-HF pretrainer.
**HyperPyYAML executes trusted pinned configuration constructors and SpeechBrain 1.0.3
uses legacy Torch pickle checkpoint loading; this is not a weights-only security claim.**
No global safe-loading bypass is applied. Model card/config:
https://huggingface.co/speechbrain/sepformer-wsj03mix/tree/21e78b44a378092c34fff32ba8e788a73584f474

## Checks

```bash
source vosk_env/bin/activate
uv run --active -m unittest discover -s tests -v
STT_RUNTIME_CHECK=1 uv run --active -m unittest discover -s tests -v
uv run --active -m compileall -q utils/stt_pipeline.py utils/stt_refiner.py tests/test_stt_pipeline.py
uv run --active -m utils.stt_pipeline --help
```

Pure tests use fake source metadata to verify triple simultaneous cues/channel
mapping, BPE grouping, offsets, unknown/manual gender and output compatibility.
Opt-in `STT_PUBLIC_SMOKE=/path/to/english.mkv` enables a **30-second clip covering the
original-media interval 60–90 seconds**, for public ASR and gender-only checks;
add `STT_SMOKE_PRESET=accurate` for a 10-second Whisper check (interval 60–70 seconds).
These are not full separation checks and not gender accuracy measurements.

### Bounded native acceptance evidence (2026-10-11)

The following is historical execution evidence. Its published test filenames were
not independently cross-checked against canonical dataset waveforms. Later checks
found that the icefall WAV named `1089-134686-0001` differs from the exact HF-row
FLAC; the historical IDs must not be read as newly verified human/corpus identity.
The new validated full-utterance fixture and its separate overlap-detection failure
are documented in [stt_api_comparison.md](stt_api_comparison.md).

After local authorization, both AMI gates returned HTTP 200. The full native pipeline
was executed on the original English TV media interval 60–90 seconds: 5 cues,
2 chronological speakers, raw source columns `[1, 0]`, times 60,320–90,000 ms.
The 60–70 second full accurate run produced 3 cues and 2 predicted speakers; its
short second speaker abstained from gender estimation. These unlabelled TV results
are execution/schema/offset checks, not ground-truth diarization or gender scores.
Crop-dependent segmentation can omit short speech: outputs need listening/review.

A controlled **real-speech**, not TTS, fixture used LibriSpeech clean utterances
`1089-134686-0001`, `1221-135766-0002`, `1272-128104-0000` (different annotated
speaker IDs; CC BY 4.0, https://www.openslr.org/12/). Gain-only RMS normalization
to 0.08, mono 16 kHz; each voice had a 4-second clean prefix at 0, 5, 10 seconds,
all three first 4.825-second source excerpts were summed at 15–19.825 seconds,
then 4-second clean suffixes at 20.825, 25.825, 30.825 seconds. Total: 35.825 seconds.
Only this known three-speaker fixture used `--num-speakers 3`, not the TV excerpt.

**Restoration provenance caveat (2026-10-12):** HF's dataset row named
`1089-134686-0001` is **3.275s**, with human gold “STUFF IT INTO YOU HIS BELLY
COUNSELLED HIM”; the public icefall test WAV named the same ID used by the newly
persisted reconstruction is **6.625s**, isolated ASR “After early nightfall…”.
The HF-row human gold must not score that different waveform. The mirror-source
35.825s production reconstruction has its own native evidence, but no historical
byte-identity or canonical first-source gold claim. Historical measured results
below remain history; content coverage is diagnostic against isolated ASR, not WER.
See [source-variant evidence](stt_api_comparison.md#billing-reconciliation-and-artifacts).

The initial fast native run produced 10 diarization turns, 9 cues and 3 nonempty
sources. Clean prefixes establish IDs 00/01/02 as reference speakers 1089/1221/1272
and raw columns 1/2/0. Predicted triple activity covered 3.624 seconds; three cues
shared 15.512–18.408 seconds. The accurate native run produced 11 cues, but the
third channel recognized the second voice's phrase within overlap, rather than its
own reference. **Neither three populated outputs nor cue overlap proves recovery
of three genuine voices.** Initial cue-based recall was invalidated because a cue
straddled clean context and overlap. Evaluation must restrict individual timed word
starts to 15–19.825 seconds and retain the identity established by clean prefixes,
never remap voices by whichever leaked transcript best matches a reference.
Saved PCM16 source replay also changed hypotheses and is diagnostic, not identical
to original float32 inference. The final controlled check saves native FLOAT WAVs
and original timed words to avoid these evaluation artifacts.
The baseline PixIT-only `pixit_triple_verified` run recovered 9/13, 6/11, **0/10** reference
content words within the true overlap, using original word timestamps and fixed
clean-prefix identities. The third hypothesis was “Yet these thoughts affected Task
to print last year”, not the Quilter/apostle/middle-classes reference. Clean FLOAT
waveform/reference absolute correlations were 0.870/0.987/0.893, but overlap still
contained substantial crosstalk. **At that stage three-voice recovery was not
accepted**, despite valid
three-source binding, activity and output schema. These recall numbers are diagnostic,
not WER or a general accuracy benchmark; the full first/third references exceed the
mixed excerpt.

Native source peaks were 85–115 (arbitrary SI-SDR/overlap-add gain). PCM16 diagnostics
therefore clipped and must not be used as lossless native evidence. Separate gain-only
ASR probes on cached FLOAT sources also failed to recover the third reference with
both presets and worsened some fast hypotheses. No speculative peak normalization
was retained. The successful later refiner uses original-speech/mix least-squares
gain calibration with a genuinely better three-source separator instead.
The pinned settings remained unchanged: segmentation threshold 0.82, clustering
threshold 0.68/minimum cluster size 15, leakage removal enabled with 0.32-second
collar. Three clusters and several seconds of triple activity were present, so the
poor second hypothesis cannot be attributed solely to a two-speaker cap or label
indexing; source leakage and missed boundary activity remain model limits.

### Bounded SepFormer3 remedial trial and production acceptance

Only one specialized model was tried: the pinned SepFormer WSJ03mix above. The
pilot used original mixture 14–20.825 seconds, explicit 16→8→16kHz resampling and
FLOAT sources. Diagnostic isolated-waveform matching found source order `[2,0,1]`,
absolute overlap correlations 0.975/0.869/0.916 and fast timed content-word matches
10/13, 11/11, 7/10. Its third hypothesis recovered Quilter/apostle/middle classes,
not a copy of the second voice. Pilot load/download took 13.08s, separation 5.07s;
process 23.62s, max RSS 1,765,404 KiB. This diagnostic association is not production
identity matching.

Full production retained canonical PixIT activity/clusters and numeric-column
binding. The activity sweep produced a single context window 14.392–20.088s from
the two nearby three-active-ID spans. **Production ECAPA matching alone** selected
`[2,0,1]`: assigned cosines 0.599/0.586/0.489, mean winning margin 0.318. Clean
fixture references were used only afterward to evaluate the fixed global IDs.
The first accurate integration run exposed Whisper collapsing the repeated clean
and overlap third phrase; disjoint ASR routing at repaired-context boundaries fixed
this without moving timestamps, deduplicating text or reassigning voices.

Final native runs on the unchanged 35.825s mixture:

| Preset/artifact prefix | Timed overlap content-word matches (1089/1221/1272) | Cues | Pipeline time / RTF | Process time / max RSS KiB |
| --- | --- | --- | --- | --- |
| `pixit_sepformer_fast_final` | **11/13, 11/11, 7/10** | 9 | 204.03s / 5.695 | 207.31s / 4,282,072 |
| `pixit_sepformer_accurate_final` | **11/13, 11/11, 7/10** | 10 | 245.30s / 6.847 | 248.23s / 4,734,520 |

Both recover all three own phrases within the scheduled 15–19.825s overlap. Fast
third text: “Mr.Quilter is the Apostle of the Middle Classes, and we are glad to
welcome him.” Accurate: “Mr. Quilter is the apostle of the middle classes, and we
are glad to welcome you.” These retain ASR errors, not a perfect transcription claim.
Own-channel overlap correlations are 0.977/0.870/0.918; off-diagonal correlations
are at most 0.053. Final source peaks are 0.765/0.641/0.480, without PCM16 clipping.
Original native timed words, FLOAT sources, JSON/SRT/metadata and evaluation are
preserved per prefix. Integer-ms schema/SRT parity, chronological IDs/raw channels
`[1,2,0]` and conservative gender evidence were validated. Offset/helper regressions
and one/two-voice no-refiner tests passed; unchanged original-media runs were not
repeated. Final jobs overlapped in time with each other: timings are not controlled
single-job benchmarks.

**The bounded fixture's all-three recovery criterion is met in both presets.** This
does not certify unknown recordings, hidden overlap missed by PixIT, noisy TV,
gender accuracy or long-window/hours-long performance. Matching thresholds remain
review heuristics; missing/ambiguous identities fail explicitly. Content-word scores
use the existing full isolated ASR references (first/third extend beyond the excerpt),
not WER or token aggregates across speakers.

Reproduce from the repository root with `vosk_env` active:

```bash
uv run --active -m temp.sepformer_pilot
uv run --active -m temp.stt_acceptance triple_run /tmp/opencode/stt_smoke/triple_recipe.json --prefix /tmp/opencode/stt_smoke/pixit_sepformer_fast_final --preset fast
uv run --active -m temp.stt_acceptance triple_run /tmp/opencode/stt_smoke/triple_recipe.json --prefix /tmp/opencode/stt_smoke/pixit_sepformer_accurate_final --preset accurate
```

The ignored `temp` helpers only capture local evidence around the production
`run_pipeline`; no installed library code is patched. The ordinary production CLI
uses the same refinement route by default when actual predicted triple activity exists.

Local inspection artifacts (not committed): `/tmp/opencode/stt_smoke/` contains
`pixit_fast.*`, `pixit_accurate.*`, `triple_mix.wav`, the three reference WAV/FLACs,
`triple_recipe.json` (source URLs, transcripts, schedule and isolated ASR), and
`pixit_triple*.*` (JSON/SRT/metadata, raw turns/mapping, separated WAVs and evaluation;
`pixit_triple_verified` retains original timed words and FLOAT WAVs).
`triple_gain_{fast,accurate}.*` are cached-source ASR diagnostics, not full native runs.
The old `triple_evaluation.json` is explicitly marked invalidated; use
`pixit_triple_verified.evaluation.json` for the PixIT-only baseline. Final remedial
evidence is `pixit_sepformer_{fast,accurate}_final.words.json` and `.evaluation.json`;
`sepformer_route_*` are cached-source routing diagnostics, not fresh full runs.
No audio was uploaded; model/fixture downloads were the only network transfers.
