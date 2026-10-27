# STT benchmarks (Australian Survivor S10, VoxConverse)

Measured 10.2026 on the RTX 4070 Ti SUPER 16 GB; the clips and model outputs were deleted from
`output/stt_bench/` on 27.10.2026, the numbers stay here. Commands: `docs/stt_local.md`, Benchmarks.

### Results (RTX 4070 Ti SUPER 16 GB, Australian Survivor S10)

ASR, three 10-minute clips (E01 10–20 min, E05 25–35 min, E10 40–50 min; confessionals,
challenges with crowd noise, music), reference = English SDH subtitles. Subtitles are not
verbatim (no fillers, repetitions, false starts), so even a perfect model scores well above 0%,
and models that write "subtitle style" (Whisper, trained on web captions) are favoured over
verbatim ones (Parakeet/Canary). WER after the repetition guard, pooled over the clips:

| Model | WER | RTFx | Peak VRAM, MB | Word times |
|---|---:|---:|---:|:---:|
| **LLM fusion: cohere + whisper-large-v3 + qwen3-asr-1.7b** (gpt-6-luna; two runs) | **16.95–17.22%** | — | — | via aligner |
| whisper-large-v3 + cast names (`--hotwords-file`) | 18.98% | 87 | 10852 | yes |
| cohere-transcribe | 19.61% | 217 | 5694 | aligner |
| whisper-large-v3 | 20.21% | 109 | 8068 | yes |
| whisper-medium.en | 20.27% | 179 | 4612 | yes |
| qwen3-asr-1.7b | 20.48% | 52 | 8340 | yes |
| qwen3-asr-0.6b | 20.63% | 39 | 5808 | yes |
| whisper-large-v3-turbo | 21.23% | 210 | 4420 | yes |
| parakeet-unified-en-0.6b | 21.88% | 343 | 7992 | yes |
| voxtral-mini-3b | 22.09% | 58 | 10106 | aligner |
| parakeet-v2 | 22.60% | 412 | 8170 | yes |
| distil-large-v3.5 | 22.63% | 219 | 4324 | yes |
| parakeet-v3 | 23.02% | 399 | 8300 | yes |
| parakeet-tdt_ctc-1.1b | 23.50% | 350 | 8612 | yes |
| kyutai-stt-2.6b | 23.62% | 3 | 6268 | aligner |
| canary-qwen-2.5b | 24.39% | 44 | 6846 | aligner |
| canary-1b-v2 | 25.32% | 29 | 14228 | aligner |
| moonshine-streaming-medium | 25.83% | 119 | 2830 | aligner |
| granite-speech-4.1-2b | 26.58% | 44 | 5756 | aligner |

E01 clip only: voxtral-realtime-4b 12.35% (RTFx 3), parakeet-tdt-1.1b 15.25%, granite-speech-4.1-2b-plus
17.76%, canary-180m-flash 69% (loops). Per clip the fusion gives 8.6–8.8% / 16.5% / 27.8–28.8%
(E01 / E05 / E10) against 10.1% / 18.4% / 32.3% for the best single model of each clip.
Whole S10E01 episode (89 min, 10.7k reference words): whisper-large-v3 14.97%, cohere 15.51%,
qwen3-asr-1.7b 16.42%, fusion 13.37%. Repetition guard: before it Canary 1B reached 62% and
Granite 37% on E05 (loops like "go, go, ..." ×500). Cast names as hints lowered Whisper's
insertions on the noisy E10 clip (160 → 97), but added nothing measurable on top of fusion.

Diarization, VoxConverse test shard 6 (21 files, 192 min, 2–15 speakers):

| Model | DER (0.25 s collar) | DER strict | Exact speaker count | RTFx | Peak VRAM, MB |
|---|---:|---:|---:|---:|---:|
| diarizen-large-v2 | 8.96% | 12.30% | 48% | 34 | 13466 (batch 32) |
| pyannote-community-1 | 11.87% | 15.64% | 43% | 86 | 2274 |

On S10E01 (89 min, ~25 people, SDH speaker labels): pyannote 16 speakers, purity 80.7%,
7 of 20 named people merged, 68 s; DiariZen 17 speakers, 77.9%, 8 merged, 172 s (batch 8,
4.6 GB; batch 32 runs out of 16 GB). DiariZen settings (`max_speakers` 20→40, `min_speakers`,
AHC 0.6→0.5) barely change this: the limit is the speaker-embedding model on short TV lines.

Cue segmentation on S10E01: speaker changes inside a sentence are moved to the nearest
sentence boundary (≤3 words) and pauses split an unfinished sentence only after 2 s. Cues
ending mid-sentence: 31% → 14% (speaker-change splits 228 → 63) with unchanged word-level
SDH purity (81.7% → 81.2%).

Gender/age on S10E01: audeering and TeraAnaly agreed on all 17 speakers (16 confident + one
mixed cluster `unknown` in both); 2.4 s for the episode (+7 s model load).

Full S10E01 runs (89.3 min audio):
- fast: pyannote 68 s + whisper-large-v3 47 s + gender 2 s ≈ 2–3 min GPU;
- accurate (recommended): pyannote 68 s + cohere 23 s + whisper 48 s + qwen3 115 s + fusion
  (API, parallel) + alignment 20 s + gender ≈ 5–6 min; LLM cost per episode ≈ $0.09
  (fusion $0.033, glossary $0.004, names $0.006, translation $0.045–0.05).

LLM steps on S10E01: naming with `openai/gpt-6-luna` found real names for 12–15 of 16–17
speakers ($0.006); a glossary pass (`stt.glossary`, $0.004) fixed name spellings
(Shonee, Jordie, Anjali, Benjamin Law, Jonathan LaPaglia) and gave stable Russian renderings.
Translation, whole episode:

| Variant | Model, reasoning | Cost, $ | Time, s | Over budget | 1st-person gender errors |
|---|---|---:|---:|---:|---:|
| luna | openai/gpt-6-luna, default | 0.037 | 460 | 208 (> 1.2× flat 14 cps) | 0 / 47 |
| luna_low | openai/gpt-6-luna, low | 0.021 | 195 | 297 (same) | 0 / 56 |
| haiku | anthropic/claude-haiku-5.5, default | 0.081 | 596 | 177 (same) | 0 / 36 |
| deepseek | deepseek/deepseek-v4.1-flash-1017, off | 0.009 | 189 | 311 (same) | 1 / 56 |
| qwen | qwen/qwen3.8-flash, off | 0.025 | 406 | 188 (same) | 2 / 44 |
| glm | z-ai/glm-5.3-flash, default | 0.022 | 937 | 389 (same) | 0 / 67 |
| luna + time windows + shortening | gpt-6-luna, medium | 0.046 | 660 | 50 → 0 (> 1.15× window budget) | 0 / 52 |
| luna + Vosk profile + shortening | gpt-6-luna, medium | 0.048 | 667 | 71 → 0 | 0 / 51 |
| luna + glossary + shortening, 4 parallel parts | gpt-6-luna, medium | 0.051 | 308 | 51 → 0 | 0 / 44 |
| existing `.ru.ass` | (machine translation shipped with the files) | — | — | — | 26 / 92 |

"Gender errors" = Russian past-tense verbs right after "я" whose ending disagrees with the
speaker's gender (heuristic). The remaining occasional slips (~2–4% before the check) are now
caught by a final check pass in `stt.translate` that re-asks only the flagged lines. deepseek and
qwen ignore `reasoning.effort` and think until the 16k token limit (empty answers), so they need
`--reasoning off`.


## Per-model summary over the three clips (`stt.bench summary`)

| Model | Clips | WER (pooled) | Audio, s | RTFx | Peak VRAM, MB |
|---|---:|---:|---:|---:|---:|
| voxtral-realtime-4b | 1 | 12.35% | 599 | 3 | 9120 |
| parakeet-tdt-1.1b | 1 | 15.25% | 599 | 249 | 10346 |
| granite-speech-4.1-2b-plus | 1 | 17.76% | 599 | 65 | 5392 |
| whisper-large-v3@hw | 3 | 18.98% | 1791 | 87 | 10852 |
| cohere-transcribe | 3 | 19.61% | 1791 | 217 | 5694 |
| whisper-large-v3 | 3 | 20.21% | 1791 | 109 | 8068 |
| whisper-medium.en | 3 | 20.27% | 1791 | 179 | 4612 |
| qwen3-asr-1.7b | 3 | 20.48% | 1791 | 52 | 8340 |
| qwen3-asr-0.6b | 3 | 20.63% | 1791 | 39 | 5808 |
| whisper-large-v3-turbo | 3 | 21.23% | 1791 | 210 | 4420 |
| parakeet-unified-en-0.6b | 3 | 21.88% | 1791 | 343 | 7992 |
| voxtral-mini-3b | 3 | 22.09% | 1791 | 58 | 10106 |
| parakeet-v2 | 3 | 22.60% | 1791 | 412 | 8170 |
| distil-large-v3.5 | 3 | 22.63% | 1791 | 219 | 4324 |
| parakeet-v3 | 3 | 23.02% | 1791 | 399 | 8300 |
| parakeet-tdt_ctc-1.1b | 3 | 23.50% | 1791 | 350 | 8612 |
| kyutai-stt-2.6b | 3 | 23.62% | 1791 | 3 | 6268 |
| canary-qwen-2.5b | 3 | 24.39% | 1791 | 44 | 6846 |
| canary-1b-v2 | 3 | 25.32% | 1791 | 29 | 14228 |
| moonshine-streaming-medium | 3 | 25.83% | 1791 | 119 | 2830 |
| granite-speech-4.1-2b | 3 | 26.58% | 1791 | 44 | 5756 |
| granite-speech-4.1-2b-plus@hw | 3 | 37.55% | 1791 | 35 | 5338 |
| canary-180m-flash | 1 | 69.09% | 599 | 11 | 10856 |
