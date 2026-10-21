# Primary model verification — cutoff 2026-01-31

These notes persist the supplied anonymous primary-source research findings, dated
**2026-10-12**, without new network access, downloads, license acceptance or runs.
Eligibility refers to capabilities available by **2026-01-31**, not arbitrary
features on today's main branches. Current-only facts and historical uncertainty
are explicitly marked. Archive dates/quotes/photos are in [Telegram notes](telegram_archive.md).

## Speaker diarizers do not output isolated waveforms

**WeSpeaker VoxBlink2 / SimAMResNet100** provides speaker embeddings and clustering,
not three isolated speaker WAVs. The official repository announced VoxBlink2 support
on **2024-09-03**. Pretrained names include `voxblink2_samresnet100` and its `_ft`
VoxCeleb2 variant; the archive's `voxblink_100m`/Voxblink2 100 labels cannot identify
the exact variant without the original configuration. Code is Apache-2.0; weight/
dataset commercial eligibility is not established by that code license alone.
([Repository](https://github.com/wenet-e2e/wespeaker),
[Pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)).

**DiariZen `BUT-FIT/diarizen-wavlm-large-s80-md`** is EEND with WavLM Large and
Conformer, producing speaker activity/RTTM rather than separated WAVs. The pinned
**2025-12-09** README revision
`a9b1b0e7974d96dcfd63af417e9da7ad8714040f` says it is optimized for **not more than
three simultaneous speakers** and lists **CC-BY-NC-4.0 weights / MIT code**.
([Pinned model card](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/blob/a9b1b0e7974d96dcfd63af417e9da7ad8714040f/README.md)).

Today's main card was updated in August 2026; do not backdate that content. History
includes November weight replacement followed by original-weight restoration on
December 9: reproducing the October screenshot requires its October revision,
not assuming today's main weights are identical.
([Commit history](https://huggingface.co/api/models/BUT-FIT/diarizen-wavlm-large-s80-md/commits/main)).
A DiariZen overlap-trigger study feeding the existing SepFormer is a **conditional
research candidate**, not an implemented repair. Noncommercial weight terms require
user approval/acceptable use before a download or run; previous gated-model acceptance
does not imply permission for commercial use of these weights.

## Separate low-level source slots from end-to-end success

**PixIT / ToTaToNet**: the low-level `pyannote/separation-ami-1.0` model uses mono
16kHz, 5-second inputs and outputs activity **[B,624,3]** and sources **[B,80000,3]**.
`pyannote/speech-separation-ami-1.0` is the long-form pipeline producing Annotation
and separated tracks with embedding-based stitching. Three output slots do not prove
automatic third-voice detection or recovery on our real fixture. PixIT dates to
Odyssey 2024. Current cards show MIT/gated access; anonymous historical pinned-card
access returned 401, so historical licensing-card details remain unverified.
([PixIT](https://github.com/joonaskalda/PixIT),
[Low-level checkpoint](https://huggingface.co/pyannote/separation-ami-1.0),
[Long-form pipeline](https://huggingface.co/pyannote/speech-separation-ami-1.0)).
The project's existing pinned models and default-negative/annotation-assisted
diagnostic distinction remain unchanged; no reinstallation or force-gating change
follows from the archive recommendation.
([Existing comparison](../../docs/stt_api_comparison.md)).

**MossFormer2** has WSJ0-2mix/3mix paper/sample evidence, but its documented ModelScope
`damo/speech_mossformer2_separation_temporal_8k` inference uses two voices/two outputs.
ClearerVoice distinguishes `MossFormer2_SS_16K` speech separation from `SE_48K`
enhancement/denoising. The linked primaries did not establish a ready blind-three-
speaker checkpoint; this is not a claim that the architecture universally supports
only two sources. Repository MIT and SS weights Apache-2.0 are distinct licenses.
SS weight revision `407cb030cd66340918ebb6c8cc63b18f8592cdbe` dates to **2024-11-25**.
([MossFormer2](https://github.com/alibabasglab/MossFormer2),
[ClearerVoice tasks](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md),
[Pinned SS weights](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/tree/407cb030cd66340918ebb6c8cc63b18f8592cdbe)).
Use it as an optional two-speaker control, not a proven replacement for the existing
three-source separator. The archive's MossFormer recommendation was in a denoise thread.

**SAM Audio** was discussed in archive post 94366 on **2025-12-17**. Text, visual-mask
or temporal-anchor guidance produces **target + residual**, not three blind human
streams. Three separate prompts would be an assisted experiment needing independent
leakage/phrase-coverage checks, not assumed three-way decomposition. Its gated **SAM
License** is not MIT/Apache or simply an NC-only license; current license text is dated
2025-11-19 but historical file pinning remains uncertain. Do not backdate newer `*-tv`
features or claim measured RTX VRAM fit.
([Repository](https://github.com/facebookresearch/sam-audio),
[Small checkpoint](https://huggingface.co/facebook/sam-audio-small),
[Current license](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE)).

## Older public table adds context, not a matched benchmark

The archive-linked **[public post 61346](https://t.me/speech_recognition_ru/61346?embed=1)**
was anonymously accessible in the supplied research. Its displayed **edited August 9,
2024 01:17** marker is not asserted as the original publication date. It is a Russian
telephony **text table**, not an imported photo: WeSpeaker `voxblink_100m` DER **20.10**,
CDER **0.115**, xRT(GPU) **0.025**; Pyannote 3.1 **24.44 / 0.587 / 0.052**.
Hardware, dataset files and collar/protocol are not specified. These values differ
from the 2025 screenshot and must not be merged into a single controlled benchmark.

**CDER** weights utterances equally, whereas DER is duration-based: compare models
within the same metric/protocol, not the numeric magnitude of DER against CDER.
Do not convert 0.115 to a percentage without establishing source units.
([CDER metric](https://github.com/SpeechClub/CDER_Metric)).

The actionable shortlist remains at most three **studies**: DiariZen→SepFormer
triggering subject to noncommercial terms, the existing PixIT-vs-SepFormer baseline
on identical true-three-speaker spans, and optional prompted SAM Audio extraction.
None is a newly measured production acceptance or a GPU speedup claim.
