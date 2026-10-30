"""Speaker diarization (who speaks when): pyannote community-1 or DiariZen v2 behind a small interface.

DiariZen v2 (BUT, WavLM Large pruned to 20%, Conformer, VBx; CC BY-NC 4.0 weights) runs in its own
venv (`diarizen_env`, torch 2.1 and a pyannote 3.1 fork: `bash scripts/setup_envs.sh diarizen`).
Exclusive turns only on request (options.exclusive); no cluster prints; `stt.transcribe` adds the prints (the voice
bank's embedding space) in subs_env. S29E05-E08 against the captions' turn marks (`stt.turn_eval`),
both with the ResNet293 refinement: speaker changes found 64% -> 80% at the same precision;
~4x slower than pyannote (~2.5 min per 43-min episode, ~4.5 GB VRAM). The first DiariZen
(2025) was not better than pyannote on Survivor, see docs/stt_local.md.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stt.audio import SAMPLE_RATE

log = logging.getLogger(__name__)

# DiariZen options and their defaults (today's behaviour); only non-default values are stored in
# `Diarization.options`, so cached files without the key mean the defaults.
DIARIZER_OPTION_DEFAULTS: dict[str, Any] = {"median_filter": True, "exclusive": False}


def non_default_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The entries of `options` that differ from DIARIZER_OPTION_DEFAULTS (unknown keys are kept)."""
    return {k: v for k, v in (options or {}).items() if k not in DIARIZER_OPTION_DEFAULTS
            or v != DIARIZER_OPTION_DEFAULTS[k]}


@dataclass(frozen=True)
class DiarSpec:
    """Registry entry: backend kind, model id and the venv it must run in."""

    kind: str
    model_id: str
    env: str = "subs_env"
    note: str = ""


DIARIZERS: dict[str, DiarSpec] = {
    "pyannote-community-1": DiarSpec(
        "pyannote",
        "pyannote/speaker-diarization-community-1",
        note="pyannote.audio 4, CC-BY-4.0, gated: accept terms + HF token",
    ),
    "diarizen-v2": DiarSpec(
        "diarizen",
        "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        env="diarizen_env",
        note="DiariZen v2, CC-BY-NC-4.0 weights, own venv: bash scripts/setup_envs.sh diarizen",
    ),
}


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Diarization:
    """Diarization output; times are seconds on the analyzed audio timeline."""

    turns: list[Turn]
    exclusive: list[Turn] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)  # non-default diarizer options only

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Diarization:
        return cls(
            turns=[Turn(**t) for t in data["turns"]],
            exclusive=[Turn(**t) for t in data.get("exclusive", [])],
            embeddings=data.get("embeddings", {}),
            options=data.get("options") or {},
        )


class Diarizer(ABC):
    def __init__(self, spec: DiarSpec, *, device: str = "cuda", options: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.device = device
        self.options = {**DIARIZER_OPTION_DEFAULTS, **(options or {})}

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Diarization: ...


def _annotation_turns(annotation: Any) -> list[Turn]:
    return [
        Turn(round(seg.start, 3), round(seg.end, 3), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]


def exclusive_turns(
    scores: np.ndarray, active: np.ndarray, start: float, step: float, duration: float
) -> list[Turn]:
    """One speaker per frame out of the (already) active ones: the one with the highest soft score.

    `scores` and `active` are (num_frames, num_clusters); `active` marks the speakers of the regular
    turns, so the result is a subset of them. Times follow pyannote's `Binarize`, which the regular
    turns come from: a turn runs from the middle of its first frame to the middle of the first frame
    after it (frame i spans [start + i*step, start + i*step + duration]), so the layers line up.
    """
    turns: list[Turn] = []
    current: int | None = None
    first = 0
    num_frames = scores.shape[0]
    masked = np.where(active, scores, -np.inf)
    for i in range(num_frames):
        speaker = int(np.argmax(masked[i])) if active[i].any() else None  # ties: first active cluster
        if speaker != current:
            if current is not None:
                turns.append(Turn(round(start + first * step + duration / 2, 3),
                                  round(start + i * step + duration / 2, 3), str(current)))
            current, first = speaker, i
    if current is not None:  # still speaking in the last frame: ends at its middle, like Binarize
        turns.append(Turn(round(start + first * step + duration / 2, 3),
                          round(start + (num_frames - 1) * step + duration / 2, 3), str(current)))
    return [t for t in turns if t.end > t.start]


class PyannoteDiarizer(Diarizer):
    """pyannote.audio 4 pipelines (community-1): segmentation + WeSpeaker + VBx clustering."""

    def load(self) -> None:
        import torch
        from huggingface_hub import get_token
        from pyannote.audio import Pipeline

        token = get_token()
        if token is None:
            raise RuntimeError(
                f"{self.spec.model_id} is gated: accept its terms on huggingface.co and run `hf auth login`"
            )
        self._torch = torch
        self._pipeline = Pipeline.from_pretrained(self.spec.model_id, token=token)
        if self._pipeline is None:
            raise RuntimeError(f"Could not load {self.spec.model_id}; were the model terms accepted?")
        self._pipeline.to(torch.device(self.device))

    def diarize(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Diarization:
        waveform = self._torch.from_numpy(np.ascontiguousarray(audio))[None]
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        annotation = getattr(output, "speaker_diarization", output)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        embeddings: dict[str, list[float]] = {}
        vectors = getattr(output, "speaker_embeddings", None)
        if vectors is not None:
            for label, vector in zip(annotation.labels(), vectors):
                if all(math.isfinite(x) for x in vector):
                    embeddings[str(label)] = [round(float(x), 6) for x in vector]
        return Diarization(
            turns=_annotation_turns(annotation),
            exclusive=_annotation_turns(exclusive) if exclusive is not None else [],
            embeddings=embeddings,
            options=non_default_options(self.options),  # ignored here, recorded so the cache check is stable
        )


class DiarizenDiarizer(Diarizer):
    """DiariZen (pyannote 3.1 fork): WavLM-Conformer segmentation + WeSpeaker ResNet34 + VBx clustering."""

    batch_size = 8  # segmentation and embedding batches: ~4.5 GB VRAM (32 by default: ~13 GB)
    # AHC threshold before VBx (0.6 in the model's config): 0.5 kept Wes and Alec apart in S29E05 and
    # changed the speaker-change F1 by at most 0.01 on 8 other episodes; a lower VBx Fb splits voices.
    ahc_threshold = 0.5

    def load(self) -> None:
        from diarizen.pipelines.inference import DiariZenPipeline

        self._pipeline = DiariZenPipeline.from_pretrained(self.spec.model_id)
        self._pipeline.segmentation_batch_size = self.batch_size
        self._pipeline.embedding_batch_size = self.batch_size
        clustering = {**self._pipeline.PIPELINE_PARAMS["clustering"], "ahc_threshold": self.ahc_threshold}
        self._pipeline.instantiate({"clustering": clustering})

    def diarize(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Diarization:
        self._pipeline.min_speakers = num_speakers or min_speakers or 1
        self._pipeline.max_speakers = num_speakers or max_speakers or 20
        turns, exclusive = self._run(np.ascontiguousarray(audio, dtype=np.float32))
        return Diarization(turns=turns, exclusive=exclusive, options=non_default_options(self.options))

    def _run(self, audio: np.ndarray) -> tuple[list[Turn], list[Turn]]:
        """DiariZenPipeline.__call__ re-implemented step by step, to switch the median filter off
        and to derive exclusive turns from the soft segmentation."""
        import torch
        from pyannote.audio.utils.signal import Binarize
        from scipy.ndimage import median_filter

        pipeline = self._pipeline
        file = {"waveform": torch.from_numpy(audio)[None], "sample_rate": SAMPLE_RATE}
        segmentations = pipeline.get_segmentations(file, soft=False)
        if self.options["median_filter"]:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode="reflect")
        count = pipeline.speaker_count(
            segmentations, pipeline._segmentation.model._receptive_field, warm_up=(0.0, 0.0)
        )
        embeddings = pipeline.get_embeddings(file, segmentations, exclude_overlap=pipeline.embedding_exclude_overlap)
        hard_clusters, _, _ = pipeline.clustering(
            embeddings=embeddings, segmentations=segmentations,
            min_clusters=pipeline.min_speakers, max_clusters=pipeline.max_speakers,
        )
        count.data = np.minimum(count.data, pipeline.max_speakers).astype(np.int8)
        hard_clusters[np.sum(segmentations.data, axis=1) == 0] = -2  # inactive local speakers
        discrete, _ = pipeline.reconstruct(segmentations, hard_clusters, count)
        annotation = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)(discrete)
        turns = _annotation_turns(annotation)
        if not self.options["exclusive"]:
            return turns, []

        t0 = time.perf_counter()
        soft = pipeline.get_segmentations(file, soft=True)  # probabilities, no median filter
        _, soft_act = pipeline.reconstruct(soft, hard_clusters, count)
        scores = np.asarray(soft_act.data, dtype=np.float64)
        active = discrete.data > 0
        if scores.shape != active.shape:
            log.warning("Soft activations %s differ from the hard ones %s; padding/cropping",
                        scores.shape, active.shape)
            fitted = np.zeros(active.shape, dtype=np.float64)
            rows, cols = min(scores.shape[0], active.shape[0]), min(scores.shape[1], active.shape[1])
            fitted[:rows, :cols] = scores[:rows, :cols]
            scores = fitted
        frames = discrete.sliding_window
        exclusive = exclusive_turns(scores, active, float(frames.start), float(frames.step),
                                    float(frames.duration))
        log.info("Exclusive turns: %d (soft pass %.1fs)", len(exclusive), time.perf_counter() - t0)
        return turns, exclusive


_BACKENDS: dict[str, type[Diarizer]] = {
    "pyannote": PyannoteDiarizer,
    "diarizen": DiarizenDiarizer,
}


def create_diarizer(name: str, *, device: str = "cuda", options: dict[str, Any] | None = None) -> Diarizer:
    """Instantiate (not load) the diarizer registered under `name`."""
    if name not in DIARIZERS:
        raise KeyError(f"Unknown diarizer {name!r}; known: {', '.join(DIARIZERS)}")
    spec = DIARIZERS[name]
    return _BACKENDS[spec.kind](spec, device=device, options=options)
