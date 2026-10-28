"""Speaker diarization (who speaks when): pyannote community-1 or DiariZen v2 behind a small interface.

DiariZen v2 (BUT, WavLM Large pruned to 20%, Conformer, VBx; CC BY-NC 4.0 weights) runs in its own
venv (`diarizen_env`, torch 2.1 and a pyannote 3.1 fork: `bash scripts/setup_envs.sh diarizen`).
It gives no exclusive turns and no cluster prints; `stt.transcribe` adds the prints (the voice
bank's embedding space) in subs_env. S29E05-E08 against the captions' turn marks (`stt.turn_eval`),
both with the ResNet293 refinement: speaker changes found 64% -> 80% at the same precision;
~4x slower than pyannote (~2.5 min per 43-min episode, ~4.5 GB VRAM). The first DiariZen
(2025) was not better than pyannote on Survivor, see docs/stt_local.md.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stt.audio import SAMPLE_RATE

log = logging.getLogger(__name__)


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
        )


class Diarizer(ABC):
    def __init__(self, spec: DiarSpec, *, device: str = "cuda") -> None:
        self.spec = spec
        self.device = device

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
        import io

        import soundfile as sf

        self._pipeline.min_speakers = num_speakers or min_speakers or 1
        self._pipeline.max_speakers = num_speakers or max_speakers or 20
        buffer = io.BytesIO()  # the pipeline reads a file (torchaudio): an in-memory WAV
        sf.write(buffer, np.ascontiguousarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="FLOAT")
        buffer.seek(0)
        return Diarization(turns=_annotation_turns(self._pipeline(buffer)))


_BACKENDS: dict[str, type[Diarizer]] = {
    "pyannote": PyannoteDiarizer,
    "diarizen": DiarizenDiarizer,
}


def create_diarizer(name: str, *, device: str = "cuda") -> Diarizer:
    """Instantiate (not load) the diarizer registered under `name`."""
    if name not in DIARIZERS:
        raise KeyError(f"Unknown diarizer {name!r}; known: {', '.join(DIARIZERS)}")
    spec = DIARIZERS[name]
    return _BACKENDS[spec.kind](spec, device=device)
