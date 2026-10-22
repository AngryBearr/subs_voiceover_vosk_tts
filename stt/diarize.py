"""Speaker diarization (who speaks when): pyannote community-1 behind a small interface.

DiariZen was also benchmarked (better DER on VoxConverse, but on real episodes not better,
2.5x slower, non-commercial weights and its own torch 2.1 venv), see docs/stt_local.md.
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
    env: str = "stt_env"
    note: str = ""


DIARIZERS: dict[str, DiarSpec] = {
    "pyannote-community-1": DiarSpec(
        "pyannote",
        "pyannote/speaker-diarization-community-1",
        note="pyannote.audio 4, CC-BY-4.0, gated: accept terms + HF token",
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


_BACKENDS: dict[str, type[Diarizer]] = {
    "pyannote": PyannoteDiarizer,
}


def create_diarizer(name: str, *, device: str = "cuda") -> Diarizer:
    """Instantiate (not load) the diarizer registered under `name`."""
    if name not in DIARIZERS:
        raise KeyError(f"Unknown diarizer {name!r}; known: {', '.join(DIARIZERS)}")
    spec = DIARIZERS[name]
    return _BACKENDS[spec.kind](spec, device=device)
