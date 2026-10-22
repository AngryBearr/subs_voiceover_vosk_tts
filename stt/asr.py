"""ASR backends behind one interface, so models are compared on identical chunks.

Every backend receives the same 16 kHz mono audio and the same VAD chunks
(`stt.vad.asr_chunks`) and returns `Segment`s with times on the audio timeline.

Kept after the Survivor benchmarks (docs/stt_local.md): Whisper large-v3 (best model with its
own word times, default), Whisper large-v3-turbo (half the VRAM), and Cohere Transcribe +
Qwen3-ASR 1.7B, which are fused with Whisper by an LLM in the accurate mode (`--fuse`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stt.audio import SAMPLE_RATE
from stt.types import Segment, Word
from stt.vad import Chunk

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsrSpec:
    """Registry entry: backend kind, model id and the venv it must run in."""

    kind: str
    model_id: str
    options: dict[str, Any] = field(default_factory=dict)
    env: str = "subs_env"
    batch_size: int = 8
    note: str = ""


ASR_MODELS: dict[str, AsrSpec] = {
    "whisper-large-v3": AsrSpec("faster-whisper", "large-v3", batch_size=16, note="OpenAI, MIT"),
    "whisper-large-v3-turbo": AsrSpec("faster-whisper", "large-v3-turbo", batch_size=16, note="OpenAI, MIT"),
    "cohere-transcribe": AsrSpec(
        "cohere",
        "CohereLabs/cohere-transcribe-03-2026",
        note="Cohere, 14 languages, Apache-2.0, gated, no word timestamps",
    ),
    "qwen3-asr-1.7b": AsrSpec(
        "qwen3-asr",
        "Qwen/Qwen3-ASR-1.7B",
        options={"aligner": "Qwen/Qwen3-ForcedAligner-0.6B"},
        env="stt_qwen_env",
        note="Alibaba, Apache-2.0, timestamps via forced aligner",
    ),
}


class AsrBackend(ABC):
    """Load once, then transcribe any number of chunks."""

    def __init__(
        self,
        spec: AsrSpec,
        *,
        device: str = "cuda",
        batch_size: int | None = None,
        hotwords: str | None = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.batch_size = batch_size or spec.batch_size
        self.hotwords = hotwords  # e.g. cast names; used by backends that support biasing

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, chunks: list[Chunk]) -> list[Segment]: ...

    def _batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]


def _slice(audio: np.ndarray, chunk: Chunk) -> np.ndarray:
    start, end = chunk
    return audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]


class FasterWhisperBackend(AsrBackend):
    """Whisper via CTranslate2 (`faster-whisper`), batched over the given chunks."""

    def load(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        from stt.gpu import preload_cuda_libs

        if self.device == "cuda":
            preload_cuda_libs()
        compute_type = "float16" if self.device == "cuda" else "int8"
        model = WhisperModel(self.spec.model_id, device=self.device, compute_type=compute_type)
        self._pipeline = BatchedInferencePipeline(model)

    def transcribe(self, audio: np.ndarray, chunks: list[Chunk]) -> list[Segment]:
        if not chunks:
            return []
        clips = [{"start": s, "end": e} for s, e in chunks]
        raw_segments, _info = self._pipeline.transcribe(
            audio,
            language=self.spec.options.get("language", "en"),
            clip_timestamps=clips,
            batch_size=self.batch_size,
            word_timestamps=True,
            hotwords=self.hotwords,
        )
        segments: list[Segment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            words = [
                Word(start=round(w.start, 3), end=round(w.end, 3), text=w.word.strip(), score=round(w.probability, 4))
                for w in (seg.words or [])
                if w.word.strip()
            ]
            segments.append(Segment(start=round(seg.start, 3), end=round(seg.end, 3), text=text, words=words))
        return segments


class Qwen3AsrBackend(AsrBackend):
    """Qwen3-ASR via the `qwen-asr` package (needs its own venv: transformers 4.x)."""

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        aligner = self.spec.options.get("aligner")
        kwargs: dict[str, Any] = {}
        if aligner:
            kwargs = {
                "forced_aligner": aligner,
                "forced_aligner_kwargs": {"dtype": torch.bfloat16, "device_map": self.device},
            }
        self._model = Qwen3ASRModel.from_pretrained(
            self.spec.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            max_inference_batch_size=self.batch_size,
            max_new_tokens=512,
            **kwargs,
        )

    def transcribe(self, audio: np.ndarray, chunks: list[Chunk]) -> list[Segment]:
        with_stamps = bool(self.spec.options.get("aligner"))
        segments: list[Segment] = []
        for batch in self._batches(chunks):
            inputs = [(_slice(audio, c), SAMPLE_RATE) for c in batch]
            results = self._model.transcribe(
                audio=inputs, language=["English"] * len(inputs), return_time_stamps=with_stamps
            )
            for (start, end), result in zip(batch, results):
                text = (result.text or "").strip()
                if not text:
                    continue
                words = []
                for item in (result.time_stamps or []) if with_stamps else []:
                    words.append(
                        Word(
                            start=round(start + float(item.start_time), 3),
                            end=round(start + float(item.end_time), 3),
                            text=str(item.text).strip(),
                        )
                    )
                segments.append(Segment(start=start, end=end, text=text, words=words))
        return segments


class CohereBackend(AsrBackend):
    """Cohere Transcribe (native transformers >= 5.4); text only, chunk-level times."""

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.spec.model_id, device_map=self.device, dtype=torch.bfloat16
        ).eval()

    def transcribe(self, audio: np.ndarray, chunks: list[Chunk]) -> list[Segment]:
        torch = self._torch
        segments: list[Segment] = []
        for batch in self._batches(chunks):
            waves = [_slice(audio, c) for c in batch]
            inputs = self._processor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", language="en")
            chunk_index = inputs.pop("audio_chunk_index", None)
            inputs = inputs.to(self.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=320)
            texts = self._processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunk_index, language="en")
            if isinstance(texts, str):
                texts = [texts]
            for (start, end), text in zip(batch, texts):
                if text.strip():
                    segments.append(Segment(start=start, end=end, text=text.strip()))
        return segments


_BACKENDS: dict[str, type[AsrBackend]] = {
    "faster-whisper": FasterWhisperBackend,
    "qwen3-asr": Qwen3AsrBackend,
    "cohere": CohereBackend,
}


def create_backend(
    name: str, *, device: str = "cuda", batch_size: int | None = None, hotwords: str | None = None
) -> AsrBackend:
    """Instantiate (not load) the backend registered under `name`."""
    if name not in ASR_MODELS:
        raise KeyError(f"Unknown ASR model {name!r}; known: {', '.join(ASR_MODELS)}")
    spec = ASR_MODELS[name]
    return _BACKENDS[spec.kind](spec, device=device, batch_size=batch_size, hotwords=hotwords)
