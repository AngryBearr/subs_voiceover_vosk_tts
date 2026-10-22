"""Offline stand-ins for the model transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from voiceover.engine import PreparedText, SynthResult
from voiceover.llm import LLMClient, LLMConfig
from voiceover.text import syllables

Reply = str | dict[str, Any]


def chat_response(content: str, model: str = "fake-model-1") -> dict[str, Any]:
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class FakeTransport:
    """Callable transport that answers from a function (or a fixed list) and records request bodies."""

    def __init__(self, replies: Callable[[dict[str, Any]], Reply] | list[Reply]) -> None:
        self.replies = replies
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if callable(self.replies):
            reply = self.replies(body)
        else:
            reply = self.replies[len(self.bodies) - 1]
        return chat_response(reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False))

    @property
    def calls(self) -> int:
        return len(self.bodies)


def make_client(tmp_path: Path, transport: Callable[[dict[str, Any]], dict[str, Any]], **overrides: Any) -> LLMClient:
    config = LLMConfig(cache_dir=tmp_path / "cache", **overrides)
    return LLMClient(config, transport, sleep=lambda _: None)


class FakeEngine:
    """Engine stand-in: speech lasts ``0.17 * syllables + 0.1`` seconds and the clip is real silence."""

    SAMPLE_RATE = 22050

    def __init__(self, fail_texts: set[str] | None = None) -> None:
        self.fail_texts = fail_texts or set()
        self.prepared_items = 0
        self.synthesized: list[str] = []

    def identity(self) -> dict[str, Any]:
        return {"name": "fake"}

    def identity_sha(self) -> str:
        return "fake-engine"

    def synth_identity_sha(self) -> str:
        return "fake-synth"

    def prepare(self, items: list[dict[str, Any]]) -> list[PreparedText]:
        self.prepared_items += len(items)
        return [PreparedText(str(i["cue_id"]), str(i["text"]), str(i["text"]), "ok") for i in items]

    def synthesize_many(self, prepared: list[PreparedText], cache_dir: Path, workers: int) -> list[SynthResult]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        results: list[SynthResult] = []
        for item in prepared:
            self.synthesized.append(item.cue_id)
            if item.tts_text in self.fail_texts:
                results.append(SynthResult(item.cue_id, "", "", 0.0, 0.0, 0, 0, "fake failure"))
                continue
            seconds = 0.17 * syllables(item.tts_text) + 0.1
            path = cache_dir / f"{hashlib.sha256(item.tts_text.encode()).hexdigest()[:16]}.wav"
            sf.write(str(path), np.zeros(int(self.SAMPLE_RATE * seconds), dtype=np.float32), self.SAMPLE_RATE)
            results.append(SynthResult(item.cue_id, str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                                       seconds, seconds, 0, 0))
        return results


def fake_speech_sec(text: str) -> float:
    return 0.17 * syllables(text) + 0.1
