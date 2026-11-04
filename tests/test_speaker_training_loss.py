"""CPU regressions for within-season speaker discrimination and synthetic overlap."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.speaker_training.train import Episodes, margin_logits


def small_dataset(path: Path) -> Episodes:
    classes = {"_HOST": 0, "S02:COLBY": 1, "S08:COLBY": 2, "S02:AMBER": 3, "S08:AMBER": 4}
    segments = [{"cls": cls, "season": season, "kind": "gold", "off": cls * 32000 + k * 16000,
                 "n": 16000, "t0": 0.5, "start": 1., "end": 1.5, "words": [[1., 1.5]]}
                for cls, season in [(0, "S02"), (0, "S08"), (1, "S02"), (2, "S08"), (3, "S02"), (4, "S08")]
                for k in range(2)]
    np.zeros(160000, dtype=np.int16).tofile(path / "audio.i16")
    (path / "index.json").write_text(json.dumps({"classes": classes, "segments": segments, "noise": []}))
    return Episodes(path, 5, 2, 1, 1., 0., False, False)


def test_prototype_loss_excludes_cross_season_returnee_but_keeps_rival_and_host(tmp_path: Path) -> None:
    dataset = small_dataset(tmp_path)
    labels = torch.tensor([1, 2, 3, 0])  # Colby in two seasons, Amber in S02, shared host
    allowed = dataset.same_season[labels[:, None], labels[None, :]]
    assert not allowed[0, 1] and not allowed[1, 0]
    assert allowed[0, 2] and allowed[0, 3]
    assert allowed[3].all() and allowed[:, 3].all()
    q = torch.tensor([[.6, .8]])
    prototypes = torch.tensor([[.8, .6], [.6, .8], [.2, .98], [.9, .1]], requires_grad=True)
    target = torch.tensor([0])
    logits = margin_logits(q, prototypes, target, allowed, scale=2.)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert torch.equal(prototypes.grad[1], torch.zeros(2))
    assert prototypes.grad[2].abs().sum() > 0 and prototypes.grad[3].abs().sum() > 0
    changed = prototypes.detach().clone()
    changed[1] = torch.tensor([-1., 0.])
    assert torch.equal(loss.detach(), F.cross_entropy(margin_logits(q, changed, target, allowed, scale=2.), target))


def test_synthetic_overlap_never_uses_a_cross_season_returnee(tmp_path: Path) -> None:
    dataset = small_dataset(tmp_path)
    mixed: list[tuple[int, int]] = []

    def clip(seg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        return np.full(16000, seg["cls"], dtype=np.float32)

    def mix(wav: np.ndarray, other: np.ndarray, snr_db: float) -> np.ndarray:
        mixed.append((int(wav[0]), int(other[0])))
        return wav

    dataset._clip = clip  # type: ignore[method-assign]
    dataset._mix = mix  # type: ignore[method-assign]
    dataset[0]
    assert mixed
    assert all(a != b and dataset.same_season[a, b] for a, b in mixed)
