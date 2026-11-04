"""Train the line-gender classifier of ``stt.line_gender``: logistic regression on resnet293-ft voice prints of whole
lines (subtitle-named lines and shout cue lines of the speaker training set, built by build_dataset.py), each person
weighing the same; 10% of the people are held out to choose the regularisation.

The genders of the people come from a JSON {person: "male"|"female"|{"gender": ...}} made by ``gender_labels.py``
(the season cast files where the pipeline made one, else asked to a language model with the show, season and name).

    subs_env/bin/python scripts/speaker_training/train_gender.py [--data output/exp_spk/data_p2]
        [--genders output/exp_gender/people_gender.json] [--out models/gender/resnet293-ft-gender.npz]
Output: the .npz with ``w`` (256), ``b`` and the voice-print settings (``max_s``, ``bucket_s``, ``min_s``, ``model``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stt.line_gender import voice_prints  # noqa: E402  (the same crop / bucket rule as the pipeline)

log = logging.getLogger("train_gender")
RATE = 16000
MODEL, MAX_S, BUCKET_S, MIN_S = "resnet293-ft", 6.0, 0.25, 0.5


def load_lines(data: Path, genders_path: Path) -> tuple[list[np.ndarray], list[float], list[str]]:
    """Training clips, labels (1 = female) and the person of each clip."""
    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    genders = json.loads(genders_path.read_text(encoding="utf-8"))
    names = {v: k for k, v in index["classes"].items()}
    audio = np.memmap(data / "audio.i16", dtype=np.int16, mode="r")
    clips, ys, people, kinds = [], [], [], []
    for s in index["segments"]:
        if s["kind"] not in ("gold", "cue"):
            continue
        person = names[s["cls"]]
        g = genders.get(person)
        g = g if isinstance(g, str) else (g or {}).get("gender")
        if g not in ("male", "female"):
            continue
        a, b = int((s["start"] - s["t0"]) * RATE), int((s["end"] - s["t0"]) * RATE)
        seg = np.asarray(audio[s["off"] + max(a, 0): s["off"] + min(b, s["n"])], dtype=np.float32) / 32768.0
        if len(seg) < 0.3 * RATE:
            continue
        clips.append(seg)
        ys.append(1.0 if g == "female" else 0.0)
        people.append(person)
        kinds.append(s["kind"])
    log.info("training lines %d (%s), female share %.2f", len(clips), dict(Counter(kinds)), float(np.mean(ys)))
    return clips, ys, people


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    import torch

    X, Y, W = (torch.from_numpy(a).float().cuda() for a in (x, y, w))
    weight = torch.zeros(x.shape[1], device="cuda", requires_grad=True)
    bias = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=300, line_search_fn="strong_wolfe")

    def closure() -> Any:
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(X @ weight + bias, Y, reduction="none") * W).mean()
        loss = loss + l2 * weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return weight.detach().cpu().numpy(), float(bias.item())


def predict(model: tuple[np.ndarray, float], x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x @ model[0] + model[1])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the line-gender classifier on resnet293-ft voice prints")
    parser.add_argument("--data", type=Path, default=Path("output/exp_spk/data_p2"))
    parser.add_argument("--genders", type=Path, default=Path("output/exp_gender/people_gender.json"))
    parser.add_argument("--out", type=Path, default=Path("models/gender/resnet293-ft-gender.npz"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    from stt.refine_speakers import WespeakerTorch

    clips, ys, people = load_lines(args.data, args.genders)
    started = time.perf_counter()
    x = voice_prints(WespeakerTorch(MODEL, "cuda").model.eval(), clips, max_s=MAX_S, bucket_s=BUCKET_S, min_s=MIN_S)
    log.info("voice prints in %.0f s", time.perf_counter() - started)
    y = np.array(ys)
    count = Counter(people)
    w = np.array([1.0 / count[p] for p in people], dtype=np.float32)
    w *= len(w) / w.sum()
    rng = np.random.default_rng(1)
    held = set(rng.choice(sorted(count), size=max(1, len(count) // 10), replace=False))
    dev = np.array([p in held for p in people])
    best: tuple[float, float] | None = None
    for l2 in (1e-4, 1e-3, 1e-2):
        model = fit(x[~dev], y[~dev], w[~dev], l2)
        acc = float(((predict(model, x[dev]) > 0.5) == (y[dev] > 0.5)).mean())
        log.info("l2 %g: held-out people %.3f", l2, acc)
        if best is None or acc > best[0]:
            best = (acc, l2)
    assert best is not None
    weight, bias = fit(x, y, w, best[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, w=weight.astype(np.float32), b=np.float32(bias), max_s=np.float32(MAX_S),
             bucket_s=np.float32(BUCKET_S), min_s=np.float32(MIN_S), model=np.array(MODEL))
    log.info("classifier saved to %s (l2 %g)", args.out, best[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
