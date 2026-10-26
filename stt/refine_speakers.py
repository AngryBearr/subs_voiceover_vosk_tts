"""Speaker refinement after diarization: re-assign turns that sound clearly like another speaker.

pyannote clusters sliding-window embeddings of a small model (WeSpeaker ResNet34). Here every
turn gets one embedding of a stronger model, every speaker a centroid of its longer turns, and a
turn that is clearly (``margin``) closer to another speaker's centroid moves there; speakers whose
centroids are near-identical (``merge``) become one. Two passes.

    subs_env/bin/python -m stt.refine_speakers bench --data output/stt_bench/diar_voxconverse \\
        --hyp output/stt_bench/diar_voxconverse/hyp/pyannote-community-1 --models resnet293,campplus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("stt.refine_speakers")

RATE = 16_000
EMBEDDERS: dict[str, tuple[str, str]] = {
    "resnet293": ("Wespeaker/wespeaker-voxceleb-resnet293-LM", "voxceleb_resnet293_LM.onnx"),
    "campplus": ("Wespeaker/wespeaker-voxceleb-campplus-LM", "voxceleb_CAM++_LM.onnx"),
}

Embed = Callable[[np.ndarray], np.ndarray]


class WespeakerOnnx:
    """WeSpeaker ONNX speaker model: Kaldi fbank (80 bins, mean-normalised) -> unit embedding."""

    def __init__(self, name: str, device: str = "cuda") -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        repo, filename = EMBEDDERS[name]
        if device == "cuda":
            ort.preload_dlls()
        # every turn has another length: the exhaustive cuDNN search per new shape would dominate the run
        cuda = ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})
        providers = [cuda, "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(hf_hub_download(repo, filename), providers=providers)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None] * 32768
        feats = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                            sample_frequency=RATE, window_type="hamming", use_energy=False)
        feats = feats - feats.mean(0, keepdim=True)
        vector = self.session.run(None, {"feats": feats[None].numpy()})[0][0]
        return vector / (np.linalg.norm(vector) + 1e-9)


def make_embedder(name: str, device: str = "cuda") -> Embed:
    if name == "pyannote":
        from stt.voice_bank import SegmentEmbedder

        embedder = SegmentEmbedder(device)
        return lambda audio: embedder(audio, RATE)
    return WespeakerOnnx(name, device)


def _overlapping(turns: list[dict[str, Any]]) -> list[bool]:
    """Whether each turn overlaps another speaker's turn (two voices: no clean embedding)."""
    order = sorted(range(len(turns)), key=lambda i: turns[i]["start"])
    flags = [False] * len(turns)
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if turns[j]["start"] >= turns[i]["end"]:
                break
            if turns[j]["speaker"] != turns[i]["speaker"]:
                flags[i] = flags[j] = True
    return flags


def turn_embedder(audio: np.ndarray, embed: Embed) -> Callable[[dict[str, Any]], np.ndarray]:
    """Embedding of a turn of ``audio`` (16 kHz), computed once per (start, end)."""
    cache: dict[tuple[float, float], np.ndarray] = {}

    def of(turn: dict[str, Any]) -> np.ndarray:
        key = (round(turn["start"], 3), round(turn["end"], 3))
        if key not in cache:
            cache[key] = embed(audio[int(turn["start"] * RATE):int(turn["end"] * RATE)])
        return cache[key]

    return of


def refine_turns(turns: list[dict[str, Any]], embed_turn: Callable[[dict[str, Any]], np.ndarray], *,
                 margin: float = 0.1, merge: float | None = None, min_turn: float = 0.5, min_anchor: float = 1.5,
                 passes: int = 2) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turns with refined speakers, and counts of moved turns / merged speakers."""
    turns = [dict(t) for t in turns]
    overlap = _overlapping(turns)
    vectors = {i: embed_turn(t) for i, t in enumerate(turns) if not overlap[i] and t["end"] - t["start"] >= min_turn}
    stats = {"moved": 0, "merged": 0, "embedded": len(vectors)}
    for _ in range(passes):
        centroids: dict[str, np.ndarray] = {}
        for speaker in {t["speaker"] for t in turns}:
            anchors = [(vectors[i], turns[i]["end"] - turns[i]["start"]) for i in vectors
                       if turns[i]["speaker"] == speaker and turns[i]["end"] - turns[i]["start"] >= min_anchor]
            if anchors:
                mean = sum(v * w for v, w in anchors)
                centroids[speaker] = mean / (np.linalg.norm(mean) + 1e-9)
        if merge is not None:  # near-identical speakers become one (the one with more speech keeps its name)
            speech = {s: sum(t["end"] - t["start"] for t in turns if t["speaker"] == s) for s in centroids}
            for a in sorted(centroids, key=lambda s: -speech[s]):
                for b in list(centroids):
                    if b != a and a in centroids and b in centroids and float(centroids[a] @ centroids[b]) >= merge:
                        for t in turns:
                            if t["speaker"] == b:
                                t["speaker"] = a
                        del centroids[b]
                        stats["merged"] += 1
        if len(centroids) < 2:
            break
        moved = 0
        for i, vector in vectors.items():
            current = turns[i]["speaker"]
            sims = {s: float(vector @ c) for s, c in centroids.items()}
            best = max(sims, key=lambda s: sims[s])
            if best != current and sims[best] - sims.get(current, -1.0) >= margin:
                turns[i]["speaker"] = best
                moved += 1
        stats["moved"] += moved
        if not moved:
            break
    return turns, stats


def _der(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], metric: Any) -> float:
    from pyannote.core import Annotation, Segment

    def annotation(turns: list[dict[str, Any]]) -> Any:
        out = Annotation()
        for i, t in enumerate(turns):
            if t["end"] > t["start"]:
                out[Segment(t["start"], t["end"]), i] = t["speaker"]
        return out

    return metric(annotation(ref), annotation(hyp))


def bench(data_dir: Path, hyp_dir: Path, models: list[str], margins: list[float], merges: list[float | None],
          device: str) -> list[dict[str, Any]]:
    import soundfile as sf
    from pyannote.metrics.diarization import DiarizationErrorRate

    files = json.loads((data_dir / "files.json").read_text(encoding="utf-8"))
    audio = {f["name"]: sf.read(str(data_dir / Path(f["wav"]).name), dtype="float32")[0] for f in files}
    hyps = {f["name"]: json.loads((hyp_dir / f"{f['name']}.json").read_text(encoding="utf-8"))["turns"] for f in files}
    rows = []
    base = DiarizationErrorRate(collar=0.25)
    for f in files:
        _der(f["turns"], hyps[f["name"]], base)
    rows.append({"model": "pyannote (as is)", "margin": None, "merge": None, **_components(base)})
    for model in models:
        embed = make_embedder(model, device)
        embedders = {f["name"]: turn_embedder(audio[f["name"]], embed) for f in files}
        for margin in margins:
            for merge in merges:
                metric = DiarizationErrorRate(collar=0.25)
                moved = merged = 0
                for f in files:
                    refined, stats = refine_turns(hyps[f["name"]], embedders[f["name"]], margin=margin, merge=merge)
                    moved += stats["moved"]
                    merged += stats["merged"]
                    _der(f["turns"], refined, metric)
                row = {"model": model, "margin": margin, "merge": merge, **_components(metric),
                       "moved": moved, "merged": merged}
                rows.append(row)
                log.info("%s", row)
    return rows


def _components(metric: Any) -> dict[str, float]:
    c = metric[:]
    return {"der": round(abs(metric), 4), "confusion": round(c["confusion"] / c["total"], 4),
            "missed": round(c["missed detection"] / c["total"], 4),
            "false_alarm": round(c["false alarm"] / c["total"], 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refine diarization speakers with a stronger speaker model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="DER on VoxConverse before/after refinement")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--hyp", type=Path, required=True)
    b.add_argument("--models", default="resnet293,campplus,pyannote")
    b.add_argument("--margins", default="0.05,0.1,0.15")
    b.add_argument("--merges", default="none,0.8")
    b.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    rows = bench(args.data, args.hyp, args.models.split(","), [float(m) for m in args.margins.split(",")],
                 [None if m == "none" else float(m) for m in args.merges.split(",")], args.device)
    print(f"{'model':18} {'margin':>6} {'merge':>6} {'DER':>7} {'confus':>7} {'moved':>6} {'merged':>6}")
    for r in rows:
        print(f"{r['model']:18} {str(r['margin']):>6} {str(r['merge']):>6} {r['der']:7.4f} {r['confusion']:7.4f} "
              f"{r.get('moved', ''):>6} {r.get('merged', ''):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
