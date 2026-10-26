"""Speaker refinement after diarization: re-assign turns that sound clearly like another speaker.

pyannote clusters sliding-window embeddings of a small model (WeSpeaker ResNet34). Here every
turn gets one embedding of a stronger model (WeSpeaker ResNet293), every speaker a centroid of
its longer turns, and a turn that is clearly (``margin``) closer to another speaker's centroid
moves there; speakers whose centroids are near-identical (``merge``) become one. Two passes.

Australian Survivor S10, all 24 episodes, 1300 lines with an SDH speaker label (mostly
off-screen or unclear voices), exclusive turns (what words are labelled by): lines on their
person's main speaker 63.6% -> 68.8% (margin 0.1; 68.9% at 0.05), ~17 s per episode.
VoxConverse: DER 11.87% -> 11.43%. Splitting speakers whose turns cluster into clearly different
voices (average linkage, similarity below 0.2-0.5) did not pay: fewer merged people (151 -> 126),
but fewer lines on their person's main speaker (65.3-67.5%), since one person's narration and
on-location voice split too.

    subs_env/bin/python -m stt.refine_speakers bench --data output/stt_bench/diar_voxconverse \\
        --hyp output/stt_bench/diar_voxconverse/hyp/pyannote-community-1 --models resnet293,pyannote
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
# WeSpeaker checkpoints loaded into pyannote's PyTorch port of the same architecture
WESPEAKER: dict[str, tuple[str, str]] = {
    "resnet293": ("Wespeaker/wespeaker-voxceleb-resnet293-LM", "WeSpeakerResNet293"),
}

Embed = Callable[[np.ndarray], np.ndarray]


class WespeakerTorch:
    """A WeSpeaker model in pyannote's PyTorch port: 16 kHz audio -> unit embedding (~35 ms per turn).

    The ONNX exports run 3-8x slower on turns of changing length (cuDNN falls back per shape)."""

    def __init__(self, name: str, device: str = "cuda") -> None:
        import pyannote.audio.models.embedding.wespeaker as wespeaker
        import torch
        from huggingface_hub import hf_hub_download

        repo, klass = WESPEAKER[name]
        state = torch.load(hf_hub_download(repo, "avg_model.pt"), map_location="cpu")
        model = getattr(wespeaker, klass)()
        model.resnet.load_state_dict({k: v for k, v in state.items() if not k.startswith("projection")})
        self.model = model.to(device).eval()
        self.device = device
        self._torch = torch

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        wav = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, None].to(self.device)
        with self._torch.inference_mode():
            vector = self.model(wav)[0].float().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-9)


def make_embedder(name: str, device: str = "cuda") -> Embed:
    if name == "pyannote":
        from stt.voice_bank import SegmentEmbedder

        embedder = SegmentEmbedder(device)
        return lambda audio: embedder(audio, RATE)
    return WespeakerTorch(name, device)


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


def refine_diarization(audio: np.ndarray, diarization: dict[str, Any], model: str, margin: float,
                       device: str = "cuda") -> dict[str, Any]:
    """The diarization with refined speakers in its turns and exclusive turns (same embeddings cache)."""
    embed_turn = turn_embedder(audio, make_embedder(model, device))
    turns, stats = refine_turns(diarization["turns"], embed_turn, margin=margin)
    exclusive, stats_x = refine_turns(diarization.get("exclusive") or [], embed_turn, margin=margin)
    return {**diarization, "turns": turns, "exclusive": exclusive,
            "refined": {"model": model, "margin": margin, "moved_turns": stats["moved"],
                        "moved_exclusive": stats_x["moved"]}}


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


def sdh_score(spans: list[tuple[str, float, float]], turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Labelled lines on their person's main speaker (line accuracy, time purity), people merged."""
    from collections import Counter, defaultdict

    from stt.sdh_eval import _GENERIC

    lines: list[tuple[str, str, float]] = []  # (label, predicted speaker, seconds)
    for label, start, end in spans:
        if label in _GENERIC:
            continue
        cover: Counter[str] = Counter()
        for t in turns:
            overlap = min(end, t["end"]) - max(start, t["start"])
            if overlap > 0:
                cover[t["speaker"]] += overlap
        if cover:
            speaker, _ = cover.most_common(1)[0]
            lines.append((label, speaker, end - start))
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for label, speaker, seconds in lines:
        by_label[label][speaker] += seconds
    main = {label: c.most_common(1)[0][0] for label, c in by_label.items()}
    owners = Counter(main.values())
    return {"lines": len(lines), "correct": sum(main[label] == speaker for label, speaker, _ in lines),
            "people": len(main), "merged": sum(1 for label in main if owners[main[label]] > 1)}


def sdh_bench(media: list[Path], cache: Path, models: list[str], margins: list[float], device: str,
              which: str = "turns") -> dict[str, dict[str, int]]:
    """Per variant, summed over episodes: labelled lines on their person's main speaker, people merged."""
    from stt.audio import load_audio, read_wav, save_wav
    from stt.diarize import create_diarizer
    from stt.sdh_eval import labelled_spans

    diarizer = None
    episodes = []
    for path in media:
        srt = path.with_name(path.stem + "_synced.srt")
        if not srt.exists():
            srt = path.with_suffix(".srt")
        folder = cache / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "audio.wav").exists():
            save_wav(folder / "audio.wav", load_audio(path, sample_rate=RATE, channels=1))
        audio = read_wav(folder / "audio.wav")
        if not (folder / "diarization.json").exists():
            if diarizer is None:
                diarizer = create_diarizer("pyannote-community-1", device=device)
                diarizer.load()
            (folder / "diarization.json").write_text(json.dumps(diarizer.diarize(audio).to_dict()), encoding="utf-8")
        turns = json.loads((folder / "diarization.json").read_text(encoding="utf-8"))[which]
        episodes.append((path.stem, audio, turns, labelled_spans(srt)))
        log.info("%s: %d turns, %d labelled lines", path.stem, len(turns), len(episodes[-1][3]))
    totals: dict[str, dict[str, int]] = {}

    def add(variant: str, score: dict[str, Any]) -> None:
        total = totals.setdefault(variant, {"lines": 0, "correct": 0, "people": 0, "merged": 0, "moved": 0})
        for key in ("lines", "correct", "people", "merged"):
            total[key] += score[key]

    for name, _, turns, spans in episodes:
        add("pyannote (as is)", sdh_score(spans, turns))
    for model in models:
        embed = make_embedder(model, device)
        for name, audio, turns, spans in episodes:
            embed_turn = turn_embedder(audio, embed)
            for margin in margins:
                refined, stats = refine_turns(turns, embed_turn, margin=margin)
                variant = f"{model} margin {margin}"
                add(variant, sdh_score(spans, refined))
                totals[variant]["moved"] += stats["moved"]
    return totals


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
    b.add_argument("--models", default="resnet293,pyannote")
    b.add_argument("--margins", default="0.05,0.1,0.15")
    b.add_argument("--merges", default="none,0.8")
    b.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    d = sub.add_parser("sdh", help="Reality TV episodes: SDH speaker labels before/after refinement")
    d.add_argument("media", nargs="+", help="Episodes (glob patterns); <name>_synced.srt or <name>.srt next to them")
    d.add_argument("--cache", type=Path, default=Path("output/stt_bench/sdh"))
    d.add_argument("--models", default="pyannote,resnet293")
    d.add_argument("--margins", default="0.05,0.1,0.2")
    d.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    d.add_argument("--turns", default="turns", choices=["turns", "exclusive"],
                   help="Diarization turns (may overlap) or exclusive turns (what words are labelled by)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.cmd == "sdh":
        import glob

        media = sorted(Path(m) for pattern in args.media for m in (glob.glob(pattern) or [pattern]))
        totals = sdh_bench(media, args.cache, args.models.split(","), [float(m) for m in args.margins.split(",")],
                           args.device, args.turns)
        print(f"{'variant':26} {'lines':>6} {'on own speaker':>15} {'people':>7} {'merged':>7} {'moved':>6}")
        for variant, t in totals.items():
            print(f"{variant:26} {t['lines']:6d} {t['correct'] / t['lines']:14.1%} {t['people']:7d} {t['merged']:7d} "
                  f"{t['moved']:6d}")
        return 0
    rows = bench(args.data, args.hyp, args.models.split(","), [float(m) for m in args.margins.split(",")],
                 [None if m == "none" else float(m) for m in args.merges.split(",")], args.device)
    print(f"{'model':18} {'margin':>6} {'merge':>6} {'DER':>7} {'confus':>7} {'moved':>6} {'merged':>6}")
    for r in rows:
        print(f"{r['model']:18} {str(r['margin']):>6} {str(r['merge']):>6} {r['der']:7.4f} {r['confusion']:7.4f} "
              f"{r.get('moved', ''):>6} {r.get('merged', ''):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
