"""Fine-tune the ResNet293 speaker embedder on Survivor speech: a 1 s window of a labelled line must land near its person
and away from the people of the same season. Queries are always gold (subtitle-named) windows, cut as the pipeline cuts
them: 1 s around one word, neighbour's speech included. Supports may also be pseudo (see build_dataset.py).

    subs_env/bin/python scripts/speaker_training/train.py --run r1 [--epochs 12 --persons 32 --per-person 8 ...]
Output: output/exp_spk/<run>/{train.log, best/avg_model.pt, last/avg_model.pt, args.json}.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd()))  # the repo root whose code runs (output/ is a symlink)
from stt.refine_speakers import WespeakerTorch  # noqa: E402

RATE = 16000
WIN = RATE  # 1 s windows, like stt.boundaries
log = logging.getLogger("train")


class Episodes(Dataset):
    """Each item is a whole batch of windows: persons x (queries + supports); workers build them on the CPU."""

    def __init__(self, data: Path, persons: int, per_person: int, steps: int, overlap_p: float, noise_p: float,
                 pseudo_support: bool, cue_queries: bool) -> None:
        index = json.loads((data / "index.json").read_text(encoding="utf-8"))
        self.audio = np.memmap(data / "audio.i16", dtype=np.int16, mode="r")
        self.noise = index["noise"]
        self.persons, self.per_person, self.steps = persons, per_person, steps
        self.overlap_p, self.noise_p = overlap_p, noise_p
        self.gold: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.support: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seasons: dict[str, set[int]] = defaultdict(set)
        for seg in index["segments"]:
            if seg["kind"] == "gold" or (seg["kind"] == "cue" and cue_queries):
                self.gold[seg["cls"]].append(seg)
                seasons[seg["season"]].add(seg["cls"])
            if seg["kind"] == "gold" or (seg["kind"] == "pseudo" and pseudo_support):
                self.support[seg["cls"]].append(seg)
        host = index["classes"].get("_HOST")
        self.host = host
        self.seasons = {s: sorted(c for c in cls if len(self.gold[c]) >= 2 and c != host) for s, cls in seasons.items()}
        self.season_names = sorted(self.seasons)
        weight = np.array([math.sqrt(sum(len(self.gold[c]) for c in self.seasons[s])) for s in self.season_names])
        self.season_p = weight / weight.sum()
        self.classes = len(index["classes"])
        # A returning player is one class per season, so the classifier only tells a query from its own season's people
        # (and the host): otherwise it would push the same voice away from itself.
        self.same_season = torch.zeros(self.classes, self.classes, dtype=torch.bool)
        for cls in seasons.values():
            ids = torch.tensor(sorted(cls))
            self.same_season[ids[:, None], ids] = True
        if host is not None:
            self.same_season[host, :] = True
            self.same_season[:, host] = True
        self.class_names = index["classes"]

    def __len__(self) -> int:
        return self.steps

    def _clip(self, seg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        """1 s around a random word of the line (gold) or a random point of the turn (pseudo), zero-padded at the edges."""
        if seg["words"]:
            a, b = seg["words"][rng.integers(len(seg["words"]))]
            centre = (a + b) / 2
        else:
            centre = rng.uniform(seg["start"], seg["end"])
        first = int((centre - seg["t0"]) * RATE) - WIN // 2
        out = np.zeros(WIN, dtype=np.float32)
        lo, hi = max(first, 0), min(first + WIN, seg["n"])
        if hi > lo:
            out[lo - first:hi - first] = np.asarray(self.audio[seg["off"] + lo:seg["off"] + hi], dtype=np.float32) / 32768.0
        return out

    def _mix(self, wav: np.ndarray, other: np.ndarray, snr_db: float) -> np.ndarray:
        power = float((wav ** 2).mean()) + 1e-9
        other_power = float((other ** 2).mean()) + 1e-9
        return wav + other * math.sqrt(power / other_power * 10 ** (-snr_db / 10))

    def _noise(self, rng: np.random.Generator) -> np.ndarray:
        item = self.noise[rng.integers(len(self.noise))]
        n = min(item["n"], WIN)
        first = item["off"] + (rng.integers(item["n"] - n + 1) if item["n"] > n else 0)
        out = np.zeros(WIN, dtype=np.float32)
        out[:n] = np.asarray(self.audio[first:first + n], dtype=np.float32) / 32768.0
        return np.resize(out[:n], WIN) if n else out

    def __getitem__(self, step: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(torch.initial_seed() % 2 ** 32 + step)
        chosen: list[int] = []
        while len(chosen) < self.persons:  # the same season fills first: the people a query must be told from
            season = self.season_names[rng.choice(len(self.season_names), p=self.season_p)]
            pool = [c for c in self.seasons[season] if c not in chosen]
            if not pool:
                continue
            weight = np.sqrt([len(self.gold[c]) for c in pool])
            take = min(self.persons - len(chosen) - (1 if self.host is not None and not chosen else 0), len(pool))
            chosen += [pool[i] for i in rng.choice(len(pool), size=take, replace=False, p=weight / weight.sum())]
            if self.host is not None and self.host not in chosen:
                chosen.append(self.host)
        queries, supports = [], []
        for cls in chosen:
            gold, support = self.gold[cls], self.support[cls]
            # Different seasonal IDs can be the same returning person; do not mix them as rival voices.
            rivals = [c for c in chosen if c != cls and self.same_season[cls, c]]
            used: set[int] = set()
            for k in range(self.per_person):
                seg = gold[rng.integers(len(gold))]
                used.add(seg["off"])
                wav = self._clip(seg, rng)
                if rng.random() < self.overlap_p and rivals:  # a distinct same-season person or the shared host
                    rival = self.gold[rivals[rng.integers(len(rivals))]]
                    other = rival[rng.integers(len(rival))]
                    if other["off"] not in used and other["cls"] != cls:
                        wav = self._mix(wav, self._clip(other, rng), rng.uniform(5.0, 15.0))
                if self.noise and rng.random() < self.noise_p:
                    wav = self._mix(wav, self._noise(rng), rng.uniform(5.0, 20.0))
                queries.append(wav)
            for k in range(self.per_person):
                seg = support[rng.integers(len(support))]
                for _ in range(5):  # a different line than the queries when there is one
                    if seg["off"] not in used:
                        break
                    seg = support[rng.integers(len(support))]
                supports.append(self._clip(seg, rng))
        return {"queries": torch.from_numpy(np.stack(queries)), "supports": torch.from_numpy(np.stack(supports)),
                "labels": torch.tensor(chosen, dtype=torch.long)}


def freeze(model: torch.nn.Module, prefixes: list[str], freeze_bn: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = not any(name.startswith(p) for p in prefixes)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and freeze_bn:
            module.eval()


def checkpoint_blocks(model: Any, layers: list[str]) -> None:
    """Run these ResNet stages block by block, recomputing on the way back: layer3 alone has 64 blocks and its saved
    activations are what fills the card. Keys of the state dict stay the same."""
    from torch.utils.checkpoint import checkpoint

    def wrap(block: torch.nn.Module) -> None:
        forward = block.forward
        block.forward = lambda x: checkpoint(forward, x, use_reentrant=False)  # type: ignore[method-assign]

    for name in layers:
        for block in getattr(model.resnet, name):
            wrap(block)


def embed(model: Any, wav: torch.Tensor) -> torch.Tensor:
    """Unit embeddings with gradients; the fbank front end is fixed."""
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        fbank = model.compute_fbank(wav[:, None].float())
    fbank = fbank.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vec = model.resnet(fbank)[1]
    return F.normalize(vec.float(), dim=-1)


def margin_logits(q: torch.Tensor, c: torch.Tensor, target: torch.Tensor,
                  allowed: torch.Tensor | None = None, *, margin: float = 0.2,
                  scale: float = 32.0) -> torch.Tensor:
    """Angular-margin logits; True in allowed means this rival may participate in the loss."""
    cos = q @ c.T
    sin = torch.sqrt((1.0 - cos ** 2).clamp(min=1e-6))
    phi = cos * math.cos(margin) - sin * math.sin(margin)
    hot = F.one_hot(target, c.size(0)).bool()
    logits = scale * torch.where(hot, phi, cos)
    return logits if allowed is None else logits.masked_fill(~allowed[target], -1e4)


class Scorer:
    """The quick check of a development episode: words of the labelled lines against the same episode's voices."""

    def __init__(self, folder: Path, srt_person_lines: list[dict[str, Any]]) -> None:
        from stt.assemble import CueOptions, assign_speakers
        from stt.audio import read_wav
        from stt.boundaries import RelabelParams, word_windows
        from stt.diarize import Diarization
        from stt.types import Word

        self.name = folder.name
        asr = json.loads((folder / "asr.whisper-large-v3.json").read_text(encoding="utf-8"))
        words = [Word(w["start"], w["end"], w["text"]) for s in asr["segments"] for w in s.get("words") or []
                 if w["end"] > w["start"]]
        dia = Diarization.from_dict(json.loads((folder / "diarization.diarizen-v2.refined.resnet293.json").read_text("utf-8")))
        self.words = assign_speakers(words, dia, CueOptions())
        self.own = [w.speaker for w in self.words]
        self.params = RelabelParams()
        self.windows = word_windows(self.words, read_wav(folder / "audio.wav"), 1.0).astype(np.float32)
        mid = np.array([(w.start + w.end) / 2 for w in self.words])
        self.person = np.full(len(self.words), "", dtype=object)
        self.line_s = np.zeros(len(self.words))
        for x in srt_person_lines:
            m = (mid >= x["start"]) & (mid <= x["end"])
            self.person[m] = x["person"]
            self.line_s[m] = x["end"] - x["start"]
        self.idx = np.flatnonzero(self.person != "")
        by: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for i in self.idx:
            if self.own[i] is not None:
                by[self.person[i]][self.own[i]] += self.words[i].end - self.words[i].start
        self.main = {p: max(c, key=c.get) for p, c in by.items() if c}  # type: ignore[arg-type]

    @torch.no_grad()
    def embed_all(self, model: Any, device: str, size: int) -> np.ndarray:
        out = []
        for i in range(0, len(self.windows), size):
            x = torch.from_numpy(self.windows[i:i + size]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(x[:, None]).float().cpu().numpy())
        emb = np.concatenate(out)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def eer(self, emb: np.ndarray) -> float:
        """Equal error rate of same/other person over the labelled words of the episode (more sensitive than the rest)."""
        rng = np.random.default_rng(3)
        idx = np.array([i for i in self.idx if self.person[i] in self.main], dtype=np.int64)
        a, b = rng.integers(len(idx), size=(2, 40000))
        cos = (emb[idx[a]] * emb[idx[b]]).sum(axis=1)
        same = self.person[idx[a]] == self.person[idx[b]]
        keep = a != b
        s, o = cos[same & keep], cos[~same & keep]
        grid = np.linspace(-0.2, 1.0, 241)
        far = np.array([(o >= t).mean() for t in grid])
        frr = np.array([(s < t).mean() for t in grid])
        k = int(np.argmin(np.abs(far - frr)))
        return float((far[k] + frr[k]) / 2)

    def metrics(self, emb: np.ndarray) -> dict[str, tuple[int, int]]:
        from stt.boundaries import _centroids, _viterbi

        sel = [i for i in self.idx if self.person[i] in self.main]
        persons = [p for p in self.main if sum(self.person[i] == p for i in sel) >= 10]
        sel = np.array([i for i in sel if self.person[i] in persons], dtype=np.int64)
        res: dict[str, tuple[int, int]] = {}
        if not len(sel):
            return res
        sums = {p: emb[self.idx[self.person[self.idx] == p]].sum(axis=0) for p in persons}
        ok = 0
        short_n = short_ok = 0
        for i in sel:
            scores = []
            for p in persons:
                c = sums[p] - emb[i] if p == self.person[i] else sums[p]
                scores.append(float(c @ emb[i]) / (float(np.linalg.norm(c)) + 1e-9))
            hit = persons[int(np.argmax(scores))] == self.person[i]
            ok += hit
            if self.line_s[i] < 1.5:
                short_n += 1
                short_ok += hit
        res["oracle"] = (ok, len(sel))
        res["oracle_short"] = (short_ok, short_n)
        cents = _centroids(self.words, self.own, emb, self.params)
        labels = sorted(cents)
        matrix = emb[sel] @ np.stack([cents[x] for x in labels]).T
        pred = [labels[j] for j in matrix.argmax(axis=1)]
        res["cluster"] = (sum(q == self.main[self.person[i]] for q, i in zip(pred, sel)), len(sel))
        path = _viterbi(self.words, self.own, emb, cents, self.params)
        res["viterbi"] = (sum(path[i] == self.main[self.person[i]] for i in sel), len(sel))
        return res


def evaluate(model: Any, scorers: list[Scorer], device: str, size: int) -> dict[str, float]:
    model.eval()
    total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    eers: list[float] = []
    for scorer in scorers:
        emb = scorer.embed_all(model, device, size)
        for key, (ok, n) in scorer.metrics(emb).items():
            total[key][0] += ok
            total[key][1] += n
        eers.append(scorer.eer(emb))
    scores = {k: round(100 * v[0] / max(v[1], 1), 2) for k, v in total.items()}
    scores["eer"] = round(100 * float(np.mean(eers)), 2) if eers else 100.0
    return scores


def head_init(model: Any, dataset: Episodes, device: str) -> torch.Tensor:
    """One unit vector per class from the current model: the loss starts near its real value."""
    rng = np.random.default_rng(1)
    head = np.zeros((dataset.classes, 256), dtype=np.float32)
    model.eval()
    for cls in range(dataset.classes):
        segs = dataset.support.get(cls) or dataset.gold.get(cls)
        if not segs:
            head[cls] = rng.normal(size=256)
            continue
        wav = np.stack([dataset._clip(segs[rng.integers(len(segs))], rng) for _ in range(32)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vec = model(torch.from_numpy(wav[:, None]).to(device)).float().mean(0).cpu().numpy()
        head[cls] = vec
    return F.normalize(torch.from_numpy(head), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="output/exp_spk/data")
    parser.add_argument("--dev", default="S46E10,S46E12,S46E13", help="development episodes (never trained on)")
    parser.add_argument("--init", default=None, help="start from this avg_model.pt instead of the VoxCeleb one")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=300, help="steps per epoch")
    parser.add_argument("--persons", type=int, default=32)
    parser.add_argument("--per-person", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--weight-sp", type=float, default=0.01, help="pull to the starting weights (L2-SP)")
    parser.add_argument("--freeze", default="layer1,layer2,conv1,bn1", help="backbone parts kept fixed")
    parser.add_argument("--checkpoint", default="layer3", help="stages run with gradient checkpointing (saves memory)")
    parser.add_argument("--bn", choices=["frozen", "adapt"], default="frozen")
    parser.add_argument("--aam-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--overlap-p", type=float, default=0.2)
    parser.add_argument("--noise-p", type=float, default=0.3)
    parser.add_argument("--no-pseudo", action="store_true", help="supports from gold lines only")
    parser.add_argument("--no-cue-lines", action="store_true", help="queries only from lines Whisper heard")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="three steps, then exit: memory and speed check")
    args = parser.parse_args()

    out = Path("output/exp_spk") / args.run
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wrapper = WespeakerTorch("resnet293", args.device)
    model = wrapper.model
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        model.resnet.load_state_dict({k: v for k, v in state.items() if not k.startswith("projection")})
    dataset = Episodes(Path(args.data), args.persons, args.per_person, args.steps, args.overlap_p, args.noise_p, not args.no_pseudo, not args.no_cue_lines)
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=True, prefetch_factor=4 if args.workers else None)
    log.info("classes %d, seasons %s, gold lines %d", dataset.classes, {s: len(c) for s, c in dataset.seasons.items()},
             sum(len(v) for v in dataset.gold.values()))

    scorers: list[Scorer] = []
    if not args.smoke and args.dev:
        train_root = Path("output/subs_train")
        for code in [c for c in args.dev.split(",") if c]:
            folder = next(train_root.glob(f"S*/{code}"))
            lines = json.loads((folder / "lines.json").read_text(encoding="utf-8"))["lines"]
            scorers.append(Scorer(folder, lines))
            log.info("dev %s: %d words, %d labelled", code, len(scorers[-1].words), len(scorers[-1].idx))
        log.info("val epoch 0 %s", evaluate(model, scorers, args.device, args.eval_batch))

    head = torch.nn.Parameter(head_init(model, dataset, args.device).to(args.device))
    prefixes = [f"resnet.{p}" for p in args.freeze.split(",") if p]
    freeze(model, prefixes, args.bn == "frozen")
    checkpoint_blocks(model, [x for x in args.checkpoint.split(",") if x])
    start_weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": args.lr, "weight_decay": 0.0},
                             {"params": [head], "lr": args.head_lr, "weight_decay": 0.0}])
    total_steps = args.epochs * args.steps
    warm = args.steps // 2

    def lr_factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    same_season = dataset.same_season.to(args.device)
    logit_options = {"margin": args.margin, "scale": args.scale}

    best = -math.inf
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze(model, prefixes, args.bn == "frozen")
        # Evaluation uses larger batches; release its unused allocator blocks before backpropagation.
        torch.cuda.empty_cache()
        for step, batch in enumerate(loader, 1):
            queries = batch["queries"].to(args.device, non_blocking=True)
            supports = batch["supports"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            persons, per = labels.size(0), args.per_person
            emb = embed(model, torch.cat([queries, supports]))
            q, s = emb[:queries.size(0)], emb[queries.size(0):]
            proto = F.normalize(s.view(persons, per, -1).mean(1), dim=-1)
            target = torch.arange(persons, device=labels.device).repeat_interleave(per)
            proto_allowed = same_season[labels[:, None], labels[None, :]]
            loss_proto = F.cross_entropy(margin_logits(q, proto, target, proto_allowed, **logit_options), target)
            class_target = labels.repeat_interleave(per)
            weights = F.normalize(head, dim=-1)
            loss_aam = F.cross_entropy(margin_logits(q, weights, class_target, same_season, **logit_options), class_target)
            loss_aam = 0.5 * (loss_aam + F.cross_entropy(margin_logits(s, weights, class_target, same_season, **logit_options), class_target))
            sp = sum(((p - start_weights[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad)
            loss = loss_proto + args.aam_weight * loss_aam + args.weight_sp * sp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
            if head.grad is not None and not torch.isfinite(head.grad).all():
                raise FloatingPointError(f"Non-finite head gradient at epoch {epoch}, step {step}")
            opt.step()
            sched.step()
            global_step += 1
            if step % 20 == 0 or args.smoke:
                top1 = (q @ proto.T).masked_fill(~proto_allowed[target], -1e4).argmax(-1).eq(target).float().mean().item()
                speed = global_step / (time.perf_counter() - t0)
                log.info("epoch %d/%d step %d/%d loss %.3f proto %.3f aam %.3f sp %.4f top1 %.3f %.2f it/s mem %.1f GB",
                         epoch, args.epochs, step, args.steps, loss.item(), loss_proto.item(), loss_aam.item(), float(sp),
                         top1, speed, torch.cuda.max_memory_allocated() / 2 ** 30)
            if args.smoke and step >= 3:
                print("SMOKE OK")
                return
        scores = evaluate(model, scorers, args.device, args.eval_batch)
        log.info("val epoch %d %s", epoch, scores)
        state = {k: v.cpu() for k, v in model.resnet.state_dict().items()}
        (out / "last").mkdir(exist_ok=True)
        torch.save(state, out / "last/avg_model.pt")
        key = -scores.get("eer", 100.0)
        if key > best:
            best = key
            (out / "best").mkdir(exist_ok=True)
            torch.save(state, out / "best/avg_model.pt")
    log.info("TRAIN DONE best %.2f", best)


if __name__ == "__main__":
    main()
