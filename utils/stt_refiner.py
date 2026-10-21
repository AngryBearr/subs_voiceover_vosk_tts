"""Bounded three-voice refinement; fixed PixIT identities, no diagnostic references."""
from __future__ import annotations

import hashlib
import itertools
import logging
import math
from pathlib import Path
from typing import Any

SR = 16000
LOG = logging.getLogger(__name__)
REPO = "speechbrain/sepformer-wsj03mix"
REVISION = "21e78b44a378092c34fff32ba8e788a73584f474"
CHECKSUMS = {
    "hyperparams.yaml": "1cef976e5e365e2c473b21a1feac33d0b027e940cee01c8b2978881c89c9fb50",
    "encoder.ckpt": "afdabc2f213c6391002088c0faade7253f62c1ad910c58b3c42a3147f3421451",
    "masknet.ckpt": "7c11786f74723c8e7021f6c9769057724c5e6833d1ed0f09fc935adfcb8a0078",
    "decoder.ckpt": "4744290ae39cee1c011bbb33e437ccf7b5b94c6c3b70fef595b6b7d53c32b0b7",
}


def match_sources(scores: list[list[float]], similarities: list[list[float]]) -> tuple[tuple[int, ...], float]:
    """Conservative one-to-one cosine assignment; thresholds are heuristics, not probabilities."""
    for matrix in (scores, similarities):
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or any(
            not math.isfinite(x) or not -1.000001 <= x <= 1.000001 for row in matrix for x in row
        ):
            raise ValueError("Triple speaker match needs review: invalid cosine matrix")
    if any(similarities[i][j] >= .90 for i in range(3) for j in range(i)):
        raise ValueError("Triple speaker match needs review: duplicate/indistinguishable source embeddings")
    ranked = sorted(((sum(scores[i][p[i]] for i in range(3)) / 3, p)
                     for p in itertools.permutations(range(3))), reverse=True)
    margin = ranked[0][0] - ranked[1][0]
    order = ranked[0][1]
    if min(scores[i][order[i]] for i in range(3)) < .45 or margin < .08:
        raise ValueError(f"Triple speaker match needs review: weak/ambiguous cosine assignment (margin={margin:.3f})")
    return order, margin


def triple_windows(turns: list[Any], duration: float) -> list[dict[str, Any]]:
    """Sweep actual activity; <=8s cores, <=1s context, short same-identity gaps merged."""
    events: dict[float, list[tuple[str, int]]] = {}
    for t in turns:
        for time, delta in ((max(0., t.start), 1), (min(duration, t.end), -1)):
            events.setdefault(time, []).append((t.speaker, delta))
    counts: dict[str, int] = {}
    spans = []
    times = sorted(events)
    for index, left in enumerate(times[:-1]):
        for speaker, delta in events[left]:
            counts[speaker] = counts.get(speaker, 0) + delta
        active = tuple(sorted(s for s, count in counts.items() if count > 0))
        right = times[index + 1]
        if len(active) > 3:
            raise ValueError("Triple refinement needs review: more than three predicted concurrent IDs")
        if len(active) == 3 and right > left:
            if (spans and spans[-1][2] == active and left - spans[-1][1] <= .32
                    and not any(t.speaker not in active and t.start < left and t.end > spans[-1][1]
                                for t in turns)):
                spans[-1] = (spans[-1][0], right, active)
            else:
                spans.append((left, right, active))
    windows = []
    for left, right, speakers in spans:
        begin, end = max(0., left - 1), min(duration, right + 1)
        # Context must not include an additional global voice.
        for t in turns:
            if t.speaker in speakers:
                continue
            if t.end <= left:
                begin = max(begin, t.end)
            elif t.start >= right:
                end = min(end, t.start)
        position = left
        while position < right:
            stop = min(position + 8, right)
            if 0 < right - stop < 2:
                stop = (position + right) / 2
            windows.append({"core_start": position, "core_end": stop,
                            "start": max(begin, position - 1), "end": min(end, stop + 1),
                            "write_start": begin if position == left else max(begin, position - .04),
                            "write_end": end if stop == right else stop,
                            "last": stop == right, "speakers": speakers})
            position = stop
    return windows


def load_sepformer(cache: Path) -> Any:
    """Only pinned local assets; trusted HyperPyYAML and legacy pickle loading, not weights-only."""
    from huggingface_hub import snapshot_download
    from speechbrain.inference.separation import SepformerSeparation

    snapshot = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=list(CHECKSUMS),
                                      cache_dir=str(cache / "hub"), token=False))
    if (snapshot / "custom.py").exists():
        raise ValueError("SepFormer snapshot unexpectedly contains remote Python")
    for name, expected in CHECKSUMS.items():
        digest = hashlib.sha256()
        with (snapshot / name).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SepFormer pinned asset checksum mismatch: {name}")
    runtime = cache / "sepformer-wsj03mix" / REVISION
    if (runtime / "custom.py").is_file():
        raise ValueError("SepFormer runtime unexpectedly contains Python; review cache")
    for name in CHECKSUMS:
        existing = runtime / name
        if existing.exists() and not existing.samefile(snapshot / name):
            raise ValueError(f"SepFormer runtime cache is not the verified snapshot: {existing}; review cache")
    model = SepformerSeparation.from_hparams(source=str(snapshot.resolve()), savedir=str(runtime.resolve()),
                                            run_opts={"device": "cpu"})
    if model.hparams.sample_rate != 8000 or model.hparams.num_spks != 3:
        raise ValueError("Pinned SepFormer must expose three sources at 8kHz")
    return model


def separated_window(model: Any, audio: Any) -> Any:
    """Explicit 16->8->16kHz resampling, exact original sample origin and length."""
    import numpy as np
    import torch
    import torchaudio

    mix = torchaudio.functional.resample(torch.from_numpy(audio)[None, :], SR, 8000)
    result = model.separate_batch(mix)
    if tuple(result.shape) != (1, mix.shape[1], 3):
        raise ValueError("SepFormer returned an invalid source shape")
    result = torchaudio.functional.resample(result[0].T, 8000, SR).T[:len(audio)].cpu().numpy()
    if result.shape != (len(audio), 3) or not np.isfinite(result).all():
        raise ValueError("SepFormer returned invalid/nonfinite resampled sources")
    # Joint least squares restores arbitrary source gain/polarity against the original mix.
    # No isolated reference or transcript is used; no hard clipping or peak guessing.
    gains, _, rank, _ = np.linalg.lstsq(result.astype(np.float64), audio.astype(np.float64), rcond=None)
    if rank != 3 or not np.isfinite(gains).all() or np.linalg.cond(result) > 100:
        raise ValueError("Triple refinement needs review: degenerate source waveforms")
    return (result * gains).astype(np.float32)


def embed_profile(embedding: Any, chunks: list[Any]) -> Any:
    import numpy as np
    import torch

    vectors = []
    for samples in chunks:
        if len(samples) < SR or float(np.sqrt(np.mean(samples ** 2))) <= 1e-5:
            raise ValueError("Triple speaker match needs review: short/silent speaker evidence")
        value = np.asarray(embedding(torch.from_numpy(samples.astype(np.float32))[None, None, :]))
        if value.ndim != 2 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError("Triple speaker match needs review: invalid ECAPA embedding")
        norm = np.linalg.norm(value[0])
        if norm <= 1e-8:
            raise ValueError("Triple speaker match needs review: zero ECAPA embedding")
        vectors.append(value[0] / norm)
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Triple speaker match needs review: invalid averaged profile")
    return mean / norm


def replace_window(data: Any, result: Any, columns: list[int], order: tuple[int, ...], *,
                   origin: int, begin: int, write_first: int, write_last: int, last: bool) -> None:
    """Replace (not sum) cores; 40ms edge crossfades and fixed sample coordinates."""
    import numpy as np

    first = max(write_first, origin, begin)
    end = min(write_last, origin + len(data), begin + len(result))
    if end <= first:
        raise ValueError("Triple replacement has no valid sample coverage")
    weight = np.ones(end - first, dtype=np.float32)
    fade = min(round(.04 * SR), len(weight) // 2)
    if fade:
        weight[:fade] = np.linspace(0, 1, fade)
        if last:
            weight[-fade:] = np.linspace(1, 0, fade)
    for i, column in enumerate(columns):
        old = data[first - origin:end - origin, column]
        new = result[first - begin:end - begin, order[i]]
        old[:] = old * (1 - weight) + new * weight


def refine_triples(audio: Any, turns: list[Any], sources: Any, mapping: dict[str, dict[str, Any]],
                   embedding: Any, cache: Path) -> None:
    """Use only PixIT clean original speech profiles; fail rather than invent identities."""
    windows = triple_windows(turns, len(audio) / SR)
    if not windows:
        return
    import numpy as np
    import torch
    from utils.stt_pipeline import clean_samples

    if embedding.sample_rate != SR:
        raise ValueError("Triple refinement requires the existing 16kHz PixIT ECAPA embedding")
    origin = round(sources.sliding_window.start * SR)
    profiles = {}
    required = sorted({s for w in windows for s in w["speakers"]})
    LOG.info("Triple refinement: %d bounded windows, %d identities; reuse PixIT ECAPA", len(windows), len(required))
    with torch.inference_mode():
        for speaker in required:
            intervals = clean_samples(turns, speaker)
            if sum(b - a for a, b in intervals) < 3:
                raise ValueError(f"Triple speaker match needs review: {speaker} lacks 3s clean original speech; "
                                 "include more clean context or review this recording")
            chunks = [audio[round(a * SR):round(b * SR)] for a, b in intervals]
            profiles[speaker] = embed_profile(embedding, chunks)
            # Calibrate the old track by a single LS gain using only this voice's clean speech.
            column = mapping[speaker]["source_channel"]
            xy, xx = 0., 0.
            for a, b in intervals:
                first, last = max(round(a * SR), origin), min(round(b * SR), origin + len(sources.data))
                x = sources.data[first - origin:last - origin, column].astype(np.float64)
                y = audio[first:last].astype(np.float64)
                xy += float(x @ y)
                xx += float(x @ x)
            gain = xy / max(xx, 1e-20)
            if not math.isfinite(gain) or abs(gain) <= 1e-8:
                raise ValueError(f"Triple refinement needs review: {speaker} has no calibratable PixIT clean source")
            sources.data[:, column] *= gain
            mapping[speaker]["pixit_clean_ls_gain"] = gain
            LOG.info("%s: clean-source LS gain %.6g", speaker, gain)
        model = load_sepformer(cache)
        for window in windows:
            first, last = round(window["start"] * SR), round(window["end"] * SR)
            result = separated_window(model, audio[first:last])
            core_first = round(window["core_start"] * SR) - first
            core_last = round(window["core_end"] * SR) - first
            if core_last - core_first < 2 * SR:
                raise ValueError("Triple speaker match needs review: overlap window has less than 2s source evidence")
            values = np.stack([embed_profile(embedding, [result[core_first:core_last, c]]) for c in range(3)])
            refs = np.stack([profiles[s] for s in window["speakers"]])
            order, margin = match_sources((refs @ values.T).tolist(), (values @ values.T).tolist())
            LOG.info("SepFormer3 %.3f–%.3fs: source order=%s, mean cosine margin=%.3f",
                     window["start"], window["end"], order, margin)
            columns = [mapping[s]["source_channel"] for s in window["speakers"]]
            replace_window(sources.data, result, columns, order, origin=origin, begin=first,
                           write_first=round(window["write_start"] * SR), write_last=round(window["write_end"] * SR),
                           last=window["last"])
            for i, speaker in enumerate(window["speakers"]):
                info = mapping[speaker]
                info.setdefault("refinement_windows", []).append({
                    "start": window["write_start"], "end": window["write_end"],
                    "source_channel": order[i], "cosine": float(refs[i] @ values[order[i]]), "margin": margin,
                })
                info["refinement_model"] = f"{REPO}@{REVISION}"
                info["refinement_status"] = "speaker_matched"
