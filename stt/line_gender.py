"""Gender of the lines of speakers whose gender stayed unknown, from the lines' own voices.

The voice estimate abstains on little clean speech and the naming LLM may find no evidence, so some
speakers stay "unknown" (voiced as male, translated in the masculine). Each of their lines of 1.2+ s
is classified by the same age/gender model (audeering) and keeps its own confident gender (p >= 0.9):
such speakers are mostly mixed clusters ("camp voices", "challenge voices"); a speaker whose confident
lines mostly agree (3+ lines, 70%) also gets that gender, for its short and unclear lines. Runs after the naming, before the translation, so both the voice and the Russian
grammar follow it. On the S10 SDH-labelled lines (E01/E12/E16, 243 lines) the right gender went from
78.2% to 84.4%; on S29E01/E05-E08 it touched 6 of 2756 lines (there are few unknown speakers).
Trusting the line's voice for speakers of known gender did not pay with audeering alone: it is often the
one that is wrong (shouting men at challenges heard as women; 37 fixed, 36 broken on S51).

The override for speakers of KNOWN gender (the gender only, never the speaker) averages audeering with a
logistic regression on a ResNet293 voice print of the whole line and acts only when the average is sure of the
OTHER gender (p_female >= 0.9 for a male speaker, <= 0.1 for a female one; ``line_gender.voice_override``). The
classifier and its own voice-print weights (a fine-tune on audited data, not the speaker model of the pipeline) are
downloaded from Hugging Face (``GENDER_REPO``, pinned; ``scripts/install.sh`` fetches them); offline without them the
older local pair (``models/gender/resnet293-ft-gender.npz`` on ``models/speaker/resnet293-ft``) is used. On 899 voiced
S51 contestant lines it changed 33: 31 fixed, 2 broken (86.2% -> 89.4% right gender; the older pair: 26, 22 fixed,
4 broken, 88.2%). The host is skipped; every
changed cue gets ``"gender_by_voice": {"p_female", "speaker_gender"}`` and is listed by ``stt.review`` for a
person; ``corrections.json`` ``{"gender": ...}`` wins over it (``stt.dub``).

    subs_env/bin/python -m stt.line_gender <episode folder> [--device cuda] [--voice-override 0.9] [--skip-speakers S01,S02]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stt.runner import load_json, save_json

log = logging.getLogger("stt.line_gender")

MIN_MS = 1200  # shorter lines carry too little voice
MAX_MS = 8000  # longer lines are classified on their first 8 s
CONFIDENT = 0.9  # p(female) >= CONFIDENT is female, <= 1 - CONFIDENT male
VOTE_LINES, VOTE_SHARE = 3, 0.7  # a speaker's gender from its lines: at least this many, this share agreeing
OVERRIDE_P = 0.9  # speakers of known gender: the averaged p must be this sure of the other gender
OVERRIDE_MIN_MS = 500  # shorter lines carry too little voice
MODELS = Path(__file__).resolve().parents[1] / "models"
CLASSIFIER = MODELS / "gender" / "resnet293-ft-gender.npz"  # the older local pair, used when the download fails
VOICE_WEIGHTS = MODELS / "speaker" / "resnet293-ft" / "avg_model.pt"
GENDER_REPO = "AngryBear9019/survivor-gender-resnet293"  # gender.npz + the avg_model.pt its prints come from
GENDER_REVISION = "2634450d433785c26fc03ecc0a1bb09cf10679dd"


def runtime_classifier() -> Path:
    """Optional classifier paired with the process-local speaker checkpoint."""
    return Path(os.environ.get("STT_GENDER_CLASSIFIER", str(CLASSIFIER))).expanduser().resolve(strict=True)


def gender_files() -> tuple[Path, Path]:
    """The classifier and the voice-print weights it was trained on.

    ``STT_GENDER_CLASSIFIER`` (experiments) pairs a classifier with ``STT_SPEAKER_CHECKPOINT``; otherwise the
    published pair is downloaded (cached by huggingface_hub), and the older local pair is the offline fallback."""
    from stt.refine_speakers import runtime_checkpoint

    if os.environ.get("STT_GENDER_CLASSIFIER"):
        return runtime_classifier(), runtime_checkpoint() or VOICE_WEIGHTS
    try:
        from huggingface_hub import hf_hub_download

        return (Path(hf_hub_download(GENDER_REPO, "gender.npz", revision=GENDER_REVISION)),
                Path(hf_hub_download(GENDER_REPO, "avg_model.pt", revision=GENDER_REVISION)))
    except Exception as exc:  # offline, hub down
        if runtime_checkpoint() is not None:  # the local pair's prints would come from that checkpoint
            raise ValueError("STT_SPEAKER_CHECKPOINT requires a paired STT_GENDER_CLASSIFIER for line gender") from exc
        log.warning("gender classifier %s not available (%s); using the local %s", GENDER_REPO, exc, CLASSIFIER)
        return CLASSIFIER, VOICE_WEIGHTS


def heard(p_female: float) -> str | None:
    return "female" if p_female >= CONFIDENT else "male" if p_female <= 1 - CONFIDENT else None


def decide(cues: list[dict[str, Any]], speakers: list[dict[str, Any]], p_female: dict[int, float]
           ) -> tuple[dict[str, str], dict[int, str]]:
    """({speaker id: gender} voted from the lines, {cue index: gender} for lines of still-unknown speakers)."""
    unknown = {s["id"] for s in speakers if s.get("gender") not in ("male", "female")}
    votes: dict[str, Counter] = defaultdict(Counter)
    for cue in cues:
        g = heard(p_female[cue["index"]]) if cue["index"] in p_female else None
        if cue.get("speaker") in unknown and g:
            votes[cue["speaker"]][g] += 1
    voted = {}
    for sid, counts in votes.items():
        g, n = counts.most_common(1)[0]
        if n >= VOTE_LINES and n / sum(counts.values()) >= VOTE_SHARE:
            voted[sid] = g
    lines = {cue["index"]: g for cue in cues  # a confident line keeps its own voice: these clusters often mix people
             if cue.get("speaker") in unknown and cue["index"] in p_female and (g := heard(p_female[cue["index"]]))}
    return voted, lines


def apply(cues: list[dict[str, Any]], speakers: list[dict[str, Any]], voted: dict[str, str],
          lines: dict[int, str]) -> int:
    """Write the decisions into the speaker cards and the cues; returns the number of cues changed."""
    for speaker in speakers:
        if speaker["id"] in voted:
            speaker.update(gender=voted[speaker["id"]], gender_source="line vote")
    changed = 0
    for cue in cues:
        gender = lines.get(cue["index"]) or voted.get(cue.get("speaker") or "")
        if gender and cue.get("gender") != gender:
            cue["gender"] = gender
            changed += 1
    return changed


def voice_prints(model: Any, clips: list[np.ndarray], *, max_s: float = 6.0, bucket_s: float = 0.25, min_s: float = 0.5,
                 batch: int = 64, device: str = "cuda", rate: int = 16000) -> np.ndarray:
    """Unit resnet293-ft prints of whole lines: centre crop to ``max_s``, length bucketed down to ``bucket_s``
    (so lines batch), at least ``min_s``. ``model`` is ``WespeakerTorch(...).model``. Shared with the training script."""
    import torch

    out = np.zeros((len(clips), 256), dtype=np.float32)
    buckets: dict[int, list[int]] = defaultdict(list)
    lo, hi = round(min_s / bucket_s), int(max_s / bucket_s)
    for i, clip in enumerate(clips):
        buckets[max(lo, min(int(len(clip) / rate / bucket_s), hi))].append(i)
    with torch.no_grad():
        for b, idx in buckets.items():
            n = int(b * bucket_s * rate)
            for k in range(0, len(idx), batch):
                part = idx[k:k + batch]
                x = np.stack([crop(clips[i], n) for i in part]).astype(np.float32)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    out[part] = model(torch.from_numpy(x).to(device)[:, None]).float().cpu().numpy()
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)


def crop(clip: np.ndarray, n: int) -> np.ndarray:
    """Centre crop to ``n`` samples (zero-padded at the end when shorter)."""
    if len(clip) >= n:
        start = (len(clip) - n) // 2
        return clip[start:start + n]
    return np.pad(clip, (0, n - len(clip)))


def override(cues: list[dict[str, Any]], speakers: list[dict[str, Any]], p_female: dict[int, float],
             threshold: float = OVERRIDE_P, skip_speakers: set[str] | frozenset[str] = frozenset()
             ) -> list[dict[str, Any]]:
    """Give a cue of a speaker of known gender the other gender when ``p_female`` is sure of it; returns the changes.

    Cues changed before are put back to their speaker's gender first, so a rerun decides afresh. Speakers of unknown
    gender (``decide`` handles them), ``skip_speakers`` (the host) and caption-only (ocr) cues keep their gender."""
    known = {s["id"]: s["gender"] for s in speakers if s.get("gender") in ("male", "female")}
    changes = []
    for cue in cues:
        sid = cue.get("speaker")
        if sid not in known:
            continue
        if cue.pop("gender_by_voice", None):
            cue["gender"] = known[sid]
        p = p_female.get(cue["index"])
        if p is None or sid in skip_speakers or cue.get("ocr"):
            continue
        heard = "female" if p >= threshold - 1e-9 else "male" if p <= 1 - threshold + 1e-9 else None  # exact ends count
        if heard and heard != known[sid]:
            cue["gender"] = heard
            cue["gender_by_voice"] = {"p_female": round(p, 3), "speaker_gender": known[sid]}
            changes.append({"index": cue["index"], "speaker": sid, "speaker_gender": known[sid], "gender": heard,
                            "p_female": round(p, 3)})
    return changes


def host_speakers(speakers: list[dict[str, Any]]) -> set[str]:
    """Speakers whose card names them the host (``stt.run`` passes the cast's hosts instead)."""
    return {s["id"] for s in speakers if re.search(r"\bhost\b", str(s.get("role", "")).lower())}


def voice_p_female(cues: list[dict[str, Any]], audio: np.ndarray, offset_ms: float, estimator: Any,
                   device: str) -> dict[int, float]:
    """p(female) per cue: audeering (raw female share, 6 s centre crop) averaged with the voice-print classifier."""
    from stt.audio import SAMPLE_RATE
    from stt.refine_speakers import WespeakerTorch

    classifier, weights = gender_files()
    data = np.load(classifier)
    if "embedding_sha256" in data:
        if str(data["embedding_sha256"]) != hashlib.sha256(weights.read_bytes()).hexdigest():
            raise ValueError(f"Gender classifier does not match its voice-print weights: {classifier}, {weights}")
    log.info("gender classifier loaded: path=%s sha256=%s", classifier,
             hashlib.sha256(classifier.read_bytes()).hexdigest())
    settings = {"max_s": float(data["max_s"]), "bucket_s": float(data["bucket_s"]), "min_s": float(data["min_s"])}
    clips: dict[int, np.ndarray] = {}
    for cue in cues:
        begin, end = (cue["start"] - offset_ms) / 1000, (cue["end"] - offset_ms) / 1000
        piece = audio[max(0, int(begin * SAMPLE_RATE)):max(0, int(end * SAMPLE_RATE))]
        if len(piece) >= SAMPLE_RATE * OVERRIDE_MIN_MS // 1000:
            clips[cue["index"]] = piece.astype(np.float32)
    if not clips:
        return {}
    ids = list(clips)
    prints = voice_prints(WespeakerTorch(str(data["model"]), device).model.eval(), [clips[i] for i in ids],
                          device=device, **settings)
    ours = 1.0 / (1.0 + np.exp(-(prints @ data["w"] + float(data["b"]))))
    n = int(settings["max_s"] * SAMPLE_RATE)
    out = {}
    for i, p_ours in zip(ids, ours):
        _, probs = estimator.predict(crop(clips[i], n) if len(clips[i]) > n else clips[i])
        out[i] = (float(p_ours) + float(probs[0])) / 2
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gender of the lines of speakers of unknown gender, from their voices")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice-override", type=float, default=OVERRIDE_P,
                        help="Speakers of known gender: change a line's gender when both models are this sure "
                             "(p_female >= x or <= 1 - x); 0 turns it off")
    parser.add_argument("--skip-speakers", default=None,
                        help="Speaker ids (comma separated) never changed by the override (the host); "
                             "default: speakers whose card role says host")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    subs_path, speakers_path = args.folder / "subs.en.json", args.folder / "speakers.json"
    cues, speakers = load_json(subs_path), load_json(speakers_path)
    unknown = {s["id"] for s in speakers if s.get("gender") not in ("male", "female")}
    todo = [c for c in cues if c.get("speaker") in unknown and c["duration"] >= MIN_MS]
    p_female: dict[int, float] = {}
    threshold = args.voice_override
    own = [c for c in cues if c.get("speaker") and c["speaker"] not in unknown and c["duration"] >= OVERRIDE_MIN_MS]
    classifier, weights = gender_files() if threshold else (CLASSIFIER, VOICE_WEIGHTS)
    if threshold and not (classifier.exists() and weights.exists()):
        log.warning("gender override skipped: %s or %s is missing (see scripts/speaker_training/train_gender.py)",
                    classifier, weights)
        threshold = 0.0
    voice_p: dict[int, float] = {}
    if todo or (threshold and own):
        from stt.audio import SAMPLE_RATE, read_wav
        from stt.voice import AgeGenderEstimator

        audio = read_wav(args.folder / "audio.wav")
        source = args.folder / "source.json"  # a test piece: audio.wav starts at its "start" (seconds)
        offset_ms = float(load_json(source).get("start") or 0.0) * 1000 if source.exists() else 0.0
        estimator = AgeGenderEstimator(device=args.device)
        estimator.load()
        if threshold and own:
            voice_p = voice_p_female(own, audio, offset_ms, estimator, args.device)
        for cue in todo:
            begin, end = cue["start"] - offset_ms, min(cue["end"], cue["start"] + MAX_MS) - offset_ms
            piece = audio[max(0, int(begin / 1000 * SAMPLE_RATE)):max(0, int(end / 1000 * SAMPLE_RATE))]
            if len(piece) < SAMPLE_RATE // 2:
                continue
            _, probs = estimator.predict(piece.astype(np.float32))
            adult = probs[:2] / max(float(probs[:2].sum()), 1e-9)
            p_female[cue["index"]] = round(float(adult[0]), 3)
    voted, lines = decide(cues, speakers, p_female)
    changed = apply(cues, speakers, voted, lines)
    hosts = set(args.skip_speakers.split(",")) if args.skip_speakers else host_speakers(speakers)
    swapped = override(cues, speakers, voice_p, threshold or 2.0, hosts)  # 2.0: never sure, only clears old marks
    save_json(speakers_path, speakers)
    save_json(subs_path, cues)
    save_json(args.folder / "line_gender.json", {"unknown_speakers": sorted(unknown), "voted": voted,
                                                 "lines": {str(k): v for k, v in lines.items()},
                                                 "p_female": {str(k): v for k, v in p_female.items()},
                                                 "voice_override": {"threshold": threshold, "hosts": sorted(hosts),
                                                                    "judged": len(voice_p), "changed": len(swapped),
                                                                    "cues": swapped}})
    log.info("line gender: %d unknown speakers, %d voted %s, %d more lines from their own voice, %d cues changed; "
             "known speakers: %d lines judged by voice, %d changed", len(unknown), len(voted), voted, len(lines),
             changed, len(voice_p), len(swapped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
