"""Model stages run as subprocesses (each in its own venv) with timing and peak VRAM.

Internal entry point used by `stt.transcribe` and the benchmarks:
    python -m stt.worker asr --audio a.wav --chunks c.json --model whisper-large-v3 --out asr.json
    python -m stt.worker diarize --audio a.wav [a2.wav ...] --model pyannote-community-1 --out d.json
    python -m stt.worker prints --audio a.wav --diarization d.json --out d.json   (cluster prints, subs_env)
    python -m stt.worker voice --audio a.wav --diarization d.json --out voice.json
    python -m stt.worker align --audio a.wav --asr asr.json --out asr.aligned.json [--aligner ctc|qwen]
                                                     (ctc: subs_env, qwen: stt_qwen_env)
    python -m stt.worker ocr --media ep.mkv --out ocr.json [--start S --duration D]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from stt.runner import load_json, run_module, save_json

log = logging.getLogger("stt.worker")


def _vram(peak: float | None) -> int | None:
    return None if peak is None else round(peak)


def run_asr(
    audio_path: Path,
    chunks_path: Path,
    model: str,
    out: Path,
    device: str,
    batch: int | None,
    hotwords: str | None = None,
) -> None:
    """Load an ASR model, warm it up on the first chunk, transcribe all chunks."""
    from stt.asr import create_backend
    from stt.audio import read_wav
    from stt.gpu import VramPeak

    audio = read_wav(audio_path)
    chunks = [tuple(c) for c in load_json(chunks_path)]
    backend = create_backend(model, device=device, batch_size=batch, hotwords=hotwords)
    with VramPeak() as vram:
        t0 = time.perf_counter()
        backend.load()
        t1 = time.perf_counter()
        backend.transcribe(audio, chunks[:1])
        t2 = time.perf_counter()
        segments = backend.transcribe(audio, chunks)
        t3 = time.perf_counter()
    save_json(
        out,
        {
            "model": model,
            "model_id": backend.spec.model_id,
            "device": device,
            "batch_size": backend.batch_size,
            "hotwords": hotwords,
            "load_s": round(t1 - t0, 3),
            "warmup_s": round(t2 - t1, 3),
            "transcribe_s": round(t3 - t2, 3),
            "peak_vram_mb": _vram(vram.peak_mb),
            "segments": [s.to_dict() for s in segments],
        },
    )


def run_diarize(
    audio_paths: list[Path],
    model: str,
    outs: list[Path],
    device: str,
    speakers: dict[str, int | None],
    options: dict[str, Any] | None = None,
) -> None:
    """Load a diarizer once and process each audio file into its own JSON."""
    from stt.audio import read_wav
    from stt.diarize import create_diarizer
    from stt.gpu import VramPeak

    diarizer = create_diarizer(model, device=device, options=options)
    with VramPeak() as vram:
        t0 = time.perf_counter()
        diarizer.load()
        load_s = time.perf_counter() - t0
        for audio_path, out in zip(audio_paths, outs):
            audio = read_wav(audio_path)
            t1 = time.perf_counter()
            result = diarizer.diarize(audio, **speakers)
            elapsed = time.perf_counter() - t1
            data: dict[str, Any] = result.to_dict()
            data.update({"model": model, "load_s": round(load_s, 3), "diarize_s": round(elapsed, 3)})
            save_json(out, data)
            log.info("%s: %d speakers in %.1fs", audio_path.name, len(result.speakers), elapsed)
    for out in outs:
        data = load_json(out)
        data["peak_vram_mb"] = _vram(vram.peak_mb)
        save_json(out, data)


def run_prints(audio_path: Path, diarization_path: Path, out: Path, device: str) -> None:
    """Add cluster prints (the voice bank's embedding space) to a diarization that has none."""
    from stt.audio import read_wav
    from stt.voice_bank import SegmentEmbedder, cluster_prints

    data = load_json(diarization_path)
    t0 = time.perf_counter()
    data["embeddings"] = cluster_prints(data["turns"], read_wav(audio_path), SegmentEmbedder(device))
    data["prints_s"] = round(time.perf_counter() - t0, 1)
    save_json(out, data)
    log.info("prints: %d speakers in %.1fs", len(data["embeddings"]), data["prints_s"])


def run_voice(audio_path: Path, diarization_path: Path, out: Path, device: str, threshold: float) -> None:
    """Gender/age profile per diarized speaker."""
    from stt.audio import read_wav
    from stt.diarize import Diarization
    from stt.gpu import VramPeak
    from stt.voice import AGE_GENDER_MODEL, AgeGenderEstimator, profile_speakers

    audio = read_wav(audio_path)
    diarization = Diarization.from_dict(load_json(diarization_path))
    estimator = AgeGenderEstimator(device=device)
    with VramPeak() as vram:
        t0 = time.perf_counter()
        estimator.load()
        t1 = time.perf_counter()
        profiles = profile_speakers(audio, diarization, estimator, threshold=threshold)
        t2 = time.perf_counter()
    save_json(
        out,
        {
            "model": AGE_GENDER_MODEL,
            "threshold": threshold,
            "load_s": round(t1 - t0, 3),
            "run_s": round(t2 - t1, 3),
            "peak_vram_mb": _vram(vram.peak_mb),
            "speakers": [p.to_dict() for p in profiles],
        },
    )


def run_refine(audio_path: Path, diarization_path: Path, out: Path, model: str, margin: float, device: str) -> None:
    """Re-assign diarized turns that sound clearly like another speaker (stt.refine_speakers)."""
    from stt.audio import read_wav
    from stt.refine_speakers import refine_diarization

    t0 = time.perf_counter()
    data = refine_diarization(read_wav(audio_path), load_json(diarization_path), model, margin, device)
    data["refine_s"] = round(time.perf_counter() - t0, 1)
    log.info("refine: %s in %.1fs", data["refined"], data["refine_s"])
    save_json(out, data)


def run_align(audio_path: Path, asr_path: Path, out: Path, device: str, kind: str = "ctc") -> None:
    """Add forced-alignment word times to an ASR result that has none."""
    from stt.align import ALIGNER_MODEL, CTC_MODEL, CtcAligner, ForcedAligner
    from stt.audio import read_wav
    from stt.gpu import VramPeak
    from stt.types import Segment

    audio = read_wav(audio_path)
    data = load_json(asr_path)
    segments = [Segment.from_dict(s) for s in data["segments"]]
    aligner: CtcAligner | ForcedAligner = CtcAligner(device=device) if kind == "ctc" else ForcedAligner(device=device)
    with VramPeak() as vram:
        t0 = time.perf_counter()
        aligner.load()
        t1 = time.perf_counter()
        aligned = aligner.align(audio, segments)
        t2 = time.perf_counter()
    save_json(
        out,
        {
            **data,
            "segments": [s.to_dict() for s in aligned],
            "aligned_by": CTC_MODEL if kind == "ctc" else ALIGNER_MODEL,
            "align_load_s": round(t1 - t0, 3),
            "align_s": round(t2 - t1, 3),
            "align_peak_vram_mb": _vram(vram.peak_mb),
        },
    )


def run_ocr(media: Path, out: Path, device: str, start: float, duration: float | None, fps: float) -> None:
    """On-screen captions and name cards (times relative to `start`)."""
    from stt.gpu import VramPeak
    from stt.ocr import FrameOcr, run_ocr as scan

    ocr = FrameOcr(device=device)
    with VramPeak() as vram:
        t0 = time.perf_counter()
        ocr.load()
        result = scan(media, ocr, start=start, duration=duration, fps=fps)
        elapsed = time.perf_counter() - t0
    save_json(out, {**result, "ocr_s": round(elapsed, 1), "peak_vram_mb": _vram(vram.peak_mb)})


# Messages of PyTorch, CTranslate2 and ONNX Runtime when the GPU has no memory left.
_GPU_OOM = ("CUDA out of memory", "OutOfMemoryError", "CUDA failed with error out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED", "cudaErrorMemoryAllocation", "Failed to allocate memory")


def gpu_out_of_memory(log_text: str) -> bool:
    return any(marker in log_text for marker in _GPU_OOM)


def _smaller_batch(args: list[str]) -> list[str]:
    """ASR args with half the batch size (the model's default if none was given)."""
    from stt.asr import ASR_MODELS

    args = list(args)
    if "--batch-size" in args:
        i = args.index("--batch-size") + 1
        args[i] = str(max(1, int(args[i]) // 2))
    else:
        model = args[args.index("--model") + 1]
        args += ["--batch-size", str(max(1, ASR_MODELS[model].batch_size // 2))]
    return args


def call(env: str, args: list[str], out: Path, log_path: Path, *, oom_retries: int = 2,
         oom_wait_s: float = 60.0) -> dict[str, Any]:
    """Run a worker command in `env`; return its JSON output or raise with the log tail.

    A stage that ran out of GPU memory (another program on the card) is retried after a pause,
    an ASR stage with half the batch size.
    """
    for attempt in range(oom_retries + 1):
        code, tail = run_module(env, "stt.worker", args, log_path)
        if code == 0 and out.exists():
            return load_json(out)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if attempt == oom_retries or not gpu_out_of_memory(log_text):
            break
        if args[0] == "asr":
            args = _smaller_batch(args)
        log.warning("stt.worker %s ran out of GPU memory; retry %d/%d in %.0f s%s", args[0], attempt + 1,
                    oom_retries, oom_wait_s, f" (batch {args[args.index('--batch-size') + 1]})"
                    if "--batch-size" in args else "")
        log_path.with_suffix(f".oom{attempt + 1}.log").write_text(log_text, encoding="utf-8")
        time.sleep(oom_wait_s)
    raise RuntimeError(f"stt.worker {args[0]} failed (exit {code}), see {log_path}: {tail}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STT model stage worker (internal)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_asr = sub.add_parser("asr")
    p_asr.add_argument("--audio", required=True)
    p_asr.add_argument("--chunks", required=True)
    p_asr.add_argument("--model", required=True)
    p_asr.add_argument("--out", required=True)
    p_asr.add_argument("--device", default="cuda")
    p_asr.add_argument("--batch-size", type=int, default=None)
    p_asr.add_argument("--hotwords", default=None, help="Biasing words, e.g. cast names (comma-separated)")

    p_diar = sub.add_parser("diarize")
    p_diar.add_argument("--audio", required=True, nargs="+")
    p_diar.add_argument("--model", required=True)
    p_diar.add_argument("--out", required=True, nargs="+")
    p_diar.add_argument("--device", default="cuda")
    p_diar.add_argument("--num-speakers", type=int, default=None)
    p_diar.add_argument("--min-speakers", type=int, default=None)
    p_diar.add_argument("--max-speakers", type=int, default=None)
    p_diar.add_argument("--no-median-filter", dest="median_filter", action="store_false",
                        help="DiariZen: keep the raw segmentation (no 11-frame median filter)")
    p_diar.add_argument("--exclusive", action="store_true",
                        help="DiariZen: also return exclusive (one speaker per frame) turns")

    p_voice = sub.add_parser("voice")
    p_voice.add_argument("--audio", required=True)
    p_voice.add_argument("--diarization", required=True)
    p_voice.add_argument("--out", required=True)
    p_voice.add_argument("--device", default="cuda")
    p_voice.add_argument("--threshold", type=float, default=0.7)

    p_prints = sub.add_parser("prints")
    p_prints.add_argument("--audio", required=True)
    p_prints.add_argument("--diarization", required=True)
    p_prints.add_argument("--out", required=True)
    p_prints.add_argument("--device", default="cuda")

    p_refine = sub.add_parser("refine")
    p_refine.add_argument("--audio", required=True)
    p_refine.add_argument("--diarization", required=True)
    p_refine.add_argument("--out", required=True)
    p_refine.add_argument("--model", default="resnet293")
    p_refine.add_argument("--margin", type=float, default=0.1)
    p_refine.add_argument("--device", default="cuda")

    p_align = sub.add_parser("align")
    p_align.add_argument("--audio", required=True)
    p_align.add_argument("--asr", required=True, help="ASR result JSON (stt.worker asr output)")
    p_align.add_argument("--out", required=True)
    p_align.add_argument("--device", default="cuda")
    p_align.add_argument("--aligner", default="ctc", choices=["ctc", "qwen"])

    p_ocr = sub.add_parser("ocr")
    p_ocr.add_argument("--media", required=True)
    p_ocr.add_argument("--out", required=True)
    p_ocr.add_argument("--device", default="cuda")
    p_ocr.add_argument("--start", type=float, default=0.0)
    p_ocr.add_argument("--duration", type=float, default=None)
    p_ocr.add_argument("--fps", type=float, default=2.0)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.command == "asr":
        run_asr(Path(args.audio), Path(args.chunks), args.model, Path(args.out), args.device, args.batch_size,
                args.hotwords)
    elif args.command == "diarize":
        if len(args.audio) != len(args.out):
            parser.error("--audio and --out need the same number of paths")
        speakers = {
            "num_speakers": args.num_speakers,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        }
        run_diarize([Path(p) for p in args.audio], args.model, [Path(p) for p in args.out], args.device, speakers,
                    {"median_filter": args.median_filter, "exclusive": args.exclusive})
    elif args.command == "ocr":
        run_ocr(Path(args.media), Path(args.out), args.device, args.start, args.duration, args.fps)
    elif args.command == "align":
        run_align(Path(args.audio), Path(args.asr), Path(args.out), args.device, args.aligner)
    elif args.command == "prints":
        run_prints(Path(args.audio), Path(args.diarization), Path(args.out), args.device)
    elif args.command == "refine":
        run_refine(Path(args.audio), Path(args.diarization), Path(args.out), args.model, args.margin, args.device)
    elif args.command == "voice":
        run_voice(Path(args.audio), Path(args.diarization), Path(args.out), args.device, args.threshold)


if __name__ == "__main__":
    main()
