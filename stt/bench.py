"""Benchmark ASR models on a media clip against its English subtitles.

Each model runs in its own subprocess (optionally from its own venv), so peak VRAM
is measured per model and one crash/OOM does not stop the comparison.

Examples (from repo root):
    subs_env/bin/python -m stt.bench asr --media ep01.mkv --srt ep01.en.srt \
        --start 600 --duration 600 --models all --out output/stt_bench/ep01_10m
    subs_env/bin/python -m stt.bench summary output/stt_bench/ep01_10m output/stt_bench/ep02_10m
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from stt.runner import load_json as _load_json
from stt.runner import run_module
from stt.runner import save_json as _save_json

log = logging.getLogger("stt.bench")


# --------------------------------------------------------------------------- clip preparation


def prepare_clip(media: Path, srt: Path, start: float, duration: float | None, out_dir: Path) -> dict[str, Any]:
    """Cut the clip at subtitle gaps, save 16 kHz WAV, reference text and VAD chunks."""
    from stt.audio import load_audio, read_wav, save_wav
    from stt.reference import load_cues, snap_window, window_reference
    from stt.vad import asr_chunks, vad_settings

    cues = load_cues(srt)
    win_start, win_end = snap_window(cues, start, duration)
    clip_meta = {
        "media": str(media.resolve()),
        "srt": str(srt.resolve()),
        "window": [win_start, win_end],
        "duration": round(win_end - win_start, 3),
        "vad": vad_settings(),
    }
    clip_json, clip_wav, chunks_json = out_dir / "clip.json", out_dir / "clip.wav", out_dir / "chunks.json"

    if clip_json.exists() and clip_wav.exists() and chunks_json.exists():
        cached = _load_json(clip_json)
        if all(cached.get(k) == v for k, v in clip_meta.items()):
            log.info("Reusing prepared clip %s", out_dir)
            return cached

    log.info("Decoding %s [%.1f-%.1fs]", media.name, win_start, win_end)
    audio = load_audio(media, start=win_start, duration=win_end - win_start)
    save_wav(clip_wav, audio)
    audio = read_wav(clip_wav)  # every model sees exactly the saved PCM16 samples

    t0 = time.perf_counter()
    chunks = asr_chunks(audio)
    vad_s = time.perf_counter() - t0
    _save_json(chunks_json, [[round(s, 3), round(e, 3)] for s, e in chunks])

    reference = window_reference(cues, win_start, win_end)
    clip_meta.update(
        {
            "reference": reference,
            "chunks": len(chunks),
            "speech_s": round(sum(e - s for s, e in chunks), 3),
            "vad_s": round(vad_s, 3),
        }
    )
    _save_json(clip_json, clip_meta)
    (out_dir / "reference.txt").write_text(reference["text"] + "\n", encoding="utf-8")
    log.info("Clip %.1fs, %d chunks, %d reference cues", clip_meta["duration"], len(chunks), len(reference["cues"]))
    return clip_meta


# --------------------------------------------------------------------------- per-model runs


def run_model(
    clip_dir: Path, model: str, device: str, batch: int | None, force: bool, hotwords: str | None = None
) -> dict[str, Any]:
    """Run the worker for `model` in its venv; returns the worker result or an error record.

    With `hotwords` the result is stored and reported as `<model>@hw`.
    """
    from stt.asr import ASR_MODELS

    tag = f"{model}@hw" if hotwords else model
    hyp_path = clip_dir / "hyp" / f"{tag}.json"
    if hyp_path.exists() and not force:
        log.info("[%s] cached result", tag)
        return {**_load_json(hyp_path), "model": tag}

    spec = ASR_MODELS[model]
    args = [
        "asr",
        "--audio", str(clip_dir / "clip.wav"),
        "--chunks", str(clip_dir / "chunks.json"),
        "--model", model,
        "--out", str(hyp_path),
        "--device", device,
    ]  # fmt: skip
    if batch:
        args += ["--batch-size", str(batch)]
    if hotwords:
        args += ["--hotwords", hotwords]
    log_path = clip_dir / "logs" / f"{tag}.log"
    log.info("[%s] running in %s ...", model, spec.env)
    try:
        code, tail = run_module(spec.env, "stt.worker", args, log_path)
    except FileNotFoundError as exc:
        log.warning("[%s] skipped: %s", model, exc)
        return {"model": model, "error": str(exc)}
    if code != 0 or not hyp_path.exists():
        log.error("[%s] failed (exit %s), see %s", tag, code, log_path)
        return {"model": tag, "error": f"exit {code}: {tail}"}
    return {**_load_json(hyp_path), "model": tag}


# --------------------------------------------------------------------------- scoring and reports


def score(clip_dir: Path, clip: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Attach WER/CER and speed to a worker result; write text and alignment files."""
    from stt.metrics import alignment_report, error_rates
    from stt.types import Segment, segments_text

    row: dict[str, Any] = {"model": result["model"]}
    if "error" in result:
        row["error"] = result["error"]
        return row
    from stt.textfix import collapse_repeats

    raw_segments = [Segment.from_dict(s) for s in result["segments"]]
    segments, removed = collapse_repeats(raw_segments)
    hypothesis = segments_text(segments)
    row["wer_raw"] = error_rates(clip["reference"]["text"], segments_text(raw_segments))["wer"]
    row["repeats_removed"] = removed
    reference = clip["reference"]["text"]
    (clip_dir / "hyp" / f"{result['model']}.txt").write_text(hypothesis + "\n", encoding="utf-8")
    (clip_dir / "diff").mkdir(exist_ok=True)
    (clip_dir / "diff" / f"{result['model']}.txt").write_text(alignment_report(reference, hypothesis), encoding="utf-8")
    row.update(error_rates(reference, hypothesis))
    row.update({k: result[k] for k in ("load_s", "warmup_s", "transcribe_s", "peak_vram_mb", "batch_size")})
    row["rtfx"] = round(clip["duration"] / result["transcribe_s"], 1) if result["transcribe_s"] > 0 else None
    row["word_timestamps"] = any(s.get("words") for s in result["segments"])
    return row


def _fmt(value: Any, pattern: str = "{}") -> str:
    return "—" if value is None else pattern.format(value)


def markdown_table(rows: list[dict[str, Any]], title: str, details: list[str]) -> str:
    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["wer"])
    lines = [f"# {title}", "", *details, ""]
    lines += [
        "| Model | WER | WER raw | Loop words cut | CER | Sub / Del / Ins | Ref words | Load, s | Transcribe, s | RTFx "
        "| Peak VRAM, MB | Word times |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in ok:
        lines.append(
            f"| {r['model']} | {r['wer'] * 100:.2f}% | {r['wer_raw'] * 100:.2f}% | {r['repeats_removed']} "
            f"| {r['cer'] * 100:.2f}% "
            f"| {r['substitutions']} / {r['deletions']} / {r['insertions']} | {r['ref_words']} "
            f"| {_fmt(r.get('load_s'), '{:.1f}')} | {_fmt(r.get('transcribe_s'), '{:.1f}')} "
            f"| {_fmt(r.get('rtfx'), '{:.0f}')} | {_fmt(r.get('peak_vram_mb'))} "
            f"| {'yes' if r.get('word_timestamps') else 'no'} |"
        )
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | failed: {r['error'][:200]} |||||||||||")
    return "\n".join(lines) + "\n"


def cmd_asr(args: argparse.Namespace) -> None:
    from stt.asr import ASR_MODELS

    models = list(ASR_MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in ASR_MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}; known: {list(ASR_MODELS)}")
    media, srt, out_dir = Path(args.media), Path(args.srt), Path(args.out)
    for path in (media, srt):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    hotwords = None
    if args.hotwords_file:
        raw = re.split(r"[,\n]", Path(args.hotwords_file).read_text(encoding="utf-8"))
        hotwords = ", ".join(w.strip() for w in raw if w.strip())

    clip = prepare_clip(media, srt, args.start, args.duration, out_dir)
    rows = []
    for model in models:
        result = run_model(out_dir, model, args.device, args.batch_size, args.force, hotwords)
        rows.append(score(out_dir, clip, result))
        if "error" not in rows[-1]:
            r = rows[-1]
            log.info("[%s] WER %.2f%%  RTFx %s  VRAM %s MB", model, r["wer"] * 100, r["rtfx"], r["peak_vram_mb"])

    # Keep rows of models benchmarked in earlier runs on the same clip.
    report_path = out_dir / "report.json"
    if report_path.exists():
        fresh = {r["model"] for r in rows}
        rows = [r for r in _load_json(report_path)["rows"] if r["model"] not in fresh] + rows
    _save_json(report_path, {"clip": {k: v for k, v in clip.items() if k != "reference"}, "rows": rows})
    w0, w1 = clip["window"]
    details = [
        f"- Media: `{Path(clip['media']).name}`, window {w0:.1f}–{w1:.1f}s ({clip['duration']:.0f}s), "
        f"speech after VAD {clip['speech_s']:.0f}s in {clip['chunks']} chunks",
        "- Reference: English subtitles (SDH markup stripped); captions are not always verbatim, "
        "so absolute WER is pessimistic — compare models relative to each other.",
        f"- Device: {args.device}; RTFx = clip seconds / transcribe seconds (after warm-up).",
        "- WER is after the repetition guard (stt.textfix: n-gram loops > 4x cut); WER raw is before.",
    ]
    report = markdown_table(rows, f"ASR benchmark: {out_dir.name}", details)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def cmd_summary(args: argparse.Namespace) -> None:
    """Aggregate several clip reports: WER weighted by reference words, mean RTFx."""
    totals: dict[str, dict[str, float]] = {}
    clips = 0
    for directory in args.dirs:
        report = _load_json(Path(directory) / "report.json")
        clips += 1
        for r in report["rows"]:
            if "error" in r:
                continue
            t = totals.setdefault(r["model"], {"errors": 0, "words": 0, "audio": 0, "time": 0, "clips": 0, "vram": 0})
            t["errors"] += r["substitutions"] + r["deletions"] + r["insertions"]
            t["words"] += r["ref_words"]
            t["audio"] += report["clip"]["duration"]
            t["time"] += r["transcribe_s"]
            t["clips"] += 1
            t["vram"] = max(t["vram"], r.get("peak_vram_mb") or 0)
    lines = [
        f"# ASR summary over {clips} clips",
        "",
        "| Model | Clips | WER (pooled) | Audio, s | RTFx | Peak VRAM, MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, t in sorted(totals.items(), key=lambda kv: kv[1]["errors"] / kv[1]["words"]):
        lines.append(
            f"| {model} | {t['clips']:.0f} | {t['errors'] / t['words'] * 100:.2f}% | {t['audio']:.0f} "
            f"| {t['audio'] / t['time']:.0f} | {t['vram']:.0f} |"
        )
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local STT benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_asr = sub.add_parser("asr", help="Compare ASR models on a clip with reference subtitles")
    p_asr.add_argument("--media", required=True, help="Video/audio file")
    p_asr.add_argument("--srt", required=True, help="English subtitles for the same media (reference)")
    p_asr.add_argument("--start", type=float, default=0.0, help="Clip start, seconds (snapped to a cue gap)")
    p_asr.add_argument("--duration", type=float, default=None, help="Clip length, seconds (default: to the end)")
    p_asr.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    p_asr.add_argument("--out", required=True, help="Output directory for this clip")
    p_asr.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_asr.add_argument("--batch-size", type=int, default=None, help="Override per-model batch size")
    p_asr.add_argument("--force", action="store_true", help="Re-run models even if cached results exist")
    p_asr.add_argument("--hotwords-file", default=None,
                       help="Names/terms (comma or newline separated) for biasing; results are tagged <model>@hw")
    p_asr.set_defaults(func=cmd_asr)

    p_sum = sub.add_parser("summary", help="Aggregate several clip reports")
    p_sum.add_argument("dirs", nargs="+", help="Clip output directories with report.json")
    p_sum.add_argument("--out", default=None, help="Write the markdown summary here")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
