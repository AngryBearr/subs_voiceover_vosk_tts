"""Benchmark diarization models on VoxConverse (TV debates/news/talk shows with speaker labels).

Data: one or more parquet shards of `diarizers-community/voxconverse` (test split), e.g.
    hf download diarizers-community/voxconverse data/test-00006-of-00011.parquet \
        --repo-type dataset --local-dir output/stt_data/voxconverse

Run (from repo root):
    subs_env/bin/python -m stt.bench_diar --parquet output/stt_data/voxconverse/data/test-00006-of-00011.parquet \
        --models pyannote-community-1 --out output/stt_bench/diar_voxconverse
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any

from stt.runner import load_json, run_module, save_json

log = logging.getLogger("stt.bench_diar")

COLLAR_S = 0.25  # VoxConverse challenge scoring collar


def prepare(parquets: list[Path], data_dir: Path, max_files: int | None) -> list[dict[str, Any]]:
    """Extract WAVs and reference turns from parquet shards (cached in `data_dir`)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    index_path = data_dir / "files.json"
    if index_path.exists():
        files = load_json(index_path)
        if max_files is None or len(files) >= max_files:
            return files[:max_files] if max_files else files

    files: list[dict[str, Any]] = []
    data_dir.mkdir(parents=True, exist_ok=True)
    for parquet in parquets:
        for row in pq.read_table(parquet).to_pylist():
            name = Path(row["audio"]["path"]).stem
            wav_path = data_dir / f"{name}.wav"
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"{name}: expected mono 16 kHz, got {sr} Hz / {audio.shape}")
            sf.write(wav_path, audio, sr, subtype="PCM_16")
            turns = [
                {"start": round(s, 3), "end": round(e, 3), "speaker": spk}
                for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"])
            ]
            files.append({"name": name, "wav": str(wav_path), "duration": round(len(audio) / sr, 3), "turns": turns})
            if max_files and len(files) >= max_files:
                break
        if max_files and len(files) >= max_files:
            break
    save_json(index_path, files)
    return files


def _annotation(turns: list[dict[str, Any]]) -> Any:
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for i, t in enumerate(turns):
        if t["end"] > t["start"]:
            annotation[Segment(t["start"], t["end"]), i] = t["speaker"]
    return annotation


def score(files: list[dict[str, Any]], hyp_dir: Path) -> dict[str, Any]:
    """DER (collar 0.25 s and strict), its components, speaker-count errors and speed."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    summary = load_json(hyp_dir / "_summary.json")
    der, strict = DiarizationErrorRate(collar=COLLAR_S), DiarizationErrorRate(collar=0.0)
    per_file = []
    count_errors = []
    for item in files:
        hyp = load_json(hyp_dir / f"{item['name']}.json")
        ref_ann, hyp_ann = _annotation(item["turns"]), _annotation(hyp["turns"])
        detail = der(ref_ann, hyp_ann, detailed=True)
        strict(ref_ann, hyp_ann)
        n_ref, n_hyp = len(ref_ann.labels()), len(hyp_ann.labels())
        count_errors.append(n_hyp - n_ref)
        per_file.append(
            {
                "name": item["name"],
                "duration": item["duration"],
                "der": round(detail["diarization error rate"], 4),
                "ref_speakers": n_ref,
                "hyp_speakers": n_hyp,
                "time_s": summary["timings"][item["name"]],
            }
        )
    components = der[:]  # accumulated totals
    total = components["total"]
    audio_s = sum(f["duration"] for f in files)
    time_s = sum(summary["timings"].values())
    return {
        "model": summary["model"],
        "der": round(abs(der), 4),
        "der_strict": round(abs(strict), 4),
        "missed": round(components["missed detection"] / total, 4),
        "false_alarm": round(components["false alarm"] / total, 4),
        "confusion": round(components["confusion"] / total, 4),
        "speaker_count_exact": round(sum(e == 0 for e in count_errors) / len(count_errors), 3),
        "speaker_count_mae": round(sum(abs(e) for e in count_errors) / len(count_errors), 2),
        "load_s": summary["load_s"],
        "time_s": round(time_s, 1),
        "rtfx": round(audio_s / time_s, 1) if time_s else None,
        "peak_vram_mb": summary["peak_vram_mb"],
        "files": per_file,
    }


def markdown(rows: list[dict[str, Any]], files: list[dict[str, Any]]) -> str:
    minutes = sum(f["duration"] for f in files) / 60
    speakers = [len({t["speaker"] for t in f["turns"]}) for f in files]
    lines = [
        "# Diarization benchmark: VoxConverse",
        "",
        f"- {len(files)} files, {minutes:.0f} min, {min(speakers)}–{max(speakers)} speakers per file",
        f"- DER with {COLLAR_S}s collar (challenge setting) and strict (no collar); overlap is scored.",
        "",
        "| Model | DER | DER strict | Missed | False alarm | Confusion | Exact spk count | Spk count MAE "
        "| Time, s | RTFx | Peak VRAM, MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted((r for r in rows if "error" not in r), key=lambda r: r["der"]):
        lines.append(
            f"| {r['model']} | {r['der'] * 100:.2f}% | {r['der_strict'] * 100:.2f}% | {r['missed'] * 100:.2f}% "
            f"| {r['false_alarm'] * 100:.2f}% | {r['confusion'] * 100:.2f}% | {r['speaker_count_exact'] * 100:.0f}% "
            f"| {r['speaker_count_mae']:.2f} | {r['time_s']:.0f} | {r['rtfx']:.0f} | {r['peak_vram_mb']} |"
        )
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | failed: {r['error'][:200]} ||||||||||")
    return "\n".join(lines) + "\n"


def cmd_run(args: argparse.Namespace) -> None:
    from stt.diarize import DIARIZERS

    models = list(DIARIZERS) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in DIARIZERS]
    if unknown:
        raise SystemExit(f"Unknown diarizers: {unknown}; known: {list(DIARIZERS)}")
    out_dir = Path(args.out)
    files = prepare([Path(p) for p in args.parquet], out_dir / "data", args.max_files)
    files_path = out_dir / "files_selected.json"
    save_json(files_path, files)
    log.info("%d files, %.0f min", len(files), sum(f["duration"] for f in files) / 60)

    rows = []
    for model in models:
        hyp_dir = out_dir / "hyp" / model
        if args.force or not (hyp_dir / "_summary.json").exists():
            spec = DIARIZERS[model]
            log.info("[%s] running in %s ...", model, spec.env)
            outs = [hyp_dir / f"{f['name']}.json" for f in files]
            worker_args = ["diarize", "--model", model, "--device", args.device,
                           "--audio", *[f["wav"] for f in files], "--out", *[str(o) for o in outs]]
            try:
                code, tail = run_module(spec.env, "stt.worker", worker_args, out_dir / "logs" / f"{model}.log")
            except FileNotFoundError as exc:
                rows.append({"model": model, "error": str(exc)})
                continue
            if code != 0 or not all(o.exists() for o in outs):
                log.error("[%s] failed, see logs/%s.log", model, model)
                rows.append({"model": model, "error": f"exit {code}: {tail}"})
                continue
            first = load_json(outs[0])
            save_json(
                hyp_dir / "_summary.json",
                {
                    "model": model,
                    "load_s": first["load_s"],
                    "peak_vram_mb": first.get("peak_vram_mb"),
                    "timings": {f["name"]: load_json(o)["diarize_s"] for f, o in zip(files, outs)},
                },
            )
        row = score(files, hyp_dir)
        rows.append(row)
        log.info("[%s] DER %.2f%%  RTFx %s", model, row["der"] * 100, row["rtfx"])

    save_json(out_dir / "report.json", rows)
    report = markdown(rows, files)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diarization benchmark on VoxConverse")
    parser.add_argument("--parquet", nargs="+", help="VoxConverse parquet shard(s)")
    parser.add_argument("--models", default="all", help="Comma-separated diarizer names or 'all'")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true", help="Re-run models even if cached results exist")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not args.parquet or not args.out:
        parser.error("--parquet and --out are required")
    cmd_run(args)


if __name__ == "__main__":
    main()
