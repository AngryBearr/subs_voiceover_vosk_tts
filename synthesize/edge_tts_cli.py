from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from synthesize.edge_tts_batch import synthesize_two_voices


def _results_to_json(results) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "voice": r.voice,
                "out_dir": str(r.out_dir),
                "manifest_path": str(r.manifest_path),
                "metrics_path": str(r.metrics_path),
                "mean_sps": r.mean_sps,
                "min_sps": r.min_sps,
                "max_sps": r.max_sps,
                "mean_spm": r.mean_spm,
                "min_spm": r.min_spm,
                "max_spm": r.max_spm,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge TTS batch synthesis for subtitle JSON + SPS/SPM metrics.")
    parser.add_argument("--json", required=True, type=Path, help="Path to JSON list with fields: index, text")
    parser.add_argument(
        "--voices",
        nargs=2,
        default=["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"],
        help="Two Edge voices to synthesize",
    )
    parser.add_argument("--output-root", type=Path, default=Path("tts_edge_out"), help="Output root folder")
    parser.add_argument("--rate", default="+0%", help="Edge rate, e.g. +10% or -10%")
    parser.add_argument("--volume", default="+0%", help="Edge volume, e.g. +0%")
    parser.add_argument("--pitch", default="+0Hz", help="Edge pitch, e.g. +0Hz")
    parser.add_argument("--retries", type=int, default=3, help="Retries per segment on transient errors")
    parser.add_argument("--connect-timeout", type=int, default=30, help="Connect timeout (seconds)")
    parser.add_argument("--receive-timeout", type=int, default=300, help="Receive timeout (seconds)")
    parser.add_argument("--keep-mp3", action="store_true", help="Keep intermediate mp3 files")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary (default: <output-root>/summary.json)",
    )

    args = parser.parse_args()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (output_root / "summary.json")

    results = synthesize_two_voices(
        json_path=args.json,
        voices=args.voices,
        output_root=output_root,
        rate=str(args.rate),
        volume=str(args.volume),
        pitch=str(args.pitch),
        keep_mp3=bool(args.keep_mp3),
        retries=int(args.retries),
        connect_timeout=int(args.connect_timeout),
        receive_timeout=int(args.receive_timeout),
    )

    # Print concise summary
    for r in results:
        print(
            "[edge-tts] {voice}: mean_sps={mean:.3f} (min={min_:.3f}, max={max_:.3f}) | mean_spm={spm:.1f}".format(
                voice=r.voice,
                mean=r.mean_sps,
                min_=r.min_sps,
                max_=r.max_sps,
                spm=r.mean_spm,
            )
        )
        print(f"  metrics: {r.metrics_path}")

    summary = {
        "input_json": str(args.json),
        "voices": list(args.voices),
        "results": _results_to_json(results),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[edge-tts] Summary: {summary_path}")


if __name__ == "__main__":
    main()
