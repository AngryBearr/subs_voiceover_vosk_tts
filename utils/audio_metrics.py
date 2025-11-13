from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf  # type: ignore

from utils.text_sanitize import count_symbols_for_sps


def compute_symbols_per_second(
    outputs_folder: str,
    text: str,
    expected_prefix: str = "tts_",
    sort: bool = True,
) -> Dict[str, object]:
    """Compute symbols-per-second (SPS) metrics for a set of WAV files.

    Policy:
      - Count punctuation.
      - Exclude spaces/newlines and RUAccent stress markers '+'.
      - Use the same symbol count for all files (single text synthesized per speaker).

    Returns a dictionary with per-file entries and aggregate stats.
    Also writes metrics.csv inside outputs_folder.
    """
    folder = Path(outputs_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Outputs folder not found: {outputs_folder}")

    wav_files = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(expected_prefix) and p.suffix.lower() == ".wav"]
    if sort:
        wav_files.sort(key=lambda p: p.name)

    chars = count_symbols_for_sps(text)
    per_file: List[Dict[str, object]] = []
    cps_values: List[float] = []

    for wf in wav_files:
        try:
            info = sf.info(str(wf))
            duration_sec = 0.0
            if info.frames and info.samplerate:
                duration_sec = info.frames / float(info.samplerate)
            cps: Optional[float] = None
            if duration_sec > 0:
                cps = chars / duration_sec
                cps_values.append(cps)
            per_file.append({
                "filename": wf.name,
                "duration_sec": duration_sec,
                "chars": chars,
                "cps": cps,
            })
        except Exception as e:
            per_file.append({
                "filename": wf.name,
                "duration_sec": None,
                "chars": chars,
                "cps": None,
                "error": str(e),
            })

    aggregate = {
        "mean_cps": statistics.mean(cps_values) if cps_values else None,
        "min_cps": min(cps_values) if cps_values else None,
        "max_cps": max(cps_values) if cps_values else None,
        "files": len(per_file),
        "chars": chars,
    }

    # Write metrics.csv
    metrics_path = folder / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("filename,duration_sec,chars,cps\n")
        for row in per_file:
            f.write(
                f"{row['filename']},{row['duration_sec'] if row['duration_sec'] is not None else ''},{row['chars']},{row['cps'] if row['cps'] is not None else ''}\n"
            )

    return {"per_file": per_file, "aggregate": aggregate, "metrics_path": str(metrics_path)}


if __name__ == "__main__":  # minimal CLI usage
    import argparse
    parser = argparse.ArgumentParser(description="Compute SPS metrics for synthesized WAV files.")
    parser.add_argument("--folder", required=True, help="Folder with WAV files")
    parser.add_argument("--text-file", required=True, help="Text file used for synthesis")
    parser.add_argument("--prefix", default="tts_", help="Filename prefix")
    args = parser.parse_args()
    text_content = Path(args.text_file).read_text(encoding="utf-8")
    result = compute_symbols_per_second(args.folder, text_content, expected_prefix=args.prefix)
    print(json.dumps(result, ensure_ascii=False, indent=2))
