from __future__ import annotations

"""SPS metrics helper based on manifest.csv.

Назначение:
        - Прочитать manifest.csv (n,speaker_id,filename,text_length),
            измерить длительность соответствующих WAV и посчитать символы/сек (SPS).
        - Используется как утилита внутри synthesize_all_voices.py
            и может запускаться как отдельный CLI-скрипт.

Ожидания:
        - manifest.csv и WAV лежат в одной папке (по умолчанию tts_out/).
        - text_length уже посчитан по политике SPS (см. count_symbols_for_sps).
"""

import csv
from pathlib import Path
from typing import List, Dict, Any

import soundfile as sf


def get_duration_sec(path: Path) -> float:
    with sf.SoundFile(str(path)) as f:
        return len(f) / float(f.samplerate)


def compute_sps_from_manifest(
    manifest_path: Path,
    audio_root: Path | None = None,
    output_csv: Path | None = None,
) -> Dict[str, Any]:
    if audio_root is None:
        audio_root = manifest_path.parent
    if output_csv is None:
        output_csv = audio_root / "metrics_from_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = audio_root / row["filename"]
            if not audio_path.exists():
                print(f"[WARN] Skip missing audio: {audio_path}")
                continue

            duration = get_duration_sec(audio_path)
            text_len = int(row["text_length"])
            sps = text_len / duration if duration > 0 else 0.0

            rows.append(
                {
                    "n": int(row["n"]),
                    "speaker_id": int(row["speaker_id"]),
                    "filename": row["filename"],
                    "text_length": text_len,
                    "duration_sec": duration,
                    "symbols_per_second": sps,
                }
            )

    if not rows:
        raise RuntimeError("No valid rows in manifest or audio files are missing")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "speaker_id",
                "filename",
                "text_length",
                "duration_sec",
                "symbols_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_sps = sum(r["symbols_per_second"] for r in rows) / len(rows)
    min_sps = min(r["symbols_per_second"] for r in rows)
    max_sps = max(r["symbols_per_second"] for r in rows)

    return {
        "metrics_path": str(output_csv),
        "mean_cps": mean_sps,
        "min_cps": min_sps,
        "max_cps": max_sps,
        "rows": rows,
    }


def main() -> None:
    manifest_path = Path("tts_out") / "manifest.csv"
    audio_root = manifest_path.parent
    result = compute_sps_from_manifest(manifest_path, audio_root=audio_root)
    print(
        "[SPS] Finished. Metrics saved to {path}. Mean={mean:.3f}, Min={min_:.3f}, Max={max_:.3f}".format(
            path=result["metrics_path"],
            mean=result["mean_cps"],
            min_=result["min_cps"],
            max_=result["max_cps"],
        )
    )


if __name__ == "__main__":
    main()
