from __future__ import annotations

"""Прогон всех голосов много-спикерной модели Vosk TTS.

Назначение:
    - Быстро синтезировать один и тот же текст для всех speaker_id модели (по умолчанию 0..56).
    - Получить набор WAV для оценки качества, темпа произношения и расчёта метрики символы/сек (SPS).
    - Сформировать manifest.csv для дальнейших анализов (индекс, speaker_id, имя файла, длина текста).

Особенности:
    - Принудительный CPU через create_synth (внутри _ForceCPUProviders).
    - Опциональное акцентирование RUAccent (можно отключить --no-accent).
    - Санитизация входного текста (кавычки, тире, управляющие символы, эмодзи, пробелы).
    - Разбиение слишком длинного текста на чанки по предложениям.

Примеры запуска:
    Быстрый прогон 2 голосов, без акцентирования (для smoke-теста):
        python -m synthesize.synthesize_all_voices --limit 2 --no-accent

    С акцентированием (CPU), кастомный префикс и внешним файлом текста:
        python -m synthesize.synthesize_all_voices --text-file .\temp\long_ru_text.txt --file-prefix vosk10_

    Увеличить максимальную длину чанка для акцентирования (до 1500 символов):
        python -m synthesize.synthesize_all_voices --max-chars-per-chunk 1500

    Сразу посчитать SPS (metrics.csv) после синтеза:
        python -m synthesize.synthesize_all_voices --compute-sps

Выходы:
    - WAV: <output-folder>/<file-prefix><n>.wav (нумерация с 1)
    - manifest.csv: n,speaker_id,filename,text_length (text_length по политике SPS)

Назначение text_length: последующее сравнение темпа (символов/сек) между голосами.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional, List

from ruaccent import RUAccent  # type: ignore

from synthesize.synthesize import create_synth  # reuse CPU provider patching
from utils.subs_utils import ensure_folder_exists
from utils.text_sanitize import sanitize_text, count_symbols_for_sps, split_into_chunks
from utils.audio_metrics import compute_symbols_per_second


DEFAULT_RU_TEXT = (
    "Это короткий нейтральный текст для оценки скорости синтеза. "
    "Он содержит базовую пунктуацию, числа 2025 и несколько стандартных оборотов. "
    "Задача — измерить длительность звука и вычислить символы в секунду. "
    "Проверим, как система справляется с простыми предложениями, паузами и многоточием..."
)


def replace_ellipsis(text: str) -> str:
    return text.replace("...", "..")


def restore_ellipsis(text: str) -> str:
    return text.replace("..", "...")


def load_text(arg_text: Optional[str], text_file: Optional[str]) -> str:
    if arg_text and arg_text.strip():
        return arg_text
    if text_file:
        p = Path(text_file)
        if not p.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")
        return p.read_text(encoding="utf-8")
    return DEFAULT_RU_TEXT


def resolve_model_dir(model_name: str, model_path: Optional[str]) -> Path:
    if model_path:
        return Path(model_path)
    return Path.cwd() / "models" / model_name


def read_num_speakers(model_dir: Path) -> int:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return 57
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("speaker_id_map"), dict):
            return max(1, len(data["speaker_id_map"]))
        if isinstance(data.get("num_speakers"), int):
            return max(1, int(data["num_speakers"]))
    return 57


def accentize_text(text: str, device: str = "CPU", max_chunk: int = 1200) -> str:
    text = sanitize_text(text)
    chunks = split_into_chunks(text, max_len=max_chunk)
    if not chunks:
        return ""
    acc = RUAccent()
    acc.load(
        omograph_model_size="turbo3.1",
        use_dictionary=True,
        custom_dict=None,
        device=device,
    )
    out: List[str] = []
    for ch in chunks:
        safe = replace_ellipsis(ch)
        accented = acc.process(safe)
        out.append(restore_ellipsis(accented))
    return " ".join(out)


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model path does not exist or is not a directory: {model_dir}")
    required = [model_dir / "model.onnx", model_dir / "dictionary", model_dir / "config.json"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory is missing required files: {', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a text with every speaker of a Vosk TTS multi-speaker model (CPU)."
    )
    parser.add_argument("--model-name", default="vosk-model-tts-ru-0.10-multi")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-folder", default="tts_out")
    parser.add_argument("--file-prefix", default="tts_")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--speech-rate", type=float, default=1.0)
    parser.add_argument("--no-accent", action="store_true")
    parser.add_argument("--max-chars-per-chunk", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of speakers for a quick run")
    parser.add_argument("--compute-sps", action="store_true", help="Compute SPS metrics (metrics.csv) after synthesis")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    text = load_text(args.text, args.text_file)
    if not text or not text.strip():
        raise ValueError("Input text must be non-empty.")

    model_dir = resolve_model_dir(args.model_name, args.model_path)
    validate_model_dir(model_dir)

    # Prepare text (optional accent)
    if args.no_accent:
        final_text = sanitize_text(text)
    else:
        try:
            final_text = accentize_text(text, device="CPU", max_chunk=args.max_chars_per_chunk)
        except Exception as e:
            logging.warning("RUAccent failed (%s). Proceeding without accent.", e)
            final_text = sanitize_text(text)

    # Symbol count per policy (spaces/newlines and '+' excluded)
    symbol_count = count_symbols_for_sps(final_text)

    # Create Synth on CPU and reuse
    synth = create_synth(str(model_dir), device="cpu")

    out_dir = Path(args.output_folder)
    ensure_folder_exists(out_dir)

    # Determine number of speakers
    num_speakers = read_num_speakers(model_dir)
    if args.limit is not None:
        num_speakers = max(1, min(num_speakers, args.limit))

    # Synthesize for each speaker
    written: List[Path] = []
    for sid in range(num_speakers):
        idx = sid + 1
        out_path = out_dir / f"{args.file_prefix}{idx}.wav"
        synth.synth(
            text=final_text,
            oname=str(out_path),
            speaker_id=int(sid),
            speech_rate=float(args.speech_rate),
        )
        logging.info("Wrote %s (speaker %d)", out_path, sid)
        written.append(out_path)

    # Write manifest.csv
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "speaker_id", "filename", "text_length"])
        for i, sid in enumerate(range(num_speakers), start=1):
            w.writerow([i, sid, f"{args.file_prefix}{i}.wav", symbol_count])
    logging.info("Manifest saved to %s", manifest_path)

    # Optional SPS metrics
    if args.compute_sps:
        try:
            metrics = compute_symbols_per_second(
                outputs_folder=str(out_dir),
                text=final_text,
                expected_prefix=args.file_prefix,
            )
            agg = metrics.get("aggregate", {})
            logging.info(
                "SPS metrics written (mean=%s, min=%s, max=%s) -> %s",
                agg.get("mean_cps"),
                agg.get("min_cps"),
                agg.get("max_cps"),
                metrics.get("metrics_path"),
            )
        except Exception as e:
            logging.error("Failed to compute SPS metrics: %s", e)


if __name__ == "__main__":
    main()
