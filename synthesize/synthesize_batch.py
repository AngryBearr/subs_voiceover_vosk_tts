import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Union

from utils.subs_utils import ensure_folder_exists
from .synthesize import synthesize_text, SynthesisError


def _load_entries(json_path: str) -> List[dict]:
    with open(json_path, encoding="utf-8") as f:
        if json_path.lower().endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _normalize_text(value: Union[str, Iterable[str]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        # Join list-of-lines into a single string
        return " ".join([str(x) for x in value if str(x).strip()])
    except Exception:
        # Fallback to string coercion
        return str(value)


def synthesize_json_lines(
    json_path: str,
    *,
    model_path: str,
    device: str = "cpu",
    output_folder: str = "./tts_out",
    file_prefix: str = "tts_",
    voice: int = 0,
    default_speech_rate: float = 1.0,
    text_key: str = "text",
    output_sample_rate: int | None = None,
) -> None:
    """Synthesize audio for each entry in a JSON/JSONL file using synthesize.synthesize.

    Expectations:
    - Input is an array of objects (or JSONL) with at least a text field (default 'text').
    - The text field may be a string or a list of strings; lists are joined with a space.
    - Output files are named '<file_prefix><n>.wav' (1-based) in output_folder.
    """
    ensure_folder_exists(output_folder)

    entries = _load_entries(json_path)

    for idx, entry in enumerate(entries):
        raw = entry.get(text_key)
        text = _normalize_text(raw)
        if not text.strip():
            # Skip empty lines but keep index continuity
            print(f"Skipping empty text at entry {idx+1}")
            continue

        outname = os.path.join(output_folder, f"{file_prefix}{idx+1}.wav")

        try:
            synthesize_text(
                text=text,
                voice=voice,
                speech_rate=default_speech_rate,
                model_path=model_path,
                device=device,
                output_path=outname,
                filename_prefix=None,
                output_sample_rate=output_sample_rate,
            )
            print(f"Synthesized: {outname}")
        except (ValueError, FileNotFoundError, OSError, SynthesisError) as e:
            # Continue batch on individual failures
            print(f"Failed to synthesize entry {idx+1}: {e}")


if __name__ == "__main__":
    # Minimal manual test (edit paths before running)
    # This section is intentionally simple; prefer using synthesize_cli.py for real runs.
    import argparse

    parser = argparse.ArgumentParser(description="Batch synthesize via synthesize.synthesize")
    parser.add_argument("json_path", help="Path to JSON/JSONL file with entries")
    parser.add_argument("--model", required=True, help="Path to Vosk TTS model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output-folder", default="./tts_out")
    parser.add_argument("--file-prefix", default="tts_")
    parser.add_argument("--voice", type=int, default=0)
    parser.add_argument("--speech-rate", type=float, default=1.0)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--output-sample-rate", type=int, default=None)
    args = parser.parse_args()

    synthesize_json_lines(
        json_path=args.json_path,
        model_path=args.model,
        device=args.device,
        output_folder=args.output_folder,
        file_prefix=args.file_prefix,
        voice=args.voice,
        default_speech_rate=args.speech_rate,
        text_key=args.text_key,
        output_sample_rate=args.output_sample_rate,
    )
