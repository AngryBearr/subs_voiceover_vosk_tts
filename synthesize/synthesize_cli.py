#!/usr/bin/env python3
"""
CLI entry point for Vosk TTS synthesis (single and batch).

Single-line mode (default): provide --text. Batch mode: provide --json.
"""
import argparse
import logging
import sys

from .synthesize import synthesize_text, SynthesisError  # type: ignore
from .synthesize_batch import synthesize_json_lines  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Vosk TTS synthesizer (single or batch)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", "-i", help="Text to synthesize (single-line mode)")
    mode.add_argument("--json", help="Path to JSON/JSONL with entries (batch mode)")

    parser.add_argument("--model", "-m", required=True, help="Path to Vosk TTS model directory")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--voice", "-s", type=int, default=0, help="Speaker id (voice)")
    parser.add_argument("--speech-rate", "-r", type=float, default=1.0, help="Speech rate (>0)")
    parser.add_argument("--output-sample-rate", type=int, default=None, help="Optional output sample rate (Hz)")

    # Single-line specific
    parser.add_argument("--output", "-o", default=None, help="Output .wav path (single mode)")
    parser.add_argument("--prefix", "-p", default=None, help="Filename prefix when --output omitted (single mode)")

    # Batch specific
    parser.add_argument("--output-folder", default="./tts_out", help="Folder for batch outputs")
    parser.add_argument("--file-prefix", default="tts_", help="Prefix for batch output files")
    parser.add_argument("--text-key", default="text", help="Key for text field in JSON entries")

    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        if args.json:
            # Batch mode
            synthesize_json_lines(
                json_path=args.json,
                model_path=args.model,
                device=args.device,
                output_folder=args.output_folder,
                file_prefix=args.file_prefix,
                voice=args.voice,
                default_speech_rate=args.speech_rate,
                text_key=args.text_key,
                output_sample_rate=args.output_sample_rate,
            )
        else:
            # Single-line mode
            out = synthesize_text(
                text=args.text,
                voice=args.voice,
                speech_rate=args.speech_rate,
                model_path=args.model,
                device=args.device,
                output_path=args.output,
                filename_prefix=args.prefix,
                output_sample_rate=args.output_sample_rate,
            )
            print(str(out))
    except (ValueError, FileNotFoundError, OSError, SynthesisError) as e:
        logging.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
