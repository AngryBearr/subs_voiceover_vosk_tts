"""Small script to test `synthesize.synthesize.synthesize_text` with a random speaker.

Usage:
    python scripts/synthesize_test.py         # uses default text and model
    python scripts/synthesize_test.py -t "Привет, мир" --device cpu

The script loads `models/vosk-model-tts-ru-0.10-multi/config.json`, selects a random
speaker from `speaker_id_map` (or falls back to `num_speakers`) and writes a .wav.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports like `synthesize.*` work
# when this script is executed directly (not as a package). This mirrors the
# behavior of running from the repository root or setting PYTHONPATH=.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from synthesize.synthesize import synthesize_text


def pick_random_speaker_from_config(config: dict) -> int:
    """Return a speaker id chosen randomly from config.

    Prefers explicit `speaker_id_map` (dictionary of name->id). If not present,
    falls back to `num_speakers` to choose an id in range(0, num_speakers).
    """
    speaker_map = config.get("speaker_id_map")
    if isinstance(speaker_map, dict) and speaker_map:
        names = list(speaker_map.keys())
        chosen_name = random.choice(names)
        return int(speaker_map[chosen_name])

    num = config.get("num_speakers")
    if isinstance(num, int) and num > 0:
        return random.randrange(num)

    # Last resort: 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test Vosk TTS synthesize_text with a random speaker")
    parser.add_argument("-t", "--text", default="Самое интересное что синтез речи может воспроизвести что угодно.", help="Text to synthesize")
    parser.add_argument("--model", default="models/vosk-model-tts-ru-0.10-multi", help="Path to model directory")
    parser.add_argument("--device", default="cuda", help="Device advisory: cpu or cuda")
    parser.add_argument("--filename-prefix", default="test_synth", help="Filename prefix for generated wav when not passing --output")
    parser.add_argument("--output", default=None, help="Optional explicit output wav path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    model_path = Path(args.model)
    config_path = model_path / "config.json"
    if not config_path.exists():
        logging.error("Model config not found at %s", config_path)
        return 2

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:
        logging.exception("Failed to read model config: %s", e)
        return 3

    speaker_id = pick_random_speaker_from_config(cfg)
    logging.info("Selected speaker id: %s", speaker_id)

    try:
        out_path = synthesize_text(
            text=args.text,
            voice=speaker_id,
            model_path=str(model_path),
            device=args.device,
            output_path=args.output,
            filename_prefix=args.filename_prefix,
            output_sample_rate=48000,
        )
        print(f"Synthesis complete: {out_path}")
        return 0
    except Exception as e:
        logging.exception("Synthesis failed: %s", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
