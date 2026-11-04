"""A short sample phrase in a chosen voice, for the "Настройки" tab ("▶ Послушать").

    subs_env/bin/python -m ui.voice_preview vosk 49 out.wav [--device cpu]
    subs_env/bin/python -m ui.voice_preview teratts ru_f1+ru_f2 out.wav [--device cpu|cuda]

The UI calls :func:`preview`, which runs this module in a subprocess, so the UI process never holds a TTS model.
Samples are kept in ``output/voice_previews/<engine>/<voice>.<phrase hash>.wav``: the second listen is instant.
Vosk goes through ``synthesize.synthesize_cli``; TeraTTS through ``voiceover.engine_teratts``. Both read ``+``
as the stress mark, so the phrase is stressed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = REPO_ROOT / "output" / "voice_previews"
VOSK_MODEL = REPO_ROOT / "models" / "vosk-model-tts-ru-0.10-multi"
PHRASE = "Прив+ет! Сег+одня на +острове двен+адцать уч+астников, и всех ждёт испыт+ание."
ENGINES = ("vosk", "teratts")
TIMEOUT_S = 300  # Vosk loads its dictionary for ~45 s on every start


def phrase_hash(phrase: str = PHRASE) -> str:
    return hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:8]


def preview_path(engine: str, voice: str | int, root: Path = PREVIEW_DIR, phrase: str = PHRASE) -> Path:
    """Where the sample of ``voice`` is kept (voice mixes like ``ru_f1+ru_f2`` stay readable in the name)."""
    name = str(voice).replace("/", "_").replace(" ", "")
    return root / engine / f"{name}.{phrase_hash(phrase)}.wav"


def build_command(engine: str, voice: str | int, out: Path, device: str = "cpu") -> list[str]:
    return [sys.executable, "-m", "ui.voice_preview", engine, str(voice), str(out), "--device", device]


def preview(engine: str, voice: str | int, device: str = "cpu", root: Path = PREVIEW_DIR) -> tuple[Path | None, str]:
    """(wav, message): the sample of the voice (made in a subprocess if it is not kept yet), or (None, why not)."""
    if engine not in ENGINES:
        return None, f"Неизвестный движок: {engine}."
    if str(voice).strip() == "":
        return None, "Сначала выберите голос."
    target = preview_path(engine, voice, root)
    if engine == "vosk" and not target.exists() and VOSK_PREPARING.is_set():
        started = time.time()  # the samples of all voices are being made right now: wait for this one
        while VOSK_PREPARING.is_set() and not target.exists() and time.time() - started < PREPARE_WAIT_S:
            time.sleep(1)
    if target.exists():
        return target, "Готово."
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.part.wav")  # a half-written file is never taken for a sample
    try:
        done = subprocess.run(build_command(engine, voice, tmp, device), cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return None, "Образец не получилось сделать вовремя: попробуйте ещё раз."
    if done.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        log.warning("voice preview %s %s failed: %s", engine, voice, (done.stderr or done.stdout)[-1500:])
        return None, f"Не получилось озвучить образец: {_reason(done.stderr or done.stdout)}"
    tmp.replace(target)
    return target, "Готово."


VOSK_PREPARING = threading.Event()  # set while prepare_vosk_voices runs: a click waits for it, not a second load
PREPARE_WAIT_S = 240


def prepare_vosk_voices(root: Path = PREVIEW_DIR, timeout_s: int = 900) -> int:
    """Make the samples of every Vosk voice not kept yet, with one model load (~45 s + ~1 s per voice, CPU, low
    priority); returns how many were made. Run in the background when the UI starts, so a click plays at once."""
    if not (VOSK_MODEL / "model.onnx").exists():
        return 0
    count = len(json.loads((VOSK_MODEL / "config.json").read_text(encoding="utf-8")).get("speaker_id_map") or {})
    missing = [v for v in range(count) if not preview_path("vosk", v, root).exists()]
    if not missing:
        return 0
    VOSK_PREPARING.set()
    try:
        return _prepare(missing, root, timeout_s)
    finally:
        VOSK_PREPARING.clear()


def _prepare(missing: list[int], root: Path, timeout_s: int) -> int:
    with tempfile.TemporaryDirectory(dir=root.parent if root.parent.exists() else None) as tmp:
        done = subprocess.run([sys.executable, "-m", "synthesize.synthesize_cli", "--text", PHRASE, "--model",
                               str(VOSK_MODEL), "--voices", ",".join(map(str, missing)), "--output-folder", tmp,
                               "--file-prefix", "v"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=timeout_s, preexec_fn=lambda: os.nice(10))
        if done.returncode != 0:
            log.warning("Vosk voice samples failed: %s", (done.stderr or done.stdout)[-1500:])
        made = 0
        for voice in missing:
            wav = Path(tmp) / f"v{voice}.wav"
            if wav.exists() and wav.stat().st_size > 0:
                target = preview_path("vosk", voice, root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(wav), target)
                made += 1
    return made


def _reason(output: str) -> str:
    """The last line of an error output that says something, shortened (no traceback for the person)."""
    lines = [line.strip() for line in output.splitlines()
             if line.strip() and not line.lstrip().startswith(("File ", "Traceback", "^"))]
    return (lines[-1] if lines else "причина не названа (подробности в логе программы)")[:200]


def _vosk(voice: str, out: Path, device: str) -> None:
    if not (VOSK_MODEL / "model.onnx").exists():
        raise SystemExit(f"нет модели Vosk TTS в {VOSK_MODEL}")
    done = subprocess.run([sys.executable, "-m", "synthesize.synthesize_cli", "--text", PHRASE, "--model",
                           str(VOSK_MODEL), "--voice", str(int(voice)), "--device", device, "--output", str(out)],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(_reason(done.stderr or done.stdout))


def _teratts(voice: str, out: Path, device: str) -> None:
    from voiceover.engine_teratts import TeraTTSEngine, load_model, set_temperature

    engine = TeraTTSEngine({"voice": voice, "device": device})
    model = load_model(engine.path, device, engine.diffusion_model)
    set_temperature(model, engine.temperature)
    engine._take(model, PHRASE, voice, out, 1.0, 1234)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample phrase in a Vosk or TeraTTS voice")
    parser.add_argument("engine", choices=ENGINES)
    parser.add_argument("voice")
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (_vosk if args.engine == "vosk" else _teratts)(args.voice, args.out, args.device)


if __name__ == "__main__":
    main()
