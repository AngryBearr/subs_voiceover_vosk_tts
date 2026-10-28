"""Self-test of an installation: GPU, environments, models and keys, one line per check.

    subs_env/bin/python -m stt.selftest            # every model loaded once (downloads ~30 GB the first time)
    subs_env/bin/python -m stt.selftest --quick    # no downloads: GPU, packages, Vosk, keys, a Vosk line heard

The full run also downloads the gated models (pyannote community-1, Cohere Transcribe): their
terms must be accepted on huggingface.co under the account of ``hf auth login``. Keys are read
from ``.env`` and never printed. Exit code 1 when a required check failed; optional parts
(ESpeech, DiariZen) only warn.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt.run import REPO_ROOT, dotenv, load_config

PHRASE = "Сегодня на острове двенадцать участников, и всех ждёт испытание."
GATED = {"pyannote/speaker-diarization-community-1": "https://huggingface.co/pyannote/speaker-diarization-community-1",
         "CohereLabs/cohere-transcribe-03-2026": "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026"}


@dataclass
class Check:
    name: str
    run: Callable[[dict[str, Any]], str]
    required: bool = True
    full_only: bool = False


def _gpu(state: dict[str, Any]) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch не видит видеокарту (драйвер NVIDIA, CUDA 12.8+)")
    props = torch.cuda.get_device_properties(0)
    memory = props.total_memory / 2**30
    return f"{props.name}, {memory:.0f} ГБ" + (" (меньше 12 ГБ: тяжёлые шаги могут не поместиться)" if memory < 12 else "")


def _ffmpeg(state: dict[str, Any]) -> str:
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    out = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, check=True).stdout
    return out.split("\n", 1)[0]


def _vosk(state: dict[str, Any]) -> str:
    model = REPO_ROOT / "models/vosk-model-tts-ru-0.10-multi"
    if not (model / "model.onnx").exists():
        raise RuntimeError(f"нет модели Vosk TTS в {model} (scripts/install.sh скачивает её)")
    wav = Path(state["tmp"]) / "vosk.wav"
    subprocess.run([sys.executable, "-m", "synthesize.synthesize_cli", "--text", PHRASE, "--model", str(model),
                    "--voice", "56", "--output", str(wav)], cwd=REPO_ROOT, check=True, capture_output=True)
    state["vosk_wav"] = wav
    return "фраза озвучена"


def _text(state: dict[str, Any]) -> str:
    from voiceover.engine import VoskEngine

    engine = VoskEngine({"model_path": str(REPO_ROOT / "models/vosk-model-tts-ru-0.10-multi")})
    row = engine.prepare([{"cue_id": "1", "text": "В 2026 году на острове было 18 человек."}])[0]
    if row.normalization != "ok" or "+" not in row.tts_text:
        raise RuntimeError(f"подготовка текста: {row.normalization} {row.error or ''}")
    state["marked"] = row.tts_text
    return row.tts_text


def _heard(state: dict[str, Any], wav: Path) -> str:
    import soundfile as sf

    from voiceover.speech_check import Recognizer, check_audio

    if "recognizer" not in state:
        state["recognizer"] = Recognizer("cuda")
    audio, rate = sf.read(str(wav), dtype="float32")
    check = check_audio(state["recognizer"], PHRASE, audio, rate)
    if check["cer"] > 0.2:
        raise RuntimeError(f"распознано «{check['heard']}» (ошибок {check['cer']:.0%})")
    return f"распознано «{check['heard']}», ошибок {check['cer']:.0%}"


def _gigaam(state: dict[str, Any]) -> str:
    if "vosk_wav" not in state:
        raise RuntimeError("нет фразы Vosk (предыдущая проверка)")
    return _heard(state, state["vosk_wav"])


def _teratts(state: dict[str, Any]) -> str:
    from voiceover.engine_teratts import TeraTTSEngine, load_model, set_temperature

    engine = TeraTTSEngine({"voice": "ru_m5"})
    model = load_model(engine.path, engine.device, engine.diffusion_model)
    set_temperature(model, engine.temperature)
    wav = Path(state["tmp"]) / "teratts.wav"
    engine._take(model, PHRASE, "ru_m5", wav, 1.0, 1234)
    return _heard(state, wav)


def _stress(state: dict[str, Any]) -> str:
    import soundfile as sf

    from voiceover.stress_listen import StressListener

    audio, rate = sf.read(str(state["vosk_wav"]), dtype="float32")
    heard = StressListener("cuda")(audio, rate, "Сег+одня на +острове двен+адцать уч+астников", {"острове", "сегодня"})
    return ", ".join(f"{w['word']}: {'верно' if not w['wrong'] else 'не так'}" for w in heard) or "нет слов"


def _download(repo: str, **kwargs: Any) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        snapshot_download(repo, **kwargs)
    except GatedRepoError as exc:
        raise RuntimeError(f"нет доступа: примите условия на {GATED.get(repo, repo)} и выполните "
                           "`subs_env/bin/hf auth login`") from exc
    return "скачано"


def _whisper(state: dict[str, Any]) -> str:
    from faster_whisper import WhisperModel

    WhisperModel("large-v3", device="cpu", compute_type="int8")
    return "загружается"


def _torchaudio_model(name: str) -> Callable[[dict[str, Any]], str]:
    def check(state: dict[str, Any]) -> str:
        import torchaudio

        getattr(torchaudio.pipelines, name).get_model()
        return "загружается"
    return check


def _env(name: str, code: str) -> Callable[[dict[str, Any]], str]:
    def check(state: dict[str, Any]) -> str:
        python = REPO_ROOT / name / "bin" / "python"
        if not python.exists():
            raise RuntimeError(f"нет {name} (bash scripts/setup_envs.sh {name.split('_')[0]})")
        done = subprocess.run([str(python), "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "ошибка")
        return done.stdout.strip()
    return check


def _openrouter(state: dict[str, Any]) -> str:
    key = dotenv().get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("нет OPENROUTER_API_KEY в .env (перевод, имена и ударения идут через OpenRouter)")
    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8")).get("data", {})
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter не принял ключ (HTTP {exc.code})") from None
    left = data.get("limit_remaining")
    return "ключ принят" + (f", осталось ${left:.2f}" if isinstance(left, (int, float)) else "")


def _subscription(state: dict[str, Any]) -> str:
    from stt.llm import subscription_error
    from stt.run import subscription_models

    models = subscription_models(load_config(REPO_ROOT / "stt_config.json"))
    if not models:
        return "не настроена (запросы идут через OpenRouter)"
    error = subscription_error(models[0])
    if error:
        raise RuntimeError(f"{models[0]}: {error}")
    return f"{models[0]} отвечает"


CHECKS = [
    Check("Видеокарта", _gpu),
    Check("ffmpeg", _ffmpeg),
    Check("Vosk TTS", _vosk),
    Check("Подготовка текста (числа, ударения)", _text),
    Check("Проверка речи (GigaAM)", _gigaam),
    Check("Ключ OpenRouter", _openrouter),
    Check("Подписка (opencode)", _subscription, required=False),
    Check("TeraTTS", _teratts, full_only=True),
    Check("Ударения на слух (TeraAnaly)", _stress, full_only=True),
    Check("Whisper large-v3", _whisper, full_only=True),
    Check("Cohere Transcribe (нужно принять условия)",
          lambda s: _download("CohereLabs/cohere-transcribe-03-2026"), full_only=True),
    Check("pyannote community-1 (нужно принять условия)",
          lambda s: _download("pyannote/speaker-diarization-community-1"), full_only=True),
    Check("Голоса: пол и возраст", lambda s: _download("audeering/wav2vec2-large-robust-24-ft-age-gender"),
          full_only=True),
    Check("Голоса: уточнение спикеров", lambda s: _download("Wespeaker/wespeaker-voxceleb-resnet293-LM"),
          full_only=True),
    Check("Выравнивание слов (wav2vec2)", _torchaudio_model("WAV2VEC2_ASR_LARGE_LV60K_960H"), full_only=True),
    Check("Голоса отдельно от музыки (Demucs)", _torchaudio_model("HDEMUCS_HIGH_MUSDB_PLUS"), full_only=True),
    Check("ESpeech (f5_env)", _env("f5_env", "import f5_tts; from huggingface_hub import hf_hub_download; "
                                   "hf_hub_download('ESpeech/ESpeech-TTS-1_RL-V2', 'espeech_tts_rlv2.pt'); "
                                   "print('скачано')"), required=False, full_only=True),
    Check("DiariZen (diarizen_env)", _env("diarizen_env", "import diarizen; from huggingface_hub import "
                                          "snapshot_download; snapshot_download('BUT-FIT/diarizen-wavlm-large-s80-md-v2'); "
                                          "print('скачано')"), required=False, full_only=True),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the installation: GPU, models, keys")
    parser.add_argument("--quick", action="store_true", help="no downloads: GPU, packages, Vosk, keys")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        state: dict[str, Any] = {"tmp": tmp}
        for check in CHECKS:
            if args.quick and check.full_only:
                continue
            t0 = time.time()
            try:
                detail = check.run(state)
                mark = "ok"
            except Exception as exc:  # noqa: BLE001 - every check reports, none stops the others
                detail = str(exc) or type(exc).__name__
                mark = "ОШИБКА" if check.required else "нет"
                failed += check.required
            print(f"[{mark:6s}] {check.name}: {detail} ({time.time() - t0:.0f} с)", flush=True)
    print("\nВсё готово." if not failed else f"\nНе прошло проверок: {failed}. Исправьте и запустите снова.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
