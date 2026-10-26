#!/usr/bin/env bash
# Create the virtual environments (Linux, NVIDIA driver with CUDA 12.8+), from the repo root:
#   bash scripts/setup_envs.sh          # subs_env: runs every script of the repo
#   bash scripts/setup_envs.sh qwen     # optional stt_qwen_env: Qwen3-ASR / Qwen3-ForcedAligner, only for
#                                       # benchmarks (`--fuse ...,qwen3-asr-1.7b`, `--aligner qwen`);
#                                       # qwen-asr pins transformers 4.57
#   bash scripts/setup_envs.sh f5       # optional f5_env: ESpeech F5-TTS voice-over in the speakers' own voices
#                                       # (engine "f5"); reuses torch of subs_env, so create subs_env first
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_HTTP_TIMEOUT=300
targets=("${@:-main}")
targets=(${targets[*]})

has() { [[ " ${targets[*]} " == *" $1 "* ]]; }

if has main; then
  uv venv subs_env --python 3.12
  VIRTUAL_ENV=$PWD/subs_env uv pip install --index-strategy unsafe-best-match \
    -r requirements.txt --override overrides-onnxruntime.txt
  # ffmpeg for tempo stretching, muxing and audio conversion: the imageio-ffmpeg binary, on PATH in subs_env
  ln -sf "$(subs_env/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" subs_env/bin/ffmpeg
  subs_env/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
fi

if has qwen; then
  uv venv stt_qwen_env --python 3.12
  VIRTUAL_ENV=$PWD/stt_qwen_env uv pip install --index-strategy unsafe-best-match -r requirements-stt-qwen.txt
fi

if has f5; then
  uv venv f5_env --python 3.12
  # torch, torchaudio and the rest of subs_env are visible to f5_env; its own packages win
  echo "$PWD/subs_env/lib/python3.12/site-packages" > f5_env/lib/python3.12/site-packages/subs_env.pth
  VIRTUAL_ENV=$PWD/f5_env uv pip install f5-tts==1.1.22 --override overrides-f5.txt
  f5_env/bin/python -c "import f5_tts, torch; print('f5-tts ok, CUDA:', torch.cuda.is_available())"
fi
