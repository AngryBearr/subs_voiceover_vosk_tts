#!/usr/bin/env bash
# Create the virtual environments (Linux, NVIDIA driver with CUDA 12.8+), from the repo root:
#   bash scripts/setup_envs.sh          # subs_env (everything) + stt_qwen_env
#   bash scripts/setup_envs.sh main     # only subs_env
#   bash scripts/setup_envs.sh qwen     # only stt_qwen_env (Qwen3-ASR + aligner, accurate STT mode)
# subs_env runs every script of the repo; stt_qwen_env is needed only because qwen-asr pins
# transformers 4.57 (the STT pipeline starts those stages in it by itself).
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_HTTP_TIMEOUT=300
targets=("${@:-main qwen}")
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
