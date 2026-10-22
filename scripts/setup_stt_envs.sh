#!/usr/bin/env bash
# Create the local GPU STT virtual environments (Linux, NVIDIA driver with CUDA 12+).
# Usage (repo root): bash scripts/setup_stt_envs.sh [main] [qwen]   (default: both)
#   main -> stt_env:      Whisper, Cohere, pyannote, gender/age, orchestration
#   qwen -> stt_qwen_env: Qwen3-ASR + Qwen3-ForcedAligner (qwen-asr pins transformers 4.57)
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_HTTP_TIMEOUT=300
targets=("${@:-main qwen}")
targets=(${targets[*]})

has() { [[ " ${targets[*]} " == *" $1 "* ]]; }

if has main; then
  uv venv stt_env --python 3.12
  VIRTUAL_ENV=$PWD/stt_env uv pip install --index-strategy unsafe-best-match -r requirements-stt-gpu.txt
fi

if has qwen; then
  uv venv stt_qwen_env --python 3.12
  VIRTUAL_ENV=$PWD/stt_qwen_env uv pip install --index-strategy unsafe-best-match -r requirements-stt-qwen.txt
fi

stt_env/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
