#!/usr/bin/env bash
# One-command install (Linux, NVIDIA GPU): checks the computer, creates the Python environments,
# downloads the Vosk voice model, logs in to Hugging Face, asks for the OpenRouter key (written to
# .env, never shown), runs the self-test (downloads the other models, ~30 GB the first time) and
# can add a menu shortcut. Run it again after `git pull` to update: finished steps are quick.
#   bash scripts/install.sh            # everything: also ESpeech (heroes' own voices) and DiariZen
#   bash scripts/install.sh --basic    # Vosk and TeraTTS voices, pyannote diarization only
# Then: bash scripts/run_ui.sh (opens http://127.0.0.1:7861 in the browser).
set -euo pipefail
cd "$(dirname "$0")/.."

basic=0
for arg in "$@"; do
  case $arg in
    --basic) basic=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Неизвестный параметр: $arg (есть --basic)" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf 'Внимание: %s\n' "$*"; }
die() { printf '\nОшибка: %s\n' "$*" >&2; exit 1; }
ask() { local answer; read -r -p "$1 [y/N] " answer; [[ $answer =~ ^[YyДд] ]]; }

step "1/6 Проверка компьютера"
[[ $(uname -s) == Linux ]] || die "нужен Linux"
command -v nvidia-smi >/dev/null || die "нет драйвера NVIDIA (команда nvidia-smi не найдена)"
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
[[ -n $cuda ]] || die "nvidia-smi не показал версию CUDA"
[[ $(printf '%s\n' 12.8 "$cuda" | sort -V | head -1) == 12.8 ]] \
  || die "драйвер NVIDIA даёт CUDA $cuda, нужна 12.8 или новее: обновите драйвер"
gpu_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "Видеокарта: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(( (gpu_mb + 512) / 1024 )) ГБ, CUDA $cuda"
(( gpu_mb >= 12000 )) || warn "меньше 12 ГБ видеопамяти: тяжёлые шаги (распознавание, ESpeech) могут не поместиться"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
cache_gb=$(mkdir -p "${HF_HOME:-$HOME/.cache/huggingface}" && df -BG --output=avail "${HF_HOME:-$HOME/.cache/huggingface}" | tail -1 | tr -dc '0-9')
echo "Свободно: $free_gb ГБ здесь, $cache_gb ГБ под моделями (${HF_HOME:-$HOME/.cache/huggingface})"
(( free_gb >= 40 )) || warn "в папке проекта меньше 40 ГБ: окружения ~15 ГБ, рабочие файлы серии ~2 ГБ"
(( cache_gb >= 40 )) || warn "под моделями меньше 40 ГБ (при первой установке они занимают ~30 ГБ)"
command -v git >/dev/null || die "нет git (sudo apt install git)"
command -v curl >/dev/null || die "нет curl (sudo apt install curl)"
if ! command -v uv >/dev/null; then
  [[ -x $HOME/.local/bin/uv ]] || { ask "Нет uv (установщик пакетов Python). Установить с astral.sh?" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh; } || die "без uv не установить: https://docs.astral.sh/uv/"
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

step "2/6 Окружения Python (первый раз 10-30 минут)"
targets=(main)
(( basic )) || targets+=(f5 diarizen)
bash scripts/setup_envs.sh "${targets[@]}"

step "3/6 Модель голосов Vosk"
vosk=models/vosk-model-tts-ru-0.10-multi
if [[ -f $vosk/model.onnx ]]; then
  echo "уже есть: $vosk"
else
  mkdir -p models
  tmp=$(mktemp -d)
  curl -fL --retry 3 -o "$tmp/vosk.zip" "https://alphacephei.com/vosk/models/$(basename "$vosk").zip"
  subs_env/bin/python -m zipfile -e "$tmp/vosk.zip" models/
  rm -rf "$tmp"
  [[ -f $vosk/model.onnx ]] || die "в архиве Vosk нет $vosk/model.onnx"
fi

step "4/6 Hugging Face (откуда скачиваются модели)"
hf_token() { subs_env/bin/python -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)"; }
if hf_token; then
  echo "вход уже выполнен"
else
  echo "Нужна бесплатная учётная запись huggingface.co и токен с правом чтения: https://huggingface.co/settings/tokens"
fi
echo "Две модели выдаются после согласия с условиями (кнопка на странице модели, под той же учётной записью):"
echo "  https://huggingface.co/pyannote/speaker-diarization-community-1"
echo "  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026"
hf_token || subs_env/bin/hf auth login

step "5/6 Ключ OpenRouter (перевод, имена, ударения)"
if grep -qE '^\s*(export\s+)?OPENROUTER_API_KEY\s*=\s*\S' .env 2>/dev/null; then
  echo "ключ уже есть в .env"
else
  echo "Языковые модели работают через OpenRouter: оплата за запросы со счёта на openrouter.ai."
  echo "Ключ: https://openrouter.ai/keys (вводимые символы не показываются; Enter без ключа: пропустить)"
  read -r -s -p "Ключ: " key; echo
  if [[ -n $key ]]; then
    touch .env && chmod 600 .env
    [[ ! -s .env || -z $(tail -c1 .env) ]] || echo >> .env  # the last line may lack its newline
    printf 'OPENROUTER_API_KEY=%s\n' "$key" >> .env
    echo "записан в .env"
  else
    warn "пропущено: без ключа озвучка остановится на переводе; добавьте OPENROUTER_API_KEY=... в .env"
  fi
  unset key
fi

step "6/6 Самопроверка (первый раз скачивает модели, ~30 ГБ)"
if ! subs_env/bin/python -m stt.selftest; then
  warn "не все проверки прошли: исправьте отмеченное и запустите scripts/install.sh снова"
fi

desktop=$HOME/.local/share/applications/survivor-voiceover.desktop
if [[ ! -f $desktop ]] && ask "Добавить ярлык «Озвучка сериалов» в меню приложений?"; then
  mkdir -p "$(dirname "$desktop")"
  cat > "$desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Озвучка сериалов
Comment=Английское видео -> русская закадровая озвучка
Exec=bash "$PWD/scripts/run_ui.sh"
Terminal=true
Categories=AudioVideo;
EOF
  echo "ярлык: $desktop"
fi
printf '\nГотово. Запуск: bash scripts/run_ui.sh (откроется http://127.0.0.1:7861)\n'
