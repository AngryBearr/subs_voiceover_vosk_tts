#!/usr/bin/env bash
# Start the blind listening page and open it (http://127.0.0.1:7862; PORT=... for another port); Ctrl+C stops it.
# Usage: bash scripts/run_listen.sh [pack or root folder] [extra ui.blind_listen options]
# Without a folder it takes the newest STT_listen_* folder on the media drive or in output/.
set -euo pipefail
cd "$(dirname "$0")/.."
port=${PORT:-7862}
url="http://127.0.0.1:$port"
[[ -x subs_env/bin/python ]] || { echo "Нет subs_env: сначала bash scripts/install.sh" >&2; exit 1; }

pack=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  pack=$1
  shift
else
  search_roots=("${LISTEN_ROOT:-/media/angrybear/new_large_storage/Survivor}" output)
  pack=$(for root in "${search_roots[@]}"; do
           [[ -d "$root" ]] && find "$root" -maxdepth 1 -mindepth 1 -type d -name 'STT_listen_*' -printf '%T@ %p\n' 2>/dev/null
         done | sort -nr | head -n1 | cut -d' ' -f2-)
  if [[ -z "$pack" ]]; then
    echo "Не нашёл ни одной папки STT_listen_* в: ${search_roots[*]}" >&2
    echo "Укажите папку набора явно: bash scripts/run_listen.sh /путь/к/папке" >&2
    exit 1
  fi
  echo "Открываю самый свежий набор: $pack"
fi
[[ -d "$pack" ]] || { echo "Папка не найдена: $pack" >&2; exit 1; }

open_browser() { xdg-open "$url" >/dev/null 2>&1 || echo "Откройте в браузере: $url"; }
if curl -fs -o /dev/null "$url/api/packs"; then
  echo "Страница уже запущена на порту $port (другой набор — PORT=7863 bash scripts/run_listen.sh ...)."
  open_browser
  exit 0
fi
( for _ in $(seq 60); do sleep 0.5; curl -fs -o /dev/null "$url/api/packs" && { open_browser; break; }; done ) &
exec subs_env/bin/python -m ui.blind_listen "$pack" --port "$port" "$@"
