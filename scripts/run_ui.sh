#!/usr/bin/env bash
# Start the browser UI and open it (http://127.0.0.1:7861; PORT=... for another port); Ctrl+C stops it.
# When the UI already runs, only opens the browser.
set -euo pipefail
cd "$(dirname "$0")/.."
port=${PORT:-7861}
url="http://127.0.0.1:$port"
[[ -x subs_env/bin/python ]] || { echo "Нет subs_env: сначала bash scripts/install.sh" >&2; exit 1; }
open_browser() { xdg-open "$url" >/dev/null 2>&1 || echo "Откройте в браузере: $url"; }
if curl -fs -o /dev/null "$url"; then
  open_browser
  exit 0
fi
( for _ in $(seq 120); do sleep 1; curl -fs -o /dev/null "$url" && { open_browser; break; }; done ) &
exec subs_env/bin/python -m ui.dub_app --port "$port" "$@"
