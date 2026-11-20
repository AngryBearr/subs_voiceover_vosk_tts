#!/usr/bin/env bash
# Link the one shared home of the environments, models, outputs and private settings into a checkout.
#
# Everything git does not keep (virtual environments, models/, output/, .env, the private settings) lives once, in
# $SUBS_SHARED (default ~/subs_voiceover_shared). Every checkout or worktree only holds symlinks to it, so a new
# worktree gets the full set at once and no environment is silently missing (e.g. lava_env for the voice widening).
#
#   bash scripts/link_shared.sh            # the current checkout (run from its root)
#   bash scripts/link_shared.sh <checkout> # another checkout
#
# Safe to run again: existing symlinks are replaced; a real file or folder of the same name is left alone (warning).
# The names are added to the checkout's .git/info/exclude so they never show up in `git status`.
set -euo pipefail

SHARED="${SUBS_SHARED:-$HOME/subs_voiceover_shared}"
TARGET="$(cd "${1:-.}" && pwd)"
# keep in sync with stt.preflight.SHARED_ITEMS (tests/test_preflight.py compares them)
ITEMS=(subs_env f5_env diarizen_env lava_env xvoice_env stt_qwen_env models output .env stt_config.local.json ui.local.json)

if [ ! -d "$SHARED" ]; then
  echo "link_shared: shared folder $SHARED not found (set SUBS_SHARED)" >&2
  exit 1
fi
exclude="$(git -C "$TARGET" rev-parse --git-path info/exclude)"
case "$exclude" in /*) ;; *) exclude="$TARGET/$exclude" ;; esac
mkdir -p "$(dirname "$exclude")"
touch "$exclude"

linked=0
for name in "${ITEMS[@]}"; do
  src="$SHARED/$name"
  dst="$TARGET/$name"
  [ -e "$src" ] || continue  # optional items (stt_qwen_env) may not exist
  if [ -L "$dst" ]; then
    rm "$dst"
  elif [ -e "$dst" ]; then
    echo "link_shared: $dst is a real file or folder, left as it is (move it into $SHARED to share it)" >&2
    continue
  fi
  ln -s "$src" "$dst"
  linked=$((linked + 1))
  grep -qxF "$name" "$exclude" || printf '%s\n' "$name" >> "$exclude"
done
echo "link_shared: $linked links in $TARGET -> $SHARED"
