#!/usr/bin/env bash
set -euo pipefail

URL="${FARMOS_UI_URL:-http://127.0.0.1:8765}"
PROFILE_DIR="${FARMOS_UI_PROFILE_DIR:-$HOME/.config/farmos-field-collector-browser}"

for _ in $(seq 1 60); do
  if curl --silent --fail --max-time 1 "$URL/api/status" >/dev/null; then
    exec chromium \
      --app="$URL" \
      --user-data-dir="$PROFILE_DIR" \
      --start-maximized \
      --no-first-run \
      --disable-session-crashed-bubble
  fi
  sleep 1
done

exit 1
