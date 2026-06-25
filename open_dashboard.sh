#!/usr/bin/env bash
set -euo pipefail

URL="${FARMOS_UI_URL:-http://127.0.0.1:8765}"

for _ in $(seq 1 60); do
  if curl --silent --fail --max-time 1 "$URL/api/status" >/dev/null; then
    exec chromium \
      --app="$URL" \
      --start-maximized \
      --no-first-run \
      --disable-session-crashed-bubble
  fi
  sleep 1
done

exit 1
