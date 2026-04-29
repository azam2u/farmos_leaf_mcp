#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local env file if present.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

VENV_PATH="${FARMOS_LEAF_VENV_PATH:-/home/cvl/farmos_mcp/farmos-venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

# Conservative defaults only (real values should come from .env or environment).
export FARMOS_HOST="${FARMOS_HOST:-https://try.farmos.net}"
export FARMOS_CLIENT_ID="${FARMOS_CLIENT_ID:-farm}"
export LEAF_YOLO_DIR="${LEAF_YOLO_DIR:-/home/cvl/farmos_yolo_model}"
export LEAF_YOLO_MODEL_PATH="${LEAF_YOLO_MODEL_PATH:-/home/cvl/farmos_yolo_model/weights/best.pt}"
export LEAF_YOLO_INFER_SCRIPT="${LEAF_YOLO_INFER_SCRIPT:-/home/cvl/farmos_yolo_model/infer_image.py}"
export LEAF_YOLO_LIVE_PREVIEW_SCRIPT="${LEAF_YOLO_LIVE_PREVIEW_SCRIPT:-/home/cvl/farmos_yolo_model/live_preview.py}"
export LEAF_YOLO_CONFIDENCE="${LEAF_YOLO_CONFIDENCE:-0.25}"
export LEAF_YOLO_LIVE_PREVIEW_ON_ASK="${LEAF_YOLO_LIVE_PREVIEW_ON_ASK:-1}"
export LEAF_CAPTURE_DIR="${LEAF_CAPTURE_DIR:-$PWD/captures}"
export MAPBOX_INPUT_DIR="${MAPBOX_INPUT_DIR:-$PWD/mapbox_inputs}"
export SEGMENTED_DIR="${SEGMENTED_DIR:-$PWD/received_segmented_mapbox_image}"
export LEAF_EXPORT_DIR="${LEAF_EXPORT_DIR:-$PWD/plant_exports}"

# GUI/camera runtime defaults.
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/cvl/.Xauthority}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

python3 farmos_leaf_mcp.py
