#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_PATH="${FARMOS_LEAF_VENV_PATH:-/home/cvl/farmos_mcp/farmos-venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

export FARMOS_HOST="${FARMOS_HOST:-https://try.farmos.net}"
export FARMOS_USER="${FARMOS_USER:-mark}"
export FARMOS_PASSWORD="${FARMOS_PASSWORD:-}"
export FARMOS_CLIENT_ID="${FARMOS_CLIENT_ID:-farm}"
export FARMOS_CLIENT_SECRET="${FARMOS_CLIENT_SECRET:-}"
export LEAF_YOLO_DIR="${LEAF_YOLO_DIR:-/home/cvl/farmos_yolo_model}"
export LEAF_YOLO_MODEL_PATH="${LEAF_YOLO_MODEL_PATH:-/home/cvl/farmos_yolo_model/weights/best.pt}"
export LEAF_YOLO_INFER_SCRIPT="${LEAF_YOLO_INFER_SCRIPT:-/home/cvl/farmos_yolo_model/infer_image.py}"
export LEAF_YOLO_LIVE_PREVIEW_SCRIPT="${LEAF_YOLO_LIVE_PREVIEW_SCRIPT:-/home/cvl/farmos_yolo_model/live_preview.py}"
export LEAF_YOLO_CONFIDENCE="${LEAF_YOLO_CONFIDENCE:-0.25}"
export LEAF_YOLO_LIVE_PREVIEW_ON_ASK="${LEAF_YOLO_LIVE_PREVIEW_ON_ASK:-1}"
export LEAF_CAPTURE_DIR="${LEAF_CAPTURE_DIR:-/home/cvl/farmos_leaf_mcp/captures}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/cvl/.Xauthority}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

python farmos_leaf_mcp.py
