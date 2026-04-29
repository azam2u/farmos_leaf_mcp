# farmOS Leaf MCP

MCP server for plant-facing farmOS workflows:
- camera capture + YOLO plant classification
- create/update plant assets in farmOS
- optional remote low-confidence LLM review
- optional map segmentation -> polygon geometry update
- capture/upload activity-log images
- retrieve/export plant asset data
- cleanup non-plant and duplicate activity-log images

## Current MCP Tools

1. `get_server_info()`
2. `classify_camera_and_sync_asset(camera_index=0, delay_seconds=5.0, mode="ask", asset_name_override="")`
3. `capture_and_upload_asset_images(image_count=5, interval_seconds=3.0, camera_index=0, start_delay_seconds=0.0, asset_id="", force_new_log=False)`
4. `apply_segmented_image_geometry_to_asset(segmented_image_path, latitude=None, longitude=None, zoom=18, asset_id="", create_new_if_missing=True, new_asset_name="", land_type="bed")`
5. `collect_data_with_coordinates(latitude=None, longitude=None, mode="ask", asset_name_override="", camera_index=0, classifier_delay_seconds=5.0, collection_interval_seconds=3.0, collection_duration_seconds=15.0, mapbox_zoom=18, mapbox_width=800, mapbox_height=800, segment_server_host="", segment_server_port=0, upload_to_activity_log=False)`
6. `retrieve_plant_asset_data(plant_name, mode="ask", selected_asset_id="", selected_asset_name="", selected_asset_index=0, export_dir="")`
7. `cleanup_asset_activity_log_images(mode="scan", asset_id="", asset_name_contains="", cleanup_scope="both", days_back=30, non_plant_conf_threshold=25.0, duplicate_hash_distance=6, keep_policy="highest_confidence", allow_multi_asset_execute=False, hard_delete_files=False, cleanup_local_captures=False, local_capture_dir="", confirmation_text="")`

## Architecture

- **MCP server**: `farmos_leaf_mcp.py`
- **farmOS API client**: `farmOS` Python package
- **Vision model**: external YOLO scripts (`infer_image.py`, `live_preview.py`)
- **Remote services (ZeroMQ REQ/REP)**:
  - segmentation server (`send_image_for_segmentation`)
  - low-confidence review server (`request_low_confidence_text`)

## Requirements

- Python 3.10+
- Camera accessible by OpenCV
- farmOS credentials + OAuth client
- YOLO model/scripts available on disk
- Optional for geometry flow: Mapbox token + remote segmentation server
- Optional for GPS: NMEA serial device (default `/dev/ttyACM0`)

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy and edit environment file:

```bash
cp .env.example .env
```

Main variables:

- farmOS auth: `FARMOS_HOST`, `FARMOS_USER`, `FARMOS_PASSWORD`, `FARMOS_CLIENT_ID`, `FARMOS_CLIENT_SECRET`
- YOLO paths: `LEAF_YOLO_DIR`, `LEAF_YOLO_MODEL_PATH`, `LEAF_YOLO_INFER_SCRIPT`, `LEAF_YOLO_LIVE_PREVIEW_SCRIPT`
- ZMQ remote servers:
  - segmentation: `SEGMENT_SERVER_HOST`, `SEGMENT_SERVER_PORT`
  - low-confidence LLM review: `LOW_CONF_SERVER_HOST`, `LOW_CONF_SERVER_PORT`
- Map image source token: `MAPBOX_TOKEN`
- Directories: `LEAF_CAPTURE_DIR`, `MAPBOX_INPUT_DIR`, `SEGMENTED_DIR`, `LEAF_EXPORT_DIR`

## Run

```bash
./run_mcp.sh
```

This script:
- activates venv (if present)
- loads `.env` (if present)
- starts `farmos_leaf_mcp.py`

## Tool Behavior Notes

### `classify_camera_and_sync_asset`
- Captures an image and classifies plant label/confidence with YOLO.
- If matching asset exists:
  - `mode="ask"`: returns prompt
  - `mode="overwrite"`: updates existing asset
  - `mode="create_new"`: creates suffixed asset
- If confidence is below threshold, sends image to remote low-confidence review server and stores LLM output in notes/activity logs.

### `collect_data_with_coordinates`
End-to-end flow combining asset sync + parallel tasks:
1. classify/upsert plant asset
2. run geometry pipeline (Mapbox image -> remote segmentation -> contour -> WKT -> update asset geometry)
3. capture interval images and optionally upload to activity log

### `cleanup_asset_activity_log_images`
- `mode="scan"`: report candidates only
- `mode="execute"`: perform unlink/delete (with safety confirmation)
- duplicate detection uses pHash only on images YOLO classifies as plant.

## Repository Hygiene

Generated runtime outputs are intentionally ignored in Git:
- `captures/`
- `mapbox_inputs/`
- `received_segmented_mapbox_image/`
- `plant_exports/`

If you need sample artifacts, commit a small curated subset under a dedicated `examples/` directory.

## Security Notes

- Do not commit real credentials, tokens, or private host IPs to the repo.
- Keep secrets in `.env` (ignored by Git).
- Rotate tokens if they were ever committed historically.
