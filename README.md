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
8. `get_collection_job_status(job_id)`
9. `list_collection_jobs(limit=20)`

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

## Automatic Collection

Run the complete pipeline continuously without Roo or an MCP prompt:

```bash
./run_auto_collect.sh
```

The automatic runner uses a foreground/background job architecture:

1. reads a coordinate
2. creates and prints a durable job ID
3. foreground: shows YOLO identification for `AUTO_CLASSIFIER_DELAY_SECONDS`
4. foreground: captures all timed data images
5. queues the job and waits `AUTO_POLL_SECONDS` (default 10 seconds)
6. background: performs remote LLM review and creates the farmOS asset
7. background: uploads captured images and updates segmented geometry in parallel
8. reads the next coordinate and starts another foreground job only after moving at least `AUTO_MIN_MOVEMENT_METERS`

It uses `mode="create_new"` so no live-preview confirmation or overwrite prompt is required.
Foreground collection can continue while earlier jobs finish remotely. Background jobs are
processed serially and stored in `AUTO_JOB_DATABASE`. Stop with `Ctrl+C`; the runner waits
for queued background jobs before exiting.

Current dummy-coordinate defaults simulate 1.2 meters of northward movement per check:

```text
AUTO_COORDINATE_SOURCE=dummy
AUTO_DUMMY_LATITUDE=35.009445
AUTO_DUMMY_LONGITUDE=135.718787
AUTO_DUMMY_MOVE_METERS_PER_CHECK=1.2
```

Set `AUTO_DUMMY_MOVE_METERS_PER_CHECK=0` to simulate a stationary device. To
use the real NMEA GPS later:

```text
AUTO_COORDINATE_SOURCE=gps
GPS_DEVICE=/dev/ttyACM0
```

Safe movement-only test, with no camera or farmOS changes:

```bash
./run_auto_collect.sh --dry-run --max-checks 3 --poll-seconds 0.1
```

Stationary dummy-coordinate test:

```bash
./run_auto_collect.sh --dry-run --max-checks 3 \
  --poll-seconds 0.1 --dummy-move-meters 0
```

Run exactly one real collection:

```bash
./run_auto_collect.sh --max-runs 1
```

List recent jobs:

```bash
./run_auto_collect.sh --list-jobs
```

Check one job:

```bash
./run_auto_collect.sh --status JOB_ID
```

Job stages include:

- `gps_trigger`
- `foreground_data_capture`
- `background_queued`
- `background_asset`
- `background_postprocess`
- `completed`, `completed_with_warnings`, or `failed`

The same status is available to MCP clients through `get_collection_job_status`
and `list_collection_jobs`.

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
