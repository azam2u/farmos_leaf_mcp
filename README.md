# farmOS Leaf Classifier MCP

This is a separate MCP server that captures a camera image, classifies the plant using a YOLO model, and syncs a plant asset in farmOS.

## Tool
- `classify_camera_and_sync_asset(camera_index=0, delay_seconds=5, mode='ask', asset_name_override='')`
- `capture_and_upload_asset_images(image_count=5, interval_seconds=3, camera_index=0, start_delay_seconds=0, asset_id='')`
- `apply_segmented_image_geometry_to_asset(segmented_image_path, latitude, longitude, zoom=18, asset_id='', create_new_if_missing=True, new_asset_name='', land_type='bed')`
- `collect_data_with_coordinates(latitude, longitude, mode='ask', asset_name_override='', camera_index=0, classifier_delay_seconds=5, collection_interval_seconds=3, collection_duration_seconds=15, mapbox_zoom=18, mapbox_width=800, mapbox_height=800, segment_server_host='', segment_server_port=0)`

Conflict behavior when an asset with predicted name already exists:
- `mode='ask'`: returns a message prompting overwrite or create_new.
- `mode='overwrite'`: updates the existing asset.
- `mode='create_new'`: creates a new asset with a timestamp suffix.

YOLO inference behavior:
- MCP captures an image, then calls `/home/cvl/farmos_yolo_model/infer_image.py`.
- The script runs `/home/cvl/farmos_yolo_model/weights/best.pt` and returns predicted label + confidence.
- When `mode='ask'`, MCP opens a live camera YOLO preview window using `/home/cvl/farmos_yolo_model/live_preview.py`.
  - Press `o` to continue with `overwrite`.
  - Press `n` to continue with `create_new`.
  - Press `Enter` to continue with normal `ask` flow.
  - Press `q` (or `Esc`) to cancel.

Low-confidence remote review behavior:
- If YOLO confidence is below `LOW_CONF_THRESHOLD_PERCENT` (default `50.0`), MCP sends the same inference image to a remote ZMQ server.
- It stores the remote server text reply in the asset `notes` together with the YOLO inference note.
- Default remote server target is `LOW_CONF_SERVER_HOST=100.113.175.27` and `LOW_CONF_SERVER_PORT=5555`.
- Timeout is controlled by `LOW_CONF_TIMEOUT_MS` (default `20000`).

Image upload behavior:
- Captures multiple images from camera and uploads them to a new `activity` log.
- That log is linked to the target asset.
- If `asset_id` is omitted, it uses the last asset created/updated by `classify_camera_and_sync_asset`.

Segmented polygon behavior:
- Extracts the red segmented region from a segmented map image.
- Converts contour pixels into WKT polygon using map projection math.
- Updates geometry of existing asset (`asset_id` or last asset), or creates a new land asset.

One-shot behavior (`collect_data_with_coordinates`):
- Classifies plant from camera capture.
- Checks existing asset name.
- If conflict and `mode='ask'`, returns overwrite/create_new prompt without continuing.
- After create/overwrite, runs both in parallel:
  - Mapbox download from provided coordinates -> ZMQ segmentation -> crop -> WKT -> asset geometry update.
  - Camera image collection every 3s for 15s (configurable) with uploads to an activity log linked to same asset.

## Run
```bash
/home/cvl/farmos_leaf_mcp/run_mcp.sh
```
