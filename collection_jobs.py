#!/usr/bin/env python3
import datetime
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Optional

def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


class CollectionJobStore:
    def __init__(self, database_path: str):
        self.database_path = os.path.abspath(database_path)
        os.makedirs(os.path.dirname(self.database_path) or ".", exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    coordinate_source TEXT NOT NULL,
                    predicted_label TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    identification_image TEXT NOT NULL DEFAULT '',
                    data_images_json TEXT NOT NULL DEFAULT '[]',
                    asset_id TEXT NOT NULL DEFAULT '',
                    asset_name TEXT NOT NULL DEFAULT '',
                    activity_log_id TEXT NOT NULL DEFAULT '',
                    mapbox_path TEXT NOT NULL DEFAULT '',
                    segmented_path TEXT NOT NULL DEFAULT '',
                    cropped_path TEXT NOT NULL DEFAULT '',
                    result_message TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection_jobs_status_created "
                "ON collection_jobs(status, created_at)"
            )

    def create(self, latitude: float, longitude: float, coordinate_source: str) -> str:
        job_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO collection_jobs (
                    job_id, status, stage, created_at, updated_at,
                    latitude, longitude, coordinate_source
                ) VALUES (?, 'foreground', 'gps_trigger', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    timestamp,
                    timestamp,
                    float(latitude),
                    float(longitude),
                    coordinate_source,
                ),
            )
        return job_id

    def update(self, job_id: str, **fields):
        allowed = {
            "status",
            "stage",
            "predicted_label",
            "confidence",
            "identification_image",
            "data_images_json",
            "asset_id",
            "asset_name",
            "activity_log_id",
            "mapbox_path",
            "segmented_path",
            "cropped_path",
            "result_message",
            "error_message",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported job field(s): {', '.join(sorted(unknown))}")
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [job_id]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE collection_jobs SET {assignments} WHERE job_id = ?",
                values,
            )

    def get(self, job_id: str) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM collection_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def list(self, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM collection_jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def recover_interrupted(self):
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE collection_jobs
                SET status = 'failed',
                    stage = 'foreground_interrupted',
                    updated_at = ?,
                    error_message = 'Foreground capture was interrupted before the job was queued.'
                WHERE status = 'foreground'
                """,
                (utc_now(),),
            )
            connection.execute(
                """
                UPDATE collection_jobs
                SET status = 'queued',
                    stage = 'background_queued',
                    updated_at = ?,
                    error_message = CASE
                        WHEN error_message = '' THEN 'Recovered after runner restart.'
                        ELSE error_message
                    END
                WHERE status = 'processing'
                """,
                (utc_now(),),
            )

    def claim_next(self) -> Optional[dict]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM collection_jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE collection_jobs
                SET status = 'processing',
                    stage = 'background_asset',
                    updated_at = ?,
                    error_message = ''
                WHERE job_id = ? AND status = 'queued'
                """,
                (utc_now(), row["job_id"]),
            )
            connection.commit()
        return self.get(row["job_id"])

    def has_unfinished(self) -> bool:
        with self._connect() as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM collection_jobs
                WHERE status IN ('foreground', 'queued', 'processing')
                """
            ).fetchone()[0]
        return bool(count)

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        item = dict(row)
        try:
            item["data_images"] = json.loads(item.pop("data_images_json"))
        except Exception:
            item["data_images"] = []
            item.pop("data_images_json", None)
        return item


def upload_job_images(job: dict, asset_id: str, asset_type: str) -> dict:
    import farmos_leaf_mcp as app

    images = job.get("data_images") or []
    if not images:
        return {"log_id": "", "uploaded": 0}
    farm = app.get_client()
    log_id = app.create_asset_log(
        farm=farm,
        asset_id=asset_id,
        asset_type=asset_type,
        name=f"Automatic Collection {job['job_id']} ({len(images)} images)",
    )
    if not log_id:
        raise RuntimeError("Could not create the automatic collection activity log.")
    uploaded = 0
    for image_path in images:
        app.upload_image_to_activity_log(farm, log_id, image_path)
        uploaded += 1
    return {"log_id": log_id, "uploaded": uploaded}


def process_collection_job(store: CollectionJobStore, job: dict):
    import farmos_leaf_mcp as app

    job_id = job["job_id"]
    try:
        farm = app.get_client()
        asset_name = app.normalize_asset_name(job["predicted_label"])
        upsert = app.upsert_plant_asset(
            farm=farm,
            asset_name=asset_name,
            predicted_label=job["predicted_label"],
            confidence=float(job["confidence"]),
            image_path=job["identification_image"],
            mode="create_new",
            latitude=float(job["latitude"]),
            longitude=float(job["longitude"]),
            coordinate_source=job["coordinate_source"],
            job_id=job_id,
        )
        asset_id = upsert["asset_id"]
        asset_type = upsert.get("asset_type", "asset--plant")
        store.update(
            job_id,
            stage="background_postprocess",
            asset_id=asset_id,
            asset_name=upsert.get("asset_name", asset_name),
            result_message=upsert["message"],
        )

        results = {"images": None, "geometry": None}
        errors = []

        def image_worker():
            upload_enabled = os.environ.get(
                "AUTO_UPLOAD_TO_ACTIVITY_LOG", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            if not upload_enabled:
                results["images"] = {"log_id": "", "uploaded": 0}
                return
            try:
                results["images"] = upload_job_images(job, asset_id, asset_type)
            except Exception as exc:
                errors.append(f"Image upload failed: {exc}")

        def geometry_worker():
            try:
                results["geometry"] = app.run_mapbox_segment_and_apply_geometry(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    latitude=float(job["latitude"]),
                    longitude=float(job["longitude"]),
                    zoom=int(os.environ.get("AUTO_MAPBOX_ZOOM", "18")),
                    width=int(os.environ.get("AUTO_MAPBOX_WIDTH", "800")),
                    height=int(os.environ.get("AUTO_MAPBOX_HEIGHT", "800")),
                    server_host=app.SEGMENT_SERVER_HOST,
                    server_port=app.SEGMENT_SERVER_PORT,
                )
            except Exception as exc:
                errors.append(f"Geometry update failed: {exc}")

        image_thread = threading.Thread(target=image_worker)
        geometry_thread = threading.Thread(target=geometry_worker)
        image_thread.start()
        geometry_thread.start()
        image_thread.join()
        geometry_thread.join()

        update_fields = {
            "status": "completed" if not errors else "completed_with_warnings",
            "stage": "completed",
            "error_message": " | ".join(errors),
        }
        messages = [upsert["message"]]
        if results["images"]:
            update_fields["activity_log_id"] = results["images"]["log_id"]
            messages.append(
                f"Uploaded {results['images']['uploaded']} images to activity log "
                f"{results['images']['log_id']}."
            )
        if results["geometry"]:
            geometry = results["geometry"]
            update_fields.update(
                {
                    "mapbox_path": geometry["mapbox_path"],
                    "segmented_path": geometry["segmented_path"],
                    "cropped_path": geometry["cropped_path"],
                }
            )
            messages.append(f"Geometry updated on asset {geometry['asset_id']}.")
        if errors:
            messages.append("Warnings: " + " | ".join(errors))
        update_fields["result_message"] = "\n".join(messages)
        store.update(job_id, **update_fields)
    except Exception as exc:
        store.update(
            job_id,
            status="failed",
            stage="failed",
            error_message=str(exc),
            result_message=f"Background processing failed: {exc}",
        )


class CollectionJobWorker:
    def __init__(self, store: CollectionJobStore, poll_seconds: float = 0.5):
        self.store = store
        self.poll_seconds = max(0.1, poll_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="farmos-collection-job-worker",
            daemon=False,
        )

    def start(self):
        self.store.recover_interrupted()
        self._thread.start()

    def stop_when_idle(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while True:
            job = self.store.claim_next()
            if job:
                process_collection_job(self.store, job)
                continue
            if self._stop.is_set():
                return
            time.sleep(self.poll_seconds)
