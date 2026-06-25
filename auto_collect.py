#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import farmos_leaf_mcp as app
from collection_jobs import CollectionJobStore, CollectionJobWorker


EARTH_RADIUS_METERS = 6_371_000.0


@dataclass(frozen=True)
class Coordinate:
    latitude: float
    longitude: float


def haversine_meters(start: Coordinate, end: Coordinate) -> float:
    lat1 = math.radians(start.latitude)
    lat2 = math.radians(end.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(end.longitude - start.longitude)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(value)))


def move_north(origin: Coordinate, distance_meters: float) -> Coordinate:
    latitude_delta = math.degrees(distance_meters / EARTH_RADIUS_METERS)
    return Coordinate(origin.latitude + latitude_delta, origin.longitude)


class CoordinateSource:
    def read(self) -> Coordinate:
        raise NotImplementedError


class DummyCoordinateSource(CoordinateSource):
    def __init__(self, latitude: float, longitude: float, move_per_check_meters: float):
        self._coordinate = Coordinate(latitude, longitude)
        self._move_per_check_meters = max(0.0, move_per_check_meters)
        self._first_read = True

    def read(self) -> Coordinate:
        if self._first_read:
            self._first_read = False
        elif self._move_per_check_meters > 0:
            self._coordinate = move_north(self._coordinate, self._move_per_check_meters)
        return self._coordinate


class GpsCoordinateSource(CoordinateSource):
    def read(self) -> Coordinate:
        latitude, longitude = app.read_gps_coordinates()
        return Coordinate(latitude, longitude)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete farmOS collection pipeline after moving a minimum distance."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test coordinate polling and movement gating without running the pipeline.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=int(os.environ.get("AUTO_MAX_RUNS", "0")),
        help="Stop after this many pipeline runs; 0 means run until interrupted.",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=0,
        help="Stop after this many coordinate checks; intended for dry-run testing.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Override AUTO_POLL_SECONDS for this run.",
    )
    parser.add_argument(
        "--dummy-move-meters",
        type=float,
        default=None,
        help="Override AUTO_DUMMY_MOVE_METERS_PER_CHECK for this run.",
    )
    parser.add_argument(
        "--status",
        metavar="JOB_ID",
        help="Show one collection job and exit.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent collection jobs and exit.",
    )
    return parser


def make_coordinate_source(dummy_move_override: Optional[float]) -> CoordinateSource:
    source_name = os.environ.get("AUTO_COORDINATE_SOURCE", "dummy").strip().lower()
    if source_name == "dummy":
        move_per_check = (
            dummy_move_override
            if dummy_move_override is not None
            else float(os.environ.get("AUTO_DUMMY_MOVE_METERS_PER_CHECK", "1.2"))
        )
        return DummyCoordinateSource(
            latitude=float(os.environ.get("AUTO_DUMMY_LATITUDE", "35.009445")),
            longitude=float(os.environ.get("AUTO_DUMMY_LONGITUDE", "135.718787")),
            move_per_check_meters=move_per_check,
        )
    if source_name == "gps":
        return GpsCoordinateSource()
    raise ValueError("AUTO_COORDINATE_SOURCE must be 'dummy' or 'gps'.")


def run_foreground_capture(
    store: CollectionJobStore,
    coordinate: Coordinate,
    coordinate_source: str,
) -> tuple[str, str]:
    job_id = store.create(
        latitude=coordinate.latitude,
        longitude=coordinate.longitude,
        coordinate_source=coordinate_source,
    )
    print(f"Job {job_id}: foreground capture started.")
    try:
        camera_index = int(os.environ.get("AUTO_CAMERA_INDEX", "0"))
        classifier_delay = float(os.environ.get("AUTO_CLASSIFIER_DELAY_SECONDS", "5"))
        interval_seconds = max(
            0.1, float(os.environ.get("AUTO_COLLECTION_INTERVAL_SECONDS", "3"))
        )
        duration_seconds = max(
            1.0, float(os.environ.get("AUTO_COLLECTION_DURATION_SECONDS", "15"))
        )

        identification_image = app.capture_image(
            camera_index=camera_index,
            delay_seconds=classifier_delay,
        )
        predicted_label, confidence = app.classify_image(identification_image)
        store.update(
            job_id,
            stage="foreground_data_capture",
            predicted_label=predicted_label,
            confidence=confidence,
            identification_image=identification_image,
        )
        print(
            f"Job {job_id}: YOLO={predicted_label} ({confidence:.2f}%). "
            "Collecting data images."
        )

        image_count = max(1, int(duration_seconds / interval_seconds))
        data_images = app.capture_frames(
            camera_index=camera_index,
            image_count=image_count,
            interval_seconds=interval_seconds,
            start_delay_seconds=interval_seconds,
        )
        store.update(
            job_id,
            status="queued",
            stage="background_queued",
            data_images_json=json.dumps(data_images),
            result_message=(
                f"Foreground complete: {predicted_label} ({confidence:.2f}%), "
                f"{len(data_images)} data images captured."
            ),
        )
        return job_id, ""
    except Exception as exc:
        store.update(
            job_id,
            status="failed",
            stage="foreground_failed",
            error_message=str(exc),
            result_message=f"Foreground capture failed: {exc}",
        )
        return job_id, str(exc)


def print_job(job: dict):
    print(
        f"{job['job_id']} | {job['status']} | {job['stage']} | "
        f"{job['latitude']:.8f},{job['longitude']:.8f} | "
        f"{job.get('predicted_label') or '-'} {job.get('confidence', 0):.2f}% | "
        f"asset={job.get('asset_id') or '-'} | log={job.get('activity_log_id') or '-'}"
    )
    if job.get("result_message"):
        print(job["result_message"])
    if job.get("error_message"):
        print("Error:", job["error_message"])


def main() -> int:
    args = build_parser().parse_args()
    database_path = os.environ.get(
        "AUTO_JOB_DATABASE",
        os.path.join(app.SCRIPT_DIR, "collection_jobs", "jobs.sqlite3"),
    )
    store = CollectionJobStore(database_path)

    if args.status:
        job = store.get(args.status)
        if not job:
            print(f"Job not found: {args.status}", file=sys.stderr)
            return 1
        print_job(job)
        return 0
    if args.list_jobs:
        jobs = store.list(limit=int(os.environ.get("AUTO_JOB_LIST_LIMIT", "20")))
        if not jobs:
            print("No collection jobs found.")
        for job in jobs:
            print_job(job)
        return 0

    runner_lock_path = os.environ.get(
        "AUTO_RUNNER_LOCK_PATH",
        "/tmp/farmos_leaf_auto_collect.lock",
    )
    runner_lock = open(runner_lock_path, "a+")
    try:
        fcntl.flock(runner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            "Error: Another automatic collection runner is already active.",
            file=sys.stderr,
        )
        runner_lock.close()
        return 1

    if (
        args.max_runs < 0
        or args.max_checks < 0
        or (args.poll_seconds is not None and args.poll_seconds < 0)
        or (args.dummy_move_meters is not None and args.dummy_move_meters < 0)
    ):
        print("Error: run, check, poll, and movement values must be >= 0.", file=sys.stderr)
        return 2

    minimum_movement = max(
        0.0, float(os.environ.get("AUTO_MIN_MOVEMENT_METERS", "1.0"))
    )
    configured_poll_seconds = (
        args.poll_seconds
        if args.poll_seconds is not None
        else float(os.environ.get("AUTO_POLL_SECONDS", "10"))
    )
    poll_seconds = max(0.1, configured_poll_seconds)
    source = make_coordinate_source(args.dummy_move_meters)
    coordinate_source_name = os.environ.get("AUTO_COORDINATE_SOURCE", "dummy").strip().lower()
    stop_requested = False
    worker = None if args.dry_run else CollectionJobWorker(store)
    if worker:
        worker.start()

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    last_run_coordinate: Optional[Coordinate] = None
    completed_runs = 0
    coordinate_checks = 0

    print(
        "Automatic collection started: "
        f"source={os.environ.get('AUTO_COORDINATE_SOURCE', 'dummy')}, "
        f"minimum_movement={minimum_movement:.2f}m, poll={poll_seconds:.1f}s, "
        f"dry_run={args.dry_run}"
    )

    while not stop_requested:
        try:
            coordinate = source.read()
            coordinate_checks += 1
        except Exception as exc:
            print(f"Coordinate read failed: {exc}")
            if stop_requested:
                break
            time.sleep(poll_seconds)
            continue

        movement = (
            None
            if last_run_coordinate is None
            else haversine_meters(last_run_coordinate, coordinate)
        )
        movement_text = "first run" if movement is None else f"moved={movement:.2f}m"
        print(
            f"Coordinate {coordinate.latitude:.8f}, {coordinate.longitude:.8f} "
            f"({movement_text})"
        )

        should_run = movement is None or movement >= minimum_movement
        if should_run:
            if args.dry_run:
                result = "Dry run: pipeline skipped."
            else:
                job_id, error = run_foreground_capture(
                    store=store,
                    coordinate=coordinate,
                    coordinate_source=coordinate_source_name,
                )
                result = (
                    f"Job {job_id} queued for background processing."
                    if not error
                    else f"Error: Job {job_id} foreground failed: {error}"
                )
                print(result)

            if not result.startswith("Error:"):
                last_run_coordinate = coordinate
                completed_runs += 1
                print(f"Foreground jobs queued: {completed_runs}")
            else:
                print("Pipeline failed; movement baseline was not advanced.")
        else:
            print(
                f"Movement is below {minimum_movement:.2f}m; "
                "waiting without collecting."
            )

        if args.max_runs > 0 and completed_runs >= args.max_runs:
            break
        if args.max_checks > 0 and coordinate_checks >= args.max_checks:
            break
        if not stop_requested:
            time.sleep(poll_seconds)

    if worker:
        print("Foreground collection stopped; waiting for background jobs to finish...")
        worker.stop_when_idle()
    fcntl.flock(runner_lock.fileno(), fcntl.LOCK_UN)
    runner_lock.close()
    print("Automatic collection stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
