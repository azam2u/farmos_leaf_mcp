#!/usr/bin/env python3
import asyncio
import collections
import json
import os
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

import farmos_leaf_mcp as farm_app
from collection_jobs import CollectionJobStore


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "collection_jobs" / "dashboard"
CONTROL_FILE = RUNTIME_DIR / "control.json"
PREVIEW_FILE = RUNTIME_DIR / "preview.jpg"
DATABASE_PATH = Path(
    os.environ.get("AUTO_JOB_DATABASE", ROOT / "collection_jobs" / "jobs.sqlite3")
)
UI_HOST = os.environ.get("FARMOS_UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("FARMOS_UI_PORT", "8765"))


def write_control(command: str):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONTROL_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"command": command}), encoding="utf-8")
    temporary.replace(CONTROL_FILE)


class CollectorSupervisor:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._state = "stopped"
        self._logs = collections.deque(maxlen=1200)
        self._next_log_id = 1

    def _log(self, text: str):
        line = text.rstrip()
        if not line:
            return
        with self._lock:
            self._logs.append(
                {
                    "id": self._next_log_id,
                    "time": time.strftime("%H:%M:%S"),
                    "text": line,
                }
            )
            self._next_log_id += 1

    def _reader(self, process):
        try:
            for line in iter(process.stdout.readline, ""):
                self._log(line)
        finally:
            return_code = process.wait()
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._state = "stopped"
            self._log(f"Collector exited with status {return_code}.")

    def start(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                return False, "Collector is already running."

            write_control("run")
            environment = os.environ.copy()
            environment.update(
                {
                    "AUTO_CONTROL_FILE": str(CONTROL_FILE),
                    "LEAF_UI_PREVIEW_PATH": str(PREVIEW_FILE),
                    "LEAF_CAPTURE_LIVE_INFERENCE": "0",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            process = subprocess.Popen(
                [str(ROOT / "run_auto_collect.sh")],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._process = process
            self._state = "running"
            threading.Thread(
                target=self._reader,
                args=(process,),
                name="dashboard-collector-log-reader",
                daemon=True,
            ).start()
        self._log("Collector started from dashboard.")
        return True, "Collector started."

    def pause(self):
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return False, "Collector is not running."
            write_control("pause")
            self._state = "paused"
        self._log("Pause requested. Current foreground capture will finish first.")
        return True, "Collector paused."

    def resume(self):
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return False, "Collector is not running."
            write_control("run")
            self._state = "running"
        self._log("Collector resumed.")
        return True, "Collector resumed."

    def stop(self):
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self._state = "stopped"
                return False, "Collector is not running."
            write_control("stop")
            self._state = "stopping"
        self._log("Stop requested. Background jobs will finish before exit.")
        return True, "Collector stopping."

    def status(self):
        with self._lock:
            process = self._process
            if process and process.poll() is not None:
                self._process = None
                self._state = "stopped"
                process = None
            return {
                "state": self._state,
                "pid": process.pid if process else None,
            }

    def logs_after(self, after: int):
        with self._lock:
            return [entry for entry in self._logs if entry["id"] > after]


class HealthMonitor:
    def __init__(self, supervisor: CollectorSupervisor):
        self.supervisor = supervisor
        self._lock = threading.Lock()
        self._state = {
            "checking": True,
            "updated_at": "",
            "checks": [],
        }

    @staticmethod
    def _check(name, status, detail, required=True):
        return {
            "name": name,
            "status": status,
            "detail": detail,
            "required": required,
        }

    @staticmethod
    def _tailscale_ping(host: str) -> tuple[bool, str]:
        if not host:
            return False, "Host is not configured."
        command = [
            "/home/cvl/tools/tailscale/tailscale",
            "--socket=/home/cvl/.tailscale/tailscaled.sock",
            "ping",
            "--timeout=3s",
            "-c",
            "1",
            host,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=6, check=False
            )
            output = (result.stdout or result.stderr).strip().splitlines()
            return result.returncode == 0, output[-1] if output else "No response."
        except Exception as exc:
            return False, str(exc)

    def run(self):
        checks = []
        try:
            farm = farm_app.get_client()
            info = farm.info()
            meta = info.get("meta", {}).get("farm", {}) if isinstance(info, dict) else {}
            checks.append(
                self._check(
                    "farmOS",
                    "ok",
                    f"Connected to {meta.get('name', 'farmOS')} {meta.get('version', '')}".strip(),
                )
            )
        except Exception as exc:
            checks.append(self._check("farmOS", "error", str(exc)))

        tool_names = [
            "get_server_info",
            "get_collection_job_status",
            "list_collection_jobs",
            "classify_camera_and_sync_asset",
            "capture_and_upload_asset_images",
            "apply_segmented_image_geometry_to_asset",
            "collect_data_with_coordinates",
            "retrieve_plant_asset_data",
            "cleanup_asset_activity_log_images",
        ]
        missing_tools = [name for name in tool_names if not hasattr(farm_app, name)]
        checks.append(
            self._check(
                "MCP tools",
                "ok" if not missing_tools else "error",
                f"{len(tool_names) - len(missing_tools)}/{len(tool_names)} tools available"
                if not missing_tools
                else "Missing: " + ", ".join(missing_tools),
            )
        )

        model_paths = [
            farm_app.YOLO_MODEL_PATH,
            farm_app.YOLO_INFER_SCRIPT,
            farm_app.YOLO_LIVE_PREVIEW_SCRIPT,
        ]
        missing_model_paths = [path for path in model_paths if not os.path.isfile(path)]
        checks.append(
            self._check(
                "YOLO",
                "ok" if not missing_model_paths else "error",
                "Model and inference scripts found"
                if not missing_model_paths
                else "Missing: " + ", ".join(missing_model_paths),
            )
        )

        if self.supervisor.status()["state"] in {"running", "paused", "stopping"}:
            checks.append(
                self._check("Camera", "ok", "In use by the active collector")
            )
        else:
            camera_index = int(os.environ.get("AUTO_CAMERA_INDEX", "0"))
            try:
                cap = farm_app._open_camera(camera_index)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    raise RuntimeError("Camera opened but returned no frame.")
                checks.append(
                    self._check(
                        "Camera",
                        "ok",
                        f"Camera {camera_index}: {frame.shape[1]}x{frame.shape[0]}",
                    )
                )
            except Exception as exc:
                checks.append(self._check("Camera", "error", str(exc)))

        coordinate_source = os.environ.get("AUTO_COORDINATE_SOURCE", "dummy").lower()
        gps_device = farm_app.GPS_DEVICE
        gps_present = os.path.exists(gps_device)
        if coordinate_source == "dummy":
            checks.append(
                self._check(
                    "GPS",
                    "warning",
                    f"Dummy coordinates active; physical device {'found' if gps_present else 'not connected'}",
                    required=False,
                )
            )
        elif not gps_present:
            checks.append(self._check("GPS", "error", f"Device not found: {gps_device}"))
        else:
            try:
                latitude, longitude = farm_app.read_gps_coordinates(
                    timeout_seconds=min(5.0, farm_app.GPS_READ_TIMEOUT_SECONDS)
                )
                checks.append(
                    self._check("GPS", "ok", f"Fix: {latitude:.6f}, {longitude:.6f}")
                )
            except Exception as exc:
                checks.append(self._check("GPS", "warning", str(exc), required=False))

        checks.append(
            self._check(
                "Mapbox",
                "ok" if bool(farm_app.MAPBOX_TOKEN) else "error",
                "Token configured" if farm_app.MAPBOX_TOKEN else "Token missing",
            )
        )

        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "tailscaled-userspace.service"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            ).stdout.strip()
            checks.append(
                self._check(
                    "Tailscale",
                    "ok" if active == "active" else "error",
                    f"Service: {active or 'inactive'}",
                )
            )
        except Exception as exc:
            checks.append(self._check("Tailscale", "error", str(exc)))

        segment_ok, segment_detail = self._tailscale_ping(farm_app.SEGMENT_SERVER_HOST)
        checks.append(
            self._check(
                "Segmentation PC",
                "ok" if segment_ok else "error",
                segment_detail,
            )
        )
        llm_ok, llm_detail = self._tailscale_ping(farm_app.LOW_CONF_SERVER_HOST)
        checks.append(
            self._check(
                "Remote LLM PC",
                "ok" if llm_ok else "warning",
                llm_detail,
                required=False,
            )
        )

        try:
            store = CollectionJobStore(str(DATABASE_PATH))
            recent_count = len(store.list(limit=20))
            checks.append(
                self._check("Job database", "ok", f"Ready; {recent_count} recent jobs")
            )
        except Exception as exc:
            checks.append(self._check("Job database", "error", str(exc)))

        with self._lock:
            self._state = {
                "checking": False,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checks": checks,
            }

    def refresh_async(self):
        with self._lock:
            self._state["checking"] = True
        threading.Thread(target=self.run, name="dashboard-health-check", daemon=True).start()

    def state(self):
        with self._lock:
            return json.loads(json.dumps(self._state))


supervisor = CollectorSupervisor()
health = HealthMonitor(supervisor)


async def homepage(_request):
    return HTMLResponse((ROOT / "ui" / "index.html").read_text(encoding="utf-8"))


async def status_api(_request):
    store = CollectionJobStore(str(DATABASE_PATH))
    jobs = store.list(limit=12)
    return JSONResponse(
        {
            "collector": supervisor.status(),
            "health": health.state(),
            "jobs": jobs,
            "preview_available": PREVIEW_FILE.is_file(),
        }
    )


async def logs_api(request):
    try:
        after = int(request.query_params.get("after", "0"))
    except ValueError:
        after = 0
    return JSONResponse({"logs": supervisor.logs_after(after)})


async def preview_api(_request):
    if not PREVIEW_FILE.is_file():
        return Response(status_code=204)
    return FileResponse(
        PREVIEW_FILE,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


async def control_api(request):
    action = request.path_params["action"]
    handlers = {
        "start": supervisor.start,
        "pause": supervisor.pause,
        "resume": supervisor.resume,
        "stop": supervisor.stop,
    }
    if action not in handlers:
        return JSONResponse({"ok": False, "message": "Unknown action."}, status_code=404)
    ok, message = await asyncio.to_thread(handlers[action])
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 409)


async def refresh_health_api(_request):
    health.refresh_async()
    return JSONResponse({"ok": True, "message": "Health checks started."})


async def job_api(request):
    job_id = request.path_params["job_id"]
    job = CollectionJobStore(str(DATABASE_PATH)).get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return JSONResponse(job)


async def on_startup():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    write_control("run")
    health.refresh_async()


@asynccontextmanager
async def lifespan(_app):
    await on_startup()
    yield


routes = [
    Route("/", homepage),
    Route("/api/status", status_api),
    Route("/api/logs", logs_api),
    Route("/api/preview", preview_api),
    Route("/api/control/{action}", control_api, methods=["POST"]),
    Route("/api/health/refresh", refresh_health_api, methods=["POST"]),
    Route("/api/jobs/{job_id}", job_api),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="info")
