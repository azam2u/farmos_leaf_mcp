#!/usr/bin/env python3
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import zmq
from farmOS import farmOS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FarmOS Leaf Classifier")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

HOSTNAME = os.environ.get("FARMOS_HOST", "https://try.farmos.net")
USERNAME = os.environ.get("FARMOS_USER", "mark")
PASSWORD = os.environ.get("FARMOS_PASSWORD", "")
CLIENT_ID = os.environ.get("FARMOS_CLIENT_ID", "farm")
CLIENT_SECRET = os.environ.get("FARMOS_CLIENT_SECRET", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_DIR = os.environ.get("LEAF_YOLO_DIR", "/home/cvl/farmos_yolo_model")
YOLO_MODEL_PATH = os.environ.get("LEAF_YOLO_MODEL_PATH", os.path.join(YOLO_MODEL_DIR, "weights", "best.pt"))
YOLO_INFER_SCRIPT = os.environ.get("LEAF_YOLO_INFER_SCRIPT", os.path.join(YOLO_MODEL_DIR, "infer_image.py"))
YOLO_LIVE_PREVIEW_SCRIPT = os.environ.get("LEAF_YOLO_LIVE_PREVIEW_SCRIPT", os.path.join(YOLO_MODEL_DIR, "live_preview.py"))
YOLO_CONFIDENCE = float(os.environ.get("LEAF_YOLO_CONFIDENCE", "0.25"))
YOLO_LIVE_PREVIEW_ON_ASK = os.environ.get("LEAF_YOLO_LIVE_PREVIEW_ON_ASK", "1").strip().lower() not in {"0", "false", "no"}
CAPTURE_LIVE_INFERENCE = os.environ.get("LEAF_CAPTURE_LIVE_INFERENCE", "1").strip().lower() not in {"0", "false", "no"}
CAPTURE_LIVE_WINDOW_TITLE = os.environ.get("LEAF_CAPTURE_LIVE_WINDOW_TITLE", "FarmOS Capture Live Inference")
CAPTURE_DIR = os.environ.get("LEAF_CAPTURE_DIR", os.path.join(SCRIPT_DIR, "captures"))

MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "",
)
MAPBOX_INPUT_DIR = os.environ.get("MAPBOX_INPUT_DIR", os.path.join(SCRIPT_DIR, "mapbox_inputs"))
SEGMENTED_DIR = os.environ.get("SEGMENTED_DIR", os.path.join(SCRIPT_DIR, "received_segmented_mapbox_image"))
SEGMENT_SERVER_HOST = os.environ.get("SEGMENT_SERVER_HOST", "")
SEGMENT_SERVER_PORT = int(os.environ.get("SEGMENT_SERVER_PORT", "5555"))
LOW_CONF_SERVER_HOST = os.environ.get("LOW_CONF_SERVER_HOST", "")
LOW_CONF_SERVER_PORT = int(os.environ.get("LOW_CONF_SERVER_PORT", "5555"))
LOW_CONF_THRESHOLD_PERCENT = float(os.environ.get("LOW_CONF_THRESHOLD_PERCENT", "50.0"))
LOW_CONF_TIMEOUT_MS = int(os.environ.get("LOW_CONF_TIMEOUT_MS", "20000"))

LAST_ASSET_ID: Optional[str] = None
LAST_ASSET_TYPE: Optional[str] = None
LAST_ASSET_NAME: Optional[str] = None
LAST_ASSET_TS: Optional[float] = None
LAST_CAPTURE_LOG_ASSET_ID: Optional[str] = None
LAST_CAPTURE_LOG_ID: Optional[str] = None
LAST_CAPTURE_LOG_TS: Optional[float] = None
CAMERA_LOCK = threading.Lock()
CAMERA_LOCK_TIMEOUT_SECONDS = float(os.environ.get("LEAF_CAMERA_LOCK_TIMEOUT_SECONDS", "120"))

GPS_DEVICE = os.environ.get("GPS_DEVICE", "/dev/ttyACM0")
GPS_READ_TIMEOUT_SECONDS = float(os.environ.get("GPS_READ_TIMEOUT_SECONDS", "25"))


def get_client():
    farm = farmOS(HOSTNAME, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    farm.authorize(USERNAME, PASSWORD)
    return farm


def _parse_nmea_latlon(lat_raw: str, lat_hemi: str, lon_raw: str, lon_hemi: str) -> Tuple[float, float]:
    if not lat_raw or not lon_raw:
        raise ValueError("Missing NMEA coordinates.")

    lat_val = float(lat_raw)
    lon_val = float(lon_raw)

    lat_deg = int(lat_val // 100)
    lon_deg = int(lon_val // 100)
    lat_min = lat_val - (lat_deg * 100)
    lon_min = lon_val - (lon_deg * 100)

    lat = lat_deg + (lat_min / 60.0)
    lon = lon_deg + (lon_min / 60.0)

    if lat_hemi == "S":
        lat *= -1.0
    if lon_hemi == "W":
        lon *= -1.0

    return lat, lon


def read_gps_coordinates(device_path: str = GPS_DEVICE, timeout_seconds: float = GPS_READ_TIMEOUT_SECONDS) -> Tuple[float, float]:
    """
    Read a valid GNSS fix from an NMEA serial device and return (latitude, longitude).
    Accepts $GNRMC/$GPRMC status A or $GNGGA/$GPGGA fix quality > 0.
    """
    end_ts = time.time() + max(1.0, timeout_seconds)
    last_error = "No GPS sentences read."
    with open(device_path, "r", encoding="ascii", errors="ignore") as stream:
        while time.time() < end_ts:
            line = stream.readline().strip()
            if not line or not line.startswith("$"):
                continue

            parts = line.split(",")
            sentence = parts[0]

            if sentence in {"$GNRMC", "$GPRMC"}:
                if len(parts) < 7:
                    last_error = f"Incomplete RMC sentence: {line}"
                    continue
                status = parts[2]
                if status == "A":
                    return _parse_nmea_latlon(parts[3], parts[4], parts[5], parts[6])
                last_error = "GPS has no valid fix yet (RMC status is not A)."

            elif sentence in {"$GNGGA", "$GPGGA"}:
                if len(parts) < 7:
                    last_error = f"Incomplete GGA sentence: {line}"
                    continue
                quality = parts[6]
                if quality and quality != "0":
                    return _parse_nmea_latlon(parts[2], parts[3], parts[4], parts[5])
                last_error = "GPS has no valid fix yet (GGA quality is 0)."

    raise RuntimeError(
        f"Failed to get GPS fix from {device_path} within {timeout_seconds:.0f}s. Last status: {last_error}"
    )


def normalize_asset_name(label: str) -> str:
    return " ".join(token.capitalize() for token in label.split())


def _run_yolo_inference(image_path: str) -> tuple[str, float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(YOLO_INFER_SCRIPT):
        raise FileNotFoundError(f"YOLO inference script not found: {YOLO_INFER_SCRIPT}")
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model file not found: {YOLO_MODEL_PATH}")

    cmd = [
        sys.executable,
        YOLO_INFER_SCRIPT,
        "--image",
        image_path,
        "--model",
        YOLO_MODEL_PATH,
        "--conf",
        str(YOLO_CONFIDENCE),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "Unknown YOLO inference failure."
        raise RuntimeError(f"YOLO inference failed: {detail}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("YOLO inference returned empty output.")

    last_line = stdout.splitlines()[-1].strip()
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"YOLO inference returned non-JSON output: {last_line}") from exc

    label = str(payload.get("label", "")).strip()
    confidence = float(payload.get("confidence", 0.0))
    if not label:
        raise RuntimeError("YOLO inference did not return a predicted label.")
    return label, confidence


def _format_camera_open_error(camera_index: int) -> str:
    device_path = f"/dev/video{camera_index}"
    detail = [f"Could not open camera index {camera_index} (tried OpenCV index and {device_path})."]

    if os.path.exists(device_path):
        perms = ""
        try:
            perms = oct(os.stat(device_path).st_mode & 0o777)
        except Exception:
            perms = "unknown"
        rw = os.access(device_path, os.R_OK | os.W_OK)
        detail.append(f"Device exists with mode {perms}; read/write access={rw}.")
        try:
            proc = subprocess.run(
                ["lsof", device_path],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            holders = [line.strip() for line in proc.stdout.splitlines()[1:] if line.strip()]
            if holders:
                preview = "; ".join(holders[:3])
                if len(holders) > 3:
                    preview += "; ..."
                detail.append(f"Device appears busy. Current holder(s): {preview}")
        except FileNotFoundError:
            pass
        except Exception:
            pass
    else:
        available = ", ".join(sorted(glob.glob("/dev/video*")))
        if available:
            detail.append(f"{device_path} does not exist. Available devices: {available}")
        else:
            detail.append("No /dev/video* devices were found.")

    detail.append("Use a free camera device/index and retry.")
    return " ".join(detail)


def _open_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap

    cap.release()
    device_path = f"/dev/video{camera_index}"
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(device_path)
    if cap.isOpened():
        return cap

    cap.release()
    raise RuntimeError(_format_camera_open_error(camera_index))


@contextmanager
def _camera_session():
    acquired = CAMERA_LOCK.acquire(timeout=max(0.0, CAMERA_LOCK_TIMEOUT_SECONDS))
    if not acquired:
        raise RuntimeError(
            f"Camera is busy: another camera task is still running (waited {CAMERA_LOCK_TIMEOUT_SECONDS:.0f}s)."
        )
    try:
        yield
    finally:
        CAMERA_LOCK.release()


def capture_image(camera_index: int, delay_seconds: float) -> str:
    with _camera_session():
        cap = _open_camera(camera_index)

        try:
            end_time = time.time() + max(0.0, delay_seconds)
            while time.time() < end_time:
                time.sleep(0.1)

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera opened but failed to read frame.")

            os.makedirs(CAPTURE_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(CAPTURE_DIR, f"capture_{ts}.jpg")
            if not cv2.imwrite(image_path, frame):
                raise RuntimeError(f"Failed to save image: {image_path}")
            return image_path
        finally:
            cap.release()


def classify_image(image_path: str) -> tuple[str, float]:
    return _run_yolo_inference(image_path)


def _run_live_preview(camera_index: int) -> dict:
    if not os.path.exists(YOLO_LIVE_PREVIEW_SCRIPT):
        raise FileNotFoundError(f"YOLO live preview script not found: {YOLO_LIVE_PREVIEW_SCRIPT}")
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model file not found: {YOLO_MODEL_PATH}")

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    cmd = [
        sys.executable,
        YOLO_LIVE_PREVIEW_SCRIPT,
        "--model",
        YOLO_MODEL_PATH,
        "--camera",
        str(camera_index),
        "--conf",
        str(YOLO_CONFIDENCE),
        "--save-dir",
        CAPTURE_DIR,
    ]
    with _camera_session():
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "Unknown live preview failure."
        raise RuntimeError(f"Live YOLO preview failed: {detail}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("Live YOLO preview returned empty output.")

    last_line = stdout.splitlines()[-1].strip()
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Live YOLO preview returned non-JSON output: {last_line}") from exc
    return payload


def find_existing_plant_assets(farm, asset_name: str):
    response = farm.asset.get("plant", params={"filter[name]": asset_name, "page[limit]": 50})
    return response.get("data", []) if response else []


def find_asset_by_id(farm, asset_id: str):
    bundles = ["plant", "animal", "land", "structure", "equipment", "water", "input", "compost"]
    for bundle in bundles:
        try:
            response = farm.asset.get_id(bundle, asset_id)
            data = response.get("data") if response else None
            if data and data.get("id") == asset_id:
                return data
        except Exception:
            continue
    return None


def ensure_plant_type_term(farm, plant_type_name: str) -> Optional[str]:
    terms = farm.term.get("plant_type", params={"filter[name]": plant_type_name, "page[limit]": 1})
    if terms and terms.get("data"):
        return terms["data"][0]["id"]

    payload = {"type": "taxonomy_term--plant_type", "attributes": {"name": plant_type_name}}
    created = farm.term.send("plant_type", payload)
    return created.get("data", {}).get("id") or created.get("id")


def make_new_name(base_name: str) -> str:
    return f"{base_name} {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_unknown_name() -> str:
    return f"unknown {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def split_text_into_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    return paragraphs if paragraphs else [normalized]


def is_non_plant_response(text: str) -> bool:
    return bool(re.search(r"\b(not\s+(a|an)\s+plant|non[-\s]?plant|no\s+plant)\b", text, re.IGNORECASE))


def _coerce_response_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("response", "text", "content", "answer", "message"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def parse_multi_llm_responses(remote_text: str) -> list[str]:
    text = remote_text.strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        responses: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                value = _coerce_response_text(item)
                if value:
                    responses.append(value)
        elif isinstance(payload, dict):
            if isinstance(payload.get("responses"), list):
                for item in payload["responses"]:
                    value = _coerce_response_text(item)
                    if value:
                        responses.append(value)
            else:
                ordered_keys = sorted(payload.keys(), key=lambda k: str(k))
                for key in ordered_keys:
                    value = _coerce_response_text(payload[key])
                    if value:
                        responses.append(value)
        if responses:
            return responses
    except Exception:
        pass

    marker_matches = list(re.finditer(r"(^|\n)\s*(llm|model)\s*([1-9])\s*[:\-]\s*", text, flags=re.IGNORECASE))
    if marker_matches:
        responses = []
        for i, match in enumerate(marker_matches):
            start = match.end()
            end = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                responses.append(chunk)
        if responses:
            return responses

    return [text]


def extract_llm_paragraph_pairs(remote_text: str) -> list[tuple[str, str, str]]:
    """
    Returns up to 3 tuples: (paragraph1, paragraph2, full_llm_text).
    """
    def normalize_pair_order(p1: str, p2: str) -> tuple[str, str]:
        left = (p1 or "").strip()
        right = (p2 or "").strip()
        if not right:
            return left, right
        # Some models occasionally emit the model/header block as paragraph 2.
        # If that happens, swap so header-style block is paragraph 1.
        left_header = left.startswith("===")
        right_header = right.startswith("===")
        if right_header and not left_header:
            return right, left
        return left, right

    responses = parse_multi_llm_responses(remote_text)

    # Structured path: one response per LLM.
    if len(responses) >= 3:
        pairs: list[tuple[str, str, str]] = []
        for response_text in responses[:3]:
            paragraphs = split_text_into_paragraphs(response_text)
            p1 = paragraphs[0].strip() if paragraphs else response_text.strip()
            p2 = paragraphs[1].strip() if len(paragraphs) > 1 else ""
            p1, p2 = normalize_pair_order(p1, p2)
            if p1:
                full = p1 if not p2 else f"{p1}\n\n{p2}"
                pairs.append((p1, p2, full))
        return pairs

    # Flattened path: all content bundled into one text block.
    if len(responses) == 1:
        paragraphs = split_text_into_paragraphs(responses[0])
        if len(paragraphs) == 5 and is_non_plant_response(paragraphs[0]):
            # LLM-1 returned only one paragraph ("not a plant"),
            # LLM-2 and LLM-3 returned normal 2-paragraph responses.
            p1_2, p2_2 = normalize_pair_order(paragraphs[1].strip(), paragraphs[2].strip())
            p1_3, p2_3 = normalize_pair_order(paragraphs[3].strip(), paragraphs[4].strip())
            return [
                (paragraphs[0].strip(), "", paragraphs[0]),
                (p1_2, p2_2, p1_2 if not p2_2 else f"{p1_2}\n\n{p2_2}"),
                (p1_3, p2_3, p1_3 if not p2_3 else f"{p1_3}\n\n{p2_3}"),
            ]
        if len(paragraphs) >= 6:
            # Expected format: first 3 are paragraph-1 (LLM1..LLM3), next 3 are paragraph-2.
            p1_1, p2_1 = normalize_pair_order(paragraphs[0].strip(), paragraphs[3].strip())
            p1_2, p2_2 = normalize_pair_order(paragraphs[1].strip(), paragraphs[4].strip())
            p1_3, p2_3 = normalize_pair_order(paragraphs[2].strip(), paragraphs[5].strip())
            return [
                (p1_1, p2_1, p1_1 if not p2_1 else f"{p1_1}\n\n{p2_1}"),
                (p1_2, p2_2, p1_2 if not p2_2 else f"{p1_2}\n\n{p2_2}"),
                (p1_3, p2_3, p1_3 if not p2_3 else f"{p1_3}\n\n{p2_3}"),
            ]

        # Fallback: sequential paragraph pairs.
        pairs: list[tuple[str, str, str]] = []
        for idx in range(0, min(len(paragraphs), 6), 2):
            p1 = paragraphs[idx].strip()
            p2 = paragraphs[idx + 1].strip() if idx + 1 < len(paragraphs) else ""
            p1, p2 = normalize_pair_order(p1, p2)
            if p1:
                full_text = p1 if not p2 else f"{p1}\n\n{p2}"
                pairs.append((p1, p2, full_text))
        return pairs[:3]

    return []


def upsert_plant_asset(farm, asset_name: str, predicted_label: str, confidence: float, image_path: str, mode: str):
    global LAST_ASSET_ID, LAST_ASSET_TYPE, LAST_ASSET_NAME, LAST_ASSET_TS

    existing = find_existing_plant_assets(farm, asset_name)
    existing_id = existing[0]["id"] if existing else ""

    # Prevent "created then asked" only for immediate duplicate retries.
    recent_retry_window_s = 6.0
    now_ts = time.time()
    if (
        existing
        and mode == "ask"
        and LAST_ASSET_ID
        and LAST_ASSET_NAME == asset_name
        and LAST_ASSET_TS is not None
        and (now_ts - LAST_ASSET_TS) <= recent_retry_window_s
    ):
        for item in existing:
            if item.get("id") == LAST_ASSET_ID:
                return {
                    "status": "reused_recent",
                    "asset_id": LAST_ASSET_ID,
                    "asset_type": LAST_ASSET_TYPE or item.get("type") or "asset--plant",
                    "asset_name": asset_name,
                    "message": (
                        f"Prediction: {predicted_label} ({confidence:.2f}%). "
                        f"Reusing recently created asset '{asset_name}' (id: {LAST_ASSET_ID})."
                    ),
                }

    if existing and mode == "ask":
        return {
            "status": "needs_confirmation",
            "asset_name": asset_name,
            "existing_asset_id": existing_id,
            "message": (
                f"Prediction: {predicted_label} ({confidence:.2f}%). Asset '{asset_name}' already exists "
                f"(id: {existing_id}). Choose mode='overwrite' or mode='create_new'."
            ),
        }

    plant_type_term_id = ensure_plant_type_term(farm, normalize_asset_name(predicted_label))
    now_iso = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    note_lines = [
        f"Auto-classified from camera capture. Class={predicted_label}, "
        f"confidence={confidence:.2f}%, image={image_path}, timestamp={now_iso}"
    ]
    remote_doc_text = ""
    non_plant_detected = False
    if confidence < LOW_CONF_THRESHOLD_PERCENT:
        try:
            remote_text = request_low_confidence_text(
                image_path=image_path,
                server_host=LOW_CONF_SERVER_HOST,
                server_port=LOW_CONF_SERVER_PORT,
                timeout_ms=LOW_CONF_TIMEOUT_MS,
            )
            llm_pairs = extract_llm_paragraph_pairs(remote_text)
            doc_sections = []
            if llm_pairs and is_non_plant_response(llm_pairs[0][2]):
                non_plant_detected = True
            for idx, (first_paragraph, second_paragraph, _) in enumerate(llm_pairs, start=1):
                note_lines.append(
                    f"Low-confidence remote review LLM-{idx} (<{LOW_CONF_THRESHOLD_PERCENT:.2f}%), "
                    f"paragraph 1: {first_paragraph.strip()}"
                )
                if second_paragraph.strip():
                    doc_sections.append(f"LLM-{idx} paragraph 2:\n{second_paragraph.strip()}")
            if doc_sections:
                remote_doc_text = "\n\n".join(doc_sections)
        except Exception as exc:
            note_lines.append(
                f"Low-confidence remote review failed (<{LOW_CONF_THRESHOLD_PERCENT:.2f}%): {exc}"
            )
    if non_plant_detected:
        asset_name = make_unknown_name()
    note_text = "\n".join(note_lines)

    if existing and mode == "overwrite":
        payload = {
            "id": existing_id,
            "attributes": {
                "name": asset_name,
                "status": "active",
                "notes": {"value": note_text, "format": "plain_text"},
            },
            "relationships": {},
        }
        if plant_type_term_id:
            payload["relationships"]["plant_type"] = {
                "data": [{"type": "taxonomy_term--plant_type", "id": plant_type_term_id}]
            }
        response = farm.asset.send("plant", payload)
        asset_id = response.get("data", {}).get("id") or response.get("id") or existing_id
        doc_note = ""
        if remote_doc_text:
            try:
                log_id = upload_text_to_asset_activity_log(
                    farm=farm,
                    asset_id=asset_id,
                    asset_type="asset--plant",
                    text_content=remote_doc_text,
                    title=f"Low-confidence review paragraph 2 (all LLMs) ({predicted_label})",
                )
                doc_note = f" Added paragraph 2 as a document in activity log {log_id}."
            except Exception as exc:
                doc_note = f" Warning: paragraph 2 document upload failed: {exc}"
        LAST_ASSET_ID = asset_id
        LAST_ASSET_TYPE = "asset--plant"
        LAST_ASSET_NAME = asset_name
        LAST_ASSET_TS = time.time()
        return {
            "status": "updated",
            "asset_id": asset_id,
            "asset_type": "asset--plant",
            "asset_name": asset_name,
            "message": (
                f"Prediction: {predicted_label} ({confidence:.2f}%). "
                f"Overwrote asset '{asset_name}' (id: {asset_id}).{doc_note}"
            ),
        }

    if existing and mode == "create_new" and not non_plant_detected:
        asset_name = make_new_name(asset_name)

    payload = {
        "attributes": {
            "name": asset_name,
            "status": "active",
            "notes": {"value": note_text, "format": "plain_text"},
        },
        "relationships": {},
    }
    if plant_type_term_id:
        payload["relationships"]["plant_type"] = {
            "data": [{"type": "taxonomy_term--plant_type", "id": plant_type_term_id}]
        }

    response = farm.asset.send("plant", payload)
    asset_id = response.get("data", {}).get("id") or response.get("id")
    doc_note = ""
    if remote_doc_text:
        try:
            log_id = upload_text_to_asset_activity_log(
                farm=farm,
                asset_id=asset_id,
                asset_type="asset--plant",
                text_content=remote_doc_text,
                title=f"Low-confidence review paragraph 2 (all LLMs) ({predicted_label})",
            )
            doc_note = f" Added paragraph 2 as a document in activity log {log_id}."
        except Exception as exc:
            doc_note = f" Warning: paragraph 2 document upload failed: {exc}"
    LAST_ASSET_ID = asset_id
    LAST_ASSET_TYPE = "asset--plant"
    LAST_ASSET_NAME = asset_name
    LAST_ASSET_TS = time.time()

    action = "Created"
    if existing and mode == "create_new":
        action = "Created new"

    return {
        "status": "created",
        "asset_id": asset_id,
        "asset_type": "asset--plant",
        "asset_name": asset_name,
        "message": (
            f"Prediction: {predicted_label} ({confidence:.2f}%). "
            f"{action} asset '{asset_name}' (id: {asset_id}).{doc_note}"
        ),
    }


def create_or_update_plant_asset(farm, asset_name: str, predicted_label: str, confidence: float, image_path: str, mode: str) -> str:
    return upsert_plant_asset(farm, asset_name, predicted_label, confidence, image_path, mode)["message"]


def _annotate_frame_with_yolo(model, frame):
    label = ""
    confidence = 0.0
    annotated = frame.copy()
    results = model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
    result = results[0] if results else None
    if result is not None:
        annotated = result.plot()
        if result.boxes is not None and len(result.boxes) > 0:
            best_idx = int(result.boxes.conf.argmax().item())
            confidence = float(result.boxes.conf[best_idx].item() * 100.0)
            cls_id = int(result.boxes.cls[best_idx].item())
            label = str(model.names.get(cls_id, cls_id))
    return annotated, label, confidence


def capture_frames(camera_index: int, image_count: int, interval_seconds: float, start_delay_seconds: float):
    with _camera_session():
        cap = _open_camera(camera_index)

        os.makedirs(CAPTURE_DIR, exist_ok=True)
        paths = []
        model = None
        show_live = CAPTURE_LIVE_INFERENCE
        if show_live:
            try:
                from ultralytics import YOLO
                model = YOLO(YOLO_MODEL_PATH)
            except Exception as exc:
                raise RuntimeError(
                    f"Live capture inference enabled but YOLO model could not be loaded: {exc}"
                ) from exc
        try:
            if show_live:
                cv2.namedWindow(CAPTURE_LIVE_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            next_capture_delay = max(0.0, start_delay_seconds)
            for i in range(image_count):
                last_frame = None
                capture_at = time.time() + next_capture_delay
                while time.time() < capture_at:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Failed to read frame {i + 1}/{image_count}.")
                    last_frame = frame.copy()
                    if show_live:
                        overlay = frame.copy()
                        label = ""
                        confidence = 0.0
                        if model is not None:
                            overlay, label, confidence = _annotate_frame_with_yolo(model, frame)
                        remaining = max(0.0, capture_at - time.time())
                        cv2.putText(
                            overlay,
                            f"Capture {i + 1}/{image_count} in {remaining:.1f}s",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        info = f"YOLO: {label or 'No detection'} ({confidence:.2f}%)"
                        cv2.putText(
                            overlay,
                            info,
                            (10, overlay.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(CAPTURE_LIVE_WINDOW_TITLE, overlay)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            raise RuntimeError("Cancelled by user during live capture preview.")

                if last_frame is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Failed to read frame {i + 1}/{image_count}.")
                    last_frame = frame.copy()

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(CAPTURE_DIR, f"asset_img_{ts}_{i + 1}.jpg")
                if not cv2.imwrite(path, last_frame):
                    raise RuntimeError(f"Failed to save image: {path}")
                paths.append(path)
                next_capture_delay = max(0.0, interval_seconds)
        finally:
            cap.release()
            if show_live:
                cv2.destroyWindow(CAPTURE_LIVE_WINDOW_TITLE)
        return paths


def create_asset_log(farm, asset_id: str, asset_type: str, name: str):
    payload = {
        "attributes": {
            "name": name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            "status": "done",
        },
        "relationships": {"asset": {"data": [{"type": asset_type, "id": asset_id}]}},
    }
    response = farm.log.send("activity", payload)
    return response.get("data", {}).get("id") or response.get("id")


def upload_image_to_activity_log(farm, log_id: str, image_path: str):
    url = f"{HOSTNAME.rstrip('/')}/api/log/activity/{log_id}/image"
    with open(image_path, "rb") as f:
        data = f.read()
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": f'file; filename="{os.path.basename(image_path)}"',
    }
    response = farm.session.post(url, data=data, headers=headers)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed for {os.path.basename(image_path)}: {response.status_code} {response.text}")


def upload_file_to_activity_log(farm, log_id: str, file_path: str):
    url = f"{HOSTNAME.rstrip('/')}/api/log/activity/{log_id}/file"
    with open(file_path, "rb") as f:
        data = f.read()
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": f'file; filename="{os.path.basename(file_path)}"',
    }
    response = farm.session.post(url, data=data, headers=headers)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed for {os.path.basename(file_path)}: {response.status_code} {response.text}")


def upload_text_to_asset_activity_log(farm, asset_id: str, asset_type: str, text_content: str, title: str) -> str:
    log_id = create_asset_log(farm=farm, asset_id=asset_id, asset_type=asset_type, name=title)
    if not log_id:
        raise RuntimeError("Could not create activity log for text document upload.")

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = os.path.join(CAPTURE_DIR, f"low_conf_review_{ts}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_content.strip() + "\n")

    upload_file_to_activity_log(farm=farm, log_id=log_id, file_path=text_path)
    return log_id


def pixel_to_latlon(px, py, width, height, center_lat, center_lon, zoom):
    n = 2.0 ** zoom
    world_size_px = 512.0 * n

    center_px_x = (center_lon + 180.0) / 360.0 * world_size_px
    lat_rad = math.radians(center_lat)
    center_px_y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * world_size_px

    target_global_x = center_px_x - (width / 2.0) + px
    target_global_y = center_px_y - (height / 2.0) + py

    res_lon = (target_global_x / world_size_px) * 360.0 - 180.0
    n2 = math.pi - (2.0 * math.pi * target_global_y) / world_size_px
    res_lat = math.degrees(math.atan(0.5 * (math.exp(n2) - math.exp(-n2))))
    return res_lat, res_lon


def contour_to_wkt(contour, width, height, center_lat, center_lon, zoom):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    if len(approx) < 3:
        return None

    points = []
    for point in approx:
        px, py = point[0]
        lat, lon = pixel_to_latlon(px, py, width, height, center_lat, center_lon, zoom)
        points.append(f"{lon} {lat}")
    points.append(points[0])
    return f"POLYGON (({', '.join(points)}))"


def red_mask_from_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 40, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_largest_red_contour(segmented_image_path: str):
    img = cv2.imread(segmented_image_path)
    if img is None:
        raise RuntimeError(f"Could not read segmented image: {segmented_image_path}")

    mask = red_mask_from_image(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No red segmented region found in image.")
    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    if area_px < 20:
        raise RuntimeError("Segmented region too small to build geometry.")
    return contour, area_px, img.shape[1], img.shape[0]


def crop_segmented_region(segmented_image_path: str):
    img = cv2.imread(segmented_image_path)
    if img is None:
        raise RuntimeError(f"Could not read segmented image: {segmented_image_path}")
    mask = red_mask_from_image(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No segmented region to crop.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    crop = img[y : y + h, x : x + w]
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    out_path = os.path.join(
        SEGMENTED_DIR,
        f"cropped_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(segmented_image_path)}",
    )
    cv2.imwrite(out_path, crop)
    return out_path


def update_asset_geometry(farm, asset_id: str, asset_type: str, wkt_geometry: str):
    bundle = asset_type.split("--", 1)[1] if "--" in asset_type else asset_type
    payload = {
        "id": asset_id,
        "attributes": {
            "intrinsic_geometry": {"value": wkt_geometry},
            "is_location": True,
            "is_fixed": True,
            "status": "active",
        },
    }
    response = farm.asset.send(bundle, payload)
    return response.get("data", {}).get("id") or response.get("id") or asset_id


def download_mapbox_satellite_image(latitude, longitude, zoom, width, height):
    if not MAPBOX_TOKEN:
        raise RuntimeError("MAPBOX_TOKEN is not configured.")
    os.makedirs(MAPBOX_INPUT_DIR, exist_ok=True)
    url = (
        "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{longitude},{latitude},{zoom}/{width}x{height}?access_token={MAPBOX_TOKEN}"
    )
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Mapbox API error: {response.status_code} {response.text}")
    path = os.path.join(
        MAPBOX_INPUT_DIR,
        f"mapbox_{latitude:.6f}_{longitude:.6f}_z{zoom}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def _extract_segmented_output_name(status_message: str, input_image_path: str) -> str:
    match = re.search(r"segmented as ([^\s]+)", status_message, flags=re.IGNORECASE)
    if match:
        return os.path.basename(match.group(1))
    return f"segmented_{os.path.basename(input_image_path)}"


def send_image_for_segmentation(
    input_image_path: str,
    server_host: str,
    server_port: int,
    timeout_ms: int = 30000,
    retries: int = 2,
):
    with open(input_image_path, "rb") as f:
        image_data = f.read()
    filename = os.path.basename(input_image_path)
    context = zmq.Context()
    try:
        last_error = "No reply from segmentation server."
        for attempt in range(1, retries + 2):
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            try:
                socket.connect(f"tcp://{server_host}:{server_port}")
                socket.send_multipart([filename.encode("utf-8"), image_data])

                poller = zmq.Poller()
                poller.register(socket, zmq.POLLIN)
                if not poller.poll(timeout_ms):
                    last_error = (
                        f"Timeout waiting for segmentation reply on attempt {attempt} "
                        f"({timeout_ms} ms)."
                    )
                    continue

                parts = socket.recv_multipart()
                if not parts:
                    last_error = "Segmentation server returned an empty response."
                    continue

                status = parts[0].decode("utf-8", errors="replace")
                if len(parts) < 2 or not parts[1]:
                    last_error = f"Segmentation server returned no image. Message: {status}"
                    continue

                os.makedirs(SEGMENTED_DIR, exist_ok=True)
                output_name = _extract_segmented_output_name(status, input_image_path)
                out_path = os.path.join(SEGMENTED_DIR, output_name)
                with open(out_path, "wb") as f:
                    f.write(parts[1])
                return out_path, status
            except Exception as exc:
                last_error = f"Segmentation attempt {attempt} failed: {exc}"
            finally:
                socket.close()

        raise RuntimeError(last_error)
    finally:
        context.term()


def request_low_confidence_text(
    image_path: str,
    server_host: str,
    server_port: int,
    timeout_ms: int = 20000,
) -> str:
    with open(image_path, "rb") as f:
        image_data = f.read()

    filename = os.path.basename(image_path)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.IMMEDIATE, 1)
    try:
        socket.connect(f"tcp://{server_host}:{server_port}")
        socket.send_multipart([filename.encode("utf-8"), image_data])
        parts = socket.recv_multipart()
        if not parts:
            raise RuntimeError("Low-confidence server returned an empty response.")
        return parts[0].decode("utf-8", errors="replace").strip() or "Low-confidence server returned blank text."
    except zmq.Again as exc:
        raise RuntimeError(
            f"Timeout waiting for low-confidence server reply from {server_host}:{server_port} "
            f"after {timeout_ms} ms."
        ) from exc
    finally:
        socket.close(0)
        context.term()


def collect_and_upload_for_asset(
    asset_id,
    asset_type,
    camera_index,
    interval_seconds,
    duration_seconds,
    upload_to_activity_log: bool = False,
):
    interval_seconds = max(0.1, interval_seconds)
    duration_seconds = max(1.0, duration_seconds)
    image_count = max(1, int(duration_seconds / interval_seconds))

    images = capture_frames(
        camera_index=camera_index,
        image_count=image_count,
        interval_seconds=interval_seconds,
        start_delay_seconds=0.0,
    )
    if not upload_to_activity_log:
        return {"log_id": "", "captured": len(images), "uploaded": 0, "saved_paths": images}

    farm = get_client()
    log_id = create_asset_log(
        farm=farm,
        asset_id=asset_id,
        asset_type=asset_type,
        name=f"Classifier Camera Images ({len(images)} imgs / {duration_seconds:.0f}s)",
    )
    if not log_id:
        raise RuntimeError("Failed to create activity log for image uploads.")

    uploaded = 0
    for image_path in images:
        upload_image_to_activity_log(farm=farm, log_id=log_id, image_path=image_path)
        uploaded += 1

    return {"log_id": log_id, "captured": len(images), "uploaded": uploaded, "saved_paths": images}


def run_mapbox_segment_and_apply_geometry(asset_id, asset_type, latitude, longitude, zoom, width, height, server_host, server_port):
    farm = get_client()
    mapbox_path = download_mapbox_satellite_image(latitude, longitude, zoom, width, height)
    segmented_path, segment_status = send_image_for_segmentation(mapbox_path, server_host, server_port)
    cropped_path = crop_segmented_region(segmented_path)

    contour, area_px, img_w, img_h = extract_largest_red_contour(segmented_path)
    wkt = contour_to_wkt(contour, img_w, img_h, latitude, longitude, zoom)
    if not wkt:
        raise RuntimeError("Failed to build polygon from segmented contour.")

    updated_id = update_asset_geometry(farm, asset_id, asset_type, wkt)
    return {
        "asset_id": updated_id,
        "mapbox_path": mapbox_path,
        "segmented_path": segmented_path,
        "cropped_path": cropped_path,
        "contour_area_px": area_px,
        "segment_status": segment_status,
    }


@mcp.tool()
def get_server_info() -> str:
    farm = get_client()
    info = farm.info()
    farm_meta = info.get("meta", {}).get("farm", {}) if isinstance(info, dict) else {}
    farm_name = farm_meta.get("name", "unknown")
    farm_version = farm_meta.get("version", "unknown")
    farm_url = farm_meta.get("url", HOSTNAME)
    return f"Connected to {farm_name} ({farm_version}) at {farm_url}"


@mcp.tool()
def classify_camera_and_sync_asset(
    camera_index: int = 0,
    delay_seconds: float = 5.0,
    mode: str = "ask",
    asset_name_override: str = "",
) -> str:
    mode = mode.strip().lower()
    if mode not in {"ask", "overwrite", "create_new"}:
        return "Error: mode must be one of: ask, overwrite, create_new."

    try:
        effective_mode = mode
        image_path = ""
        predicted_label = ""
        confidence = 0.0

        preview_warning = ""
        if mode == "ask" and YOLO_LIVE_PREVIEW_ON_ASK:
            try:
                preview = _run_live_preview(camera_index=camera_index)
                if preview.get("status") == "cancelled":
                    return "Cancelled by user during live YOLO preview."
                effective_mode = str(preview.get("mode", "ask")).strip().lower() or "ask"
                if effective_mode not in {"ask", "overwrite", "create_new"}:
                    effective_mode = "ask"
                image_path = str(preview.get("image_path", "")).strip()
                predicted_label = str(preview.get("label", "")).strip()
                confidence = float(preview.get("confidence", 0.0))
                if not image_path or not predicted_label:
                    raise RuntimeError("Live YOLO preview did not return image_path/label.")
            except Exception as preview_exc:
                preview_warning = (
                    "Live preview unavailable; continued with snapshot classification. "
                    f"Reason: {preview_exc}"
                )
                image_path = capture_image(camera_index=camera_index, delay_seconds=delay_seconds)
                predicted_label, confidence = classify_image(image_path)
        else:
            image_path = capture_image(camera_index=camera_index, delay_seconds=delay_seconds)
            predicted_label, confidence = classify_image(image_path)

        asset_name = asset_name_override.strip() or normalize_asset_name(predicted_label)
        farm = get_client()
        result = create_or_update_plant_asset(
            farm=farm,
            asset_name=asset_name,
            predicted_label=predicted_label,
            confidence=confidence,
            image_path=image_path,
            mode=effective_mode,
        )
        if preview_warning:
            return f"{preview_warning}\n{result}"
        return result
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def capture_and_upload_asset_images(
    image_count: int = 5,
    interval_seconds: float = 3.0,
    camera_index: int = 0,
    start_delay_seconds: float = 0.0,
    asset_id: str = "",
    force_new_log: bool = False,
) -> str:
    global LAST_ASSET_ID, LAST_ASSET_TYPE, LAST_CAPTURE_LOG_ASSET_ID, LAST_CAPTURE_LOG_ID, LAST_CAPTURE_LOG_TS
    if image_count <= 0:
        return "Error: image_count must be > 0."

    try:
        farm = get_client()
        target_asset_id = asset_id.strip() or LAST_ASSET_ID
        target_asset_type = LAST_ASSET_TYPE
        if not target_asset_id:
            return "Error: No target asset available. Run classify_camera_and_sync_asset first or pass asset_id."

        # Prevent repeated uploads when clients accidentally re-call this tool in a loop.
        if (
            not force_new_log
            and LAST_CAPTURE_LOG_ASSET_ID == target_asset_id
            and LAST_CAPTURE_LOG_ID
        ):
            return (
                f"Images were already captured and uploaded for asset {target_asset_id} in this session "
                f"(activity log {LAST_CAPTURE_LOG_ID}). Skipping duplicate upload. "
                "Set force_new_log=true to capture and upload again."
            )

        if not target_asset_type:
            asset_data = find_asset_by_id(farm, target_asset_id)
            if not asset_data:
                return f"Error: Asset not found for id {target_asset_id}."
            target_asset_type = asset_data.get("type")

        os.makedirs(CAPTURE_DIR, exist_ok=True)
        images = capture_frames(
            camera_index=camera_index,
            image_count=image_count,
            interval_seconds=max(0.0, interval_seconds),
            start_delay_seconds=max(0.0, start_delay_seconds),
        )

        log_id = create_asset_log(
            farm=farm,
            asset_id=target_asset_id,
            asset_type=target_asset_type,
            name=f"Classifier Camera Images ({len(images)})",
        )
        if not log_id:
            return "Error: Could not create activity log."

        uploaded = 0
        for path in images:
            upload_image_to_activity_log(farm=farm, log_id=log_id, image_path=path)
            uploaded += 1

        LAST_CAPTURE_LOG_ASSET_ID = target_asset_id
        LAST_CAPTURE_LOG_ID = log_id
        LAST_CAPTURE_LOG_TS = time.time()

        return (
            f"Captured {len(images)} images and uploaded {uploaded} to activity log {log_id}, "
            f"linked to asset {target_asset_id}."
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def apply_segmented_image_geometry_to_asset(
    segmented_image_path: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    zoom: int = 18,
    asset_id: str = "",
    create_new_if_missing: bool = True,
    new_asset_name: str = "",
    land_type: str = "bed",
) -> str:
    global LAST_ASSET_ID, LAST_ASSET_TYPE
    try:
        if latitude is None or longitude is None:
            latitude, longitude = read_gps_coordinates()
        farm = get_client()
        contour, area_px, width, height = extract_largest_red_contour(segmented_image_path)
        wkt = contour_to_wkt(contour, width, height, latitude, longitude, zoom)
        if not wkt:
            return "Error: Failed to build polygon from segmented contour."

        target_asset_id = asset_id.strip() or LAST_ASSET_ID
        target_asset_type = LAST_ASSET_TYPE

        if target_asset_id:
            if not target_asset_type:
                asset_data = find_asset_by_id(farm, target_asset_id)
                if not asset_data:
                    return f"Error: Asset not found for id {target_asset_id}."
                target_asset_type = asset_data.get("type")

            updated_id = update_asset_geometry(farm, target_asset_id, target_asset_type, wkt)
            LAST_ASSET_ID = updated_id
            LAST_ASSET_TYPE = target_asset_type
            return (
                f"Updated asset {updated_id} geometry from segmented image. "
                f"Contour area: {area_px:.1f}px. Zoom: {zoom}."
            )

        if not create_new_if_missing:
            return "Error: No target asset available. Provide asset_id or set create_new_if_missing=True."

        asset_name = new_asset_name.strip() or f"Segmented Field {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        payload = {
            "attributes": {
                "name": asset_name,
                "status": "active",
                "land_type": land_type,
                "is_location": True,
                "intrinsic_geometry": {"value": wkt},
            }
        }
        response = farm.asset.send("land", payload)
        created_id = response.get("data", {}).get("id") or response.get("id")
        LAST_ASSET_ID = created_id
        LAST_ASSET_TYPE = "asset--land"
        return (
            f"Created land asset '{asset_name}' (id: {created_id}) with segmented polygon geometry. "
            f"Contour area: {area_px:.1f}px. Zoom: {zoom}."
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def collect_data_with_coordinates(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    mode: str = "ask",
    asset_name_override: str = "",
    camera_index: int = 0,
    classifier_delay_seconds: float = 5.0,
    collection_interval_seconds: float = 3.0,
    collection_duration_seconds: float = 15.0,
    mapbox_zoom: int = 18,
    mapbox_width: int = 800,
    mapbox_height: int = 800,
    segment_server_host: str = "",
    segment_server_port: int = 0,
    upload_to_activity_log: bool = False,
) -> str:
    """
    End-to-end flow:
    1) Capture and classify plant asset, ask on conflict if mode='ask'.
    2) Create/overwrite asset when allowed.
    3) Run in parallel:
       - Mapbox download -> remote segmentation -> crop -> WKT -> asset geometry update.
       - Camera image collection every N seconds for duration (local only by default).
         Set upload_to_activity_log=True to upload captured images to a single activity log.
    """
    mode = mode.strip().lower()
    if mode not in {"ask", "overwrite", "create_new"}:
        return "Error: mode must be one of: ask, overwrite, create_new."

    host = segment_server_host.strip() or SEGMENT_SERVER_HOST
    port = segment_server_port if segment_server_port > 0 else SEGMENT_SERVER_PORT

    try:
        if latitude is None or longitude is None:
            latitude, longitude = read_gps_coordinates()

        effective_mode = mode
        image_path = ""
        predicted_label = ""
        confidence = 0.0
        preview_warning = ""
        if mode == "ask" and YOLO_LIVE_PREVIEW_ON_ASK:
            try:
                preview = _run_live_preview(camera_index=camera_index)
                if preview.get("status") == "cancelled":
                    return "Cancelled by user during live YOLO preview."
                effective_mode = str(preview.get("mode", "ask")).strip().lower() or "ask"
                if effective_mode not in {"ask", "overwrite", "create_new"}:
                    effective_mode = "ask"
                image_path = str(preview.get("image_path", "")).strip()
                predicted_label = str(preview.get("label", "")).strip()
                confidence = float(preview.get("confidence", 0.0))
                if not image_path or not predicted_label:
                    raise RuntimeError("Live YOLO preview did not return image_path/label.")
            except Exception as preview_exc:
                preview_warning = (
                    "Live preview unavailable; continued with snapshot classification. "
                    f"Reason: {preview_exc}"
                )
                image_path = capture_image(camera_index=camera_index, delay_seconds=classifier_delay_seconds)
                predicted_label, confidence = classify_image(image_path)
        else:
            image_path = capture_image(camera_index=camera_index, delay_seconds=classifier_delay_seconds)
            predicted_label, confidence = classify_image(image_path)

        asset_name = asset_name_override.strip() or normalize_asset_name(predicted_label)

        farm = get_client()
        upsert_result = upsert_plant_asset(
            farm=farm,
            asset_name=asset_name,
            predicted_label=predicted_label,
            confidence=confidence,
            image_path=image_path,
            mode=effective_mode,
        )

        if upsert_result["status"] == "needs_confirmation":
            return upsert_result["message"]

        asset_id = upsert_result["asset_id"]
        asset_type = upsert_result.get("asset_type", "asset--plant")

        results = {"images": None, "geometry": None}
        errors = []

        def image_worker():
            try:
                results["images"] = collect_and_upload_for_asset(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    camera_index=camera_index,
                    interval_seconds=collection_interval_seconds,
                    duration_seconds=collection_duration_seconds,
                    upload_to_activity_log=upload_to_activity_log,
                )
            except Exception as exc:
                errors.append(f"Image collection failed: {exc}")

        def geometry_worker():
            try:
                results["geometry"] = run_mapbox_segment_and_apply_geometry(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    latitude=latitude,
                    longitude=longitude,
                    zoom=mapbox_zoom,
                    width=mapbox_width,
                    height=mapbox_height,
                    server_host=host,
                    server_port=port,
                )
            except Exception as exc:
                errors.append(f"Geometry pipeline failed: {exc}")

        t1 = threading.Thread(target=image_worker, daemon=True)
        t2 = threading.Thread(target=geometry_worker, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        lines = [upsert_result["message"]]
        if preview_warning:
            lines.insert(0, preview_warning)

        if results["geometry"]:
            g = results["geometry"]
            lines.append(
                f"Geometry updated on asset {g['asset_id']} from segmented map. "
                f"Mapbox: {g['mapbox_path']} Segmented: {g['segmented_path']} Cropped: {g['cropped_path']}"
            )

        if results["images"]:
            i = results["images"]
            if i.get("log_id"):
                lines.append(
                    f"Captured {i['captured']} images and uploaded {i['uploaded']} to activity log {i['log_id']}."
                )
            else:
                lines.append(
                    f"Captured {i['captured']} images locally in {CAPTURE_DIR}. "
                    "No activity log updates were made."
                )

        if errors:
            lines.append("Warnings: " + " | ".join(errors))

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
