# camera_bridge_dcc.py - digiCamControl (USB) edition
import os
import time
import logging
import traceback
import shutil
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

CAPTURE_DIR = os.path.join(os.getcwd(), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# digiCamControl HTTP server (enable in DCC: Tools > Settings > Webserver, port 5513)
DCC_URL = os.environ.get("DCC_URL", "http://localhost:5513")

# Folder where digiCamControl saves photos (DCC: Session > Session folder)
# Override with DCC_SAVE_DIR env var to match your DCC session folder
DCC_SAVE_DIR = os.environ.get("DCC_SAVE_DIR",
    r"C:\Users\OmniV\OneDrive\Pictures\digiCamControl\Session1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camerabridge")


def get_jpg_files(folder):
    result = set()
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg')):
            result.add(os.path.join(folder, f))
    return result


class DigiCamControlCamera:
    def __init__(self):
        self.connected = False

    def connect(self):
        resp = requests.get(f"{DCC_URL}/", timeout=5)
        resp.raise_for_status()
        self.connected = True
        log.info(f"Connected to digiCamControl at {DCC_URL}")
        log.info(f"Watching folder: {DCC_SAVE_DIR}")

    def capture(self) -> str:
        # Baseline snapshot
        before = get_jpg_files(DCC_SAVE_DIR)
        log.info(f"Baseline: {len(before)} images")

        # Trigger capture - no AF (manual focus lens)
        resp = requests.get(f"{DCC_URL}/?CMD=Capture", timeout=15)
        resp.raise_for_status()
        log.info("Shutter triggered")

        # Poll for new file
        deadline = time.time() + 30
        while time.time() < deadline:
            after = get_jpg_files(DCC_SAVE_DIR)
            new_files = after - before
            if new_files:
                src = max(new_files, key=os.path.getmtime)
                log.info(f"New image: {src}")
                ts = time.strftime("%Y%m%d-%H%M%S")
                dst = os.path.join(CAPTURE_DIR, f"IMG_{ts}.jpg")
                shutil.copy2(src, dst)
                log.info(f"Saved to {dst}")
                return dst
            time.sleep(0.3)

        raise RuntimeError("No new image appeared after 30s")

    def status(self) -> dict:
        try:
            resp = requests.get(f"{DCC_URL}/", timeout=5)
            return {"ready": resp.ok, "battery": None}
        except Exception:
            return {"ready": False, "battery": None}


camera = DigiCamControlCamera()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    try:
        camera.connect()
    except Exception as e:
        log.error(f"Camera connect failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "camera_connected": camera.connected}


@app.post("/capture")
def capture():
    try:
        path = camera.capture()
        return FileResponse(path, media_type="image/jpeg", filename=os.path.basename(path))
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.get("/camera/status")
def camera_status():
    try:
        return camera.status()
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7777)
