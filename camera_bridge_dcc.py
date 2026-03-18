# camera_bridge_dcc.py - digiCamControl (USB) edition
import os
import time
import logging
import traceback
import shutil
import argparse
import requests
import cv2
import numpy as np

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


# ---------------------------------------------------------------------------
# Image processing pipeline for graded card in a lightbox
# ---------------------------------------------------------------------------

def _order_corners(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _detect_card_contour(img):
    """
    Find the largest 4-sided contour in the image (the card slab).
    Uses CLAHE for contrast enhancement and tries multiple Canny thresholds
    and polygon epsilon values for robustness in lightbox conditions.
    Returns a (4,2) float32 array of corner points, or None if not found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE boosts local contrast — critical for even lightbox lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    best_corners = None
    best_area = 0
    min_area = w * h * 0.15  # slab should cover at least 15% of frame

    kernel = np.ones((5, 5), np.uint8)
    for blur_k in [5, 9]:
        blurred = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
        for t1, t2 in [(30, 90), (50, 150), (80, 220)]:
            edges = cv2.Canny(blurred, t1, t2)
            edges = cv2.dilate(edges, kernel, iterations=3)
            edges = cv2.erode(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < min_area or area <= best_area:
                    break
                peri = cv2.arcLength(cnt, True)
                for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4:
                        best_corners = approx.reshape(4, 2).astype("float32")
                        best_area = area
                        break  # found a quad for this contour, move on

    if best_corners is not None:
        log.info(f"Card contour found, area={best_area:.0f}")
    return best_corners


def _expand_corners(corners, margin):
    """Push each corner outward from the center by `margin` pixels."""
    center = corners.mean(axis=0)
    expanded = []
    for pt in corners:
        direction = pt - center
        norm = np.linalg.norm(direction)
        expanded.append(pt + direction / norm * margin if norm > 0 else pt)
    return np.array(expanded, dtype="float32")


def _deskew_fallback(img):
    """
    When full quad detection fails, use Otsu threshold + minAreaRect to detect
    the card's tilt and rotate the image to straighten it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return img  # negligible tilt, don't touch it
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    log.info(f"Deskew fallback: correcting {angle:.1f}°")
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _perspective_crop(img, corners, margin=20):
    """Perspective-transform the image so the card fills the frame.
    `margin` expands the source corners outward so the card edge isn't clipped.
    """
    corners = _expand_corners(corners, margin)
    rect = _order_corners(corners)
    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))
    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst_pts)
    return cv2.warpPerspective(img, M, (width, height))


def _sharpen(img):
    """Unsharp mask: brings out edge detail and text on the card label."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1.7, blurred, -0.7, 0)


def process_card_image(src_path: str, dst_path: str, rotate_only: bool = False):
    """
    Full processing pipeline:
      1. Rotate 90° left (counterclockwise)
      2. Detect card slab and perspective-crop  (skipped when rotate_only=True)
      3. Sharpen                                (skipped when rotate_only=True)
    Falls back to a plain file copy if anything goes wrong.
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            raise ValueError(f"cv2 could not open {src_path}")

        # 1. Rotate 90° left
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        log.info("Rotated 90° left")

        if not rotate_only:
            # 2. Detect card and crop
            corners = _detect_card_contour(img)
            if corners is not None:
                img = _perspective_crop(img, corners)
                log.info("Card detected and cropped")
            else:
                log.warning("Card quad not detected — attempting deskew fallback")
                img = _deskew_fallback(img)

            # 3. Sharpen
            img = _sharpen(img)
        else:
            log.info("rotate-only mode: skipping detection and sharpening")

        cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.info(f"Processed image saved: {dst_path}")

    except Exception:
        log.error(f"Image processing failed, falling back to plain copy:\n{traceback.format_exc()}")
        shutil.copy2(src_path, dst_path)


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
                process_card_image(src, dst, rotate_only=ROTATE_ONLY)
                return dst
            time.sleep(0.3)

        raise RuntimeError("No new image appeared after 30s")

    def status(self) -> dict:
        try:
            resp = requests.get(f"{DCC_URL}/", timeout=5)
            return {"ready": resp.ok, "battery": None}
        except Exception:
            return {"ready": False, "battery": None}


parser = argparse.ArgumentParser(description="OmniVault camera bridge")
parser.add_argument(
    "--rotate-only",
    action="store_true",
    help="Only rotate 90° left — skip card detection, crop, and sharpening",
)
args, _ = parser.parse_known_args()  # parse_known_args so uvicorn args don't cause errors
ROTATE_ONLY: bool = args.rotate_only
if ROTATE_ONLY:
    log.info("Starting in rotate-only mode")

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
    return {"status": "ok", "camera_connected": camera.connected, "rotate_only": ROTATE_ONLY}


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
