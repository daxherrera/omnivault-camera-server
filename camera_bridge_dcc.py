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

# Image processing tuning (hardcoded so no env vars are needed)
CROP_MARGIN_PX = 20
MIN_CROP_AREA_RATIO = 0.08
BLACK_BG_DELTA = 22
BRIGHT_SLAB_PERCENTILE = 88
BRIGHT_SLAB_DELTA = 8
BRIGHT_SLAB_PAD_X_RATIO = 0.28
BRIGHT_SLAB_PAD_Y_RATIO = 0.18
EDGE_DIFF_THRESHOLD = 24.0
EDGE_HIT_RATIO = 0.12
EDGE_SCAN_BAND = 4
EDGE_SCAN_MAX_RATIO = 0.45
EDGE_SCAN_MARGIN_PX = 10
WHITE_DESKEW_MIN_DEG = 0.3
WHITE_DESKEW_MAX_DEG = 15.0
LINE_DESKEW_MIN_DEG = 0.25
LINE_DESKEW_MAX_DEG = 12.0
LINE_DESKEW_HOUGH_THRESHOLD = 70
ENABLE_COLOR_CORRECTION = False
SATURATION_GAIN = 1.05
CLAHE_CLIP_LIMIT = 1.25
GAMMA = 1.0
ENABLE_DEGLARE = False
DEGLARE_VALUE_THRESH = 242
DEGLARE_SAT_THRESH = 35
DEGLARE_MAX_MASK_RATIO = 0.14
DEGLARE_INPAINT_RADIUS = 1.2
COLOR_VIBRANCE = 0.08
HIGHLIGHT_ROLL_OFF = 6.0
ENABLE_LIGHT_COLOR_BOOST = True
LIGHT_COLOR_SAT_GAIN = 1.10
LIGHT_COLOR_VAL_GAIN = 1.03

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


def _wait_for_file_ready(path, timeout=10.0, stable_checks=3, poll=0.2):
    """Wait until file exists, has non-zero size, and size is stable across checks."""
    deadline = time.time() + timeout
    last_size = -1
    stable_count = 0

    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= stable_checks:
                    return True
            else:
                stable_count = 0
            last_size = size
        except OSError:
            stable_count = 0
            last_size = -1

        time.sleep(poll)

    return False


def _read_image_with_retry(path, timeout=8.0, poll=0.25):
    """Retry cv2 image load while camera software is still flushing file writes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        img = cv2.imread(path)
        if img is not None and img.size > 0:
            return img
        time.sleep(poll)
    return None


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


def _detect_card_on_black_background(img):
    """
    Black-foam-core specific detector.
    Segments non-black foreground and uses minAreaRect to get a robust 4-corner box.
    Returns a (4,2) float32 array of points or None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu picks a split between dark background and brighter slab/card foreground.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < (w * h * 0.08):
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    log.info(f"Black-bg detector box found, area={area:.0f}")
    return box.astype("float32")


def _foreground_mask_from_black_background(img):
    """
    Build a mask for the slab/card foreground when background is black foam core.
    Threshold is derived from border pixels (where background is expected).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bw = max(10, int(w * 0.06))
    bh = max(10, int(h * 0.06))

    border = np.concatenate([
        gray[:bh, :].ravel(),
        gray[h - bh:, :].ravel(),
        gray[:, :bw].ravel(),
        gray[:, w - bw:].ravel(),
    ])

    # Use a robust bright quantile from borders as black-bg estimate, then offset.
    bg_level = int(np.percentile(border, 85))
    threshold = min(255, bg_level + BLACK_BG_DELTA)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _crop_non_black_foreground(img, margin=CROP_MARGIN_PX):
    """
    Crop to the largest non-black connected foreground region.
    Returns (cropped_img, True) on success, else (original_img, False).
    """
    h, w = img.shape[:2]
    mask = _foreground_mask_from_black_background(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < (w * h * MIN_CROP_AREA_RATIO):
        return img, False

    x, y, cw, ch = cv2.boundingRect(largest)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + cw + margin)
    y1 = min(h, y + ch + margin)

    if (x1 - x0) < 50 or (y1 - y0) < 50:
        return img, False

    log.info(
        f"Black-bg bbox crop: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}, area={area:.0f}"
    )
    return img[y0:y1, x0:x1], True


def _crop_bright_slab_foreground(img, margin=CROP_MARGIN_PX):
    """
    Background-agnostic crop tuned for white slabs.
    Uses LAB lightness to isolate bright slab region and crops to the largest
    plausible connected component.
    """
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    light = cv2.GaussianBlur(light, (5, 5), 0)

    # Adaptive threshold from image brightness distribution.
    bright_cutoff = int(np.percentile(light, BRIGHT_SLAB_PERCENTILE))
    thresh_val = max(30, min(250, bright_cutoff - BRIGHT_SLAB_DELTA))
    _, mask = cv2.threshold(light, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False

    min_area = w * h * MIN_CROP_AREA_RATIO
    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 50 or ch < 50:
            continue

        rect_area = float(cw * ch)
        fill_ratio = area / rect_area if rect_area > 0 else 0

        # Prefer large, rectangular-ish components.
        score = area * (0.5 + min(fill_ratio, 1.0))
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch, area)

    if best is None:
        return img, False

    x, y, cw, ch, area = best

    # White detection can lock onto the card face; pad aggressively to retain slab edges/label.
    pad_x = max(margin, int(cw * BRIGHT_SLAB_PAD_X_RATIO))
    pad_y = max(margin, int(ch * BRIGHT_SLAB_PAD_Y_RATIO))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)

    if (x1 - x0) < 50 or (y1 - y0) < 50:
        return img, False

    log.info(
        f"Bright-slab bbox crop: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}, "
        f"area={area:.0f}, thresh={thresh_val}, pad_x={pad_x}, pad_y={pad_y}"
    )
    return img[y0:y1, x0:x1], True


def _estimate_white_slab_angle(img):
    """Estimate slab tilt angle from bright/white slab region using minAreaRect."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    light = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)

    bright_cutoff = int(np.percentile(light, BRIGHT_SLAB_PERCENTILE))
    thresh_val = max(30, min(250, bright_cutoff - BRIGHT_SLAB_DELTA))
    _, mask = cv2.threshold(light, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = w * h * MIN_CROP_AREA_RATIO
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        rect_area = float(cw * ch)
        fill_ratio = area / rect_area if rect_area > 0 else 0
        score = area * (0.5 + min(fill_ratio, 1.0))
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None

    (_, _), (rw, rh), angle = cv2.minAreaRect(best)
    if rw < rh:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0
    return float(angle)


def _deskew_from_white_slab(img):
    """Deskew image by rotating opposite of white slab angle estimate."""
    angle = _estimate_white_slab_angle(img)
    if angle is None:
        return img, False

    abs_angle = abs(angle)
    if abs_angle < WHITE_DESKEW_MIN_DEG or abs_angle > WHITE_DESKEW_MAX_DEG:
        return img, False

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    log.info(f"White-slab deskew: corrected {angle:.2f}°")
    return deskewed, True


def _estimate_skew_from_lines(img):
    """
    Estimate skew angle from dominant long edges.
    Returns angle in degrees (positive means CCW), or None if no stable estimate.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    h, w = img.shape[:2]
    min_line_len = max(40, int(min(h, w) * 0.35))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=LINE_DESKEW_HOUGH_THRESHOLD,
        minLineLength=min_line_len,
        maxLineGap=20,
    )
    if lines is None:
        return None

    weighted_sum = 0.0
    total_weight = 0.0

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line_len:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Normalize to nearest axis deviation in [-45, 45]
        if angle > 45.0:
            angle -= 90.0
        elif angle < -45.0:
            angle += 90.0

        # Keep mostly horizontal/vertical edges only.
        if abs(angle) > 20.0:
            continue

        weighted_sum += angle * length
        total_weight += length

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


def _deskew_from_lines(img):
    """Deskew using line-based angle estimate on the cropped slab image."""
    angle = _estimate_skew_from_lines(img)
    if angle is None:
        return img, False

    abs_angle = abs(angle)
    if abs_angle < LINE_DESKEW_MIN_DEG or abs_angle > LINE_DESKEW_MAX_DEG:
        return img, False

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    log.info(f"Line deskew: corrected {angle:.2f}°")
    return deskewed, True


def _crop_by_border_contrast(img, margin=EDGE_SCAN_MARGIN_PX):
    """
    Shrink from all 4 edges until strip color is significantly different from
    background color. This targets a bright slab on a more uniform background.
    Returns (cropped_img, True) on success, else (original_img, False).
    """
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return img, False

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Estimate background color from outer border frame.
    bh = max(6, int(h * 0.04))
    bw = max(6, int(w * 0.04))
    border_pixels = np.concatenate([
        lab[:bh, :, :].reshape(-1, 3),
        lab[h - bh:, :, :].reshape(-1, 3),
        lab[:, :bw, :].reshape(-1, 3),
        lab[:, w - bw:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border_pixels, axis=0)

    # Color-distance map from background in LAB space.
    diff = np.linalg.norm(lab - bg, axis=2)
    band = max(1, EDGE_SCAN_BAND)

    x0_scan = int(w * 0.10)
    x1_scan = int(w * 0.90)
    y0_scan = int(h * 0.10)
    y1_scan = int(h * 0.90)

    max_scan_y = max(1, int(h * EDGE_SCAN_MAX_RATIO))
    max_scan_x = max(1, int(w * EDGE_SCAN_MAX_RATIO))

    def scan_top():
        hits = 0
        for y in range(0, max_scan_y):
            strip = diff[y:min(h, y + band), x0_scan:x1_scan]
            ratio = float(np.mean(strip > EDGE_DIFF_THRESHOLD))
            hits = hits + 1 if ratio >= EDGE_HIT_RATIO else 0
            if hits >= 2:
                return max(0, y - 1)
        return None

    def scan_bottom():
        hits = 0
        for i in range(0, max_scan_y):
            y = h - 1 - i
            strip = diff[max(0, y - band + 1):y + 1, x0_scan:x1_scan]
            ratio = float(np.mean(strip > EDGE_DIFF_THRESHOLD))
            hits = hits + 1 if ratio >= EDGE_HIT_RATIO else 0
            if hits >= 2:
                return min(h - 1, y + 1)
        return None

    def scan_left():
        hits = 0
        for x in range(0, max_scan_x):
            strip = diff[y0_scan:y1_scan, x:min(w, x + band)]
            ratio = float(np.mean(strip > EDGE_DIFF_THRESHOLD))
            hits = hits + 1 if ratio >= EDGE_HIT_RATIO else 0
            if hits >= 2:
                return max(0, x - 1)
        return None

    def scan_right():
        hits = 0
        for i in range(0, max_scan_x):
            x = w - 1 - i
            strip = diff[y0_scan:y1_scan, max(0, x - band + 1):x + 1]
            ratio = float(np.mean(strip > EDGE_DIFF_THRESHOLD))
            hits = hits + 1 if ratio >= EDGE_HIT_RATIO else 0
            if hits >= 2:
                return min(w - 1, x + 1)
        return None

    top = scan_top()
    bottom = scan_bottom()
    left = scan_left()
    right = scan_right()

    if None in (top, bottom, left, right):
        return img, False

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(w - 1, right + margin)
    bottom = min(h - 1, bottom + margin)

    cw = right - left + 1
    ch = bottom - top + 1
    if cw < 80 or ch < 80:
        return img, False
    if (cw * ch) < (w * h * MIN_CROP_AREA_RATIO):
        return img, False

    log.info(
        f"Edge-scan crop: x={left}, y={top}, w={cw}, h={ch}, "
        f"thr={EDGE_DIFF_THRESHOLD}, hit_ratio={EDGE_HIT_RATIO}"
    )
    return img[top:bottom + 1, left:right + 1], True


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
    return cv2.addWeighted(img, 1.25, blurred, -0.25, 0)


def _light_color_boost(img):
    """Conservative color pop without aggressive local contrast edits."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * LIGHT_COLOR_SAT_GAIN, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * LIGHT_COLOR_VAL_GAIN, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _suppress_specular_scratches(img):
    """Reduce bright low-saturation streaks from slab plastic reflections."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = np.zeros_like(v, dtype=np.uint8)
    mask[(v >= DEGLARE_VALUE_THRESH) & (s <= DEGLARE_SAT_THRESH)] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    ratio = float(np.mean(mask > 0))
    if ratio <= 0.0:
        return img
    if ratio > DEGLARE_MAX_MASK_RATIO:
        # Skip to avoid flattening the whole image when glare mask is too broad.
        log.info(f"Deglare skipped, mask too large: {ratio:.3f}")
        return img

    fixed = cv2.inpaint(img, mask, DEGLARE_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    out = cv2.addWeighted(fixed, 0.72, img, 0.28, 0)
    log.info(f"Deglare applied, mask ratio={ratio:.3f}")
    return out


def _color_correct(img):
    """Post-crop color correction tuned for slab/card photos."""
    out = img.astype(np.float32)

    if ENABLE_DEGLARE:
        out = _suppress_specular_scratches(out.astype(np.uint8)).astype(np.float32)

    # Gray-world white balance to neutralize color cast from lightbox/room light.
    b_mean = float(np.mean(out[:, :, 0]))
    g_mean = float(np.mean(out[:, :, 1]))
    r_mean = float(np.mean(out[:, :, 2]))
    avg = (b_mean + g_mean + r_mean) / 3.0
    if b_mean > 1 and g_mean > 1 and r_mean > 1:
        out[:, :, 0] *= avg / b_mean
        out[:, :, 1] *= avg / g_mean
        out[:, :, 2] *= avg / r_mean

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Mild local contrast on L channel, kept conservative to avoid halo artifacts.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)

    # Gentle highlight roll-off to reduce glare clipping without whitening nearby color.
    highlight_mask = l_f > 200.0
    l_f[highlight_mask] -= ((l_f[highlight_mask] - 200.0) / 55.0) * HIGHLIGHT_ROLL_OFF
    l = np.clip(l_f, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Very mild vibrance/saturation boost for safer color across different card sets.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    vib_boost = 1.0 + COLOR_VIBRANCE * (1.0 - (sat / 255.0))
    sat = np.clip(sat * vib_boost, 0, 255)
    sat = np.clip(sat * SATURATION_GAIN, 0, 255)
    hsv[:, :, 1] = sat
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Optional gamma for quick brightness tuning.
    if abs(GAMMA - 1.0) > 1e-3:
        gamma = max(0.2, min(3.0, GAMMA))
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, table)

    return out


def process_card_image(src_path: str, dst_path: str, rotate_only: bool = False):
    """
    Full processing pipeline:
      1. Rotate 90° left (counterclockwise)
      2. Detect card slab and perspective-crop  (skipped when rotate_only=True)
      3. Sharpen                                (skipped when rotate_only=True)
    Falls back to a plain file copy if anything goes wrong.
    """
    try:
        img = _read_image_with_retry(src_path, timeout=8.0, poll=0.25)
        if img is None:
            raise ValueError(f"cv2 could not open {src_path}")

        # 1. Rotate 90° left
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        log.info("Rotated 90° left")

        if not rotate_only:
            # 2. Primary crop: shrink edges inward until color diverges from background.
            img, cropped = _crop_by_border_contrast(img)

            # 2a. Secondary crop: white slab foreground (background agnostic).
            if not cropped:
                img, cropped = _crop_bright_slab_foreground(img)

            # 2b. Tertiary crop for darker backgrounds.
            if not cropped:
                img, cropped = _crop_non_black_foreground(img)

            # 2c. Optional perspective cleanup on already-cropped image.
            if not cropped:
                corners = _detect_card_contour(img)
                if corners is not None:
                    img = _perspective_crop(img, corners, margin=CROP_MARGIN_PX)
                    cropped = True
                    log.info("Card detected and perspective-cropped")
                else:
                    bg_corners = _detect_card_on_black_background(img)
                    if bg_corners is not None:
                        img = _perspective_crop(img, bg_corners, margin=CROP_MARGIN_PX)
                        cropped = True
                        log.info("Card perspective-cropped using black-background detector")
                    else:
                        log.warning("Card not detected for crop — leaving orientation as-is")

            # 3. Color correction intentionally disabled for color-safe output.
            if ENABLE_COLOR_CORRECTION:
                img = _color_correct(img)
                log.info("Applied color correction")

            # 3a. Mild color boost for less washed-out results.
            if ENABLE_LIGHT_COLOR_BOOST:
                img = _light_color_boost(img)
                log.info("Applied light color boost")

            # 4. Sharpen
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

        # Trigger capture without autofocus first. Fallback keeps compatibility.
        try:
            resp = requests.get(f"{DCC_URL}/?CMD=CaptureNoAf", timeout=15)
            resp.raise_for_status()
            log.info("Shutter triggered (CaptureNoAf)")
        except Exception:
            log.warning("CaptureNoAf failed; falling back to Capture")
            resp = requests.get(f"{DCC_URL}/?CMD=Capture", timeout=15)
            resp.raise_for_status()
            log.info("Shutter triggered (Capture)")

        # Poll for new file
        deadline = time.time() + 30
        while time.time() < deadline:
            after = get_jpg_files(DCC_SAVE_DIR)
            new_files = after - before
            if new_files:
                src = max(new_files, key=os.path.getmtime)
                if not _wait_for_file_ready(src, timeout=8.0, stable_checks=3, poll=0.2):
                    log.warning(f"New image not ready yet, continuing wait: {src}")
                    time.sleep(0.3)
                    continue
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
