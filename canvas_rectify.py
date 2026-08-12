#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canvas_rectify.py
=================

Calibrate and rectify the perspective of a canvas using a target of six
64 mm squares printed on an 8.5x11" sheet that sits flush with the right
and bottom edges of a 12x16" canvas.

Modes:
  calibrate  - find the corners of the visible squares and compute the
               homography (image <-> canvas plane in mm). Saved to a file.
  run        - show the rectified (fronto-parallel) canvas in real time
               using the saved calibration.

The camera must NOT move between calibrate and run.

Usage:
  conda activate p3124
  python canvas_rectify.py calibrate           # calibration
  python canvas_rectify.py run                 # live rectification
  python canvas_rectify.py list                # list cameras/resolutions

In the calibrate window:
  col_off / row_off trackbars - shift the visible block over the 3x2 grid
  c  - compute and save the homography
  q / ESC - quit

In the run window:
  c  - go back to calibration
  g  - toggle the metric grid
  q / ESC - quit
"""

import argparse
import sys
import numpy as np
import cv2

# ==========================================================================
#  GEOMETRY CONFIG  (all sizes in millimeters)
#  --- EDIT TO MATCH YOUR SHEET ---
# ==========================================================================
MM_PER_IN = 25.4

CANVAS_W_IN = 12.0                 # canvas width, inches
CANVAS_H_IN = 16.0                 # canvas height, inches
CANVAS_W = CANVAS_W_IN * MM_PER_IN   # 304.8 mm
CANVAS_H = CANVAS_H_IN * MM_PER_IN   # 406.4 mm

SQUARE_MM = 64.0                   # square side
COLS = 3                           # squares horizontally
ROWS = 2                           # squares vertically

# Gap between adjacent squares (white spacing).
# Center-to-center pitch = SQUARE_MM + GAP.
# From the photo: horizontally the squares TOUCH (3*64=192 mm exactly) -> GAP_X=0.
# Vertically there is a small gap between rows -> MEASURE it and set GAP_Y.
GAP_X_MM = 0.0
GAP_Y_MM = 0.6   # measured: white gap between rows is 0.6 mm (nearly touching)

# Anchoring to the canvas edges (= sheet edges, since the sheet is flush right/bottom):
RIGHT_MARGIN_MM = 14.0             # from the canvas right edge to the right edge of the RIGHTMOST square
BOTTOM_MARGIN_MM = 145.0           # from the canvas bottom edge to the bottom edge of the BOTTOMMOST square

PITCH_X = SQUARE_MM + GAP_X_MM
PITCH_Y = SQUARE_MM + GAP_Y_MM

# ==========================================================================
#  CAMERA / OUTPUT CONFIG
# ==========================================================================
CAM_INDEX = 0
# The GXI-IMX179 camera via OpenCV/AVFoundation reliably delivers frames ONLY
# in 1920x1080 MJPG. Higher modes (2048/2592/3264) return 0 frames, and so do
# 1280/640 - so leave it alone. This is the working maximum.
REQ_WIDTH = 1920
REQ_HEIGHT = 1080
USE_MJPG = True                    # the USB camera delivers frames in MJPG

PX_PER_MM = 2.0                    # scale of the rectified image (2 px/mm ~ 610x813)
CALIB_FILE = "calibration.npz"

# Square detector thresholds
MIN_QUAD_AREA_FRAC = 0.0004        # min square area as a fraction of the frame
MAX_QUAD_AREA_FRAC = 0.20          # max square area as a fraction of the frame
ASPECT_TOL = 0.6                   # aspect ratio tolerance (|1 - w/h|); higher = more perspective-tolerant


# ==========================================================================
#  MODEL: mm corner coordinates of each square in the canvas frame.
#  Origin - the BOTTOM-RIGHT corner of the canvas (= sheet corner), X right, Y down.
#  Thus the canvas occupies X in [-CANVAS_W, 0], Y in [-CANVAS_H, 0], and the
#  square coordinates do NOT depend on the canvas size (the sheet is always in
#  the bottom-right corner) - no need to recalibrate when the canvas size changes.
# ==========================================================================
def build_model():
    """Return dict {(col,row): (4,2) float32} of corners TL,TR,BR,BL in mm."""
    x_right_anchor = -RIGHT_MARGIN_MM                    # right edge of the right square
    y_bottom_anchor = -BOTTOM_MARGIN_MM                  # bottom edge of the bottom square
    l_right = x_right_anchor - SQUARE_MM                 # left edge of the right column (col=COLS-1)
    t_bottom = y_bottom_anchor - SQUARE_MM               # top edge of the bottom row (row=ROWS-1)

    model = {}
    for col in range(COLS):
        x0 = l_right - (COLS - 1 - col) * PITCH_X        # left edge of column col
        for row in range(ROWS):
            y0 = t_bottom - (ROWS - 1 - row) * PITCH_Y   # top edge of row row
            s = SQUARE_MM
            corners = np.array([
                [x0,     y0],       # TL
                [x0 + s, y0],       # TR
                [x0 + s, y0 + s],   # BR
                [x0,     y0 + s],   # BL
            ], dtype=np.float32)
            model[(col, row)] = corners
    return model


# ==========================================================================
#  CAMERA
# ==========================================================================
def _configure(cap):
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)


def open_camera(index=CAM_INDEX, attempts=6):
    """Open the camera and make sure it ACTUALLY delivers frames.

    The camera sometimes opens but does not stream on the first try -
    in that case we reopen it.
    """
    import time
    for a in range(attempts):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            _configure(cap)
            # verify that frames really arrive
            t = time.time() + 3.0
            while time.time() < t:
                ok, f = cap.read()
                if ok and f is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cam] camera #{index}: {w}x{h} (attempt {a + 1})")
                    return cap
                time.sleep(0.05)
        cap.release()
        time.sleep(0.4)
    raise RuntimeError(
        f"Camera #{index} opened but delivers no frames. Make sure it is not "
        f"held by FaceTime/another app, and check the index (python canvas_rectify.py list).")


def read_frame(cap, retries=30):
    """Read a frame, tolerating occasional empty responses."""
    import time
    for _ in range(retries):
        ok, f = cap.read()
        if ok and f is not None:
            return f
        time.sleep(0.02)
    return None


# ==========================================================================
#  SQUARE DETECTOR
# ==========================================================================
def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL (in image coordinates)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_squares(frame):
    """Return a list of (4,2) corner arrays for the squares (TL,TR,BR,BL).

    Works with OUTLINED squares (black frame on light paper): black lines ->
    foreground, take all 4-gon contours of the right size (outer frame
    contours AND inner grid-cell "holes"), then deduplicate by center
    (keep the larger one = outer edge).
    """
    h, w = frame.shape[:2]
    area_img = float(h * w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Black lines on a light background -> foreground (white).
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 41, 12)
    # bridge possible gaps in thin lines
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cand = []  # (area, quad, center)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_QUAD_AREA_FRAC * area_img or area > MAX_QUAD_AREA_FRAC * area_img:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (rw, rh) = cv2.minAreaRect(c)[1]
        if rw < 1 or rh < 1 or abs(1.0 - (rw / rh)) > ASPECT_TOL:
            continue
        # convex-hull fill ratio - rejects ragged/letter-like contours
        if area / (cv2.contourArea(cv2.convexHull(c)) + 1e-6) < 0.85:
            continue
        quad = order_quad(approx)
        cand.append((area, quad, quad.mean(axis=0)))

    # deduplicate by center proximity: keep the larger contour
    cand.sort(key=lambda t: -t[0])
    quads, centers, areas = [], [], []
    for area, quad, cen in cand:
        r = 0.45 * np.sqrt(area)
        if any(np.linalg.norm(cen - pc) < r for pc in centers):
            continue
        quads.append(quad)
        centers.append(cen)
        areas.append(area)

    # size-consistency filter: real squares are ~equal in size, so drop
    # outliers (stray rectangles at a different scale)
    if len(quads) >= 3:
        med = float(np.median(areas))
        keep = [i for i, a in enumerate(areas) if 0.4 * med <= a <= 2.5 * med]
        quads = [quads[i] for i in keep]

    # refine corners to sub-pixel accuracy
    if quads:
        corners = np.vstack(quads).astype(np.float32)
        cv2.cornerSubPix(
            gray, corners, (7, 7), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        quads = [corners[i * 4:(i + 1) * 4] for i in range(len(quads))]
    return quads


def assign_local_grid(quads):
    """Lay the squares out into a local grid -> list of (col,row,quad).

    Returns (items, n_cols, n_rows), where col/row are local indices
    (0 = left/top) within the visible block.
    """
    if not quads:
        return [], 0, 0
    centers = np.array([q.mean(axis=0) for q in quads])
    heights = np.array([np.linalg.norm(q[3] - q[0]) for q in quads])
    tol = 0.5 * float(np.median(heights))

    # group into rows (clustering by Y)
    order = np.argsort(centers[:, 1])
    rows = []          # list of index lists
    for idx in order:
        placed = False
        for r in rows:
            if abs(centers[idx, 1] - np.mean([centers[j, 1] for j in r])) < tol:
                r.append(idx)
                placed = True
                break
        if not placed:
            rows.append([idx])

    items = []
    n_cols = 0
    for r_i, r in enumerate(rows):
        r_sorted = sorted(r, key=lambda j: centers[j, 0])
        n_cols = max(n_cols, len(r_sorted))
        for c_i, j in enumerate(r_sorted):
            items.append((c_i, r_i, quads[j]))
    return items, n_cols, len(rows)


# ==========================================================================
#  CALIBRATION
# ==========================================================================
def compute_homography(matches, model):
    """matches: list of (col,row,quad_img). Returns H (world_mm -> image)."""
    world_pts, img_pts = [], []
    for (col, row, quad) in matches:
        if (col, row) not in model:
            continue
        world_pts.append(model[(col, row)])
        img_pts.append(quad)
    if len(world_pts) < 1:
        return None
    world_pts = np.vstack(world_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)
    if len(world_pts) < 4:
        return None
    H, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, 3.0)
    return H


def output_transform():
    """A: output-px -> world-mm (for warpPerspective with WARP_INVERSE_MAP).

    The world origin is the bottom-right corner of the canvas, so the canvas
    lies in X in [-CANVAS_W,0], Y in [-CANVAS_H,0]. Output pixel (0,0) = the
    top-left corner of the canvas (world (-CANVAS_W,-CANVAS_H)).
    """
    out_w = int(round(CANVAS_W * PX_PER_MM))
    out_h = int(round(CANVAS_H * PX_PER_MM))
    A = np.array([
        [1.0 / PX_PER_MM, 0.0, -CANVAS_W],
        [0.0, 1.0 / PX_PER_MM, -CANVAS_H],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return A, out_w, out_h


def corrected_view_transform(H, cam_w, cam_h, target=1200):
    """Rectified view CROPPED to the camera coverage (not the whole canvas).

    The frame corners are projected onto the canvas plane (H^-1), and the
    output window is built from their bbox with correct proportions. Returns
    (A, out_w, out_h), where A: output-px -> world-mm (for warpPerspective
    with WARP_INVERSE_MAP)."""
    Hinv = np.linalg.inv(H)
    corners = np.array([[0, 0], [cam_w, 0], [cam_w, cam_h], [0, cam_h]], np.float64)
    world = []
    for x, y in corners:
        p = Hinv @ np.array([x, y, 1.0])
        world.append(p[:2] / p[2])
    world = np.array(world)
    xmin, ymin = world.min(axis=0)
    xmax, ymax = world.max(axis=0)
    bw, bh = xmax - xmin, ymax - ymin
    ppm = target / max(bw, bh)
    out_w = int(round(bw * ppm))
    out_h = int(round(bh * ppm))
    A = np.array([[1.0 / ppm, 0.0, xmin],
                  [0.0, 1.0 / ppm, ymin],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return A, out_w, out_h


def draw_metric_grid(img, step_mm=50.0):
    out_h, out_w = img.shape[:2]
    color = (0, 180, 0)
    x = 0.0
    while x <= CANVAS_W:
        px = int(round(x * PX_PER_MM))
        cv2.line(img, (px, 0), (px, out_h), color, 1, cv2.LINE_AA)
        x += step_mm
    y = 0.0
    while y <= CANVAS_H:
        py = int(round(y * PX_PER_MM))
        cv2.line(img, (0, py), (out_w, py), color, 1, cv2.LINE_AA)
        y += step_mm
    return img


def calibrate(cap, model):
    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("col_off", win, 0, max(0, COLS - 1), lambda v: None)
    cv2.createTrackbar("row_off", win, 0, max(0, ROWS - 1), lambda v: None)

    H_saved = None
    print("[calib] point the camera at the squares. Use the col_off/row_off\n"
          "        trackbars to align the labels with the real positions. 'c' saves.")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[calib] no frame"); break
        vis = frame.copy()
        quads = detect_squares(frame)
        items, ncols, nrows = assign_local_grid(quads)

        col_off = cv2.getTrackbarPos("col_off", win)
        row_off = cv2.getTrackbarPos("row_off", win)

        matches = []
        for (lc, lr, quad) in items:
            gc, gr = lc + col_off, lr + row_off
            valid = (0 <= gc < COLS) and (0 <= gr < ROWS)
            color = (0, 220, 0) if valid else (0, 0, 255)
            cv2.polylines(vis, [quad.astype(np.int32)], True, color, 2, cv2.LINE_AA)
            for p in quad:
                cv2.circle(vis, tuple(p.astype(int)), 4, (255, 0, 0), -1)
            cen = quad.mean(axis=0).astype(int)
            cv2.putText(vis, f"{gc},{gr}", tuple(cen), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)
            if valid:
                matches.append((gc, gr, quad))

        # auto hint: if all 6 (3x2) are visible, the mapping is unambiguous
        auto = (ncols == COLS and nrows == ROWS)
        txt = (f"detected={len(quads)} grid={ncols}x{nrows} "
               f"matched={len(matches)}  off=({col_off},{row_off})"
               f"{'  [AUTO 3x2]' if auto else ''}")
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "c=save  q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            H = compute_homography(matches, model)
            if H is None:
                print("[calib] need >=4 corners (>=1 valid square, 2+ is better). "
                      "Check the offsets.")
                continue
            np.savez(CALIB_FILE, H=H, px_per_mm=PX_PER_MM,
                     canvas_w=CANVAS_W, canvas_h=CANVAS_H)
            H_saved = H
            print(f"[calib] homography saved to {CALIB_FILE} "
                  f"(from {len(matches)} squares)")
            # quick confirmation - reprojection error
            err = reprojection_error(matches, model, H)
            print(f"[calib] mean reprojection error: {err:.2f} px")
            break

    cv2.destroyWindow(win)
    return H_saved


def reprojection_error(matches, model, H):
    errs = []
    for (col, row, quad) in matches:
        wp = model[(col, row)]
        wp_h = np.hstack([wp, np.ones((4, 1), np.float32)]).T
        proj = H @ wp_h
        proj = (proj[:2] / proj[2]).T
        errs.append(np.linalg.norm(proj - quad, axis=1))
    return float(np.mean(errs)) if errs else float("nan")


# ==========================================================================
#  RUN MODE - live rectification
# ==========================================================================
def load_calibration():
    try:
        data = np.load(CALIB_FILE)
    except FileNotFoundError:
        return None
    return data["H"]


def run(cap, model):
    H = load_calibration()
    if H is None:
        print(f"[run] no {CALIB_FILE} - run calibrate first.")
        H = calibrate(cap, model)
        if H is None:
            return
    A, out_w, out_h = output_transform()
    M = H @ A                                  # output-px -> source-image-px

    win = "rectified"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_grid = False
    print("[run] live rectification. g=grid  c=recalibrate  q=quit")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[run] no frame"); break
        rect = cv2.warpPerspective(frame, M, (out_w, out_h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        if show_grid:
            draw_metric_grid(rect, 50.0)
            cv2.putText(rect, "50mm grid", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, rect)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('g'):
            show_grid = not show_grid
        if key == ord('c'):
            cv2.destroyWindow(win)
            newH = calibrate(cap, model)
            if newH is not None:
                H = newH
                M = H @ A
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.destroyWindow(win)


# ==========================================================================
#  OVERLAY MODE - reference contours over the live camera frame (AR)
# ==========================================================================
OVERLAY_ADJUST_FILE = "overlay_adjust.npz"
OVERLAY_COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (255, 255, 255)]


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def extract_edges(ref_bgr, canny_lo, canny_hi, blur=3, max_side=1600):
    """Reference contours: Canny -> uint8 mask (0/255). The ref is shrunk to max_side."""
    h, w = ref_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        ref_bgr = cv2.resize(ref_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    if blur >= 3 and blur % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    return edges  # (rh, rw) uint8, resolution of the shrunk reference


def ref_to_world(ref_w, ref_h, dx, dy, sx, sy, theta_deg):
    """T: reference pixel -> world (mm). The ref is stretched to fill the whole
    canvas, plus adjustment: independent scales sx/sy, rotation about the canvas
    center, and offset (dx,dy) mm."""
    # base: ref pixel -> world (canvas in X[-W,0], Y[-H,0])
    S0 = np.array([[CANVAS_W / ref_w, 0.0, -CANVAS_W],
                   [0.0, CANVAS_H / ref_h, -CANVAS_H],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    cx, cy = -CANVAS_W / 2.0, -CANVAS_H / 2.0        # canvas center in the world
    th = np.deg2rad(theta_deg)
    cos, sin = np.cos(th), np.sin(th)
    R = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    Sc = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    Rs = R @ Sc                                       # scale first, then rotate
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, cx + dx], [0, 1, cy + dy], [0, 0, 1]], dtype=np.float64)
    return T2 @ Rs @ T1 @ S0


def apply_view(img, v):
    """Viewport: zoom/pan on the finished image. v: {z, ox, oy} (ox/oy are fractions [0..1])."""
    z = v["z"]
    if z <= 1.0001 and v["ox"] == 0.0 and v["oy"] == 0.0:
        return img
    h, w = img.shape[:2]
    rw, rh = int(round(w / z)), int(round(h / z))
    rx = min(max(int(round(v["ox"] * w)), 0), max(0, w - rw))
    ry = min(max(int(round(v["oy"] * h)), 0), max(0, h - rh))
    crop = img[ry:ry + rh, rx:rx + rw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def load_overlay_adjust(path):
    """Return (dx,dy,sx,sy,theta,alpha). Compatible with the old format (single 's')."""
    try:
        d = np.load(path)
    except FileNotFoundError:
        return 0.0, 0.0, 1.0, 1.0, 0.0, 0.5
    dx = float(d["dx"]) if "dx" in d else 0.0
    dy = float(d["dy"]) if "dy" in d else 0.0
    theta = float(d["theta"]) if "theta" in d else 0.0
    alpha = float(d["alpha"]) if "alpha" in d else 0.5
    if "sx" in d and "sy" in d:
        sx, sy = float(d["sx"]), float(d["sy"])
    elif "s" in d:
        sx = sy = float(d["s"])
    else:
        sx = sy = 1.0
    return dx, dy, sx, sy, theta, alpha


def overlay(cap, ref_path, adjust_path=OVERLAY_ADJUST_FILE):
    H = load_calibration()
    if H is None:
        print(f"[overlay] no {CALIB_FILE} - run calibrate first."); return
    ref = cv2.imread(ref_path)
    if ref is None:
        print(f"[overlay] could not read the reference: {ref_path}"); return

    # shrunk color copy of the reference (for image mode and for contours)
    ref_small = _resize_max(ref, 1600)
    canny_lo, canny_hi, blur = 50, 150, 3
    edges = extract_edges(ref_small, canny_lo, canny_hi, blur, max_side=10 ** 9)
    rh, rw = ref_small.shape[:2]
    dx, dy, sx, sy, theta, alpha = load_overlay_adjust(adjust_path)
    color_i = 0
    show = True
    show_help = True
    rectified = False
    render_mode = "contours"       # "contours" | "image"

    MOVE, SCALE, ROT, DA = 2.0, 1.01, 0.5, 0.05  # steps
    A = None  # rectified-view transform; lazy (needs the frame size)

    win = "overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # --- viewport (zoom/pan of the finished image) ---
    view = {"z": 1.0, "ox": 0.0, "oy": 0.0, "W": 1, "H": 1,
            "pan": False, "zoomdrag": False, "lx": 0, "ly": 0}

    def clampv(val, lo, hi):
        return max(lo, min(hi, val))

    def zoom_at(sx, sy, factor):
        W, H = view["W"], view["H"]
        z = view["z"]
        nz = clampv(z * factor, 1.0, 25.0)
        nx = view["ox"] + (sx / W) / z          # normalized point under the cursor
        ny = view["oy"] + (sy / H) / z
        view["ox"] = clampv(nx - (sx / W) / nz, 0.0, max(0.0, 1.0 - 1.0 / nz))
        view["oy"] = clampv(ny - (sy / H) / nz, 0.0, max(0.0, 1.0 - 1.0 / nz))
        view["z"] = nz

    def on_mouse(event, x, y, flags, param):
        W, H = view["W"], view["H"]
        if W <= 1 or H <= 1:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            hi = (flags >> 16) & 0xFFFF          # high 16 bits = signed delta
            if hi >= 0x8000:
                hi -= 0x10000
            if hi != 0:
                zoom_at(x, y, 1.2 if hi > 0 else 1 / 1.2)
        elif event == cv2.EVENT_LBUTTONDOWN:
            view["pan"] = True; view["lx"], view["ly"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            view["pan"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            view["zoomdrag"] = True; view["lx"], view["ly"] = x, y
        elif event == cv2.EVENT_RBUTTONUP:
            view["zoomdrag"] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if view["pan"]:
                dsx, dsy = x - view["lx"], y - view["ly"]
                view["lx"], view["ly"] = x, y
                z = view["z"]
                view["ox"] = clampv(view["ox"] - (dsx / W) / z, 0.0, max(0.0, 1.0 - 1.0 / z))
                view["oy"] = clampv(view["oy"] - (dsy / H) / z, 0.0, max(0.0, 1.0 - 1.0 / z))
            elif view["zoomdrag"]:
                dsy = y - view["ly"]
                view["lx"], view["ly"] = x, y
                zoom_at(x, y, 1.0 - dsy * 0.005)   # drag up = zoom in

    cv2.setMouseCallback(win, on_mouse)
    print("[overlay] w/a/s/d move, z/x scale, [ ] X, - = Y, ,/. rotate,\n"
          "          m contours/image, 9/0 opacity -/+,\n"
          "          1/2 3/4 Canny, o toggle, c color, r raw/corrected,\n"
          "          MOUSE: wheel or right button (drag) = zoom at cursor,\n"
          "          left button (drag) = pan, SPACE = reset zoom,\n"
          "          p save, i reset adjustment, h help, q quit")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[overlay] no frame"); break
        base = frame
        if rectified:
            if A is None:
                A, ow, oh = corrected_view_transform(H, frame.shape[1], frame.shape[0])
            base = cv2.warpPerspective(frame, H @ A, (ow, oh),
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        canvas = base.copy()
        ch, cw = canvas.shape[:2]

        if show:
            T = ref_to_world(rw, rh, dx, dy, sx, sy, theta)
            if rectified:
                M = np.linalg.inv(A) @ T        # ref pixel -> output pixel
            else:
                M = H @ T                        # ref pixel -> camera pixel
            if render_mode == "image":
                warped = cv2.warpPerspective(ref_small, M, (cw, ch),
                                             flags=cv2.INTER_LINEAR)
                mcov = cv2.warpPerspective(
                    np.full(ref_small.shape[:2], 255, np.uint8), M, (cw, ch),
                    flags=cv2.INTER_NEAREST) > 0
                canvas[mcov] = (canvas[mcov] * (1.0 - alpha)
                                + warped[mcov] * alpha).astype(np.uint8)
            else:  # contours
                warped = cv2.warpPerspective(edges, M, (cw, ch),
                                             flags=cv2.INTER_NEAREST)
                warped = cv2.dilate(warped, np.ones((2, 2), np.uint8))
                m = warped > 0
                col = np.array(OVERLAY_COLORS[color_i], np.float32)
                canvas[m] = (canvas[m] * (1.0 - alpha) + col * alpha).astype(np.uint8)

        view["W"], view["H"] = canvas.shape[1], canvas.shape[0]
        disp = apply_view(canvas, view)

        if show_help:
            hud = [f"dx={dx:+.0f}mm dy={dy:+.0f}mm  sx={sx:.3f} sy={sy:.3f}  rot={theta:+.1f}deg",
                   f"mode={render_mode}  alpha={alpha:.2f}  zoom={view['z']:.1f}x  "
                   f"Canny={canny_lo}/{canny_hi}  overlay={'on' if show else 'off'}  "
                   f"view={'corrected' if rectified else 'raw (perspective)'}",
                   "w/a/s/d move  z/x scale  [ ] X  - = Y  ,/. rot  m mode  9/0 alpha  "
                   "1/2 3/4 canny  o c r  |  mouse: wheel/RMB zoom, LMB pan, SPACE reset"]
            y0 = 26
            for ln in hud:
                cv2.putText(disp, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 1, cv2.LINE_AA)
                y0 += 24

        cv2.imshow(win, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            view["z"], view["ox"], view["oy"] = 1.0, 0.0, 0.0
        elif k == ord('a'):
            dx -= MOVE
        elif k == ord('d'):
            dx += MOVE
        elif k == ord('w'):
            dy -= MOVE
        elif k == ord('s'):
            dy += MOVE
        elif k == ord('x'):
            sx *= SCALE; sy *= SCALE
        elif k == ord('z'):
            sx /= SCALE; sy /= SCALE
        elif k == ord(']'):
            sx *= SCALE
        elif k == ord('['):
            sx /= SCALE
        elif k == ord('='):
            sy *= SCALE
        elif k == ord('-'):
            sy /= SCALE
        elif k == ord('.'):
            theta += ROT
        elif k == ord(','):
            theta -= ROT
        elif k == ord('m'):
            render_mode = "image" if render_mode == "contours" else "contours"
        elif k == ord('0'):
            alpha = min(1.0, alpha + DA)
        elif k == ord('9'):
            alpha = max(0.0, alpha - DA)
        elif k == ord('o'):
            show = not show
        elif k == ord('c'):
            color_i = (color_i + 1) % len(OVERLAY_COLORS)
        elif k == ord('r'):
            rectified = not rectified
        elif k == ord('h'):
            show_help = not show_help
        elif k == ord('i'):
            dx, dy, sx, sy, theta = 0.0, 0.0, 1.0, 1.0, 0.0
        elif k == ord('p'):
            np.savez(adjust_path, dx=dx, dy=dy, sx=sx, sy=sy, theta=theta, alpha=alpha)
            print(f"[overlay] adjustment saved to {adjust_path}")
        elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
            if k == ord('1'):
                canny_lo = max(0, canny_lo - 5)
            elif k == ord('2'):
                canny_lo = min(canny_hi - 1, canny_lo + 5)
            elif k == ord('3'):
                canny_hi = max(canny_lo + 1, canny_hi - 5)
            elif k == ord('4'):
                canny_hi = min(500, canny_hi + 5)
            edges = extract_edges(ref, canny_lo, canny_hi, blur)
            rh, rw = edges.shape[:2]

    cv2.destroyWindow(win)


def world_to_template_px(pt_mm, ppm):
    """World mm (bottom-right corner = 0) -> canvas template pixel."""
    x, y = pt_mm
    return (int(round((x + CANVAS_W) * ppm)), int(round((y + CANVAS_H) * ppm)))


def generate_template(out_path, ppm=4.0):
    """Render the WHOLE 12x16 canvas with the 6 squares in their real positions.

    With `overlay --ref <template>` this template lands the green squares
    exactly on the real ones (a check of the whole chain). It is also handy as
    a base: draw the reference on top of this canvas and it will land with the
    correct proportions.
    """
    W = int(round(CANVAS_W * ppm))
    H = int(round(CANVAS_H * ppm))
    img = np.full((H, W, 3), 255, np.uint8)
    model = build_model()
    th = max(1, int(round(ppm)))  # line thickness
    for (col, row), corners in model.items():
        pts = np.array([world_to_template_px(p, ppm) for p in corners], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 0), th, cv2.LINE_AA)
    # canvas border to make the proportions clear
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 0, 0), th)
    cv2.imwrite(out_path, img)
    print(f"[gen-template] saved {out_path}  ({W}x{H}px, {ppm} px/mm, "
          f"canvas {CANVAS_W:.0f}x{CANVAS_H:.0f}mm)")


def list_devices():
    print("Probing camera indices 0..4 at the requested resolution:")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            if USE_MJPG:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  #{i}: OPEN  max~{w}x{h}")
            cap.release()
        else:
            print(f"  #{i}: -")


def main():
    ap = argparse.ArgumentParser(description="Canvas perspective calibration/rectification")
    ap.add_argument("mode", choices=["calibrate", "run", "overlay", "gen-template", "list"])
    ap.add_argument("--cam", type=int, default=CAM_INDEX)
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="scale of the rectified image")
    ap.add_argument("--ref", default="nadya111.jpg",
                    help="reference image for overlay mode")
    ap.add_argument("--canvas-w-in", type=float, default=None,
                    help="canvas width in inches (the sheet is always bottom-right)")
    ap.add_argument("--canvas-h-in", type=float, default=None,
                    help="canvas height in inches")
    ap.add_argument("--out", default=None, help="output file for gen-template")
    ap.add_argument("--adjust", default=OVERLAY_ADJUST_FILE,
                    help="file for saved overlay adjustment parameters (.npz)")
    args = ap.parse_args()

    global PX_PER_MM, CANVAS_W, CANVAS_H
    if args.px_per_mm:
        PX_PER_MM = args.px_per_mm
    if args.canvas_w_in:
        CANVAS_W = args.canvas_w_in * MM_PER_IN
    if args.canvas_h_in:
        CANVAS_H = args.canvas_h_in * MM_PER_IN

    if args.mode == "list":
        list_devices()
        return
    if args.mode == "gen-template":
        generate_template(args.out or "calib_template.png")
        return

    model = build_model()
    cap = open_camera(args.cam)
    try:
        if args.mode == "calibrate":
            calibrate(cap, model)
        elif args.mode == "overlay":
            overlay(cap, args.ref, args.adjust)
        else:
            run(cap, model)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
