#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
artprojector.py
===============

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
  python artprojector.py calibrate           # calibration
  python artprojector.py run                 # live rectification
  python artprojector.py list                # list cameras/resolutions

In the calibrate window:
  col_off / row_off trackbars - shift the visible block over the 3x2 grid
  a  - auto-fit the lens distortion (straightens the target's grid lines)
  1/2, 3/4 - tune the radial distortion k1 / k2 by hand
  5/6 - halve / double the tuning step;  0 - reset the distortion
  f  - switch the homography fit: consensus quad <-> individual corners
  c  - compute and save the homography (+ the distortion)
  q / ESC - quit

In the run window:
  c  - go back to calibration
  g  - toggle the metric grid
  q / ESC - quit
"""

import argparse
import os
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
# On Linux/V4L2 the GXI-IMX179 delivers valid MJPG in every advertised mode
# EXCEPT 1920x1080 (the only 25fps mode): reads succeed but the payload is not
# JPEG, so frames come out all-zero (black). ffmpeg fails on that mode too.
# 2592x1944 @15fps is the highest mode that actually works - good for
# calibration, since square corners land more precisely.
# Override per-run with --width/--height.
REQ_WIDTH = 2592
REQ_HEIGHT = 1944
USE_MJPG = True                    # the USB camera delivers frames in MJPG

PX_PER_MM = 2.0                    # scale of the rectified image (2 px/mm ~ 610x813)
CALIB_FILE = "calibration.npz"

# Overlay-mode rendering: longest side of the rendered view, and how much wall
# to keep around the canvas in the corrected view.
DISPLAY_MAX = 2000
CORRECTED_MARGIN_MM = 20.0

# Square detector thresholds
MIN_QUAD_AREA_FRAC = 0.0004        # min square area as a fraction of the frame
MAX_QUAD_AREA_FRAC = 0.20          # max square area as a fraction of the frame
ASPECT_TOL = 0.6                   # aspect ratio tolerance (|1 - w/h|); higher = more perspective-tolerant

# How the homography is fitted:
#   "consensus" - from the outer quad fitted to all six squares at once
#                 (default; see the CONSENSUS QUAD section)
#   "corners"   - least squares over all 24 individual square corners
FIT_MODE = "consensus"
# BGR - magenta, distinct from the green squares and from the red used for
# squares that fall outside the grid
CONSENSUS_COLOR = (255, 0, 255)


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
            # verify that frames really arrive AND carry a picture: some modes
            # (e.g. 1920x1080 on the IMX179) return ok=True with all-zero pixels
            blank = 0
            t = time.time() + 3.0
            while time.time() < t:
                ok, f = cap.read()
                if ok and f is not None:
                    if f.max() == 0:
                        blank += 1
                        if blank >= 15:
                            break          # mode is dead - retry / report below
                        time.sleep(0.05)
                        continue
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[cam] camera #{index}: {w}x{h} (attempt {a + 1})")
                    return cap
                time.sleep(0.05)
        cap.release()
        time.sleep(0.4)
    raise RuntimeError(
        f"Camera #{index} delivers no usable frames at {REQ_WIDTH}x{REQ_HEIGHT}. "
        f"If frames arrive but are all black, the camera does not really support "
        f"this mode - try another one (--width/--height), e.g. 1280x960 or "
        f"2592x1944. Check the index with 'python artprojector.py list', and "
        f"make sure no other app holds the camera.")


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
#  LENS DISTORTION
#
#  A wide USB camera bows straight lines near the frame edges, so the printed
#  squares are not really quadrilaterals there: their edges curve, corner
#  detection lands off the true corner, and the fitted lines (and with them the
#  perspective) get dragged along. Correcting the radial distortion FIRST makes
#  the squares genuinely straight-edged, so everything downstream - detection,
#  the consensus quad, the homography - works on a proper pinhole image.
#
#  Only the two radial terms k1/k2 are exposed: they cover the bowing, and they
#  can be dialled in by eye against the straight edges of the target, which the
#  tangential terms cannot. The focal length is pinned to max(w,h), so k1 is
#  O(0.1) for a typical webcam - a scale that is comfortable to tune by hand.
# ==========================================================================
DIST_K1 = 0.0
DIST_K2 = 0.0
DIST_STEP = 0.01                   # k1/k2 increment per keypress in calibrate

_undistort_maps = {}               # (w,h,k1,k2) -> (map1, map2); holds one entry


def camera_matrix(w, h):
    f = float(max(w, h))
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], np.float64)


def dist_coeffs(k1, k2):
    return np.array([k1, k2, 0.0, 0.0, 0.0], np.float64)


def distort_points(pts, k1, k2, w, h):
    """Apply the radial model to points (inverse of undistort_points)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    f, cx, cy = float(max(w, h)), w / 2.0, h / 2.0
    xy = (pts - [cx, cy]) / f
    r2 = (xy ** 2).sum(axis=1, keepdims=True)
    return (xy * (1.0 + k1 * r2 + k2 * r2 ** 2) * f + [cx, cy]).astype(np.float32)


def undistort_points(pts, k1, k2, w, h):
    """Where points would sit on the undistorted image."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    K = camera_matrix(w, h)
    return cv2.undistortPoints(pts, K, dist_coeffs(k1, k2), P=K).reshape(-1, 2)


def auto_distortion(matches, w, h, k1_cur=0.0, k2_cur=0.0):
    """Solve for the k1/k2 that make the target's grid lines straightest.

    The residual from collinearity_residual() has a clean minimum at the true
    distortion, so it can be searched directly instead of dialled in by eye.
    The search runs on the 24 detected corners rather than on the image: the
    corners are pushed back to raw sensor coordinates and re-undistorted per
    candidate, which costs microseconds, so a coarse-to-fine sweep is instant
    even at 2592x1944. The result is a starting point - the manual keys still
    refine it against what the image actually looks like.

    Caveat: the target only covers part of the frame, so k1/k2 are constrained
    over that radial range and extrapolate less well to the far corners. It
    corrects what matters here (the region the canvas is measured in), not the
    whole lens. Returns (k1, k2, residual)."""
    if not matches:
        return k1_cur, k2_cur, float("nan")
    shape = [(col, row) for (col, row, _) in matches]
    flat = np.vstack([q for (_, _, q) in matches])
    raw = distort_points(flat, k1_cur, k2_cur, w, h)   # back to sensor coords

    def residual(a1, a2):
        u = undistort_points(raw, a1, a2, w, h)
        return collinearity_residual(
            [(c, r, u[i * 4:(i + 1) * 4]) for i, (c, r) in enumerate(shape)])

    best = (k1_cur, k2_cur, residual(k1_cur, k2_cur))
    c1, c2, span1, span2 = 0.0, 0.0, 0.4, 0.2
    for _ in range(5):
        for a1 in np.linspace(c1 - span1, c1 + span1, 9):
            for a2 in np.linspace(c2 - span2, c2 + span2, 9):
                r = residual(a1, a2)
                if r < best[2]:
                    best = (float(a1), float(a2), r)
        c1, c2 = best[0], best[1]
        span1, span2 = span1 / 4.0, span2 / 4.0
    return best


def undistort(frame, k1, k2):
    """Remove radial distortion, keeping the camera matrix (and hence the
    framing and the center scale) unchanged.

    The remap is a per-frame cost on a 2592x1944 image, so k1=k2=0 short-circuits
    to the original frame and the maps are built only when the params change."""
    if abs(k1) < 1e-9 and abs(k2) < 1e-9:
        return frame
    h, w = frame.shape[:2]
    key = (w, h, round(k1, 6), round(k2, 6))
    maps = _undistort_maps.get(key)
    if maps is None:
        K = camera_matrix(w, h)
        maps = cv2.initUndistortRectifyMap(K, dist_coeffs(k1, k2), None, K,
                                           (w, h), cv2.CV_16SC2)
        _undistort_maps.clear()    # only the current setting is ever needed
        _undistort_maps[key] = maps
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)


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
#  CONSENSUS QUAD - one big quadrilateral fitted to all six squares
#
#  Corners of the individual squares are localized unreliably (thick printed
#  lines, blur, sub-pixel refinement pulling to the wrong side of an edge), so
#  a homography from any single square - or even a plain least-squares fit over
#  all 24 corners - inherits that noise. But the six squares are printed on one
#  sheet, so their outer edges are collinear by construction: the top edges of
#  the three squares of row 0 lie on ONE straight line, the left edges of the
#  two squares of column 0 lie on ONE straight line, and so on. Fitting each of
#  those four lines through ALL the corner points that belong to it averages the
#  per-corner errors out, and the four line intersections give a much steadier
#  outer quad than any corner measured directly. The homography is then computed
#  from that quad.
# ==========================================================================
def fit_line(pts):
    """Total-least-squares line through points -> (vx, vy, x0, y0), or None.

    DIST_HUBER instead of DIST_L2: if one of the contributing corners is badly
    off (a missed edge on one square), Huber down-weights it instead of letting
    it tilt the whole line."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_HUBER,
                                 0, 0.01, 0.01).reshape(4)
    return float(vx), float(vy), float(x0), float(y0)


def line_intersect(l1, l2):
    """Intersection of two point+direction lines. None if (near-)parallel."""
    if l1 is None or l2 is None:
        return None
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    den = vx1 * vy2 - vy1 * vx2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def block_bounds(matches):
    """(cmin, cmax, rmin, rmax) of the squares actually seen. None if empty.

    The fit is anchored to the observed sub-block rather than to the full 3x2
    model, so it still works when only part of the target is in view - with,
    say, one row visible the outer quad is that row's bounding quad, which is
    still fitted from all its squares at once."""
    if not matches:
        return None
    cols = [c for (c, _, _) in matches]
    rows = [r for (_, r, _) in matches]
    return min(cols), max(cols), min(rows), max(rows)


def consensus_lines(matches, bounds):
    """Fit the four outer edge lines of the block from all its squares.

    matches: list of (col, row, quad) with quad ordered TL,TR,BR,BL.
    Returns dict {"top","bottom","left","right"} of lines (value None if that
    edge had too few points).
    """
    cmin, cmax, rmin, rmax = bounds
    pts = {"top": [], "bottom": [], "left": [], "right": []}
    for (col, row, quad) in matches:
        tl, tr, br, bl = quad
        if row == rmin:              # top edge of the block
            pts["top"] += [tl, tr]
        if row == rmax:              # bottom edge of the block
            pts["bottom"] += [bl, br]
        if col == cmin:              # left edge of the block
            pts["left"] += [tl, bl]
        if col == cmax:              # right edge of the block
            pts["right"] += [tr, br]
    return {k: fit_line(v) for k, v in pts.items()}


def consensus_quad(matches):
    """Outer quad (TL,TR,BR,BL) of the block from the fitted edge lines.

    Returns (quad, lines, bounds); quad is None when an edge could not be
    fitted or two of them turned out parallel."""
    bounds = block_bounds(matches)
    if bounds is None:
        return None, {}, None
    lines = consensus_lines(matches, bounds)
    corners = [line_intersect(lines["left"], lines["top"]),
               line_intersect(lines["right"], lines["top"]),
               line_intersect(lines["right"], lines["bottom"]),
               line_intersect(lines["left"], lines["bottom"])]
    if any(c is None for c in corners):
        return None, lines, bounds
    return np.array(corners, np.float32), lines, bounds


def collinearity_residual(matches):
    """Mean distance (px) from corners to the grid line they should lie on.

    Every row of the target contributes two physically straight lines (the top
    and bottom edges of its squares) and every column two more, each sampled by
    several corners. On an ideal pinhole image those corners are exactly
    collinear, so what is left is detection noise plus whatever lens distortion
    is still uncorrected - which makes this number the thing to minimize when
    tuning k1/k2 by hand. Lines with fewer than 3 points are skipped: two points
    always fit perfectly and would only dilute the average."""
    groups = {}
    for (col, row, quad) in matches:
        tl, tr, br, bl = quad
        groups.setdefault(("r_top", row), []).extend([tl, tr])
        groups.setdefault(("r_bot", row), []).extend([bl, br])
        groups.setdefault(("c_left", col), []).extend([tl, bl])
        groups.setdefault(("c_right", col), []).extend([tr, br])

    dists = []
    for pts in groups.values():
        if len(pts) < 3:
            continue
        ln = fit_line(pts)
        if ln is None:
            continue
        vx, vy, x0, y0 = ln
        for p in pts:
            dists.append(abs((p[0] - x0) * vy - (p[1] - y0) * vx))
    return float(np.mean(dists)) if dists else float("nan")


def outer_world_quad(model, bounds=None):
    """Corners of the block in mm: TL, TR, BR, BL.

    bounds=(cmin,cmax,rmin,rmax) selects a sub-block; the default is the whole
    3x2 grid."""
    cmin, cmax, rmin, rmax = bounds or (0, COLS - 1, 0, ROWS - 1)
    return np.array([model[(cmin, rmin)][0],
                     model[(cmax, rmin)][1],
                     model[(cmax, rmax)][2],
                     model[(cmin, rmax)][3]], np.float32)


def draw_consensus(vis, quad, lines, color=CONSENSUS_COLOR):
    """Draw the fitted lines (thin) and the consensus quad (thick)."""
    h, w = vis.shape[:2]
    span = float(max(h, w)) * 2.0
    for ln in lines.values():
        if ln is None:
            continue
        vx, vy, x0, y0 = ln
        p1 = (int(x0 - vx * span), int(y0 - vy * span))
        p2 = (int(x0 + vx * span), int(y0 + vy * span))
        cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
    if quad is None:
        return
    cv2.polylines(vis, [np.round(quad).astype(np.int32)], True, color, 3,
                  cv2.LINE_AA)
    for p in quad:
        cv2.circle(vis, tuple(np.round(p).astype(int)), 6, color, -1)


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


def compute_homography_consensus(matches, model):
    """Homography (world_mm -> image) from the consensus quad.

    Four point pairs define a homography exactly, and these four are the
    intersections of lines fitted through all the detected corners, so every
    square contributes to every corner. Returns (H, quad, lines); H is None if
    the quad could not be built."""
    quad, lines, bounds = consensus_quad(matches)
    if quad is None:
        return None, None, lines
    world = outer_world_quad(model, bounds).astype(np.float32)
    H = cv2.getPerspectiveTransform(world, quad.astype(np.float32))
    return H, quad, lines


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


def plane_px_per_mm(H, pts_mm):
    """Local scale of H (camera px per mm on the canvas plane) at given points.

    Returns the max over the points of sqrt(|det J|), i.e. the highest source
    resolution actually available anywhere in the region of interest. Used to
    pick the rectified-view scale so that rectification does not throw away
    camera detail."""
    best = 0.0
    for x, y in pts_mm:
        o = H @ np.array([x, y, 1.0]);        o = o[:2] / o[2]
        ex = H @ np.array([x + 1.0, y, 1.0]); ex = ex[:2] / ex[2] - o
        ey = H @ np.array([x, y + 1.0, 1.0]); ey = ey[:2] / ey[2] - o
        det = abs(ex[0] * ey[1] - ex[1] * ey[0])
        best = max(best, float(np.sqrt(det)))
    return best if best > 1e-6 else 1.0


def corrected_view_transform(H, cam_w, cam_h,
                             margin_mm=CORRECTED_MARGIN_MM, max_side=None):
    """Rectified view framed on the CANVAS (+ margin), at ~native resolution.

    Framing on the canvas rather than on the projected camera frustum matters
    a lot: at a steep viewing angle the frustum covers several meters of wall,
    so a fixed-size output would spend ~90% of its pixels on background and
    resolve the canvas itself at ~1 px/mm - far below the 3-6 px/mm the camera
    actually delivers there. The scale is taken from H's local scale over the
    canvas, capped so the output stays within max_side.

    Returns (A, out_w, out_h) with A: output-px -> world-mm (for
    warpPerspective with WARP_INVERSE_MAP)."""
    max_side = max_side or DISPLAY_MAX     # read late: --view-max can change it
    xmin, ymin = -CANVAS_W - margin_mm, -CANVAS_H - margin_mm
    xmax, ymax = margin_mm, margin_mm
    bw, bh = xmax - xmin, ymax - ymin
    probe = [(-CANVAS_W, -CANVAS_H), (0.0, -CANVAS_H), (0.0, 0.0), (-CANVAS_W, 0.0),
             (-CANVAS_W / 2.0, -CANVAS_H / 2.0)]
    ppm = min(plane_px_per_mm(H, probe), max_side / max(bw, bh))
    out_w = int(round(bw * ppm))
    out_h = int(round(bh * ppm))
    A = np.array([[1.0 / ppm, 0.0, xmin],
                  [0.0, 1.0 / ppm, ymin],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return A, out_w, out_h


def raw_view_transform(cam_w, cam_h, max_side=None):
    """B: output-px -> camera-px for the un-rectified view.

    Native resolution (identity) by default, so the raw view stays exactly as
    sharp as the sensor; the viewport zoom is folded in separately."""
    scale = min(1.0, max_side / float(max(cam_w, cam_h))) if max_side else 1.0
    out_w, out_h = int(round(cam_w * scale)), int(round(cam_h * scale))
    B = np.array([[cam_w / out_w, 0.0, 0.0],
                  [0.0, cam_h / out_h, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return B, out_w, out_h


def view_zoom_matrix(view, out_w, out_h):
    """Z: display-px -> base-output-px for the current zoom/pan viewport.

    Folding the viewport into the warp (instead of cropping and upscaling the
    finished frame) is what makes zoom actually resolve more detail: every
    display pixel is resampled straight from the source frame, and the
    reference contours are re-rasterized at the zoomed scale."""
    z = max(1.0, float(view["z"]))
    rw, rh = out_w / z, out_h / z
    rx = min(max(view["ox"] * out_w, 0.0), max(0.0, out_w - rw))
    ry = min(max(view["oy"] * out_h, 0.0), max(0.0, out_h - rh))
    return np.array([[1.0 / z, 0.0, rx],
                     [0.0, 1.0 / z, ry],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


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
    global FIT_MODE, DIST_K1, DIST_K2
    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("col_off", win, 0, max(0, COLS - 1), lambda v: None)
    cv2.createTrackbar("row_off", win, 0, max(0, ROWS - 1), lambda v: None)

    k1, k2, kstep = DIST_K1, DIST_K2, DIST_STEP
    H_saved = None
    print("[calib] point the camera at the squares. Use the col_off/row_off\n"
          "        trackbars to align the labels with the real positions.\n"
          "        The magenta quad is the consensus fit over all the squares -\n"
          "        its sides should sit on the outer edges of the whole block.\n"
          "        'a' auto-fits the lens distortion; 1/2 3/4 tune k1/k2 by hand\n"
          "        (watch 'line residual': lower = straighter = better),\n"
          "        5/6 change the step, 0 resets the distortion,\n"
          "        'f' switches consensus/per-corner fit, 'c' saves.")

    while True:
        raw = read_frame(cap)
        if raw is None:
            print("[calib] no frame"); break
        # everything below runs on the undistorted frame, so the homography
        # that comes out maps mm -> UNDISTORTED image px (run/overlay undistort
        # to match)
        frame = undistort(raw, k1, k2)
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

        # the consensus quad averaged over all matched squares (magenta)
        cq, clines, _ = consensus_quad(matches)
        draw_consensus(vis, cq, clines, CONSENSUS_COLOR)

        # auto hint: if all 6 (3x2) are visible, the mapping is unambiguous
        auto = (ncols == COLS and nrows == ROWS)
        txt = (f"detected={len(quads)} grid={ncols}x{nrows} "
               f"matched={len(matches)}  off=({col_off},{row_off})"
               f"{'  [AUTO 3x2]' if auto else ''}"
               f"{'' if cq is not None else '  [no consensus quad]'}")
        res = collinearity_residual(matches)
        hud = [txt,
               f"k1={k1:+.4f} k2={k2:+.4f} step={kstep:.4f}   "
               f"line residual={res:.3f} px   fit={FIT_MODE}",
               "1/2 k1  3/4 k2  5/6 step  a auto-fit dist  0 reset dist  "
               "f fit  c save  q quit"]
        y = 30
        for ln in hud:
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            y += 32

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('f'):
            FIT_MODE = "corners" if FIT_MODE == "consensus" else "consensus"
        elif key == ord('1'):
            k1 -= kstep
        elif key == ord('2'):
            k1 += kstep
        elif key == ord('3'):
            k2 -= kstep
        elif key == ord('4'):
            k2 += kstep
        elif key == ord('5'):
            kstep = max(1e-5, kstep / 2.0)
        elif key == ord('6'):
            kstep = min(1.0, kstep * 2.0)
        elif key == ord('0'):
            k1 = k2 = 0.0
        elif key == ord('a'):
            nk1, nk2, r = auto_distortion(matches, frame.shape[1],
                                          frame.shape[0], k1, k2)
            print(f"[calib] auto distortion: k1={nk1:+.4f} k2={nk2:+.4f} "
                  f"(residual {res:.3f} -> {r:.3f} px, over {len(matches)} squares)")
            k1, k2 = nk1, nk2
        if key == ord('c'):
            H_pts = compute_homography(matches, model)
            H_con, cq, _ = compute_homography_consensus(matches, model)
            H = H_con if FIT_MODE == "consensus" else H_pts
            if H is None and H_pts is not None:
                print("[calib] no consensus quad (an outer edge of the block is "
                      "not covered) - falling back to the per-corner fit.")
                H = H_pts
            if H is None:
                print("[calib] need >=4 corners (>=1 valid square, 2+ is better). "
                      "Check the offsets.")
                continue
            np.savez(CALIB_FILE, H=H, px_per_mm=PX_PER_MM,
                     canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                     k1=k1, k2=k2,
                     cam_w=frame.shape[1], cam_h=frame.shape[0])
            DIST_K1, DIST_K2 = k1, k2
            H_saved = H
            print(f"[calib] homography saved to {CALIB_FILE} "
                  f"(fit={FIT_MODE}, from {len(matches)} squares, "
                  f"k1={k1:+.4f} k2={k2:+.4f})")
            print(f"[calib] line residual: {collinearity_residual(matches):.3f} px")
            # Quick confirmation. Both errors are measured over all detected
            # corners, so per-corner is expected to win here almost by
            # definition - it is the least-squares minimizer of exactly this
            # quantity. Read it as a sanity check (a consensus value far above
            # the per-corner one means a square was mis-detected), not as a
            # ranking of the two fits.
            if H_con is not None:
                print(f"[calib] mean reprojection error: "
                      f"consensus={reprojection_error(matches, model, H_con):.2f} px  "
                      f"per-corner={reprojection_error(matches, model, H_pts):.2f} px")
            else:
                print(f"[calib] mean reprojection error: "
                      f"{reprojection_error(matches, model, H):.2f} px")
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
    """Return H, also restoring the saved lens distortion into DIST_K1/DIST_K2.

    H maps mm -> UNDISTORTED image pixels, so every consumer must push its
    frames through undistort(frame, DIST_K1, DIST_K2) first. Files written
    before distortion tuning existed simply carry no k1/k2 and give 0."""
    global DIST_K1, DIST_K2
    try:
        data = np.load(CALIB_FILE)
    except FileNotFoundError:
        return None
    DIST_K1 = float(data["k1"]) if "k1" in data else 0.0
    DIST_K2 = float(data["k2"]) if "k2" in data else 0.0
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
        frame = undistort(frame, DIST_K1, DIST_K2)
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
RENDER_MODES = ["contours", "image", "multiply"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Arrow keys, as reported by cv2.waitKeyEx on the various HighGUI backends
# (GTK, Qt, Win32). They have to be matched BEFORE the usual `& 0xFF`, because
# masking collapses them onto letter codes (65361 & 0xFF == ord('Q'), 65363 &
# 0xFF == ord('S')); the masked forms are listed too, since some builds hand
# back the already-truncated value.
KEY_PREV = {81, 65361, 2424832, 63234}
KEY_NEXT = {83, 65363, 2555904, 63235}


def list_reference_images(path):
    """Sorted list of the reference images that --ref names.

    A directory expands to every image inside it (alphabetically); a single
    file becomes a one-element list, so nothing downstream special-cases the
    two forms."""
    if os.path.isdir(path):
        return [os.path.join(path, n) for n in sorted(os.listdir(path))
                if n.lower().endswith(IMAGE_EXTS)]
    return [path]


def draw_ref_dots(img, idx, total, margin=16, r=7, gap=22):
    """Top-right row of dots marking which reference of the set is showing.

    Past a couple of dozen files the dots stop being countable at a glance, so
    it falls back to a plain N/M counter."""
    if total <= 1:
        return
    h, w = img.shape[:2]
    if total > 24:
        label = f"{idx + 1}/{total}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        org = (w - margin - tw, margin + th)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return
    x, y = w - margin - r, margin + r
    for i in range(total - 1, -1, -1):
        cv2.circle(img, (x, y), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (0, 255, 255) if i == idx else (80, 80, 80),
                   -1, cv2.LINE_AA)
        x -= gap


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


def edges_to_polylines(edges):
    """Vectorize an edge mask -> (points (N,1,2) float32, per-contour lengths).

    The overlay draws transformed polylines instead of warping the binary mask,
    because a 1-px mask survives neither direction of resampling: warped into a
    smaller destination it decimates into dots, into a larger one it blows up
    into blocks. Polylines stay thin and continuous at any scale."""
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if len(c) >= 2]
    if not cnts:
        return np.zeros((0, 1, 2), np.float32), []
    return np.vstack(cnts).astype(np.float32), [len(c) for c in cnts]


def draw_polylines_blended(canvas, pts, lens, M, color, alpha):
    """Draw ref-space polylines through M onto canvas, alpha-blended."""
    if len(pts) == 0:
        return
    tp = cv2.perspectiveTransform(pts, M)
    # a point mapped near the horizon can be astronomically far away; clamp
    # before the int cast (it is off-screen either way)
    np.clip(tp, -1e5, 1e5, out=tp)
    tp = np.round(tp).astype(np.int32)
    # work inside the bbox of the contours only - blending over the full frame
    # costs more than the drawing itself
    ch, cw = canvas.shape[:2]
    x0 = int(np.clip(tp[:, 0, 0].min() - 1, 0, cw))
    y0 = int(np.clip(tp[:, 0, 1].min() - 1, 0, ch))
    x1 = int(np.clip(tp[:, 0, 0].max() + 2, 0, cw))
    y1 = int(np.clip(tp[:, 0, 1].max() + 2, 0, ch))
    if x1 <= x0 or y1 <= y0:
        return
    tp -= np.array([x0, y0], np.int32)      # lines crossing in from outside still clip correctly
    polys = np.split(tp, np.cumsum(lens)[:-1])
    roi = canvas[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.polylines(mask, polys, False, 255, 1, cv2.LINE_AA)
    m = mask > 0
    if not m.any():
        return
    w = (mask[m].astype(np.float32) * (alpha / 255.0))[:, None]
    roi[m] = (roi[m] * (1.0 - w) + np.array(color, np.float32) * w).astype(np.uint8)


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
    refs = list_reference_images(ref_path)
    if not refs:
        print(f"[overlay] no reference images in {ref_path}"); return

    canny_lo, canny_hi, blur = 50, 150, 3
    # shrunk color copy of the current reference (for image mode and contours)
    ref_small, epts, elens, rh, rw, ref_i = None, None, None, 0, 0, 0

    def rebuild_edges():
        nonlocal epts, elens
        edges = extract_edges(ref_small, canny_lo, canny_hi, blur,
                              max_side=10 ** 9)
        epts, elens = edges_to_polylines(edges)

    def load_ref(i):
        """Swap in reference #i, leaving every other setting alone.

        Only ref_small/edges change, so the alignment, opacity, Canny
        thresholds, render mode and viewport all carry over to the next image -
        which is the point of stepping through a folder."""
        nonlocal ref_small, rh, rw, ref_i
        img = cv2.imread(refs[i])
        if img is None:
            print(f"[overlay] could not read the reference: {refs[i]}")
            return False
        ref_i = i
        ref_small = _resize_max(img, 1600)
        rh, rw = ref_small.shape[:2]
        rebuild_edges()
        print(f"[overlay] reference {i + 1}/{len(refs)}: {refs[i]}")
        return True

    if not load_ref(0):
        return
    dx, dy, sx, sy, theta, alpha = load_overlay_adjust(adjust_path)
    color_i = 0
    show = True
    show_help = True
    rectified = False
    render_mode = "contours"       # "contours" | "image" | "multiply"

    MOVE, SCALE, ROT, DA = 2.0, 1.01, 0.5, 0.05  # steps
    # view transforms; lazy (they need the frame size)
    A = B = None            # corrected: out-px -> world-mm; raw: out-px -> cam-px
    ow = oh = 0             # rendered view size for the current mode

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
          "          m contours/image/multiply, 9/0 opacity -/+,\n"
          "          LEFT/RIGHT arrows = previous/next reference in the folder,\n"
          "          1/2 3/4 Canny, o toggle, c color, r raw/corrected,\n"
          "          MOUSE: wheel or right button (drag) = zoom at cursor,\n"
          "          left button (drag) = pan, SPACE = reset zoom,\n"
          "          p save, i reset adjustment, h help, q quit")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[overlay] no frame"); break
        # H was fitted on undistorted pixels, so BOTH views (raw and corrected)
        # have to start from the undistorted frame or the overlay drifts at the
        # edges exactly where the lens bends the most
        frame = undistort(frame, DIST_K1, DIST_K2)
        # --- view transform: display px -> camera px, zoom folded in ---------
        if rectified:
            if A is None:
                A, ow, oh = corrected_view_transform(H, frame.shape[1], frame.shape[0])
                print(f"[overlay] corrected view {ow}x{oh} px "
                      f"({1.0 / A[0, 0]:.2f} px/mm; camera gives "
                      f"{plane_px_per_mm(H, [(-CANVAS_W / 2, -CANVAS_H / 2)]):.2f} "
                      f"px/mm at the canvas center)")
            src = H @ A @ view_zoom_matrix(view, ow, oh)
        else:
            if B is None:
                B, ow, oh = raw_view_transform(frame.shape[1], frame.shape[0])
            src = B @ view_zoom_matrix(view, ow, oh)

        if np.allclose(src, np.eye(3)):
            disp = frame.copy()                  # raw at zoom 1: no resampling
        else:
            disp = cv2.warpPerspective(frame, src, (ow, oh),
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        ch, cw = disp.shape[:2]

        if show:
            T = ref_to_world(rw, rh, dx, dy, sx, sy, theta)
            M = np.linalg.inv(src) @ H @ T        # ref pixel -> display pixel
            if render_mode == "contours":
                draw_polylines_blended(disp, epts, elens, M,
                                       OVERLAY_COLORS[color_i], alpha)
            else:
                warped = cv2.warpPerspective(ref_small, M, (cw, ch),
                                             flags=cv2.INTER_LINEAR)
                mcov = cv2.warpPerspective(
                    np.full(ref_small.shape[:2], 255, np.uint8), M, (cw, ch),
                    flags=cv2.INTER_NEAREST) > 0
                over = warped[mcov].astype(np.float32)
                if render_mode == "multiply":
                    # multiply only ever darkens, so the reference reads as ink
                    # laid over the canvas: white paper in the ref leaves the
                    # camera image untouched and what is drawn on the real
                    # canvas stays visible through the dark areas.
                    over *= disp[mcov] / 255.0
                disp[mcov] = (disp[mcov] * (1.0 - alpha)
                              + over * alpha).astype(np.uint8)

        view["W"], view["H"] = cw, ch

        if show_help:
            hud = [f"ref {ref_i + 1}/{len(refs)}: {os.path.basename(refs[ref_i])}",
                   f"dx={dx:+.0f}mm dy={dy:+.0f}mm  sx={sx:.3f} sy={sy:.3f}  rot={theta:+.1f}deg",
                   f"mode={render_mode}  alpha={alpha:.2f}  zoom={view['z']:.1f}x  "
                   f"Canny={canny_lo}/{canny_hi}  overlay={'on' if show else 'off'}  "
                   f"view={'corrected' if rectified else 'raw (perspective)'}"
                   f"{'' if (DIST_K1 or DIST_K2) else ' no-undistort'}  "
                   f"render={cw}x{ch}"
                   + (f" {view['z'] / A[0, 0]:.1f}px/mm" if rectified else ""),
                   "w/a/s/d move  z/x scale  [ ] X  - = Y  ,/. rot  m mode  9/0 alpha  "
                   "1/2 3/4 canny  <-/-> ref  o c r  |  mouse: wheel/RMB zoom, LMB pan"]
            y0 = 26
            for ln in hud:
                cv2.putText(disp, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 1, cv2.LINE_AA)
                y0 += 24

        draw_ref_dots(disp, ref_i, len(refs))

        cv2.imshow(win, disp)
        # waitKeyEx, not waitKey: the arrow keys have no 8-bit code, and their
        # full codes collide with letters once masked (see KEY_PREV/KEY_NEXT).
        key = cv2.waitKeyEx(1)
        if key in KEY_PREV or key in KEY_NEXT:
            # deliberately no wrap-around: the ends of the folder are a stop,
            # so stepping through it never silently loops back
            nxt = ref_i + (1 if key in KEY_NEXT else -1)
            if 0 <= nxt < len(refs):
                load_ref(nxt)
            continue
        k = key & 0xFF
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
            render_mode = RENDER_MODES[
                (RENDER_MODES.index(render_mode) + 1) % len(RENDER_MODES)]
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
            rebuild_edges()

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
            # actually look at the pixels - a mode can stream pure black
            lit = False
            for _ in range(20):
                ok, f = cap.read()
                if ok and f is not None and f.max() > 0:
                    lit = True
                    break
            state = "OK" if lit else "BLACK (mode not really supported)"
            print(f"  #{i}: OPEN  {w}x{h}  {state}")
            cap.release()
        else:
            print(f"  #{i}: -")


def main():
    global PX_PER_MM, CANVAS_W, CANVAS_H, REQ_WIDTH, REQ_HEIGHT, DISPLAY_MAX
    global FIT_MODE, DIST_K1, DIST_K2
    ap = argparse.ArgumentParser(description="Canvas perspective calibration/rectification")
    ap.add_argument("mode", choices=["calibrate", "run", "overlay", "gen-template", "list"])
    ap.add_argument("--cam", type=int, default=CAM_INDEX)
    ap.add_argument("--width", type=int, default=None,
                    help=f"capture width (default {REQ_WIDTH})")
    ap.add_argument("--height", type=int, default=None,
                    help=f"capture height (default {REQ_HEIGHT})")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="scale of the rectified image")
    ap.add_argument("--ref", default="nadya111.jpg",
                    help="reference image for overlay mode, or a folder of "
                         "them - then the first one (alphabetically) opens and "
                         "LEFT/RIGHT step through the rest")
    ap.add_argument("--canvas-w-in", type=float, default=None,
                    help="canvas width in inches (the sheet is always bottom-right)")
    ap.add_argument("--canvas-h-in", type=float, default=None,
                    help="canvas height in inches")
    ap.add_argument("--out", default=None, help="output file for gen-template")
    ap.add_argument("--adjust", default=OVERLAY_ADJUST_FILE,
                    help="file for saved overlay adjustment parameters (.npz)")
    ap.add_argument("--fit", choices=["consensus", "corners"], default=None,
                    help=f"how to fit the homography (default {FIT_MODE}): "
                         f"'consensus' = one quad fitted to all six squares, "
                         f"'corners' = least squares over the individual corners")
    ap.add_argument("--k1", type=float, default=None,
                    help="initial radial distortion k1 for calibrate "
                         "(tunable there with 1/2)")
    ap.add_argument("--k2", type=float, default=None,
                    help="initial radial distortion k2 for calibrate "
                         "(tunable there with 3/4)")
    ap.add_argument("--view-max", type=int, default=None,
                    help=f"longest side of the corrected overlay view "
                         f"(default {DISPLAY_MAX}; higher = sharper but slower)")
    args = ap.parse_args()

    if args.width:
        REQ_WIDTH = args.width
    if args.height:
        REQ_HEIGHT = args.height
    if args.px_per_mm:
        PX_PER_MM = args.px_per_mm
    if args.view_max:
        DISPLAY_MAX = args.view_max
    if args.canvas_w_in:
        CANVAS_W = args.canvas_w_in * MM_PER_IN
    if args.canvas_h_in:
        CANVAS_H = args.canvas_h_in * MM_PER_IN
    if args.fit:
        FIT_MODE = args.fit
    # start calibration from the distortion tuned last time instead of from
    # zero, so recalibrating does not throw the lens settings away
    if args.mode == "calibrate":
        load_calibration()
    if args.k1 is not None:
        DIST_K1 = args.k1
    if args.k2 is not None:
        DIST_K2 = args.k2

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
