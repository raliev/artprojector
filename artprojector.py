#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
artprojector.py
===============

Calibrate and rectify the perspective of a canvas using a target of six
~63 mm squares (calibr.svg) printed on an 8.5x11" sheet that sits flush with
the right and bottom edges of a 12x16" canvas.

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
  python artprojector.py list                # list cameras and monitors

  --fullscreen --display DP-1                # where the window opens
  --no-keep-awake                            # let the machine sleep

While a session runs, sleep and the screensaver are held off two ways at once
(a systemd-inhibit/caffeinate child, plus a poke every 45 s) - a suspend would
re-enumerate the camera, and the camera must not move.

In the calibrate window:
  col_off / row_off trackbars - shift the visible block over the 3x2 grid
  a  - auto-fit the lens distortion (straightens the target's grid lines)
  1/2, 3/4 - tune the radial distortion k1 / k2 by hand
  5/6 - halve / double the tuning step;  0 - reset the distortion
  f  - switch the homography fit: consensus quad <-> individual corners
  e  - toggle the sub-pixel snap of the fit onto the printed lines
  r  - let the snap retune k1/k2 as well
  c  - compute and save the homography (+ the distortion)
  q / ESC - quit

In the run window:
  c  - go back to calibration
  g  - toggle the metric grid
  q / ESC - quit
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
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

COLS = 3                           # squares horizontally
ROWS = 2                           # squares vertically

# --------------------------------------------------------------------------
# THE PRINTED TARGET - taken verbatim from calibr.svg.
#
# The SVG is an A4 page with 1 user unit = 1 mm, so its numbers ARE the
# millimeters on the paper (printed at 100%, "actual size" - see below). The
# six rectangles were placed by hand in Inkscape and are NOT on an exact grid:
# the column pitch is 64.407 and 64.681 mm, the rows are 64.333 mm apart, and
# the tops of the three squares in a row differ by up to 0.22 mm. So the real
# coordinates are used as they are instead of a nominal side + pitch, which is
# what "64 mm squares, touching, 0.6 mm between rows" used to assume - that
# model was ~1.1 mm too tall over the block and its width was right by luck.
#
# All numbers below are the STROKE CENTRELINE (an SVG stroke straddles the
# rectangle path, 0.406 mm to each side), which is what detect_squares()
# measures once it averages the outer and the inner contour of a frame.
#
# Sanity checks that this really is the printed sheet: the white gap left
# between the strokes comes out at 0.62 mm horizontally and 0.54 mm vertically
# - the ~0.6 mm that was measured off the photo - and the sheet's own bottom
# margin, 279.4 (Letter) - 134.93 = 144.5 mm, is the 145 mm measured with a
# ruler. Both only work if the page was printed at 100%, not fitted to Letter
# (fit-to-page would have scaled everything by 0.9407).
# --------------------------------------------------------------------------
SQUARE_MM = 62.977863              # rectangle side, stroke centreline
STROKE_MM = 0.811876               # printed line width

# top-left corner of each rectangle in SVG page mm, keyed (col, row)
SHEET_RECTS_MM = {
    (0, 0): (7.7011156, 6.9964180),
    (1, 0): (72.107773, 7.0334225),
    (2, 0): (136.78879, 7.2159476),
    (0, 1): (7.6948276, 71.329369),
    (1, 1): (72.101486, 71.366371),
    (2, 1): (136.78250, 71.548897),
}

# Anchoring to the canvas edges (= sheet edges, since the sheet is flush
# right/bottom). Measured with a ruler to the OUTER edge of the black line;
# build_model() takes off half a stroke to get to the centreline. An error
# here is a pure translation of everything - fix it once with w/a/s/d in
# overlay and save with 'p'.
RIGHT_MARGIN_MM = 14.0             # canvas right edge -> right line of the RIGHTMOST square
BOTTOM_MARGIN_MM = 145.0           # canvas bottom edge -> bottom line of the BOTTOM square

# ==========================================================================
#  KEEPING THE MACHINE AWAKE
#
#  A painting session is hours of looking at the screen and not touching the
#  keyboard, which is exactly what every idle timer is built to punish. Worse,
#  a suspend is not merely annoying here: the camera must not move between
#  calibrate and overlay, and coming back from sleep usually means the USB
#  camera is re-enumerated and the stream has to be reopened.
#
#  Two independent mechanisms run at once, because they fail in different
#  ways. The inhibitor is the correct one - it tells the session manager not
#  to sleep, and it holds for as long as the child process is alive - but it
#  depends on logind (or macOS) being there and on the desktop honouring it,
#  and a lock screen may come up regardless. The poke is the crude one: every
#  45 seconds it tells the screensaver that somebody is still here. Neither is
#  reliable across every desktop; both failing at once is unlikely.
#
#  Everything is best-effort and silent: a machine that cannot be kept awake
#  is not a reason to refuse to run.
# ==========================================================================
KEEP_AWAKE = True
_POKE_PERIOD_S = 45.0


class KeepAwake:
    """Hold off sleep and the screensaver for as long as this is alive."""

    def __init__(self, why="artprojector session"):
        self.why = why
        self._proc = None
        self._stop = threading.Event()
        self._thread = None
        self._poke = None          # the poke command that worked, if any

    # -- mechanism 1: a held inhibitor ------------------------------------
    def _inhibitor_cmd(self):
        if platform.system() == "Darwin":
            # -dimsu: display, idle, disk, system; -w dies with us
            return ["caffeinate", "-dimsu", "-w", str(os.getpid())]
        return ["systemd-inhibit", "--what=idle:sleep:handle-lid-switch",
                "--who=artprojector", f"--why={self.why}", "--mode=block",
                "sleep", "infinity"]

    # -- mechanism 2: periodic "somebody is still here" -------------------
    def _poke_cmds(self):
        if platform.system() == "Darwin":
            return [["caffeinate", "-u", "-t", "5"]]
        return [
            # freedesktop, spoken by both KDE and GNOME
            ["gdbus", "call", "--session", "--dest", "org.freedesktop.ScreenSaver",
             "--object-path", "/org/freedesktop/ScreenSaver",
             "--method", "org.freedesktop.ScreenSaver.SimulateUserActivity"],
            ["xdg-screensaver", "reset"],
            ["xset", "s", "reset"],
        ]

    def _run(self, cmd):
        try:
            return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  timeout=10).returncode == 0
        except Exception:
            return False

    def _loop(self):
        while not self._stop.wait(_POKE_PERIOD_S):
            if self._poke and not self._run(self._poke):
                self._poke = None          # it stopped working - stop trying
            if self._poke is None:
                break

    def start(self):
        cmd = self._inhibitor_cmd()
        if shutil.which(cmd[0]):
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            except Exception:
                self._proc = None
        for c in self._poke_cmds():
            if shutil.which(c[0]) and self._run(c):
                self._poke = c
                break
        if self._poke:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        held = ", ".join(
            n for n, ok in (("inhibitor", self._proc is not None),
                            (f"poke ({self._poke[0]})" if self._poke else "poke",
                             self._poke is not None)) if ok)
        print(f"[awake] {held or 'nothing worked - the machine may still sleep'}")
        return self

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# ==========================================================================
#  DISPLAYS AND WINDOW PLACEMENT
# ==========================================================================
FULLSCREEN = False
DISPLAY_TARGET = None              # monitor index or name to open windows on


def list_displays():
    """[(name, x, y, w, h)] for the monitors, as the WINDOWS see them.

    xrandr comes first on purpose. An OpenCV window is a Qt window on an X11
    screen - XWayland included - so it is xrandr's idea of where the monitors
    are that decides where cv2.moveWindow() actually puts things, even when
    the session is Wayland and the compositor's own layout says otherwise.
    kscreen-doctor is only a fallback for naming them.

    Note that under XWayland a mirrored pair reports both monitors at +0+0,
    and then no coordinate can tell them apart - `--display` cannot help there
    and the window will land on whichever one the compositor picks."""
    try:
        out = subprocess.run(["xrandr", "--listmonitors"], timeout=5,
                             capture_output=True, text=True).stdout
        mons = []
        for ln in out.splitlines()[1:]:
            m = re.search(r"(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)\s+(\S+)", ln)
            if m:
                w, h, x, y, name = m.groups()
                mons.append((name, int(x), int(y), int(w), int(h)))
        if mons:
            return mons
    except Exception:
        pass
    try:
        from screeninfo import get_monitors
        return [(m.name or f"display{i}", m.x, m.y, m.width, m.height)
                for i, m in enumerate(get_monitors())]
    except Exception:
        return []


def _display_index(mons, target):
    """Resolve --display (an index, or a monitor name) to a position in mons."""
    if target is None or not mons:
        return None
    try:
        i = int(target)
        return i if 0 <= i < len(mons) else None
    except (TypeError, ValueError):
        pass
    for i, (name, *_rest) in enumerate(mons):
        if name.lower() == str(target).lower():
            return i
    return None


def make_window(name, fullscreen=None, display=None):
    """Create a resizable window, optionally on a given monitor / fullscreen.

    The move has to happen before the fullscreen flag: what a window manager
    fullscreens onto is the monitor the window is currently on."""
    fullscreen = FULLSCREEN if fullscreen is None else fullscreen
    display = DISPLAY_TARGET if display is None else display
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    mons = list_displays()
    i = _display_index(mons, display)
    if i is not None:
        _n, x, y, w, h = mons[i]
        try:
            cv2.moveWindow(name, x, y)
            cv2.resizeWindow(name, w, h)
        except cv2.error:
            pass
    elif display is not None:
        print(f"[display] no monitor '{display}' - "
              f"{len(mons)} found; see 'artprojector.py list'")
    if fullscreen:
        try:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            pass
    return name


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
REFINE_COLOR = (255, 255, 0)       # cyan - the refined fit, drawn over the ink


# ==========================================================================
#  MODEL: mm corner coordinates of each square in the canvas frame.
#  Origin - the BOTTOM-RIGHT corner of the canvas (= sheet corner), X right, Y down.
#  Thus the canvas occupies X in [-CANVAS_W, 0], Y in [-CANVAS_H, 0], and the
#  square coordinates do NOT depend on the canvas size (the sheet is always in
#  the bottom-right corner) - no need to recalibrate when the canvas size changes.
# ==========================================================================
def build_model():
    """Return dict {(col,row): (4,2) float32} of corners TL,TR,BR,BL in mm.

    The sheet layout comes from SHEET_RECTS_MM (page coordinates); it is
    shifted so that the outermost printed lines sit at the measured margins
    from the canvas corner."""
    # centreline of the rightmost / bottom-most printed line, in page mm
    right = max(x for (x, _) in SHEET_RECTS_MM.values()) + SQUARE_MM
    bottom = max(y for (_, y) in SHEET_RECTS_MM.values()) + SQUARE_MM
    # the margins are measured to the outer edge of that line
    x_anchor = -(RIGHT_MARGIN_MM - STROKE_MM / 2.0)
    y_anchor = -(BOTTOM_MARGIN_MM - STROKE_MM / 2.0)

    model = {}
    s = SQUARE_MM
    for (col, row), (px, py) in SHEET_RECTS_MM.items():
        x0 = px - right + x_anchor                       # left line of this square
        y0 = py - bottom + y_anchor                      # top line of this square
        model[(col, row)] = np.array([
            [x0,     y0],       # TL
            [x0 + s, y0],       # TR
            [x0 + s, y0 + s],   # BR
            [x0,     y0 + s],   # BL
        ], dtype=np.float32)
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


# Capture modes offered for switching at runtime, largest first. Only the ones
# matching the calibrated aspect ratio are ever used (see scale_homography), so
# this list can name modes of several shapes safely.
CAPTURE_MODES = [(2592, 1944), (1920, 1440), (1600, 1200), (1280, 960),
                 (1920, 1080), (1280, 720), (800, 600), (640, 480)]


def set_capture_mode(cap, w, h, settle=8, timeout=2.5):
    """Retune a live capture to w x h. Returns the size actually in effect.

    The request is only a request: V4L2 substitutes the nearest supported mode
    without saying so, the first frames after a switch are still the old size or
    black while the sensor re-exposes, and some modes on this class of camera
    stream all-black forever (open_camera fights the same thing). So the frames
    decide, not CAP_PROP_FRAME_*, and a mode that will not produce a picture
    within `timeout` returns None for the caller to fall back on."""
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    good = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, f = cap.read()
        if ok and f is not None and f.max() > 0:
            good = f
            settle -= 1
            if settle <= 0:
                break
        else:
            time.sleep(0.02)
    if good is None:
        return None
    return good.shape[1], good.shape[0]


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
    """Return (quads, n_paired): (4,2) corner arrays (TL,TR,BR,BL) and how
    many of them came from a complete outer+inner contour pair.

    Works with OUTLINED squares (black frame on light paper): black lines ->
    foreground, take all 4-gon contours of the right size, then group them by
    center. Each printed frame gives TWO of them - the outside of the black
    stroke and the hole inside it - and they are averaged into the STROKE
    CENTRELINE.

    Keeping the outer contour instead (as this used to) measures every square
    a full stroke width too big - 0.81 mm, which is 0.4% of the block and
    ends up as ~1 mm of drift at the far corner of the canvas. Worse, it is
    not even a stable bias: whether the outer contour survives at all depends
    on exposure and on whether the blur merges the strokes of neighbouring
    squares, so the same sheet can measure 0.8 mm differently between two
    runs. The centreline is what the SVG rectangle is and what build_model()
    returns, and it is immune to how fat the printer laid the ink down.
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

    if not cand:
        return [], 0

    # sub-pixel corners, per contour, BEFORE the two contours of a frame are
    # merged (afterwards there is no image feature at the averaged corner)
    corners = np.vstack([q for (_, q, _) in cand]).astype(np.float32)
    cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    cand = [(a, corners[i * 4:(i + 1) * 4], corners[i * 4:(i + 1) * 4].mean(axis=0))
            for i, (a, _, _) in enumerate(cand)]

    # Group by center: the outside and the hole of one frame share a center.
    # Size has to agree too - the two sides of a stroke differ by ~2%, so a
    # quad that is merely concentric and much bigger (the outline of the whole
    # block, or the sheet itself) is a different object, not a partner.
    cand.sort(key=lambda t: -t[0])
    groups = []                        # list of [(area, quad), ...]
    for area, quad, cen in cand:
        for g in groups:
            if (np.linalg.norm(cen - g[0][1].mean(axis=0)) < 0.45 * np.sqrt(area)
                    and np.sqrt(area / g[0][0]) > 0.75):
                g.append((area, quad))
                break
        else:
            groups.append([(area, quad)])

    quads, areas, solo = [], [], []
    for g in groups:
        outer, inner = g[0], g[-1]     # sorted by area, descending
        ratio = np.sqrt(inner[0] / outer[0])
        if len(g) >= 2 and 0.80 <= ratio < 1.0:
            quads.append((outer[1] + inner[1]) / 2.0)   # <- the centreline
            areas.append(0.5 * (outer[0] + inner[0]))
            solo.append(False)
        else:
            quads.append(outer[1])
            areas.append(outer[0])
            solo.append(True)

    # size-consistency filter: real squares are ~equal in size, so drop
    # outliers (stray rectangles at a different scale)
    if len(quads) >= 3:
        med = float(np.median(areas))
        keep = [i for i, a in enumerate(areas) if 0.4 * med <= a <= 2.5 * med]
        quads = [quads[i] for i in keep]
        areas = [areas[i] for i in keep]
        solo = [solo[i] for i in keep]

    # Drop anything that swallows another square's centre. When the 0.6 mm
    # white gap between two squares closes up - at a distance, or under blur -
    # their strokes merge and the pair (or the whole block, or a column) traces
    # one more rounded rectangle, which is convex, four-sided and close enough
    # in size to survive the filters above. No printed square ever contains the
    # centre of another one, so containment identifies those cleanly, and
    # without them assign_local_grid() does not see a phantom extra column.
    if len(quads) > 1:
        cents = [q.mean(axis=0) for q in quads]
        drop = set()
        for i, q in enumerate(quads):
            for j, c in enumerate(cents):
                if i != j and cv2.pointPolygonTest(q.astype(np.float32),
                                                   tuple(float(v) for v in c),
                                                   False) > 0:
                    drop.add(i)
                    break
        if len(drop) < len(quads):
            keep = [i for i in range(len(quads)) if i not in drop]
            quads = [quads[i] for i in keep]
            solo = [solo[i] for i in keep]

    # A frame whose partner contour was lost keeps half a stroke of bias; it is
    # reported (n_paired) rather than guessed at, and refine_homography() takes
    # it out anyway by measuring the printed lines directly.
    return quads, sum(1 for s in solo if not s)


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


def draw_model(vis, H, model, color=(255, 255, 0), thickness=1):
    """Draw the model squares projected through H (a look at the fit itself)."""
    for corners in model.values():
        p = np.hstack([np.asarray(corners, np.float64), np.ones((4, 1))]) @ H.T
        p = p[:, :2] / p[:, 2:3]
        cv2.polylines(vis, [np.round(p).astype(np.int32)], True, color,
                      thickness, cv2.LINE_AA)


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
#  SUB-PIXEL REFINEMENT - "the calibration of the calibration"
#
#  Everything above measures the target through its CORNERS: 24 of them, and
#  they are the worst-localised features on the sheet - a corner is where two
#  fat printed lines meet and where the contour tracer, the blur and the
#  sub-pixel snapper all disagree the most. The consensus quad then throws
#  away all but four of them.
#
#  But the sheet knows far more about itself than that. It has 24 straight
#  printed edges, several thousand pixels of them, and each one is a dark
#  ridge whose centre can be found across the profile to a fraction of a
#  pixel, far better than any endpoint. This stage uses them: project every
#  model edge into the RAW frame with the current estimate, walk across it,
#  find the darkest point, and solve for the homography (and optionally
#  k1/k2) that lands the projected lines on the measured ones.
#
#  That also makes the result self-checking, which corner fits never were.
#  A homography fitted to four corners always maps the model block onto the
#  imaged block exactly, so it reports a small error even when the model
#  geometry is wrong - the error just moves somewhere else on the canvas,
#  where nothing is measuring it. Fitting 24 lines at once cannot be fooled
#  that way: a model whose squares are the wrong size or pitch cannot put all
#  of them on ink simultaneously, and the leftover residual (printed in mm)
#  says so.
# ==========================================================================
REFINE_STEP_MM = 1.5       # spacing of sample points along an edge
REFINE_TRIM_MM = 4.0       # skip this much at both ends (corners are rounded)
REFINE_SEARCH_MM = 0.62    # how far to look across a line. MUST stay below half
                           # the 1.355 mm gap to the neighbouring square's line
                           # (0.68) or a sample can lock onto the wrong one, and
                           # above half a stroke (0.41), which is how far off the
                           # start can be when only one contour of a frame was
                           # found
REFINE_COARSE_MM = 3.0     # ditto for the coarse pass, which only uses the
                           # four outer edges of the block - nothing else is
                           # within 60 mm of them, so it can look much further
REFINE_MIN_CONTRAST = 12.0  # gray levels between the paper and the line
REFINE_ITERS = 6


EDGE_NAMES = ("top", "right", "bottom", "left")


def model_edge_samples(model, keys=None, step_mm=REFINE_STEP_MM,
                       trim_mm=REFINE_TRIM_MM, outer_only=False,
                       with_labels=False):
    """Points along every printed edge, with the edge normal. Both in mm.

    Returns (pts (N,2), normals (N,2)), plus a per-point (col,row,edge) label
    list when with_labels is set. The normal points INTO the square, which
    measure_line_offsets() relies on: that is the side with 63 mm of blank
    paper behind it.

    outer_only keeps just the four edges that bound the whole block. Those are
    the only lines with no other line within 60 mm of them, so they can be
    searched for from far away without any risk of locking onto a neighbour -
    which is what the coarse pass in refine_homography() needs."""
    keys = set(model) if keys is None else set(keys)
    cmin = min(c for (c, _) in keys); cmax = max(c for (c, _) in keys)
    rmin = min(r for (_, r) in keys); rmax = max(r for (_, r) in keys)
    pts, nrm, lab = [], [], []
    for key, c in model.items():
        if key not in keys:
            continue
        col, row = key
        # TL->TR is the top edge, TR->BR the right one, and so on
        use = [row == rmin, col == cmax, row == rmax, col == cmin]
        c = np.asarray(c, np.float64)
        for i in range(4):
            if outer_only and not use[i]:
                continue
            a, b = c[i], c[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            if L <= 2 * trim_mm + step_mm:
                continue
            d = (b - a) / L
            n = np.array([-d[1], d[0]])          # 90 deg from the edge
            t = np.arange(trim_mm, L - trim_mm + 1e-9, step_mm)
            pts.append(a + np.outer(t, d))
            nrm.append(np.repeat(n[None, :], len(t), axis=0))
            lab += [(col, row, i)] * len(t)
    if not pts:
        empty = (np.zeros((0, 2)), np.zeros((0, 2)))
        return empty + ([],) if with_labels else empty
    out = (np.vstack(pts), np.vstack(nrm))
    return out + (lab,) if with_labels else out


def project_raw(pts_mm, H, k1, k2, w, h):
    """world mm -> RAW (still distorted) sensor pixels.

    Refinement works on the raw frame on purpose: no resampling stands between
    the measurement and the sensor, and k1/k2 stay ordinary parameters of the
    projection chain instead of something baked into the image."""
    p = np.hstack([np.asarray(pts_mm, np.float64),
                   np.ones((len(pts_mm), 1))]) @ np.asarray(H, np.float64).T
    p = p[:, :2] / p[:, 2:3]
    if abs(k1) < 1e-12 and abs(k2) < 1e-12:
        return p
    return distort_points(p, k1, k2, w, h).astype(np.float64)


def sample_gray(gray, pts):
    """Bilinear sample of a gray image at float points (...,2). NaN outside."""
    h, w = gray.shape[:2]
    x, y = pts[..., 0], pts[..., 1]
    ok = (x >= 0) & (x <= w - 2) & (y >= 0) & (y <= h - 2)
    x0 = np.clip(np.floor(x), 0, w - 2).astype(np.int32)
    y0 = np.clip(np.floor(y), 0, h - 2).astype(np.int32)
    fx, fy = x - x0, y - y0
    g = gray.astype(np.float32)
    v = (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy) +
         g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def measure_line_offsets(gray, pts_mm, nrm_mm, H, k1, k2,
                         search_mm=REFINE_SEARCH_MM, ink_mm=None):
    """How far each projected sample sits from the printed line it belongs to.

    Returns (p_img (N,2), n_img (N,2), offset_px (N,), good (N,) bool), the
    offset signed along n_img.

    The line is located by its INNER FLANK - the step from the square's empty
    interior into the ink - and not by the darkest point across it.

    The darkest point is the obvious thing to look for, and it is what this did
    first, but it does not survive contact with the sheet. Neighbouring squares
    are 1.355 mm apart while their lines are 0.81 mm wide, so the white gap
    between an inner pair of lines is barely half a millimetre: at the ~7 px/mm
    a real frame gives, that is three or four pixels. Blur fills it in, each
    line's dip gets dragged toward its neighbour, and the bottom of the merged
    trough goes flat enough for noise to move the minimum a whole pixel between
    adjacent samples. Measured on a real sheet, the fourteen inner edges came
    out at 0.145 mm against 0.098 mm for the ten edges on the block outline,
    which have no neighbour within 60 mm.

    Every square is blank inside for 63 mm, so the inner flank is clean for all
    24 edges, with no special cases. And the position of a blurred step is
    where its gradient peaks - somewhere symmetric blur, unlike a merging
    neighbour, does not move it. The price is that the measurement now depends
    on how wide the printer actually laid the ink down, so refine_homography()
    solves for that as one more unknown (info["ink_mm"])."""
    h, w = gray.shape[:2]
    ink = STROKE_MM / 2.0 if ink_mm is None else float(ink_mm)
    p0 = project_raw(pts_mm, H, k1, k2, w, h)
    # the normal in image space: where 0.5 mm along it lands. nrm_mm points
    # INTO the square (see model_edge_samples), so +u walks off the ink into
    # the empty interior
    p1 = project_raw(pts_mm + nrm_mm * 0.5, H, k1, k2, w, h)
    nv = p1 - p0
    ln = np.linalg.norm(nv, axis=1, keepdims=True)
    px_per_mm = np.maximum(ln / 0.5, 1e-9)                  # (N,1) local scale
    nv = nv / np.maximum(ln, 1e-9)

    # window around where the flank should be, wide enough to still find it
    # when the start is search_mm out, and stopping short of the neighbouring
    # square's ink (which begins 0.95 mm on the other side of the line)
    lo, hi = ink - search_mm - 0.25, ink + search_mm + 0.25
    n_samples = max(21, int(np.ceil((hi - lo) / 0.07)) | 1)
    u = np.linspace(lo, hi, n_samples)[None, :]             # (1,S) mm
    off = u * px_per_mm                                     # (N,S) px
    prof = sample_gray(gray, p0[:, None, :] + nv[:, None, :] * off[:, :, None])

    good = ~np.isnan(prof).any(axis=1)
    prof = np.where(np.isnan(prof), 0.0, prof)
    # brightening across the step, per sample interval
    grad = np.diff(prof, axis=1)
    i = np.argmax(grad, axis=1)
    idx = np.arange(len(prof))
    good &= (i > 0) & (i < grad.shape[1] - 1)
    i = np.clip(i, 1, grad.shape[1] - 2)
    y0_, y1_, y2_ = grad[idx, i - 1], grad[idx, i], grad[idx, i + 1]
    # a real step: the ink is darker than the paper by a visible margin
    good &= (prof.max(axis=1) - prof.min(axis=1)) > REFINE_MIN_CONTRAST
    den = (y0_ - 2 * y1_ + y2_)
    good &= den < -1e-9                                     # a real maximum
    sub = np.where(np.abs(den) > 1e-9, 0.5 * (y0_ - y2_) / np.where(
        np.abs(den) > 1e-9, den, 1.0), 0.0)
    sub = np.clip(sub, -1.0, 1.0)
    step = u[0, 1] - u[0, 0]
    # gradient index i sits between samples i and i+1
    u_flank = u[0, i] + (0.5 + sub) * step
    return p0, nv, (u_flank - ink) * px_per_mm[:, 0], good


def _refine_pass(gray_raw, H, k1, k2, pts, nrm, iters, search_mm, refine_dist,
                 ink=None, refine_ink=False):
    """One Gauss-Newton fit of H (and optionally k1/k2, and the ink width) to
    the sampled lines. Returns (H, k1, k2, ink).

    The update is parametrised as H <- H @ Minv @ (I+F) @ M with M normalising
    the target to a unit box, so all eight entries of F are of comparable
    weight and a plain finite-difference Gauss-Newton stays well conditioned
    even though H itself spans six orders of magnitude. Two IRLS rounds with a
    Huber weight keep one sample that latched onto a shadow or a pencil mark
    from steering the fit.

    `ink` is the half-width the flank measurement subtracts to get back to the
    centre of a line. Printers do not lay ink down at exactly the width asked
    for, and getting it wrong dilates every square about its own centre by the
    same amount - which is not a homography, so it does not hide in H and can
    be solved for. It is solved for jointly rather than assumed, but only when
    several squares are in view: with one square, dilating it about its centre
    IS a scale change, and the two become the same unknown."""
    h, w = gray_raw.shape[:2]
    ink = STROKE_MM / 2.0 if ink is None else float(ink)
    cen = pts.mean(axis=0)
    scale = float(np.max(np.linalg.norm(pts - cen, axis=1)))
    M = np.array([[1 / scale, 0, -cen[0] / scale],
                  [0, 1 / scale, -cen[1] / scale],
                  [0, 0, 1.0]])
    Minv = np.linalg.inv(M)
    H = np.asarray(H, np.float64).copy()

    n_geo = 8 + (2 if refine_dist else 0)
    n_par = n_geo + (1 if refine_ink else 0)
    delta = 1e-4

    def perturb(j, eps):
        """Apply a small step in parameter j -> (H, k1, k2)."""
        if j >= 8:
            return H, k1 + (eps if j == 8 else 0.0), k2 + (eps if j == 9 else 0.0)
        F = np.zeros(9)
        F[j] = eps                       # F[8] (the 3,3 entry) stays fixed
        return H @ Minv @ (np.eye(3) + F.reshape(3, 3)) @ M, k1, k2

    for _ in range(iters):
        p0, nv, off, good = measure_line_offsets(gray_raw, pts, nrm, H, k1, k2,
                                                 search_mm, ink)
        if good.sum() < 12:
            break
        P, N, d = pts[good], nv[good], off[good]
        J = np.empty((len(P), n_par))
        base = project_raw(P, H, k1, k2, w, h)
        for j in range(n_geo):
            Hj, k1j, k2j = perturb(j, delta)
            pj = project_raw(P, Hj, k1j, k2j, w, h)
            J[:, j] = ((pj - base) * N).sum(axis=1) / delta   # motion along n
        if refine_ink:
            # widening the ink by 1 mm moves the measured centre 1 mm inwards,
            # i.e. by the local scale in pixels
            step_mm = project_raw(P + nrm[good] * 0.5, H, k1, k2, w, h) - base
            J[:, n_geo] = np.linalg.norm(step_mm, axis=1) / 0.5
        r = d.copy()
        wgt = np.ones(len(P))
        for _irls in range(2):
            A = J * wgt[:, None]
            step, *_ = np.linalg.lstsq(A, r * wgt, rcond=None)
            res = r - J @ step
            s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
            wgt = np.minimum(1.0, 2.0 * s / np.maximum(np.abs(res), 1e-9))
        F = np.zeros(9)
        F[:8] = step[:8]
        H = H @ Minv @ (np.eye(3) + F.reshape(3, 3)) @ M
        H = H / H[2, 2]
        if refine_dist:
            k1 += float(step[8]); k2 += float(step[9])
        if refine_ink:
            # half a printed line, between half and two and a half times the
            # width it was drawn at - beyond that something else is wrong
            ink = float(np.clip(ink + step[n_geo], 0.25 * STROKE_MM,
                                1.25 * STROKE_MM))
        if np.abs(step[:8]).max() < 1e-7:
            break
    return H, k1, k2, ink


def refine_homography(gray_raw, H, model, k1=0.0, k2=0.0, keys=None,
                      iters=REFINE_ITERS, refine_dist=False):
    """Fit H (and optionally k1/k2) to the printed lines themselves.

    gray_raw is the RAW camera frame in gray; H maps mm -> undistorted px, as
    everywhere else. Returns (H, k1, k2, info).

    Two passes, coarse to fine. The fine pass may only look 0.62 mm to each
    side of where it expects the flank, because the neighbouring square's ink
    starts about a millimetre past it and a sample that walks that far measures
    the wrong line - and nothing downstream would notice, since a fit locked
    one line over still reports a tiny residual while being a millimetre wrong
    out on the canvas. The corner fit it starts from can easily be off by more
    than that, so the coarse pass first lines the block up using ONLY its four
    outer edges, which have no neighbour within 60 mm and can be searched for
    from 3 mm away. After that every inner line is well inside its capture
    range."""
    h, w = gray_raw.shape[:2]
    pts, nrm = model_edge_samples(model, keys)
    info = {"n": 0, "n_total": len(pts), "rms_px": float("nan"),
            "rms_mm": float("nan"), "ink_mm": STROKE_MM / 2.0, "ok": False}
    if len(pts) < 20:
        return H, k1, k2, info
    H = np.asarray(H, np.float64).copy()
    ink = STROKE_MM / 2.0
    # see _refine_pass: with fewer squares the ink width is not separable
    refine_ink = len(set(model) if keys is None else set(keys)) >= 3

    if iters > 0:
        o_pts, o_nrm = model_edge_samples(model, keys, outer_only=True)
        if len(o_pts) >= 12:
            H, k1, k2, _ = _refine_pass(gray_raw, H, k1, k2, o_pts, o_nrm,
                                        max(2, iters // 2), REFINE_COARSE_MM,
                                        False, ink)
        H, k1, k2, ink = _refine_pass(gray_raw, H, k1, k2, pts, nrm, iters,
                                      REFINE_SEARCH_MM, refine_dist,
                                      ink, refine_ink)

    p0, nv, off, good = measure_line_offsets(gray_raw, pts, nrm, H, k1, k2,
                                             ink_mm=ink)
    if good.sum() >= 12:
        rms = float(np.sqrt(np.mean(off[good] ** 2)))
        ppm = plane_px_per_mm(H, pts[good][::7])
        info.update(n=int(good.sum()), rms_px=rms, rms_mm=rms / max(ppm, 1e-6),
                    ink_mm=ink, ok=True)
    return H, k1, k2, info


def refine_report(gray_raw, H, model, k1=0.0, k2=0.0, keys=None,
                  out_path="snap_residual.png", frame=None, ink_mm=None):
    """Say what SHAPE the leftover snap residual has, and write a picture of it.

    A residual well above the ~0.03 mm the geometry alone gives is not noise -
    noise averages out over 900 samples - so it is worth knowing what it is
    before chasing it. Three candidates are separable here:

      * lens - the two-term radial model with the principal point pinned to the
        centre of the frame is an approximation, and what it misses is a smooth
        field that grows with the radius. Regressing the residual onto the
        radial signature says how much of it that is; if it is most of it, 'r'
        (refit k1/k2 through the lines) and re-centring the target in the frame
        will help.
      * paper - a sheet that is not perfectly flat breaks the one assumption
        this whole tool rests on. Curl is smooth across the sheet, so a
        quadratic in the sheet coordinates captures it.
      * print - the sheet not matching calibr.svg. A uniform scale error is
        invisible here (a homography absorbs it, which is exactly why it must
        be checked with a ruler instead), but a non-uniform one is not: it
        shows up as a per-edge offset pattern, so the table below is printed
        edge by edge.

    None of them can be told apart by the size of the residual alone, which is
    why this prints the decomposition rather than a verdict."""
    h, w = gray_raw.shape[:2]
    pts, nrm, lab = model_edge_samples(model, keys, with_labels=True)
    if len(pts) < 20:
        print("[snap] not enough sample points")
        return
    p0, nv, off, good = measure_line_offsets(gray_raw, pts, nrm, H, k1, k2,
                                             ink_mm=ink_mm)
    if good.sum() < 20:
        print("[snap] no lock")
        return
    P, N, D, IMG = pts[good], nv[good], off[good], p0[good]
    ppm = plane_px_per_mm(H, P[::7])
    mm = D / ppm                                   # residual in mm on the sheet
    lab = [l for l, g in zip(lab, good) if g]
    ink = STROKE_MM / 2.0 if ink_mm is None else ink_mm

    print(f"[snap] {len(D)} points, rms {np.sqrt((mm ** 2).mean()):.3f} mm "
          f"({np.sqrt((D ** 2).mean()):.2f} px), scale {ppm:.2f} px/mm, "
          f"ink {2 * ink:.3f} mm wide (drawn {STROKE_MM:.3f})")

    # --- per edge: a signed mean is a real offset, not scatter ---
    print("       square  edge     n   mean(mm)  rms(mm)")
    for key in sorted(set((c, r) for (c, r, _) in lab)):
        for e in range(4):
            m = np.array([(l[0], l[1]) == key and l[2] == e for l in lab])
            if m.sum() < 3:
                continue
            print(f"       ({key[0]},{key[1]})     {EDGE_NAMES[e]:<7}{m.sum():4d}"
                  f"   {mm[m].mean():+7.3f}  {np.sqrt((mm[m] ** 2).mean()):7.3f}")

    def explained(A):
        """Share of the residual variance a model A can account for."""
        coef, *_ = np.linalg.lstsq(A, mm, rcond=None)
        return max(0.0, 1.0 - np.var(mm - A @ coef) / max(np.var(mm), 1e-12))

    # lens: an error in k1/k2 moves a point along the radius, so only the part
    # of that motion across the line shows up in the residual
    f = float(max(w, h))
    rad = (IMG - [w / 2.0, h / 2.0]) / f
    r2 = (rad ** 2).sum(axis=1)
    proj = (rad * N).sum(axis=1)
    lens = np.stack([proj * r2, proj * r2 ** 2], 1)
    # paper: a smooth bow over the sheet, seen across the line
    x = (P[:, 0] - P[:, 0].mean()) / 100.0
    y = (P[:, 1] - P[:, 1].mean()) / 100.0
    one = np.ones_like(x)
    bow = np.stack([one, x, y, x * x, x * y, y * y], 1)
    print(f"[snap] residual explained by: lens (k1/k2) {100 * explained(lens):4.0f}%"
          f"   a smooth bow of the sheet {100 * explained(bow):4.0f}%")
    horiz = np.array([l[2] in (0, 2) for l in lab])
    print(f"       across the horizontal lines {np.sqrt((mm[horiz] ** 2).mean()):.3f} mm"
          f"   across the vertical ones {np.sqrt((mm[~horiz] ** 2).mean()):.3f} mm")

    if out_path and frame is not None:
        vis = frame.copy()
        for p, n, d in zip(IMG, N, D):
            a = tuple(np.round(p).astype(int))
            b = tuple(np.round(p + n * d * 50).astype(int))
            cv2.line(vis, a, b, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, a, 2, (0, 255, 255), -1)
        cv2.imwrite(out_path, vis)
        print(f"[snap] residual map (x50) -> {out_path}")


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
    make_window(win)
    cv2.createTrackbar("col_off", win, 0, max(0, COLS - 1), lambda v: None)
    cv2.createTrackbar("row_off", win, 0, max(0, ROWS - 1), lambda v: None)

    k1, k2, kstep = DIST_K1, DIST_K2, DIST_STEP
    H_saved = None
    do_refine, refine_dist = True, False
    print("[calib] point the camera at the squares. Use the col_off/row_off\n"
          "        trackbars to align the labels with the real positions.\n"
          "        The magenta quad is the consensus fit over all the squares -\n"
          "        its sides should sit on the outer edges of the whole block.\n"
          "        The cyan outlines are the refined fit snapped onto the\n"
          "        printed lines; 'snap' in the HUD is how far off they still\n"
          "        are, in mm on the sheet - that is the number to minimise.\n"
          "        'a' auto-fits the lens distortion; 1/2 3/4 tune k1/k2 by hand\n"
          "        (watch 'line residual': lower = straighter = better),\n"
          "        5/6 change the step, 0 resets the distortion,\n"
          "        'e' toggles the sub-pixel refinement, 'r' lets it retune\n"
          "        k1/k2 as well, 'f' switches consensus/per-corner fit,\n"
          "        'c' saves.")

    while True:
        raw = read_frame(cap)
        if raw is None:
            print("[calib] no frame"); break
        # everything below runs on the undistorted frame, so the homography
        # that comes out maps mm -> UNDISTORTED image px (run/overlay undistort
        # to match)
        frame = undistort(raw, k1, k2)
        vis = frame.copy()
        quads, n_paired = detect_squares(frame)
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

        # the corner fit, then snapped onto the printed lines (cyan)
        H_corner = None
        H_live, rinfo = None, {"ok": False}
        if matches:
            H_corner, _, _ = compute_homography_consensus(matches, model)
            if H_corner is None or FIT_MODE == "corners":
                H_pts = compute_homography(matches, model)
                if H_pts is not None:
                    H_corner = H_pts
        if do_refine and H_corner is not None:
            H_live, nk1, nk2, rinfo = refine_homography(
                cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), H_corner, model,
                k1, k2, keys=set((c, r) for (c, r, _) in matches),
                refine_dist=refine_dist)
            if rinfo["ok"]:
                # feed the tuned distortion back in - the next frame is
                # undistorted with it, so the loop settles after a few frames
                k1, k2 = nk1, nk2
                draw_model(vis, H_live, model, REFINE_COLOR)
            else:
                H_live = None

        # auto hint: if all 6 (3x2) are visible, the mapping is unambiguous
        auto = (ncols == COLS and nrows == ROWS)
        txt = (f"detected={len(quads)} ({n_paired} centred) grid={ncols}x{nrows} "
               f"matched={len(matches)}  off=({col_off},{row_off})"
               f"{'  [AUTO 3x2]' if auto else ''}"
               f"{'' if cq is not None else '  [no consensus quad]'}")
        res = collinearity_residual(matches)
        snap = (f"snap={rinfo['rms_mm']:.3f} mm ({rinfo['rms_px']:.2f} px) "
                f"on {rinfo['n']}/{rinfo['n_total']} pts  "
                f"ink={2 * rinfo['ink_mm']:.2f}mm"
                f"{'  +k1k2' if refine_dist else ''}"
                if rinfo["ok"] else ("snap: no lock" if do_refine else "snap: off"))
        hud = [txt,
               f"k1={k1:+.4f} k2={k2:+.4f} step={kstep:.4f}   "
               f"line residual={res:.3f} px   fit={FIT_MODE}",
               snap,
               "1/2 k1  3/4 k2  5/6 step  a auto-fit dist  0 reset dist  "
               "e refine  r +k1k2  v report  f fit  c save  q quit"]
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
        elif key == ord('e'):
            do_refine = not do_refine
        elif key == ord('r'):
            refine_dist = not refine_dist
        elif key == ord('v') and H_live is not None:
            refine_report(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), H_live, model,
                          k1, k2, keys=set((c, r) for (c, r, _) in matches),
                          frame=raw, ink_mm=rinfo.get("ink_mm"))
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
            # the refinement gets the last word: same model, same frame, but
            # fitted to the ink instead of to four corner estimates
            if do_refine:
                gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                keys = set((c, r) for (c, r, _) in matches)
                _, _, _, before = refine_homography(gray_raw, H, model, k1, k2,
                                                    keys, iters=0)
                Hr, k1r, k2r, after = refine_homography(
                    gray_raw, H, model, k1, k2, keys, refine_dist=refine_dist)
                if after["ok"]:
                    print(f"[calib] line snap: {before['rms_mm']:.3f} -> "
                          f"{after['rms_mm']:.3f} mm "
                          f"({before['rms_px']:.2f} -> {after['rms_px']:.2f} px, "
                          f"{after['n']}/{after['n_total']} sample points)")
                    if after["rms_mm"] > 0.15:
                        print("[calib] NOTE: >0.15 mm left after the snap. The "
                              "fit cannot put all 24 printed lines on the ink "
                              "at once, which points at the SHEET GEOMETRY "
                              "(is this really the calibr.svg print?) or at "
                              "leftover lens distortion, not at this frame.")
                    H, k1, k2 = Hr, k1r, k2r
                else:
                    print("[calib] line snap: no lock (too few usable samples) "
                          "- saving the corner fit.")
            np.savez(CALIB_FILE, H=H, px_per_mm=PX_PER_MM,
                     canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                     k1=k1, k2=k2, model_sig=model_signature(),
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
            if n_paired < len(matches):
                print(f"[calib] ({len(matches) - n_paired} of {len(matches)} "
                      f"squares gave only one contour - their corners sit half a "
                      f"stroke off the centreline, so the reprojection number "
                      f"above carries that offset. The snap residual does not.)")
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
CALIB_CAM_W = 0                    # capture size H was fitted at (0 = unknown)
CALIB_CAM_H = 0


def model_signature():
    """A fingerprint of the sheet geometry the calibration was made with."""
    m = build_model()
    return np.round(np.vstack([m[k] for k in sorted(m)]), 4).astype(np.float32)


def load_calibration():
    """Return H, also restoring the saved lens distortion into DIST_K1/DIST_K2.

    H maps mm -> UNDISTORTED image pixels, so every consumer must push its
    frames through undistort(frame, DIST_K1, DIST_K2) first. Files written
    before distortion tuning existed simply carry no k1/k2 and give 0.

    The capture size H was fitted at lands in CALIB_CAM_W/H: H is in pixels, so
    it only means anything together with the resolution that produced it (see
    scale_homography)."""
    global DIST_K1, DIST_K2, CALIB_CAM_W, CALIB_CAM_H
    try:
        data = np.load(CALIB_FILE)
    except FileNotFoundError:
        return None
    DIST_K1 = float(data["k1"]) if "k1" in data else 0.0
    DIST_K2 = float(data["k2"]) if "k2" in data else 0.0
    CALIB_CAM_W = int(data["cam_w"]) if "cam_w" in data else 0
    CALIB_CAM_H = int(data["cam_h"]) if "cam_h" in data else 0
    # H is only meaningful together with the sheet geometry it was fitted to:
    # change the model and every millimetre it reports moves, silently.
    sig = model_signature()
    old = data["model_sig"] if "model_sig" in data else None
    if old is None or old.shape != sig.shape or not np.allclose(old, sig, atol=1e-3):
        print(f"[calib] WARNING: {CALIB_FILE} was made with a different sheet "
              f"geometry than the one configured now - recalibrate, or the "
              f"reference will land a millimetre or two out.")
    return data["H"]


def scale_homography(H, from_wh, to_wh):
    """H (mm -> pixels at `from_wh`) retargeted to a capture size of `to_wh`.

    Pixel coordinates scale with the resolution, so the retarget is a plain
    diagonal premultiply - but only if the other mode is the SAME field of view
    sampled more coarsely (binned or scaled down), not a crop of the sensor. A
    crop keeps the pixel scale and moves the principal point instead, and
    nothing in H can tell the two apart. Hence the mode list sticks to the
    calibrated aspect ratio, where a crop is unlikely; past that the check is
    visual - the contours either still land on the printed squares or they do
    not.

    k1/k2 need no retarget: camera_matrix pins f to max(w,h) and the principal
    point to the frame center, so the distortion model is expressed in units of
    the frame itself and survives any uniform change of resolution."""
    sx = to_wh[0] / float(from_wh[0])
    sy = to_wh[1] / float(from_wh[1])
    return np.diag([sx, sy, 1.0]) @ H


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
    make_window(win)
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
            make_window(win)
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


_MORPH_K3 = np.ones((3, 3), np.uint8)
DELTA_SCALE = 2                    # the delta mask is built at 1/this of capture


def delta_mask(snap, frame, thr, scale=DELTA_SCALE, blur=3, grow=2):
    """Where `frame` differs from `snap`, as a 0/255 mask at 1/scale of capture.

    The largest per-channel difference, not the gray one: paint whose luminance
    happens to match the paper (yellows, light reds) would barely register in
    gray. Speckle from sensor noise is averaged away by the downscale, blurred,
    then opened; what survives is dilated a little, because the interesting part
    of the brush is its outline and the outline is exactly where the difference
    fades out.

    Two things here are performance, not image processing, and both are worth
    ~10x. The mask is computed small - a brush is hundreds of pixels wide, so
    half resolution costs nothing visible (measured IoU 0.97 against the
    full-res mask, the difference being a 3% wider edge) and turns ~18 ms per
    frame into ~1.5 ms at 2592x1944. And every step is an OpenCV call: the
    obvious numpy spellings (`absdiff(...).max(axis=2)`, `np.where(d >= thr)`)
    are single-threaded passes over the whole array and cost more than all of
    this together. The caller gets the reduced mask and folds the scale into the
    warp that brings it to display resolution, so it is never resized twice."""
    sw, sh = max(1, frame.shape[1] // scale), max(1, frame.shape[0] // scale)
    if scale > 1:
        # INTER_AREA averages, which is the noise suppression as well
        a = cv2.resize(snap, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        a, b = snap, frame
    if blur >= 3:
        a = cv2.GaussianBlur(a, (blur, blur), 0)
        b = cv2.GaussianBlur(b, (blur, blur), 0)
    c0, c1, c2 = cv2.split(cv2.absdiff(a, b))
    d = cv2.max(cv2.max(c0, c1), c2)
    _, m = cv2.threshold(d, thr - 1, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _MORPH_K3)
    if grow:
        m = cv2.dilate(m, _MORPH_K3, iterations=grow)
    return m


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

    # --- delta layer (experimental) ------------------------------------------
    # A snapshot of the canvas as it was, kept in undistorted camera pixels so
    # it survives every view change (raw/corrected, zoom, pan). It replaces the
    # live frame as the bottom layer, and whatever now differs from it - the
    # brush, the hand, fresh paint - is painted back on top of the reference at
    # full opacity. Three layers instead of two: snapshot, reference, delta.
    snap = None                    # undistorted BGR frame, or None
    delta_on = False
    delta_thr = 18                 # per-channel levels of difference
    DTHR = 2

    # --- capture resolution, switchable at runtime ---------------------------
    # Dropping the capture resolution is the one big lever left on the camera
    # side: the USB transfer and the MJPG decode of every frame scale with the
    # pixel count (~26 ms per frame at 2592x1944 against ~6 ms at 1280x960),
    # and no amount of GPU touches either. The cost is resolved detail, which
    # matters when zoomed in - hence a key rather than a setting.
    H_calib = H                    # H exactly as fitted, never rescaled
    calib_w, calib_h = CALIB_CAM_W, CALIB_CAM_H
    H_wh = None                    # capture size the live H currently matches
    cap_modes, cap_i = [], 0
    fps = 0.0
    t_prev = None

    MOVE, SCALE, ROT, DA = 2.0, 1.01, 0.5, 0.05  # steps
    # view transforms; lazy (they need the frame size)
    A = B = None            # corrected: out-px -> world-mm; raw: out-px -> cam-px
    ow = oh = 0             # rendered view size for the current mode

    win = "overlay"
    make_window(win)

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
          "          n snapshot (of the empty canvas) + delta layer on,\n"
          "          t delta layer on/off, 5/6 delta threshold -/+,\n"
          "          v cycle capture resolution (lower = faster, less detail),\n"
          "          MOUSE: wheel or right button (drag) = zoom at cursor,\n"
          "          left button (drag) = pan, SPACE = reset zoom,\n"
          "          p save, i reset adjustment, h help, q quit")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[overlay] no frame"); break
        now = time.perf_counter()
        if t_prev is not None:
            dt = now - t_prev
            if dt > 0:                       # smoothed, or it is unreadable
                fps = 1.0 / dt if fps == 0.0 else 0.9 * fps + 0.1 / dt
        t_prev = now

        fw, fh = frame.shape[1], frame.shape[0]
        if not cap_modes:
            if not calib_w:
                # a calibration file from before cam_w/cam_h were saved: it can
                # only have been made at whatever the camera gives right now
                calib_w, calib_h = fw, fh
                print(f"[overlay] {CALIB_FILE} carries no capture size - "
                      f"assuming it was made at {fw}x{fh}")
            cap_modes = [m for m in CAPTURE_MODES
                         if abs(m[0] / m[1] - calib_w / calib_h) < 0.01]
            if (fw, fh) not in cap_modes:
                cap_modes.append((fw, fh))
                cap_modes.sort(key=lambda m: -m[0] * m[1])
            cap_i = cap_modes.index((fw, fh))
        if H_wh != (fw, fh):
            # the capture size changed (by keypress, or by the camera itself):
            # H is in pixels, so it has to follow
            H = scale_homography(H_calib, (calib_w, calib_h), (fw, fh))
            H_wh = (fw, fh)
            A = B = None                     # both view transforms depend on it

        # H was fitted on undistorted pixels, so BOTH views (raw and corrected)
        # have to start from the undistorted frame or the overlay drifts at the
        # edges exactly where the lens bends the most
        frame = undistort(frame, DIST_K1, DIST_K2)
        if snap is not None and snap.shape != frame.shape:
            print("[overlay] camera frame size changed - snapshot dropped")
            snap, delta_on = None, False
        use_delta = delta_on and snap is not None
        # the snapshot is the bottom layer while the mode is on; the live frame
        # only comes back where it differs from it
        base = snap if use_delta else frame
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

        live = dmask = None
        identity = np.allclose(src, np.eye(3))
        if identity:
            disp = base.copy()                   # raw at zoom 1: no resampling
        else:
            disp = cv2.warpPerspective(base, src, (ow, oh),
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        if use_delta:
            # the live frame and the mask go through separately, not as one
            # 4-channel warp of the two stacked: warpPerspective has no fast
            # path for 4 channels and the stacking copies 20 MB, which measures
            # 25 ms against 1.6 ms for the pair.
            live = frame if identity else cv2.warpPerspective(
                frame, src, (ow, oh),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            dm = delta_mask(snap, frame, delta_thr)
            # the mask arrives reduced; its own scale folds into the warp that
            # brings it to display resolution, so it is resampled exactly once
            # (this is also what upscales it when src is the identity). NEAREST
            # keeps it strictly 0/255, which cv2.copyTo below needs.
            Ms = np.diag([dm.shape[1] / float(frame.shape[1]),
                          dm.shape[0] / float(frame.shape[0]), 1.0]) @ src
            dmask = cv2.warpPerspective(
                dm, Ms, (ow, oh),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        ch, cw = disp.shape[:2]

        if show:
            T = ref_to_world(rw, rh, dx, dy, sx, sy, theta)
            M = np.linalg.inv(src) @ H @ T        # ref pixel -> display pixel
            if render_mode == "contours":
                draw_polylines_blended(disp, epts, elens, M,
                                       OVERLAY_COLORS[color_i], alpha)
            else:
                # Both modes blend only where the reference actually lands, and
                # both get that for free from the warp border instead of from a
                # coverage mask - a mask would mean a second warp plus a
                # boolean-indexed float32 gather over the whole frame, which
                # measures ~80 ms against ~3.5 ms for the two calls below.
                if render_mode == "multiply":
                    # multiply only ever darkens, so the reference reads as ink
                    # laid over the canvas: white paper in the ref leaves the
                    # camera image untouched and what is drawn on the real
                    # canvas stays visible through the dark areas. A white
                    # border extends that no-op to everything outside the ref.
                    over = cv2.warpPerspective(
                        ref_small, M, (cw, ch), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255))
                    over = cv2.multiply(over, disp, scale=1.0 / 255.0)
                else:
                    # BORDER_TRANSPARENT leaves dst alone where the ref does not
                    # reach, so those pixels end up blending disp with disp
                    over = disp.copy()
                    cv2.warpPerspective(ref_small, M, (cw, ch), dst=over,
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_TRANSPARENT)
                disp = cv2.addWeighted(disp, 1.0 - alpha, over, alpha, 0.0)

        if use_delta:
            # last, so the brush is never washed out by the reference: opaque,
            # which is the whole point - everything around it keeps the usual
            # blended look, the brush itself reads as bare camera.
            # copyTo, not disp[dmask > 127] = live[...]: same result, ~0.5 ms
            # instead of ~9 ms (no boolean gather, no scatter).
            cv2.copyTo(live, dmask, disp)

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
                   f"delta={'on' if use_delta else ('armed' if delta_on else 'off')}"
                   f"  thr={delta_thr}"
                   f"  snapshot={'yes' if snap is not None else 'none (n)'}"
                   f"  cam={fw}x{fh}"
                   + ("" if (fw, fh) == (calib_w, calib_h) else " (scaled H)")
                   + f"  {fps:.1f} fps",
                   "w/a/s/d move  z/x scale  [ ] X  - = Y  ,/. rot  m mode  9/0 alpha  "
                   "1/2 3/4 canny  5/6 delta thr  n snap  t delta  v cam res  <-/-> ref"]
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
        elif k == ord('n'):
            # take the snapshot with nothing in front of the camera, then go
            # draw: mode on straight away, since that is always what follows
            snap = frame.copy()
            delta_on = True
            print(f"[overlay] snapshot taken ({snap.shape[1]}x{snap.shape[0]}), "
                  f"delta on (thr={delta_thr})")
        elif k == ord('t'):
            if snap is None:
                snap = frame.copy()
                print("[overlay] no snapshot yet - taking one now")
            delta_on = not delta_on
            print(f"[overlay] delta {'on' if delta_on else 'off'}")
        elif k == ord('v'):
            # step to the next (smaller) capture mode, wrapping back to full;
            # a mode the camera cannot really stream is skipped, not fatal
            for step in range(1, len(cap_modes)):
                want = cap_modes[(cap_i + step) % len(cap_modes)]
                print(f"[overlay] switching capture to {want[0]}x{want[1]}...")
                got = set_capture_mode(cap, *want)
                if got is None:
                    print(f"[overlay] {want[0]}x{want[1]} delivers nothing - skipped")
                    continue
                cap_i = (cap_modes.index(got) if got in cap_modes
                         else (cap_i + step) % len(cap_modes))
                if got != want:
                    print(f"[overlay] camera substituted {got[0]}x{got[1]}")
                # the snapshot belongs to the old resolution; the loop drops it
                # on the size change, so say why rather than let it vanish
                if snap is not None and got != (fw, fh):
                    print("[overlay] retake the snapshot (n) at the new resolution")
                break
        elif k == ord('5'):
            delta_thr = max(1, delta_thr - DTHR)
        elif k == ord('6'):
            delta_thr = min(255, delta_thr + DTHR)
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

    This is the honest thing to check the overlay against: it is the model
    itself, drawn to the same tolerance the model is stated to, so anything
    that does not line up on the sheet is the calibration and not the drawing.
    A reference traced over a photo of the sheet by hand is good to a few
    tenths of a millimetre at best, and its errors are indistinguishable from
    the ones being hunted.
    """
    W = int(round(CANVAS_W * ppm))
    H = int(round(CANVAS_H * ppm))
    img = np.full((H, W, 3), 255, np.uint8)
    model = build_model()
    th = max(1, int(round(STROKE_MM * ppm)))   # the printed line width
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

    mons = list_displays()
    print("\nDisplays (pass the index or the name to --display):")
    if not mons:
        print("  none detected - --display will be ignored, --fullscreen "
              "still works on whichever monitor the window opens on")
    for i, (name, x, y, w, h) in enumerate(mons):
        print(f"  {i}: {name:<12} {w}x{h}+{x}+{y}")
    if len({(x, y) for (_n, x, y, _w, _h) in mons}) < len(mons):
        print("  NOTE: two monitors report the same position - they are "
              "mirrored, and --display cannot tell them apart.")


def main():
    global PX_PER_MM, CANVAS_W, CANVAS_H, REQ_WIDTH, REQ_HEIGHT, DISPLAY_MAX
    global FIT_MODE, DIST_K1, DIST_K2, FULLSCREEN, DISPLAY_TARGET, KEEP_AWAKE
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
    ap.add_argument("--ppm", type=float, default=4.0,
                    help="pixels per mm for gen-template (default 4.0)")
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
    ap.add_argument("--fullscreen", action="store_true",
                    help="open the window fullscreen")
    ap.add_argument("--display", default=None,
                    help="monitor to open on: an index or a name such as "
                         "DP-1 ('list' prints them)")
    ap.add_argument("--no-keep-awake", action="store_true",
                    help="do not hold off sleep and the screensaver")
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
    FULLSCREEN = args.fullscreen
    DISPLAY_TARGET = args.display
    KEEP_AWAKE = not args.no_keep_awake
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
        generate_template(args.out or "calib_template.png", args.ppm)
        return

    model = build_model()
    awake = KeepAwake(f"artprojector {args.mode}").start() if KEEP_AWAKE else None
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
        if awake is not None:
            awake.stop()


if __name__ == "__main__":
    main()
