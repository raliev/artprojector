#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projcalib.py
============

Calibrate the PROJECTOR against the canvas, using the already-calibrated
camera as the measuring instrument.

The camera calibration (artprojector.py calibrate) answers "where is the
canvas in the camera frame": a homography H_cam from canvas millimetres to
undistorted camera pixels. This tool answers the other half - "where does the
projector put its pixels on that canvas":

    projector window px --H_pc--> camera px --H_cam^-1--> canvas mm

and it is the composition, H_pm, that is saved. Its inverse, H_mp (mm ->
projector px), is what everything downstream actually needs: give it a
reference image in canvas millimetres and it says which projector pixel to
light up.

How it works. The projector is filled with a lattice of squares - the same
idea as the printed 1-inch target, only made of light and measured in
projector pixels instead of millimetres - with an ArUco marker centred in
every cell. The camera looks at the canvas and reads whichever markers happen
to fall in its frame; each one says which cell it is, so the correspondence
projector-px <-> camera-px comes for free and the camera does not have to see
the whole projection (it usually cannot). A homography is fitted through them,
composed with the camera's, and the result is reported in the terms that
matter to a painter: does the projection cover the whole canvas, how much
light is spilling past it, how many projector pixels there are per millimetre,
and how badly the projector is off-axis.

The projected markers use DICT_5X5_1000 while the printed target uses
DICT_4X4_1000, so the printed board may stay on the canvas while this runs -
which is worth doing, because then the verification modes below show the
projected geometry landing on top of printed lines whose position is known
independently.

Two things this cannot know and will not warn about:

  * The camera must not have moved since `artprojector.py calibrate`. Every
    millimetre here is measured through H_cam, and a nudged camera makes a
    perfectly self-consistent fit that is simply in the wrong place.
  * The canvas must be flat and the projector must be a pinhole. A homography
    has no term for a bowed canvas or for projector lens distortion; both come
    out as a residual, which is why the fit residual is printed in millimetres
    on the canvas rather than hidden.

And one it fails loudly at: a projector set to mirror its image (rear
projection, some ceiling mounts) throws a mirrored marker on the canvas, and a
mirrored marker does not decode at all - nothing is found, rather than found
wrong. Turn the flip off in the projector's menu.

What is saved is the mapping of the *window content* to the canvas: if the
window is fullscreen at the monitor's own resolution - which is what happens
by default - window pixels are projector pixels, but even if the compositor
scales it, the calibration stays true as long as later rendering goes through
a window of the same geometry.

Usage:
  python projcalib.py --cam 0 --display 1 --board 12x16
  python projcalib.py --display DP-2 --cells 10          # denser lattice
  python projcalib.py --display 1 --cam-display 0        # where each window goes

Two windows open: the pattern, fullscreen on --display, and the camera view,
which goes to the first monitor that is NOT the projector unless
--cam-display says otherwise.

An existing projector.npz is loaded at startup, so the verification views work
immediately and a session that only wants to check - or to nudge - the
alignment never has to measure again.

In the window (keep the keyboard focus on the camera window):
  c     - measure: average several frames, fit, and report the geometry
  ENTER - save the fit, adjustment included (projector.npz); 'S' does too
  p     - project the marker lattice (the calibration pattern)
  g     - project a test grid drawn in canvas millimetres
  r     - project a white rectangle exactly the size of the canvas
  v     - project --verify-image, mapped onto the canvas
  k     - project black (to see what the room looks like unlit)
  9/0   - dim / brighten the pattern (a blown-out white blooms and biases corners)
  n/m   - fewer / more cells across
  q     - quit

and the hand adjustment, on the same keys and with the same steps as
`artprojector.py overlay`, because it is the same gesture on the same canvas:
  a/d w/s - move the projection 2 mm on the canvas
  z/x     - scale it about the canvas centre;  [ ] and - = scale X and Y alone
  , .     - rotate half a degree
  TAB     - cycle five states: the whole projection, then each of the four
            corners. In a corner state a/d/w/s move THAT corner alone and a
            big arrow, drawn on the canvas itself, points at it.
  i       - back to no adjustment, corners included

Move, scale and rotate are a similarity, and a similarity cannot bend a
rectangle: when the projection is short at one corner and long at the
opposite one, nothing in it can help. The four corners are the remaining four
degrees of freedom - together they make the adjustment a full homography -
and they are where a residual keystone, a canvas that is not quite
rectangular, or a wall the canvas is not quite parallel to end up.
"""

import argparse
import time

import numpy as np
import cv2

import artprojector as ap

PROJ_FILE = "projector.npz"

# The projected markers must not be confusable with the printed ones, so they
# come from a different dictionary altogether rather than from a different id
# range of the same one: a 4x4 codeword cannot be read as a 5x5 codeword at
# all, so the printed board can stay on the canvas while the projector is
# calibrated - which is the only way to see the two geometries at once.
PROJ_DICT_ID = cv2.aruco.DICT_5X5_1000

# How fine the lattice is. The trade is not obvious in either direction, and
# it is worth stating because the natural instinct - "the squares look huge,
# make them small" - runs into the other end of it quickly.
#
# Coarse cells give big markers, which decode from further away and refine
# their corners more precisely, but a camera looking at part of the canvas
# then holds only a handful of them, and four corners in one corner of the
# frame extrapolate badly across the rest of it. Fine cells give many
# correspondences spread over the frame, which is what the homography actually
# wants, until the marker gets small enough that its modules stop being
# resolved by the projector, the canvas texture or the camera - and a marker
# that is not read is not a marker.
#
# 16 across a 1920 px projector is 120 px cells with 63 px markers, i.e. 9
# projector pixels per module: still comfortable, and typically a dozen or
# more cells in a close-up camera frame instead of two or three.
DEFAULT_COLS = 16                  # lattice cells across the projector frame
MARKER_FRAC = 0.55                 # marker side as a fraction of the cell
LINE_PX = 2                        # width of the lattice lines, projector px
MIN_CELL_PX = 56                   # below this a 5x5 marker stops decoding

_DICT = None
_DET = None


def projector_dictionary():
    global _DICT
    if _DICT is None:
        _DICT = cv2.aruco.getPredefinedDictionary(PROJ_DICT_ID)
    return _DICT


def projector_detector():
    """Detector for the projected markers.

    The parameters are the printed target's (see artprojector.aruco_detector)
    for the same reasons - a marker may be small in the frame and seen at an
    angle - with one addition: a projected marker is a bright thing on a
    brighter background, and its edges are softened by the projector's focus
    and by the paint texture underneath, so the sub-pixel corner refinement is
    given a slightly larger window to work in."""
    global _DET
    if _DET is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 7
        p.cornerRefinementMaxIterations = 60
        p.cornerRefinementMinAccuracy = 0.01
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMax = 43
        p.adaptiveThreshWinSizeStep = 8
        p.perspectiveRemovePixelPerCell = 8
        _DET = cv2.aruco.ArucoDetector(projector_dictionary(), p)
    return _DET


# ==========================================================================
#  THE PROJECTED PATTERN
#
#  Defined entirely in window pixels: cells of `cell` px on a side, the
#  lattice centred in the frame, a marker of 7*k px (a 5x5 marker is 5 data
#  modules plus a one-module black border, so a multiple of 7 keeps every
#  module an exact whole number of pixels and the edges crisp) centred in
#  each cell. Cell (c,r) carries marker id r*cols + c.
# ==========================================================================
def pattern_spec(proj_w, proj_h, cols=DEFAULT_COLS, white=255, cell_px=None):
    """The lattice. `cell_px` sets the cell size directly and wins over `cols`.

    The clamp at MIN_CELL_PX is silent arithmetic, so whoever asked for more
    cells than fit has to be told - see `spec['asked']`."""
    cell = int(cell_px) if cell_px else int(proj_w // max(1, cols))
    cell = max(MIN_CELL_PX, cell)
    ncols = max(1, int(proj_w // cell))
    nrows = max(1, int(proj_h // cell))
    k = max(2, int(round(cell * MARKER_FRAC / 7.0)))
    return {"proj_w": int(proj_w), "proj_h": int(proj_h),
            "cell": int(cell), "cols": int(ncols), "rows": int(nrows),
            "marker": int(7 * k),
            "ox": int((proj_w - ncols * cell) // 2),
            "oy": int((proj_h - nrows * cell) // 2),
            "white": int(white),
            "asked": int(cols) if not cell_px else ncols}


def marker_corners(spec, mid):
    """(4,2) window px of marker `mid`, TL,TR,BR,BL - the OUTER edge of its
    black border, which is what detectMarkers() measures.

    The half pixel is not pedantry with a projector this coarse: a marker
    drawn into pixels [x0 .. x0+m-1] has its outer edge at x0-0.5, and at ~2
    projector px per mm half a pixel is a quarter of a millimetre on the
    canvas - the same order as everything else being fought for here."""
    c, r = mid % spec["cols"], mid // spec["cols"]
    m = spec["marker"]
    x0 = spec["ox"] + c * spec["cell"] + (spec["cell"] - m) // 2
    y0 = spec["oy"] + r * spec["cell"] + (spec["cell"] - m) // 2
    return np.array([[x0 - 0.5, y0 - 0.5], [x0 + m - 0.5, y0 - 0.5],
                     [x0 + m - 0.5, y0 + m - 0.5], [x0 - 0.5, y0 + m - 0.5]],
                    np.float64)


def pattern_image(spec):
    """The calibration pattern: lattice lines plus a marker in every cell."""
    w, h, cell, m = spec["proj_w"], spec["proj_h"], spec["cell"], spec["marker"]
    ox, oy, white = spec["ox"], spec["oy"], spec["white"]
    img = np.full((h, w, 3), white, np.uint8)
    for i in range(spec["cols"] + 1):
        x = ox + i * cell
        cv2.line(img, (x, 0), (x, h), (0, 0, 0), LINE_PX)
    for j in range(spec["rows"] + 1):
        y = oy + j * cell
        cv2.line(img, (0, y), (w, y), (0, 0, 0), LINE_PX)
    # the edge of the projector's own frame, so it can be seen on the wall
    cv2.rectangle(img, (1, 1), (w - 2, h - 2), (0, 0, 0), 2)
    dic = projector_dictionary()
    for mid in range(spec["cols"] * spec["rows"]):
        q = marker_corners(spec, mid)
        x0, y0 = int(round(q[0][0] + 0.5)), int(round(q[0][1] + 0.5))
        tile = cv2.aruco.generateImageMarker(dic, mid, m)
        img[y0:y0 + m, x0:x0 + m] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    return img


# ==========================================================================
#  MEASURING
# ==========================================================================
def xform(H, pts):
    """Apply a homography to (N,2) points."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


def detect_projected(frame, spec, k1=0.0, k2=0.0):
    """Find the projected markers -> ([(id, quad)], n_foreign, n_clipped).

    `frame` is the RAW camera frame and the quads come back in UNDISTORTED
    pixels, exactly as in artprojector.detect_grid_cells and for the same
    reasons: nothing resamples the image before a corner is measured, and the
    black wedges undistort() leaves in the frame corners are not thresholded
    against a marker sitting in one."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    corners, ids, _ = projector_detector().detectMarkers(gray)
    out, foreign, clipped = [], 0, 0
    if ids is None:
        return out, foreign, clipped
    n_ids = spec["cols"] * spec["rows"]
    for quad, mid in zip(corners, ids.reshape(-1)):
        q = np.asarray(quad, np.float64).reshape(4, 2)
        side = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4])
                              for i in range(4)]))
        if side < ap.GRID_MIN_MARKER_PX:
            continue
        mrg = max(2.0, ap.GRID_BORDER_FRAC * side)
        if (q[:, 0].min() < mrg or q[:, 1].min() < mrg
                or q[:, 0].max() > w - 1 - mrg or q[:, 1].max() > h - 1 - mrg):
            clipped += 1
            continue
        if not (0 <= int(mid) < n_ids):
            foreign += 1                     # not from this lattice at all
            continue
        if k1 or k2:
            q = ap.undistort_points(q.astype(np.float32), k1, k2, w, h)
            q = np.asarray(q, np.float64).reshape(4, 2)
        out.append((int(mid), q))
    return out, foreign, clipped


def average_observations(seen, min_frames):
    """{id: [quad,...]} -> [(id, mean quad)], keeping the steady ones.

    Averaging several frames is worth the second it costs: the corner noise of
    a projected edge is a good fraction of a camera pixel and it is
    zero-mean, while the things that make a marker appear in one frame only -
    a hand passing, a flicker, an ArUco error-correction fantasy - are not,
    which is what the min_frames threshold is for."""
    obs = []
    for mid, quads in sorted(seen.items()):
        if len(quads) >= min_frames:
            obs.append((mid, np.mean(np.stack(quads), axis=0)))
    return obs


def fit_projector(obs, spec, H_cam):
    """Fit projector px -> canvas mm. Returns a dict, or None.

    RANSAC first, then a least-squares refit on the markers that agreed. The
    vote is not ceremony: a marker read through a moving hand, or one whose
    id ArUco's error correction invented, is a geometric outlier, and one
    wrong cell id drags the whole extrapolation across the canvas."""
    if len(obs) < 2:
        return None
    P = np.vstack([marker_corners(spec, mid) for mid, _ in obs])
    C = np.vstack([q for _, q in obs])
    H_pc, mask = cv2.findHomography(P, C, cv2.RANSAC, 3.0)
    if H_pc is None:
        return None
    if mask is not None:
        per = mask.reshape(-1, 4).sum(axis=1)
        keep = [i for i, s in enumerate(per) if s >= 2]
        if len(keep) >= 2:
            obs = [obs[i] for i in keep]
            P = np.vstack([marker_corners(spec, mid) for mid, _ in obs])
            C = np.vstack([q for _, q in obs])
            H2, _ = cv2.findHomography(P, C, 0)
            if H2 is not None:
                H_pc = H2
    try:
        H_pm = np.linalg.inv(np.asarray(H_cam, np.float64)) @ H_pc
        H_mp = np.linalg.inv(H_pm)
    except np.linalg.LinAlgError:
        return None

    # Residual where it means something: on the canvas, in millimetres. The
    # camera-pixel residual is reported too, but it is the one that changes
    # with how close the camera happens to be sitting.
    mm_seen = xform(np.linalg.inv(np.asarray(H_cam, np.float64)), C)
    mm_model = xform(H_pm, P)
    d_mm = np.linalg.norm(mm_seen - mm_model, axis=1)
    d_px = np.linalg.norm(xform(H_pc, P) - C, axis=1)
    x, y = P[:, 0], P[:, 1]
    return {"H_pc": H_pc, "H_pm": H_pm, "H_mp": H_mp,
            "H_mp_raw": H_mp, "adj": list(ZERO_ADJ),
            "corners": np.zeros((4, 2)), "measured": True,
            "n": len(obs), "npts": len(P),
            "rms_mm": float(np.sqrt(np.mean(d_mm ** 2))),
            "max_mm": float(d_mm.max()),
            "rms_px": float(np.sqrt(np.mean(d_px ** 2))),
            "span_x": float(x.max() - x.min()) / spec["proj_w"],
            "span_y": float(y.max() - y.min()) / spec["proj_h"],
            "ids": [mid for mid, _ in obs]}


# ==========================================================================
#  THE HAND ADJUSTMENT
#
#  Everything above measures. This is the knob for what measuring cannot
#  reach: the fit can be perfect and the projected rectangle still miss the
#  canvas, because "the canvas" is a number in the model (12x16") and the
#  wooden thing on the easel may not be that size, and because every
#  millimetre here is inherited from the printed board - print it at 96% and
#  every mm in the whole tool is 4% short, invisibly and self-consistently.
#
#  Both of those come out as the same gesture: nudge and scale until the
#  projection lands on the real edges. Which is why the adjustment is kept
#  apart from the fit rather than folded into it - a measured mapping and a
#  hand-tuned one should never become indistinguishable. The file keeps the
#  raw fit, the adjustment, and their product, and says so on load.
# ==========================================================================
MOVE_MM, SCALE_STEP, ROT_DEG = 2.0, 1.01, 0.5      # the steps overlay uses

ZERO_ADJ = (0.0, 0.0, 1.0, 1.0, 0.0)               # dx, dy, sx, sy, theta
CORNER_NAMES = ("TL", "TR", "BR", "BL")            # order of canvas_quad_mm()


def adjust_matrix(adj):
    """(dx,dy,sx,sy,theta) -> world mm -> world mm.

    The same convention as artprojector's ref_to_world: scale then rotate
    about the CANVAS CENTRE, then translate by (dx,dy) mm - so the numbers
    mean in this window exactly what they mean in overlay."""
    dx, dy, sx, sy, th = adj
    cx, cy = -ap.CANVAS_W / 2.0, -ap.CANVAS_H / 2.0
    c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    T1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    T2 = np.array([[1.0, 0.0, cx + dx], [0.0, 1.0, cy + dy], [0.0, 0.0, 1.0]])
    return T2 @ R @ S @ T1


def corner_matrix(corners):
    """4 corner offsets in mm -> world mm -> world mm.

    Move, scale and rotate are a similarity: four numbers of freedom, and they
    cannot bend a rectangle. What is left over when they run out is a
    projective error - one corner of the projection short and the opposite one
    long - and that has exactly the four extra degrees of freedom that four
    draggable corners give. So the correction is built the only way it can be:
    the canvas quad mapped onto the canvas quad with each corner displaced,
    which is a homography and composes with everything else for free.

    A degenerate drag (a corner pushed across its neighbours) has no
    homography; it falls back to no correction rather than to a matrix full of
    infinities."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    if not c.any():
        return np.eye(3)
    Q = canvas_quad_mm().astype(np.float32)
    try:
        return cv2.getPerspectiveTransform(Q, (Q + c).astype(np.float32))
    except cv2.error:
        return np.eye(3)


def apply_adjust(fit):
    """Recompute the effective mapping of `fit` from its raw fit + adjustment.

    A drawing is authored in mm and has to LAND at adjusted mm, so the
    adjustment goes on the mm side of the mapping: projector px = H_mp(A(p)).
    The corner warp is applied after the similarity, which is what makes a
    corner offset mean the millimetres it says on the finished projection."""
    fit.setdefault("corners", np.zeros((4, 2)))
    A = corner_matrix(fit["corners"]) @ adjust_matrix(fit["adj"])
    fit["H_mp"] = np.asarray(fit["H_mp_raw"], np.float64) @ A
    fit["H_pm"] = np.linalg.inv(fit["H_mp"])
    return fit


def adjusted(fit):
    return (tuple(fit["adj"]) != ZERO_ADJ
            or bool(np.asarray(fit.get("corners", 0)).any()))


def adjust_text(fit):
    dx, dy, sx, sy, th = fit["adj"]
    s = (f"adjust dx={dx:+.1f}mm dy={dy:+.1f}mm sx={sx:.3f} sy={sy:.3f} "
         f"rot={th:+.1f}deg")
    c = np.asarray(fit.get("corners", np.zeros((4, 2))), np.float64).reshape(4, 2)
    if c.any():
        s += "  corners " + " ".join(
            f"{n}({x:+.0f},{y:+.0f})" for n, (x, y) in zip(CORNER_NAMES, c)
            if x or y)
    return s


def corner_arrow(img, H, i, color=(0, 160, 255)):
    """Point a big arrow at canvas corner `i`, through any mm -> px map.

    Drawn on the PROJECTION as well as in the camera window, because the
    corner being nudged is a corner of the canvas across the room, and the
    person nudging it is looking at the canvas, not at the screen."""
    q = canvas_quad_mm()
    c = q[i]
    m = np.array([-ap.CANVAS_W / 2.0, -ap.CANVAS_H / 2.0])
    d = m - c
    n = np.linalg.norm(d)
    if n < 1e-6:
        return img
    a, b = c + d * 0.30, c + d * 0.05
    pts = xform(H, [a, b])
    if not np.all(np.isfinite(pts)) or np.abs(pts).max() > 1e6:
        return img
    p0, p1 = (tuple(np.round(pts[0]).astype(int)),
              tuple(np.round(pts[1]).astype(int)))
    th = max(3, int(np.linalg.norm(pts[1] - pts[0]) / 22))
    cv2.arrowedLine(img, p0, p1, (0, 0, 0), th + 4, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, p0, p1, color, th, cv2.LINE_AA, tipLength=0.3)
    lab = np.round(pts[0] + (pts[0] - pts[1]) * 0.12).astype(int)
    for col, wid in (((0, 0, 0), th + 4), (color, max(2, th))):
        cv2.putText(img, CORNER_NAMES[i], tuple(lab), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.9, th / 3.0), col, wid, cv2.LINE_AA)
    return img


# ==========================================================================
#  WHAT THE FIT MEANS
# ==========================================================================
def canvas_quad_mm():
    """The canvas as TL,TR,BR,BL in world mm (origin: bottom-right corner)."""
    W, H = ap.CANVAS_W, ap.CANVAS_H
    return np.array([[-W, -H], [0.0, -H], [0.0, 0.0], [-W, 0.0]], np.float64)


def jacobian(H, x, y, eps=0.5):
    """d(image)/d(x,y) of a homography at (x,y), by central differences."""
    p = xform(H, [[x - eps, y], [x + eps, y], [x, y - eps], [x, y + eps]])
    return np.column_stack([(p[1] - p[0]) / (2 * eps), (p[3] - p[2]) / (2 * eps)])


def local_scale(H, x, y):
    """Scalar px per mm of H at (x,y): the geometric mean of the two axes."""
    J = jacobian(H, x, y)
    return float(np.sqrt(abs(np.linalg.det(J))))


def _ccw(poly):
    """cv2.intersectConvexConvex wants consistent orientation."""
    p = np.asarray(poly, np.float32)
    return p if cv2.contourArea(p, True) >= 0 else p[::-1].copy()


def geometry_report(fit, spec):
    """The lines that say whether the projector is aimed well enough."""
    H_pm, H_mp = fit["H_pm"], fit["H_mp"]
    pw, ph = spec["proj_w"], spec["proj_h"]
    W, H = ap.CANVAS_W, ap.CANVAS_H
    canvas_mm = canvas_quad_mm()
    frame_px = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float64)
    frame_mm = xform(H_pm, frame_px)
    canvas_px = xform(H_mp, canvas_mm)
    out = []

    out.append(f"[proj] fit: {fit['n']} markers / {fit['npts']} corners, "
               f"residual {fit['rms_mm']:.2f} mm rms on the canvas "
               f"(max {fit['max_mm']:.2f} mm; {fit['rms_px']:.2f} px in the camera)")
    out.append(f"[proj] measured over {100 * fit['span_x']:.0f}% x "
               f"{100 * fit['span_y']:.0f}% of the projector frame - "
               f"everything outside that is extrapolated")

    names = ("TL", "TR", "BR", "BL")
    out.append("[proj] the projected frame lands on the canvas plane at "
               + "  ".join(f"{n}({p[0]:+.0f},{p[1]:+.0f})"
                           for n, p in zip(names, frame_mm))
               + f" mm   [the canvas is (-{W:.0f},-{H:.0f})..(0,0)]")

    inter_area, inter_poly = cv2.intersectConvexConvex(
        _ccw(frame_mm), _ccw(canvas_mm))
    canvas_area = W * H
    cover = 100.0 * inter_area / canvas_area if canvas_area else 0.0
    if inter_poly is not None and len(inter_poly) >= 3:
        lit = xform(H_mp, np.asarray(inter_poly, np.float64).reshape(-1, 2))
        used = 100.0 * abs(cv2.contourArea(lit.astype(np.float32))) / float(pw * ph)
    else:
        used = 0.0
    out.append(f"[proj] the projection covers {cover:.1f}% of the canvas, and "
               f"the canvas takes {used:.0f}% of the projector's pixels "
               f"({max(0.0, 100 - used):.0f}% falls past it)")

    # Clearance, corner by corner, in millimetres on the canvas: how much
    # further the canvas could move before it ran out of projection.
    sides = {"left": [], "right": [], "top": [], "bottom": []}
    for (mx, my), (px, py) in zip(canvas_mm, canvas_px):
        s = local_scale(H_mp, mx, my)
        s = s if s > 1e-9 else 1.0
        if mx <= -W + 1e-6:
            sides["left"].append(px / s)
        if mx >= -1e-6:
            sides["right"].append((pw - px) / s)
        if my <= -H + 1e-6:
            sides["top"].append(py / s)
        if my >= -1e-6:
            sides["bottom"].append((ph - py) / s)
    clear = {k: min(v) for k, v in sides.items() if v}
    out.append("[proj] clearance from the canvas edge to the edge of the "
               "projection: "
               + "  ".join(f"{k} {v:+.0f} mm" for k, v in clear.items()))
    short = {k: v for k, v in clear.items() if v < 0}
    if short:
        out.append("[proj] !! the projection does NOT reach the canvas on the "
                   + ", ".join(f"{k} ({-v:.0f} mm short)"
                               for k, v in short.items())
                   + " - move the projector back, or aim it that way and "
                     "measure again.")

    probes = [(-W, -H), (0.0, -H), (0.0, 0.0), (-W, 0.0), (-W / 2, -H / 2)]
    scales = [local_scale(H_mp, x, y) for x, y in probes]
    lo, hi = min(scales), max(scales)
    out.append(f"[proj] resolution on the canvas: {lo:.2f}..{hi:.2f} projector "
               f"px per mm ({lo * 25.4:.0f}..{hi * 25.4:.0f} dpi); "
               f"keystone {100 * (hi / max(lo, 1e-9) - 1):.0f}% across the canvas")

    s = local_scale(H_mp, -W / 2, -H / 2)
    cell_mm, mark_mm = spec["cell"] / max(s, 1e-9), spec["marker"] / max(s, 1e-9)
    out.append(f"[proj] the lattice lands at {cell_mm:.0f} mm per cell "
               f"({mark_mm:.0f} mm markers), i.e. about "
               f"{W / max(cell_mm, 1e-9):.0f}x{H / max(cell_mm, 1e-9):.0f} cells "
               f"on the canvas - '-'/'=' (or --cells / --cell-px) change it")

    J = jacobian(H_mp, -W / 2, -H / 2)
    U, S, Vt = np.linalg.svd(J)
    R = U @ Vt
    rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    aniso = 100.0 * (S[0] / max(S[1], 1e-9) - 1.0)
    out.append(f"[proj] the projector's image is rotated {rot:+.1f} deg from "
               f"the canvas axes, and stretched {aniso:.1f}% more along one "
               f"direction than the other at the canvas centre")

    if fit["rms_mm"] > 1.0:
        out.append("[proj] NOTE: >1 mm of residual. A homography has no term "
                   "for a bowed canvas or for the projector's own lens "
                   "distortion, and both show up here; so does a camera that "
                   "has moved since it was calibrated.")
    if min(fit["span_x"], fit["span_y"]) < 0.4:
        out.append("[proj] NOTE: the camera saw a small part of the projected "
                   "frame, so most of the canvas is being extrapolated from "
                   "it. Back the camera off, or accept that the fit is only "
                   "good where the markers were.")
    return out


# ==========================================================================
#  WHAT THE PROJECTOR SHOWS
# ==========================================================================
def canvas_test_image(spec, H_mp, step_mm=50.0):
    """A test pattern authored in canvas mm and pushed out to projector px.

    Black everywhere except the drawing, so the lit lines can be compared with
    the physical edges of the canvas by eye - which is the only check that
    closes the loop without involving the camera again."""
    pw, ph = spec["proj_w"], spec["proj_h"]
    img = np.zeros((ph, pw, 3), np.uint8)
    W, H = ap.CANVAS_W, ap.CANVAS_H

    def line(p0, p1, color, th):
        a, b = xform(H_mp, [p0, p1])
        if np.all(np.isfinite([a, b])):
            cv2.line(img, tuple(np.round(a).astype(int)),
                     tuple(np.round(b).astype(int)), color, th, cv2.LINE_AA)

    x = 0.0
    while x <= W + 1e-6:
        line((-x, -H), (-x, 0.0), (90, 90, 90), 1)
        x += step_mm
    y = 0.0
    while y <= H + 1e-6:
        line((-W, -y), (0.0, -y), (90, 90, 90), 1)
        y += step_mm
    q = canvas_quad_mm()
    for i in range(4):
        line(q[i], q[(i + 1) % 4], (255, 255, 255), 3)
    line((-W, -H), (0.0, 0.0), (0, 160, 255), 1)
    line((0.0, -H), (-W, 0.0), (0, 160, 255), 1)
    tl = xform(H_mp, [(-W + 12, -H + 34)])[0]
    if np.all(np.isfinite(tl)):
        cv2.putText(img, "TOP-LEFT", tuple(np.round(tl).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 160, 255), 2, cv2.LINE_AA)
    cen = xform(H_mp, [(-W / 2, -H / 2)])[0]
    if np.all(np.isfinite(cen)):
        cv2.putText(img, f"{W:.0f} x {H:.0f} mm  ({step_mm:.0f} mm grid)",
                    tuple(np.round(cen + [-140, 0]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def canvas_rect_image(spec, H_mp, level=255):
    """The canvas, lit white, and nothing else - the fastest way to see how
    far the mapping is out: the lit rectangle either sits on the canvas or it
    does not, and a centimetre is obvious across the room."""
    pw, ph = spec["proj_w"], spec["proj_h"]
    img = np.zeros((ph, pw, 3), np.uint8)
    q = np.round(xform(H_mp, canvas_quad_mm())).astype(np.int32)
    cv2.fillPoly(img, [q], (level, level, level))
    return img


def canvas_image_projection(spec, H_mp, img_bgr):
    """A canvas-sized reference image, warped so it lands on the canvas.

    The image is taken to span the whole canvas - which is what
    `artprojector.py gen-template` writes and what calibr-1216-exact.png is -
    so its own pixel grid maps linearly onto the canvas rectangle, and that
    map composed with mm -> projector px is the warp."""
    pw, ph = spec["proj_w"], spec["proj_h"]
    ih, iw = img_bgr.shape[:2]
    A = np.array([[ap.CANVAS_W / iw, 0.0, -ap.CANVAS_W],
                  [0.0, ap.CANVAS_H / ih, -ap.CANVAS_H],
                  [0.0, 0.0, 1.0]], np.float64)          # image px -> mm
    return cv2.warpPerspective(img_bgr, np.asarray(H_mp, np.float64) @ A,
                               (pw, ph), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


# ==========================================================================
#  SAVING
# ==========================================================================
def save_projector(path, fit, spec, cam_wh, display):
    """Write the mapping. H_mm_to_proj is the one to USE - the raw fit with the
    hand adjustment already in it - and the raw fit and the adjustment are
    written beside it so that later runs can go on tuning, and so that nobody
    has to guess afterwards which part of it was measured."""
    np.savez(path,
             H_proj_to_mm=fit["H_pm"], H_mm_to_proj=fit["H_mp"],
             H_mm_to_proj_raw=fit["H_mp_raw"], adjust=np.array(fit["adj"]),
             corner_adjust=np.asarray(fit.get("corners", np.zeros((4, 2)))),
             H_proj_to_cam=fit.get("H_pc", np.zeros((3, 3))),
             proj_w=spec["proj_w"], proj_h=spec["proj_h"],
             cell_px=spec["cell"], cols=spec["cols"], rows=spec["rows"],
             marker_px=spec["marker"], dict_id=PROJ_DICT_ID,
             cam_w=cam_wh[0], cam_h=cam_wh[1],
             canvas_w=ap.CANVAS_W, canvas_h=ap.CANVAS_H,
             rms_mm=fit["rms_mm"], rms_px=fit["rms_px"],
             n_markers=fit["n"], span=np.array([fit["span_x"], fit["span_y"]]),
             display=str(display), target=ap.TARGET,
             board=ap.GRID_BOARD if ap.TARGET == "grid" else "",
             when=time.strftime("%Y-%m-%d %H:%M:%S"))


def load_projector(path=PROJ_FILE):
    """The saved mapping as a fit dict (raw + adjustment + product), or None.

    Note what this does NOT depend on: the camera. Once fitted, the mapping is
    between the projector and the canvas, and it stays true if the camera is
    moved, re-aimed or unplugged. It goes stale when the projector or the
    canvas moves - and nothing can notice that on its own."""
    try:
        d = np.load(path)
    except FileNotFoundError:
        return None
    H_mp = np.asarray(d["H_mm_to_proj"], np.float64)
    raw = np.asarray(d["H_mm_to_proj_raw"], np.float64) if "H_mm_to_proj_raw" in d else H_mp
    adj = list(np.asarray(d["adjust"], np.float64)) if "adjust" in d else list(ZERO_ADJ)
    cor = (np.asarray(d["corner_adjust"], np.float64).reshape(4, 2)
           if "corner_adjust" in d else np.zeros((4, 2)))
    fit = {"H_mp_raw": raw, "adj": adj, "corners": cor, "measured": False,
           "n": int(d["n_markers"]) if "n_markers" in d else 0,
           "npts": 0,
           "rms_mm": float(d["rms_mm"]) if "rms_mm" in d else float("nan"),
           "max_mm": float("nan"),
           "rms_px": float(d["rms_px"]) if "rms_px" in d else float("nan"),
           "span_x": float(d["span"][0]) if "span" in d else float("nan"),
           "span_y": float(d["span"][1]) if "span" in d else float("nan"),
           "proj_w": int(d["proj_w"]), "proj_h": int(d["proj_h"]),
           "canvas_w": float(d["canvas_w"]), "canvas_h": float(d["canvas_h"]),
           "display": str(d["display"]) if "display" in d else "",
           "when": str(d["when"]) if "when" in d else ""}
    return apply_adjust(fit)


# ==========================================================================
#  THE SESSION
# ==========================================================================
def projector_size(mons, display, override=None):
    """(w, h, name, index) of the monitor the pattern will go to."""
    i = ap._display_index(mons, display)
    if i is None and display is not None:
        print(f"[proj] no monitor '{display}' - falling back to the first one")
    if i is None:
        i = 0 if mons else None
    if override:
        w, h = (int(v) for v in override.lower().split("x"))
        return w, h, (mons[i][0] if i is not None else str(display)), i
    if i is None:
        print("[proj] no monitors reported; assuming 1920x1080 "
              "(--proj-size WxH overrides)")
        return 1920, 1080, str(display), None
    n, x, y, w, h = mons[i]
    return w, h, n, i


def camera_display(mons, proj_i, want):
    """Which monitor the camera window goes to.

    The projector is showing a fullscreen pattern and the camera window has to
    be looked at while that happens, so the one place it must not open is the
    projector - which is exactly where a window inherits its position from by
    default once --display has been given to something else. Without
    --cam-display it therefore goes to the first monitor that is not the
    projector; None means "wherever the window manager likes", which is the
    only honest answer when there is just the one screen."""
    if want is not None:
        i = ap._display_index(mons, want)
        if i is None:
            print(f"[proj] no monitor '{want}' for the camera window - "
                  f"leaving it where the window manager puts it")
        return i
    for i in range(len(mons)):
        if i != proj_i:
            return i
    return None


def camera_window(name, mons, i, frac=0.62):
    """The camera window, placed on monitor `i` but NOT resized to fill it.

    artprojector.make_window() sizes a window to the whole monitor, which is
    right for a fullscreen view and wrong here: this window is a preview of a
    frame that has already been scaled down, and blowing it up to a 3440 px
    desktop only makes it soft."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if i is not None and 0 <= i < len(mons):
        _n, x, y, w, h = mons[i]
        try:
            cv2.moveWindow(name, x + 40, y + 40)
            cv2.resizeWindow(name, int(w * frac), int(h * frac))
        except cv2.error:
            pass
    return name


def flush(cap, n=4):
    """Drop the frames that were already in flight when the screen changed."""
    for _ in range(n):
        ap.read_frame(cap)


def main():
    p = argparse.ArgumentParser(
        description="Calibrate a projector against the canvas, through the "
                    "already-calibrated camera")
    p.add_argument("--cam", type=int, default=ap.CAM_INDEX)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--display", default=None,
                   help="monitor the projector is: an index or a name such as "
                        "HDMI-1 ('artprojector.py list' prints them)")
    p.add_argument("--cam-display", default=None,
                   help="monitor for the camera window (index or name). "
                        "Without it: the first monitor that is not the "
                        "projector")
    p.add_argument("--proj-size", default=None,
                   help="WxH of the projector frame, if the monitor list is "
                        "wrong about it")
    p.add_argument("--cells", type=int, default=DEFAULT_COLS,
                   help=f"lattice cells across the projector frame "
                        f"(default {DEFAULT_COLS}); more cells = more "
                        f"correspondences but smaller markers. '-'/'=' change "
                        f"it while running")
    p.add_argument("--cell-px", type=int, default=None,
                   help="cell size in projector pixels, instead of --cells "
                        f"(the floor is {MIN_CELL_PX} px, below which the "
                        f"markers stop decoding)")
    p.add_argument("--white", type=int, default=255,
                   help="brightness of the pattern's white (default 255); "
                        "lower it if the camera blows out and the markers bloom")
    p.add_argument("--frames", type=int, default=12,
                   help="frames averaged per measurement (default 12)")
    p.add_argument("--out", default=PROJ_FILE,
                   help=f"where to save the fit (default {PROJ_FILE})")
    p.add_argument("--verify-image", default="calibr-1216-exact.png",
                   help="a canvas-sized image to project with 'v'")
    p.add_argument("--target", choices=["squares", "grid"], default=None)
    p.add_argument("--board", default=None,
                   help="printed grid board, as in artprojector.py")
    p.add_argument("--canvas-w-in", type=float, default=None)
    p.add_argument("--canvas-h-in", type=float, default=None)
    p.add_argument("--no-keep-awake", action="store_true")
    args = p.parse_args()

    if args.width:
        ap.REQ_WIDTH = args.width
    if args.height:
        ap.REQ_HEIGHT = args.height

    # Mirror artprojector.main()'s setup, in the same order and for the same
    # reason: which target the calibration file was made with decides the
    # model, and the model has to exist before H means anything.
    if args.target or args.board:
        ap.TARGET_EXPLICIT = True
        if args.target == "grid" or (args.board and args.target != "squares"):
            ap.use_grid_target(args.board)
    H_cam = ap.load_calibration()
    if H_cam is None:
        print(f"[proj] no {ap.CALIB_FILE} - calibrate the CAMERA first:\n"
              f"       python artprojector.py calibrate --target grid "
              f"--board 12x16")
        return
    if ap.TARGET == "grid" and not (args.canvas_w_in or args.canvas_h_in):
        ap.CANVAS_W, ap.CANVAS_H = ap.gt.board_size_mm(ap.GRID_BOARD)
    if args.canvas_w_in:
        ap.CANVAS_W = args.canvas_w_in * ap.MM_PER_IN
    if args.canvas_h_in:
        ap.CANVAS_H = args.canvas_h_in * ap.MM_PER_IN

    mons = ap.list_displays()
    if mons:
        print("[proj] monitors: " + ", ".join(
            f"[{i}] {n} {w}x{h}+{x}+{y}" for i, (n, x, y, w, h) in enumerate(mons)))
    pw, ph, pname, proj_i = projector_size(mons, args.display, args.proj_size)
    cam_i = camera_display(mons, proj_i, args.cam_display)
    cname = mons[cam_i][0] if cam_i is not None else "wherever it opens"
    spec = pattern_spec(pw, ph, args.cells, args.white, args.cell_px)
    args.cells = spec["cols"]            # so '-'/'=' start from what is shown
    print(f"[proj] projector '{pname}' {pw}x{ph}: lattice {spec['cols']}x"
          f"{spec['rows']} cells of {spec['cell']} px, "
          f"{spec['marker']} px markers (DICT_5X5_1000)")
    print(f"[proj] camera window on '{cname}' (--cam-display moves it)")
    if spec["asked"] > spec["cols"]:
        print(f"[proj] {spec['asked']} cells across would be "
              f"{pw // spec['asked']} px cells, under the {MIN_CELL_PX} px "
              f"floor where the markers stop decoding - using {spec['cols']}")
    print(f"[proj] canvas {ap.CANVAS_W:.0f}x{ap.CANVAS_H:.0f} mm; camera "
          f"calibration from {ap.CALIB_FILE} "
          f"(k1={ap.DIST_K1:+.4f} k2={ap.DIST_K2:+.4f})")
    print("[proj] the camera must not have moved since it was calibrated - "
          "nothing here can tell if it has.\n"
          "       Dim the room, aim the camera at the canvas, and keep the "
          "keyboard focus on the camera window.\n"
          "       c measure   ENTER (or S) save   p pattern  g test grid  "
          "r white rectangle  v image  k black\n"
          "       a/d w/s move 2mm   z/x scale   [ ] scale X   - = scale Y   "
          ", . rotate   i reset\n"
          "       TAB picks what a/d/w/s move: everything, or one corner "
          "(arrowed on the canvas) - five states\n"
          "       9/0 dim/brighten   n/m fewer/more cells   q quit")

    fit = load_projector(args.out)
    if fit is not None:
        print(f"[proj] {args.out} loaded: a fit from {fit['when']} "
              f"({fit['rms_mm']:.2f} mm rms over {fit['n']} markers). "
              f"'r'/'g'/'v' project through it straight away; 'c' only if the "
              f"projector or the canvas has moved.")
        if abs(fit["canvas_w"] - ap.CANVAS_W) > 0.5 or \
                abs(fit["canvas_h"] - ap.CANVAS_H) > 0.5:
            print(f"[proj] NOTE: it was made for a "
                  f"{fit['canvas_w']:.0f}x{fit['canvas_h']:.0f} mm canvas and "
                  f"the canvas configured now is "
                  f"{ap.CANVAS_W:.0f}x{ap.CANVAS_H:.0f} mm. The mapping itself "
                  f"is unaffected - it is projector px to mm - but everything "
                  f"drawn 'the size of the canvas' has just changed size.")
        if adjusted(fit):
            print(f"[proj] !! it carries a HAND ADJUSTMENT, applied to "
                  f"everything projected: {adjust_text(fit)}\n"
                  f"        That is someone's correction for the canvas not "
                  f"being the size the model says, or for a scaled print of "
                  f"the board - not a measurement. 'i' zeroes it.")
        if (fit["proj_w"], fit["proj_h"]) != (pw, ph):
            print(f"[proj] !! it was made on a {fit['proj_w']}x{fit['proj_h']} "
                  f"projector frame and this one is {pw}x{ph} - the mapping is "
                  f"in pixels, so it is wrong by that ratio. Measure again.")

    verify_img = cv2.imread(args.verify_image) if args.verify_image else None
    if args.verify_image and verify_img is None:
        print(f"[proj] could not read --verify-image {args.verify_image}")

    awake = None if args.no_keep_awake else ap.KeepAwake("projcalib").start()
    cap = ap.open_camera(args.cam)
    pwin = ap.make_window("projector", fullscreen=True,
                          display=proj_i if proj_i is not None else args.display)
    cwin = camera_window("projector calibration", mons, cam_i)
    mode, dirty = "pattern", True
    corner_i = None                  # None = a/d/w/s move the whole projection
    try:
        while True:
            frame = ap.read_frame(cap)
            if frame is None:
                print("[proj] no frame from the camera"); break
            cam_h, cam_w = frame.shape[:2]
            # H_cam is in the pixels of the resolution it was fitted at
            Hc = H_cam
            if ap.CALIB_CAM_W and (cam_w, cam_h) != (ap.CALIB_CAM_W, ap.CALIB_CAM_H):
                Hc = ap.scale_homography(H_cam, (ap.CALIB_CAM_W, ap.CALIB_CAM_H),
                                         (cam_w, cam_h))

            if dirty:
                if mode == "pattern":
                    shown = pattern_image(spec)
                elif mode == "grid" and fit is not None:
                    shown = canvas_test_image(spec, fit["H_mp"])
                elif mode == "rect" and fit is not None:
                    shown = canvas_rect_image(spec, fit["H_mp"], spec["white"])
                elif mode == "image" and fit is not None and verify_img is not None:
                    shown = canvas_image_projection(spec, fit["H_mp"], verify_img)
                else:
                    shown = np.zeros((ph, pw, 3), np.uint8)
                if corner_i is not None and fit is not None and mode != "pattern":
                    corner_arrow(shown, fit["H_mp"], corner_i)
                cv2.imshow(pwin, shown)
                cv2.waitKey(1)
                flush(cap)
                dirty = False
                continue

            live, foreign, clipped = (detect_projected(frame, spec, ap.DIST_K1,
                                                       ap.DIST_K2)
                                      if mode == "pattern" else ([], 0, 0))

            vis = ap.undistort(frame, ap.DIST_K1, ap.DIST_K2).copy()
            for mid, q in live:
                cv2.polylines(vis, [np.round(q).astype(np.int32)], True,
                              (0, 220, 0), 2, cv2.LINE_AA)
                cen = np.round(q.mean(axis=0)).astype(int)
                cv2.putText(vis, str(mid), tuple(cen), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2, cv2.LINE_AA)
            # the canvas, where the CAMERA calibration says it is
            cq = np.round(xform(Hc, canvas_quad_mm())).astype(np.int32)
            cv2.polylines(vis, [cq], True, (255, 0, 255), 2, cv2.LINE_AA)
            if fit is not None:
                # and the projected frame, where THIS calibration says it is.
                # Composed from H_pm rather than taken from the fit, so it is
                # in the camera pixels of THIS frame and follows the hand
                # adjustment - and so a fit loaded from the file, which never
                # saw this camera, draws it too.
                fq = xform(Hc @ fit["H_pm"], [[0, 0], [pw, 0], [pw, ph], [0, ph]])
                if np.all(np.isfinite(fq)) and np.abs(fq).max() < 1e6:
                    cv2.polylines(vis, [np.round(fq).astype(np.int32)], True,
                                  (0, 200, 255), 2, cv2.LINE_AA)
                if corner_i is not None:
                    corner_arrow(vis, Hc, corner_i)

            hud = [f"mode={mode}  markers={len(live)}"
                   f"{f'  [{clipped} at the frame edge]' if clipped else ''}"
                   f"{f'  [{foreign} not from this lattice]' if foreign else ''}"
                   f"  lattice={spec['cols']}x{spec['rows']}@{spec['cell']}px"
                   f"  white={spec['white']}",
                   (f"fit{'' if fit.get('measured') else ' (from ' + args.out + ')'}"
                    f": {fit['n']} markers, {fit['rms_mm']:.2f} mm rms; cell = "
                    f"{spec['cell'] / max(local_scale(fit['H_mp'], -ap.CANVAS_W / 2, -ap.CANVAS_H / 2), 1e-9):.0f}"
                    f" mm on the canvas" if fit else
                    "no fit yet - get markers into the frame and press 'c'"),
                   (adjust_text(fit) + ("" if adjusted(fit) else "  (none)")
                    if fit else ""),
                   (f"TAB: a/d w/s move the {CORNER_NAMES[corner_i]} CORNER "
                    f"alone (arrow on the canvas)" if corner_i is not None else
                    "TAB: a/d w/s move the whole projection; TAB picks a "
                    "corner instead"),
                   "magenta = canvas (camera calib)   orange = projector frame "
                   "(this calib)",
                   "c measure  ENTER save  p/g/r/v/k project  a/d w/s move  "
                   "z/x scale  , . rot  i reset  n/m cells  9/0 white  q quit"]
            y = 34
            for ln in hud:
                cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                y += 32
            cv2.imshow(cwin, ap._resize_max(vis, 1400))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                if mode != "pattern":
                    mode, dirty = "pattern", True
                    continue
                seen = {}
                for _ in range(args.frames):
                    f = ap.read_frame(cap)
                    if f is None:
                        break
                    for mid, q in detect_projected(f, spec, ap.DIST_K1,
                                                   ap.DIST_K2)[0]:
                        seen.setdefault(mid, []).append(q)
                obs = average_observations(seen, max(2, args.frames // 2))
                newfit = fit_projector(obs, spec, Hc)
                if newfit is None:
                    print(f"[proj] not enough steady markers "
                          f"({len(obs)} of {len(seen)} seen) - two are the "
                          f"minimum and four spread across the frame is what "
                          f"it wants. More light on the pattern, less on the "
                          f"room; or fewer cells ('n') for bigger markers.")
                    if not seen:
                        print("[proj] nothing at all decoded. If the pattern is "
                              "plainly there on the canvas, check the "
                              "projector's own image flip: a mirrored marker "
                              "is not a marker any more and will never decode "
                              "(rear-projection / ceiling-mount settings do "
                              "this). Blown-out white does it too - '9' dims "
                              "the pattern.")
                else:
                    # A hand adjustment outlives the measurement it was made
                    # against, deliberately: it usually compensates something
                    # about the canvas or the printed board, which a new
                    # measurement of the projector does not change. It is
                    # announced every time rather than silently re-applied.
                    if fit is not None and adjusted(fit):
                        newfit["adj"] = list(fit["adj"])
                        newfit["corners"] = np.array(fit.get(
                            "corners", np.zeros((4, 2))), np.float64)
                        apply_adjust(newfit)
                        print(f"[proj] the hand adjustment carried over onto "
                              f"the new fit: {adjust_text(newfit)}  ('i' zeroes it)")
                    fit = newfit
                    for ln in geometry_report(fit, spec):
                        print(ln)
                    print("[proj] 'r'/'g'/'v' project it back onto the canvas "
                          "to check it by eye; ENTER saves.")
            elif key in (13, 10, ord('S')):
                if fit is None:
                    print("[proj] nothing measured yet.")
                else:
                    save_projector(args.out, fit, spec, (cam_w, cam_h), pname)
                    print(f"[proj] saved to {args.out}: mm -> projector px, "
                          f"{fit['rms_mm']:.2f} mm rms over {fit['n']} markers"
                          + (f", plus the hand adjustment ({adjust_text(fit)}), "
                             f"which is in H_mm_to_proj; the raw fit is kept "
                             f"beside it." if adjusted(fit) else ".")
                          + " It stays true until the projector or the canvas "
                            "moves.")
            elif key in (ord('p'), ord('g'), ord('r'), ord('v'), ord('k')):
                want = {"p": "pattern", "g": "grid", "r": "rect",
                        "v": "image", "k": "black"}[chr(key)]
                if want not in ("pattern", "black") and fit is None:
                    print("[proj] measure first ('c') - there is nothing to "
                          "project the canvas through yet.")
                elif want == "image" and verify_img is None:
                    print("[proj] no --verify-image to show.")
                else:
                    mode, dirty = want, True
                    if want == "grid":
                        # The two ways the projected rectangle can miss the
                        # canvas look identical on the wall and are fixed in
                        # completely different places, so say how to tell them
                        # apart while the grid is up.
                        print("[proj] measure the 50 mm squares on the canvas "
                              "with a ruler. If they are not 50 mm, the "
                              "millimetre itself is wrong - a scaled print of "
                              "the board - and EVERYTHING is off by that "
                              "factor, overlay included. If they are 50 mm but "
                              "the white border misses the canvas edge, the "
                              "canvas is simply not the size the model thinks "
                              "(--canvas-w-in/--canvas-h-in), and a/s/d/w z/x "
                              "will paper over it here only.")
            elif key in (ord('9'), ord('0')):
                spec["white"] = int(np.clip(spec["white"]
                                            + (-15 if key == ord('9') else 15),
                                            60, 255))
                dirty = True
            elif key in (ord('n'), ord('m')):
                # The lattice is only how the fit was MEASURED; changing it
                # does not invalidate a mapping that is already fitted.
                args.cells = max(2, args.cells + (-1 if key == ord('n') else 1))
                spec = pattern_spec(pw, ph, args.cells, spec["white"])
                args.cells = spec["cols"]
                print(f"[proj] lattice {spec['cols']}x{spec['rows']} cells of "
                      f"{spec['cell']} px, {spec['marker']} px markers")
                mode, dirty = "pattern", True
            elif key == 9:                                   # TAB
                # five states: the whole projection, then each corner. The
                # corner ones are what the similarity cannot reach.
                corner_i = None if corner_i == 3 else (
                    0 if corner_i is None else corner_i + 1)
                if corner_i is None:
                    print("[proj] a/d w/s move the whole projection again")
                else:
                    off = np.asarray(fit["corners"])[corner_i] if fit is not None \
                        else (0.0, 0.0)
                    print(f"[proj] a/d w/s now move the "
                          f"{CORNER_NAMES[corner_i]} corner alone "
                          f"(now {off[0]:+.1f},{off[1]:+.1f} mm) - the arrow on "
                          f"the canvas points at it")
                dirty = True
            elif (fit is not None and corner_i is not None
                  and key in (ord('a'), ord('d'), ord('w'), ord('s'))):
                c = np.asarray(fit["corners"], np.float64).reshape(4, 2).copy()
                c[corner_i] += {ord('a'): (-MOVE_MM, 0.0),
                                ord('d'): (+MOVE_MM, 0.0),
                                ord('w'): (0.0, -MOVE_MM),
                                ord('s'): (0.0, +MOVE_MM)}[key]
                fit["corners"] = c
                apply_adjust(fit)
                if mode == "pattern":
                    print(f"[proj] {CORNER_NAMES[corner_i]} corner "
                          f"{c[corner_i][0]:+.1f},{c[corner_i][1]:+.1f} mm - "
                          f"press 'r' or 'g' to see it on the canvas")
                dirty = True
            elif fit is not None and key in (ord('a'), ord('d'), ord('w'),
                                             ord('s'), ord('z'), ord('x'),
                                             ord('['), ord(']'), ord('-'),
                                             ord('='), ord(','), ord('.'),
                                             ord('i')):
                dx, dy, sx, sy, th = fit["adj"]
                if key == ord('a'):
                    dx -= MOVE_MM
                elif key == ord('d'):
                    dx += MOVE_MM
                elif key == ord('w'):
                    dy -= MOVE_MM
                elif key == ord('s'):
                    dy += MOVE_MM
                elif key == ord('x'):
                    sx, sy = sx * SCALE_STEP, sy * SCALE_STEP
                elif key == ord('z'):
                    sx, sy = sx / SCALE_STEP, sy / SCALE_STEP
                elif key == ord(']'):
                    sx *= SCALE_STEP
                elif key == ord('['):
                    sx /= SCALE_STEP
                elif key == ord('='):
                    sy *= SCALE_STEP
                elif key == ord('-'):
                    sy /= SCALE_STEP
                elif key == ord('.'):
                    th += ROT_DEG
                elif key == ord(','):
                    th -= ROT_DEG
                elif key == ord('i'):
                    dx, dy, sx, sy, th = ZERO_ADJ
                    fit["corners"] = np.zeros((4, 2))   # corners go too
                fit["adj"] = [dx, dy, sx, sy, th]
                apply_adjust(fit)
                if mode == "pattern":
                    # the lattice is drawn in projector px and does not move,
                    # so there would be nothing to see
                    print(f"[proj] {adjust_text(fit)} - press 'r' or 'g' to "
                          f"see it on the canvas")
                dirty = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if awake is not None:
            awake.stop()


if __name__ == "__main__":
    main()
