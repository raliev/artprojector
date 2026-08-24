#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthtest.py - end-to-end accuracy check on a synthetic camera frame.

Renders the printed sheet (from the model, at the real stroke width) as seen
by a camera with a known homography and known radial distortion, then runs the
real pipeline over it - detect_squares -> consensus quad -> refine_homography -
and reports the error where it actually matters: how far a point drawn at a
canvas corner would land from where it belongs, in millimetres on the canvas.

  python synthtest.py            # the default scene
  python synthtest.py --n 20     # 20 random viewpoints, summary statistics

The point of the test is that the ground truth is a homography the test picked,
so nothing in the measurement chain can quietly agree with itself: a bias in
the detector or a wrong square size shows up as millimetres at the corners.

  python synthtest.py --target grid --n 20        # the ArUco grid instead
  python synthtest.py --target grid --dist 90     # ...seen from 9 cm away

The grid runs are stricter still, because the scene is not rendered from the
model at all: it is the PRINTED PDF out of templates/, rasterised. If
gridtarget.py and artprojector.py ever disagree about where a marker or a line
is, that shows up here as millimetres, which is the one way the two can be
held to the same geometry.
"""
import argparse
import os
import subprocess
import numpy as np
import cv2

import artprojector as ap
import gridtarget as gt


def render_scene(H_true, k1, k2, w, h, ppm=8.0, blur=1.2, noise=2.0,
                 paper=(226, 232, 240), ink=28, seed=0):
    """A camera frame of the sheet: the model rendered in mm, warped by H_true,
    then distorted. The sheet is drawn on a paper-coloured quad over a mildly
    textured background, like a real photo of paper on a canvas."""
    rng = np.random.default_rng(seed)
    model = ap.build_model()
    pts = np.vstack(list(model.values()))
    x0, y0 = pts.min(axis=0) - 12.0
    x1, y1 = pts.max(axis=0) + 12.0
    # sheet image, ppm px per mm
    sw, sh = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    sheet = np.full((sh, sw, 3), paper, np.uint8)
    th = max(1, int(round(ap.STROKE_MM * ppm)))
    for corners in model.values():
        q = np.round((np.asarray(corners) - [x0, y0]) * ppm).astype(np.int32)
        cv2.polylines(sheet, [q], True, (ink, ink, ink), th, cv2.LINE_AA)
    # sheet px -> mm -> image
    S = np.array([[1 / ppm, 0, x0], [0, 1 / ppm, y0], [0, 0, 1.0]])
    frame = np.full((h, w, 3), 60, np.uint8)
    frame = cv2.GaussianBlur(rng.integers(40, 110, (h, w, 3)).astype(np.uint8),
                             (0, 0), 25)
    warped = cv2.warpPerspective(sheet, H_true @ S, (w, h),
                                 flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), H_true @ S,
                               (w, h), flags=cv2.INTER_AREA)
    frame[mask > 127] = warped[mask > 127]
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    if noise:
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0,
                        255).astype(np.uint8)
    if k1 or k2:
        # push the ideal image through the lens: sample it where each raw
        # pixel looks
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        p = ap.undistort_points(np.stack([xx.ravel(), yy.ravel()], 1), k1, k2, w, h)
        frame = cv2.remap(frame, p[:, 0].reshape(h, w).astype(np.float32),
                          p[:, 1].reshape(h, w).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return frame


_SHEET_CACHE = {}


def rasterize_board(board, ppm, pdf=None):
    """The printed PDF as pixels -> (image, px_per_mm actually achieved).

    Rendering the real PDF instead of re-drawing the board from the model is
    the whole point of the grid test: the marker bits, the 4.7 mm clear ring,
    the line width and the 1-inch pitch all come from the file that goes to the
    printer, so a disagreement between the generator and the detector cannot
    cancel out."""
    pdf = pdf or os.path.join("templates", f"grid-{board}-full.pdf")
    key = (pdf, round(ppm, 4))
    if key in _SHEET_CACHE:
        return _SHEET_CACHE[key]
    if not os.path.exists(pdf):
        raise SystemExit(f"{pdf} not found - run `python gridtarget.py` first")
    out = f"/tmp/synthtest-{board}-{ppm:.2f}"
    subprocess.run(["pdftoppm", "-r", f"{ppm * 25.4:.6f}", "-png", "-gray",
                    "-singlefile", pdf, out], check=True)
    img = cv2.imread(out + ".png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not rasterise {pdf}")
    # ppm is exactly what was asked for: pdftoppm honours -r and rounds the
    # image size UP, so deriving the scale back from img.shape would be off by
    # the rounding. Getting this wrong is not academic - it silently rescales
    # the whole ground truth and every error below inherits it.
    _SHEET_CACHE[key] = (img, ppm)
    return _SHEET_CACHE[key]


TRUE_LENS = None                   # (k3, p1, p2, cx, cy) of the rendered lens,
                                   # cx/cy in units of f = max(w,h); see --lens


def true_lens_maps(k1, k2, w, h):
    """Where each RAW pixel's content sits on the ideal image, per TRUE_LENS.

    Deliberately not ap.undistort_points(): that reads the module's own lens
    state, which is the thing under test here. A test whose ground truth is
    produced by the estimator cannot catch the estimator being wrong."""
    k3, p1, p2, cx, cy = TRUE_LENS or (0.0, 0.0, 0.0, 0.0, 0.0)
    f = float(max(w, h))
    K = np.array([[f, 0.0, w / 2.0 + cx * f],
                  [0.0, f, h / 2.0 + cy * f],
                  [0.0, 0.0, 1.0]])
    D = np.array([k1, k2, p1, p2, k3])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    p = cv2.undistortPoints(pts, K, D, P=K).reshape(h, w, 2)
    return p[:, :, 0].copy(), p[:, :, 1].copy()


def render_grid_scene(H_true, k1, k2, w, h, ppm=8.0, blur=1.2, noise=2.0,
                      paper=232, ink=28, seed=0, board=None):
    """A camera frame of the printed grid, warped by H_true and distorted."""
    rng = np.random.default_rng(seed)
    board = board or ap.GRID_BOARD
    sheet, real_ppm = rasterize_board(board, ppm)
    # paper white -> the paper grey, ink black -> the ink grey
    sheet = (ink + sheet.astype(np.float32) * (paper - ink) / 255.0)
    sheet = cv2.cvtColor(sheet.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    sh, sw = sheet.shape[:2]
    # Sheet pixel -> world mm. The half pixel is not a fudge: the rasteriser
    # maps board mm 0 to the left EDGE of pixel 0, while OpenCV puts pixel 0 at
    # coordinate 0, i.e. at its centre. Without it the whole ground truth sits
    # half a sheet pixel off and every fit below is condemned to be exactly
    # that wrong, which at 8 px/mm is 0.06 mm of pure bookkeeping.
    ox, oy = ap.grid_origin_mm()
    S = np.array([[1 / real_ppm, 0, ox + 0.5 / real_ppm],
                  [0, 1 / real_ppm, oy + 0.5 / real_ppm],
                  [0, 0, 1.0]])

    frame = cv2.GaussianBlur(rng.integers(40, 110, (h, w, 3)).astype(np.uint8),
                             (0, 0), 25)
    warped = cv2.warpPerspective(sheet, H_true @ S, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), H_true @ S,
                               (w, h), flags=cv2.INTER_AREA)
    frame[mask > 127] = warped[mask > 127]
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    if noise:
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0,
                        255).astype(np.uint8)
    if k1 or k2 or TRUE_LENS:
        mx, my = true_lens_maps(k1, k2, w, h)
        frame = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return frame


def aim_homography(w, h, yaw, pitch, roll, dist_mm, aim_mm):
    """Like scene_homography, but pointed at a chosen world point.

    dist_mm doubles as the framing: with f pinned to max(w,h) the frame covers
    roughly dist_mm across its long side, so dist=90 is a close-up of three or
    four cells and dist=700 takes in a whole 16x20 board."""
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.deg2rad([pitch, yaw, roll]).astype(np.float64))
    t = np.array([[0.0], [0.0], [dist_mm]])
    Rt = np.hstack([R[:, :2],
                    R @ np.array([[-aim_mm[0]], [-aim_mm[1]], [0.0]]) + t])
    H = K @ Rt
    return H / H[2, 2]


def run_one_grid(seed, w=2592, h=1944, k1=-0.19, k2=0.02, dist=None,
                 verbose=True, blur=1.2, ink=28, noise=2.0, fit_lens=False):
    """One grid viewpoint, end to end. Returns (marker-only, refined, info).

    fit_lens=False hands the pipeline the true k1/k2, which measures the fit of
    H and nothing else. fit_lens=True makes it earn them from the frame the way
    'a' does in calibrate, which is the only way the lens MODEL is on trial:
    with --lens the frame is rendered through a lens the two-parameter model
    cannot express, and the millimetres that come out are what a real camera
    costs."""
    rng = np.random.default_rng(seed)
    cell_model = ap.build_grid_cell_model()
    marker_model = ap.build_grid_marker_model()
    cols, rows = gt.board_spec(ap.GRID_BOARD)

    yaw = rng.uniform(-35, 35); pitch = rng.uniform(-30, 30)
    roll = rng.uniform(-8, 8)
    d = dist if dist else rng.uniform(90, 700)
    # aim somewhere on the board, keeping the aim point off the very edge so a
    # close-up still has cells all round it
    aim = (rng.uniform(1.5, cols - 1.5) * gt.CELL_MM - cols * gt.CELL_MM,
           rng.uniform(1.5, rows - 1.5) * gt.CELL_MM - rows * gt.CELL_MM)
    H_true = aim_homography(w, h, yaw, pitch, roll, d, aim)
    frame = render_grid_scene(H_true, k1, k2, w, h, seed=seed, blur=blur,
                              ink=ink, noise=noise)

    # --- the pipeline, as calibrate_grid() runs it ---
    lens_note = ""
    if fit_lens:
        # exactly the 'a' key: detect with no correction at all, fit the lens to
        # the corners, then carry on with what was fitted
        ap.set_lens_extras()
        first, _, _ = ap.detect_grid_cells(frame, 0.0, 0.0)
        if not first:
            if verbose:
                print(f"  seed {seed}: no marker detected (d={d:.0f}mm) - skipped")
            return None
        k1, k2, li = ap.fit_lens(first, w, h, 0.0, 0.0)
        lens_note = (f"  fitted k1={k1:+.3f} k2={k2:+.3f} "
                     f"{'+full' if li['took'] else 'only'} "
                     f"({li['r2']:.3f}->{li['r7']:.3f}px)")
    matches, foreign, clipped = ap.detect_grid_cells(frame, k1, k2)
    if not matches:
        if verbose:
            print(f"  seed {seed}: no marker detected (d={d:.0f}mm) - skipped")
        return None
    # Every detected cell must be the cell it really is: a mis-decode is not a
    # millimetre error, it is a 25 mm one, and it must never pass silently. The
    # bar is a fraction of the marker's own size, because that is the scale a
    # real mis-identification has - the nearest wrong answer is a whole cell
    # away, i.e. 160% of a marker side. `worst` reports the corner error itself
    # separately, which is the quantity that is merely accuracy: ArUco puts the
    # boundary of a blurred marker a pixel or two inside the ink, and that is
    # what the line snap afterwards is for.
    wrong, worst = [], 0.0
    for (col, row, quad) in matches:
        truth = marker_model[(col, row)]
        p = np.hstack([truth, np.ones((4, 1))]) @ H_true.T
        p = p[:, :2] / p[:, 2:3]          # H_true is in undistorted px, as is quad
        e = np.linalg.norm(p - quad, axis=1).max()
        side = np.linalg.norm(p[1] - p[0])
        worst = max(worst, e)
        if e > 0.4 * side:
            wrong.append((col, row))
    H_mark, matches = ap.compute_homography_markers(matches, marker_model)
    keys = set((c, r) for (c, r, _) in matches)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H_ref, rk1, rk2, info = ap.refine_homography(gray, H_mark, cell_model,
                                                 k1, k2, keys=keys)

    e_mark = canvas_error(H_true, H_mark)
    e_ref = canvas_error(H_true, H_ref)
    # error over the cells actually in view - what the overlay draws on
    seen = np.vstack([cell_model[k] for k in keys]).astype(np.float64)
    s_mark = canvas_error(H_true, H_mark, seen)
    s_ref = canvas_error(H_true, H_ref, seen)
    n, nc, nr, span = ap.grid_view_span(matches)
    if verbose:
        print(f"  seed {seed}: yaw={yaw:+5.1f} pitch={pitch:+5.1f} d={d:3.0f}mm "
              f" {n} markers over {nc}x{nr} cells (span {span:3.0f}mm)"
              f"  snap={info['rms_mm']:.3f}mm ({info['n']}/{info['n_total']}pts)"
              f"  worst marker {worst:.1f}px"
              + (f"  MIS-ID {wrong}" if wrong else "")
              + (f"  {clipped} clipped" if clipped else "")
              + (f"  {foreign} foreign" if foreign else "")
              + lens_note)
        print(f"      in view: markers {s_mark.max():.3f} -> refined "
              f"{s_ref.max():.3f} mm    whole canvas: {e_mark.max():.3f} -> "
              f"{e_ref.max():.3f} mm")
    return s_ref.max(), e_ref.max(), info, bool(wrong), span


def scene_homography(w, h, yaw, pitch, roll, dist_mm, seed=0):
    """A plausible easel viewpoint -> H (mm -> undistorted px)."""
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.deg2rad([pitch, yaw, roll]).astype(np.float64))
    # centre the target in the frame
    model = ap.build_model()
    c = np.vstack(list(model.values())).mean(axis=0)
    t = np.array([[0.0], [0.0], [dist_mm]])
    Rt = np.hstack([R[:, :2], R @ np.array([[-c[0]], [-c[1]], [0.0]]) + t])
    H = K @ Rt
    return H / H[2, 2]


def canvas_error(H_true, H_est, probes=None):
    """mm on the canvas between where a feature belongs and where the estimate
    puts it: project with the estimate, read back with the truth.

    Deliberately in the UNDISTORTED plane, where both homographies live. Going
    through the lens instead would measure nothing at the far canvas corners:
    those sit well outside the frame, at a radius where the two-term radial
    polynomial is no longer monotonic, so inverting it there produces over a
    millimetre of pure arithmetic - the ground truth scores 1.8 mm against
    itself. That is a statement about the distortion model outside the image,
    not about the calibration."""
    if probes is None:
        probes = np.array([[-ap.CANVAS_W, -ap.CANVAS_H], [0, -ap.CANVAS_H],
                           [-ap.CANVAS_W, 0], [0, 0],
                           [-ap.CANVAS_W / 2, -ap.CANVAS_H / 2]], np.float64)
    q = np.hstack([probes, np.ones((len(probes), 1))]) @ np.asarray(H_est).T
    q = q[:, :2] / q[:, 2:3]
    p = np.hstack([q, np.ones((len(q), 1))]) @ np.linalg.inv(H_true).T
    p = p[:, :2] / p[:, 2:3]
    return np.linalg.norm(p - probes, axis=1)


def ridge_measure_line_offsets(gray, pts_mm, nrm_mm, H, k1, k2,
                               search_mm=0.62, ink_mm=None, n_samples=21):
    """The measurement artprojector used before: the darkest point across the
    line, parabola-refined. Kept here to A/B against the flank estimator - on
    a clean render the two are equivalent, and the whole question is what
    happens once the frame is soft enough for neighbouring lines to bleed
    into each other. Use --ridge --blur 2.5."""
    h, w = gray.shape[:2]
    p0 = ap.project_raw(pts_mm, H, k1, k2, w, h)
    p1 = ap.project_raw(pts_mm + nrm_mm * 0.5, H, k1, k2, w, h)
    nv = p1 - p0
    ln = np.linalg.norm(nv, axis=1, keepdims=True)
    px_per_mm = np.maximum(ln / 0.5, 1e-9)
    nv = nv / np.maximum(ln, 1e-9)
    t = np.linspace(-1.0, 1.0, n_samples)[None, :]
    off = t * (search_mm * px_per_mm)
    prof = ap.sample_gray(gray, p0[:, None, :] + nv[:, None, :] * off[:, :, None])
    good = ~np.isnan(prof).any(axis=1)
    prof = np.where(np.isnan(prof), 1e9, prof)
    i = np.argmin(prof, axis=1)
    idx = np.arange(len(prof))
    good &= (i > 0) & (i < n_samples - 1)
    i = np.clip(i, 1, n_samples - 2)
    y0_, y1_, y2_ = prof[idx, i - 1], prof[idx, i], prof[idx, i + 1]
    good &= (np.where(prof > 1e8, -np.inf, prof).max(axis=1) - y1_) > 12.0
    den = (y0_ - 2 * y1_ + y2_)
    good &= den > 1e-6
    sub = np.clip(np.where(np.abs(den) > 1e-9,
                           0.5 * (y0_ - y2_) / np.maximum(den, 1e-9), 0.0), -1, 1)
    return p0, nv, off[idx, i] + sub * (off[:, 1] - off[:, 0]), good


def legacy_model():
    """The model this tool used before the sheet was read off calibr.svg:
    64 mm squares, touching horizontally, 0.6 mm between the rows, margins
    measured to the edge of the black line rather than to its centre.

    Run against a scene rendered from the real geometry it shows what the old
    calibration was actually doing - and, more to the point, that its own
    reprojection error stayed small while it did it."""
    square, gap_y, right, bottom = 64.0, 0.6, 14.0, 145.0
    model = {}
    for col in range(3):
        x0 = -right - square - (2 - col) * square
        for row in range(2):
            y0 = -bottom - square - (1 - row) * (square + gap_y)
            model[(col, row)] = np.array(
                [[x0, y0], [x0 + square, y0], [x0 + square, y0 + square],
                 [x0, y0 + square]], np.float32)
    return model


def run_one(seed, w=2592, h=1944, k1=-0.19, k2=0.02, verbose=True,
            legacy=False, blur=1.2, ink=28, noise=2.0):
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-25, 25); pitch = rng.uniform(-20, 20)
    roll = rng.uniform(-8, 8); dist = rng.uniform(280, 460)
    H_true = scene_homography(w, h, yaw, pitch, roll, dist)
    frame = render_scene(H_true, k1, k2, w, h, seed=seed, blur=blur,
                         ink=ink, noise=noise)

    # --- the pipeline, as calibrate() runs it ---
    und = ap.undistort(frame, k1, k2)          # start from the true distortion
    quads, n_paired = ap.detect_squares(und)
    items, nc, nr = ap.assign_local_grid(quads)
    if (nc, nr) != (ap.COLS, ap.ROWS):
        if verbose:
            print(f"  seed {seed}: detected {nc}x{nr} squares - skipped")
        return None
    matches = [(c, r, q) for (c, r, q) in items]
    # the scene is always rendered from the true sheet; only what the pipeline
    # BELIEVES about it changes
    model = legacy_model() if legacy else ap.build_model()
    H_con, _, _ = ap.compute_homography_consensus(matches, model)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H_ref, rk1, rk2, info = ap.refine_homography(gray, H_con, model, k1, k2)

    e_con = canvas_error(H_true, H_con)
    e_ref = canvas_error(H_true, H_ref)
    blk = np.vstack(list(ap.build_model().values())).astype(np.float64)
    b_con = canvas_error(H_true, H_con, blk)
    b_ref = canvas_error(H_true, H_ref, blk)
    if verbose:
        print(f"  seed {seed}: yaw={yaw:+5.1f} pitch={pitch:+5.1f} "
              f"d={dist:3.0f}mm  paired={n_paired}/6  "
              f"snap={info['rms_mm']:.3f}mm ({info['n']}/{info['n_total']}pts)")
        print(f"      canvas-corner error  consensus {e_con.max():.3f} mm   "
              f"refined {e_ref.max():.3f} mm   "
              f"(on the sheet: {b_con.max():.3f} -> {b_ref.max():.3f} mm)")
    return e_con.max(), e_ref.max(), info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--k1", type=float, default=-0.19)
    p.add_argument("--k2", type=float, default=0.02)
    p.add_argument("--legacy", action="store_true",
                   help="run the pipeline with the pre-SVG square model")
    p.add_argument("--ridge", action="store_true",
                   help="measure lines at their darkest point, as before")
    p.add_argument("--blur", type=float, default=1.2, help="render blur, px")
    p.add_argument("--ink", type=int, default=28, help="ink gray level")
    p.add_argument("--noise", type=float, default=2.0)
    p.add_argument("--target", choices=["squares", "grid"], default="squares")
    p.add_argument("--board", default="16x20", help="which grid board to test")
    p.add_argument("--dist", type=float, default=None,
                   help="camera distance in mm; also the width of the frame in "
                        "mm, so 90 is a 3-cell close-up (default: random)")
    p.add_argument("--lens", default=None,
                   help="render through a fuller lens: k3,p1,p2,cx,cy with the "
                        "principal point offset in units of f=max(w,h) "
                        "(0.0135 is 35 px at 2592). A plausible webcam is "
                        "'-0.02,0.001,-0.0005,0.0135,-0.0096'. Implies "
                        "--fit-lens: being told the truth about a lens the "
                        "model cannot express would test nothing")
    p.add_argument("--fit-lens", action="store_true",
                   help="fit the lens from the frame ('a' in calibrate) instead "
                        "of handing the pipeline the k1/k2 it was rendered with")
    p.add_argument("--two-param", action="store_true",
                   help="fit k1/k2 only, as before the full lens model existed - "
                        "run it against --lens to see what the extra terms buy")
    a = p.parse_args()

    if a.two_param:
        ap.LENS_FIT_GAIN = -1.0          # no gain can ever be good enough
    if a.lens:
        vals = [float(v) for v in a.lens.split(",")]
        TRUE_LENS_SET = tuple(vals + [0.0] * (5 - len(vals)))
        globals()["TRUE_LENS"] = TRUE_LENS_SET
        a.fit_lens = True

    if a.target == "grid":
        ap.use_grid_target(a.board)
        ap.CANVAS_W, ap.CANVAS_H = gt.board_size_mm(a.board)
        cols, rows = gt.board_spec(a.board)
        print(f"scene: templates/grid-{a.board}-full.pdf, rasterised - "
              f"{cols}x{rows} cells of {gt.CELL_MM} mm, {gt.MARKER_MM} mm "
              f"markers, {gt.LINE_MM} mm lines (blur {a.blur} px, ink {a.ink})")
        print(f"lens : k1={a.k1:+.3f} k2={a.k2:+.3f}"
              + (f" k3={TRUE_LENS[0]:+.4f} p1={TRUE_LENS[1]:+.5f} "
                 f"p2={TRUE_LENS[2]:+.5f} principal point "
                 f"{TRUE_LENS[3] * 2592:+.0f},{TRUE_LENS[4] * 2592:+.0f}px off centre"
                 if TRUE_LENS else "")
              + ("   (fitted from the frame)" if a.fit_lens else "   (given to the fit)"))
        res = [run_one_grid(s, k1=a.k1, k2=a.k2, dist=a.dist, blur=a.blur,
                            ink=a.ink, noise=a.noise, fit_lens=a.fit_lens)
               for s in range(a.n)]
        res = [r for r in res if r]
        if not res:
            return
        bad = sum(1 for r in res if r[3])
        seen = np.array([r[0] for r in res])
        whole = np.array([r[1] for r in res])
        snap = np.array([r[2]["rms_mm"] for r in res])
        print(f"\n{len(res)} scenes, {bad} with a mis-identified cell")
        print(f"  refined error over the cells in view, mm: "
              f"mean {seen.mean():.3f}  p95 {np.percentile(seen, 95):.3f}  "
              f"max {seen.max():.3f}")
        print(f"  ...extrapolated to the whole canvas, mm: "
              f"mean {whole.mean():.3f}  p95 {np.percentile(whole, 95):.3f}  "
              f"max {whole.max():.3f}")
        print(f"  snap residual, mm: mean {np.nanmean(snap):.3f}  "
              f"max {np.nanmax(snap):.3f}")
        return

    print(f"scene: the real sheet (square {ap.SQUARE_MM:.3f} mm, stroke "
          f"{ap.STROKE_MM:.3f} mm, block 192.07 x 127.53 mm)")
    print(f"model: {'LEGACY 64 mm grid' if a.legacy else 'the same, from calibr.svg'}"
          f"   lines measured at their {'darkest point' if a.ridge else 'inner flank'}"
          f"   (blur {a.blur} px, ink {a.ink})")
    if a.ridge:
        ap.measure_line_offsets = ridge_measure_line_offsets
    res = [run_one(s, k1=a.k1, k2=a.k2, verbose=True, legacy=a.legacy,
                   blur=a.blur, ink=a.ink, noise=a.noise) for s in range(a.n)]
    res = [r for r in res if r]
    if len(res) > 1:
        con = np.array([r[0] for r in res]); ref = np.array([r[1] for r in res])
        print(f"\n{len(res)} scenes - worst canvas-corner error, mm")
        print(f"  consensus  mean {con.mean():.3f}  p95 {np.percentile(con,95):.3f}"
              f"  max {con.max():.3f}")
        print(f"  refined    mean {ref.mean():.3f}  p95 {np.percentile(ref,95):.3f}"
              f"  max {ref.max():.3f}")


if __name__ == "__main__":
    main()
