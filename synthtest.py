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
"""
import argparse
import numpy as np
import cv2

import artprojector as ap


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
    a = p.parse_args()
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
