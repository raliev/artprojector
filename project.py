#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project.py
==========

Throw a reference image onto the canvas, pre-distorted so that it looks right
ON THE CANVAS rather than on the projector's own frame.

`projcalib.py` measured where every projector pixel lands, in canvas
millimetres. This is the other end of that: a picture is placed on the canvas
in millimetres, pushed backwards through the same mapping, and what the
projector actually displays is a keystoned, rotated, oversized shape that
nobody would recognise - which lands on the canvas as the picture, square to
its edges and the right size, whatever angle the projector is sitting at.

Nothing is projected outside the canvas rectangle: the frame, the easel and
the wall stay dark, which is both politer to look at and a permanent visual
check that the mapping is still true (the lit area IS the canvas, so if it
creeps off the edge, something has moved).

Usage:
  python project.py --ref ref/nadya-1/4/bw --display 1
  python project.py --ref check.png --display 1 --fit contain

  --ref is a file or a folder; a folder opens at the first image and LEFT /
  RIGHT step through the rest, exactly as in `artprojector.py overlay`.

Two windows: the picture, fullscreen on the projector, and a small control
window with the readout, on another monitor - the readout must not be painted
onto the canvas.

Keys (the control window has the focus), all as in overlay:
  LEFT/RIGHT - previous / next reference in the folder
  a/d w/s    - move the picture 2 mm on the canvas
  z/x        - scale about the canvas centre;  [ ] and - = scale X and Y alone
  , .        - rotate half a degree
  i          - back to no adjustment;  p - save the adjustment
  9/0        - dimmer / brighter;  I - invert (white lines on black)
  b          - outline the canvas;  k - blackout;  f - stretch <-> contain
  q          - quit

The adjustment saved here is the PICTURE's placement - where this reference
sits on the canvas - and is kept in its own file (project_adjust.npz),
separate from the projector calibration and from overlay's. Which one is
wrong, when something does not line up, is not a question to have to guess
about: if the lit rectangle no longer matches the canvas edges, the
calibration has gone stale (re-run projcalib); if it matches and the drawing
sits badly inside it, that is this adjustment.
"""

import argparse

import numpy as np
import cv2

import artprojector as ap
import projcalib as pc

ADJUST_FILE = "project_adjust.npz"


def image_to_mm(iw, ih, mode="stretch"):
    """Reference pixel -> canvas mm.

    'stretch' fills the canvas exactly, as overlay does - the references made
    by make_refs.py are cut from a canvas-shaped picture, so their proportions
    are already the canvas's. 'contain' keeps the picture's own aspect ratio
    and centres it, which is what an arbitrary photograph wants."""
    W, H = ap.CANVAS_W, ap.CANVAS_H
    if mode == "contain":
        s = min(W / iw, H / ih)
        ox, oy = -W + (W - iw * s) / 2.0, -H + (H - ih * s) / 2.0
        return np.array([[s, 0.0, ox], [0.0, s, oy], [0.0, 0.0, 1.0]], np.float64)
    return np.array([[W / iw, 0.0, -W], [0.0, H / ih, -H], [0.0, 0.0, 1.0]],
                    np.float64)


def render(img, H_mp, adj, spec_wh, fit_mode="stretch", gain=1.0,
           invert=False, outline=False):
    """The frame to hand the projector."""
    pw, ph = spec_wh
    ih, iw = img.shape[:2]
    M = np.asarray(H_mp, np.float64) @ pc.adjust_matrix(adj) @ image_to_mm(iw, ih, fit_mode)
    src = cv2.bitwise_not(img) if invert else img
    out = cv2.warpPerspective(src, M, (pw, ph), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    # Clip to the canvas: the picture may have been moved or scaled past its
    # edge, and light on the frame or the wall is both distracting and a false
    # reading of where the canvas is.
    quad = np.round(pc.xform(H_mp, pc.canvas_quad_mm())).astype(np.int32)
    mask = np.zeros((ph, pw), np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    out = cv2.bitwise_and(out, out, mask=mask)
    if gain != 1.0:
        out = cv2.convertScaleAbs(out, alpha=float(gain), beta=0)
    if outline:
        cv2.polylines(out, [quad], True, (0, 160, 255), 2, cv2.LINE_AA)
    return out


def hud_image(lines, w=980, lh=30):
    img = np.zeros((lh * len(lines) + 24, w, 3), np.uint8)
    y = 30
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 255) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)
        y += lh
    return img


def load_adjust(path):
    try:
        d = np.load(path)
    except FileNotFoundError:
        return list(pc.ZERO_ADJ)
    return [float(d["dx"]), float(d["dy"]), float(d["sx"]), float(d["sy"]),
            float(d["theta"])]


def save_adjust(path, adj, gain):
    dx, dy, sx, sy, th = adj
    np.savez(path, dx=dx, dy=dy, sx=sx, sy=sy, theta=th, gain=gain)


def main():
    p = argparse.ArgumentParser(
        description="Project a reference image onto the canvas through a "
                    "projector calibration")
    p.add_argument("--ref", default="check.png",
                   help="image file, or a folder of them (LEFT/RIGHT step)")
    p.add_argument("--calib", default=pc.PROJ_FILE,
                   help=f"projector calibration (default {pc.PROJ_FILE}), "
                        f"written by projcalib.py")
    p.add_argument("--adjust", default=ADJUST_FILE,
                   help=f"file for this picture's placement (default "
                        f"{ADJUST_FILE})")
    p.add_argument("--display", default=None,
                   help="monitor the projector is (index or name)")
    p.add_argument("--ctl-display", default=None,
                   help="monitor for the control window; without it, the first "
                        "one that is not the projector")
    p.add_argument("--fit", choices=["stretch", "contain"], default="stretch",
                   help="stretch the picture to the canvas (default) or keep "
                        "its aspect ratio and centre it")
    p.add_argument("--gain", type=float, default=1.0,
                   help="brightness multiplier (default 1.0); 9/0 change it")
    p.add_argument("--invert", action="store_true",
                   help="project the negative - white lines on black is far "
                        "easier to trace by, and lights the canvas far less")
    p.add_argument("--canvas-w-in", type=float, default=None)
    p.add_argument("--canvas-h-in", type=float, default=None)
    p.add_argument("--no-keep-awake", action="store_true")
    args = p.parse_args()

    fit = pc.load_projector(args.calib)
    if fit is None:
        print(f"[project] no {args.calib} - calibrate the projector first:\n"
              f"          python projcalib.py --display 1 --board 12x16")
        return
    # The canvas the calibration was made for, unless told otherwise: what
    # "the size of the canvas" means has to be the same here as it was there,
    # or the picture is placed on a different rectangle than the one measured.
    ap.CANVAS_W, ap.CANVAS_H = fit["canvas_w"], fit["canvas_h"]
    if args.canvas_w_in:
        ap.CANVAS_W = args.canvas_w_in * ap.MM_PER_IN
    if args.canvas_h_in:
        ap.CANVAS_H = args.canvas_h_in * ap.MM_PER_IN
    print(f"[project] {args.calib}: fitted {fit['when']}, "
          f"{fit['rms_mm']:.2f} mm rms, canvas "
          f"{fit['canvas_w']:.0f}x{fit['canvas_h']:.0f} mm, projector frame "
          f"{fit['proj_w']}x{fit['proj_h']}")
    if pc.adjusted(fit):
        print(f"[project] the calibration carries a hand adjustment and it is "
              f"applied to everything below: {pc.adjust_text(fit)}")

    refs = ap.list_reference_images(args.ref)
    if not refs:
        print(f"[project] no images in {args.ref}"); return

    mons = ap.list_displays()
    if mons:
        print("[project] monitors: " + ", ".join(
            f"[{i}] {n} {w}x{h}+{x}+{y}" for i, (n, x, y, w, h) in enumerate(mons)))
    pw, ph, pname, proj_i = pc.projector_size(mons, args.display)
    if (pw, ph) != (fit["proj_w"], fit["proj_h"]):
        print(f"[project] !! the calibration was made on a "
              f"{fit['proj_w']}x{fit['proj_h']} frame and this monitor is "
              f"{pw}x{ph}. The mapping is in pixels: recalibrate, or the "
              f"picture lands wrong by that ratio.")
    ctl_i = pc.camera_display(mons, proj_i, args.ctl_display)

    adj = load_adjust(args.adjust)
    if tuple(adj) != pc.ZERO_ADJ:
        print(f"[project] a saved placement was loaded from {args.adjust} and "
              f"is applied to EVERY reference: dx={adj[0]:+.1f}mm "
              f"dy={adj[1]:+.1f}mm sx={adj[2]:.3f} sy={adj[3]:.3f} "
              f"rot={adj[4]:+.1f}deg  ('i' zeroes it)")

    img, ref_i = None, 0

    def load_ref(i):
        nonlocal img, ref_i
        got = cv2.imread(refs[i])
        if got is None:
            print(f"[project] could not read {refs[i]}")
            return False
        img, ref_i = got, i
        print(f"[project] reference {i + 1}/{len(refs)}: {refs[i]} "
              f"({got.shape[1]}x{got.shape[0]})")
        return True

    if not load_ref(0):
        return

    awake = None if args.no_keep_awake else ap.KeepAwake("project").start()
    pwin = ap.make_window("projection", fullscreen=True,
                          display=proj_i if proj_i is not None else args.display)
    cwin = pc.camera_window("projection control", mons, ctl_i, frac=0.45)
    gain, invert, outline, blackout = args.gain, args.invert, False, False
    fit_mode, dirty = args.fit, True
    try:
        while True:
            if dirty:
                shown = (np.zeros((ph, pw, 3), np.uint8) if blackout else
                         render(img, fit["H_mp"], adj, (pw, ph), fit_mode,
                                gain, invert, outline))
                cv2.imshow(pwin, shown)
                ppm = pc.local_scale(fit["H_mp"], -ap.CANVAS_W / 2,
                                     -ap.CANVAS_H / 2)
                dx, dy, sx, sy, th = adj
                cv2.imshow(cwin, hud_image([
                    f"{refs[ref_i]}   [{ref_i + 1}/{len(refs)}]"
                    f"   {img.shape[1]}x{img.shape[0]} px",
                    f"canvas {ap.CANVAS_W:.0f} x {ap.CANVAS_H:.0f} mm at "
                    f"{ppm:.2f} projector px/mm ({ppm * 25.4:.0f} dpi)"
                    f"   fit={fit_mode}",
                    f"placement  dx={dx:+.1f}mm dy={dy:+.1f}mm  sx={sx:.3f} "
                    f"sy={sy:.3f}  rot={th:+.1f}deg",
                    f"gain={gain:.2f}  invert={'on' if invert else 'off'}  "
                    f"outline={'on' if outline else 'off'}"
                    f"{'  BLACKOUT' if blackout else ''}",
                    "LEFT/RIGHT ref   a/d w/s move   z/x scale   [ ] - = axes",
                    ", . rotate   i reset   p save   9/0 gain   I invert",
                    "b outline   k blackout   f stretch/contain   q quit",
                ]))
                dirty = False

            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            if key in ap.KEY_PREV or key in ap.KEY_NEXT:
                nxt = ref_i + (1 if key in ap.KEY_NEXT else -1)
                if 0 <= nxt < len(refs) and load_ref(nxt):
                    dirty = True
                continue
            k = key & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('a'):
                adj[0] -= pc.MOVE_MM
            elif k == ord('d'):
                adj[0] += pc.MOVE_MM
            elif k == ord('w'):
                adj[1] -= pc.MOVE_MM
            elif k == ord('s'):
                adj[1] += pc.MOVE_MM
            elif k == ord('x'):
                adj[2] *= pc.SCALE_STEP; adj[3] *= pc.SCALE_STEP
            elif k == ord('z'):
                adj[2] /= pc.SCALE_STEP; adj[3] /= pc.SCALE_STEP
            elif k == ord(']'):
                adj[2] *= pc.SCALE_STEP
            elif k == ord('['):
                adj[2] /= pc.SCALE_STEP
            elif k == ord('='):
                adj[3] *= pc.SCALE_STEP
            elif k == ord('-'):
                adj[3] /= pc.SCALE_STEP
            elif k == ord('.'):
                adj[4] += pc.ROT_DEG
            elif k == ord(','):
                adj[4] -= pc.ROT_DEG
            elif k == ord('i'):
                adj = list(pc.ZERO_ADJ)
            elif k == ord('p'):
                save_adjust(args.adjust, adj, gain)
                print(f"[project] placement saved to {args.adjust}")
            elif k == ord('9'):
                gain = max(0.05, gain - 0.05)
            elif k == ord('0'):
                gain = min(4.0, gain + 0.05)
            elif k == ord('I'):
                invert = not invert
            elif k == ord('b'):
                outline = not outline
            elif k == ord('k'):
                blackout = not blackout
            elif k == ord('f'):
                fit_mode = "contain" if fit_mode == "stretch" else "stretch"
            else:
                continue
            dirty = True
    finally:
        cv2.destroyAllWindows()
        if awake is not None:
            awake.stop()


if __name__ == "__main__":
    main()
