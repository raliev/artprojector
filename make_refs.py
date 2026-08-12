#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_refs.py - build posterized reference variations for drawing.

For an input image and a number of posterize levels N it creates:

  ref/<stem>/<N>/posterized.jpg           posterized image
  ref/<stem>/<N>/posterized-contours.jpg  contours (posterize region boundaries)
  ref/<stem>/<N>/color/<stem>_<i>_<hex>.jpg   N b/w variations (color posterize)
  ref/<stem>/<N>/bw/<stem>_<i>_g<val>.jpg     N b/w variations (desaturate, then levels)

Each variation is a black-and-white image: the pixels that fall into a given
posterize level/color are filled BLACK on a white background (a tone stencil).
Index i runs from dark to light.

  color - posterize the color image (k-means into N colors), mask per color.
  bw    - desaturate first, then split into N tonal levels.

Usage:
  conda activate p3124
  python make_refs.py nadya111.jpg                    # 4 levels, both modes
  python make_refs.py nadya111.jpg --levels 3 4 6
  python make_refs.py nadya111.jpg --levels 5 --modes bw
"""
import argparse
import os
import numpy as np
import cv2


def color_posterize(img_bgr, n):
    """k-means into n colors. Returns (labels HxW int, centers_bgr (n,3),
    quantized BGR). Labels are ordered from dark to light."""
    h, w = img_bgr.shape[:2]
    Z = img_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = centers  # (n,3) BGR float
    lum = centers @ np.array([0.114, 0.587, 0.299])   # luminance (BGR)
    order = np.argsort(lum)
    rank = np.zeros(n, int)
    rank[order] = np.arange(n)
    labels = rank[labels]
    centers = centers[order].astype(np.uint8)
    quant = centers[labels].reshape(h, w, 3)
    return labels.reshape(h, w), centers, quant


def bw_posterize(img_bgr, n):
    """Desaturate and split into n equal tonal levels.
    Returns (labels HxW int, gray_values (n,), posterized_gray HxW)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    labels = np.clip((gray.astype(np.int32) * n) // 256, 0, n - 1)
    if n > 1:
        vals = np.round(np.arange(n) * 255.0 / (n - 1)).astype(np.uint8)
    else:
        vals = np.array([128], np.uint8)
    post = vals[labels]
    return labels, vals, post


def contours_from_labels(labels):
    """Boundaries between posterize regions -> b/w (black lines on white)."""
    h, w = labels.shape
    edge = np.zeros((h, w), bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    out = np.full((h, w), 255, np.uint8)
    out[edge] = 0
    return out


def save_layer_mask(path, labels, i):
    """Mask of level i: level pixels -> black on white."""
    out = np.full(labels.shape, 255, np.uint8)
    out[labels == i] = 0
    cv2.imwrite(path, out)


def process(img_path, levels, modes, out_root):
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Could not read: {img_path}")
    stem = os.path.splitext(os.path.basename(img_path))[0]

    for n in levels:
        base = os.path.join(out_root, stem, str(n))
        os.makedirs(base, exist_ok=True)

        c_labels, centers, quant = color_posterize(img, n)
        bw_labels, gray_vals, post_bw = bw_posterize(img, n)

        # shared posterized.jpg + contours: color if it is in modes, else b/w
        if "color" in modes:
            canon_img, canon_labels = quant, c_labels
        else:
            canon_img, canon_labels = post_bw, bw_labels
        cv2.imwrite(os.path.join(base, "posterized.jpg"), canon_img)
        cv2.imwrite(os.path.join(base, "posterized-contours.jpg"),
                    contours_from_labels(canon_labels))

        if "color" in modes:
            d = os.path.join(base, "color")
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                b, g, r = (int(v) for v in centers[i])
                suffix = f"{i}_{r:02x}{g:02x}{b:02x}"
                save_layer_mask(os.path.join(d, f"{stem}_{suffix}.jpg"),
                                c_labels, i)

        if "bw" in modes:
            d = os.path.join(base, "bw")
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                suffix = f"{i}_g{int(gray_vals[i])}"
                save_layer_mask(os.path.join(d, f"{stem}_{suffix}.jpg"),
                                bw_labels, i)

        print(f"[{stem}] N={n}: posterized + contours + "
              f"{'color ' if 'color' in modes else ''}{'bw' if 'bw' in modes else ''}"
              f" variations -> {base}")


def main():
    ap = argparse.ArgumentParser(description="Posterized reference variations")
    ap.add_argument("image", help="input image")
    ap.add_argument("--levels", type=int, nargs="+", default=[4],
                    help="number of posterize levels (one or more)")
    ap.add_argument("--modes", nargs="+", default=["color", "bw"],
                    choices=["color", "bw"], help="which branches to build")
    ap.add_argument("--out", default="ref", help="output root folder")
    args = ap.parse_args()
    process(args.image, args.levels, args.modes, args.out)


if __name__ == "__main__":
    main()
