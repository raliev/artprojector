#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture.py - grab a single frame from the USB camera to a file (detector debugging).

  conda activate p3124
  python capture.py                     # -> frame.png from camera #0, working resolution
  python capture.py --cam 1 --out shot.png
  python capture.py --warmup 15         # more warm-up frames (auto-exposure)
"""
import argparse
import cv2

REQ_WIDTH = 1920      # working mode of the GXI-IMX179 via OpenCV
REQ_HEIGHT = 1080
USE_MJPG = True


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default="frame.png")
    ap.add_argument("--warmup", type=int, default=10,
                    help="how many frames to skip before the snapshot")
    args = ap.parse_args()

    frame = None
    # the camera sometimes does not stream on the first try -> reopen
    for attempt in range(6):
        cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            time.sleep(0.4); continue
        if USE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        good = 0
        deadline = time.time() + 4.0
        while time.time() < deadline:
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                good += 1
                if good >= max(1, args.warmup):
                    break
            else:
                time.sleep(0.05)
        cap.release()
        if frame is not None:
            print(f"camera #{args.cam}: {w}x{h} (attempt {attempt + 1})")
            break
        time.sleep(0.4)

    if frame is None:
        raise SystemExit("No frame captured (check camera access and the --cam index, "
                         "see python canvas_rectify.py list)")
    cv2.imwrite(args.out, frame)
    print(f"saved: {args.out}  ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()
