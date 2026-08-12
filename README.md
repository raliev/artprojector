# artprojector

A small OpenCV tool for painters. A USB camera looks at a canvas on an easel;
the tool figures out the perspective of the canvas from a printed calibration
target and can rectify the view or overlay a reference image (as contours or as
a semi-transparent picture) aligned to the canvas plane.

The camera may sees only part of the canvas. A sheet with a grid of six
64 mm squares is placed flush with the right and bottom edges of the canvas to calibrate.
From the squares the tool computes a homography between the image and the
canvas plane (in millimeters), which is enough to rectify what the camera sees
and to project a reference onto it.

## Requirements

- Python 3, `numpy`, `opencv-python` (developed with OpenCV 4.13)
- A USB camera. Development used a GXI-IMX179 board camera on macOS.

## Calibration target

Print `calibr-1216.png` and place it in the bottom-right corner of the canvas,
flush with the right and bottom edges. The default geometry assumes:

- Canvas 12x16 inches (configurable, see below).
- Six 64 mm squares, 3 columns by 2 rows.
- Columns touch horizontally (3 x 64 = 192 mm), rows have a ~0.6 mm gap.
- The right edge of the rightmost square is 14 mm from the canvas right edge;
  the bottom edge of the bottom square is 145 mm from the canvas bottom edge.

The world origin is the bottom-right corner of the canvas, which coincides with
the sheet corner. Because of that the square coordinates do not depend on the
canvas size, so changing the canvas size does not require recalibration.

## Usage

```
python canvas_rectify.py list                 # list cameras and resolutions
python canvas_rectify.py calibrate            # detect squares, compute homography
python canvas_rectify.py run                  # live rectified (fronto-parallel) view
python canvas_rectify.py overlay --ref img.jpg  # reference contours/image over the feed
python canvas_rectify.py gen-template         # generate a full-canvas target template
```

Common options: `--cam N` (camera index, default 0), `--px-per-mm 2.0` (output
scale), `--ref FILE` (reference for overlay), `--canvas-w-in 12 --canvas-h-in 16`
(canvas size), `--adjust FILE.npz` (where overlay saves/loads its adjustment).

The camera must not move between `calibrate` and `run`/`overlay`.

### calibrate

Point the camera so the squares are visible; all six is best, since a single
row constrains the plane poorly. A green outline means the square was detected
and mapped into the 3x2 model; red means it is outside the grid. The `col_off`
and `row_off` trackbars shift the label mapping when only part of the block is
visible (leave them at 0 for a full 3x2 view). Press `c` to compute and save the
homography to `calibration.npz`; the mean reprojection error is printed.

### run

Loads `calibration.npz` and shows the rectified canvas plane in real time.
`g` toggles a 50 mm grid, `c` recalibrates, `q` quits.

### overlay

Draws a reference over the live frame, projected into the canvas perspective.
The reference is stretched to the whole canvas, so only the part inside the
camera view is shown. Two render modes: contours (Canny edges) and the original
image blended with adjustable opacity.

Controls:

| Keys | Action |
|------|--------|
| `w` `a` `s` `d` | move the reference |
| `z` / `x` | scale down / up (both axes) |
| `[` / `]` | stretch X down / up |
| `-` / `=` | stretch Y down / up |
| `,` / `.` | rotate |
| `m` | switch contours / original image |
| `9` / `0` | opacity down / up |
| `1` `2` / `3` `4` | Canny low / high thresholds |
| `o` | toggle overlay |
| `c` | contour color |
| `r` | raw (perspective) vs corrected proportions view |
| `p` `i` `h` `q` | save adjustment / reset adjustment / help / quit |

Mouse: wheel or right-button drag zooms at the cursor, left-button drag pans,
Space resets the zoom. Zoom and pan work on top of any mode without disturbing
the overlay, which is useful for magnifying a fragment in the center of the
frame. The adjustment (offset, scale, rotation, opacity) is saved to the
`--adjust` file and loaded on the next run.

The `r` view: in raw mode a square filmed at an angle appears as a trapezoid and
the reference follows the same perspective; in corrected mode the proportions
are undone, so the square looks square and the reference is square. The
corrected view is cropped to what the camera sees.

### gen-template

`gen-template` renders the whole canvas with the six squares drawn in their real
positions:

```
python canvas_rectify.py gen-template --out calib_template.png
```

Overlaying this template (`overlay --ref calib_template.png`) lands the contours
on the real squares, which is a quick way to confirm the calibration and the
adjustment. It is also a convenient base to draw a reference on with the correct
proportions.

## Posterized reference variations - make_refs.py

Builds per-tone black-and-white stencils and contour drawings from an image, for
tracing or overlaying:

```
python make_refs.py nadya111.jpg --levels 4
python make_refs.py nadya111.jpg --levels 3 4 6 --modes bw
```

Output layout:

```
ref/<stem>/<N>/posterized.jpg            posterized image
ref/<stem>/<N>/posterized-contours.jpg   contours (posterize region boundaries)
ref/<stem>/<N>/color/<stem>_<i>_<hex>.jpg   N masks per color (k-means)
ref/<stem>/<N>/bw/<stem>_<i>_g<val>.jpg     N masks per tonal level
```

Each mask fills the pixels of one level/color in black on white; index `i` runs
from dark to light. `color` posterizes by color; `bw` desaturates first and then
splits into levels. Any of these files can be passed to `overlay --ref ...`.

## Geometry configuration

See the "GEOMETRY CONFIG" block in `canvas_rectify.py`:

- Canvas size (`CANVAS_W_IN`, `CANVAS_H_IN`) or the `--canvas-w-in/--canvas-h-in`
  flags.
- Square side (`SQUARE_MM`), grid (`COLS`, `ROWS`), gaps (`GAP_X_MM`, `GAP_Y_MM`).
- Anchoring (`RIGHT_MARGIN_MM`, `BOTTOM_MARGIN_MM`).

## Hardware notes

- Through OpenCV/AVFoundation the GXI-IMX179 reliably streams only 1920x1080
  MJPG; 2048/2592/3264 and 1280/640 return zero frames. That is the working
  maximum here.
- The camera sometimes does not stream on the first open, so opening and single
  captures retry with reopen.
- Make sure no other application (FaceTime and similar) is holding the camera.

## Files

- `canvas_rectify.py` - the main tool (calibrate / run / overlay / gen-template / list).
- `capture.py` - grab one frame to a file (for debugging).
- `make_refs.py` - posterized reference variations.
- `calibr-1216.png` - printable calibration target.
