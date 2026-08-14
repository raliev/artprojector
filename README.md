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
python artprojector.py list                 # list cameras and resolutions
python artprojector.py calibrate            # detect squares, compute homography
python artprojector.py run                  # live rectified (fronto-parallel) view
python artprojector.py overlay --ref img.jpg  # reference contours/image over the feed
python artprojector.py overlay --ref ref/nadya111/4/bw  # a folder: arrows switch reference
python artprojector.py gen-template         # generate a full-canvas target template
```

Common options: `--cam N` (camera index, default 0), `--px-per-mm 2.0` (output
scale), `--ref FILE|DIR` (reference for overlay; a folder is stepped through
with the arrow keys), `--canvas-w-in 12 --canvas-h-in 16`
(canvas size), `--adjust FILE.npz` (where overlay saves/loads its adjustment),
`--view-max 2000` (render size of the corrected overlay view),
`--fit consensus|corners` (how the homography is fitted), `--k1/--k2` (starting
lens distortion for `calibrate`; normally just press `a` there instead).

The camera must not move between `calibrate` and `run`/`overlay`.

### calibrate

Point the camera so the squares are visible; all six is best, since a single
row constrains the plane poorly. A green outline means the square was detected
and mapped into the 3x2 model; red means it is outside the grid. The `col_off`
and `row_off` trackbars shift the label mapping when only part of the block is
visible (leave them at 0 for a full 3x2 view). Press `c` to compute and save the
homography to `calibration.npz`; the mean reprojection error is printed.

**Consensus quad.** The magenta quadrilateral is the fit the perspective is
actually computed from. Corners of the individual squares are localized
unreliably, so instead of trusting any one of them, the four outer edges of the
whole block are fitted as straight lines through every corner that belongs to
them - the top edge from all three squares of the top row, the left edge from
both squares of the left column, and so on - and the quad is where those lines
intersect. The per-square errors average out, and the six squares correct each
other. Its sides should sit right on the outer edges of the block; if one is
visibly off, a square was mis-detected. `f` switches to the old per-corner
least-squares fit (`--fit corners` to start there). On synthetic targets with
1.5 px corner noise the consensus fit is ~10% more accurate on average and
~25% at the 95th percentile.

**Lens distortion.** A wide camera bows straight lines near the frame edges, so
squares out there are not really quadrilaterals and both the detection and the
fit suffer. Press `a` to auto-fit the radial coefficients: it solves for the
`k1`/`k2` that make the target's grid lines straightest, which takes a fraction
of a second. `1`/`2` and `3`/`4` adjust `k1`/`k2` by hand, `5`/`6` change the
step, `0` resets. The `line residual` in the HUD is the mean distance from the
corners to the line they should lie on - minimize it. Detection, the consensus
quad and the homography all run on the corrected frame, and the coefficients
are saved into `calibration.npz`, so `run` and `overlay` undistort every frame
automatically (both the raw and the corrected view). In a synthetic test with
`k1=-0.12` the correction cut the resulting canvas-corner error from ~13 px to
~0.5 px.

### run

Loads `calibration.npz` and shows the rectified canvas plane in real time.
`g` toggles a 50 mm grid, `c` recalibrates, `q` quits.

### overlay

Draws a reference over the live frame, projected into the canvas perspective.
The reference is stretched to the whole canvas, so only the part inside the
camera view is shown. Three render modes: contours (Canny edges), the original
image, and multiply - all blended with adjustable opacity.

Controls:

| Keys | Action |
|------|--------|
| `w` `a` `s` `d` | move the reference |
| `z` / `x` | scale down / up (both axes) |
| `[` / `]` | stretch X down / up |
| `-` / `=` | stretch Y down / up |
| `,` / `.` | rotate |
| `m` | cycle contours / image / multiply |
| `9` / `0` | opacity down / up |
| `1` `2` / `3` `4` | Canny low / high thresholds |
| left / right | previous / next reference (when `--ref` is a folder) |
| `o` | toggle overlay |
| `c` | contour color |
| `r` | raw (perspective) vs corrected proportions view |
| `p` `i` `h` `q` | save adjustment / reset adjustment / help / quit |

**Multiply mode** blends the reference multiplicatively instead of by
interpolation, so it only ever darkens: white areas of the reference leave the
camera image untouched and whatever is already drawn on the real canvas stays
visible through the dark ones. That is what you want for checking work against
a reference, where plain image mode would wash the canvas out.

**A folder of references.** If `--ref` points to a directory, the first image in
it (alphabetically) opens and the left/right arrows step to the previous/next
one. There is no wrap-around: the ends of the list are a stop. Everything else -
alignment, opacity, Canny thresholds, render mode, zoom and pan, raw/corrected
view - carries over untouched, so switching swaps only the picture. Dots in the
top-right corner show the position in the set (a `N/M` counter past 24 files).
This suits the layered output of `make_refs.py`: point `--ref` at
`ref/<stem>/<N>/bw` and flip through the tonal levels while painting.

Mouse: wheel or right-button drag zooms at the cursor, left-button drag pans,
Space resets the zoom. Zoom and pan work on top of any mode without disturbing
the overlay, which is useful for magnifying a fragment in the center of the
frame. Zoom is folded into the warp rather than applied to the finished image,
so magnifying resamples the sensor frame directly and does resolve more detail
(up to the sensor limit), and the contours are re-rasterized thin at every zoom
level. The adjustment (offset, scale, rotation, opacity) is saved to the
`--adjust` file and loaded on the next run.

The `r` view: in raw mode a square filmed at an angle appears as a trapezoid and
the reference follows the same perspective; in corrected mode the proportions
are undone, so the square looks square and the reference is square. The
corrected view is framed on the canvas plus a 20 mm margin - not on everything
the camera sees, which at a steep angle spans meters of wall and would leave
the canvas itself resolved at ~1 px/mm. Its scale is taken from the local scale
of the homography over the canvas (typically 3-6 px/mm), capped by `--view-max`
(default 2000 px on the longest side; raise it for a sharper view at a lower
frame rate).

### gen-template

`gen-template` renders the whole canvas with the six squares drawn in their real
positions:

```
python artprojector.py gen-template --out calib_template.png
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

See the "GEOMETRY CONFIG" block in `artprojector.py`:

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

- `artprojector.py` - the main tool (calibrate / run / overlay / gen-template / list).
- `capture.py` - grab one frame to a file (for debugging).
- `make_refs.py` - posterized reference variations.
- `calibr-1216.png` - printable calibration target.
- `calibration.npz` - written by `calibrate`: the homography `H` (mm -> undistorted
  image px), the lens distortion `k1`/`k2`, and the canvas size it was made for.
