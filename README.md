# artprojector

A small OpenCV tool for painters. A USB camera looks at a canvas on an easel;
the tool figures out the perspective of the canvas from a printed calibration
target and can rectify the view or overlay a reference image (as contours or as
a semi-transparent picture) aligned to the canvas plane.

The camera may sees only part of the canvas. From the printed target the tool
computes a homography between the image and the canvas plane (in millimeters),
which is enough to rectify what the camera sees and to project a reference onto
it.

There are two printed targets, chosen with `--target`:

| | `--target squares` (default) | `--target grid` |
|---|---|---|
| what it is | six 63 mm squares on one A4 sheet (`calibr.svg`) | a 1-inch grid the size of the canvas, ArUco marker in every cell (`templates/*.pdf`) |
| where it goes | bottom-right corner of the canvas | over the whole canvas |
| what must be in frame | the whole sheet | any two or three cells |
| which square is which | set by hand, `col_off`/`row_off` | the marker says |
| how close can the camera get | far enough to see the sheet | as close as you like |

The old target is untouched and stays the default; a calibration file records
which target it was made with, so `run` and `overlay` pick the right one on
their own.

## Requirements

- Python 3, `numpy`, `opencv-python` (developed with OpenCV 4.13)
- A USB camera. Development used a GXI-IMX179 board camera on macOS.

## Calibration target 2 - the 1-inch ArUco grid (`--target grid`)

```
python gridtarget.py                  # writes templates/*.pdf
python gridtarget.py --check          # ...and verifies the tiling (needs pdftoppm)
python artprojector.py calibrate --target grid --board 16x20
```

`templates/` gets, for each of 12x16" and 16x20":

- `grid-<size>-full.pdf` - one page, exactly that many inches, for a large printer;
- `grid-<size>-a4.pdf`, `grid-<size>-letter.pdf` - the same board cut into
  ordinary sheets (4 sheets for 12x16, 6 for 16x20, on either paper).

The board is a 1-inch grid the size of the canvas, with a 16 mm ArUco marker
(`DICT_4X4_1000`) centred in every cell. Print at 100%, assemble, and mount it -
see below. `templates/README.md` (written by the generator) has the assembly
instructions; every tile also carries a 100 mm ruler to catch a print that was
scaled to fit the page.

### Mounting: the one thing the target cannot tell you

The reference is the printed **border line**, never the paper edge. A printer
cannot print to the edge, so the bottom-right sheet keeps ~5 mm of blank paper
outside the border (`PRINT_MARGIN_MM`), and a large printer adds its own margin
to the full-size PDF. Tape the sheet down by its paper edge and the board sits a
centimetre up and to the left of where the software thinks it is.

That error is invisible to everything else. The perspective, the scale and the
cell identities are all measured off the ink and are perfectly correct; only the
link between the board and the *canvas* is wrong, and nothing in the frame marks
the canvas edge. Snap residual, reprojection error and the marker count all stay
excellent while every millimetre reported is out by a centimetre.

Two ways to make it true:

- **trim** along the printed border on the right and bottom and tape that edge
  flush to the canvas. Nothing to measure, nothing to configure. Preferred.
- or **declare it**: measure the gap from the canvas right edge to the board's
  right border, and from the canvas bottom edge to the board's bottom border,
  and pass them as negative millimetres (`--grid-anchor-x <-your measurement>
  --grid-anchor-y <-your measurement>`). Do not copy the numbers out of any
  example, including this one - they are a property of your easel, and nothing
  in the software can notice that they are wrong.

  They are saved into `calibration.npz` and **picked up automatically on later
  runs**, which is convenient until you trim or remount the board, at which
  point a stale offset survives a recalibration untouched. `calibrate` shouts
  about an inherited offset for that reason; pass `--grid-anchor-x 0
  --grid-anchor-y 0` explicitly once the board is flush. They are part of the
  model, so changing them makes the old calibration stale - `calibrate` again.

**Checking it.** Use `gen-template` output as the overlay reference, not a PDF
converted to an image:

```
python artprojector.py gen-template --target grid --board 12x16 --out check.png
python artprojector.py overlay --target grid --board 12x16 --ref check.png --adjust /tmp/fresh.npz
```

`check.png` is the whole *canvas* with the board drawn where the model says the
board is, mounting offset included, so its lines land on the real ones. A
straight `convert` of `templates/grid-12x16-full.pdf` is the board and nothing
else, and `overlay` stretches any reference across the whole canvas - so with a
non-zero mounting offset it is guaranteed to miss, by exactly that offset. That
is not a calibration error, and chasing it as one wastes an evening.

A leftover **`overlay_adjust.npz`** does the same kind of damage from the other
side: `overlay` loads `dx/dy/sx/sy/theta` at start-up and applies them to every
reference, so an 8 mm `dy` tuned for an older sheet silently shifts the new one
by 8 mm. Check it (`python -c "import numpy;d=numpy.load('overlay_adjust.npz');
print({k:float(d[k]) for k in d.files})"`) or point `--adjust` at a fresh file.

### Why markers and not counted dots or QR

Counted dots were the first idea and the arithmetic kills them: a 16x20" board
has 320 cells, and 320 countable dots do not fit in a 25.4 mm cell alongside the
4.7 mm of blank paper the line snap needs. Splitting the count into row and
column groups still needs up to 20 dots a group, at which point a dot is about a
millimetre and a misread is one glance away.

A QR code fits but is the wrong tool: a Version-1 symbol is 21x21 modules, so in
an 18 mm cell the module is 0.86 mm, and QR decoding is all-or-nothing - at 70
degrees off-axis with a little motion blur it mostly does not decode.

ArUco is what is built for this. `DICT_4X4_1000` is 6x6 modules including the
border, so a 16 mm marker has a 2.7 mm module - three times the QR module - the
detector is designed around the projective distortion of a flat marker seen at
an angle, and it returns four sub-pixel corners *with* an identity, so one
marker already pins a homography. It is also in OpenCV already.

### How a two-cell view can be enough

The two features do different jobs, and it is not "markers instead of lines":

- the **markers** are read for identity and for a first homography. Four corners
  of one 16 mm marker determine one - which is what makes a close-up work at all
  - but 16 mm is a poor lever for a 400 mm canvas, so it is a starting point.
- the **grid lines** are what the answer is measured on: `refine_homography()`
  snaps the fit onto the printed ink exactly as it does for the squares, and the
  lines are the longest and best-localised features on the sheet.

That is why the marker is 16 mm inside a 25.4 mm cell. The 4.7 mm ring of blank
paper keeps it out of the ~2 mm the line snap searches through; ink that close
to a line would be measured *as* the line.

**About the lines bending.** At a short distance and a wide angle the printed
grid really does curve across the frame, and on two or three cells the curve is
barely visible - which is exactly why it has to be dealt with rather than
ignored. It is lens distortion, not perspective (perspective keeps straight
lines straight), so it belongs in `k1`/`k2` and not in `H`: press `a` to fit
them to the marker corners, `r` to let the line snap keep refining them. Fitting
a homography to a bent frame without that buries the bend in `H`, where it comes
back as millimetres somewhere else on the canvas.

### What it is worth

`synthtest.py --target grid` renders the real PDF out of `templates/`, warps it
through a known homography with known distortion, and runs the whole pipeline
over it, so a disagreement between the generator and the detector cannot cancel
out. Over 30 random viewpoints from 90 to 700 mm (12 of them close-ups with 2-4
cells in frame), and 50 scenes in total:

- cells identified correctly: **all of them**, in every scene;
- error where the camera is looking: **0.04 mm mean, 0.08 mm worst** (0.07 /
  0.12 mm for the 90 mm close-ups);
- error extrapolated to the *whole* canvas from a close-up: 4 mm mean, 14 mm
  worst.

That last number is not a defect to tune away, it is what a close-up can know: a
fit measured over 90 mm and used over 640 mm is extrapolated sevenfold. It
matters much less than it sounds, because the overlay only ever draws on the
part of the canvas that is in the frame - but if you want a number to trust
across the whole canvas, back the camera off so the cells spread across the
frame. The calibrate window prints the span and warns when it is small.

### When it recognises nothing

`markers=0` in the calibrate window is three different failures wearing the same
face, so there is a mode that tells them apart:

```
python artprojector.py grid-probe --board 16x20 --cam 2      # grab a frame and analyse it
python artprojector.py grid-probe --board 16x20 --frame shot.png
```

It reports focus and exposure, how many four-sided candidates were found, how
many of those decoded, the marker size in pixels, and whether the ids belong to
the board you named - and writes an annotated image (green = decoded and on the
board, orange = decoded but on the *other* board, red = candidate that would not
decode). The usual answers, in order of likelihood:

- **decoded, but "not on the 16x20 board"** - `--board` names the wrong sheet.
  The two boards use disjoint id ranges precisely so this is visible rather than
  silently wrong.
- **candidates found, none decoded** - too few pixels per module. A 4x4 marker
  is 6 modules across and wants ~30 px of side at the very least; get closer,
  focus, or raise the capture resolution.
- **no candidates at all** - the grid is not in view, not printed, or not lit.

You do not need the whole board glued together to test recognition: print one
sheet of the tiling and point the camera at it. Every marker names itself, so
detection works immediately - only the *geometry* waits until the board is
assembled and taped to the canvas in the right place.

### Tiling and assembly

The sheets are laid out from the **bottom-right corner of the board outwards**,
which is the corner everything else in this project measures from. Consequences,
by design:

- the bottom-right sheet is used exactly as it comes out of the printer - no
  trimming at all;
- every other sheet is trimmed along a printed dashed line on its **right and/or
  bottom** edge and glued **on top** of the sheet it laps over;
- so all the cutting and gluing happens on the left and the top, and each sheet
  covers its neighbour's unprintable margin.

Each sheet laps 14 mm over its neighbour and stays 5 mm clear of the paper edge
(`--overlap`, `--margin`). Sheet labels, instructions and the ruler are printed
only where they cannot show in the finished board: in the strip the next sheet
covers, or on paper outside the board.

## Calibration target 1 - the six squares (`--target squares`, default)

Print `calibr.svg` at 100% ("actual size", NOT fit-to-page) and place the sheet
in the bottom-right corner of the canvas, flush with the right and bottom
edges. The geometry is taken straight out of that SVG (see `SHEET_RECTS_MM`):

- Canvas 12x16 inches (configurable, see below).
- Six squares of 62.978 mm, 3 columns by 2 rows, printed with a 0.812 mm line.
- They are *not* on an exact grid - the file was drawn by hand, so the column
  pitch is 64.407 and 64.681 mm, the rows are 64.333 mm apart, and the tops of
  a row differ by up to 0.22 mm. The real coordinates are used as they are.
- The right edge of the rightmost square is 14 mm from the canvas right edge;
  the bottom edge of the bottom square is 145 mm from the canvas bottom edge.

All sheet numbers are the **centreline** of the printed line, which is what the
detector measures once it averages the outer and the inner contour of a frame.

The world origin is the bottom-right corner of the canvas, which coincides with
the sheet corner. Because of that the square coordinates do not depend on the
canvas size, so changing the canvas size does not require recalibration.

The earlier description of this sheet - "64 mm squares, touching, 0.6 mm
between the rows" - was measured off a photo and was wrong by about a
millimetre over the block (0.6 mm is the white gap *between the strokes*, not
between the rectangles). Since a homography fitted to four corners always maps
the model block onto the imaged block, that error did not show up as a
reprojection error; it showed up as ~2 mm of drift across the sheet and up to
5 mm at the far corner of the canvas. If you print a different target, measure
it from the file, not from a photograph.

## Usage

```
python artprojector.py list                 # list cameras and resolutions
python artprojector.py calibrate            # detect squares, compute homography
python artprojector.py calibrate --target grid --board 16x20   # the ArUco grid instead
python artprojector.py run                  # live rectified (fronto-parallel) view
python artprojector.py overlay --ref img.jpg  # reference contours/image over the feed
python artprojector.py overlay --ref ref/nadya111/4/bw  # a folder: arrows switch reference
python artprojector.py gen-template         # generate a full-canvas target template
```

Common options: `--cam N` (camera index, default 0), `--px-per-mm 2.0` (output
scale), `--ref FILE|DIR` (reference for overlay; a folder is stepped through
with the arrow keys), `--target squares|grid` and `--board 12x16|16x20` (which
printed target; `--board` implies `--target grid` and also sets the canvas size,
since the board is canvas-sized), `--canvas-w-in 12 --canvas-h-in 16`
(canvas size), `--adjust FILE.npz` (where overlay saves/loads its adjustment),
`--view-max 2000` (render size of the corrected overlay view),
`--fit consensus|corners` (how the homography is fitted), `--k1/--k2` (starting
lens distortion for `calibrate`; normally just press `a` there instead),
`--ppm 4` (resolution of the `gen-template` output), `--fullscreen`,
`--display N|NAME`, `--no-keep-awake`.

### Window placement

`--fullscreen` opens the window fullscreen; `--display` picks the monitor, by
index or by name (`--display 1`, `--display DP-1`). `artprojector.py list`
prints the monitors along with the cameras. The move happens before the
fullscreen flag, because what a window manager fullscreens onto is whichever
monitor the window is already on.

The monitors come from `xrandr --listmonitors` on purpose: an OpenCV window is
a Qt window on an X11 screen, XWayland included, so xrandr's idea of the
layout is the one that decides where the window lands - even in a Wayland
session whose compositor reports something else. The catch is that a mirrored
pair shows up at the same position, and then no coordinate can tell the two
apart; `list` says so when it happens, and `--display` cannot help there.

### Staying awake

A painting session is hours of not touching the keyboard, which is what every
idle timer is built to punish - and a suspend is worse than annoying here,
since the camera must not move between `calibrate` and `overlay`, and waking
up usually re-enumerates the USB camera. So both of these are held for the
life of the session:

- a `systemd-inhibit` child (`caffeinate -dimsu` on macOS) blocking idle,
  sleep and the lid switch;
- a poke every 45 s telling the screensaver somebody is still here, via
  `org.freedesktop.ScreenSaver.SimulateUserActivity` over `gdbus`, falling
  back to `xdg-screensaver reset` and then `xset s reset`.

Two of them because they fail differently: the inhibitor is the correct
mechanism but needs logind and a desktop that honours it, while the poke works
through the screensaver but not against a scheduled suspend. Everything is
best-effort - a line saying which ones came up is printed at startup, and a
machine that cannot be kept awake still runs. `--no-keep-awake` turns it off.

The camera must not move between `calibrate` and `run`/`overlay`.

### calibrate

Point the camera so the squares are visible; all six is best, since a single
row constrains the plane poorly. A green outline means the square was detected
and mapped into the 3x2 model; red means it is outside the grid. The `col_off`
and `row_off` trackbars shift the label mapping when only part of the block is
visible (leave them at 0 for a full 3x2 view). Press `c` to compute and save the
homography to `calibration.npz`; the snap residual and the mean reprojection
error are printed.

**Centrelines, not edges.** A printed square is a black frame ~0.8 mm wide, so
thresholding gives two contours for it - the outside of the stroke and the hole
inside. They are averaged into the stroke centreline. Taking either one instead
measures the square a full stroke width wrong, and *which* one survives depends
on the exposure and on whether the blur has merged the 0.6 mm gap between
neighbouring squares, so the bias is not even stable between runs. `n centred`
in the HUD counts the squares where both contours were found.

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

**The line snap** (`e`, on by default). Corners are the worst-localised
features on the sheet, and the consensus quad keeps only four of them. The
snap puts the rest of the target back to work: it projects all 24 printed
edges into the raw frame, walks the intensity profile across each one at ~900
sample points, finds the dark ridge to a fraction of a pixel, and re-solves
the homography (`r` adds `k1`/`k2` to the unknowns) so the projected lines land
on the ink. The cyan outlines are that fit - they should sit on the printed
lines, not beside them - and `snap` in the HUD is what is left over, in mm on
the sheet.

That residual is the one number here that cannot lie to you. A corner fit maps
the model block onto the imaged block by construction, so it reports a small
error even when the model is wrong - the error just moves out onto the canvas
where nothing is measuring it. Twenty-four lines at once cannot all be
satisfied by a wrong model. Expect **≤0.05 mm**; anything above ~0.15 mm means
the sheet is not the sheet the code thinks it is (wrong print scale, a
different target) or there is lens distortion left - not that this particular
frame was unlucky. On synthetic scenes the snap takes the error at the far
canvas corner from 3.3 mm to 0.24 mm, and from 1.0 mm to 0.06 mm on the sheet
itself (`python synthtest.py --n 12`).

The search runs coarse-to-fine on purpose. The fine pass may only look 0.62 mm
to each side of where it expects the line, because the neighbouring square's
ink starts about a millimetre past it and a sample that walks that far
measures the wrong line while still reporting a beautiful residual. So a
coarse pass first lines the block up using only its four outer edges, which
have no neighbour within 60 mm.

**Where a line is measured.** Not at its darkest point across the profile -
at its inner flank, the step from the square's blank interior into the ink.
Squares are 1.355 mm apart and their lines are 0.81 mm wide, so the white gap
between an inner pair is barely half a millimetre: three or four pixels at a
realistic 7 px/mm. Blur fills it in, each line's dip is dragged toward its
neighbour, and the bottom of the merged trough goes flat enough for noise to
move the minimum a whole pixel. On a real frame that showed as 0.145 mm on
the fourteen inner edges against 0.098 mm on the ten outer ones, and in
synthetic frames the ratio tracks the blur: 1.11 sharp, 1.31 soft, 1.61 very
soft. Measured at the flank it stays near 1.1, because every square is blank
inside for 63 mm and a blurred step does not move.

The price is that the flank sits half a line width off the centre, and
printers do not lay ink down at exactly the width asked for. So the fit solves
for that too (`ink=` in the HUD, `info["ink_mm"]`): getting it wrong dilates
every square about its own centre by the same amount, which is not a
homography and therefore cannot hide in `H`. Reading much above the drawn
0.81 mm means the frame is soft rather than the printer generous - blur makes
a narrow line's two flanks lean outwards - and it is a decent proxy for how
much sharpness is costing you.

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
image, and multiply - all blended with adjustable opacity. An optional third
layer shows what changed since a snapshot (the brush, fresh paint) at full
opacity on top.

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
| `n` | snapshot the canvas + delta layer on (see below) |
| `t` | delta layer on / off |
| `5` / `6` | delta threshold down / up |
| `v` | cycle the capture resolution (lower = faster, less detail) |
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

**Delta layer (experimental).** Normally there are two layers: the live camera
frame and the reference blended over it. `n` adds a third. It snapshots the
current frame - take it with nothing in front of the camera - and from then on
the *snapshot* is the bottom layer instead of the live feed, with the reference
blended over it as usual; on top of both, every pixel that now differs from the
snapshot is painted straight from the live frame at full opacity. So the brush,
the hand and fresh paint stay perfectly sharp and unblended while everything
around them keeps the calm blended look, instead of the brush swimming under a
half-transparent reference.

`t` switches the layer off and on again without losing the snapshot; `n` retakes
it. The difference is measured per channel (not on gray, so paint whose
brightness matches the paper still registers), blurred against sensor noise,
opened, and grown slightly so the fading outline of the brush is covered too.
`5`/`6` set the threshold: too low and lighting drift makes the whole canvas
count as delta, too high and thin strokes drop out. If the light in the room
changes, or the canvas moves, retake the snapshot with `n`.

**Speed.** Per frame at 2592x1944 into a 1545x2000 corrected view, camera
excluded (measured on a 24-thread CPU, OpenCV 4.10):

| | ms/frame |
|---|---|
| contours | 11 |
| image | 9 |
| multiply | 10 |
| + delta layer | +5 |

Everything is plain CPU OpenCV, and deliberately so. There is no CUDA in a
`pip install opencv-python` build (`cv2.cuda.getCudaEnabledDeviceCount()` is 0),
so the only GPU path available is OpenCL through `UMat`, and measured against
these sizes it wins ~2x on `remap` (2.5 -> 1.1 ms) and nothing at all on
`warpPerspective` (0.9 ms either way): the pixel counts here are small enough
that IPP and 24 threads already saturate memory bandwidth, and the upload plus
download alone costs 1.7 ms. What did matter was avoiding numpy on whole frames -
boolean-indexed float32 gathers and `axis=2` reductions are single-threaded
passes over 15 MB, and replacing three of them with OpenCV calls took image mode
from 114 to 9 ms and multiply from 159 to 10 ms.

The remaining per-frame cost is mostly the camera: MJPG decode of one
2592x1944 frame is ~26 ms against ~6 ms at 1280x960, plus the USB transfer of
4.5 MB against 1.1 MB, and no GPU touches either. That is what `v` is for - it
retunes the live capture to the next smaller mode of the same aspect ratio and
rescales H to match (H is in pixels, so it has to follow; `scale_homography`
explains when that is legitimate and when it is not). Rectified alignment holds
to ~0.1 mm at 1280x960 and ~0.3 mm at 640x480 on synthetic frames, so the cost
of the switch is resolved detail when zoomed in, not accuracy. `v` wraps back
round to full resolution, and a mode the camera will not actually stream is
skipped rather than left broken.

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
python artprojector.py gen-template --out calib_template.png --ppm 4
```

Overlaying this template (`overlay --ref calib_template.png`) lands the contours
on the real squares, which is a quick way to confirm the calibration and the
adjustment. It is also a convenient base to draw a reference on with the correct
proportions.

Check the overlay against **this**, not against a reference traced over a photo
of the sheet by hand: the template is the model itself, so whatever does not
line up is the calibration. A hand-traced one is good to a few tenths of a
millimetre at best - `calibr-1216.png` here turned out to be within 0.15 mm on
the block size but shifted 0.65 mm up - and those errors are the same size as
the ones being hunted. `calibr-1216-exact.png` is the generated equivalent at
the same 1152x1536.

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

For the grid target the printed geometry lives in `gridtarget.py` (`CELL_MM`,
`MARKER_MM`, `LINE_MM`, `BOARDS`) and `artprojector.py` imports it, so the PDF
and the detector cannot drift apart. Change any of it and you have to reprint
*and* recalibrate. `GRID_ANCHOR_X_MM`/`GRID_ANCHOR_Y_MM` in `artprojector.py`
are the equivalent of `RIGHT_MARGIN_MM`/`BOTTOM_MARGIN_MM`: they stay zero when
the board is taped flush to the canvas edges, and are where a millimetre of
mounting error goes if it is not.

## Hardware notes

- Through OpenCV/AVFoundation the GXI-IMX179 reliably streams only 1920x1080
  MJPG; 2048/2592/3264 and 1280/640 return zero frames. That is the working
  maximum here.
- The camera sometimes does not stream on the first open, so opening and single
  captures retry with reopen.
- Make sure no other application (FaceTime and similar) is holding the camera.

## Files

- `artprojector.py` - the main tool (calibrate / run / overlay / gen-template / list).
- `gridtarget.py` - the 1-inch ArUco grid: its geometry (imported by
  `artprojector.py`) and the PDF generator. `--check` reads the tiles back
  through the detector to prove the board is fully covered.
- `templates/` - the generated grids, plus a `README.md` on printing and gluing
  them. Regenerate with `python gridtarget.py`.
- `capture.py` - grab one frame to a file (for debugging).
- `make_refs.py` - posterized reference variations.
- `synthtest.py` - accuracy check on synthetic frames with a known homography;
  `--legacy` runs the pre-SVG square model against the real geometry,
  `--target grid` runs the ArUco board rendered from its actual PDF.
- `calibr.svg` - the calibration target to print (100%, A4 artwork).
- `calibr-1216.png` - the target drawn over a 12x16" canvas, by hand.
- `calibr-1216-exact.png` - the same from `gen-template`, exact.
- `calibration.npz` - written by `calibrate`: the homography `H` (mm -> undistorted
  image px), the lens distortion `k1`/`k2`, the canvas size it was made for,
  which `target` (and `board`) it was fitted to, and `model_sig`, the target
  geometry itself - change the geometry and `run`/`overlay` will tell you the
  file is stale instead of quietly drifting.
