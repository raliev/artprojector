# artprojector

A small OpenCV tool for painters. A USB camera looks at a canvas on an easel;
the tool figures out the perspective of the canvas from a printed calibration
target and can rectify the view or overlay a reference image (as contours or as
a semi-transparent picture) aligned to the canvas plane.

The camera may sees only part of the canvas. From the printed target the tool
computes a homography between the image and the canvas plane (in millimeters),
which is enough to rectify what the camera sees and to project a reference onto
it.

There is a second half, for an easel with a real projector aimed at it:
[`projcalib.py`](#projector-calibration---projcalibpy) measures where the
projector's pixels land on the canvas - using the calibrated camera as the
instrument - and [`project.py`](#projecting-a-reference---projectpy) throws a
reference onto the canvas, pre-distorted so that it lands square on it whatever
angle the projector sits at.

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
python gridtarget.py                  # writes templates/*.pdf (16x20 and 20x16)
python gridtarget.py --boards all     # every size below
python gridtarget.py --check          # ...and verifies the tiling (needs pdftoppm)
python artprojector.py calibrate --target grid --board 16x20
```

The sizes, in inches, `--board` and `gridtarget.BOARDS`:

| portrait | landscape |
|---|---|
| 8x10, 11x14, 12x16, 16x20 | 10x8, 14x11, 16x12, 20x16 |

plus 4x4, which is one sheet and useful for a quick test.

`templates/` gets, for each size:

- `grid-<size>-full.pdf` - one page, exactly that many inches, for a large printer;
- `grid-<size>-a4.pdf`, `grid-<size>-letter.pdf` - the same board cut into
  ordinary sheets (1 for 4x4, 4 for 12x16, 6 for 16x20, on either paper).

The board is a 1-inch grid the size of the canvas, with a 16 mm ArUco marker
(`DICT_4X4_1000`) centred in every cell. Print at 100%, assemble, and mount it -
see below. `templates/README.md` (written by the generator) has the assembly
instructions; every tile also carries a 100 mm ruler to catch a print that was
scaled to fit the page.

### One printed board, several canvas sizes

A marker's id says how far its cell is from the board's **bottom-right** corner,
and nothing else - and that is the corner the board is mounted by and the corner
the whole project measures from. So every board *is* the bottom-right part of
every larger board, ids and world coordinates included. Mounted flush:

- a printed **16x20** is also a 12x16, an 11x14, an 8x10 and a 4x4;
- a printed **20x16** is also a 16x12, a 14x11, a 10x8 and a 4x4.

Two sheets therefore cover every size, and there are two ways to use one:

```
# name the canvas: only the cells over it are used, the rest are foreign
python artprojector.py calibrate --target grid --board 14x11

# name the sheet, then the canvas: the overhanging cells are used too, which
# spreads the fit over more of the frame and is slightly more accurate
python artprojector.py calibrate --target grid --board 20x16 \
    --canvas-w-in 14 --canvas-h-in 11
```

`calibrate` prints which larger sheet fits, and `grid-probe` says so too when it
sees ids from beyond the configured board - that is not an error.

`gridtarget.py --check` verifies the claim rather than asserting it: every cell
of every board must carry the same id, at the same distance from the bottom-right
corner, in every board that covers it.

### Already printed a board? Keep it (`--board-rev`)

Boards printed before this used template **rev 1**: cells numbered row-major from
the **top-left**, out of a range reserved per board (`12x16` → 0..191,
`16x20` → 500..819), which is exactly what stopped them being interchangeable.
Same paper, same cells, same lines, same mounting - only the id in each cell
differs. So a rev 1 board is still a perfectly good board **at its own size**:

```
python artprojector.py calibrate --board 12x16 --board-rev 1
```

- `--board-rev auto` is the **default**: the revision is worked out from the ids
  that come out of the first frame with markers in it, and printed. The wrong
  scheme only ever explains about a third of them, so the vote is not close.
- The revision is saved in `calibration.npz` (`board_rev`), so it is asked once.
  A calibration file written before the revisions existed is taken as rev 1,
  which is what it must have been fitted to.
- `grid-probe` reports how many ids fit each revision - that is the answer to
  "this board worked yesterday and now everything is foreign".
- `python gridtarget.py --rev 1 --boards 12x16` reprints the original sheet
  unchanged, as `templates/grid-12x16-rev1-*.pdf`, for replacing a torn sheet of
  a board already on the wall.

What a rev 1 sheet cannot do is stand in for a smaller canvas - that is the thing
rev 2 bought, and it needs a rev 2 print. Tiles carry `rev1`/`rev2` in their
header; the full-size PDFs carry it in the PDF title, since nothing may be
printed on the board itself.

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

  The board's *thickness* is the third measurement of the same kind, and it
  matters for the same reason: glued to card, the printed grid sits a
  millimetre or two above the canvas, which a camera at this angle sees as a
  millimetre or two sideways. See [The thickness of the
  target](#calibrate) - `--thickness`, or `[` / `]` in `calibrate`.

**Checking it.** Use `gen-template` output as the overlay reference, not a PDF
converted to an image:

```
python artprojector.py gen-template --target grid --board 12x16 --out check.png
python artprojector.py overlay --target grid --board 12x16 --ref check.png --adjust none
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
by 8 mm. `--adjust none` is the answer when the point is to check the
calibration: nothing is loaded, nothing is saved, and the command means the same
thing on any machine. (Naming a file that does not exist yet also gives zeros -
`--adjust /tmp/fresh.npz` used to be the advice here - but only until it exists,
and only on the machine where it is missing, which is the opposite of
reproducible.) To see what is in the file rather than bypass it: `python -c
"import numpy;d=numpy.load('overlay_adjust.npz');print({k:float(d[k]) for k in
d.files})"`.

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
lines straight), so it belongs in the lens model and not in `H`: press `a` to fit
it to the marker corners, `r` to let the line snap keep refining `k1`/`k2`.
Fitting a homography to a bent frame without that buries the bend in `H`, where
it comes back as millimetres somewhere else on the canvas. And a bend the model
cannot express - a radial field about a principal point that is not the middle of
the frame - buries itself in `H` the same way while every residual on screen
still reads perfect, which is why `a` fits the principal point too (see *Lens
distortion* below).

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
with the arrow keys), `--target squares|grid` and `--board 12x16|20x16|...`
(which printed target; `--board` implies `--target grid` and also sets the canvas
size, since the board is canvas-sized - a larger printed sheet serves a smaller
board, see [above](#one-printed-board-several-canvas-sizes)),
`--canvas-w-in 12 --canvas-h-in 16`
(canvas size), `--adjust FILE.npz|none` (where overlay saves/loads its
adjustment; `none` = no adjustment, nothing saved),
`--view-max 2000` (render size of the corrected overlay view),
`--fit consensus|corners` (how the homography is fitted), `--thickness 1.5`
(how thick the printed target is, in mm - see below), `--k1/--k2` (starting
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

The inhibitor is let go however the program ends. A clean exit, a Ctrl-C, a
`SIGTERM` or a closed terminal go through an atexit hook and signal handlers,
which kill the whole process group - the inhibitor holds a child of its own,
and killing only the parent leaves that child running. Underneath both sits a
dead man's switch for the cases nobody gets told about (`kill -9`, a segfault
in a camera driver): the command the inhibitor holds is a `cat` reading a pipe
whose other end is ours, so the kernel closing our file descriptors is itself
the release. Anything an older version still left behind is swept up at the
next startup - an inhibitor of ours whose parent is no longer an artprojector
belongs to nobody, and is reported as `released N inhibitor(s)`.

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
fit suffer. Press `a` to auto-fit the lens: it solves for the coefficients that
make the target's grid lines straightest, which takes a fraction of a second.
`1`/`2` and `3`/`4` adjust `k1`/`k2` by hand, `5`/`6` change the step, `0`
resets. The `line residual` in the HUD is the mean distance from the corners to
the line they should lie on - minimize it. Detection, the consensus quad and the
homography all run on the corrected frame, and the coefficients are saved into
`calibration.npz`, so `run` and `overlay` undistort every frame automatically
(both the raw and the corrected view). In a synthetic test with `k1=-0.12` the
correction cut the resulting canvas-corner error from ~13 px to ~0.5 px.

*The whole lens, not just `k1`/`k2`.* `a` fits `k1`, `k2`, `k3`, the two
tangential terms **and the principal point**, and the last of those is the one
that matters. A wrong focal length costs nothing - the model is a polynomial in
`r/f`, so pinning `f` to `max(w,h)` is absorbed exactly by rescaling `k1`/`k2` -
but the optical axis of a cheap module is tens of pixels off the middle of the
sensor, and correcting a radial field about the wrong centre leaves a residual
that is not a homography. `H` cannot swallow it, and **nothing in the HUD sees
it**: over the few cells in view the leftover field is nearly affine, so the fit
absorbs it locally, the line snap keeps reporting 0.04 mm, and the millimetres
go wrong somewhere else on the canvas. On the synthetic bench (14x11 board,
320 mm span, otherwise perfect lens) a principal point 35 px off centre costs
**1.2 mm over the cells in view** while the snap still reads 0.04 mm; fitting it
brings that to 0.16 mm. The one number that does react is the straight-line
residual `a` prints, which is why it prints both the before and the after.

The extra terms are only fitted when the frame can locate them - six cells or
more, spread across at least 35% of the frame in both directions. A close-up of
three cells cannot say where the optical axis is, so `a` says so and stays with
`k1`/`k2`. Given the spread, any improvement in the residual is taken:
`synthtest.py --fit-lens --lens=...` says demanding a 15% improvement was worse
in both worlds, including the one where the lens really is two-parameter.
`--two-param` there reproduces the old behaviour for comparison.

**The thickness of the target** (`[` / `]`, `--thickness`). The target is
printed on something - paper on cardboard, a mounted print, foam board - so the
ink lies a millimetre or two *above* the canvas, and everything in this window
measures the ink. A camera looking straight down would not care: two parallel
planes differ only in depth, and depth is what a top-down view collapses. This
camera looks from the side on purpose, so that it is not between the painter
and the canvas, and at that angle the point of ink and the point of canvas
underneath it are two different pixels. The gap is `thickness x tan(angle from
the normal)`: on a 50° view, every millimetre of card is a millimetre of error,
and at 70° it is nearly three.

Set it with `[` and `]` (0.2 mm a press) or `--thickness 1.5`, and watch the
yellow outline: it is where the same coordinates land on the canvas *under* the
target, next to the cyan fit on the ink. The HUD says what the correction is
worth in millimetres in the middle of the frame. Default is 0 - the ink taken
to lie flat on the canvas.

This is the second number, with the mounting offset above, that nothing in the
frame can check for you: the snap residual is measured against the ink, and the
ink is exactly where the fit says it is, so every diagnostic here reads
beautifully while the overlay lands a millimetre off in the same direction
everywhere. Measure the card with calipers, or measure the miss on the canvas
and dial the value until it goes away.

It is stored *beside* the fit in `calibration.npz` rather than folded into it,
so `--thickness 1.6` on any later `run`/`overlay` re-measures the cardboard
without recalibrating, and the saved `H` stays what the camera actually saw.
Internally the correction turns `H` back into a camera pose - `H = K[x y o]`
for the plane it was fitted to, so the parallel plane `dz` further away is
`K[x y o + dz*n]` - and hands the result on as an ordinary 3x3 homography;
nothing downstream learns that a pose was involved. The weak link is that
`camera_matrix()` pins the focal length to `max(w, h)` instead of measuring it,
and the focal length is what sets the estimated tilt, so read the correction as
very nearly right rather than exact. It is exactly linear in the thickness,
which is why the number has hotkeys: tuning it by eye absorbs whatever the
focal length gets wrong.

### run

Loads `calibration.npz` and shows the rectified canvas plane in real time.
`g` toggles a 50 mm grid, `c` recalibrates, `q` quits.

### overlay

Draws a reference over the live frame, projected into the canvas perspective.
The reference is stretched to the whole canvas, so only the part inside the
camera view is shown. Four render modes: contours (Canny edges), the original
image, multiply, and overlay - all blended with adjustable opacity. An optional third
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
| `m` | cycle contours / image / multiply / overlay |
| `I` | invert the reference (the overlaid image, not the camera) |
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

**Overlay mode** is multiply's answer to a dark canvas. Multiply only darkens,
so once there is dark paint under it a black reference line disappears into it -
which is exactly where the drawing is still needed. Overlay keeps the
reference's *coverage* (how black the ref pixel is) but takes the ink's
lightness from the canvas underneath: white ink over dark paint, black ink over
light paper, decided per pixel at gray 118. So the line reads at the same
strength everywhere, and white areas of the reference still leave the frame
untouched, as in multiply. It is not Photoshop's "overlay" blend, which fades
the ink out over mid greys - and mid greys are most of a painting; the
lightness is chosen with a hard threshold instead.

**`I` inverts the reference itself** - the overlaid image, never the camera
frame. A negative (white lines on black) becomes usable ink for multiply and
overlay, and it is the quickest way to read a value study the other way round.
Contours mode does not change: Canny sees the same edges either way.

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
| overlay | 20 |
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
`--adjust` file by `p` and loaded at start-up on the next run; `i` zeroes it for
this session. `--adjust none` neither loads nor saves - the reference is then
placed by the calibration alone, which is what a check of the calibration wants.

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

The **mounting offset comes from `calibration.npz`**, like the board and the
thickness do, and `gen-template` prints the figure it used. It has to: the
template is drawn where the board is mounted, and a template drawn flush against
a board that is mounted 10 mm in fails the check by 10 mm - with the lens, the
snap and the print scale all lining up as suspects. `--grid-anchor-x/-y`
override it, but then they have to be the same numbers `calibrate` was given, and
`overlay` now says so out loud when they are not.

Check the overlay against **this**, not against a reference traced over a photo
of the sheet by hand: the template is the model itself, so whatever does not
line up is the calibration. A hand-traced one is good to a few tenths of a
millimetre at best - `calibr-1216.png` here turned out to be within 0.15 mm on
the block size but shifted 0.65 mm up - and those errors are the same size as
the ones being hunted. `calibr-1216-exact.png` is the generated equivalent at
the same 1152x1536.

## Projector calibration - projcalib.py

The camera calibration says where the canvas is in the camera frame. A
projector aimed at the same canvas needs the other half of the answer: which
projector pixel lands on which millimetre of the canvas. `projcalib.py` measures
it, using the already-calibrated camera as the instrument:

```
python artprojector.py list                        # which monitor is the projector
python projcalib.py --cam 0 --display 1 --board 12x16
```

Two windows open: the pattern goes fullscreen on `--display`, and the camera
view goes to the first monitor that is *not* the projector - `--cam-display`
(index or name) overrides that, and with a single screen both land on it and the
camera window has to be moved by hand.

It fills the projector with a lattice of squares - the same idea as the printed
target, made of light and measured in projector pixels - with an ArUco marker
centred in every cell. The camera reads whichever markers fall in its frame,
each one says which cell it is, and the chain

```
projector px --H_pc--> camera px --H_cam^-1--> canvas mm
```

is fitted and composed. What gets saved (`projector.npz`) is that composition
and its inverse, `mm -> projector px`, which is what a reference image has to be
warped through to land on the canvas.

The projected markers are `DICT_5X5_1000` while the printed board is
`DICT_4X4_1000`, so the printed target can stay on the canvas while this runs -
and it is worth leaving there, because then the verification views show the
projected geometry landing on printed lines whose position is known
independently.

### How fine the lattice should be

`--cells` (default 16 across the projector frame), `--cell-px` in projector
pixels, or `n`/`m` while it runs; the report says what a cell comes to in
millimetres on the canvas and roughly how many cells fall on it.

The instinct on seeing the pattern is that the squares are far too big, and it
is usually right, but not for the reason it feels like. Big markers decode from
further away and refine their corners better - the trouble is that a camera
framed on part of the canvas then holds two or three of them, and a homography
fitted to four corners in one corner of the frame extrapolates badly across
everything else. On a synthetic close-up view (the camera on about a third of
the canvas, the projector covering half again more than the canvas), 8 cells
gave 2 markers and 0.45 px of error at the far corner of the canvas; 16 gave 10
markers and 0.19 px; 24 gave 24 markers and 0.08 px.

The other end of the trade is real too, and synthetic frames cannot show it: a
projected marker is softened by the projector's focus, by the canvas texture and
by the camera, so past some fineness the modules stop being resolved and the
marker is simply not read. Below 56 px per cell the lattice is clamped for that
reason. Between the two, go as fine as still decodes reliably in *your* room -
the marker count in the HUD is the thing to watch while pressing `m`.

### What it reports

Pressing `c` measures (several frames averaged, RANSAC over the markers) and
prints the things that decide whether the projector is aimed well enough:

- the residual **in millimetres on the canvas**, not in camera pixels - a
  homography has no term for a bowed canvas or for the projector's own lens
  distortion, and both land here;
- how much of the projector frame the camera actually saw the markers over -
  everything outside that is extrapolated;
- where the projected frame lands in canvas millimetres, what share of the
  canvas it covers, and what share of the light falls past it;
- the clearance from each canvas edge to the edge of the projection, in mm, and
  a loud line when the projection does not reach the canvas at all;
- projector pixels per mm on the canvas (i.e. the dpi actually available for
  painting), the keystone across the canvas, and the rotation of the projector's
  image relative to the canvas axes.

### Checking it by eye

`r` projects a white rectangle exactly the size of the canvas, `g` a test grid
drawn in canvas millimetres, `v` a canvas-sized image (`--verify-image`, by
default `calibr-1216-exact.png`). All three are authored in millimetres and
pushed through the fit, so they close the loop without the camera: the lit
rectangle either sits on the canvas or it does not, and a centimetre of error is
obvious across the room. ENTER saves.

An existing `projector.npz` is loaded at startup, so all of that works
immediately and a session that only wants to check the alignment - or to nudge
it - never has to measure again. Re-measuring (`c`) is for when the projector or
the canvas has moved.

### When the rectangle misses the canvas: the hand adjustment

`a`/`d`/`w`/`s` move the projection 2 mm, `z`/`x` scale it about the canvas
centre, `[`/`]` and `-`/`=` scale one axis, `,`/`.` rotate, `i` zeroes it - the
keys, the steps and the meaning are overlay's, because it is the same gesture on
the same canvas. It is saved into `projector.npz` (in `H_mm_to_proj`, with the
raw fit and the adjustment kept beside it), and announced on every load: a
measured mapping and a hand-tuned one must never become indistinguishable.

`TAB` cycles five states: the whole projection, then each of the four corners.
In a corner state `a`/`d`/`w`/`s` move that corner alone, and a big arrow -
drawn on the canvas itself, not just in the camera window - points at the one
being moved, because the person nudging it is looking across the room at the
canvas.

The corners are not a convenience, they are the other half of the freedom. Move,
scale and rotate together are a similarity, and a similarity cannot bend a
rectangle: if the projection is short at one corner and long at the opposite
one, nothing in it can help. The four corners add exactly the four degrees of
freedom that make the adjustment a full homography, which is where a residual
keystone, a canvas that is not quite rectangular, or a bowed corner can be taken
out. Drag a corner across its neighbours and there is no homography at all; that
falls back to no correction rather than to a matrix full of infinities.

Before reaching for it, note **what shape the error is**, because it names the
cause. An error that is zero at the right and bottom edges and grows towards the
left and top is not an offset - those two edges are the origin of the world
frame - it is a **scale**. If the rectangle stops 0.5" short on the left of a
12" canvas and 5/8" short at the top of a 16" one, that is ~4% in both axes, and
there are only two things it can be:

- **the canvas is not the size the model thinks.** Nothing in the frame marks
  the canvas edge, so the model takes it from the board (`--board 12x16`). Give
  it the real size with `--canvas-w-in/--canvas-h-in` and the projection lands
  where it should, with the calibration untouched.
- **the printed board is not the size it says**, i.e. it was printed at ~96%.
  Then the millimetre itself is 4% short - *everywhere*, `overlay` included -
  and this is simply the first view honest enough to show it.

`g` tells them apart in one measurement: put a ruler on the projected 50 mm
grid. Squares that are not 50 mm mean the millimetre is wrong, and the fix
belongs at the printer, not here. Squares that are 50 mm while the border misses
the canvas edge mean the canvas is a different size than declared. The hand
adjustment papers over either one, but only for the projector.

Two things the fit cannot notice on its own. The camera must not have moved
since `artprojector.py calibrate` - a nudged camera gives a perfectly
self-consistent fit that is simply in the wrong place. And once saved, the
mapping is between the projector and the canvas alone: it survives the camera
being moved or unplugged, and goes stale the moment the projector or the canvas
moves.

If the projector is set to mirror its image (rear projection, some ceiling
mounts), nothing is detected at all rather than detected wrong - a mirrored
ArUco marker is not a codeword. Turn the flip off in the projector's menu.

## Projecting a reference - project.py

```
python project.py --ref ref/nadya-1/4/bw --display 1
python project.py --ref check.png --display 1 --fit contain --invert
```

The other end of the calibration: a picture is placed on the canvas in
millimetres, pushed backwards through `mm -> projector px`, and what the
projector displays is a keystoned, rotated, oversized shape nobody would
recognise - which lands on the canvas as the picture, square to its edges and
the right size, whatever angle the projector sits at.

Nothing is projected outside the canvas rectangle: the frame, the easel and the
wall stay dark. That is politer to look at and it is also a standing check - the
lit area *is* the canvas, so if it creeps off the edge, something has moved.

`--ref` takes a file or a folder, and a folder steps with LEFT/RIGHT, as in
`overlay`; the masks and contour sheets from `make_refs.py` are what this is for.
`I` inverts (white lines on black - far easier to trace by, and it lights the
canvas far less), `9`/`0` dim and brighten, `f` switches stretch-to-canvas for
keep-aspect, `b` outlines the canvas, `k` blacks out.

The placement keys are overlay's again (`a/d/w/s`, `z/x`, `[ ]`, `- =`, `,/.`,
`i`, and `p` to save), and this adjustment goes to its own file,
`project_adjust.npz`. Three adjustments now exist and they are deliberately
separate files, because they answer different questions: `overlay_adjust.npz`
places a reference in the *camera* view, `project_adjust.npz` places it in the
*projection*, and the one inside `projector.npz` corrects the projector-canvas
mapping itself. If the lit rectangle no longer matches the canvas edges, that
last one (or the calibration) is what is wrong; if it matches and the drawing
sits badly inside it, it is this one.

Two windows open, as in `projcalib.py`: the picture goes fullscreen on
`--display`, the readout to a small control window on another monitor
(`--ctl-display`) - a HUD painted onto the canvas would be a HUD painted onto
the painting.

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
- `projcalib.py` - projector calibration: fills the projector with an ArUco
  lattice, reads it through the calibrated camera and writes `projector.npz`
  (`mm -> projector px`, the hand adjustment included, with the raw fit and the
  adjustment kept beside it).
- `project.py` - projects a reference onto the canvas through that mapping,
  pre-distorted so it lands square on the canvas; `project_adjust.npz` is where
  its placement is saved.
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
  image px, on the plane of the ink), the lens distortion `k1`/`k2`, the
  `thickness` of the printed target (applied to `H` on load, which is what puts
  it on the canvas), the canvas size it was made for,
  which `target` (and `board`, `board_rev`) it was fitted to, and `model_sig`, the target
  geometry itself - change the geometry and `run`/`overlay` will tell you the
  file is stale instead of quietly drifting.
- `projector.npz` - written by `projcalib.py`: `H_mm_to_proj` (the one to use,
  hand adjustment included) and its inverse, `H_mm_to_proj_raw` and
  `adjust`/`corner_adjust` (what was measured and what was tuned by hand, kept
  apart on purpose), the projector frame size the pixels refer to, the canvas
  size, and the residual and marker count of the fit.
- `overlay_adjust.npz`, `project_adjust.npz` - where a reference sits, in the
  camera view and in the projection respectively. Three adjustments exist in
  total and they are three files because they answer three different questions;
  see the `project.py` section.
