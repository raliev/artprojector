#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gridtarget.py
=============

The 1-inch calibration GRID: the geometry of it, and the PDFs to print it.

The old target (calibr.svg - six 63 mm squares on one A4 sheet) has to be in
view as a whole, in a known place, to mean anything: it is six anonymous
rectangles, so which is which comes from the operator setting col_off/row_off
by hand. Point the camera at a corner of the canvas from 20 cm away and there
is nothing usable in the frame at all.

This target fixes that by covering the WHOLE canvas with a 1-inch grid and
putting an ArUco marker in every cell. Every cell then says out loud which
cell it is, so:

  * two or three cells anywhere in the frame are enough - the markers give the
    absolute position on the canvas, and the grid lines around them give the
    perspective;
  * nothing has to be aligned by hand, and nothing can be off by one cell;
  * the camera can sit as close to the canvas as you like.

WHY ARUCO AND NOT DOTS OR QR
----------------------------
Counted dots were the first idea, and they do not survive the arithmetic: a
16x20" board has 320 cells, and 320 countable dots do not fit in a 25.4 mm cell
next to a 4.7 mm clear zone. Splitting the count into row/column groups helps
but still needs up to 20 dots per group, at which point a dot is ~1 mm and a
misread is one glance away.

A QR code fits, but a Version-1 symbol is 21x21 modules: in an 18 mm cell that
is a 0.86 mm module, and QR decoding is all-or-nothing - a symbol either reads
or it does not, and at 70 degrees off-axis with a bit of motion blur it mostly
does not.

ArUco is what is actually built for this. DICT_4X4_1000 is 6x6 modules
including the border, so a 16 mm marker has a 2.7 mm module - three times the
QR module - and the detector is designed around the projective distortion of a
flat marker seen at an angle. It also returns four sub-pixel corners with a
known identity, which means ONE marker already pins a homography, and OpenCV
ships it (cv2.aruco), so there is no new dependency for the detection side.

THE LAYOUT
----------
  cell       25.4 mm (1")           the grid pitch, and what the lines draw
  marker     16.0 mm centred        leaves a 4.7 mm clear ring inside the cell
  line       0.6 mm

The 4.7 mm ring is the point of the marker being smaller than the cell: the
sub-pixel line snap in artprojector.py walks about 2 mm to each side of a grid
line looking for the edge of the ink, and it must never find the marker
instead. 4.7 mm of blank paper is more than twice that.

The board is anchored to the BOTTOM-RIGHT corner of the canvas, the same corner
everything else in this project measures from, and it is the same size as the
canvas - so the printed border IS the canvas edge and there is nothing to
measure with a ruler.

PRINTING
--------
  python gridtarget.py                    # writes templates/*.pdf

Full-size PDFs (the page IS 12x16" / 16x20") for a large printer, plus A4 and
Letter tilings for an ordinary one. Print at 100% / "actual size" - NEVER "fit
to page", which silently scales everything by ~6% and quietly ruins every
millimetre this project reports. Each tile carries a 100 mm ruler to check that.

The tiling grows from the bottom-right corner of the board: the bottom-right
sheet is used as it comes out of the printer, and every other sheet is trimmed
along the printed line on its RIGHT and/or BOTTOM edge and glued on top of its
neighbour, so all the trimming and gluing happens on the left and top. See
templates/README.md, which this script writes too.
"""

import argparse
import math
import os

import cv2
import numpy as np

MM_PER_IN = 25.4

# --------------------------------------------------------------------------
#  THE PRINTED GEOMETRY.  artprojector.py imports these - they are the single
#  source of truth for what is on the paper, so a regenerated PDF and the
#  detector cannot drift apart. Changing any of them means reprinting AND
#  recalibrating.
# --------------------------------------------------------------------------
CELL_MM = 25.4                     # grid pitch (1 inch)
MARKER_MM = 16.0                   # ArUco side, centred in the cell
LINE_MM = 0.6                      # printed width of a grid line

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_1000
ARUCO_BORDER_BITS = 1
ARUCO_MODULES = 4 + 2 * ARUCO_BORDER_BITS      # 6x6 modules on the paper

# board name -> (cols, rows, first marker id)
#
# The id ranges are kept apart on purpose. Both boards would otherwise start at
# id 0 and a 12x16 board photographed while the software is configured for
# 16x20 would decode into a real - and wrong - cell. With disjoint ranges the
# ids simply fall outside the configured board and the mistake is visible
# instead of silent.
BOARDS = {
    "12x16": (12, 16, 0),          # ids   0..191
    "16x20": (16, 20, 500),        # ids 500..819
}

# The clear ring of paper between the marker and the grid lines. Not a setting -
# a consequence - but worth having by name, because the line-snap search range
# in artprojector.py has to stay well inside it.
CLEAR_MM = (CELL_MM - MARKER_MM) / 2.0


def board_spec(name):
    """'16x20' -> (cols, rows, id_base). Also accepts '16x20in', '16 x 20'."""
    key = str(name).lower().replace(" ", "").replace("in", "").replace('"', "")
    if key not in BOARDS:
        raise ValueError(f"unknown board {name!r}; known: {', '.join(BOARDS)}")
    return BOARDS[key]


def board_size_mm(name):
    cols, rows, _ = board_spec(name)
    return cols * CELL_MM, rows * CELL_MM


def marker_id(col, row, cols, id_base):
    return id_base + row * cols + col


def cell_of_id(mid, cols, rows, id_base):
    """Marker id -> (col, row), or None if it is not on this board."""
    i = int(mid) - id_base
    if i < 0 or i >= cols * rows:
        return None
    return i % cols, i // cols


_DICT = None


def aruco_dictionary():
    global _DICT
    if _DICT is None:
        _DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    return _DICT


def marker_bits(mid):
    """(6,6) bool array of the marker, True where the paper is BLACK.

    Row 0 / column 0 of the array is the top-left of the marker as printed,
    which is also the corner cv2.aruco.detectMarkers() returns first."""
    img = cv2.aruco.generateImageMarker(aruco_dictionary(), int(mid),
                                        ARUCO_MODULES,
                                        borderBits=ARUCO_BORDER_BITS)
    return img == 0


# ==========================================================================
#  BOARD COORDINATES
#
#  Board-local millimetres: bx to the right from the board's LEFT edge, by
#  DOWNWARD from its TOP edge. That is the paper's own frame and the one the
#  PDF is laid out in. artprojector.py shifts it to the world frame (origin at
#  the bottom-right corner of the canvas) when it builds the model.
# ==========================================================================
def cell_rect_mm(col, row):
    """(bx0, by0, bx1, by1) of a cell - the CENTRELINES of its four lines."""
    return (col * CELL_MM, row * CELL_MM, (col + 1) * CELL_MM, (row + 1) * CELL_MM)


def marker_rect_mm(col, row):
    """(bx0, by0, bx1, by1) of the marker printed in that cell."""
    x0, y0, _, _ = cell_rect_mm(col, row)
    return (x0 + CLEAR_MM, y0 + CLEAR_MM,
            x0 + CLEAR_MM + MARKER_MM, y0 + CLEAR_MM + MARKER_MM)


# ==========================================================================
#  PDF OUTPUT
#
#  reportlab is only needed to PRINT the target, never to use it, so it is
#  imported here rather than at module load - artprojector.py imports this
#  module for the geometry alone and must not need a PDF library to run.
# ==========================================================================
PAGES = {                          # page sizes in mm, portrait
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
}

PRINT_MARGIN_MM = 5.0              # unprintable border to stay out of
OVERLAP_MM = 14.0                  # how far a sheet laps over its neighbour

# Every module of every marker is drawn as a filled rectangle rather than as an
# embedded bitmap: a bitmap gets resampled by the printer driver and the module
# edges come out soft, which is exactly the thing the detector measures. The
# runs are merged horizontally and then grown by a hair, because two rectangles
# that share an edge exactly show a white hairline in some PDF viewers.
_HAIRLINE_MM = 0.02


def _draw_marker(c, mm, mid, x0, y0_top, page_h_mm):
    """Draw one marker with its top-left at board->page point (x0, y0_top)."""
    bits = marker_bits(mid)
    m = MARKER_MM / ARUCO_MODULES
    for r in range(ARUCO_MODULES):
        col = 0
        while col < ARUCO_MODULES:
            if not bits[r, col]:
                col += 1
                continue
            run = col
            while run < ARUCO_MODULES and bits[r, run]:
                run += 1
            x = x0 + col * m - _HAIRLINE_MM
            w = (run - col) * m + 2 * _HAIRLINE_MM
            y_top = y0_top + r * m - _HAIRLINE_MM
            h = m + 2 * _HAIRLINE_MM
            c.rect((x) * mm, (page_h_mm - y_top - h) * mm, w * mm, h * mm,
                   stroke=0, fill=1)
            col = run


def _draw_board(c, mm, board, ox, oy, page_h_mm, clip=None):
    """Draw the whole board with its top-left corner at page mm (ox, oy).

    `clip` is (x0, y0, x1, y1) in page mm, top-down - the region of the page
    that may receive ink. Everything is clipped to it, so a tile is literally
    a window onto the same drawing as the full-size sheet: there is no separate
    'tile geometry' that could disagree with it."""
    cols, rows, id_base = board_spec(board)
    bw, bh = cols * CELL_MM, rows * CELL_MM

    c.saveState()
    if clip is not None:
        cx0, cy0, cx1, cy1 = clip
        p = c.beginPath()
        p.rect(cx0 * mm, (page_h_mm - cy1) * mm, (cx1 - cx0) * mm, (cy1 - cy0) * mm)
        c.clipPath(p, stroke=0, fill=0)

    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setLineWidth(LINE_MM * mm)
    c.setLineCap(0)                                  # butt: lines end where told

    for i in range(cols + 1):                        # verticals
        x = (ox + i * CELL_MM) * mm
        c.line(x, (page_h_mm - oy) * mm, x, (page_h_mm - oy - bh) * mm)
    for j in range(rows + 1):                        # horizontals
        y = (page_h_mm - oy - j * CELL_MM) * mm
        c.line(ox * mm, y, (ox + bw) * mm, y)

    # only the cells that can actually land inside the clip window
    if clip is None:
        c0, r0, c1, r1 = 0, 0, cols - 1, rows - 1
    else:
        c0 = max(0, int(math.floor((clip[0] - ox) / CELL_MM)) - 1)
        r0 = max(0, int(math.floor((clip[1] - oy) / CELL_MM)) - 1)
        c1 = min(cols - 1, int(math.ceil((clip[2] - ox) / CELL_MM)))
        r1 = min(rows - 1, int(math.ceil((clip[3] - oy) / CELL_MM)))
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            mx, my, _, _ = marker_rect_mm(col, row)
            _draw_marker(c, mm, marker_id(col, row, cols, id_base),
                         ox + mx, oy + my, page_h_mm)
    c.restoreState()


def _text(c, mm, page_h_mm, x, y, s, size=8, gray=0.0, rotate=0):
    c.saveState()
    c.setFillGray(gray)
    c.setFont("Helvetica", size)
    c.translate(x * mm, (page_h_mm - y) * mm)
    c.rotate(rotate)
    c.drawString(0, 0, s)
    c.restoreState()


def _ruler(c, mm, page_h_mm, x, y, length_mm=100.0, vertical=True):
    """A printed 100 mm bar. If it does not measure 100 mm, the print scaled."""
    c.saveState()
    c.setStrokeGray(0.35)
    c.setFillGray(0.35)
    c.setLineWidth(0.25 * mm)
    if vertical:
        c.line(x * mm, (page_h_mm - y) * mm, x * mm, (page_h_mm - y - length_mm) * mm)
        for t in range(0, int(length_mm) + 1, 10):
            w = 3.0 if t % 50 == 0 else 1.8
            c.line(x * mm, (page_h_mm - y - t) * mm,
                   (x + w) * mm, (page_h_mm - y - t) * mm)
    else:
        c.line(x * mm, (page_h_mm - y) * mm, (x + length_mm) * mm, (page_h_mm - y) * mm)
        for t in range(0, int(length_mm) + 1, 10):
            w = 3.0 if t % 50 == 0 else 1.8
            c.line((x + t) * mm, (page_h_mm - y) * mm,
                   (x + t) * mm, (page_h_mm - y + w) * mm)
    c.restoreState()


def _cut_line(c, mm, page_h_mm, x0, y0, x1, y1):
    c.saveState()
    c.setStrokeGray(0.0)
    c.setLineWidth(0.2 * mm)
    c.setDash(2 * mm, 2 * mm)
    c.line(x0 * mm, (page_h_mm - y0) * mm, x1 * mm, (page_h_mm - y1) * mm)
    c.restoreState()


def write_full_pdf(path, board):
    """One page, exactly the size of the board.

    The outermost line is centred on the board edge, so the printer clips its
    outer half. That is cosmetic and not an error: the snap in artprojector.py
    locates a line by its INNER flank - the step from the blank cell into the
    ink - which does not move when the outer half is missing."""
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import mm

    cols, rows, _ = board_spec(board)
    bw, bh = cols * CELL_MM, rows * CELL_MM
    c = rl_canvas.Canvas(path, pagesize=(bw * mm, bh * mm))
    c.setTitle(f"artprojector calibration grid {board}")
    _draw_board(c, mm, board, 0.0, 0.0, bh)
    c.showPage()
    c.save()
    return 1


def tile_layout(board_w, board_h, page_w, page_h,
                margin=PRINT_MARGIN_MM, overlap=OVERLAP_MM):
    """How many sheets, and where each one's window sits on the board.

    Returns (n_x, n_y, tiles) with tiles[(i, j)] = (bx0, by0, bx1, by1), the
    board rectangle that sheet gets. i counts columns from the RIGHT edge of
    the board, j counts rows from the BOTTOM - because that is the corner the
    assembly starts from and the corner the canvas is measured from."""
    pw, ph = page_w - 2 * margin, page_h - 2 * margin
    step_x, step_y = pw - overlap, ph - overlap
    if step_x <= 0 or step_y <= 0:
        raise ValueError("overlap is as wide as the printable area")
    n_x = 1 + max(0, math.ceil((board_w - pw) / step_x - 1e-9))
    n_y = 1 + max(0, math.ceil((board_h - ph) / step_y - 1e-9))
    tiles = {}
    for j in range(n_y):
        for i in range(n_x):
            x1 = board_w - i * step_x
            y1 = board_h - j * step_y
            tiles[(i, j)] = (x1 - pw, y1 - ph, x1, y1)
    return n_x, n_y, tiles


def write_tiled_pdf(path, board, page, margin=PRINT_MARGIN_MM,
                    overlap=OVERLAP_MM):
    """The board cut into printable sheets, one per page.

    Each sheet carries its window of the board placed so that the window's
    RIGHT and BOTTOM edges sit `margin` in from the paper's right and bottom
    edges. Sheets therefore lap over their right and lower neighbours by
    `overlap`, and the assembly is: put the bottom-right sheet down as it is,
    then trim every other sheet along its printed right/bottom line and glue it
    on top of the sheet it laps over. Nothing is ever trimmed on the left or
    the top, and the bottom-right sheet is not trimmed at all."""
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import mm

    cols, rows, _ = board_spec(board)
    bw, bh = cols * CELL_MM, rows * CELL_MM
    pw_mm, ph_mm = PAGES[page]
    n_x, n_y, tiles = tile_layout(bw, bh, pw_mm, ph_mm, margin, overlap)

    c = rl_canvas.Canvas(path, pagesize=(pw_mm * mm, ph_mm * mm))
    c.setTitle(f"artprojector calibration grid {board} on {page}")

    # bottom-right sheet first, then leftwards and upwards - the order they are
    # meant to be glued in, so "sheet 3 of 6" is also "the third one you place"
    for j in range(n_y):
        for i in range(n_x):
            bx0, by0, bx1, by1 = tiles[(i, j)]
            # board origin on the page: the window's top-left goes to (margin, margin)
            ox, oy = margin - bx0, margin - by0
            clip = (margin, margin, pw_mm - margin, ph_mm - margin)
            _draw_board(c, mm, board, ox, oy, ph_mm, clip)

            if i > 0:              # a neighbour to the right: trim to it
                _cut_line(c, mm, ph_mm, pw_mm - margin, margin,
                          pw_mm - margin, ph_mm - margin)
            if j > 0:              # a neighbour below: trim to it
                _cut_line(c, mm, ph_mm, margin, ph_mm - margin,
                          pw_mm - margin, ph_mm - margin)

            _tile_info(c, mm, ph_mm, board, page, i, j, n_x, n_y,
                       bx0, by0, bx1, by1, bw, bh, margin, overlap)
            c.showPage()
    c.save()
    return n_x * n_y


def _tile_info(c, mm, page_h_mm, board, page, i, j, n_x, n_y,
               bx0, by0, bx1, by1, bw, bh, margin, overlap):
    """Labels, instructions and the scale ruler - only where they cannot show.

    Two places qualify. The first is the strip of the board that the next sheet
    to the left laps over: it is covered in the finished target, so ink there
    is invisible. The second is the paper outside the board on the outermost
    sheets, which is blank by construction. Anything that fits in neither is
    not printed at all - a label inside a live cell would sit in the few
    millimetres the line snap searches through, and it would be measured as if
    it were a printed line."""
    cols, rows, _ = board_spec(board)
    # which cells of the board this sheet carries (1-based, for a human)
    c0 = max(1, int(math.floor(bx0 / CELL_MM)) + 1)
    r0 = max(1, int(math.floor(by0 / CELL_MM)) + 1)
    c1 = min(cols, int(math.ceil(bx1 / CELL_MM)))
    r1 = min(rows, int(math.ceil(by1 / CELL_MM)))
    n = j * n_x + i + 1
    head = f"{board}  {page.upper()}  sheet {n} of {n_x * n_y}"
    where = (f"col {i + 1} from the RIGHT, row {j + 1} from the BOTTOM"
             f"   |   covers cells x {c0}-{c1}, y {r0}-{r1}")

    # the strip the left-hand neighbour will cover, in page mm
    strip_x = margin + 1.0
    _text(c, mm, page_h_mm, strip_x, page_h_mm - margin - 2.0,
          f"{head}   {where}   PRINT AT 100%", size=6, gray=0.45, rotate=90)

    # blank paper outside the board, if this sheet has any
    top_blank = max(0.0, -by0)              # mm of board-space above the board
    left_blank = max(0.0, -bx0)
    lines = [head, where,
             "PRINT AT 100% / ACTUAL SIZE - never 'fit to page'.",
             "Assemble from the BOTTOM-RIGHT sheet outwards: lay it down as printed,",
             "then trim each other sheet along its dashed right/bottom line and glue",
             "it ON TOP of the sheet it overlaps, matching the grid lines.",
             "The finished border is the canvas edge - tape it flush to the canvas."]
    if top_blank >= 34.0:
        y = margin + 6.0
        for k, s in enumerate(lines):
            _text(c, mm, page_h_mm, margin + 2.0, y + k * 4.2, s,
                  size=8 if k == 0 else 6.5, gray=0.25)
        _ruler(c, mm, page_h_mm, margin + 2.0, y + len(lines) * 4.2 + 6.0,
               100.0, vertical=False)
        _text(c, mm, page_h_mm, margin + 2.0, y + len(lines) * 4.2 + 10.5,
              "100 mm - measure it", size=6, gray=0.35)
    elif left_blank >= 34.0:
        x = margin + 6.0
        for k, s in enumerate(lines):
            _text(c, mm, page_h_mm, x + k * 4.2, page_h_mm - margin - 2.0, s,
                  size=8 if k == 0 else 6.5, gray=0.25, rotate=90)
        _ruler(c, mm, page_h_mm, x + len(lines) * 4.2 + 6.0,
               page_h_mm - margin - 2.0, 100.0, vertical=True)


README = """\
artprojector calibration grids
==============================

What is here
------------
`grid-<size>-full.pdf`   one page, exactly <size> inches - for a large printer.
`grid-<size>-a4.pdf`     the same board cut into A4 sheets.
`grid-<size>-letter.pdf` the same board cut into Letter sheets.

The board is a 1-inch grid the size of the canvas, with an ArUco marker in every
cell. The marker says which cell it is, so the software needs only two or three
cells anywhere in the frame to know both the perspective and where on the canvas
the camera is looking.

Printing
--------
100% / "actual size". NOT "fit to page", NOT "shrink oversized pages".
Every tile has a 100 mm ruler printed on it - measure it before you cut
anything. If it reads 94 mm, the print was scaled to fit and the whole target
is useless; print it again.

Assembling a tiled board
------------------------
The sheets overlap, and they are meant to be assembled from the bottom-right
corner outwards, so that all the trimming and gluing happens on the left and the
top and the bottom-right sheet is used exactly as it came out of the printer.

  1. Lay sheet 1 (bottom-right) down. Do not trim it.
  2. Take the sheet to its left. Trim it along the dashed line on its RIGHT
     edge. Lay it ON TOP of sheet 1 so the grid lines and the markers continue
     without a step, and glue it there.
  3. Carry on leftwards, then upwards row by row. A sheet with a neighbour
     below it is also trimmed along its dashed BOTTOM line.

Alignment is easy to check: a grid line that crosses a seam must stay straight,
and the two half-cells on either side must add up to one 1-inch cell.

Mounting - read this one carefully
---------------------------------
The reference is the printed BORDER LINE, never the paper. The board is exactly
canvas-sized, so its outer border is where the canvas edge is supposed to be.

The paper does not end there. A printer cannot print to the edge, so the
bottom-right sheet of a tiling keeps about 5 mm of blank paper outside the
border (and the full-size PDF picks up whatever margin your large printer adds).
If you tape the sheet down by its PAPER edge, the whole board sits a centimetre
or so up and to the left of where the software thinks it is - and everything
downstream is out by that centimetre while every diagnostic still reads perfect,
because the calibration measures the ink and the ink is fine.

So do one of these two things:

  a. Trim along the printed border line on the right and the bottom, and tape
     that cut edge flush with the canvas edges. Nothing to measure, nothing to
     configure. (The left and top edges do not matter - trim or not.)

  b. Leave the paper as it is, measure the two gaps with a ruler - from the
     canvas right edge to the board's right border, and from the canvas bottom
     edge to the board's bottom border - and tell the software:

         python artprojector.py calibrate --target grid --board 12x16 \
             --grid-anchor-x <minus your measurement> \
             --grid-anchor-y <minus your measurement>

     Negative, because the board sits left of and above the canvas corner. Do
     not copy numbers out of an example: this is the one quantity nothing in
     the software can check, so a wrong value calibrates perfectly and is
     wrong by exactly that amount forever after.

     The numbers are saved with the calibration and reused on later runs, so
     if you later trim or remount the board, pass them again (0 0 when the
     border ends up flush) - otherwise the old offset quietly survives.

(a) is more accurate and needs no ruler, so prefer it unless you would rather
not cut the sheet.

Using it
--------
    python artprojector.py calibrate --target grid --board 16x20

Aim the camera anywhere on the board, from as close as you like. The window
shows the markers it recognised, the cell each one is, and how far the fitted
grid still is from the printed ink ("snap", in mm). Press 'c' to save.
"""


def generate_all(outdir="templates", boards=("12x16", "16x20"),
                 pages=("a4", "letter"), margin=PRINT_MARGIN_MM,
                 overlap=OVERLAP_MM, write_readme=True):
    os.makedirs(outdir, exist_ok=True)
    made = []
    for b in boards:
        cols, rows, base = board_spec(b)
        p = os.path.join(outdir, f"grid-{b}-full.pdf")
        write_full_pdf(p, b)
        made.append((p, 1))
        for pg in pages:
            p = os.path.join(outdir, f"grid-{b}-{pg}.pdf")
            n = write_tiled_pdf(p, b, pg, margin, overlap)
            made.append((p, n))
        print(f"[grid] {b}: {cols}x{rows} cells, "
              f"{cols * CELL_MM:.1f}x{rows * CELL_MM:.1f} mm, "
              f"marker ids {base}..{base + cols * rows - 1}")
    if write_readme:
        with open(os.path.join(outdir, "README.md"), "w") as f:
            f.write(README)
        made.append((os.path.join(outdir, "README.md"), 0))
    for p, n in made:
        print(f"[grid] wrote {p}" + (f"  ({n} page{'s' * (n != 1)})" if n else ""))
    return made


def check(outdir="templates", boards=("12x16", "16x20"),
          pages=("a4", "letter"), ppm=6.0):
    """Rasterise every tile and read it back with the real detector.

    The tiling is the one part of this that is easy to get wrong and hard to
    see: an off-by-one in the step leaves a strip of the board on no sheet at
    all, and the missing strip is a few millimetres wide in the middle of a
    stack of pages nobody checks until they are glued down. So this asks the
    only questions that matter - is every cell of the board on some sheet, and
    does every marker land where tile_layout() says it does - and it asks them
    of the actual PDF, through the actual detector.

    Needs pdftoppm (poppler) and is not part of printing; run it after changing
    the page sizes, the margin or the overlap."""
    import subprocess
    import tempfile
    import artprojector as ap

    ok = True
    for board in boards:
        cols, rows, _ = board_spec(board)
        bw, bh = board_size_mm(board)
        for page in pages:
            pw, ph = PAGES[page]
            n_x, n_y, tiles = tile_layout(bw, bh, pw, ph)
            pdf = os.path.join(outdir, f"grid-{board}-{page}.pdf")
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["pdftoppm", "-r", f"{ppm * 25.4:.6f}", "-png",
                                "-gray", pdf, os.path.join(tmp, "p")], check=True)
                ap.use_grid_target(board)
                seen, worst = set(), 0.0
                for j in range(n_y):
                    for i in range(n_x):
                        n = j * n_x + i + 1
                        f = os.path.join(tmp, f"p-{n}.png")
                        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"  {pdf}: page {n} missing"); ok = False; continue
                        bx0, by0, _, _ = tiles[(i, j)]
                        ox, oy = PRINT_MARGIN_MM - bx0, PRINT_MARGIN_MM - by0
                        found, _foreign, _clipped = ap.detect_grid_cells(img)
                        for (c, r, q) in found:
                            seen.add((c, r))
                            x0, y0, x1, y1 = marker_rect_mm(c, r)
                            exp = np.array([[x0 + ox, y0 + oy], [x1 + ox, y0 + oy],
                                            [x1 + ox, y1 + oy], [x0 + ox, y1 + oy]])
                            worst = max(worst, float(np.abs(q - exp * ppm).max()))
            missing = {(c, r) for r in range(rows) for c in range(cols)} - seen
            bad = missing or worst / ppm > 0.5
            ok = ok and not bad
            print(f"  {'FAIL' if bad else 'ok  '} {board:6s} {page:6s} "
                  f"{n_x}x{n_y}={n_x * n_y} sheets   "
                  f"cells on some sheet {len(seen)}/{cols * rows}   "
                  f"marker placement within {worst / ppm:.3f} mm"
                  + (f"   MISSING {sorted(missing)[:6]}" if missing else ""))
    print("[grid] check: " + ("all good" if ok else "PROBLEMS ABOVE"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Generate the 1-inch ArUco calibration grids as PDFs")
    ap.add_argument("--out", default="templates", help="output directory")
    ap.add_argument("--boards", default="12x16,16x20",
                    help=f"comma separated; known: {','.join(BOARDS)}")
    ap.add_argument("--pages", default="a4,letter",
                    help=f"comma separated; known: {','.join(PAGES)}")
    ap.add_argument("--margin", type=float, default=PRINT_MARGIN_MM,
                    help="unprintable page border to stay out of, mm")
    ap.add_argument("--overlap", type=float, default=OVERLAP_MM,
                    help="how far a sheet laps over its neighbour, mm")
    ap.add_argument("--check", action="store_true",
                    help="after writing, read every tile back through the "
                         "detector and verify the board is fully covered "
                         "(needs pdftoppm)")
    args = ap.parse_args()
    boards = [b for b in args.boards.split(",") if b]
    pages = [p for p in args.pages.split(",") if p]
    generate_all(args.out, boards, pages, args.margin, args.overlap)
    if args.check:
        raise SystemExit(0 if check(args.out, boards, pages) else 1)


if __name__ == "__main__":
    main()
