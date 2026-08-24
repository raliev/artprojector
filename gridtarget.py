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
everything else in this project measures from, and it is at most the size of the
canvas - so the printed border IS the canvas edge and there is nothing to
measure with a ruler.

ONE BOARD, MANY CANVAS SIZES
----------------------------
An id says where its cell is relative to that bottom-right corner and nothing
else, so every board is the bottom-right corner of every bigger board, markers
and all. A printed 16x20 mounted flush IS a 12x16, an 11x14, an 8x10 or a 4x4 -
the extra rows and columns simply hang off the edge of the smaller canvas - and
a printed 20x16 does the same for the landscape sizes. Two sheets cover the lot.

Two ways to use a big board on a small canvas, and they are not the same:

  * `--board 14x11` on a printed 16x12: the software only looks at the 14x11
    corner. Simple, and the canvas size comes out right on its own; the markers
    outside that corner are counted as foreign and ignored.
  * `--board 16x12 --canvas-w-in 14 --canvas-h-in 11`: every marker on the
    sheet is used, including the ones hanging over the edge, which is a wider
    spread of cells to fit through and therefore a better fit. Name the board
    you printed, name the canvas you are painting on.

PRINTING
--------
  python gridtarget.py                    # writes templates/*.pdf
  python gridtarget.py --boards all       # every size in BOARDS
  python gridtarget.py --boards 4x4       # one sheet, for a quick test

Full-size PDFs (the page IS 16x20" / 20x16") for a large printer, plus A4 and
Letter tilings for an ordinary one. Print at 100% / "actual size" - NEVER "fit
to page", which silently scales everything by ~6% and quietly ruins every
millimetre this project reports. Each tile carries a 100 mm ruler to check that.

ALREADY PRINTED A BOARD?  Keep it.
----------------------------------
The sheets printed before this (template rev 1) number their cells row-major
from the TOP-LEFT out of a range reserved per board, which is what stopped them
being interchangeable. Same paper, same cells, same lines, same mounting - only
the id in each cell differs. So they are still perfectly good boards, at their
own size, and nothing needs reprinting:

  python artprojector.py calibrate --board 12x16 --board-rev 1

`--board-rev auto`, the default, works it out from the ids it decodes and says
so; a calibration file made before the revisions existed is taken as rev 1. What
a rev 1 sheet cannot do is stand in for other sizes - that came with rev 2 and
needs a rev 2 print. `python gridtarget.py --rev 1 --boards 12x16` reprints the
original sheet unchanged (as templates/grid-12x16-rev1-*.pdf), for replacing one
that tore without recalibrating.

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

# --------------------------------------------------------------------------
#  BOARD SIZES AND MARKER IDS
#
#  Every board is a window onto ONE lattice of ids, and the corner the window
#  is fixed to is the BOTTOM-RIGHT - the corner the canvas is measured from and
#  the corner the board is mounted by. So an id says how far its cell is from
#  the right and the bottom edge of the board, and nothing else:
#
#      id = ID_STRIDE * (cells up from the bottom) + (cells left of the right)
#
#  which is what makes the sizes interchangeable. The bottom-right 14x11 corner
#  of a 16x12 sheet carries exactly the markers a 14x11 sheet would - the same
#  ids, and (because artprojector.py places the board by that same corner) the
#  same world coordinates. One printed 16x20 therefore serves every portrait
#  size below it and one printed 20x16 every landscape one; a smaller canvas
#  just leaves the rest of the sheet hanging over the edge, or unused. See
#  boards_covering() / minimal_boards().
#
#  What this gives up is worth stating, because it used to be the whole design:
#  the ids used to be a disjoint range per board, so a sheet photographed while
#  the software was configured for another size decoded to ids that were simply
#  not on the configured board, and the mistake was visible instead of silent.
#  Most of that survives. Two boards' ids agree only where the two boards agree
#  GEOMETRICALLY - in the shared bottom-right rectangle, where reading one as
#  the other is not a mistake but the point - and every cell outside it still
#  falls off the configured board and is counted as foreign. ID_STRIDE is also
#  deliberately wider than the widest board, which leaves 12 ids per lattice row
#  that no sheet ever carries, so a garbled read has a fair chance of landing on
#  one and being thrown away.
#
#  Changing ID_STRIDE or the formula means REPRINTING every board: the ink is
#  what says which cell it is. Hence TEMPLATE_REV, and hence rev 1 below - a
#  printed board is a physical object that someone had to find a printer for,
#  so a new id scheme does not get to invalidate one.
#
#  REV 1 was the scheme this started with: ids row-major from the board's
#  TOP-LEFT, from a base kept apart per board (12x16 -> 0..191,
#  16x20 -> 500..819). Those two sizes are all that was ever printed that way,
#  and the two sheets are NOT interchangeable with each other or with anything
#  else, because an id row-major from the top-left depends on the width of the
#  board it was printed for. Everything else about the sheet - the cells, the
#  lines, the 16 mm markers, the mounting - is identical between the revisions;
#  only which id sits in which cell changed. So a rev 1 sheet stays perfectly
#  usable, at its own size, by telling the software which revision it is:
#
#      python artprojector.py calibrate --board 12x16 --board-rev 1
#
#  and `--board-rev auto` (the default) works it out from the ids it decodes.
#  The revision is stored in the calibration file, so it is asked once.
# --------------------------------------------------------------------------
ID_STRIDE = 32                     # ids per lattice row; > the widest board
N_IDS = 1000                       # what DICT_4X4_1000 holds
TEMPLATE_REV = 2                   # bumped when the ids or the geometry change

BOARDS = {                         # name (inches) -> (cols, rows) in 1" cells
    "4x4":   (4, 4),
    "8x10":  (8, 10),
    "10x8":  (10, 8),
    "11x14": (11, 14),
    "14x11": (14, 11),
    "12x16": (12, 16),
    "16x12": (16, 12),
    "16x20": (16, 20),
    "20x16": (20, 16),
}

# The rev 1 sheets, and their id bases. Nothing else was ever printed at rev 1,
# so nothing else can be read at rev 1.
LEGACY_BOARDS = {
    (12, 16): 0,                   # ids   0..191
    (16, 20): 500,                 # ids 500..819
}

# Which revision the ink in front of us is. Module state because it is a
# property of the printed object, like CELL_MM - everything that reads or draws
# an id needs it, and threading it through every call site would only mean the
# detector and the PDF writer could disagree about it.
ID_REV = TEMPLATE_REV

# The clear ring of paper between the marker and the grid lines. Not a setting -
# a consequence - but worth having by name, because the line-snap search range
# in artprojector.py has to stay well inside it.
CLEAR_MM = (CELL_MM - MARKER_MM) / 2.0


def set_id_rev(rev):
    """Read and draw ids as printed by template revision `rev`."""
    global ID_REV
    rev = int(rev)
    if rev not in (1, TEMPLATE_REV):
        raise ValueError(f"unknown template revision {rev}; "
                         f"known: 1, {TEMPLATE_REV}")
    ID_REV = rev
    return ID_REV


def board_spec(name):
    """'16x20' -> (cols, rows). Also accepts '16x20in', '16 x 20', '16x20"'."""
    key = str(name).lower().replace(" ", "").replace("in", "").replace('"', "")
    if key not in BOARDS:
        raise ValueError(f"unknown board {name!r}; known: {', '.join(BOARDS)}")
    return BOARDS[key]


def board_size_mm(name):
    cols, rows = board_spec(name)
    return cols * CELL_MM, rows * CELL_MM


def legacy_boards():
    """The board names that exist at rev 1."""
    return [n for n, cr in BOARDS.items() if cr in LEGACY_BOARDS]


def _legacy_base(cols, rows):
    base = LEGACY_BOARDS.get((cols, rows))
    if base is None:
        raise ValueError(
            f"there is no rev 1 board of {cols}x{rows} cells - rev 1 was only "
            f"ever printed as {', '.join(legacy_boards())}. Print a rev "
            f"{TEMPLATE_REV} sheet for this size, or use --board-rev "
            f"{TEMPLATE_REV}.")
    return base


def marker_id(col, row, cols, rows, rev=None):
    """The id printed in cell (col, row) of a cols x rows board.

    col/row count from the board's TOP-LEFT, because that is the frame the PDF
    is laid out in. The id counts from the BOTTOM-RIGHT, because that is the
    corner every board shares - except at rev 1, which counted row-major from
    the top-left and so tied an id to the width of one board."""
    if (ID_REV if rev is None else int(rev)) == 1:
        return _legacy_base(cols, rows) + row * cols + col
    return ID_STRIDE * (rows - 1 - row) + (cols - 1 - col)


def cell_of_id(mid, cols, rows, rev=None):
    """Marker id -> (col, row) on a cols x rows board, or None if not on it."""
    i = int(mid)
    if (ID_REV if rev is None else int(rev)) == 1:
        i -= _legacy_base(cols, rows)
        if i < 0 or i >= cols * rows:
            return None
        return i % cols, i // cols
    if i < 0:
        return None
    left, up = i % ID_STRIDE, i // ID_STRIDE
    if left >= cols or up >= rows:
        return None
    return cols - 1 - left, rows - 1 - up


def board_ids(name, rev=None):
    """The ids printed on that board. Not a contiguous range at rev 2."""
    cols, rows = board_spec(name)
    return {marker_id(c, r, cols, rows, rev)
            for r in range(rows) for c in range(cols)}


def boards_with_id(mid, rev=None):
    """Which known boards carry that id - for 'those are the 20x16 board'."""
    use = ID_REV if rev is None else int(rev)
    names = legacy_boards() if use == 1 else list(BOARDS)
    return [n for n in names if cell_of_id(mid, *BOARDS[n], rev=use)]


def boards_covering(name, rev=None):
    """The boards that can be printed and used AS a `name` board.

    Any board at least as wide and as tall: mounted bottom-right-flush, its
    markers over the smaller board's area are that board. `name` is in the
    list, and the rest is what you may already have on the wall.

    A rev 1 board covers nothing but itself - that is the one thing rev 2
    changed."""
    cols, rows = board_spec(name)
    if (ID_REV if rev is None else int(rev)) == 1:
        return [n for n, cr in BOARDS.items() if cr == (cols, rows)]
    return [n for n, (c, r) in BOARDS.items() if c >= cols and r >= rows]


def boards_covered_by(name, rev=None):
    """The sizes a printed `name` board can stand in for, `name` included."""
    cols, rows = board_spec(name)
    if (ID_REV if rev is None else int(rev)) == 1:
        return [n for n, cr in BOARDS.items() if cr == (cols, rows)]
    return [n for n, (c, r) in BOARDS.items() if c <= cols and r <= rows]


def minimal_boards():
    """The boards nothing else covers. Print these and you have every size."""
    return [n for n in BOARDS if len(boards_covering(n)) == 1]


def _check_boards():
    """Every board's ids have to fit the dictionary, and the lattice row."""
    for name, (cols, rows) in BOARDS.items():
        if cols > ID_STRIDE:
            raise ValueError(f"board {name} is wider than ID_STRIDE={ID_STRIDE}")
        top_left = marker_id(0, 0, cols, rows, rev=TEMPLATE_REV)
        if top_left >= N_IDS:                        # the largest id on it
            raise ValueError(f"board {name} needs id {top_left}, "
                             f"the dictionary holds {N_IDS}")
    for (cols, rows), base in LEGACY_BOARDS.items():
        if (cols, rows) not in BOARDS.values():
            raise ValueError(f"rev 1 board {cols}x{rows} is not in BOARDS")
        if base + cols * rows > N_IDS:
            raise ValueError(f"rev 1 board {cols}x{rows} runs past {N_IDS}")


_check_boards()


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


def _draw_board(c, mm, board, ox, oy, page_h_mm, clip=None, rev=None):
    """Draw the whole board with its top-left corner at page mm (ox, oy).

    `clip` is (x0, y0, x1, y1) in page mm, top-down - the region of the page
    that may receive ink. Everything is clipped to it, so a tile is literally
    a window onto the same drawing as the full-size sheet: there is no separate
    'tile geometry' that could disagree with it."""
    cols, rows = board_spec(board)
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
            _draw_marker(c, mm, marker_id(col, row, cols, rows, rev),
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


def write_full_pdf(path, board, rev=None):
    """One page, exactly the size of the board.

    The outermost line is centred on the board edge, so the printer clips its
    outer half. That is cosmetic and not an error: the snap in artprojector.py
    locates a line by its INNER flank - the step from the blank cell into the
    ink - which does not move when the outer half is missing."""
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import mm

    cols, rows = board_spec(board)
    bw, bh = cols * CELL_MM, rows * CELL_MM
    c = rl_canvas.Canvas(path, pagesize=(bw * mm, bh * mm))
    # the only place a full-size sheet can say which revision it is: nothing may
    # be printed on the board itself, where the line snap would measure it
    c.setTitle(f"artprojector calibration grid {board} "
               f"rev{ID_REV if rev is None else int(rev)}")
    _draw_board(c, mm, board, 0.0, 0.0, bh, rev=rev)
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
                    overlap=OVERLAP_MM, rev=None):
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

    cols, rows = board_spec(board)
    bw, bh = cols * CELL_MM, rows * CELL_MM
    pw_mm, ph_mm = PAGES[page]
    n_x, n_y, tiles = tile_layout(bw, bh, pw_mm, ph_mm, margin, overlap)

    c = rl_canvas.Canvas(path, pagesize=(pw_mm * mm, ph_mm * mm))
    c.setTitle(f"artprojector calibration grid {board} "
               f"rev{ID_REV if rev is None else int(rev)} on {page}")

    # bottom-right sheet first, then leftwards and upwards - the order they are
    # meant to be glued in, so "sheet 3 of 6" is also "the third one you place"
    for j in range(n_y):
        for i in range(n_x):
            bx0, by0, bx1, by1 = tiles[(i, j)]
            # board origin on the page: the window's top-left goes to (margin, margin)
            ox, oy = margin - bx0, margin - by0
            clip = (margin, margin, pw_mm - margin, ph_mm - margin)
            _draw_board(c, mm, board, ox, oy, ph_mm, clip, rev)

            if i > 0:              # a neighbour to the right: trim to it
                _cut_line(c, mm, ph_mm, pw_mm - margin, margin,
                          pw_mm - margin, ph_mm - margin)
            if j > 0:              # a neighbour below: trim to it
                _cut_line(c, mm, ph_mm, margin, ph_mm - margin,
                          pw_mm - margin, ph_mm - margin)

            _tile_info(c, mm, ph_mm, board, page, i, j, n_x, n_y,
                       bx0, by0, bx1, by1, bw, bh, margin, overlap, rev)
            c.showPage()
    c.save()
    return n_x * n_y


def _tile_info(c, mm, page_h_mm, board, page, i, j, n_x, n_y,
               bx0, by0, bx1, by1, bw, bh, margin, overlap, rev=None):
    """Labels, instructions and the scale ruler - only where they cannot show.

    Two places qualify. The first is the strip of the board that the next sheet
    to the left laps over: it is covered in the finished target, so ink there
    is invisible. The second is the paper outside the board on the outermost
    sheets, which is blank by construction. Anything that fits in neither is
    not printed at all - a label inside a live cell would sit in the few
    millimetres the line snap searches through, and it would be measured as if
    it were a printed line."""
    cols, rows = board_spec(board)
    # which cells of the board this sheet carries (1-based, for a human)
    c0 = max(1, int(math.floor(bx0 / CELL_MM)) + 1)
    r0 = max(1, int(math.floor(by0 / CELL_MM)) + 1)
    c1 = min(cols, int(math.ceil(bx1 / CELL_MM)))
    r1 = min(rows, int(math.ceil(by1 / CELL_MM)))
    n = j * n_x + i + 1
    use = ID_REV if rev is None else int(rev)
    also = [b for b in boards_covered_by(board, use) if b != board]
    head = f"{board} rev{use}  {page.upper()}  sheet {n} of {n_x * n_y}"
    where = (f"col {i + 1} from the RIGHT, row {j + 1} from the BOTTOM"
             f"   |   covers cells x {c0}-{c1}, y {r0}-{r1}"
             + (f"   |   also serves {', '.join(also)}" if also else ""))

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
    # How much clear space the block below actually wants: the lines, then the
    # ruler and its label. Measured rather than guessed, because a block that
    # does not fit spills onto the board - into the few millimetres around a
    # printed line that the snap searches through, where it would be measured
    # as if it were the line.
    # 6 mm in from the paper edge, then the lines, then the ruler and its label
    # (the vertical ruler needs only its ticks), and a hair of glyph height.
    need_top = 6.0 + len(lines) * 4.2 + 10.5 + 2.5
    need_left = 6.0 + len(lines) * 4.2 + 6.0 + 4.0

    if top_blank >= need_top:
        y = margin + 6.0
        for k, s in enumerate(lines):
            _text(c, mm, page_h_mm, margin + 2.0, y + k * 4.2, s,
                  size=8 if k == 0 else 6.5, gray=0.25)
        _ruler(c, mm, page_h_mm, margin + 2.0, y + len(lines) * 4.2 + 6.0,
               100.0, vertical=False)
        _text(c, mm, page_h_mm, margin + 2.0, y + len(lines) * 4.2 + 10.5,
              "100 mm - measure it", size=6, gray=0.35)
    elif left_blank >= need_left:
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
`grid-<size>-rev1-*.pdf` the ORIGINAL id scheme, for boards printed before it
                         changed - see "An older board" at the end.

The board is a 1-inch grid the size of the canvas, with an ArUco marker in every
cell. The marker says which cell it is, so the software needs only two or three
cells anywhere in the frame to know both the perspective and where on the canvas
the camera is looking.

One board fits several canvases
-------------------------------
A marker's id says how far its cell is from the board's BOTTOM-RIGHT corner, and
that is the corner the board is mounted by - so every board is exactly the
bottom-right part of every bigger board, ids included. Mounted flush, one
printed 16x20 is also a 12x16, an 11x14, an 8x10 and a 4x4; one printed 20x16 is
also a 16x12, a 14x11, a 10x8 and a 4x4. The rows and columns beyond the smaller
canvas hang over its edge, and that is all that happens.

So print the big one once and tell the software which canvas you are on:

    python artprojector.py calibrate --target grid --board 14x11

Or, to get some use out of the markers hanging over the edge - more cells spread
across the frame is a better fit - name the board you actually printed and the
canvas separately:

    python artprojector.py calibrate --target grid --board 20x16 \\
        --canvas-w-in 14 --canvas-h-in 11

Both are correct. The first is easier; the second is slightly more accurate.

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

An older board
--------------
If you printed a board before the ids moved to the shared bottom-right lattice
(template rev 1: 12x16 and 16x20 only, numbered row-major from the top-left),
KEEP IT. Nothing about the paper changed - the cells, the lines, the markers and
the mounting are identical - so it is still a calibration board:

    python artprojector.py calibrate --board 12x16 --board-rev 1

`--board-rev auto` is the default and works it out from the ids, and the answer
is stored with the calibration, so in practice it is said once or not at all.
`grid-probe` prints which revision the ids fit, which is the answer to "this
board used to work and now everything is foreign".

The one thing a rev 1 sheet cannot do is stand in for a smaller canvas - that is
what rev 2 bought. For that, print a rev 2 board.

`python gridtarget.py --rev 1 --boards 12x16` regenerates the old sheets
(`grid-12x16-rev1-*.pdf`), identical to the ones printed before, for replacing a
damaged sheet of a board already on the wall.
"""


def board_stem(board, rev=None):
    """The filename stem for a board's PDFs.

    The current revision keeps the plain name; an older one is spelled out,
    because two sheets that look identical and decode differently must not be
    one filename."""
    use = ID_REV if rev is None else int(rev)
    return f"grid-{board}" if use == TEMPLATE_REV else f"grid-{board}-rev{use}"


def generate_all(outdir="templates", boards=("16x20", "20x16"),
                 pages=("a4", "letter"), margin=PRINT_MARGIN_MM,
                 overlap=OVERLAP_MM, write_readme=True, rev=None):
    """The default is the two boards that between them cover every size.

    Nothing stops you printing a small one - a 4x4 fits on one sheet and is
    handy for a test - but a printed 16x20 already IS every portrait size in
    BOARDS and a printed 20x16 every landscape one, so those two are the honest
    default. See boards_covered_by().

    `rev=1` reprints a sheet from the old id scheme, byte for byte the board
    that was on the wall before - for replacing a torn one without recalibrating
    or reprinting the rest."""
    use = ID_REV if rev is None else int(rev)
    os.makedirs(outdir, exist_ok=True)
    made = []
    for b in boards:
        cols, rows = board_spec(b)
        stem = board_stem(b, use)
        p = os.path.join(outdir, f"{stem}-full.pdf")
        write_full_pdf(p, b, use)
        made.append((p, 1))
        for pg in pages:
            p = os.path.join(outdir, f"{stem}-{pg}.pdf")
            n = write_tiled_pdf(p, b, pg, margin, overlap, use)
            made.append((p, n))
        ids = board_ids(b, use)
        also = [n for n in boards_covered_by(b, use) if n != b]
        print(f"[grid] {b} rev{use}: {cols}x{rows} cells, "
              f"{cols * CELL_MM:.1f}x{rows * CELL_MM:.1f} mm, "
              f"{len(ids)} marker ids in {min(ids)}..{max(ids)}"
              + (f"\n[grid]   mounted bottom-right-flush it also serves "
                 f"--board {', '.join(also)}" if also else "")
              + (f"\n[grid]   rev 1 ids: read it with --board-rev 1"
                 if use == 1 else ""))
    if write_readme:
        with open(os.path.join(outdir, "README.md"), "w") as f:
            f.write(README)
        made.append((os.path.join(outdir, "README.md"), 0))
    for p, n in made:
        print(f"[grid] wrote {p}" + (f"  ({n} page{'s' * (n != 1)})" if n else ""))
    return made


def check(outdir="templates", boards=("16x20", "20x16"),
          pages=("a4", "letter"), ppm=6.0, rev=None):
    """Rasterise every tile and read it back with the real detector.

    The tiling is the one part of this that is easy to get wrong and hard to
    see: an off-by-one in the step leaves a strip of the board on no sheet at
    all, and the missing strip is a few millimetres wide in the middle of a
    stack of pages nobody checks until they are glued down. So this asks the
    only questions that matter - is every cell of the board on some sheet, and
    does every marker land where tile_layout() says it does - and it asks them
    of the actual PDF, through the actual detector.

    One detection has to be thrown away, and it is worth knowing why. The
    printable-area clip slices the markers at the edge of a tile's window in
    half - they are the ones the neighbouring sheet is glued over, so on the
    finished board they do not exist - and ArUco reads a sliced marker anyway,
    sometimes correcting the missing third into a valid id of some other cell.
    That is a garbled read of ink that is not on the assembled board, so it
    says nothing about the tiling either way, and a detection whose cell cannot
    be on this page at all is discarded and counted. (The count is printed. A
    tiling bug that put a marker on the wrong page would show up there rather
    than vanish.)

    Needs pdftoppm (poppler) and is not part of printing; run it after changing
    the page sizes, the margin or the overlap."""
    import artprojector as ap

    use = ID_REV if rev is None else int(rev)
    was = (ID_REV, ap.BOARD_REV_AUTO, ap.BOARD_REV_RESOLVED)
    try:
        return _check(outdir, boards, pages, ppm, use, ap)
    finally:
        # pinning the revision is this function's business and nobody else's
        set_id_rev(was[0])
        ap.BOARD_REV_AUTO, ap.BOARD_REV_RESOLVED = was[1], was[2]


def _check(outdir, boards, pages, ppm, use, ap):
    import subprocess
    import tempfile

    ok = True
    for board in boards:
        cols, rows = board_spec(board)
        bw, bh = board_size_mm(board)
        for page in pages:
            pw, ph = PAGES[page]
            n_x, n_y, tiles = tile_layout(bw, bh, pw, ph)
            pdf = os.path.join(outdir, f"{board_stem(board, use)}-{page}.pdf")
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["pdftoppm", "-r", f"{ppm * 25.4:.6f}", "-png",
                                "-gray", pdf, os.path.join(tmp, "p")], check=True)
                ap.use_grid_target(board)
                ap.set_board_rev(use)     # pinned: this PDF's revision is known
                seen, worst, dropped = set(), 0.0, 0
                for j in range(n_y):
                    for i in range(n_x):
                        n = j * n_x + i + 1
                        f = os.path.join(tmp, f"p-{n}.png")
                        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"  {pdf}: page {n} missing"); ok = False; continue
                        bx0, by0, _, _ = tiles[(i, j)]
                        ox, oy = PRINT_MARGIN_MM - bx0, PRINT_MARGIN_MM - by0
                        clip = (PRINT_MARGIN_MM, PRINT_MARGIN_MM,
                                pw - PRINT_MARGIN_MM, ph - PRINT_MARGIN_MM)
                        found, _foreign, _clipped = ap.detect_grid_cells(img)
                        for (c, r, q) in found:
                            x0, y0, x1, y1 = marker_rect_mm(c, r)
                            exp = np.array([[x0 + ox, y0 + oy], [x1 + ox, y0 + oy],
                                            [x1 + ox, y1 + oy], [x0 + ox, y1 + oy]])
                            # where this cell's marker would be on this page
                            px0, py0, px1, py1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
                            if (px1 <= clip[0] or py1 <= clip[1]
                                    or px0 >= clip[2] or py0 >= clip[3]):
                                dropped += 1          # not on this page at all
                                continue
                            seen.add((c, r))
                            if (px0 < clip[0] or py0 < clip[1]
                                    or px1 > clip[2] or py1 > clip[3]):
                                continue              # sliced: its corners are
                            worst = max(worst,        # the cut's, not the ink's
                                        float(np.abs(q - exp * ppm).max()))
            missing = {(c, r) for r in range(rows) for c in range(cols)} - seen
            bad = missing or worst / ppm > 0.5
            ok = ok and not bad
            print(f"  {'FAIL' if bad else 'ok  '} rev{use} {board:6s} {page:6s} "
                  f"{n_x}x{n_y}={n_x * n_y} sheets   "
                  f"cells on some sheet {len(seen)}/{cols * rows}   "
                  f"marker placement within {worst / ppm:.3f} mm"
                  + (f"   ({dropped} sliced-marker misreads dropped)"
                     if dropped else "")
                  + (f"   MISSING {sorted(missing)[:6]}" if missing else ""))
    print("[grid] check: " + ("all good" if ok else "PROBLEMS ABOVE"))
    return ok


def check_ids():
    """The interchangeability claim, checked rather than asserted in a comment.

    For every pair of boards where one covers the other, every cell of the
    smaller one must carry the SAME id in the bigger one, and sit the same
    distance from the bottom-right corner. That is the whole promise: print the
    big board, tell the software the small one, and nothing downstream can
    tell the difference. This is a property of rev 2 alone - rev 1 tied an id
    to one board's width - so the revision is named here rather than taken from
    ID_REV."""
    ok = True
    for small, (sc, sr) in BOARDS.items():
        for big in boards_covering(small, rev=TEMPLATE_REV):
            bc, br = BOARDS[big]
            for r in range(sr):
                for c in range(sc):
                    mid = marker_id(c, r, sc, sr, rev=TEMPLATE_REV)
                    # the same cell of the big board: same distance from the
                    # right edge and from the bottom edge
                    cell = cell_of_id(mid, bc, br, rev=TEMPLATE_REV)
                    want = (bc - (sc - c), br - (sr - r))
                    if cell != want:
                        print(f"  FAIL id {mid}: cell ({c},{r}) of {small} is "
                              f"{cell} of {big}, expected {want}")
                        ok = False
    # and the guard: an id no board carries must be refused by all of them
    stray = [i for i in range(N_IDS) if not boards_with_id(i, rev=TEMPLATE_REV)]
    print(f"  ok   {len(BOARDS)} boards agree on every shared cell at rev "
          f"{TEMPLATE_REV}; "
          f"{len(stray)} of {N_IDS} dictionary ids are on no board and are "
          f"rejected as foreign" if ok else "  FAIL see above")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Generate the 1-inch ArUco calibration grids as PDFs")
    ap.add_argument("--out", default="templates", help="output directory")
    ap.add_argument("--boards", default=",".join(minimal_boards()),
                    help=f"comma separated, or 'all'; known: {','.join(BOARDS)}. "
                         f"The default ({','.join(minimal_boards())}) is enough "
                         f"for every size: a board mounted flush with the "
                         f"bottom-right canvas corner also serves every smaller "
                         f"size, ids and all")
    ap.add_argument("--pages", default="a4,letter",
                    help=f"comma separated; known: {','.join(PAGES)}")
    ap.add_argument("--margin", type=float, default=PRINT_MARGIN_MM,
                    help="unprintable page border to stay out of, mm")
    ap.add_argument("--overlap", type=float, default=OVERLAP_MM,
                    help="how far a sheet laps over its neighbour, mm")
    ap.add_argument("--rev", type=int, default=TEMPLATE_REV, choices=(1, TEMPLATE_REV),
                    help=f"which id scheme to print (default {TEMPLATE_REV}). "
                         f"1 reproduces the original sheets ({', '.join(legacy_boards())} "
                         f"only, ids row-major from the top-left, a private range "
                         f"per board) - for replacing a damaged sheet of a board "
                         f"already on the wall, which is then read with "
                         f"--board-rev 1. Files get a -rev1 in the name.")
    ap.add_argument("--check", action="store_true",
                    help="after writing, read every tile back through the "
                         "detector and verify the board is fully covered "
                         "(needs pdftoppm)")
    args = ap.parse_args()
    if args.boards.strip().lower() == "all":
        boards = legacy_boards() if args.rev == 1 else list(BOARDS)
    else:
        boards = [b for b in args.boards.split(",") if b]
    pages = [p for p in args.pages.split(",") if p]
    generate_all(args.out, boards, pages, args.margin, args.overlap, rev=args.rev)
    if args.check:
        good = check_ids() if args.rev == TEMPLATE_REV else True
        raise SystemExit(
            0 if check(args.out, boards, pages, rev=args.rev) and good else 1)


if __name__ == "__main__":
    main()
