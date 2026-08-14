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

         python artprojector.py calibrate --target grid --board 12x16              --grid-anchor-x <minus your measurement>              --grid-anchor-y <minus your measurement>

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
