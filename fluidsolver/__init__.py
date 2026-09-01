"""fluidsolver -- a 2-D incompressible RANS solver with a Qt front end.

The package is layered so each level can be tested without the one above it:

    geometry/  closed body contours (NACA, circle, square, DXF import)
    mesh/      body-fitted structured O-grid generation and finite-volume metrics
    solver/    cell-centred finite-volume RANS discretisation and the SIMPLE loop
    gui/       PySide6 front end; imports the layers below, never the reverse
"""

__version__ = "0.1.0"
