# fluidsolver

A 2-D incompressible RANS solver with a Qt front end. Everything is solved from
first principles -- a finite-volume discretisation of the RANS equations on a
body-fitted mesh -- rather than by wrapping an existing CFD package.

Set up, one command at a time. On Windows (PowerShell):

```
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m fluidsolver
```

On macOS or Linux:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fluidsolver
```

These are listed separately rather than chained with `&&` on purpose: Windows
PowerShell 5.1, which is what ships with Windows 10 and 11, has no `&&` operator
and fails with a parser error. Its separator is `;`, or `A; if ($?) { B }` to
continue only on success.

Python 3.11 to 3.14 is the tested range; 3.10 is the floor, set by numpy and
scipy rather than by anything in this code. The front end is a desktop window, so
it needs a real graphical session -- it will not run over plain SSH.

The window walks through five steps: **fluid and flow**, **body**, **mesh**,
**model and numerics**, **solve**. The last runs the case in a background thread
and updates the field, the residuals and the force coefficients as it converges.

## Status, honestly

| Part | State |
|---|---|
| Geometry: NACA 4-digit, circle, square, DXF import | working, tested |
| Body-fitted O-grid mesher | working, tested |
| Finite-volume discretisation | working, second order (verified by manufactured solution) |
| Laminar Navier-Stokes | **validated** against published cylinder benchmarks |
| k-omega SST | implemented and unit-tested, but **does not converge** -- see below |
| Qt front end | working, tested |

**The turbulence model is the honest gap.** Its algebra is implemented and
checked against the analytic properties it is derived from: the log-layer eddy
viscosity, the blending functions, the constants' defining relation. For the
first few hundred iterations it produces sensible results on a NACA 0012 at
Re = 2e6 -- skin friction within a few per cent of the flat-plate correlation, a
y+ distribution near 1, an eddy-viscosity peak in the trailing-edge boundary
layer, `Cd` around 0.010 against a published 0.008.

It then stops improving. Around iteration 350 the residuals stall and the
solution enters a limit cycle; sometimes it diverges outright. This was measured
across four configurations -- NACA 0012 at Re = 2e6 and at 2e5, a cylinder at
Re = 2e6, and a coarser y+ = 30 mesh -- and **none of them converged**, the best
residual reached being about 1e-2. It is a general failure of the coupled
iteration, not something peculiar to one geometry or Reynolds number.

The symptom is a slow growth of `k` outside the boundary layer, which raises the
eddy viscosity, which raises turbulent diffusion, which spreads `k` further. The
stagnation-point anomaly at the leading edge feeds it. Menter's production
limiter, the Kato-Launder production form, under-relaxation of the eddy viscosity
and a conservation-consistent convective term each helped, and none was
sufficient.

Do not use the turbulence model for numbers you intend to rely on. The laminar
path is a different matter, and is validated below.

## Validation

`validation/cylinder.py` solves steady flow past a circular cylinder and compares
against the published benchmarks. Nothing is tuned to hit these.

```
.\.venv\Scripts\python.exe -m validation.cylinder
```

| | computed | published |
|---|---|---|
| Re = 20, Cd | 2.023 | 2.00 - 2.09 |
| Re = 20, wake L/D | 0.933 | 0.91 - 0.94 |
| Re = 20, separation from rear | 43.6 deg | 43 - 45 deg |
| Re = 40, Cd | 1.514 | 1.50 - 1.58 |
| Re = 40, wake L/D | 2.122 | 2.13 measured, 2.21 - 2.35 computed |
| Re = 40, separation from rear | 53.7 deg | 52 - 54 deg |

Lift comes out identically zero, as symmetry requires. The Re = 40 drag splits as
0.993 pressure and 0.522 friction, against a published split of roughly 0.99 and
0.53.

## How it works

**Geometry** (`fluidsolver/geometry/`) produces a closed, validated contour from
whichever source. The resampler builds a sizing field from surface curvature --
no cell may turn through more than a few degrees -- and splits the loop at
detected corners so they survive exactly.

**Meshing** (`fluidsolver/mesh/`) marches an O-grid outward from the wall by
imposing orthogonality and a prescribed cell area, solved implicitly as a
periodic block-banded system. On a circle it reproduces the exact concentric
answer to machine precision. The march is stopped once cell quality starts to
degrade, and the far field is completed analytically by interpolating to a circle
in polar coordinates -- a construction that cannot fold. The first cell height is
set from the target y+, because that is what the turbulence model's wall
condition requires.

**Discretisation** (`fluidsolver/solver/`) is cell-centred finite volume.
Convection uses deferred correction: upwind in the matrix for stability, the
high-order difference lagged in the source for accuracy. Diffusion splits into an
implicit orthogonal part and an explicit non-orthogonal correction. Gradients are
weighted least squares, exact for a linear field on any mesh -- Green-Gauss is
not usable here, because on a boundary-layer cell its skewness error enters
divided by the cell volume.

**Pressure-velocity coupling** is SIMPLE with Rhie-Chow interpolation on the
collocated arrangement. The far field is characteristic: each face decides, on
the sign of `u . n`, whether it fixes velocity or pressure.

## Testing

```
.\.venv\Scripts\python.exe -m pytest -q
```

180 tests. The centrepiece is a method-of-manufactured-solutions check on the
discrete operators, which measures their *order of accuracy* rather than their
error: diffusion and the high-order convection schemes come out second order,
upwind first, which is what each is by construction. A scheme that is second
order on paper and first order in practice has a bug, and this is the only test
that says so.

Many tests are regressions for specific defects found during development, and
each records which. Those are worth reading -- they are the parts of a CFD code
that fail quietly rather than loudly.

## Layout

```
fluidsolver/geometry/   closed contours: NACA, circle, square, DXF
fluidsolver/mesh/       O-grid generation, finite-volume metrics, quality
fluidsolver/solver/     discretisation, SIMPLE, turbulence, post-processing
fluidsolver/gui/        PySide6 front end (imports the solver; never the reverse)
validation/             benchmarks the solver must reproduce
docs/compressible.md    what a compressible extension would involve
docs/optional-deps.md   pyamg, and why it is not required
```
