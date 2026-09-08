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

On the solve page the field plot is the whole point, so it gets the room:

* **Zoom and pan.** The wheel zooms about the cursor, a left-drag pans, a
  double-click goes back to the preset view, and `+` / `-` / **Reset view** do
  the same from the keyboard-free route. The preset views (body, near field,
  wake, far field) are still there as a starting point. Whatever view you are
  looking at survives the redraws, so you can zoom into a wake and watch it
  develop rather than being thrown back to the preset every twenty iterations.
* **Fill window** hides the residuals and results panel and gives the whole page
  to the field.
* **Colour map.** *Automatic* uses a red-blue map for the signed fields and
  viridis for the rest; red-blue, cool-warm, plasma, turbo and greyscale can be
  chosen explicitly for any field.
* **Replay.** Snapshots taken during the run are kept, and when the solver stops
  the run plays back: play/pause, a slider to scrub with, a speed control and a
  loop. The colour range is pinned to the final frame during a replay, so what
  moves is the solution rather than the colour bar. The buffer is capped by
  memory, and thins itself by dropping every second frame, so a long run on a
  fine mesh replays the whole run at a coarser interval instead of exhausting
  memory.

## Status, honestly

| Part | State |
|---|---|
| Geometry: NACA 4-digit, circle, square, DXF import | working, tested |
| Body-fitted O-grid mesher | working, tested |
| Finite-volume discretisation | working; the operators are second order on orthogonal, stretched and non-orthogonal meshes (manufactured solution). The *solved* order of `Cd` on the cylinder is 1.26 -- see below |
| Laminar Navier-Stokes | **validated** against published cylinder benchmarks |
| k-omega SST | working; NACA 0012 at Re 2e6 converges to 9.85e-07 in 479 iterations on factory defaults |
| Qt front end | working, tested |

**The turbulence model was blamed for four bugs that were not in it.** The
symptom used to be a NACA 0012 at Re = 2e6 diverging around iteration 350, with
`k` growing outside the boundary layer, and it looked exactly like a turbulence
closure failing. It was not. The test that settled it: freeze the eddy viscosity
at a *constant* 1e5 times molecular, so there is no turbulence model in the loop
at all and the effective Reynolds number is 20 -- creeping flow, where nothing
physical can go wrong. On a circle the solver converged monotonically. On the
aerofoil, same mesher, same spacing, same everything, it diverged at iteration
230 with velocities reaching 1e12. The difference was the mesh.

What was actually wrong, in the order it mattered:

1. **The O-grid had a 68-degree kink in it.** The hyperbolic march hands over to
   an analytic polar far field, and the handover matched position but not
   direction: the march arrives along the surface normal and the polar
   construction leaves along a ray from the body centroid. On a circle those are
   the same direction, so every test passed. On an aerofoil the face normal
   rotated 68 degrees across one layer and the outer half of the mesh carried 18
   degrees of non-orthogonality on average.
2. **The march stopped two thirds of a chord early**, which is what put so much
   of the mesh into that far field. Its guard against sawtooth cell collapse
   measured widths with the same central difference the metric uses -- and a
   central difference cannot see a sawtooth, which is the entire reason the
   fourth-difference smoothing term exists. It tripped on smooth stretching at
   layer 48 and would have sailed past the real collision at layer 54.
3. **The `omega` residual was measured on a row that is thrown away.** `omega` in
   the wall cell is prescribed, not solved, and its row is replaced with the
   identity before the solve -- but after the residual was taken. That row is the
   stiffest in the mesh and carried 99.6% of the reported imbalance: the number
   printed was 9.2e-2 while the system being solved stood at 1.9e-4. Since the
   convergence test takes the worst of all residuals, **no run could ever report
   convergence**, whatever the physics was doing.
4. **The far-field flux correction was assembled and never applied.** The
   pressure equation put a diagonal entry on every outflow face asserting a flux
   correction through it; the flux update did not make it. 62% of all the mass
   imbalance left after each pressure correction sat in that one row of cells.

With those fixed, the constant-viscosity cases converge -- Re_eff = 20
monotonically, Re_eff = 2000 to 1.6e-5 -- and the NACA 0012 at Re = 2e6 with SST
reaches 8.3e-4 by iteration 200 with a peak eddy-viscosity ratio of 90 where the
flat-plate estimate is 84.

**That case now converges, and two earlier descriptions of it here were wrong.**
On factory defaults with the divergence monitor armed it reaches 9.85e-07 at
iteration 479, with `Cd` = 0.009076, `Cl` = -5e-5 against the zero symmetry
requires, and an eddy-viscosity ratio steady at 117.7.

An earlier version of this file reported a bounded limit cycle around 1e-2 that
never settled. A later one reported a residual excursion to 1.8e-1 near iteration
400 followed by a recovery, and said the divergence monitor stopped the run at
iteration 367 in the middle of it -- "a false positive on the primary use case and
the first thing to fix". Neither reproduces. The 2026-09-07 physics audit ran the
case on that same commit with the monitor armed and got 1600 iterations without an
exception and a post-transient peak of 7.45e-03; the monitor's own trip condition
requires a ratio of 100 against a run that peaks at 1.8. The item has been
withdrawn from the hardening plan, and the monitor is left alone until a real
false positive appears.

What was real on that case was a residual plateau near 4e-05 that never reached
tolerance, so the run would exhaust `max_iterations` while its forces sat steady in
the sixth decimal. Its cause was the pressure correction dropping the
non-orthogonal cross term of `(grad p')_f . S`, and with that restored the plateau
is gone. `Cd` = 0.0091 against a published 0.008 is a separate matter, and is what
transition modelling is for.

So: the turbulence model's algebra checks out against every analytic property it
is derived from, and the coupled iteration converges when it is allowed to. The
laminar path is a different matter, and is validated below.

## Validation

`validation/cylinder.py` solves steady flow past a circular cylinder and compares
against the published benchmarks. Nothing is tuned to hit these.

```
.\.venv\Scripts\python.exe -m validation.cylinder
```

| | computed | published |
|---|---|---|
| Re = 20, Cd | 2.027 | 2.00 - 2.09 |
| Re = 20, wake L/D | 0.933 | 0.91 - 0.94 |
| Re = 20, separation from rear | 43.6 deg | 43 - 45 deg |
| Re = 40, Cd | 1.516 | 1.50 - 1.58 |
| Re = 40, wake L/D | 2.122 | 2.13 measured, 2.21 - 2.35 computed |
| Re = 40, separation from rear | 53.7 deg | 52 - 54 deg |

Lift comes out identically zero, as symmetry requires. The Re = 40 drag splits as
0.994 pressure and 0.522 friction, against a published split of roughly 0.99 and
0.53.

`Cd` moved from 1.514 to 1.516 during the response to the 2026-09-07 physics
audit, by two deliberate changes measured separately: the wall-pressure
reconstruction (+1.955e-03), which removed a first-order term from the force
integral, and the Rhie-Chow mobility (-2.738e-04), which removed the converged
answer's dependence on the velocity relaxation factor. Both are argued in
`docs/audit-response-plan.md`.

**Read those figures with the discretisation uncertainty in mind, which the table
does not yet carry.** The 2026-09-07 physics audit ran the first grid-convergence
study this project has had, on three systematically refined cylinder meshes, and
measured the observed order of `Cd` at **1.261** with a Richardson limit of
1.515358 -- measured on the code as it stood before that response, where the
reported 1.5142 sat about 0.08% below the solver's own continuum answer. Two
first-order terms have been removed since and the study has not been re-run, so
the observed order today is unmeasured. The wake length has not converged at all: observed order 0.735,
limit 2.200. The reported 2.1219 agrees with Coutanceau and Bouard's measured
2.13 because the mesh is coarse, and the solver's own grid-converged answer sits
with the computations at 2.20. That is a coincidence being read as agreement.

`validation/convergence.py` now computes observed order, Richardson limit and the
ASME (Celik et al. 2008) GCI from a mesh family, so those numbers can be produced
rather than quoted:

```
.\.venv\Scripts\python.exe -m validation.convergence
```

The order is measured, never assumed: a GCI computed at the formal order of 2
rather than the observed 1.26 comes out 1.87 times narrower, and reporting the
narrower one is not the conservative choice.

`validation/aerofoil.py` is a second gate -- a NACA 2412 at 5 degrees with SST.
It is **not** a validation case and certifies nothing; its job is to notice
change. The cylinder cannot do that job alone, because its mesh is orthogonal to
0.0000 degrees and it carries no circulation, which makes it structurally blind
to any error proportional to either.

## How it works

**Geometry** (`fluidsolver/geometry/`) produces a closed, validated contour from
whichever source. The resampler builds a sizing field from surface curvature --
no cell may turn through more than a few degrees -- and splits the loop at
detected corners so they survive exactly.

**Meshing** (`fluidsolver/mesh/`) marches an O-grid outward from the wall by
imposing orthogonality and a prescribed cell area, solved implicitly as a
periodic block-banded system. On a circle it reproduces the exact concentric
answer to machine precision. The march is stopped once neighbouring cells within
a layer start to collide, and the far field is completed analytically by
interpolating to a circle in polar coordinates -- a construction that cannot
fold. The first cell height is set from the target y+, because that is what the
turbulence model's wall condition requires.

The handover between the two is the delicate part, and getting it wrong cost this
project the whole turbulence model. Polar interpolation matches the marched
layer's position but not the direction it was travelling, and on any body that is
not a circle those differ. The mitigation is to march as far as the grid stays
clean and to relax the angular distribution on log radius, so the turn is spread
evenly over the layers rather than crammed into the seam. Some non-orthogonality
at the transition is inherent; the quality report now says how much of the mesh
carries it, not just how bad the single worst face is.

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

316 tests, about 410 seconds. The centrepiece is a method-of-manufactured-
solutions check on the discrete operators, which measures their *order of
accuracy* rather than their error: diffusion and the high-order convection
schemes come out second order, upwind first, which is what each is by
construction. A scheme that is second order on paper and first order in practice
has a bug, and this is the only test that says so.

It now runs on three mesh families rather than one. The original is an
orthogonal, unstretched circle -- non-orthogonality 0.0000 degrees mean and peak,
aspect ratio 2.5 -- which is not the kind of mesh the solver runs on. A stretched
family and an analytically sheared one (36.5 degrees of non-orthogonality, held
constant under refinement) were added, and the interior order survives both:
1.87 to 1.94 across diffusion, convection and the two together. The boundary rows
are now measured rather than excluded, and they are *zeroth* order, not the first
order previously claimed -- see `tests/test_solver.py`.

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
docs/handover.md        start here: scope, how to run it, and the known traps
docs/hardening-plan.md  what is done, what was tried and failed, what is left
docs/compressible.md    what a compressible extension would involve
docs/optional-deps.md   pyamg, and why it is not required
```
