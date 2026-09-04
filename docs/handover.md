# Handover

Read this first, then [`hardening-plan.md`](hardening-plan.md) for what has been
done, what was tried and failed, and what is left. This file covers the things
that are not recoverable from the code or the git history: what the project is
for, how it is to be worked on, and where the traps are.

## What this is

`fluidsolver` is a 2-D incompressible RANS solver with a Qt front end, written
from first principles -- a finite-volume discretisation on a body-fitted
structured mesh -- rather than wrapping an existing CFD package.

The goal is a solver whose answers can be defended: consistent and trustworthy
enough to be used the way a commercial code is used, within its stated scope.

**In scope.** External aerodynamics -- aerofoils and wing sections, bluff bodies
and separated flow, low-Reynolds-number cases. 2-D only. Self-contained and
standalone: numpy, scipy and PyQt, nothing that has to be fetched or compiled at
run time.

**Out of scope**, and this list is deliberate rather than a backlog: three
dimensions, compressible and transonic flow, multiphase, combustion, radiation,
unstructured or overset meshing, parallel or GPU execution, LES and DES,
conjugate heat transfer, and any third-party mesh generator as a dependency.

**Transition modelling is required, not optional.** Drag below Re 1e6 is a case
the solver has to get right, and fully turbulent SST structurally cannot -- it
has no way to represent a laminar separation bubble.

**"Validated" means the published standards**, not a bar invented here: ASME
V&V 20-2009 for validation, Celik et al. 2008 for grid-convergence uncertainty,
the NASA Turbulence Modeling Resource for model verification. See Stage 7.

## How the work is to be done

These are standing instructions from the owner, and they have shaped every
decision so far.

- **Found every decision on mathematics and physics.** Not on what makes a test
  pass, and not on what a plausible-sounding default would be. Where the code
  departs from a published model, that departure is argued for in the commit
  message and in the plan.
- **No cheap or quick fixes.** Do it right once. If the right fix is large, the
  right fix is still the one to make.
- **Robustness before features.** A wrong answer delivered confidently is worse
  than a refusal, which is why `solver/health.py` refuses hopeless cases before
  spending time on them and `solver/guard.py` bounds and diagnoses the rest.
- **Measure; do not estimate.** Every figure in the plan and in the commit
  messages was measured. This one has been learned the hard way more than once --
  see the traps below.
- **Record negative results as plainly as successes.** Several approaches in the
  plan are written up precisely because they failed, so the ground is not covered
  twice. Stage 1 shipped switched off for this reason.

## Running it

The virtual environment lives at the repository root, outside any worktree:

```bash
C:/AI_CFD_Analysis/.venv/Scripts/python.exe -m pytest -q
```

252 tests, about 100 seconds. The GUI needs a real graphical session:

```bash
C:/AI_CFD_Analysis/.venv/Scripts/python.exe -m fluidsolver
```

**The regression gate.** Laminar flow over a cylinder at Re 40 -- the
best-documented benchmark in incompressible CFD, and nothing in the code is
tuned to hit it:

```bash
C:/AI_CFD_Analysis/.venv/Scripts/python.exe -m validation.cylinder
```

```
Cd 1.5142   (literature 1.50 - 1.58)
wake L/D 2.1219   (2.10 - 2.35)
separation 53.717 deg from the rear   (52.0 - 54.0)
Cl -0.00000   residual 9.96e-08
```

Those numbers have been unchanged through every stage so far. **Run this after
any change to the solver core** -- it is the first thing to check, and it has
caught real regressions, including a wall treatment leaking into laminar runs
where it had no business being. The three quantities test different things:
drag integrates pressure and shear over the surface, wake length tests the
momentum balance away from the wall, separation angle tests near-wall shear.

Note that the environment is Windows with PowerShell 5.1, which has no `&&`
operator. Use `;` or `A; if ($?) { B }`. The Bash tool is available and is
usually the easier route.

## Repository map

```
fluidsolver/
  geometry/   contour, NACA generation, DXF import, primitives
  mesh/       hyperbolic marching, O-grid, spacing, metrics, quality
  solver/
    simple.py       SIMPLE coupling, Rhie-Chow, Numerics, pseudo-transient
    bc.py           boundary conditions; the Stage 3 wall treatment lives here
    case.py         build_case -- meshing, sizing and assembly in one place
    operators.py    convection schemes, diffusion, gradients
    linalg.py       StructuredMatrix, five-band; assumes periodicity in i
    health.py       refuses cases no code could solve
    guard.py        bounds the solution, diagnoses divergence
    post.py         forces, surface data, separation
    turbulence/     laminar, SST 2003
  gui/        Qt front end; five pages, background solve thread
tests/        252 tests
validation/   cylinder.py -- the regression gate
docs/         this file, hardening-plan.md, compressible.md, optional-deps.md
```

## Traps

Things that have already cost time, in rough order of how expensive they were.

- **Do not claim a result before the run returns.** Twice a conclusion was stated
  ahead of the measurement and twice the measurement contradicted it: once
  asserting the cylinder had no steady solution, when relaxation of 0.5/0.2
  reaches 9.2e-04; once attributing a convergence fix to domain size, when a
  control run showed relaxation did all of it. Wait for the number.
- **Tests written from the same wrong assumption as the code will pass.** The
  divergence monitor demanded a tenfold residual rise and its unit tests fed it a
  fivefold-per-iteration climb, so both agreed while the monitor sat through 900
  iterations of a real divergence without objecting. Where a threshold encodes a
  belief about physical behaviour, test it end to end against a real case.
- **Bounds must be sized from measured healthy behaviour.** The first pressure
  limiter was set inside the healthy band and clipped 126 cells on the first
  iteration of the benchmark.
- **March depth *is* mesh quality.** Every wall-normal layer the hyperbolic march
  gives up is built instead by the polar blend. An improvement that shortens the
  march is not an improvement; this is what sank adaptive dissipation.
- **A C0 corner does not resolve.** Refining the surface at a trailing edge
  leaves the turning angle unchanged and raises the effective curvature, so the
  march gets worse, not better.
- **`ndarray.ptp()` was removed in NumPy 2.0.** Use `np.ptp(a)`.
- **Check what the model is actually specified on.** Menter's "wall shear varies
  under 2%" is Couette flow. Applied to an aerofoil it is not a like-for-like
  target, because changing the y+ target there changes the boundary-layer
  resolution as well as the wall condition. This produced an acceptance criterion
  Stage 3 could not meet and should never have been asked to.

## Where things stand

Stages 0, 1, 2, 3 and the first part of 6 are done and on `main` or in flight;
the plan's status table is authoritative. `docs/hardening-plan.md` carries the
measured results for each.

### First: the divergence monitor aborts the primary use case

A NACA 0012 at Re 2e6 with SST on factory defaults stops at iteration 367 with
`SolverDiverged`. It is not diverging. Measured with the monitor disarmed, the
residual peaks at 1.8e-01 around iteration 400 as the eddy-viscosity ratio passes
100, then recovers monotonically:

```
iterations    median residual
 300 -  500      2.68e-02   (peak 1.78e-01)
 500 -  800      2.53e-03
 800 - 1100      4.73e-05
1100 - 1500      2.81e-05      Cd 0.009487 +/- 2e-6, Cl -8e-6
```

The clause that fires is `_MONITOR_LOST` in `solver/guard.py`: the residual is
more than a hundred times the best the run had reached by then. The flaw is in
what "best" means. A SIMPLE run's early residual minimum is a transient artefact,
not a standard the run should be held to for the rest of its life -- here the
"best" it is measured against is a passing 5.6e-04 at iteration 200, long before
the turbulence field has developed. So a case that recovers completely is
indistinguishable, to the monitor, from one that is failing.

This is a Stage 2 regression that only became visible once Stages 0 and 3 made
the case recoverable, which is why it was not caught then.

**Do not just move the threshold.** That is how the monitor was mis-set the first
time. The right fix needs measurement across several cases -- what excursion size
and duration is actually recoverable, and whether the reference should be a
trailing window rather than the all-time best -- and both the recovering case
above and a genuinely diverging one (the laminar cylinder at Re 2e6, which grinds
upward at about 1.3% per iteration) as end-to-end tests.

### Then: Stage 6's remainder, which gates what follows

1. **The marched-to-analytic seam** -- the worst region left on an aerofoil mesh,
   a single face at 60.07 degrees, because the handover between the two
   constructions happens in one layer. Blend it over several.
2. **Wake refinement** -- an O-grid falls below four cells per diameter about two
   diameters downstream, so a shed vortex is smeared within a couple of its own
   spacings. **Stage 4 cannot work until this is fixed**; there is no point
   solving for a Karman street the mesh will not carry.
3. **C-grid topology** for bodies with a trailing edge -- the structural fix for
   the wake. Costed already, and it is solver work rather than only mesher work:
   63 uses of periodic `np.roll` across ten modules, and `StructuredMatrix`'s
   five-band assumption breaks, because a wake cut makes cell `(i, 0)` a
   neighbour of `(Ni-1-i, 0)`, far apart in the `k = i*Nj + j` ordering.

Then Stage 4 (URANS and vortex shedding), Stage 5 (Spalart-Allmaras, k-epsilon,
the one-equation gamma transition model) and Stage 7 (MMS, the NASA TMR cases,
GCI, and geometric multigrid with line-implicit smoothing).
