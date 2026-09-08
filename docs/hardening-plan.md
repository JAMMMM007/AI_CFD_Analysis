# Hardening plan: state and remaining work

A staged programme to take the solver from "converges on the easy cases" to one
whose answers can be defended. Scope is 2-D, self-contained, targeting external
aerodynamics: aerofoils, bluff bodies and low-Reynolds-number studies.

Every figure below was measured, not estimated. Where a stage produced a
negative result that is recorded as plainly as the successes, because the point
of the record is to stop the same ground being covered twice.

## Status at a glance

| Stage | What | State |
|---|---|---|
| 0 | Land the SST correction, shading, bounded scheme | **done**, merged (#3) |
| 1 | Pseudo-transient continuation | **done**, merged (#4) — negative result, shipped off |
| 2 | Refuse, bound, diagnose | **done**, merged (#4) |
| 3 | A wall treatment that works above y+ 1 | **done** — validated range measured, see below |
| 4 | URANS and vortex shedding | not started |
| 5 | Turbulence and transition models | not started |
| 6 | Meshing | **part done** — spacing constraint (#5) and the marching Newton; seam, wake and C-grid remain |
| 7 | Verification, validation and speed | **part done** — GCI machinery and a second gate exist; TMR cases remain |
| 8 | Response to the 2026-09-07 physics audit | **in progress** — see `docs/audit-response-plan.md` |

316 tests. Regression gate: cylinder at Re 40 gives Cd 1.5161, wake 2.1219 D,
separation 53.718 degrees, Cl -0.00000, residual 9.98e-08. `Cd` moved from 1.5142
by two deliberate changes -- the wall-pressure reconstruction (+1.955e-03) and the
Rhie-Chow mobility (-2.738e-04) -- each measured on its own and recorded with the
commit that made it. It is the first thing to check after any change, and it is now
checked alongside `validation/aerofoil.py`, which can see seven findings the
cylinder structurally cannot.

**The open defect that headed this list has been withdrawn.** Stage 2's
divergence monitor was recorded here, in `README.md` and in `docs/handover.md` as
stopping a NACA 0012 at Re 2e6 with SST at iteration 367, blocking the primary use
case, and as the first thing to fix. It does not reproduce. The physics audit
could not reproduce it on `main` at 37fabd2, and a direct run of that case here
converged to 9.85e-07 at iteration 479 on factory defaults with the monitor armed.
The monitor is left alone until a real false positive appears. What the same case
*did* have -- a residual plateau near 4e-05 that never reached tolerance -- was
real, and it is gone: its cause was the missing non-orthogonal term in the
pressure correction. See Stage 8.

---

## Done

### Stage 0 — bounded convection, and the SST correction

The default momentum scheme was unbounded central differencing, labelled
"recommended" in the interface. Central differencing is unbounded above a cell
Peclet number of 2, and an external-aerodynamics mesh is nowhere near that
outside the boundary layer: on a 240x99 cylinder mesh at Re 2e6 the median cell
Peclet number is 3.2e4 and the peak 2.6e7. A sawtooth seeded at the mesh seam
grew unchecked and killed runs, having passed through a lift coefficient of
37525 on the way.

Default is now `limited_linear`. The NACA 2412 at 15 degrees converges to
4.6e-05 on factory defaults; it used to diverge.

Also landed: the k production term now follows Menter 2003 equation (5),
`mu_t S^2`, in place of the Kato-Launder `mu_t S Omega` that was there. Omega is
exactly zero on a stagnation streamline by symmetry, so the old form put a line
of zero production along it — k came out five orders of magnitude low on the
stagnation ray — and Menter's production limiter, the mechanism the paper
specifies for exactly that problem, never activated anywhere in the field. It
now fires in about 7% of cells. `alpha_1` and `alpha_2` are the published 5/9
and 0.44.

Field plots use gouraud shading. Flat shading paints each finite-volume cell one
colour, which on a polar O-grid draws a smooth field as concentric rings and a
smooth wake as radial spokes; neither is in the solution, and both were being
read as solver artefacts. Flat remains available behind a **Shading** control,
because seeing the actual cells is what answers whether an oscillation is real.

### Stage 1 — pseudo-transient continuation, and why it is off

Added expecting a robustness win. It does not deliver one for a steady
segregated solver, and `Numerics.pseudo_transient` defaults to `False`.

```
NACA 2412, 5 deg attached, iterations to reach 1e-6
    relaxation only 0.7/0.3                        1010
    pseudo-transient, cfl_max 10/30/100/300   1155/1060/1032/1023

cylinder Re 2.03e6, SST, 800 iterations, final residual
    relaxation only 0.7/0.3                    3.47e-02
    relaxation only 0.5/0.2 (hand-tuned)       1.57e-02
    pseudo-transient 0.7/0.3                   4.00e-02
```

It converges *towards* plain relaxation as the ceiling rises and never past it,
and on the bluff body it is worse than doing nothing. The cause is structural:
the time step is convection-only, which leaves near-wall cells essentially
undamped, and on a cylinder the trouble is the shedding mode working through the
boundary layer — precisely the region it declines to touch. It cannot stand
alone either: `relax_velocity = 1.0` diverged at every ceiling tried, down to
`cfl_max = 2`.

It is kept because **Stage 4 needs exactly this term** with a global physical
time step in place of the local pseudo one, and because it is verified: the
fixed point is preserved to 0.0055% in Cl at a residual of 1e-9.

One trap for anyone revisiting it. The convection-only step is load-bearing.
Pair it with a viscous limit and `dtau` becomes `CFL rho V / a_P`, so the
effective relaxation is `CFL/(1+CFL)` in *every* cell — global under-relaxation
with a renamed knob. A test asserts the resulting spread (alpha_eff above 0.9 at
the wall, below 0.75 in the far field) so that collapse cannot creep back in.

### Stage 2 — refuse, bound, diagnose

The solver could be handed a case no code could solve, spend 459 iterations on
it, and die inside the linear solver with nothing to say about where or why.

**Refuse** (`solver/health.py`) judges mesh, fluid and model against each other
before any time is spent — a separate question from whether the mesh is well
formed, which `mesh/quality.py` already answers. Thresholds are measured:
sweeping only Reynolds number, the laminar cylinder converges at a median cell
Peclet of 98, loses symmetry at 328, and diverges at 32000. The 328 case is
allowed through deliberately — that is the cylinder genuinely shedding, not the
discretisation failing. The Peclet test is laminar-only: a turbulent case runs
at high *molecular* Peclet because the eddy viscosity supplies the diffusion.
`allow_unhealthy=True` opens the door for a deliberate laminar reference.

**Bound and diagnose** (`solver/guard.py`) applies generous physical limits
before the forces are read, so a runaway cell cannot produce a nonsense
coefficient on its way out. Velocity clipping scales the vector rather than the
components, so the flow keeps its direction. The counting matters more than the
clipping: a limiter active on a fifth of iterations means the numbers are
worthless, and the report says so.

Both bounds are sized from measurement, and the first attempt at the pressure
one was wrong. Healthy cold-start peaks are |Cp| of 123.5 on the Re 40 cylinder,
50.4 on the cylinder at Re 2e6 and 30.3 on a NACA 2412 at 15 degrees; a cap of
100 dynamic heads clipped 126 cells on the first iteration of the Re 40
benchmark. Raised to 1000, two orders below the ~18000 the diverging case
carried.

The divergence monitor was also wrong first time, and only an end-to-end test
showed it. It demanded a tenfold rise of the trailing mean inside a
fifty-iteration window; real divergences grind upward at about 1.3% per
iteration, which is 1.9x over fifty. It let the laminar cylinder run 900
iterations to a residual of 7.7 without objecting. It now compares trailing
window *medians* — robust to the spikes a mean would trip on — at a 1.5x
threshold, gated behind an absolute floor and a hundredfold loss against the
run's own best.

**It is now known to be wrong in the other direction, and that is the first job
on the list.** See "The divergence monitor's false positive" below.

### Stage 6 (part) — the surface spacing constraint

`surface_points` fixes the tangential spacing, `target_y_plus` fixes the
wall-normal spacing, and nothing connected them. Ask for 240 points at y+ 100
and the trailing-edge cluster comes out six times finer than the first layer, so
every wall cell there is taller than it is wide, the march folds within two
layers, and the polar blend builds almost the whole mesh.

| first layer / tightest spacing | layers marched (of 30) |
|---|---|
| 0.97 | 30 |
| 2.19 | 29 |
| 5.80 | 2 |
| 12.58 | 3 |

`Contour.resample` now takes a `min_spacing` floor and `build_case` sizes the
first layer before resampling.

| y+ | marched | non-orthogonality mean / peak / %>60 | before |
|---|---|---|---|
| 1 | 54/89 | 3.9 / 60.1 / 0.00% | unchanged |
| 5 | 43/78 | 4.4 / 59.4 / 0.00% | unchanged |
| 30 | 37/65 | 3.8 / 48.9 / 0.00% | 10.0 / 75.0 / 2.89% |
| 100 | 30/56 | 4.6 / 30.6 / 0.00% | 24.5 / 84.3 / 17.88% |

Three approaches failed first and are recorded so they are not repeated:
adaptive fourth-difference dissipation (controls the sawtooth, but march depth
*is* mesh quality — every layer given up is built by the polar blend, and it
took the y+ 1 mesh from no faces above 60 degrees to 16.6% of them); refining
the surface to resolve the corner (a C0 corner turns through a fixed angle at
any resolution, so refinement only raises the effective curvature, and the march
got monotonically worse from 52 layers to 26); and tuning the width-ratio and
alternation guards against each other, when they measure the same quantity.

### Stage 3 — the wall treatment above y+ 1

Esch and Menter's automatic wall treatment (IGTC 2003, eqs 15-18): omega blended
as `sqrt(omega_vis^2 + omega_log^2)`; a friction velocity blended as the fourth
root of the sum of fourth powers; the momentum wall shear carried by an
effective wall viscosity `tau_w y1 / U1`, which recovers the low-Re treatment
exactly as `y1 -> 0`; and k switched from a fixed zero to **zero flux**.

A NACA 2412 previously diverged at iteration 137 on a y+ 30 mesh and at 101 on
y+ 100. Both now converge, y+ 100 to 1.4e-10.

The friction velocity is iterated to a fixed point rather than seeded once.
Seeding y+ from the viscous branch alone underestimates `u_tau`, which shrinks
`ln(y+)` and therefore *raises* the logarithmic branch — an overestimate of wall
shear growing with coarsening, +21% at y+ 300. Iterated, +0.6% at y+ 100 and
+0.1% at y+ 300. Five passes; y+ enters only through a logarithm.

**The part not in the paper, which the measurements demanded.** With zero-flux k
and nothing else the case still would not converge. At y+ 30 momentum fell by 86
to 120 times and continuity by 133, while k *rose* from 4.4e-03 to 7.0e-03 and
stuck on the solution limiter — 165 cells clipped on 554 of 600 iterations, k
pinned at the ceiling, mu_t/mu at 2522.

The cause is arithmetic, not modelling. The discrete strain in a wall cell is
`U1/y1`, the *average* gradient between wall and cell centre; the local gradient
in the log layer is `u_tau/(kappa y1)`. Their ratio is `kappa U+`, 5.5 at y+ 30,
and production goes as the square — about thirty times too much. The wall row now
takes its production strain from the two-layer profile

```
dU/dy = min( u_tau^2 / nu , u_tau / (kappa y1) )
```

which picks the viscous branch below y+ 2.4 and the logarithmic one above.
Standard wall-function machinery rather than an invention, and it costs nothing
where it does not apply: 1.001 of the resolved gradient at y+ 1, 0.20 at y+ 30.
Afterwards k falls to 6.3e-08, mu_t/mu to 166, and the limiter never fires.

Note this closes the open question the parked version left. The earlier
suspicion was that the wall-adjacent k cell wanted the log-layer production
`tau_w^2 / (rho kappa u_tau y1)`. That form is right in the log layer but does
not reduce to the resolved value as y+ -> 0, so it would have corrupted
wall-resolved meshes; the `mu_t (dU/dy)^2` form above does reduce, exactly.

**The stage does not meet the acceptance criterion the plan set, and that
criterion was not well posed.** It asked for Cf within 3% across y+ 0.5 to 100,
taking Menter's "under 2%" — which is Couette flow, where the layer is the whole
channel and always resolved. On an aerofoil, changing the y+ target changes the
wall condition *and* the boundary-layer resolution: 35 cells across the layer at
y+ 0.5, 19 at y+ 5, 8 at y+ 30, 3 at y+ 100, where the first cell alone spans
21% of it. No wall condition repairs a profile carried by three cells.

```
y+     achieved      Cl        Cd      Cd_friction   residual
0.5   0.15..  1.18  0.75184  0.012754   0.007635     3.9e-07
1     0.30..  2.36  0.75368  0.012501   0.007454     2.4e-07
5     0.67.. 12.08  0.74818  0.013166   0.007896     9.7e-09
30    5.20.. 68.15  0.77143  0.012171   0.007058     2.0e-10
100  11.04..200.20  0.78065  0.012504   0.005581     1.4e-10
```

What can be defended, measured:

| range | BL cells | Cl | Cd | Cd_friction |
|---|---|---|---|---|
| y+ 0.5 - 5 | >= 19 | 0.73% | 5.19% | 5.77% |
| y+ 0.5 - 30 | >= 8 | 3.07% | 7.87% | 11.16% |
| y+ 0.5 - 100 | >= 3 | 4.27% | 7.88% | 32.49% |

Lift is insensitive to near-wall spacing below 1% where the layer is resolved,
drag to about 5%, and the solver is stable and convergent to y+ 100 where it
previously diverged above 5.

Two cautions for anyone re-running this. The five meshes differ in more than
their first cell — 89 wall-normal layers down to 56, peak non-orthogonality 60.2
to 30.6 degrees — so part of the drag spread is mesh, not wall model, and this
experiment cannot separate them. And the treatment is confined to turbulent runs:
a laminar case has no friction velocity and no log layer, and the fourth-power
blend always exceeds the larger of its branches, so applying it there returned
slightly more than molecular viscosity even where the viscous branch was exact.

Note for anyone comparing against sources: Esch and Menter (IGTC 2003) list
`alpha_1 = 0.5532, alpha_2 = 0.4403`, the derived log-layer values, where Menter,
Kuntz and Langtry (2003) state 5/9 and 0.44. This code follows the latter. Same
author, same year, 0.4% apart; the mismatch is deliberate.

### Stage 8 — the response to the 2026-09-07 physics audit

The audit is `docs/physics-audit-2026-09-07.md`; the order of attack and the
argument for it are in `docs/audit-response-plan.md`. What follows is what has
landed and what it measured. Three of its findings were re-derived independently
before anything was implemented, and all three reproduced exactly.

**The instruments came first, because four of the seventeen findings are
invisible to every measurement the project owned.** F4, F13 and F14 are
identically zero on an orthogonal mesh and F2's vortex term is identically zero
on a non-lifting body; the cylinder gate is both, and so are the
manufactured-solution meshes. Four instruments were built and none of them moves
a number:

- a linear-pressure-field identity for the Rhie-Chow flux, which is exact rather
  than asymptotic and needs no solve;
- the manufactured solution on a stretched family and on an analytically sheared
  one holding 36.5 degrees of non-orthogonality constant under refinement, plus a
  separate measurement of the boundary rows;
- `validation/aerofoil.py`, a NACA 2412 at 5 degrees with SST, whose job is to
  detect change rather than to certify accuracy;
- `validation/convergence.py`, Celik's GCI with the observed order and the cell
  size both measured rather than assumed.

**Two things the instruments found that the audit did not.**

The boundary rows of the discrete operator are *zeroth* order, not the first
order this project's own test docstring claimed and the audit repeated. Measured
4.0608e-01, 4.0576e-01, 4.0635e-01 across a refinement by four, an observed order
of +0.001. The derivation agrees: the wall flux error is `O(h)|S| = O(h^2)` and
the operator divides by a volume that is also `O(h^2)`. This does not make the
solution zeroth order at the wall -- for an elliptic operator the boundary
truncation error is damped -- but it does mean the boundary treatment is the
weakest part of the discretisation and that nothing had ever measured it.

The interior order *survives* non-orthogonality and stretching: 1.87 to 1.94
across diffusion, convection and the two together on both new families. That is a
positive result and it narrows F4 rather than widening it -- the operators are
fine, and the first-order term is in the flux definition, which no manufactured
solution evaluates.

**The flux definition.** Rhie-Chow was comparing a compact pressure difference
taken along `d` against an interpolated gradient dotted with the full area vector
`S`. For a linear field, where the damping must be identically zero, that leaves
`-(grad p)_f . T` -- verified against the closed form to 8e-14. Correcting it
alone made the solver *worse*: the NACA 2412 stopped converging and the NACA 0012
diverged at iteration 163, with its fastest cell inside the polar-blended region.
The spurious flux had been paying for a second omission -- the pressure correction
applied only the two-cell part of `(grad p')_f . S` and dropped the cross term
entirely -- and the two errors had been partly cancelling. Both are now fixed, the
cross term lagged into the source in one corrector pass.

A control separates the two halves of that corrector, because the loop both adds
the cross term and runs a second warm-started solve. With the same number of
solves and the cross term forced to zero, the case stops diverging and plateaus at
1.773e-04; with the cross term it converges at 1044 iterations. The extra solve
turns divergence into a plateau, and the cross term turns the plateau into
convergence.

Measured on the NACA 2412, same mesh throughout: `Cd` 0.012506 to 0.011773,
-5.87%, of which 14.5% comes out of the pressure drag while friction drag moves
0.03%. That split is the signature -- the spurious flux lived in the blended far
field and the marched near-wall region is orthogonal to 0.25 degrees.

**Relaxation-independence, which four docstrings asserted and none of them had.**
The Rhie-Chow mobility is built from the under-relaxed momentum diagonal, so
`D_f = alpha_u V / a_P`, and the damping term it multiplies does not vanish at
convergence. Measured on the cylinder at a residual of 1e-9, `Cd` spanned
2.038e-04 across `alpha_u` 0.70 to 0.40 -- while the `alpha_p` control spanned
1.164e-06, which is the convergence floor. Carrying the damping between
iterations (Choi) makes the relaxation cancel out of the fixed point by
construction: the `alpha_u` spread falls to 8.83e-07, *at* the floor and not below
it. Iteration counts are unchanged.

**The marching Newton.** A fixed four passes became a residual test with a
ceiling. At y+ 1 the march reaches all 63 requested near-field layers where it
reached 54, and the fraction of faces past 30 degrees falls from 2.92% to 0.09%.
The peak does not improve -- 60.098 to 60.863 -- because that face is the
marched-to-analytic seam, which is a discontinuity in the construction and needs
its own fix. The cylinder is untouched and *faster*, because a circle marches
exactly and now exits after one pass.

**A negative result on a proposed remedy.** The audit's F3 asks for the wall
pressure to be reconstructed as `p_P + (grad p)_P . d` using the solver's own
least-squares gradient. Implemented and measured, that is order 1.13, not 2, and
the cause is that the least-squares gradient *in the wall row does not converge at
all* -- observed order 0.00. The stencil weights go as `1/|d|^2` and the wall face
is the nearest stencil point, so a wall value asserting zero normal gradient is
the most heavily weighted member of the fit, and the reconstruction inherits the
error of the assumption it exists to remove. A linear extrapolation along the wall
normal through the first two cell centres, which needs no gradient, measures 2.05
to 2.13 on all three families and is what landed.

**Also landed:** the `omega` residual now excludes the identity-substituted wall
row, which was 56.89% of its normaliser and 65.54% of its imbalance; `Cm` is
reported nose-up positive; separation takes its sign from a one-sided wall
gradient rather than the first cell, worth 0.253 degrees; and the mesh page no
longer calls a flat-plate correlation the achieved `y+`.

**Landed since:** F14 and F15 (which turned out to be the same defect), F6, F9,
F2, F13 and F7.

**F2 is the largest of them and it corrects the audit's own formula.** A lifting
body's bound circulation induces a `1/r` velocity that the bare freestream
condition imposes away over an arc of length `2 pi R`, so the spurious flux is
`O(Gamma)` however far out the boundary goes. Superposing the vortex collapses the
sensitivity to the domain size by a factor of 24.7 in `Cl` and 63 in `Cd` between
20 and 40 chords. The audit writes the induced field with `Gamma = +Cl U c / 2`,
which puts the flow *slower* over the suction side; three independent physical
checks -- faster above, upwash ahead, downwash behind -- fail together with that
sign and hold together with the other. Taking it as written would have doubled the
error while still changing the domain sensitivity, just the wrong way.

**F7 corrects the audit's arithmetic as well as the code.** Its suggested ambient
values, `k_amb = 1e-6 U^2` and `omega_amb = 5 U/L`, are outside two of the three
NASA TMR bands the same finding quotes: `k` twenty times the top of its band, and
`mu_t/mu` at 4.06e-01 where the audit states 7.4e-3. The defaults are now
`I = 5e-5` and a ratio of `1e-3`, inside all three with margin, and `SST-sust`
holds them there -- `k` arrives at the body at 1.0000 of its set value against
0.0423 without. Since no fixed default can satisfy bands that scale differently
from the parameterisation, `health.assess` now measures the case actually being
run against them.

**Still open:** F1, which is blocked rather than deferred -- deciding the `omega`
wall constant needs the NASA TMR `2DZP` flat plate, and this mesher builds O-grids
around closed contours, so a flat plate needs rectangular structured meshing that
does not exist here. K2's seam, and the grid-convergence study on a properly
refined family, also remain.

**One thing this response created.** The NACA 2412 no longer reaches `1e-6`: F5's
correction makes the damping `1/alpha_u` larger and the case plateaus at about
3.4e-06 with the forces steady to five figures. The gate carries a measured
`RESIDUAL_FLOOR` for it, which is an admission rather than a fix, and diagnosing
it is the first thing to do next. The binding equation is `Uy`.

---

## Remaining

### The divergence monitor's false positive — withdrawn, does not reproduce

**This item is closed and the work it was directing should not be done.** It said
that a NACA 0012 at Re 2e6 with SST on factory defaults raises `SolverDiverged` at
iteration 367 on a case that recovers completely, that the residual peaks at
1.8e-01 near iteration 400, and that this blocks the primary use case and comes
before the meshing work.

Measured twice, on two code states, and it does not happen. The physics audit ran
it on `main` at 37fabd2 with the monitor armed: 1600 iterations, no exception, no
excursion, largest residual after iteration 100 of 7.45e-03 against a claimed peak
of 1.8e-01. A direct run here on the audit-response branch converged at iteration
479 to 9.85e-07. The plan's own residual table disagrees with measurement by three
orders in its first window and agrees in its last two, which is the signature of a
transient that something has since removed rather than of a different case.

The plan's *diagnosis* of the monitor's design was sound as far as it went, and it
is left recorded: `_MONITOR_LOST` holds a run to the best residual it ever reached,
and an early minimum before the turbulence field has developed is not a
meaningful reference. But the monitor is not close to firing on any case that can
be reproduced -- a shadow monitor over 1600 iterations peaked at a ratio of 1.8
against the 100 it requires -- so there is nothing to tune against. **Do not touch
it until a real false positive appears, and record that case's full specification
when it does.** None of the three documents that carried this item recorded the
angle of attack, surface point count, `y+` target or far-field ratio of the case
it described, and all four change the answer.

There *was* a real defect on that case and it was a different one: the residual
stopped falling at about 4e-05 and oscillated there indefinitely, so the run would
exhaust `max_iterations` rather than converge while the forces sat steady in the
sixth decimal. The audit guessed the mechanism -- "the deferred correction's
lagged source is the usual cause of a plateau at this level" -- and the guess was
right. The pressure correction was dropping the non-orthogonal cross term of
`(grad p')_f . S` entirely. With it restored the same case converges at iteration
479. See Stage 8.

### Stage 6 (rest) — meshing

- **The marched-to-analytic seam.** Still the worst region on an aerofoil mesh:
  the single face at 60.07 degrees on the y+ 1 mesh is it. The handover happens
  in one layer, so non-orthogonality jumps from about 3 degrees to the peak and
  back. Blend the two constructions over several layers instead.
- **Wake refinement.** An O-grid wraps the wake and falls below four cells per
  diameter about two diameters downstream, so a shed vortex is smeared within a
  couple of its own spacings. Stage 4 cannot work on that.
- **C-grid topology for bodies with a trailing edge.** The structural fix for
  the wake, and it removes the trailing edge from the interior of the marched
  line. Keep the O-grid for bluff bodies, where it is optimal — a circle meshes
  at 0.0 degrees non-orthogonality, mean and peak.

  Honest cost, measured: 63 uses of periodic `np.roll` across ten modules, and
  `StructuredMatrix`'s five-band assumption breaks, because a wake cut makes
  cell `(i, 0)` a neighbour of `(Ni-1-i, 0)` — far apart in the `k = i*Nj + j`
  ordering. This is solver work, not only mesher work.
- **Systematically refined mesh families** from one specification, which Stage 7
  needs for grid convergence.

### Stage 4 — URANS and vortex shedding

No steady solver can produce a Karman street: shedding is a travelling wave and
the current equations contain no time. Stage 1 already built most of the
machinery.

Second order in time so the shedding amplitude is not damped by the scheme:

```
BDF2:  (3 phi^n+1 - 4 phi^n + phi^n-1) rho V / (2 dt)
   =>  a_P += 3 rho V / (2 dt)     b += rho V (4 phi^n - phi^n-1) / (2 dt)
```

Two stored time levels in `State`, started with one implicit-Euler step; each
step runs the existing SIMPLE loop as inner iterations. Step size from the
Strouhal number, `dt = 1 / (N f)` with `f = St U / D`, `St ~ 0.2` and 50 to 100
steps per shedding period.

Interface work travels with it: the iteration axis becomes a time axis, force
histories gain an FFT so the Strouhal number can be read off, and the replay
buffer becomes a genuine animation.

**Depends on** wake refinement, or the street will not survive to where it can
be seen.

**Acceptance**: cylinder at Re 100 gives St 0.164 +/- 0.005 and mean Cd
1.35 +/- 0.05; at Re 3900, St about 0.21 and mean Cd 0.98 to 1.05; halving `dt`
changes mean Cd by under 1%.

### Stage 5 — turbulence and transition models

- **Spalart-Allmaras**, negative-`nu~` variant. One equation, markedly more
  robust than SST on a poor mesh, which makes it the right default for a first
  look at a new geometry.
- **k-epsilon with wall functions**, for bluff bodies where the boundary layer
  is not the point. Stage 3 supplies the wall machinery it needs.
- **Transition: the one-equation gamma model** (Menter et al. 2015), *not* the
  four-equation Langtry-Menter. It adds one transport equation instead of two,
  computes the transition-onset momentum-thickness Reynolds number algebraically
  from local variables, and is therefore **Galilean invariant** where the 2009
  model is not. Specified on the NASA Turbulence Modeling Resource as
  `SST-2003-Menter-Gamma-2015`, sitting directly on the `SST-2003` this code
  already implements.

  Required, not optional: fully turbulent SST cannot represent a laminar
  separation bubble, which is why cylinder drag through the critical range and
  aerofoil drag below Re 1e6 come out wrong for structural reasons.

### Stage 7 — verification, validation and speed

"Validated" follows the published standards rather than a bar of our own.

- **Code verification** — method of manufactured solutions; observed order
  within 10% of formal order. The only way to *prove* second-order accuracy on
  the curvilinear mesh rather than assert it.
- **Model verification** — NASA TMR, which publishes authoritative model
  definitions and reference results from previously verified codes. Cases
  `2DZP` (zero-pressure-gradient flat plate), `2DB` (bump-in-channel), `2DANW`
  (airfoil near-wake); validation adds NACA 0012, NACA 4412 with separation, the
  backward-facing step and the wall-mounted hump.
- **Discretisation uncertainty** — ASME Journal of Fluids Engineering procedure
  (Celik et al. 2008):

  ```
  p   = ln( (f3 - f2) / (f2 - f1) ) / ln(r)        r >= 1.3
  GCI = Fs |f2 - f1| / (|f1| (r^p - 1))
        Fs = 1.25 for three or more grids, 3.0 for two
  ```

  Every force coefficient the solver reports should carry one.
- **Validation** — ASME V&V 20-2009: comparison error `E = S - D`, validated
  when `|E| <= u_val`, with `u_val` combining numerical, input and experimental
  uncertainty. Note the consequence: a result is validated *at a stated
  uncertainty level*, not against a fixed percentage.
- **Speed — geometric multigrid, written here.** The pressure correction is a
  Poisson problem and dominates the cost; ILU-preconditioned BiCGSTAB leaves the
  iteration count growing with mesh size. `pyamg` is not the route: no wheel for
  the interpreter in use, no compiler here, and a bought-in algebraic solver cuts
  against the program being self-contained. On a structured grid the mesh
  hierarchy is free.

  One design point decides whether it works. Boundary-layer cells reach aspect
  ratios above 450, and a point smoother on that anisotropy makes multigrid no
  faster than the smoother alone. The fix is line-implicit smoothing along the
  wall-normal direction, and the existing `k = i*Nj + j` ordering already makes
  those lines contiguous in memory and in the matrix — which is precisely the
  property a line solver needs.

---

## Out of scope

Three dimensions; compressible and transonic flow; multiphase, combustion,
radiation; unstructured or overset meshing; parallel or GPU execution; LES and
DES; conjugate heat transfer; third-party mesh generators as a dependency.

The target is a trustworthy 2-D incompressible and low-Mach RANS and URANS
solver with verified numerics and a real validation suite — not a general-purpose
commercial code.
