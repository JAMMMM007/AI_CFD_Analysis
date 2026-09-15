# Breaking the periodic-`i` assumption

The plan required before a structural change, per `docs/agents/solver-engineer.md`
§2. Agreed with the owner: go to (a) and then (b).

## What is assumed today, measured

The solver assumes the `i` direction wraps. Counted: **75 uses of `np.roll` across
ten modules** — `geometry/contour.py` 10, `mesh/hyperbolic.py` 10,
`mesh/metrics.py` 5, `mesh/quality.py` 5, `solver/faces.py` 1,
`solver/health.py` 2, `solver/linalg.py` 6, `solver/operators.py` 16,
`solver/post.py` 3, `solver/simple.py` 9.

It also assumes `j = 0` is a no-slip wall and `j = -1` is the far field, which is
not being changed here.

## What it blocks

| blocked | needs |
|---|---|
| NASA TMR `2DZP`, zero-pressure-gradient flat plate | (a) |
| NASA TMR `2DB`, bump in a channel | (a) |
| **F1 — the `omega` wall constant, 8.94% in `Cd`** | `2DZP`, so (a) |
| NASA TMR `2DANW`, aerofoil near-wake | (b) |
| Wake resolution for Stage 4 | (b) |

F1 is the reason to do this now rather than later. It is the largest single
number the audit found and the only finding left untouched, and everything built
on top of the current wall constant would have to be re-baselined once it moves.

## The design: make `i` look like `j`

The two directions are already asymmetric in exactly the way that matters, and
the asymmetry is the whole problem:

```
face_j_area   (Ni, Nj+1, 2)    Nj-1 interior faces + 2 boundary families
face_i_area   (Ni,   Nj, 2)    Ni interior faces, wrapping
```

`FaceGeometry` already carries `j_faces` (interior) alongside `wall` and
`far_field` (boundary). So the change is not a new mechanism — it is giving `i`
the mechanism `j` already has:

```
face_i_area   (Ni+1, Nj, 2)    Ni-1 interior faces + i_start + i_end
```

That framing matters for review: every place that special-cases `j`'s boundaries
already exists and is tested, and the non-periodic `i` path follows it.

**The five-band matrix survives (a).** A non-periodic `i` removes couplings at
the two ends rather than adding any; `west` at `i = 0` and `east` at `i = Ni-1`
become boundary contributions to the diagonal and the source, exactly as `south`
at `j = 0` and `north` at `j = -1` already do. `StructuredMatrix` keeps its five
bands. This is why (a) is much cheaper than (b), and the hardening plan's costing
of "63 `np.roll` uses and the five-band assumption breaks" bundles the two.

**(b) is where the bands break.** A C-grid's wake cut makes cell `(i, 0)` a
neighbour of `(Ni-1-i, 0)`, and those are far apart in the `k = i*Nj + j`
ordering. That is a sixth and seventh band at a data-dependent offset, and it is
a separate piece of work with its own plan.

## The invariant that protects this

**The periodic path must stay bit-identical at every step.** The cylinder gate is
the check and it is a good one: `Cd 1.5161`, `wake 2.1219`, `separation 53.7183`,
992 iterations. Any movement means the refactor has leaked into the path it was
supposed to leave alone. `periodic=True` is the default everywhere, so an
untouched caller gets untouched behaviour.

## Order of work

1. `Metrics` and `compute_metrics` carry a topology flag and produce the
   `(Ni+1, Nj)` face arrays when `i` is open.
2. `FaceGeometry` gains `i_start` and `i_end`; `i_faces` becomes interior-only.
3. `operators`: gradient stencil, convection, diffusion, divergence.
4. `linalg.StructuredMatrix`: bands stop wrapping.
5. `simple`: `face_fluxes`, `pressure_correction`, `apply_correction`.
6. A rectangular mesh generator, and boundary conditions for the `i` ends —
   inflow, outflow, symmetry.
7. `validation/flat_plate.py`: the TMR `2DZP` case, `Cf(Re_x)` against the
   published distribution.
8. F1 decided on that measurement, not on authority.

Each step keeps both gates green, and the two mesh families and the manufactured
solutions keep running on the periodic path throughout.

## What will legitimately change, predicted before running

Nothing on the periodic path. No number in either gate, no test in `tests/`
whose mesh is an O-grid. If any of them moves, that is a regression and not a
result. Stated here so that a moving number is a signal rather than something to
rationalise afterwards.

## Step 6b: what an open boundary actually is

Step 6a built the mesh. This is what has to be true at its four edges, and the
survey came first because the answer changes the shape of the work.

The solver's boundary vocabulary is **positional**: `wall` means `j = 0` and
`far_field` means `j = -1`, and every consumer -- `momentum`, `face_fluxes`,
`pressure_correction`, both turbulence equations, the force integral -- is
written in those terms. The 2DZP plate needs four boundaries and one of them is
*mixed*, so the vocabulary has to stop being positional. What makes that
tractable is that there are only **three kinds** of condition in the whole
solver, and two of them are already written:

1. **Solid wall.** No slip, no flux, wall functions, `omega` pinned in the
   adjacent cell. Written, at `j = 0`.
2. **Characteristic open boundary.** Each face decides on the sign of `u . n`:
   inflow fixes velocity and turbulence and floats pressure, outflow fixes
   pressure and floats velocity. Written, at `j = -1`, under the name
   `far_field`. Inflow and outflow are not separate conditions -- an inlet is
   this condition on a boundary that happens to be inflow everywhere, and the
   code will select that for itself.
3. **Symmetry.** New. No flux, no shear, zero normal gradient for scalars.

So the plate needs no new *physics* at its `i` ends or its top: those are three
more instances of (2). It needs (3), and it needs (1) and (3) to coexist along
one row.

### Symmetry is a wall with the shear removed

The finite-volume statement is the mirrored face value

    u_face = u_cell - (u_cell . n) n

carried by the interior viscosity through the same diffusive coupling the
no-slip wall uses. The tangential flux is then identically zero -- no shear --
and the normal part is `-mu g (u_cell . n)`, a penalty driving the face-normal
velocity to zero. That is exactly what a solid wall's condition reduces to when
the tangential target stops being zero and starts being whatever the cell has,
which is why this is a *mask over the wall row* rather than a fourth code path.

Everything the wall row does therefore needs the mask:

| what | solid | symmetry |
|---|---|---|
| `wall_velocity` | `0, 0` | `u_cell - (u_cell.n) n` |
| `wall_viscosity` | blended `mu_wall` | interior `mu + mu_t` |
| `wall_turbulence` `omega` | pinned in the cell | not pinned; zero gradient |
| `k` | zero flux | zero flux -- already the same |
| `wall_velocity_gradient` | two-layer profile | the resolved strain |
| force integral | contributes | does not: it is not the body |

The last line is the one that would be silently wrong. `compute_forces`
integrates the whole `j = 0` row, so a plate would report the symmetry plane's
pressure as part of its drag.

### What the `i` ends need that the far field did not

The characteristic condition is written against `flux_j[:, -1]`, whose sign
convention is outward because `j = -1` is the high end. At `i = 0` the outward
direction is *decreasing* `i`, so the same test on `flux_i[0]` has the opposite
sign. `faces.i_start` already stores the outward area vector -- it was built
negated in step 2, mirroring `wall` against `face_j` -- so the fix belongs at
the one place the flux is read, not inside the condition.

Beyond that the `i` ends need what the far field already has and step 5 left
impermeable:

* `face_fluxes` must build `flux_i[0]` and `flux_i[-1]` from the boundary
  velocity instead of writing zeros.
* `add_convection` must carry the two end faces. It does not: `interior_i` is
  `flux_i[1:-1]` and the ends are dropped.
* `pressure_correction` and `apply_correction` need the Dirichlet coupling that
  `_far_field_coupling` provides at an outflow face, at whichever `i` end holds
  the pressure.
* `add_diffusion` already takes `i_start_value` and `i_end_value` from step 3.

### Order, and what each commit must show

* **A. Symmetry as a masked wall.** A shear flow over a symmetry plane feels no
  wall: measured shear zero to rounding, and a uniform stream stays uniform.
* **B. The `i` ends carry flow.** Mass in equals mass out on a channel with no
  body in it, and a uniform stream crosses an open box unchanged.
* **C. A plate case runs.** `Case` accepts a `RectilinearGrid`, and the run
  converges.

The invariant stands: the periodic path stays bit-identical. It is checked the
same way step 6a was checked, against `%.17e` on the cylinder's metrics, `Cd`,
`Cl`, wake length, separation angle and iteration count.

### What happened in 6b

All three commits kept the periodic path bit-identical, measured at `%.17e`
against the previous commit on two cases rather than one: the laminar Re 40
cylinder (metrics, `Cd`, `Cl`, wake length, separation angle, iteration count)
and a k-omega SST NACA 0012 at 5 degrees after 400 iterations (`Cd`, `Cl`,
`Cm`, the sums of `k`, `omega` and `mu_t`, the residual). The second case was
added because most of what A touched -- the wall function, the pinned `omega`
row, the production strain -- is never reached by a laminar run.

**Two O-grid assumptions the survey above missed**, both silent:

* `health.cell_peclet` rolled the node array along `i`. On an open mesh that
  joins the outlet to the inlet and reports a cell as long as the domain.
* The laminar force integral receives no `Boundaries`, because a laminar case
  has no wall model, so a mask carried only by `Boundaries` never reached it.
  `compute_forces` and `surface_data` now take the mask separately. Before the
  second fix a laminar plate reported `Cf` up to 0.93 along the symmetry plane
  -- sixty times the plate's own -- in its surface data.

And one the plan did not list: **SST's wall distance was measured to the whole
`j = 0` row**, symmetry plane included. `compute_metrics` takes the mask and
measures to the solid segments only; on the plate it is exact, `hypot(x, y)`
ahead of the leading edge and `y` over the plate.

**First end-to-end run.** Laminar, `Re_L = 1e4`, 80 x 49 cells, first layer
1e-3, growth 1.1. Converged to 9.7e-08 in 503 iterations, 17 s. Mass through the
outlet 0.97694 of the inlet and through the top 0.02306, summing to one: the
top carries the boundary layer's displacement, as it should.

    x       Re_x      Cf / Blasius
    0.047     234     1.0875
    0.100     499     1.0688
    0.251    1253     1.0587
    0.502    2511     1.0528
    1.005    5025     1.0396
    1.482    7410     1.0314
    1.947    9736     0.9887

Friction `Cd` 1.4001e-02 against Blasius's 1.3280e-02, +5.4%. Blasius is the
`Re_x -> infinity` limit and is approached from above, which is the direction
of the first six rows; **none of this is claimed as agreement** until a grid
study says how much of each figure is discretisation error.

**The peak `|v|` of 0.156 U is the leading-edge singularity**, located rather
than assumed: it sits in the first plate column, `x = 0.002`, `Re_x = 10`. Over
the plate the column maximum tracks Blasius's edge normal velocity
`0.8604 U / sqrt(Re_x)` at ratios 0.919, 0.969, 0.988, 0.997, 1.008 from
`x = 0.02` to `0.5`.

**Open, and not explained:** that ratio then climbs to 1.039, 1.150 and 1.286
at `x = 1.0`, `1.5` and `1.95`, where `Cf` also turns from above Blasius to
below it. Both point at the outlet or the top boundary, not at the plate. The
outlet holds `p = 0` on a boundary one plate length downstream of nothing --
the domain ends at the trailing edge -- and the streamwise spacing is coarsest
there. Step 7 separates those: outlet distance, domain height, and streamwise
refinement, one at a time.
