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
