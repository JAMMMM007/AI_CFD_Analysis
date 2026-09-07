# Response to the physics audit of 2026-09-07

Mode A: implementing an audit report. This document is the written plan the
workflow requires before anything is implemented, and it is a proposal — nothing
in it has been built. The decisions the owner has to make are at the end, and
three of them gate the work.

---

## 0. Re-derivation before anything else

The audit's findings were not accepted on the strength of being written down.
Three were re-derived from the code and the mathematics, independently, and the
measurements reproduced. All three ran from a scratch script importing
`fluidsolver` read-only; nothing in the repository was modified.

**F4 — Rhie-Chow omits the non-orthogonal correction. Holds, exactly.**

The claim is that for a linear pressure field, where a consistent damping term is
identically zero, the coded `compact - smooth` evaluates to `-(grad p)_f . T`.
Setting `p = 1.7 x - 0.9 y` on both meshes — the gradient operator reproduces it
to `2.4e-14`, so the field is exact:

```
cylinder Re 40, exactly the gate's settings: 180x53
  i-faces  max|compact-smooth| 3.8030e-08   max|residue - (-(grad p).T)| 4.780e-14
  j-faces  max|compact-smooth| 8.2787e-09   max|residue - (-(grad p).T)| 2.515e-15
           residue/|smooth|  mean 0.0000  p99 0.0000  max 0.0000
           non-orthogonality  i 0.0000 / j 0.0000 deg, mean and peak

NACA 2412, 5 deg, y+ 1: 240x89
  i-faces  max|compact-smooth| 5.0675e-01   max|residue - (-(grad p).T)| 4.798e-14
  j-faces  max|compact-smooth| 1.0040e-01   max|residue - (-(grad p).T)| 8.399e-14
           residue/|smooth|  i: mean 0.7533  p99 9.3917
                             j: mean 1.6352  p99 9.5039
           non-orthogonality  i mean 3.8493 peak 60.0979 deg
```

The residue equals `-(grad p)_f . T` to `8e-14` on both meshes, which is the
algebra confirmed to machine precision, and the 99th percentiles of `9.39` and
`9.50` match the audit's `9.3784` and `9.5073`. The finding holds and the
diagnosis is right.

**F8 — the marching Newton iteration is truncated. Holds, exactly.**

```
     iterations  4   marched 52
     iterations  8   marched 49
     iterations 16   marched 54
     iterations 40   marched 62
```

Identical to the audit's 52 / 49 / 54 / 62, including the non-monotonicity at
eight passes.

**F14 — two wall distances in use. Holds, exactly.**

```
  first cell row, wall_normal_distance / metrics.wall_distance:
     mean 0.997940  min 0.851965  max 1.000000
     implied omega ratio at the worst cell: 1.3777, i.e. 37.8%
```

The audit reports `0.851920` and 38%.

**One correction to the audit, at the level of the remedy rather than the
finding.** F11 says of the non-orthogonal manufactured solution: "This alone
would have caught F4." It would not. The MMS exercises the assembled convection
and diffusion operator against a *prescribed* flux field; F4 lives in
`simple.face_fluxes`, which builds the flux, and no manufactured solution in
`tests/` touches it. What catches F4 is the linear-pressure-field identity used
above — on any mesh, for any linear `p`, the damping must be identically zero —
and that is cheap, exact, and needs no solve at all. The non-orthogonal MMS is
still worth building; it is the instrument for F13 and for the diffusion cross
term, and those are different findings.

The remaining fourteen findings are taken on the audit's evidence for the purpose
of *planning*. Each is re-derived at the point it is implemented, as the workflow
requires, and not before.

---

## 1. Baseline

Recorded on `main` at `37fabd2` before a line was changed. The two runs shared
the machine, so the wall times are not comparable with the audit's; the numbers
are.

```
.\.venv\Scripts\python.exe -m pytest -q
252 passed in 266.08s (0:04:26)

.\.venv\Scripts\python.exe -m validation.cylinder
Re = 20   (1179 iterations, residual 9.97e-08)
  Cd                          2.0232      expect 2.0-2.09  [ok ]
  wake L/D                    0.9333      expect 0.91-0.94  [ok ]
  separation (from rear)     43.6382deg   expect 43.0-45.0  [ok ]
  Cl (symmetry)             -0.00000   expect 0
  Cd split                 pressure 1.2191, friction 0.8041

Re = 40   (992 iterations, residual 9.96e-08)
  Cd                          1.5142      expect 1.5-1.58  [ok ]
  wake L/D                    2.1219      expect 2.1-2.35  [ok ]
  separation (from rear)     53.7170deg   expect 52.0-54.0  [ok ]
  Cl (symmetry)             -0.00000   expect 0
  Cd split                 pressure 0.9925, friction 0.5217

PASS
```

`Cd 1.5142`, `wake 2.1219`, `separation 53.717`, `Cl -0.00000`, residual `1e-7`:
the tree is the one the documentation describes. The test count is **252**, which
settles the README's 188 against the handover's 252 in the handover's favour.

---

## 2. The ordering argument

Seventeen findings have no natural order, so this one is argued rather than
asserted. Three facts fix it.

**Four findings are invisible to every measurement the project owns.** F4, F13
and F14 are identically zero on an orthogonal mesh, and F2's vortex term is
identically zero on a non-lifting body. The cylinder gate is orthogonal *and*
non-lifting; the manufactured-solution meshes are orthogonal, unstretched, and
exclude both boundary rows. Implementing any of those four now would mean
changing code whose effect cannot be measured, which standing instruction 4
forbids in as many words.

**The two largest numerical findings are the ones capping the observed order.**
F3's wall-pressure gap is measured at order `1.005, 1.002`; F4's spurious flux is
`O(h)` wherever the mesh is non-orthogonal. Together they are why the observed
order of `Cd` is 1.261 rather than 2. Fixing them is the highest-value work in
the report — but the *proof* that they are fixed is an observed order, and there
is no mesh-family harness to measure one with.

**F3 makes the gate's error worse at the present resolution before it makes it
better.** Correcting it alone moves `Cd` from 1.5142 to 1.5164, taking the error
against the Richardson limit from `−0.076%` to `+0.079%`, because it removes one
of two cancelling first-order terms. That is still the right change, and it is
defensible only if the gate reports an uncertainty rather than a digit. So the
gate policy has to be settled before F3 lands, not rationalised afterwards.

Hence: **build the instruments, then fix the terms the instruments measure.**

The alternative ordering — go at F1 first, because 8.94% in `Cd` is the largest
number in the report — is rejected for the reason the audit itself gives. F1 is a
calibration question whose answer is a flat-plate measurement the project has
never run, and a 9% drag change adopted on authority ahead of that measurement is
precisely the kind of decision the standing instructions exist to prevent.

---

## 3. Stage I — instruments

Nothing here changes an answer. Every item is a measurement that does not
currently exist, run on unmodified code so that a before-and-after has a before.

### I.1 A linear-field consistency test for the Rhie-Chow flux

**The mathematics.** For `p = G . x` we have `p_N - p_P = G . d` and
`(grad p)_f = G` exactly, so a consistent damping term satisfies

    D_f [ g (p_N - p_P) + (grad p)_f . T  −  (grad p)_f . S ]
      = D_f [ g (G . d) + G . T − G . (g d + T) ] = 0

identically, on any mesh, for any `G`. This is an algebraic identity, not an
asymptotic statement, so the test carries no discretisation error to allow for
and the tolerance is machine precision.

**What lands where.** `tests/test_solver.py`. Two meshes — the gate's cylinder
and the NACA 2412 at 5 degrees — and the assertion
`max|damping| < 1e-12 * max|smooth|`.

**What breaks.** It fails on the NACA mesh at `5.07e-01` against a `smooth` of
order `5e-2`, and passes on the cylinder at `3.8e-08`. That is its purpose. It
lands in Stage I as `xfail(strict=True)` naming the defect it guards, and the
`xfail` comes off in Stage II.

### I.2 Manufactured solution on a non-orthogonal, stretched mesh

F11 items 1 and 2. The existing `uniform_mesh` helper gains a variant built with
`growth=1.15` and the polar blend left in, which is what `build_ogrid` produces
by default; the measured mesh properties then move from
`non-orth 0.0000, aspect 2.5, expansion 1.08` towards the working mesh's
`3.9 / 453 / 4.77`. The boundary rows are measured and reported separately rather
than excluded, so that a change from `O(h)` to `O(h²)` at the wall is visible
instead of hidden.

**Acceptance.** The observed order is *reported*, not asserted, in Stage I — this
is a measurement, and asserting a number before it has been measured is the trap
the handover records twice. The assertion goes in during Stage II, once the value
is known and understood.

### I.3 A second regression gate: the NACA 2412

Audit open question 7. The cylinder is orthogonal, non-lifting, laminar and run
at one mesh density, and is structurally incapable of detecting seven of the
seventeen findings. The NACA 2412 at 5 degrees, Re 2.03e6, SST, y+ 1, 40 chords
sees all seven, converges to `1e-6` in about 1000 iterations, and takes roughly a
quarter of an hour.

Its job is to **detect change, not to certify accuracy** — it is not a validated
case — so it carries wide bands and a full specification recorded in the module
docstring: geometry, incidence, Reynolds number, fluid, surface points, `y+`
target, far-field ratio, scheme, relaxation, tolerance. The audit notes that
three separate documents currently point the next engineer at a NACA 0012 case
whose specification is written down nowhere and which no longer reproduces. This
gate is written so that cannot happen to it.

### I.4 A mesh-family and grid-convergence harness

K5, and the precondition for any statement about observed order or
discretisation uncertainty. One specification produces a systematically refined
family — `MeshSettings` with an explicit `first_layer` and a scaled
`surface_points` is enough for the cylinder; an aerofoil family also needs the
resampler's `min_spacing` to scale with it, or the refinement is not uniform.
Celik's procedure on top:

    p   = ln( (f3 − f2) / (f2 − f1) ) / ln(r)
    GCI = 1.25 |f2 − f1| / (|f1| (r^p − 1))

**Acceptance.** Re-running the audit's own three-mesh cylinder family through the
harness must return its published numbers: observed order of `Cd` 1.261,
Richardson limit 1.515358; wake order 0.735, limit 2.200; separation order 2.88,
limit 53.756. That is a real check on the harness, because those numbers came out
of an independent script.

**Cost.** The three-mesh family took 35 minutes in the audit. It is a `validation`
entry point, not a pytest test.

---

## 4. Stage II — the flux definition

F4 and F5 live in the same function and both change every answer on a
non-orthogonal mesh. Three commits, one mathematical change each.

### II.1 The non-orthogonal Rhie-Chow correction (F4)

**The mathematics.** With `S = g d + T`, the compact operator must be the one the
diffusion assembly already uses — implicit orthogonal part plus explicit cross
term — so that

    compact_full − smooth = g [ (p_N − p_P) − (grad p)_f . d ]

which is zero for linear `p` on any mesh and `O(h³ p''')` in general. The second
form is also the cheaper implementation: the whole damping term is `D_f g` times
the difference between the compact pressure difference and the interpolated
gradient projected on `d`. `InteriorFaces` already carries `diffusion_factor`,
`cross` and `delta`, so nothing new is computed.

**What lands where.** `simple.face_fluxes`, both face families; and the matching
compact operator in `simple.apply_correction`, which must change with it or the
flux update and the pressure equation describe different things again — the
failure `_far_field_coupling`'s docstring already records.

**The pressure-correction matrix stays orthogonal-only.** It does not need the
cross term for correctness, because `p' → 0`; adding it costs an extra lagged
term and risks diagonal dominance on a mesh with 39% of its cells in the blended
region. This follows the audit's recommendation on open question 4, and is
revisited only if the outer iteration slows.

**What breaks, predicted before running.**

- I.1's linear-field test goes from `5.07e-01` to machine zero. The `xfail` comes
  off.
- **The cylinder gate does not move.** The term being removed is `3.8e-08` there,
  measured above, so `Cd 1.5142`, `wake 2.1219` and `separation 53.717` must all
  stand to every printed digit. If the gate moves, the change has leaked into
  somewhere it does not belong — a stronger check than any assertion I could
  write, and the same one that caught the wall treatment reaching into laminar
  runs.
- The NACA 2412 gate moves, and by how much is the measurement. The audit
  establishes that the spurious flux is 0.002% of the mean physical face flux but
  34% on the worst single face, and introduces `1.37e-04` of the domain's mass
  throughput — two orders above the `1e-6` tolerance the run stops at, so it is
  not buried under iteration error. No prediction of the sign or size of the
  change in `Cd` is offered here, because none has been measured.
- No existing test asserts a force coefficient on a non-orthogonal mesh, so none
  is expected to change. If one does, that is a signal rather than something to
  rationalise.

**Acceptance.** The identity holds to machine precision on both meshes; the
cylinder gate is unmoved; the NACA gate's movement is measured and reported.

### II.2 Relaxation-independence of the converged answer (F5)

**The mathematics.** `momentum()` returns the *relaxed* diagonal `a_P/alpha_u`,
so `D_f = alpha_u V / a_P`, and the damping term — which does not vanish at
convergence — carries `alpha_u` into the fixed point. Choi's form retains the
previous face flux explicitly,

    F^m = rho [ u_f^m . S − D_f (damping)^m
                + (1 − alpha_u) ( F^{m−1}/rho − u_f^{m−1} . S ) ]

so the relaxation-dependent part telescopes and the fixed point is independent of
`alpha_u` by construction. Taking the mobility from the unrelaxed diagonal is one
line and removes the leading dependence only; Choi's form is chosen because
Stage 4 needs exactly this machinery to make Rhie-Chow time-step-independent, and
otherwise the work is done twice. This follows the audit's recommendation on open
question 5.

**Structural cost, honestly.** This needs `u_f^{m−1} . S` stored between outer
iterations. `F^{m−1}` is already `state.flux_i` / `state.flux_j`; the interpolated
face-velocity flux is not, so `State.ARRAYS` gains one or two entries — and
`copy()`, `nbytes` and `is_finite` follow automatically, because they are already
driven off `ARRAYS` rather than written out by hand. That is the only structural
change in Stage II. It does not touch `StructuredMatrix`'s five bands, the
`k = i*Nj + j` ordering, or any of the 63 periodic `np.roll` uses.

**Acceptance, with a null control.** Re-run the audit's sweep on the cylinder at
`tolerance = 1e-9`: `alpha_u` 0.70 / 0.55 / 0.40 at fixed `alpha_p`, and
`alpha_p` 0.15 / 0.30 / 0.45 at fixed `alpha_u`. The `alpha_p` family measures the
convergence floor at `1.5e-05` in `Cd` and is unaffected by this change — that is
the control. The criterion is that the `alpha_u` spread falls from `2.76e-04` to
the `alpha_p` family's level: not to zero, which would be a suspiciously good
answer at a finite residual, and not below the floor the other family measures.
Cost: six runs of roughly 1400 to 4200 iterations.

**What breaks.** Four docstrings assert relaxation-independence — the `simple`
module docstring, `linalg.Coefficients.under_relax`, `simple.CflRamp` and
`sst.KOmegaSST.update`. They are what stopped this being noticed, and all four are
corrected in the same commit. The cylinder gate's fourth decimal may move by up to
`2.8e-04`; it is inside the band either way, and whether it moves at all is a
measurement rather than a prediction.

### II.3 The zero-gradient condition along `n` rather than `d` (F13)

Small in effect — `tan(0.53 deg)` typical, `tan(31.6 deg)` at the worst face —
and grouped here because it is the same class of defect and I.2 is the instrument
that measures it. It lands last of the three so that if the observed order moves,
the cause is unambiguous.

---

## 5. Stages III to VII, in outline

Each is detailed to the depth of §4 when it is reached, not now.

**III — the boundary and reporting terms.** F3 (wall-pressure reconstruction,
moves the gate), F12 (separation from a one-sided wall gradient), F17 (mask the
substituted wall rows out of the `omega` residual), F14 (one wall distance and one
constant, not two of each), F10 (the moment sign), F16 (the mesh page's label),
F15 (wall-distance sampling). F3 is the one with consequences, and it cannot land
until the gate policy is decided.

**IV — the mesher.** F8 first: iterate the marching Newton to a residual with a
pass ceiling. One loop, and it buys ten layers. Then K2's seam, blending the
*marching direction* into the polar ray over eight to ten layers — the audit
establishes that blending positions alone cannot work, because the discontinuity
is a 36-degree direction mismatch and the polar construction has no direction to
blend. Marched-layer counts reported alongside both, because march depth is mesh
quality.

**V — the turbulence model definition.** F6 first: `gamma P̃_k / nu_t` is what
SST-2003 specifies and the code claims to implement SST-2003, so it is a
definitional correction needing no decision, and its measured effect is +0.021%
in `Cd`. Then F7, then F1 — and F1 only behind the NASA TMR `2DZP` flat plate,
for the reason the audit gives.

**VI — the far field.** F2's point-vortex correction, `O(1/R)` to `O(1/R²)`. It
changes every lifting result. Needs the current `C_l` channelled into
`Boundaries`, which `Case` already computes every iteration.

**VII — what is left.** The residual plateau at `4e-05` that K1 identifies as the
genuine open item on the NACA 0012, in place of the divergence-monitor false
positive that no longer reproduces.

**K1 is closed as written, on the audit's evidence and pending one confirmation
run.** Three documents currently direct the next engineer at a defect the audit
could not reproduce on `main`. Whatever is decided about the monitor itself, that
statement has to come out of `README.md`, `docs/handover.md` and
`docs/hardening-plan.md` — and the standing instruction cuts both ways: the
documented failure is itself a claim that has now failed to reproduce.

---

## 6. Documentation debt to clear along the way

- `README.md` says 188 tests, `docs/handover.md` says 252; the measured count is
  **252**. Whichever document is touched, the count is made to match.
- `docs/handover.md` gives the interpreter as
  `C:/AI_CFD_Analysis/.venv/Scripts/python.exe`, which does not exist in this
  checkout, and the suite as "about 100 seconds" against a measured 266.
- `README.md`'s status table claims "second order (verified by manufactured
  solution)". What the verification supports is "second order for the isolated
  convection and diffusion operators, on an orthogonal unstretched polar mesh,
  away from the boundaries". The narrower claim goes in until the wider one is
  earned.
- `build_ogrid`'s docstring says less than about twenty chords "starts to
  interfere with the circulation". The measured interference at *forty* is 1.24%
  in `Cl` and 10.1% in `Cd`. Those numbers belong in the sentence.

---

## 7. Decisions — settled 2026-09-07

The three blocking questions were put to the owner and answered. The remaining
seven were delegated with them, so they are settled here as recommended and are
recorded rather than raised.

**Answered by the owner**

1. **Scope: all stages, in the order above, without further check-ins.** Work
   proceeds through I to VII, reporting as each lands. The consequence is that
   items 4 to 10 below are mine to decide, and they are decided as recommended.
2. **The gate reports an uncertainty, not a digit.** `Cd`, wake and separation
   each carry a discretisation uncertainty from a mesh family, per ASME V&V 20,
   and the headline digits are accepted to move once. This unblocks F3.
3. **The second gate is `validation.aerofoil`,** a deliberate entry point beside
   `validation.cylinder`, not a pytest test. `pytest -q` stays at four minutes.

**Settled as recommended**

4. F1 — bring the NASA TMR `2DZP` flat plate forward and let it decide the
   constant, rather than adopting 60 on authority.
5. F2 — implement the point-vortex correction rather than move the boundary out.
6. F4 — leave the pressure-correction matrix orthogonal-only.
7. F5 — Choi/Pascau rather than the unrelaxed diagonal.
8. F7 — `SST-sust` sustaining terms rather than changed freestream defaults.
9. F9 — remove `enforce_global_mass_balance`, and enforce compatibility on the
   source only where there is no fixed-pressure face.
10. F10 — flip the sign to the standard nose-up-positive `Cm`, and offer a
    quarter-chord moment reference alongside the centroid default.
