# Physics and mathematics audit — 2026-09-07

## Scope and method

**Read in full**, in this order: `README.md`, `docs/handover.md`,
`docs/hardening-plan.md`, `docs/compressible.md`, `docs/optional-deps.md`; then
`mesh/metrics.py`, `mesh/spacing.py`, `mesh/hyperbolic.py`, `mesh/ogrid.py`,
`mesh/quality.py`, `solver/faces.py`, `solver/operators.py`, `solver/linalg.py`,
`solver/simple.py`, `solver/bc.py`, `solver/fields.py`, `solver/fluid.py`,
`solver/turbulence/{base,laminar,sst}.py`, `solver/post.py`, `solver/case.py`,
`solver/health.py`, `solver/guard.py`, `validation/cylinder.py`, and the test
suite — `tests/test_solver.py` and `tests/test_turbulence.py` line by line, the
rest for what they cover. `gui/pages/mesh.py` and `gui/pages/setup.py` were read
only where they report a physical quantity.

**Baseline established before any opinion was formed.**

```
.\.venv\Scripts\python.exe -m pytest -q
252 passed in 259.87s (0:04:19)

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

The numbers match `docs/handover.md` exactly, so the code audited is the code
the documentation describes.

**Nothing in the repository was modified.** Every numerical experiment ran from
a script under
`%TEMP%\claude\C--Jack-CFD-Work\...\scratchpad\`, importing `fluidsolver`
read-only. Where a model variant had to be tried it was applied as a runtime
monkey-patch in the scratch script (`bc._OMEGA_WALL_FACTOR = 60.0`;
`KOmegaSST._solve_omega = ...`), never as an edit to source.

**Three cases carry the measurements.** The laminar cylinder at Re 40 — the
project's own regression gate, and its only validated result. A NACA 2412 at
5 degrees, Re 2.03e6, `AIR_15C`, U = 30 m/s, 240 surface points, target y+ = 1,
far field 40 chords, k-omega SST, which is the configuration
`tests/test_turbulence.py::TestAutomaticWallTreatment` uses and the closest thing
the project has to a primary use case. And a NACA 0012 at zero incidence in the
same conditions, run only to re-examine the hardening plan's first open item
(K1).

Turbulent runs had the divergence monitor disarmed in the scratch script, because
its documented false positive would otherwise stop them. K1's first run is the
deliberate exception — it was run on factory defaults with the monitor **armed**,
which is the whole point of it, and the result is not what the documentation
says.

**Primary sources fetched, not recalled.** The NASA Turbulence Modeling
Resource has moved from `turbmodels.larc.nasa.gov` to
`https://tmbwg.github.io/turbmodels/`; the SST page there was fetched and is
quoted verbatim below. That page is decisive on two of the items in this report
and it contradicts a premise of the brief on a third.

**One grid-convergence study was run**, because the project intends Celik-style
GCI and has never measured an observed order on a solved field. Three
systematically refined cylinder meshes, `r = 1.5` in both directions
simultaneously (180x53, 270x56, 405x59; first layer scaled by the same ratio),
each converged to `tolerance = 1e-8`. It took 35 minutes and it is the single
most informative measurement in this report. It is reported under F11.

**Deliberately not examined.** The Qt front end except where it reports a
physical number; the DXF importer and the geometry resampler; the linear solver's
performance; anything requiring three dimensions, compressibility, unsteady time
integration or a transition model, all of which are out of scope by declaration.
Convergence *rate* is treated only where it bears on whether a converged answer
exists.

---

## Summary of findings

Ranked by effect on a number the project reports, worst first. The IDs are in the
order the findings appear in the sections below, so the table is not in ID order
— F17 outranks F15 and F16 on effect and is written last. The last column is
where the change lands, not how long it takes.

Nothing here is SUSPECTED. Every row was either derived in closed form and
verified numerically, or measured on a converged run whose command and raw output
are pasted into the finding.

| ID | Area | Statement | Confidence | Effect on a reported quantity | Cost |
|---|---|---|---|---|---|
| F1 | Turbulence, wall | `_OMEGA_WALL_FACTOR = 6.0` is one tenth of what SST-2003 specifies at that point, and the docstring's arithmetic for why is wrong | CONFIRMED | NACA 2412: `Cd` **−8.94%**, `Cd_friction` −10.23%, `Cl` +1.24% when corrected | one constant in `bc.py`; but see the caveat |
| F2 | Far field | No far-field vortex correction; the error is `O(1/R)` and 40 chords is not far enough | CONFIRMED | at the default 40 chords: aerofoil `Cd` **+10.1%**, `Cd_pressure` +29%, `Cl` −1.24%; cylinder `Cd` +1.32% | `bc.far_velocity`, `bc.far_pressure` |
| F3 | Post-processing | Wall pressure is the cell-centre value: a zeroth-order extrapolation, measured first order (1.005, 1.002) | CONFIRMED | cylinder Re 40: `Cd` moves **0.143%**, and it is one of the terms capping the observed order | `post.compute_forces` |
| F4 | Rhie-Chow | The damping omits the non-orthogonal correction, leaving a spurious flux `O(h)` where the intended term is `O(h³)` | CONFIRMED | zero on the cylinder; on the NACA mesh the term as coded is **9.5x** what it should be, 93% of it spurious | `simple.face_fluxes` |
| F5 | Rhie-Chow | The converged answer is linear in `relax_velocity` and independent of `relax_pressure`, contradicting four docstrings | CONFIRMED | cylinder `Cd` 1.514209 at α_u 0.7 against 1.514486 at 0.4 — the fourth decimal the project reports | `simple.face_fluxes` |
| F6 | Turbulence | The omega production is `gamma rho S^2`; SST-2003 specifies `gamma P̃_k / nu_t`, limiter on both equations | CONFIRMED | model definition wrong; measured effect here only +0.02% in `Cd` | `sst._solve_omega` |
| F7 | Turbulence | Freestream `k` is 30x and `mu_t/mu` 100x above the NASA TMR band | CONFIRMED | measured ambient `mu_t/mu` 0.85 at the body, `Tu` 0.031% against the 0.1% set; blocks Stage 5 | `fluid.Freestream` defaults |
| F8 | Meshing | The marching Newton iteration is a fixed four passes with no convergence test | CONFIRMED | 52 of 63 layers at four passes, 62 at forty — and march depth *is* mesh quality | one loop in `hyperbolic.py` |
| F9 | Far field | `enforce_global_mass_balance` is still active at convergence, so it is part of the model | CONFIRMED | converged outflow rescaled by −0.0252% | `bc.enforce_global_mass_balance` |
| F10 | Post-processing | `Forces.moment_coefficient` is the negative of the standard nose-up-positive `Cm` | CONFIRMED | every reported `Cm` has the wrong sign | one line in `post.compute_forces` |
| F11 | Verification | The MMS runs on an orthogonal unstretched polar mesh, excludes both boundary rows, and tests operators in isolation | CONFIRMED | **observed order of `Cd` is 1.261, not 2**; wake length not grid-converged (order 0.735, limit 2.200 against a reported 2.1219) | `tests/`, then everything |
| F12 | Post-processing | The separation angle is reported to three decimals; the method-dependence is 0.25 degrees | CONFIRMED | `53.717` carries two digits the discretisation does not support | `post.separation_points` |
| F13 | Operators | A zero-gradient condition is imposed as `grad phi . d = 0`, not `grad phi . n = 0` | CONFIRMED | exact on the cylinder; `tan(0.53 deg)` typical and `tan(31.6 deg)` worst on NACA wall faces | `operators.Gradient` callers |
| F14 | Meshing | Two different wall distances are in use, disagreeing by up to 15% in the first cell row | CONFIRMED | `omega` seed and `omega` wall value disagree by 38% at start-up | `fields.State.uniform` |
| F17 | Residuals | 57% of the `omega` normaliser and 65% of its imbalance come from the identity-substituted wall row | CONFIRMED | no coefficient changes; the stopping criterion is a max over incommensurable numbers | `linalg.Coefficients.residual` |
| F15 | Meshing | Wall-distance sub-sampling at 8 points per segment is 17% off in the first cell row | CONFIRMED | none measurable — the SST blending functions are saturated there; the error is at the trailing edge, not the nose | `metrics._wall_distance` |
| F16 | Reporting | The mesh page calls the flat-plate estimate the "achieved" y+ | CONFIRMED | reads 1.00 where the solver delivers 0.30 .. 2.36 | `gui/pages/mesh.py:185` |

---

## Findings

### F1 — The omega wall value is a factor of ten below what SST-2003 specifies

**Statement.** `bc._OMEGA_WALL_FACTOR = 6.0` prescribes
`omega = 6 nu / (beta1 y1^2)` in the wall-adjacent cell, where the NASA
Turbulence Modeling Resource specifies `10 (6 nu) / (beta1 (Delta d1)^2)` at
exactly the point this code applies it; the code is ten times low, not the 0.4
its docstring's reasoning implies.

**Location.** `fluidsolver/solver/bc.py:32-43`:

```python
# Wilcox's exact near-wall solution, omega -> 6 nu / (beta1 y^2) as y -> 0.
#
# Menter quotes this with a factor of ten, as 60 nu / (beta1 dy^2), and that form
# is widely copied -- but it is for codes that need a value *on the wall face*,
# where omega is formally infinite, and the ten is deliberate over-specification
# to force the right asymptotic behaviour. Here the value is prescribed in the
# first cell instead, at a point where the asymptote is simply valid, so the
# factor does not belong: including it puts omega ten times too high in the
# stiffest cell of the mesh.
_OMEGA_WALL_FACTOR = 6.0
```

used at `bc.py:246-251`, and duplicated as a bare `6.0` in
`fields.State.uniform`, `fields.py:105-108`.

**The mathematics.**

*Continuous statement.* In the viscous sublayer the omega equation reduces, as
`y -> 0`, to viscous diffusion against destruction,

    nu d²omega/dy² = beta_1 omega²,

whose exact solution is

    omega(y) = 6 nu / (beta_1 y²),

singular at the wall. This is Wilcox's near-wall asymptote and the `6` in it is
not adjustable.

*Discrete statement, two constructions.*

(a) The published prescription. Because `omega` is infinite on the wall itself,
Menter and the TMR prescribe the asymptote evaluated at the first solution point
off the wall and multiply it by ten:

    omega_wall = 60 nu / (beta_1 (Delta d_1)²),
    Delta d_1 = distance to the next point away from the wall.

The TMR writes the factor out as `10 (6 nu)/(beta_1 (Delta d_1)^2)`, which makes
the over-specification explicit rather than hiding it in a constant.

(b) This code. It replaces the wall row of the `omega` system with the identity
(`sst._fix_wall_row`) and prescribes `omega` in the first *cell* rather than on
the wall face. The docstring argues that at a real point the asymptote is simply
valid, so the ten does not belong.

*Where they part company.* They do not part company on the point of evaluation —
they agree on it. `Delta d_1` in the published form is the distance to the first
solution point, which in a cell-centred finite-volume code *is* the first cell
centre, which is what `faces.wall.wall_normal_distance` returns and what this
code substitutes for `y1`. The two prescriptions are therefore evaluated at the
same place and differ by exactly 10.

The brief's premise — that `Delta d_1` is the first cell *height*, so that
`y1 = dy1/2` makes the code 0.4 of the published value rather than 0.1 — is not
what the sources say. Settling that was worth the fetch: the discrepancy is a
factor of ten, and the docstring's justification for removing it does not
survive.

*Why ten is not merely a fudge.* The asymptote solves the *reduced* equation. The
discrete first cell has finite height, the diffusion operator across a
1000:1-stretched cell under-resolves a `1/y²` profile, and the discrete solution
relaxes *below* the asymptote if the asymptote itself is what is prescribed. Both
Menter and Wilcox state that the solution is insensitive to the prescribed value
*provided it is large enough*, and both set the margin at ten. Removing the
margin removes the insensitivity, and a low near-wall `omega` is a high
`mu_t = rho k / omega` which propagates outward through the `omega` diffusion into
the log layer.

**Evidence.** NASA Turbulence Modeling Resource, Menter Shear Stress Transport
page, fetched 2026-09-04 from `https://tmbwg.github.io/turbmodels/sst.html`,
quoted:

> "ω_wall = 10 (6 ν)/(β₁ (Δ d₁)²)" and "k_wall = 0"

Four runs of the NACA 2412 case, identical but for the monkey-patched constant,
each converged to `worst < 1e-6`:

```
RESULT base:   iters 1007  worst 9.9385e-07  Cl 0.7535096  Cd 0.01250649  Cdp 0.00505258  Cdf 0.00745391  Cm -0.0825130
RESULT base:   y+ 0.301..2.357  mut/mu peak 186.16  limiter-fires 8.59% of cells  omega_wall 8.042e+06
RESULT wall60: iters 1056  worst 9.8109e-07  Cl 0.7628901  Cd 0.01138820  Cdp 0.00469699  Cdf 0.00669120  Cm -0.0826105
RESULT wall60: y+ 0.264..2.366  mut/mu peak 158.82  limiter-fires 8.60% of cells  omega_wall 7.993e+07
```

**Magnitude.** NACA 2412, 5 degrees, Re 2.03e6, y+ ~ 1:

| | as coded (6) | TMR (60) | change |
|---|---|---|---|
| `Cl` | 0.7535096 | 0.7628901 | **+1.245%** |
| `Cd` | 0.01250649 | 0.01138820 | **−8.94%** |
| `Cd_pressure` | 0.00505258 | 0.00469699 | −7.04% |
| `Cd_friction` | 0.00745391 | 0.00669120 | **−10.23%** |
| peak `mu_t / mu` | 186.16 | 158.82 | −14.7% |

This is the largest effect in this report, and it appears on the *wall-resolved*
mesh where the wall treatment is supposed to be invisible. For orientation: the
plan's Stage 3 table reports `Cd 0.012501` at y+ 1 on this geometry, which is
this run; and `README.md` reports a NACA 0012 at Re 2e6 giving `Cd = 0.009487`
against a published 0.008. A 9% reduction would close roughly half of that gap
without touching transition modelling.

**A caveat that matters more than the number.** The measurement shows the answer
is *not* insensitive to the prescribed `omega`, which is the assumption both
prescriptions rest on. A 9% swing in `Cd` from a boundary value means the
near-wall `omega` field is under-resolved in a way that neither constant repairs.
The right response is not to swap 6 for 60 on the strength of one aerofoil but to
run the NASA TMR zero-pressure-gradient flat plate (`2DZP`), where `Cf(Re_x)` is
published, and establish which prescription reproduces it. That is a Stage 7
measurement and it is the one that settles this.

**The correct formulation.**

    omega_1 = 60 nu / (beta_1 d_1²),    beta_1 = 0.075,
    d_1 = perpendicular distance from the wall face to the first cell centre,

prescribed exactly as now, by identity substitution on the wall row. The seed in
`fields.State.uniform` must move with it.

**References.** NASA Turbulence Modeling Resource, "Menter Shear Stress Transport
Model", `https://tmbwg.github.io/turbmodels/sst.html`, section on wall boundary
conditions (accessed 2026-09-04). Menter, F. R. (1994), "Two-Equation
Eddy-Viscosity Turbulence Models for Engineering Applications", *AIAA Journal*
32(8), 1598–1605, appendix. Wilcox, D. C. (2006), *Turbulence Modeling for CFD*,
3rd ed., DCW Industries, §4.6.2 and §7.3.1.

**Confidence.** CONFIRMED — definition from the primary source, magnitude by
measurement.

**Cost.** Two constants, both local (`bc.py`, `fields.py`). The *decision* is not
local: it needs `2DZP` first.

---

### F2 — There is no far-field vortex correction, and 40 chords is not far enough without one

**Statement.** For a lifting body the bound circulation induces a `1/r` velocity
that the freestream Dirichlet condition imposes away on every inflow face; the
resulting error decays as `1/R`, and at the default 40 chords it costs 1.24% of
`Cl` and 10.1% of `Cd` on a NACA 2412 at 5 degrees.

**Location.** `fluidsolver/solver/bc.py:263-274`:

```python
    def far_velocity(
        self, u: np.ndarray, v: np.ndarray, far_flux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Freestream where flow enters, extrapolated where it leaves."""
        entering = self.inflow_mask(far_flux)
        stream = self.freestream.vector
        return (
            np.where(entering, stream[0], u[:, -1]),
            np.where(entering, stream[1], v[:, -1]),
        )
```

and the default `MeshSettings.far_field_radius_ratio = 40.0`
(`case.py:56`), together with `bc.far_pressure`, which pins `p = 0` on every
outflow face.

**The mathematics.**

*Continuous statement.* Outside the viscous region the flow past a
two-dimensional lifting body is, to leading order in `1/r`,

    u(r, theta) = U_inf + u_vortex + u_doublet + O(r^-3),
    u_vortex    = (Gamma / 2 pi r) e_theta,
    u_doublet   = O(r^-2),

with `Gamma = (1/2) C_l U_inf c` from the Kutta–Joukowski theorem
`L = rho U Gamma`, `C_l = L / ((1/2) rho U² c)`.

*Discrete statement.* On the inflow arc the code imposes `u = U_inf` exactly. The
error it introduces on the boundary is therefore `-u_vortex`, of magnitude

    |Delta u| / U_inf = Gamma / (2 pi R U_inf) = C_l c / (4 pi R).

For `C_l = 0.75` and `R = 40 c` that is `1.49e-3`, one and a half parts in a
thousand of the freestream — which sounds negligible and is not, because it is
imposed as a *velocity* over an arc of length `2 pi R`, so the total spurious
volume flux it removes is `O(Gamma)`, i.e. the whole circulation.

*Order.* The induced velocity is `O(1/R)`, so the leading error in every integrated
coefficient is `O(1/R)` as well. That is the prediction to test: a `1/R` law
means halving `R` doubles the error and the successive differences under
doubling stand in the ratio 2.

**Evidence.** NACA 2412, 5 degrees, Re 2.03e6, SST, y+ 1, 900 iterations each
(`worst` from `3.75e-06` down to `1.79e-06`), far-field radius the only change:

```
FAR  20.0  iters 900  worst 3.750e-06  Cl 0.74403030  Cd 0.013556492  Cdp 0.006096063  Cdf 0.007460429  Cm -0.0809117
FAR  20.0  Gamma 11.16045   v_theta(R)/U = 2.9604e-03
FAR  40.0  iters 900  worst 2.334e-06  Cl 0.75323071  Cd 0.012515253  Cdp 0.005061984  Cdf 0.007453269  Cm -0.0825028
FAR  40.0  Gamma 11.29846   v_theta(R)/U = 1.4985e-03
FAR  80.0  iters 900  worst 1.790e-06  Cl 0.75790082  Cd 0.011969533  Cdp 0.004519799  Cdf 0.007449735  Cm -0.0833241
FAR  80.0  Gamma 11.36851   v_theta(R)/U = 7.5390e-04
```

and, for the non-lifting case, the cylinder at Re 40 converged to `1e-8`:

```
  r/L =  20.0   Cd 1.54308446   Cdp 1.01292921   Cdf 0.53015525   wake 2.10736   sep 53.8280  (1141 it)
  r/L =  40.0   Cd 1.51421004   Cdp 0.99248838   Cdf 0.52172166   wake 2.12192   sep 53.7171  (1177 it)
  r/L =  80.0   Cd 1.50249332   Cdp 0.98420530   Cdf 0.51828802   wake 2.12671   sep 53.6621  (1207 it)
```

**Magnitude.**

*Aerofoil.* Successive differences and their ratios:

| | r/c 20 | 40 | 80 | Δ(20→40) | Δ(40→80) | ratio |
|---|---|---|---|---|---|---|
| `Cl` | 0.744030 | 0.753231 | 0.757901 | +9.200e-3 | +4.670e-3 | **1.97** |
| `Cd` | 0.0135565 | 0.0125153 | 0.0119695 | −1.041e-3 | −5.458e-4 | **1.91** |
| `Cd_pressure` | 0.0060961 | 0.0050620 | 0.0045198 | −1.034e-3 | −5.422e-4 | 1.91 |
| `Cd_friction` | 0.0074604 | 0.0074533 | 0.0074497 | −7.1e-6 | −3.6e-6 | 1.97 |

The ratios are 1.9 to 2.0 against a predicted 2. The error is `O(1/R)`, exactly
as the vortex law requires, and it is confined to the *pressure* drag —
`Cd_friction` moves by 0.1% across a factor of four in domain size while
`Cd_pressure` moves by 35%.

Richardson extrapolation in `1/R`, using the *observed* exponent
`p = ln(ratio)/ln(2)` in each case rather than assuming 1:

| | observed `p` | limit as R → ∞ | error at R = 20 | at 40 | at 80 |
|---|---|---|---|---|---|
| `Cl` | 0.978 | 0.762715 | −2.45% | **−1.24%** | −0.61% |
| `Cd` | 0.932 | 0.011368 | +19.3% | **+10.1%** | +5.3% |
| `Cd_pressure` | 0.931 | 0.003922 | +55.4% | **+29.1%** | +15.2% |
| `Cd_friction` | 1.019 | 0.0074463 | +0.19% | +0.094% | +0.045% |

At the **default 40 chords**, the reported `Cd` of this aerofoil is ten per cent
above the value the same solver gives with the boundary taken to infinity, and
essentially all of it is in the pressure drag. `Cd_friction` — the quantity
Stage 3 spent its effort on — is untouched at 0.09%.

*Cylinder.* No circulation, so the leading term is the doublet and the wake
rather than a vortex. Successive differences `−2.887e-2`, `−1.172e-2`, ratio
2.464, i.e. `p = 1.30`. Richardson limit `Cd(inf) = 1.49449`, so the gate's
reported `1.51421` carries a **+1.32%** domain-truncation error.

That last number deserves saying plainly. The published band for the Re 40
cylinder is 1.50–1.58 for *unbounded* flow. The solver's own domain-independent
limit is 1.4945, below the band; the reported 1.5142 sits inside it, and the
difference between the two is the 40-chord truncation. The gate passes on a
number that carries an uncounted 1.3% domain error. Nothing about that is
dishonest — every code has a domain — but ASME V&V 20 requires it to be in
`u_num`, and at present it is neither measured nor reported.

**A caveat on the aerofoil numbers.** Changing `far_field_radius_ratio` changes
the mesh as well as the boundary: `geometric_layers` produces more layers, and
because `transition_distance` stays at one chord, all of the extra layers are
built by the polar blend. So the sweep conflates boundary placement with mesh
extent. Three things argue that the boundary dominates: the `1/R` scaling matches
the vortex law to within 5%; `Cd_friction`, which is a near-wall quantity
untouched by the far field's mesh, moves by 0.05% while `Cd_pressure` moves by
27%; and the same experiment on the cylinder, where there is no circulation,
gives a visibly different exponent (1.30 against 1.91). A clean separation would
hold the mesh fixed and change only the boundary condition, which is what the
correction below makes possible.

**The correct formulation.** Superpose the analytic point-vortex field on the
freestream at the boundary, the standard treatment since Thomas & Salas:

    u_far = U_inf + (Gamma / 2 pi) ( -(y - y_0), (x - x_0) ) / |r - r_0|²
    Gamma = (1/2) C_l U_inf c ,

with `C_l` taken from the previous outer iteration and `r_0` the aerodynamic
centre (quarter chord is the usual choice), applied on the inflow faces in place
of the bare freestream. The pressure on the outflow faces should carry the
matching Bernoulli correction rather than being pinned to zero:

    p_far = (1/2) rho ( U_inf² - |u_far|² ) .

The incompressible form above is what this solver needs; Thomas & Salas give the
compressible (Prandtl–Glauert-corrected) version and Vassberg & Jameson give the
grid-convergence evidence for how far out one must go without it. With the
correction the error falls to `O(1/R²)` and 20 to 40 chords becomes adequate;
without it, the measurements above say the boundary would have to go to several
hundred chords for `Cl` to be converged to 0.1%.

`build_ogrid`'s docstring currently says "Thirty to fifty reference lengths is
usual for a lifting case; less than about twenty and the boundary starts to
interfere with the circulation." The measurement says the interference at forty
is 1.24% in `Cl` and 10.1% in `Cd`; that sentence should carry those numbers.

**References.** Thomas, J. L. & Salas, M. D. (1986), "Far-Field Boundary
Conditions for Transonic Lifting Solutions to the Euler Equations", *AIAA
Journal* 24(7), 1074–1080. Vassberg, J. C. & Jameson, A. (2010), "In Pursuit of
Grid Convergence for Two-Dimensional Euler Solutions", *Journal of Aircraft*
47(4), 1152–1166 — grids to 150 chords, with and without the point-vortex
far-field influence. Usab, W. J. & Murman, E. M. (1983), "Embedded Mesh
Solutions of the Euler Equation Using a Multiple-Grid Method", AIAA 83-1946, for
the original vortex far field. Celik et al. (2008), *JFE* 130(7), 078001, for
the extrapolation procedure used above.

**Confidence.** CONFIRMED — measured on two geometries, with the `1/R` scaling
predicted in advance and recovered to within 5%.

**Cost.** Local to `bc.far_velocity` and `bc.far_pressure`, plus a channel for
the current `C_l` into `Boundaries` — which `Case` already computes every
iteration for its residual record. It changes every lifting result.

---

### F3 — The wall pressure in the force integral is a zeroth-order extrapolation

**Statement.** `post.compute_forces` takes the wall face pressure to be the
adjacent cell-centre value, which is an `O(h)` extrapolation feeding a scheme
claimed to be `O(h²)`, and it costs 0.14% of `Cd` on the validation gate.

**Location.** `fluidsolver/solver/post.py:139-141`:

```python
    # Zero normal pressure gradient at a wall, so the face value is the cell value.
    wall_pressure = state.pressure[:, 0]
    pressure_force = np.sum(wall_pressure[:, None] * area, axis=0)
```

**The mathematics.**

*Continuous statement.* The exact normal momentum balance at a solid wall, with
`u = 0` there, is

    dp/dn |_wall = mu (grad² u) . n,

because the convective term `rho (u . grad) u . n` vanishes identically at a
no-slip surface. So `dp/dn = 0` at the wall is correct *only* to the order at
which the viscous term is neglected — it is the boundary-layer approximation,
not an identity. The often-quoted curvature term `rho u_t² / R` is zero *at* the
wall for the same reason, and it is not the mechanism here.

*Discrete statement.* What the code needs is `p` on the wall *face*; what it has
is `p` at the first cell *centre*, a distance `y1` away. Taylor:

    p_face = p_cell - y1 (dp/dn)|_cell + (y1²/2)(d²p/dn²) + ...

`dp/dn` at the cell centre is **not** zero even when it is zero at the wall: the
first cell centre sits inside the layer where `u_t != 0`, so the balance there
carries both a curvature term and a viscous term. Using `p_cell` for `p_face`
therefore drops a term of order `y1 (dp/dn)|_cell = O(h)`.

*Order.* A zeroth-order extrapolation is `O(h)` pointwise. Integrated round the
body with weights proportional to `h`, the leading error is
`∮ y1 (dp/dn) n ds`, which does not cancel because `dp/dn` at the first cell
centre is not fore-aft symmetric — it scales with the local `u_t²`, largest at
the shoulder. So `Cd` inherits an `O(h)` error, and no amount of second-order
accuracy in the interior operators recovers it. This is the term that caps the
observed order of every force coefficient, and therefore of every GCI the project
intends to compute.

**Evidence.** On the converged Re 40 cylinder (`worst 9.992e-09`), the pressure
drag was re-integrated three ways from the same field: with the cell value, with
a linear extrapolation to the wall face through the first two cell centres, and
with a Lagrange quadratic through the first three.

```
converged in 1177 iters, worst 9.992e-09
Cd 1.514210039  Cdp 0.992488375  Cdf 0.521721664  Cl -1.450e-10

  first cell centre at y/D = 1.860e-03  (y0 1.8596e-03, y1 5.8520e-03, y2 1.0440e-02)
  Cd_pressure  cell value      0.992488375   (this is what the solver reports)
               linear extrap.  0.994573056   delta +2.085e-03 (+0.1377% of total Cd)
               quadratic       0.994655894   delta +2.168e-03 (+0.1431% of total Cd)
  max |p_wall - p_cell| / q    linear 4.0432e-03   quadratic 4.2373e-03
  normal momentum estimate rho*U1^2*y0/R  max/q = 3.4151e-06
```

The linear and quadratic reconstructions agree to `8e-5` in `Cdp`, so the
correction is well determined and is not an artefact of the extrapolation order.
The last line is worth reading: the *curvature* estimate is `3.4e-6 q`, three
orders below the `4.0e-3 q` actually present, confirming that the mechanism is
the viscous normal-momentum term and not centrifugal.

The order was then measured directly, on three systematically refined meshes
(`r = 1.5`, both directions, first layer scaled with them), each converged to
`1e-8`:

```
r=1.00  180x53  1177 it  Cd 1.514210039  Cdp(cell) 0.992488375  Cdp(quad) 0.994655894  gap 2.168e-03  y0 1.8596e-03
r=1.50  270x56  1861 it  Cd 1.514669624  Cdp(cell) 0.992910142  Cdp(quad) 0.994352498  gap 1.442e-03  y0 1.2393e-03
r=2.25  405x59  3223 it  Cd 1.514945209  Cdp(cell) 0.993164917  Cdp(quad) 0.994125697  gap 9.608e-04  y0 8.2603e-04

wall-pressure gap:  2.1675e-03  1.4424e-03  9.6078e-04
observed order of the gap: 1.005, 1.002
```

**1.005 and 1.002.** The wall-pressure term is first order to three significant
figures, which is what a zeroth-order extrapolation must be, and it is the only
quantity in this audit whose predicted order was recovered exactly.

**Magnitude.** On the gate's own mesh, `Cd_pressure` read at the wall rather than
at the cell centre is higher by `2.17e-03`, i.e. **0.143% of `Cd`**: 1.5142
becomes 1.5164.

One honest complication, which the refinement study exposes and which a
single-mesh measurement would have hidden. The Richardson limit of the *reported*
`Cd` is 1.515358 (F11), so the reported value at `r = 1` is `1.15e-03` **below**
the continuum answer, while the wall-pressure term is `2.17e-03` **above** it.
The two have opposite signs and comparable size: on this mesh the zeroth-order
wall pressure is partly cancelling other first-order errors. Correcting it alone
moves `Cd` from 1.5142 to 1.5164 and *increases* the error at this resolution,
from `−0.076%` to `+0.079%`.

Applying the correction to all three meshes gives the sequence 1.516378,
1.516112, 1.515906, whose observed order is 0.63 and whose Richardson limit is
1.515194 — against the uncorrected limit of 1.515358. The two limits agree to
`1.6e-04`, which is the check that the correction is right rather than merely
different: both sequences must extrapolate to the same continuum answer, and they
do, to one part in ten thousand.

The corrected sequence's lower observed order (0.63 against 1.26) is not the
correction making things worse. It is what happens when two first-order terms of
opposite sign are cancelling and one of them is removed: the surviving term is
then exposed rather than masked. On the cylinder that surviving term is not F4 —
the mesh is orthogonal — so it is something else, and identifying it is work this
audit did not do.

The conclusion is therefore not "fix this and `Cd` improves". It is that a
first-order term worth `0.14%` sits in every reported force, that it happens to
be cancelling something else of similar size that has not been identified, and
that neither is measured. That is precisely the situation ASME V&V 20 exists to
forbid, and it is invisible without a mesh family.

**The correct formulation.** Reconstruct the face value from the cell value and
the cell gradient, which the solver already computes:

    p_face = p_P + (grad p)_P . d_{P->f}

with `d_{P->f} = faces.wall.delta`. That is `O(h²)` and costs one dot product.
The gradient must be the one built with the *correct* wall value, which makes it
a one-step fixed point (seed with `p_face = p_P`, rebuild, apply); a single pass
is enough because the correction is `O(h)`. The same treatment belongs in
`post.surface_data`, whose `pressure_coefficient` is `state.pressure[:, 0] /
dynamic` and carries the identical error into every plotted `Cp`.

**References.** Ferziger & Perić (2002), *Computational Methods for Fluid
Dynamics*, 3rd ed., §7.1 and §8.6 on boundary-value reconstruction and the order
of surface integrals. Roache (1998), *Verification and Validation in
Computational Science and Engineering*, ch. 5, on why a first-order boundary
treatment defeats an interior scheme of higher order. Oberkampf & Roy (2010),
*Verification and Validation in Scientific Computing*, §5.3.

**Confidence.** CONFIRMED — derived, and measured on the project's own gate.

**Cost.** Local: `post.compute_forces` and `post.surface_data`. It will move the
gate's `Cd` from 1.5142 to about 1.5164 and every regression that pins those
digits will have to be re-baselined, which is the honest consequence and not a
reason to leave it.

---

### F4 — The Rhie-Chow damping omits the non-orthogonal correction

**Statement.** `simple.face_fluxes` forms the compact pressure difference with
the orthogonal factor `g` alone and subtracts from it a smooth gradient dotted
with the *full* area vector; on a non-orthogonal face those are not the same
operator, and the difference leaves a spurious mass flux that is `O(h)` where the
intended damping is `O(h³)`.

**Location.** `fluidsolver/solver/simple.py:392-419`:

```python
        compact = (
            state.pressure - np.roll(state.pressure, 1, axis=0)
        ) * self.faces.i_faces.diffusion_factor
        smooth = np.sum(
            self.faces.i_faces.interpolate(grad_p, np.roll(grad_p, 1, axis=0))
            * self.faces.metrics.face_i_area,
            axis=-1,
        )
        flux_i = self.fluid.density * (
            np.sum(interpolated * self.faces.metrics.face_i_area, axis=-1)
            - d_i * (compact - smooth)
        )
```

and the identical construction for the `j` faces at `simple.py:406-420`.

**The mathematics.**

Write the face area vector in the decomposition `faces._decompose` already uses,

    S = g d + T,    g = |S|² / (d . S),    T . S = 0,

with `d` the centroid-to-centroid vector. Let `theta` be the non-orthogonality
angle between `d` and `S`. Then

    g = |S| / (|d| cos theta),    |T| = |S| tan theta,

and `T` lies in the face plane.

*Continuous statement.* The Rhie-Chow damping is the difference between the
compact and the interpolated pressure *gradient normal flux*, and it must vanish
whenever the two agree — in particular for any field whose second and higher
derivatives vanish, i.e. any linear `p`.

*Discrete statement, as coded.* For a linear field `p = G . x`,

    p_N - p_P = G . d
    compact   = g (G . d)
    smooth    = G . S
    compact - smooth = g (G . d) - G . (g d + T) = -G . T.

So the coded damping term for a linear pressure field is

    -rho D_f (compact - smooth) = +rho D_f (grad p)_f . T,

which is **not zero**, and vanishes only when `T = 0`, i.e. on an orthogonal
mesh.

*Discrete statement, done consistently.* The compact operator must be the same
one the diffusion assembly uses — implicit orthogonal part plus explicit cross
term:

    compact_full = g (p_N - p_P) + (grad p)_f . T,

whereupon, for linear `p`,

    compact_full - smooth = g (G . d) + G . T - G . S = 0

identically, and for a general smooth `p`,

    compact_full - smooth = g [ (p_N - p_P) - (grad p)_f . d ] = O(h³ p'''),

which is the third-derivative damping the docstring says it is.

*Order.* With `a_P ~ rho U |S|` in the convection-dominated region and
`V ~ h |S|`, the mobility `D_f = V / a_P ~ h / (rho U)`, so the spurious flux is

    Phi_spur ~ rho . (h / rho U) . |grad p| . |S| tan theta ~ h² |grad p| tan theta / U,

against a physical face flux `Phi ~ rho U |S| ~ rho U h`. The ratio is

    Phi_spur / Phi ~ (h / L) tan theta ,

first order in `h`, and it does not vanish under refinement of a self-similar
mesh family because `theta` does not. The consistent term, by the same estimate,
is `O(h³)` relative. The scheme is therefore formally **first order on any
non-orthogonal mesh**, whatever the interior operators do.

**Evidence.** Set an exactly linear pressure field `p = 1.7 x - 0.9 y` on each
mesh — the gradient operator reproduces it to `7.6e-12`, so the field is exact —
and evaluate the coded `compact - smooth`:

```
  cylinder  i-faces max|compact-smooth| 3.8030e-08   |residue + gradp.cross| 4.780e-14
            j-faces max|compact-smooth| 8.2787e-09   |residue + gradp.cross| 2.515e-15
            residue / |smooth|   i: mean 0.0000 median 0.0000 p99 0.0000 max 0.0000
            residue / |smooth|   j: mean 0.0000 median 0.0000 p99 0.0000 max 0.0000
  naca      i-faces max|compact-smooth| 5.0676e-01   |residue + gradp.cross| 5.435e-14
            j-faces max|compact-smooth| 1.0040e-01   |residue + gradp.cross| 8.395e-14
            residue / |smooth|   i: mean 0.8774 median 0.0001 p99 9.3784 max 7943.4095
            residue / |smooth|   j: mean 1.8233 median 0.0015 p99 9.5073 max 20895.0708
```

Two things are established at once. The residue equals `-(grad p)_f . T` to
`8e-14`, confirming the algebra above exactly. And on the NACA mesh it reaches
**9.4 times** the physical pressure-difference term on the worst 1% of faces,
for a pressure field on which a consistent damping term is identically zero.

The cylinder line is the other half of the point: on the mesh the project
validates against, the residue is `4e-9` relative — zero to rounding. The
cylinder O-grid is exactly orthogonal (measured: mean and peak non-orthogonality
0.0000 degrees). **The validation gate is structurally incapable of detecting
this class of error**, and the same is true of the manufactured-solution suite
(F11), which runs on the same orthogonal polar mesh.

On the **converged** NACA 2412 solution (SST, 900 iterations, `worst 2.3e-06`),
with the real pressure field and the real mobility `D_f = V / a_P`:

```
  i-faces: mean|physical flux|          1.05156e+01
           mean|Rhie-Chow term as coded| 2.69223e-04  (0.003% of the flux)
           mean|consistent RC term|      2.61712e-05  (0.000%)
           mean|spurious residue|        2.49652e-04  (0.002% of the flux)
           max |spurious| / |flux| on a single face: 0.340
           ratio |spurious| / |consistent RC term|: mean 9.54  p99 2326.50
  j-faces: mean|spurious| 9.40076e-05   = 0.005% of mean|flux| 1.87391e+00
  total continuity imbalance introduced: sum|div(spurious)| / sum|flux_far| = 1.3691e-04
```

**Magnitude.** Zero on the cylinder — `3.6e-09` relative, which is rounding.

On the NACA 2412 mesh the Rhie-Chow term as coded is **9.5 times larger than the
term it is supposed to be**, and 93% of what it contains is the spurious
residue rather than the third-derivative damping. In absolute terms the spurious
flux is 0.002% of the mean physical face flux, which is small; on the worst
single face it is **34%** of the physical flux, which is not. Removing it would
change the flux field by `1.37e-04` of the domain's mass throughput, which is two
orders above the `1e-06` convergence tolerance the run was stopped at — so this
is not a term buried under the iteration error.

Where it lives is the point. The mesh's non-orthogonality is not uniform:

```
  marched layers 54 of 89
  j-faces, marched region : mean 0.248  p99 5.159  peak 20.558 deg
  j-faces, blended region : mean 9.502  p99 49.082  peak 57.387 deg
  i-faces, marched region : mean 0.106  peak 47.005 deg
  i-faces, blended region : mean 9.623  peak 60.069 deg
  all faces               : mean 3.932  peak 60.069  fraction > 30 deg 2.92%
  quality report says     : mean 3.932  peak 60.069
```

The marched near field is almost perfectly orthogonal (0.1 to 0.25 degrees mean)
and the spurious term is negligible there. The blended far field — 35 of 89
layers, 39% of the cells — averages 9.5 degrees with a 99th percentile near 50,
and `tan(50 deg) = 1.19`, so on those faces the spurious term exceeds the
physical pressure difference it is derived from. The quality report's headline
"3.9 degrees mean" is the average of those two populations and describes neither.
Its warning threshold at 60 degrees reports "0.00% of faces", while 2.92% of
faces are past 30 degrees, where `tan` is already 0.58.

No single force coefficient can be attributed to this in isolation, because
removing it changes the converged flux everywhere at once. The defensible
statement is: on any non-orthogonal mesh the flux definition carries an `O(h)`
error, the discretisation is therefore first order there whatever the interior
operators do, and the observed order of `Cd` measured in F11 — 1.26, not 2 — is
consistent with exactly that.

**The correct formulation.** Replace `compact` with the same split the diffusion
operator uses:

    compact_i = g_i (p_P - p_W) + (grad p)_f . T_i
    flux_i    = rho [ u_f . S - D_f (compact_i - (grad p)_f . S) ]
              = rho [ u_f . S - D_f g_i ( (p_P - p_W) - (grad p)_f . d ) ]

The second line is the same thing written once the cancellation is done, and it
is the cheaper implementation: the whole damping term is `D_f g` times the
difference between the compact pressure difference and the interpolated gradient
projected on `d` — a quantity that is manifestly zero for a linear field on any
mesh. `faces.InteriorFaces` already carries `diffusion_factor` (`g`), `cross`
(`T`) and `delta` (`d`), so nothing new has to be computed.

**References.** Rhie, C. M. & Chow, W. L. (1983), "Numerical Study of the
Turbulent Flow Past an Airfoil with Trailing Edge Separation", *AIAA Journal*
21(11), 1525–1532. Jasak, H. (1996), *Error Analysis and Estimation for the
Finite Volume Method with Applications to Fluid Flows*, PhD thesis, Imperial
College, §3.3 (over-relaxed decomposition) and §3.9. Demirdžić, I. &
Muzaferija, S. (1995), "Numerical method for coupled fluid flow, heat transfer
and stress analysis using unstructured moving meshes with cells of arbitrary
topology", *Computer Methods in Applied Mechanics and Engineering* 125, 235–255.
Ferziger & Perić (2002), §8.6.

**Confidence.** CONFIRMED — derived in closed form and verified numerically to
machine precision on the project's own meshes.

**Cost.** Local to `simple.face_fluxes` (both face families) and to the matching
compact operator in `simple.apply_correction`, which must use the same form or
the flux update and the pressure equation will disagree again — the failure mode
`_far_field_coupling`'s docstring already records. Structural in consequence:
every reported coefficient on a non-orthogonal mesh moves.

---

### F5 — The converged answer depends on `relax_velocity`

**Statement.** Rhie-Chow builds its mobility from the *under-relaxed* momentum
diagonal, so `D_f` is proportional to `alpha_u` and the damping term — which does
not vanish at convergence — carries the relaxation factor into the fixed point.
Four docstrings assert the opposite.

**Location.** `simple.py:306-310`:

```python
            coefficients.under_relax(field, self.numerics.relax_velocity)
        ...
        return components[0], components[1], components[0].centre
```

and `simple.py:384`, `simple.py:535`:

```python
        mobility = self.volume / diagonal
```

The claim under audit, `simple.py` module docstring:

> "and because the relaxation is applied in Patankar's implicit form, it changes
> only the path taken, never the converged answer."

repeated at `linalg.Coefficients.under_relax`, `simple.CflRamp` and
`sst.KOmegaSST.update`.

**The mathematics.**

Patankar's implicit under-relaxation replaces

    a_P phi_P = sum a_nb phi_nb + b

by

    (a_P/alpha) phi_P = sum a_nb phi_nb + b + ((1-alpha)/alpha) a_P phi_P^old.

At a fixed point `phi_P = phi_P^old` and the two added terms cancel exactly, so
*for the solution of that equation* the claim is true, and
`Coefficients.under_relax` implements it correctly (the source is added after the
diagonal has been divided, so the increment is `(1-alpha) (a_P/alpha) phi^old`,
which is `((1-alpha)/alpha) a_P phi^old` as intended).

The claim fails one level up. `momentum()` returns `components[0].centre`, the
**relaxed** diagonal `a_P/alpha_u`, and `face_fluxes` builds

    D_f = V / (a_P / alpha_u) = alpha_u V / a_P,

so the mass flux

    F = rho [ u_f . S - D_f ( compact - smooth ) ]

is linear in `alpha_u` through its second term. That term is not an iteration
artefact: it is part of the definition of the converged face flux, and it is what
suppresses the checkerboard mode. Therefore the converged flux field, and with it
the converged velocity, pressure and forces, depend on `alpha_u`.

Note which relaxation factor is implicated. `alpha_p` appears only in
`state.pressure += relax_pressure * correction`, and `p' -> 0` at convergence, so
`alpha_p` genuinely cannot move the fixed point. That asymmetry is the signature
of the mechanism and is what the measurement below tests.

**Evidence.** Laminar cylinder at Re 40, `scheme="linear"`, every run converged to
`tolerance = 1e-9` — two orders tighter than the gate:

```
alpha_u 0.70  alpha_p 0.30    1363 it  res 9.84e-10  Cd 1.514209454  Cdp 0.992488333  Cdf 0.521721121  wake 2.121923  sep 53.71710
alpha_u 0.70  alpha_p 0.15    1364 it  res 9.94e-10  Cd 1.514224767  Cdp 0.992499178  Cdf 0.521725589  wake 2.121922  sep 53.71710
alpha_u 0.70  alpha_p 0.45    1361 it  res 9.99e-10  Cd 1.514215272  Cdp 0.992492452  Cdf 0.521722820  wake 2.121923  sep 53.71710
alpha_u 0.55  alpha_p 0.30    2415 it  res 9.96e-10  Cd 1.514350779  Cdp 0.992617191  Cdf 0.521733589  wake 2.121921  sep 53.71644
alpha_u 0.40  alpha_p 0.30    4215 it  res 9.98e-10  Cd 1.514485661  Cdp 0.992747480  Cdf 0.521738181  wake 2.121919  sep 53.71577
```

(A fourth velocity relaxation, `alpha_u = 0.9`, diverged at iteration 126 and is
not part of the comparison.)

**Magnitude.** Read the two families separately, because the difference between
them is the whole argument.

*Velocity relaxation.* `Cd` moves monotonically: 1.514209 at `alpha_u = 0.70`,
1.514351 at 0.55, 1.514486 at 0.40. The successive differences are `1.41e-04`
and `1.35e-04` for equal steps of 0.15 — **linear in `alpha_u` to within 5%**,
which is exactly what `D_f = alpha_u V / a_P` predicts and is far stronger
evidence than the endpoints alone. The full span is `+2.76e-04`, i.e.
**+0.0182%** in `Cd`, and 94% of it is in the pressure drag. Extrapolating the
straight line to `alpha_u -> 0` gives 1.51486, so the total reach of the
mechanism is about `6.5e-04` in `Cd`.

*Pressure relaxation.* `alpha_p` at 0.15, 0.30 and 0.45 gives 1.514225,
1.514209, 1.514215 — a spread of `1.5e-05`, an order of magnitude smaller than
the `alpha_u` effect, and **not monotone**. That is the signature of the
convergence floor rather than of a real dependence, and it is what the derivation
predicts: `alpha_p` appears only in `state.pressure += relax_pressure *
correction`, and `p' -> 0` at the fixed point, so it cannot move the answer.

The asymmetry is the proof. If the dependence were an artefact of stopping at a
finite residual, both factors would show it equally. One does and one does not,
and the one that does is the one Rhie-Chow's mobility is built from.

The wake length is unmoved at `2.1219` across the whole sweep; the separation
angle moves `0.0013` degrees.

Read against what the project reports: the gate prints `Cd 1.5142`, and at
`alpha_u = 0.4` the same converged solver prints `1.5145`. The fourth decimal is
not a property of the discretisation. On the cylinder that is all it is, because
the mesh is orthogonal and the damping term is only the third-derivative one; on
a non-orthogonal mesh the term carrying `alpha_u` is the much larger spurious
residue of F4, and the sensitivity will be correspondingly larger. This is a
first-order question for a project that intends ASME V&V 20 uncertainty
quantification, because a numerical parameter that moves the answer is an input
uncertainty that has to be carried in `u_num`.

**The correct formulation.** Two standard remedies, in increasing order of
correctness:

1. Build the Rhie-Chow mobility from the **unrelaxed** diagonal `a_P`, keeping
   the relaxed one for the velocity correction. This removes the leading
   `alpha_u` dependence but not the dependence hidden in `u_f` itself.
2. Choi's / Pascau's correction: retain the previous-iteration face velocity
   explicitly,

       F^m = rho [ u_f^m . S - D_f ( compact - smooth )^m
                   + (1 - alpha_u) ( F^{m-1}/rho - u_f^{m-1} . S ) ],

   so that the relaxation-dependent part telescopes and the fixed point is
   independent of `alpha_u` by construction. This is the same device used to
   remove the time-step dependence of Rhie-Chow in unsteady runs, and Stage 4
   will need it anyway.

Whichever is chosen, the four docstrings that assert relaxation-independence
must be corrected, because they are what stopped this being noticed.

**References.** Majumdar, S. (1988), "Role of Underrelaxation in Momentum
Interpolation for Calculation of Flow with Nonstaggered Grids", *Numerical Heat
Transfer* 13(1), 125–132. Choi, S. K. (1999), "Note on the Use of Momentum
Interpolation Method for Unsteady Flows", *Numerical Heat Transfer Part A* 36(5),
545–550. Yu, B., Tao, W.-Q., Wei, J.-J., Kawaguchi, Y. et al. (2002),
"Discussion on Momentum Interpolation Method for Collocated Grids of
Incompressible Flow", *Numerical Heat Transfer Part B* 42(2), 141–166. Cubero, A.
& Fueyo, N. (2007), "A compact momentum interpolation procedure for unsteady
flows and relaxation", *Numerical Heat Transfer Part B* 52(6), 507–529.
Pascau, A. (2011), "Cell face velocity alternatives in a structured colocated
grid for the unsteady Navier–Stokes equations", *International Journal for
Numerical Methods in Fluids* 65(7), 812–833. Van Doormaal, J. P. & Raithby, G. D.
(1984), "Enhancements of the SIMPLE Method for Predicting Incompressible Fluid
Flows", *Numerical Heat Transfer* 7(2), 147–163.

**Confidence.** CONFIRMED — derived, and measured at a residual two orders below
the gate's.

**Cost.** Local to `simple.face_fluxes` if remedy 1; local plus one stored field
in `State` if remedy 2. Every pinned regression digit moves.

---

### F6 — The omega-equation production is the form the SST-2003 paper had to correct

**Statement.** `sst._solve_omega` uses `gamma rho S^2`; the NASA TMR records that
this is an erratum in Menter, Kuntz & Langtry (2003) and that SST-2003 specifies
`gamma P̃_k / nu_t` with the *limited* production, the limiter applying to both
equations. The definition is wrong; the measured effect on this case is small.

**Location.** `fluidsolver/solver/turbulence/sst.py:338`:

```python
        coefficients.source += gamma * density * strain**2 * self.volume
```

against `sst.py:280-288`, where the k equation correctly limits its own
production.

**The mathematics.**

The two forms coincide whenever the limiter is inactive:

    P̃_k / nu_t = (mu_t S²) / (mu_t / rho) = rho S²,

so `gamma P̃_k / nu_t = gamma rho S²` exactly. They differ precisely where

    P_k = mu_t S²  >  10 beta* rho k omega,

in which case the specified source is `gamma . 10 beta* rho k omega / nu_t` and
the coded one is `gamma rho S²`, larger by the same ratio the limiter was cutting.
They also differ in the wall-adjacent row, where the k equation substitutes the
two-layer strain (`bc.wall_velocity_gradient`) and the omega equation does not —
but that row's `omega` is replaced by the identity, so there it is moot.

The consequence of the coded form is that Menter's stagnation-point limiter,
which the project's Stage 0 went to some trouble to make active, is switched off
for one of the two equations it is specified to act on. Where it fires, `omega`
is over-produced relative to `k`, and `mu_t = rho a1 k / max(a1 omega, S F2)` is
correspondingly suppressed.

**Evidence.** NASA Turbulence Modeling Resource, Menter SST page, fetched
2026-09-04, quoted verbatim:

> "In the omega equation (2nd part of eqn (1) in the paper), the production term
> was incorrectly given as α ρ S² (using the paper's notation)."
>
> "Instead, it should have read α P̃ₖ / νₜ (again using the paper's notation)."
>
> "In this expression, the Pk term has a tilde over it, which refers to the
> limited value of the k production term min(P, 10 β* ρ ω k)."
>
> "Another minor difference from (SST) is that the production limiter is used for
> both k and omega equations, and the constant is changed from 20 to 10."

Two NACA 2412 runs, identical but for `KOmegaSST._solve_omega` replaced in the
scratch script with

    source += gamma rho min( S_prod² , 10 beta* rho k omega / mu_t ) V

which is `gamma P̃_k / nu_t` written so that `mu_t -> 0` is harmless:

```
RESULT base:   iters 1007  worst 9.9385e-07  Cl 0.7535096  Cd 0.01250649  Cdp 0.00505258  Cdf 0.00745391
RESULT omegaP: iters 1002  worst 9.9857e-07  Cl 0.7535790  Cd 0.01250910  Cdp 0.00505375  Cdf 0.00745535
RESULT base:   mut/mu peak 186.16  limiter-fires 8.59% of cells
RESULT omegaP: mut/mu peak 186.95  limiter-fires 17.11% of cells
```

**Magnitude.** `Cl` +0.0092%, `Cd` +0.021%, `Cd_friction` +0.019%. That is
between one and two orders below F1 and comparable to F5. The reason it is small
here is visible in the last line: the limiter fires in 8.6% of cells in the base
run — matching the 9% the plan records — but overwhelmingly in the freestream,
where both arguments of the `min` are near zero and which of them wins is
immaterial. It would not be small on a bluff body at high Reynolds number, where
the limiter's purpose is to control a genuine stagnation-point anomaly, and it is
exactly the case the plan's Stage 0 entry describes.

The brief nominated this as "the single most consequential item in this section".
It is not, on this case; F1 is, by a factor of four hundred. But it is a
*definitional* error where F1 is a calibration one, and it is the one that would
be caught first by a NASA TMR verification exercise, because it makes the model
something other than the `SST-2003` the code claims.

**The correct formulation.**

    P̃_k = min( mu_t S² , 10 beta* rho k omega )
    k equation:      source += P̃_k
    omega equation:  source += gamma rho min( S² , 10 beta* rho k omega / mu_t )

the second being `gamma P̃_k / nu_t` with the division arranged so that a
vanishing `mu_t` selects `S²`.

**References.** NASA Turbulence Modeling Resource, "Menter Shear Stress Transport
Model", `https://tmbwg.github.io/turbmodels/sst.html` (accessed 2026-09-04),
"SST-2003" section and the erratum note. Menter, F. R., Kuntz, M. & Langtry, R.
(2003), "Ten Years of Industrial Experience with the SST Turbulence Model", in
*Turbulence, Heat and Mass Transfer 4*, Begell House, eq. (1) and eq. (5).

**Confidence.** CONFIRMED — definition from the primary source, magnitude
measured.

**Cost.** Local: `sst._solve_omega` gains three lines and needs `state.eddy_
viscosity` passed through, which it already holds.

---

### F7 — The freestream turbulence defaults are two orders outside the recommended band

**Statement.** `Freestream.turbulence_intensity = 0.001` with
`eddy_viscosity_ratio = 1.0` puts the ambient `k` about 30 times and the ambient
`mu_t/mu` about 100 times above the top of the range the NASA TMR specifies for
external aerodynamics.

**Location.** `fluidsolver/solver/fluid.py:102-103`:

```python
    turbulence_intensity: float = 0.001
    eddy_viscosity_ratio: float = 1.0
```

with `turbulent_kinetic_energy()` and `specific_dissipation()` immediately below.

**The mathematics.**

For the NACA 2412 case (`U = 30`, `L = 1`, `rho = 1.225`, `mu = 1.81e-5`,
`Re_L = 2.03e6`):

    k_inf     = 1.5 (I U)²          = 1.5 (0.001 x 30)² = 1.350e-03 m²/s²
    omega_inf = rho k_inf / (mu . 1) = 1.225 x 1.35e-3 / 1.81e-5 = 91.4 s^-1

against the TMR's stated bands:

    U/L  <  omega_farfield  <  10 U/L        =>   30 < omega < 300      : 91.4 is inside
    1e-5 U²/Re_L < k_farfield < 0.1 U²/Re_L  =>   4.4e-9 < k < 4.4e-5   : 1.35e-3 is 30x above
    1e-5 < mu_t_farfield/mu < 1e-2                                       : 1.0 is 100x above

*Decay.* In the freestream `F1 -> 0`, production vanishes, and the transported
equations reduce to

    U dk/dx = -beta* k omega,    U domega/dx = -beta_2 omega²

with solutions

    omega(x) = omega_0 / (1 + beta_2 omega_0 x / U)
    k(x)     = k_0 (1 + beta_2 omega_0 x / U)^(-beta*/beta_2)

With `beta_2 = 0.0828`, `beta* = 0.09`, `beta*/beta_2 = 1.087`, and a far field at
40 chords so `x = 40 m`:

    beta_2 omega_0 x / U = 0.0828 x 91.4 x 40 / 30 = 10.09
    omega falls by 11.1x,  k falls by 11.1^1.087 = 13.7x,
    mu_t/mu = (k/omega) x const falls from 1.00 to 0.81.

So the turbulence arrives at the body with an intensity of 0.027% rather than the
0.1% set, and an ambient eddy viscosity ratio of about 0.8 — still eighty times
the top of the TMR band. The `omega` value itself is inside the band, so the
problem is the `k` level, i.e. the intensity.

**Evidence.** NASA Turbulence Modeling Resource, SST page, fetched 2026-09-04,
quoted:

> "U∞/L < ω_farfield < 10 U∞/L"
>
> "10⁻⁵U∞²/Re_L < k_farfield < 0.1 U∞²/Re_L"
>
> "the combination of the two farfield values should yield a freestream turbulent
> viscosity between 10⁻⁵ and 10⁻² times freestream laminar viscosity"
>
> "Note that the turbulence variables decay (sometimes dramatically) from their
> set values in the farfield for external aerodynamic problems."

The decay analysis above is a closed-form solution of the model's own equations.
Measured in the converged field, sampling along the upstream centreline:

```
  set at the boundary:  k 1.350000e-03   omega 9.136740e+01   mu_t/mu 1.0000
  omega * L / U_inf = 3.046   (TMR band 1 .. 10)
  k / (U^2/Re_L) = 3.046e+00   (TMR band 1e-5 .. 1e-1)
    x/c ~ -   8.00   k 1.6282e-04 (0.121 of set)   omega 1.2771e+01   mu_t/mu 0.8629
    x/c ~ -   2.00   k 1.3573e-04 (0.101 of set)   omega 1.0794e+01   mu_t/mu 0.8511
    x/c ~ -   1.00   k 1.3294e-04 (0.098 of set)   omega 1.0551e+01   mu_t/mu 0.8528
```

`k` arrives at the body at 0.098 of its set value against 0.073 predicted by the
closed-form decay, and `mu_t/mu` at 0.853 against 0.81 — agreement to within the
accuracy of a one-dimensional decay estimate applied to a two-dimensional
approach. The turbulence intensity at the body is `sqrt(2 k / 3)/U = 0.031%`,
not the 0.1% the user set, and the difference is a function of where the boundary
was placed.

**Magnitude.** No direct effect on a reported coefficient in a fully turbulent
run has been isolated — an ambient `mu_t/mu` of 0.8 is small against a
boundary-layer peak of 186. The consequence is elsewhere and it is structural:

1. It puts a floor under the eddy viscosity everywhere in the domain, including
   inside any potential core and in the near wake, which is where Stage 4's
   shedding will live.
2. It makes the answer depend on the far-field radius through the decay length,
   which is a numerical parameter masquerading as a physical one. This compounds
   F2.
3. **It blocks Stage 5.** The one-equation gamma transition model takes its
   onset criterion from the local turbulence intensity. With `Tu` decaying from
   0.1% at the boundary to 0.027% at the body — a factor of 3.7 that depends on
   where the boundary was put — transition onset would be set by the mesh rather
   than by the flow. The plan calls transition "required, not optional"; this is
   a prerequisite for it.

**The correct formulation.** Either adopt the TMR's ambient values,

    omega_amb = 5 U_inf / L,    k_amb = 1e-6 U_inf²

which for this case give `omega = 150`, `k = 9e-4`, `mu_t/mu = 7.4e-3` — inside
the band — or, better, implement the `SST-sust` sustaining terms, which add
`beta* rho k_amb omega_amb` and `beta rho omega_amb²` to the two source terms so
that the destruction is exactly cancelled at the ambient level and the decay
disappears. The TMR specifies both, and the second removes the far-field-radius
dependence entirely. `eddy_viscosity_ratio = 1.0` should not be the default
whichever is chosen.

**References.** Spalart, P. R. & Rumsey, C. L. (2007), "Effective Inflow
Conditions for Turbulence Models in Aerodynamic Calculations", *AIAA Journal*
45(10), 2544–2553. NASA Turbulence Modeling Resource, "Menter Shear Stress
Transport Model", farfield boundary conditions and the `SST-sust` variant,
`https://tmbwg.github.io/turbmodels/sst.html` (accessed 2026-09-04).

**Confidence.** CONFIRMED — the code's values against the published band, and the
decay derived in closed form from the model's own equations.

**Cost.** Two defaults in `fluid.Freestream` if the first route; two extra source
terms in `sst._solve_k` and `sst._solve_omega` if the second. Local either way.
It changes every turbulent result.

---

### F8 — The marching Newton iteration is truncated, and it costs ten layers of mesh

**Statement.** `hyperbolic._march_one_layer` runs a fixed four Newton passes with
no convergence test; raising it to forty carries the march ten layers further,
and the handover doc's own trap entry says march depth *is* mesh quality.

**Location.** `fluidsolver/mesh/hyperbolic.py:217-224`:

```python
def _march_one_layer(
    previous: np.ndarray,
    step: np.ndarray,
    thickness: float,
    equalisation: float,
    epsilon: float,
    second_difference: float,
    iterations: int = 4,
) -> np.ndarray:
```

The docstring states the problem precisely and then does not act on it:

> "A single linearisation leaves a relative error of order ``(step / radius)^2``
> per layer. That is negligible near the wall but not in the far field, where
> geometric growth makes the layers comparable to the local radius of curvature,
> and it accumulates over the march. Re-linearising a few times drives each layer
> onto the true nonlinear solution instead."

**The mathematics.** The system solved per layer is

    f(r) = ( x_xi x_eta + y_xi y_eta , x_xi y_eta - y_xi x_eta ) = (0, V)

with `r_xi` evaluated on the layer being solved for, so `f` is quadratic in the
unknowns. A Newton iteration on a quadratic converges quadratically once inside
its basin; the error after `n` passes from a starting error `e_0` is
`e_n ~ C^(2^n - 1) e_0^(2^n)`. With `e_0 = O((step/radius)²)` and `step/radius`
approaching order one in the far field, four passes is not obviously enough and
the docstring says so. There is no residual test, so nothing detects the case
where it is not.

Crucially, the failure does not show up as a wrong node position — it shows up
through `max_width_ratio`, the guard that stops the march. An under-converged
layer carries more width variation, trips the guard sooner, and hands more of the
mesh to the polar blend, which is where all of the non-orthogonality lives.

**Evidence.** The NACA 2412 wall line at zero incidence (240 points, y+ 1 first
layer — the unrotated body, so 52 layers where the 5-degree case that
`build_case` produces marches 54), marched with `iterations` monkey-patched and
everything else identical:

```
  iterations  4  layers 52  (reference)
  iterations  8  layers 49  max|dr| 1.1520e-03  max|dr|/r 1.879e-03  last-layer max|dr| 1.1520e-03 (thickness 1.951e-02)
  iterations 16  layers 54  max|dr| 2.5474e-03  max|dr|/r 3.949e-03  last-layer max|dr| 2.5474e-03 (thickness 2.963e-02)
  iterations 40  layers 62  max|dr| 2.9288e-03  max|dr|/r 4.455e-03  last-layer max|dr| 2.9288e-03 (thickness 2.963e-02)
```

and the resulting mesh at four passes:

```
  marched 52 of 89 layers; notes: ['march stopped after 52 of 63 near-field layers
  (wall distance 0.2278 rather than 1); the analytic far field took over there.']
```

**Magnitude.** Four passes reach 52 of the 63 requested near-field layers, at a
wall distance of 0.228 where 1.0 was asked for. Forty passes reach 62 — within
one layer of the request. The node positions differ by at most `4.5e-3` of the
local radius, which is small; the ten extra layers are not, because each one is a
layer the polar blend does not have to build. Note the non-monotonicity at eight
passes (49 layers): the guard is a threshold on a max-over-the-layer, so it is
noisy in the layer count. The trend from 4 to 16 to 40 is clear.

This does not by itself change a force coefficient, which is why it is ranked
where it is. It changes the mesh that F4's spurious flux lives on, and it is the
cheapest available improvement to the seam that Stage 6 is otherwise proposing to
solve structurally.

**The correct formulation.** Iterate to a residual rather than a count:

    repeat:
        solve the linearised system for r^(m+1)
        until  max_i | r^(m+1)_i - r^(m)_i |  <  tol . thickness
        or a pass ceiling is reached

with `tol` of order `1e-6` and a ceiling of, say, 20, falling back to the current
behaviour if the ceiling is hit. The cost is a sparse solve per extra pass on a
2Ni system, which is negligible against the rest of meshing.

**References.** Steger, J. L. & Chaussee, D. S. (1980), "Generation of Body-Fitted
Coordinates Using Hyperbolic Partial Differential Equations", *SIAM Journal on
Scientific and Statistical Computing* 1(4), 431–437. Chan, W. M. & Steger, J. L.
(1992), "Enhancements of a Three-Dimensional Hyperbolic Grid Generation Scheme",
*Applied Mathematics and Computation* 51, 181–205. Chan, W. M., Rogers, S. E.,
Nash, S. M. et al., *HYPGEN User's Manual*, NASA TM 108791.

**Confidence.** CONFIRMED — measured.

**Cost.** One loop in `hyperbolic._march_one_layer`. Local. It changes every mesh,
so the gate must be re-run — on the cylinder it will not move, because a circle
marches exactly.

---

### F9 — The global mass balance is still active at convergence

**Statement.** `bc.enforce_global_mass_balance` rescales every outflow face
multiplicatively, and at convergence the factor is not one, so it is a boundary
condition rather than a solvability repair.

**Location.** `fluidsolver/solver/bc.py:321-341`, in particular:

```python
        balanced[~entering] *= inflow / outflow
```

**The mathematics.**

The stated justification is:

> "The pressure-correction equation is a discrete Poisson problem, and it has a
> solution only if its source integrates to zero -- that is, only if the boundary
> fluxes balance."

That is the compatibility condition of a **pure Neumann** Poisson problem. This
one is not pure Neumann: `pressure_correction` adds
`self._far_field_coupling(...)` to the diagonal of every cell behind an outflow
face, which is a Dirichlet coupling to `p' = 0`. A matrix with any Dirichlet row
is non-singular and needs no compatibility condition. The stated reason for the
rescaling does not apply.

What the rescaling actually does. Let `F_i` be the raw extrapolated flux on
outflow face `i`, `M_in` the (exactly known) inflow total. The code sets

    F_i  ->  F_i . (M_in / M_out),   M_out = sum over outflow faces of F_i.

This is a *proportional* redistribution: a face already carrying a large outflow
absorbs most of the correction, a face carrying almost none absorbs almost none.
The alternatives are a redistribution by face area (`F_i -> F_i + (M_in - M_out)
|S_i| / sum|S_i|`), which spreads it evenly, or by `|F_i|`, which is what this
is. There is no strong argument between them *if* the correction vanishes; there
is if it does not.

It does not. At the fixed point, `p' = 0`, so the far-field flux is exactly the
rescaled `F`. Nothing forces the raw `F` — built from the *extrapolated*
velocities `u[:, -1]` on outflow faces — to balance the inflow, so the scale
factor settles at some value `c != 1` and stays there. The converged outflow
velocity is therefore `c` times what the interior produced, uniformly, and the
docstring's "the correction vanishes as the solution converges" is not what
happens.

**Evidence.** Converged Re 40 cylinder (`worst 9.992e-09`), factor evaluated from
the converged state:

```
  raw inflow  8.000000000e+01
  raw outflow 8.002020240e+01
  multiplicative factor applied to every outflow face: 0.999747534
  i.e. the converged outflow velocity is rescaled by -0.0252%
  outflow faces 90 of 180
  smallest / largest outflow face flux  2.620e-02 / 1.409e+00
```

**Magnitude.** The converged outflow is rescaled by `-2.52e-04`. That is small,
and on this case it is harmless. Two things make it worth recording anyway. It is
active at convergence, so by the project's own standard ("anything active at
convergence is part of the model whether or not it is labelled as one") it is a
model term that nothing documents. And the guard

```python
        if outflow <= 0.0 or inflow <= 0.0:
            return far_flux
```

silently returns an unbalanced flux exactly when the situation is worst — a
transient in which almost the whole boundary is inflow, or a start-up in which
the sign pattern has not settled. In that case the rescaling does nothing and no
diagnostic says so. The factor `M_in/M_out` is also unbounded as `M_out -> 0`.

**The correct formulation.** Enforce compatibility on the *source* of the
pressure equation rather than on the boundary flux, which is the standard
treatment and is exact:

    b_P  ->  b_P - (1/V_total) . V_P . sum over cells of b

(a zero-mean projection of the divergence field). Where there is at least one
fixed-pressure face this is unnecessary and can be skipped entirely; where there
is none — the case the guard is really worried about — it is what makes the
system solvable, and it does not touch the boundary condition. If a boundary
correction is kept as well, distribute by face area, not by flux, and raise a
diagnostic when `|c - 1|` exceeds a threshold, because a large `c` means the
outflow extrapolation is failing and that is worth knowing.

**References.** Ferziger & Perić (2002), §7.2 and §8.4 on compatibility of the
pressure equation and on outflow conditions. Patankar, S. V. (1980), *Numerical
Heat Transfer and Fluid Flow*, §6.7 on mass-flux correction at outlets.

**Confidence.** CONFIRMED — measured on the converged gate case.

**Cost.** Local to `bc.enforce_global_mass_balance` and one line in
`simple.pressure_correction`.

---

### F10 — The reported moment coefficient has the opposite sign to the convention

**Statement.** `Forces.moment_coefficient` returns the `+z` component of the
moment about the reference, which for a freestream along `+x` and lift along
`+y` is the negative of the standard nose-up-positive aerodynamic `Cm`.

**Location.** `fluidsolver/solver/post.py:146-148`:

```python
    lever = faces.wall.centre - moment_reference
    element = wall_pressure[:, None] * area + traction * length[:, None]
    moment = float(np.sum(lever[:, 0] * element[:, 1] - lever[:, 1] * element[:, 0]))
```

**The mathematics.** `Forces.lift` is `total[1]`, so lift is `+y`; `Forces.drag`
is `total[0]`, so the freestream runs `+x` and the leading edge is at smaller
`x`. The expression above is `r_x F_y - r_y F_x`, the `+z` component of
`r x F` with `z` out of the page.

Take a unit lift applied one length *ahead* of the reference, `r = (-1, 0)`,
`F = (0, 1)`:

    M = (-1)(1) - (0)(0) = -1.

Lift acting ahead of the moment reference is a **nose-up** moment. The standard
aerodynamic convention (x aft along the chord, pitching moment positive nose-up)
would report `+1` here. The code reports `-1`. The two conventions differ by a
sign, uniformly, for every contribution.

**Evidence.**

```
  a unit LIFT (+y) applied one length AHEAD of the reference gives
    the code's moment integrand = -1.0
```

and, on the NACA 2412 at 5 degrees with the moment reference at the rotated
contour centroid, the converged runs report `Cm = -0.0825`. A NACA 2412 has
`Cm_ac ~ -0.05`; about a reference aft of the aerodynamic centre at roughly
`0.42 c` with `Cl = 0.75`, the standard convention gives
`Cm ~ -0.05 + 0.17 x 0.75 = +0.08`. The magnitude agrees, the sign is inverted,
which is what the unit test above predicts.

**Magnitude.** Every reported `Cm` — `Case.summary()`, `Forces.moment_
coefficient`, and whatever the GUI shows — has the wrong sign. No other reported
quantity is affected: `Cl`, `Cd` and their splits are unsigned by this.

**The correct formulation.**

    moment = -sum( lever_x element_y - lever_y element_x )

or, equivalently, document the convention explicitly as "positive
counter-clockwise in the mesh frame" and stop calling it `Cm`. Since the project
intends comparison against published NACA data, where `Cm` is nose-up positive,
the sign flip is the right choice.

There is a second, smaller point in the same lines. `moment_reference` defaults
to `contour.centroid` — the area centroid of the *rotated* body. Published
aerofoil moments are quoted about the quarter chord. Nothing is wrong with the
centroid as a reference, but it must be stated, and a quarter-chord option is
what a user comparing against data will want.

**References.** Anderson, J. D. (2016), *Fundamentals of Aerodynamics*, 6th ed.,
§1.5 and §4.3, for the sign convention. Abbott, I. H. & von Doenhoff, A. E.
(1959), *Theory of Wing Sections*, for the quarter-chord reference and the
tabulated `Cm_ac` values this would be compared against.

**Confidence.** CONFIRMED — derived from the code and verified by a unit force.

**Cost.** One sign. Local. Two tests will need their expectation flipped.

---

### F11 — The manufactured-solution check does not cover the meshes the solver runs on

**Statement.** The order-of-accuracy verification runs on an orthogonal,
unstretched polar mesh, excludes both boundary rows, and exercises the operators
in isolation, so it cannot support the claim that the discretisation is second
order on a body-fitted mesh.

**Location.** `tests/test_solver.py:80-90`:

```python
def uniform_mesh(surface_points: int, outer: float = 3.0):
    """A circle in a circular far field, with uniform radial spacing."""
    radial = outer - 0.5
    grid = build_ogrid(
        circle(1.0, surface_points),
        first_layer=radial / (surface_points // 3),
        far_field_radius=outer,
        growth=1.0,
    )
```

and `tests/test_solver.py:393-401`:

> "Only interior cells are measured. Boundary rows use a one-sided gradient which
> is first order there by construction; that is a known and accepted property of
> the scheme, not a defect, and including them would mask the interior order."

**The mathematics and what it leaves out.** The claim being verified is that
`apply(phi) - source` reproduces `div(rho u phi) - div(Gamma grad phi)` to second
order. On the mesh above:

- **Non-orthogonality is identically zero.** Measured on the two finer meshes of
  the sequence: mean and peak both `0.0000` degrees, and `0.0006` peak on the
  coarsest. So `faces.cross` is zero, the entire non-orthogonal correction path
  in `add_diffusion` is dead code under test, and F4's residue — which is exactly
  `-(grad p)_f . T` — cannot appear.
- **Stretching is switched off.** `growth=1.0`, uniform radial spacing, measured
  aspect ratio 2.51 and expansion ratio 1.08 to 1.27. The working meshes carry
  expansion ratios to 4.77 and aspect ratios to 453. Truncation error on a
  stretched mesh loses one order in the leading term unless the stretching is
  smooth, and nothing here tests that.
- **Skewness is negligible.** Measured `3.8e-03` on the finest MMS mesh against
  `0.477` on the NACA mesh — two orders apart.
- **Both boundary rows are excluded** (`error[:, 1:-1]`), and the exclusion is
  argued for on the grounds that they are first order by construction. That is
  true and it is the point: a scheme with first-order boundary rows is not
  globally second order unless the boundary error is `O(h²)` *in the integrated
  quantity*, which F3 shows it is not for the wall pressure.
- **The coupled system is not verified at all.** There is no manufactured
  solution for the Navier–Stokes system with the pressure–velocity coupling in
  the loop, none for the boundary conditions, and none for the SST source terms.
  Every one of F1, F4, F5, F6 and F9 lives in code the MMS never touches.

**Evidence.** The mesh, from the test source above; and the properties of the
three meshes the MMS actually runs on, measured by importing `uniform_mesh`
directly from the test module:

```
uniform_mesh( 48): 48x16  marched 7   non-orth mean 0.0001 peak 0.0006 deg  skew peak 2.987e-02  aspect peak 2.511  expansion peak 1.2703
uniform_mesh( 96): 96x32  marched 13  non-orth mean 0.0000 peak 0.0000 deg  skew peak 1.073e-02  aspect peak 2.513  expansion peak 1.1449
uniform_mesh(192): 192x64 marched 26  non-orth mean 0.0000 peak 0.0000 deg  skew peak 3.821e-03  aspect peak 2.513  expansion peak 1.0752
```

against the mesh `build_case` produces for the primary use case:

```
naca 2412 mesh: 240x89  non-orth mean 3.932 peak 60.069 deg  skew peak 0.477
                aspect peak 452.7  expansion peak 4.767
```

The verification mesh has an aspect ratio of 2.5 where the working mesh has 453,
a skewness of 0.004 where the working mesh has 0.48, and no non-orthogonality at
all where the working mesh averages 3.9 degrees and reaches 60. It is not a
harder version of the working mesh; it is a different object.

The observed-order tests themselves pass:

```
252 passed in 259.87s
```

**The measurement the project has never made.** Three systematically refined
cylinder meshes at Re 40, refinement ratio `r = 1.5` applied to the surface
points, the layer count and the first-layer thickness together, each converged to
`tolerance = 1e-8`:

```
r=1.00  180x53  1177 it  Cd 1.514210039  Cdf 0.521721664  wake 2.121923  sep 53.71710  y0 1.8596e-03  (298s)
r=1.50  270x56  1861 it  Cd 1.514669624  Cdf 0.521759481  wake 2.142097  sep 53.74419  y0 1.2393e-03  (493s)
r=2.25  405x59  3223 it  Cd 1.514945209  Cdf 0.521780293  wake 2.157075  sep 53.75262  y0 8.2603e-04  (1337s)

Cd as reported:     1.514210039  1.514669624  1.514945209
Cd differences:     4.5958e-04  2.7559e-04
observed order of Cd: 1.261
Richardson limit Cd:  1.515358
```

**The observed order of the reported drag coefficient is 1.261.** Not 2. The
sequence is monotone and the differences are clean, so this is not a
non-asymptotic artefact — it is the order the solver delivers on the case it is
validated against.

**Magnitude.** `README.md`'s status table says

> "Finite-volume discretisation | working, second order (verified by manufactured
> solution)"

The verification that exists supports a narrower claim: *second order for the
isolated convection and diffusion operators, on an orthogonal unstretched polar
mesh, away from the boundaries*. That is a real and valuable result — it caught
two sign errors in the deferred correction that had reduced the scheme to first
order — but it is not the claim in the table, and the measurement above shows the
gap. On a solved field, with the boundary conditions, the pressure–velocity
coupling and the force integral in the loop, the order is 1.26.

That number is consistent with the two first-order terms this audit identifies:
F3, whose wall-pressure gap is measured at order `1.005, 1.002` on these same
three meshes, and F4, whose spurious flux is `O(h)` wherever the mesh is
non-orthogonal — which on the cylinder is nowhere, so on *this* case F3 is the
mechanism and F4 is not. On an aerofoil both would be present.

For Celik's procedure this matters twice over. The GCI formula
`GCI_21 = F_s |f2 - f1| / (|f1| (r^p - 1))` uses the *observed* `p`, so
`p = 1.261` gives, on these meshes,

    GCI_21 = 1.25 x 2.7559e-4 / (1.514945 x (1.5^1.261 - 1)) = 0.034%

against `0.018%` if second order were assumed — a band 1.87 times wider. And a
code whose observed order is not close to its formal order fails Roache's own
precondition for the extrapolation to mean anything in the first place.

**Additional caution from the same study.** The wake length has *not* converged.
2.121923, 2.142097, 2.157075; differences `2.017e-2` and `1.498e-2`, ratio 1.347,
observed order **0.735**, Richardson limit **2.200**. The gate reports
`wake L/D 2.1219` — the coarsest of the three — and quotes it against a band of
2.10 to 2.35 that spans a measured 2.13 and a computational cluster at 2.21–2.35.
The solver's own grid-converged answer is 2.20, which sits with the computations;
the reported 2.1219 sits with the experiment, and it does so because the mesh is
coarse. That is a coincidence being read as agreement, and it is the same class
of error the handover's traps section warns about.

The separation angle is better behaved: 53.71710, 53.74419, 53.75262, observed
order 2.88, Richardson limit 53.756 — 0.039 degrees above the reported figure,
which is well inside the 0.25-degree method-dependence of F12.

**The correct formulation.** Three additions, in order of value:

1. **Run the existing MMS on a non-orthogonal, stretched mesh.** The cheapest
   version is the same circle with `growth=1.15` and the far field built by the
   polar blend, which is what `build_ogrid` produces by default; the expensive
   version is a deliberately sheared mesh with a prescribed non-orthogonality
   angle. This alone would have caught F4.
2. **Include the boundary rows in a separate measurement**, reporting their order
   explicitly rather than excluding them, so that a change from `O(h)` to `O(h²)`
   at the wall is visible.
3. **A full-system manufactured solution.** Add a body-force source to the
   momentum equations and a source to the continuity equation, choose an analytic
   `(u, v, p)`, and measure the order of the *solved* fields, not of the
   assembled operator. Salari & Knupp give the recipe for exactly this, including
   the treatment of the pressure's arbitrary constant.

**References.** Roache, P. J. (1998), *Verification and Validation in
Computational Science and Engineering*, Hermosa, ch. 3 and 5. Salari, K. &
Knupp, P. (2000), *Code Verification by the Method of Manufactured Solutions*,
Sandia report SAND2000-1444. Oberkampf, W. L. & Roy, C. J. (2010),
*Verification and Validation in Scientific Computing*, CUP, ch. 6. Celik, I. B.,
Ghia, U., Roache, P. J. et al. (2008), "Procedure for Estimation and Reporting of
Uncertainty Due to Discretization in CFD Applications", *Journal of Fluids
Engineering* 130(7), 078001.

**Confidence.** CONFIRMED — the test source and the mesh properties are both
measured.

**Cost.** Tests only, but the results will force changes elsewhere.

---

### F12 — The separation angle is reported to two more digits than it has

**Statement.** `post.separation_points` takes the traction direction from the
first-cell velocity; a one-sided wall gradient on the same converged field moves
the separation angle by 0.25 degrees, and the gate reports three decimals.

**Location.** `fluidsolver/solver/post.py:189-192` (`wall_shear_stress` with
`boundaries=None`) and `post.py:200-218`.

**The mathematics.** For a laminar run the traction is
`mu u_t(y1) / y1`, whose sign is the sign of the first-cell tangential velocity.
The true wall shear is `mu (du_t/dy)|_0`. In a separating boundary layer the
profile is inflected: `du_t/dy` changes sign at the wall before `u_t(y1)` does,
because the reversed region grows outward from the wall. The two therefore
disagree, and the disagreement is `O(y1)` — first order in the wall spacing.

**Evidence.** Converged Re 40 cylinder, same field, two ways of taking the sign:

```
  separation from first-cell velocity      53.71710 deg
  separation from one-sided wall gradient  53.97023 deg
  reported by the validation gate          53.71710 deg
```

The one-sided estimate fits `u_t(y) = a y + b y²` through the wall and the first
two cell centres and takes `a`.

**Magnitude.** 0.253 degrees, on a quantity the gate prints as `53.717` and
compares against a published band of 52–54. The published band is wide enough
that nothing fails; the last two digits of `53.717` are not supported by the
discretisation.

There is a second, smaller issue in the same function:

```python
    crossing = np.flatnonzero(np.sign(along) != np.sign(following))
```

`np.sign` returns `0` for an exact zero, so a face where the traction is exactly
zero registers as two crossings rather than one. On a symmetric case the
stagnation faces can hit this; `separation_angle` filters them by angle, which
works for a cylinder and would not for an aerofoil.

**The correct formulation.** Take the sign from a one-sided wall gradient rather
than from the first-cell value:

    (du_t/dy)|_0 ~ ( u_t1 y2² - u_t2 y1² ) / ( y1 y2² - y2 y1² )

using the first two cell centres and the no-slip wall, which is `O(h²)`; and
report the angle to one decimal, or attach the mesh-resolution uncertainty to it.
Replace the `np.sign` comparison with `along * following < 0`.

**References.** Simpson, R. L. (1989), "Turbulent Boundary-Layer Separation",
*Annual Review of Fluid Mechanics* 21, 205–234, on the definition of separation as
the zero of wall shear. Ferziger & Perić (2002), §8.6.

**Confidence.** CONFIRMED — measured on the gate case.

**Cost.** Local to `post.wall_shear_stress` and `post.separation_points`. It moves
the gate's separation angle by a quarter of a degree, still inside the band.

---

### F13 — A zero-gradient condition is imposed along `d`, not along `n`

**Statement.** `operators.Gradient` implements a Neumann boundary by passing the
adjacent cell value, which asserts that the derivative along the
centroid-to-face vector `d` is zero, not that the *normal* derivative is; on the
NACA mesh those directions differ by up to 31.6 degrees on the wall.

**Location.** `fluidsolver/solver/operators.py:131-136`:

> "``wall`` and ``far_field`` are the values on those boundary faces. For a
> zero-gradient condition pass the adjacent cell value: the difference is then
> zero, which is precisely what a vanishing normal gradient asserts."

Callers that rely on it: `sst.update` for `k` (`grad_k = self.gradient(state.k,
state.k[:, 0], far_k)`), `simple._transpose_stress` for the far-field viscosity,
`simple.face_fluxes` and `simple.momentum` for the pressure on inflow faces, and
`post.vorticity`.

**The mathematics.** The least-squares stencil minimises

    sum_N w_N [ (grad phi . d_N) - (phi_N - phi_P) ]²

with `d_N = faces.wall.centre - centroid` on the wall row. Setting
`phi_wall = phi_P` makes that residual `grad phi . d`, so the fit is driven
towards `grad phi . d = 0`. What a zero-flux condition requires is
`grad phi . n = 0` with `n` the face normal. Decomposing
`d = |d|(cos(theta) n + sin(theta) t)`,

    grad phi . d = 0    =>    grad phi . n = -tan(theta) (grad phi . t),

so the reconstructed normal derivative is not zero but `tan(theta)` times the
tangential one. The condition is exact only where `d` is parallel to `n`.

**Evidence.**

```
  cylinder  angle(delta, wall normal): mean 0.000 deg  peak 0.000 deg
  naca      angle(delta, wall normal): mean 0.530 deg  peak 31.581 deg
```

**Magnitude.** `tan(0.53 deg) = 0.0093` on average and `tan(31.58 deg) = 0.615`
at the worst face. So on a typical NACA wall face the spurious normal gradient is
under 1% of the tangential one — negligible — and on the handful of worst faces
it is 60% of it. Since the tangential gradient of `k` along a wall is far smaller
than its normal gradient, the absolute error is small everywhere; this is ranked
low for that reason. It is recorded because it is exact on the cylinder and
therefore invisible to every test the project has, and because the docstring
states it as an identity when it is an approximation.

**The correct formulation.** Impose the condition on the flux rather than through
a face value, which `add_diffusion` already does correctly by passing
`wall_value=None`. For the *gradient* operator, the consistent statement is to
constrain the fit,

    grad phi = P grad phi_unconstrained,   P = I - n n^T

on rows carrying a zero-flux face, or equivalently to supply the face value

    phi_face = phi_P + (grad phi . (d - (d . n) n))

i.e. the cell value transported along the tangential part of `d` only, evaluated
with the previous iteration's gradient. The second is one line and lags by one
iteration, which at convergence is exact.

**References.** Demirdžić & Muzaferija (1995), *CMAME* 125, 235–255, on boundary
treatment for non-orthogonal cells. Jasak (1996), §3.3 and §3.8.

**Confidence.** CONFIRMED — the misalignment is measured; the consequence is
derived and bounded, not measured in a coefficient.

**Cost.** Local to `operators.Gradient` plus its callers' boundary arguments.

---

### F14 — Two different wall distances are in use and they disagree by 15%

**Statement.** `bc.py` uses `faces.wall.wall_normal_distance` (perpendicular
distance from cell centroid to its own wall face) while `sst.py` and
`fields.State.uniform` use `metrics.wall_distance` (true minimum distance to the
surface polyline); on the NACA mesh they differ by up to 15% in the first cell
row, and the `omega` seed and the `omega` boundary condition therefore disagree
by 38%.

**Location.** `bc.py:120`, `bc.py:246` versus `sst.py:110` and `fields.py:105-108`:

```python
        distance = np.maximum(faces.metrics.wall_distance, 1e-300)
        omega = freestream.specific_dissipation(fluid) + 6.0 * fluid.kinematic_viscosity / (
            0.075 * distance**2
        )
```

against

```python
        distance = self.faces.wall.wall_normal_distance
        viscous = (
            _OMEGA_WALL_FACTOR * self.fluid.kinematic_viscosity
            / (_BETA_1 * distance**2)
        )
```

**The mathematics.** Both quantities are correct for their own purpose. `y+`, the
friction velocity and the effective wall viscosity all describe a flux through a
*particular* face, so the perpendicular distance to that face is right. SST's
blending functions need the distance to the nearest wall, whichever face that is,
so the polyline minimum is right. The defect is that `State.uniform` seeds
`omega` with the *asymptote formula from `bc.py`* evaluated on the *distance from
`metrics`*, mixing the two.

Since `omega ~ 1/d²`, a ratio `r` in the distances is a ratio `r^-2` in `omega`.

**Evidence.**

```
  wall_normal_distance vs metrics.wall_distance in the first cell row:
    ratio mean 0.997931  min 0.851920  max 1.000000
```

`0.85192^-2 = 1.378`.

**Magnitude.** The initial `omega` field in the worst wall cells is 38% below the
value the boundary condition will impose on the first iteration. On the cylinder
the ratio is exactly 1 and there is no discrepancy at all, which is why the gate
does not see it. The effect is transient — the wall row is overwritten by the
identity substitution on the first `model.update` — so this is a start-up
inconsistency, not a converged-solution error, and it is ranked accordingly.
`State.uniform`'s own docstring explains at length why the `omega` seed matters,
which is the reason to fix it.

**The correct formulation.** Seed `omega` from `Boundaries.wall_turbulence`
directly for the wall row, and from the same `_OMEGA_WALL_FACTOR` and the same
distance definition elsewhere; do not duplicate the constant `6.0` and the
constant `0.075` in `fields.py`.

**References.** None needed; this is an internal consistency defect.

**Confidence.** CONFIRMED — measured.

**Cost.** A few lines in `fields.State.uniform`. Local.

---

### F15 — Wall-distance sub-sampling is coarse in the first cell row

**Statement.** `metrics.compute_metrics(wall_samples=8)` over-estimates the wall
distance of first-row cells by up to 17% relative to a converged sub-sampling.

**Location.** `fluidsolver/mesh/metrics.py:100-110` and `metrics._wall_distance`.

**The mathematics.** The surface is a polyline; the distance is taken to a
KD-tree of points sampled along it at `1/8` of each segment. A cell centroid a
distance `y1` off the wall, sitting opposite the middle of a gap of length
`h_s/8` between samples, reads `sqrt(y1² + (h_s/16)²)` instead of `y1`. With
`y1 ~ 6e-6` and a surface spacing `h_s ~ 1e-2` on the NACA mesh, `h_s/16 ~ 6e-4`
would be a hundredfold error; that it is only 17% means the sampling usually
lands well, and the 17% is the tail.

**Evidence.** NACA 2412 mesh, distances recomputed at 8, 32, 128 and 512 samples
per segment:

```
    samples   32   max|dd|/d 3.779e-02   max|dd| 3.552e-06   worst at wall distance 9.399e-05
    samples  128   max|dd|/d 1.481e-01   max|dd| 3.552e-06   worst at wall distance 9.399e-05
    samples  512   max|dd|/d 1.481e-01   max|dd| 3.554e-06   worst at wall distance 9.399e-05
    leading-edge cells only: max|d8-d512|/d = 0.000e+00
    first cell row overall : max|d8-d512|/d = 1.738e-01
```

**Magnitude.** 17.4% in the worst first-row cell; 3.8% at the worst cell
anywhere, at a wall distance of `9.4e-05` which is inside the buffer layer. The
brief asked specifically about the leading edge: measured, the leading-edge cells
are exact to 512 samples (`0.000e+00`), because that is where the surface
resampler clusters points most finely. The error is at the *trailing* edge and on
the long flat segments, not at the nose.

Effect on the model: the viscous argument `500 nu / (d² omega)` of the SST
blending function `F1` moves by 8% at the worst cell, and `F1` is saturated at 1
throughout that region, so `tanh(x⁴)` is 1 either way. No measurable effect on any reported quantity. Recorded because
it is cheap to remove and because it would matter if the blending functions were
ever evaluated where they are not saturated.

**The correct formulation.** Sample at a spacing tied to the *wall-normal* scale
rather than at a fixed count: `samples = ceil(h_s / (2 y1_min))`, capped. Or
compute the exact point-to-segment distance analytically, which is a closed-form
projection and removes the question entirely — for a polyline of `Ni` segments
and `Ni x Nj` centroids that is affordable if restricted to the nearest few
segments found by the existing KD-tree.

**References.** None needed.

**Confidence.** CONFIRMED — measured. No effect on a reported quantity, also
measured.

**Cost.** Local to `metrics._wall_distance`.

---

### F16 — The mesh page reports a correlation estimate as the "achieved" y+

**Statement.** `gui/pages/mesh.py` labels the flat-plate correlation's prediction
`achieved` and prints it as the mesh's `y+`; on the NACA 2412 case it reads 1.00
where the solver delivers 0.30 to 2.36.

**Location.** `fluidsolver/gui/pages/mesh.py:185-193`:

```python
        achieved = spacing.y_plus_of(
            first_layer,
            session.freestream.velocity,
            ...
        )
        lines = [
            f"first cell        {first_layer:.4g} m  (y+ approx {achieved:.2f})",
```

`spacing.y_plus_of` inverts `first_layer_thickness`, so it returns the target
back, computed from `C_f = 0.026 Re^(-1/7)` — a flat-plate correlation with no
knowledge of the pressure gradient or the stagnation point. `health.assess` uses
the same function for `estimated_y_plus`, but calls it *estimated*, which is
honest.

**The mathematics.** The correlation gives a single `u_tau` for the whole body.
The real `u_tau` varies by roughly an order of magnitude between the stagnation
point and the suction peak, so a single number cannot describe the distribution
whatever correlation is used. `spacing.friction_velocity`'s own docstring says
so:

> "The solver reports the ``y+`` distribution actually achieved, and that is the
> number to trust."

The mesh page then prints the other one and calls it achieved.

**Evidence.** Same case, both numbers:

```
  mesh page  : y+ approx 1.00      (spacing.y_plus_of on the first layer)
  solver     : y+ 0.301 .. 2.357   (post.surface_data, from the blended tau_w)
```

**Magnitude.** A factor of 3.3 across the surface, presented as a single exact
figure at the moment the user is deciding whether the mesh is adequate for the
wall treatment. `y+` is the one quantity on which the validity of the whole SST
wall condition rests, so a misleading readout of it is worth a line even though
no computed number changes.

**The correct formulation.** Call it `y+ target (flat-plate estimate)` on the
mesh page, and add the achieved range to the solve page from
`post.surface_data`, which already computes it consistently with `bc`'s friction
velocity (verified: `surface_data` takes `u_tau = sqrt(tau_w/rho)` from
`boundaries.wall_shear`, which is the same `friction_velocity` the wall condition
uses, and the same perpendicular distance).

**References.** Schlichting & Gersten (2017), *Boundary-Layer Theory*, 9th ed.,
§18.2, for the `1/7`-power correlation's range of validity.

**Confidence.** CONFIRMED.

**Cost.** A label and one extra readout. Trivial.

---

### F17 — The five residuals `worst` takes a maximum over are not commensurable

**Statement.** `Residuals.worst` takes `max(u, v, continuity, k, omega)`, but the
five are normalised by scales of entirely different character; in particular 57%
of the `omega` normaliser and 65% of its imbalance come from the
identity-substituted wall row, which is a prescribed value rather than a solved
equation.

**Location.** `fluidsolver/solver/fields.py:159-161`:

```python
    @property
    def worst(self) -> float:
        return max(self.u, self.v, self.continuity, self.k, self.omega)
```

and `linalg.Coefficients.residual`, `linalg.py:106-121`.

**The mathematics.** The normaliser is

    N = sum_P ( |A phi - A phi_bar|_P + |b - A phi_bar|_P ),   phi_bar = mean(phi),

which is the standard finite-volume scaling — it is exactly OpenFOAM's
`normFactor` and is not an invention here. Three of the brief's questions about
its robustness have different answers.

*Field mean near zero.* Harmless. For `v` on a symmetric cylinder `phi_bar -> 0`,
so `A phi_bar -> 0` and `N -> sum(|A phi| + |b|)`, which is the natural
normaliser. Nothing degenerates.

*Near-null mode.* Also harmless for the equations here, because the pressure
correction — the only near-singular operator — is excluded from `worst`
deliberately, and its own residual is measured at the obtained correction rather
than at zero (a defect the project has already fixed and documented).

*A row replaced by the identity.* This is the one that bites. After
`_fix_wall_row` the wall row reads `1 . omega_0 = omega_wall`. Its contribution to
the imbalance is `|omega_wall(u^m, v^m) - omega_0^{m-1}|` — the amount the
*boundary condition itself* moved since the last iteration, not a measure of how
well any transport equation was solved. Its contribution to the normaliser is
`2 |omega_wall - omega_bar|`, and with `omega_wall ~ 8e6` against
`omega_bar ~ 1.2e5` that is `~1.6e7` per wall cell over 240 cells.

**Evidence.** The residual was recomputed inside `_solve_and_clip`, on the same
coefficients the solver uses, with the wall row's share separated out:

```
 100 k      reported 5.1741e-04   |b-A phi| 4.6928e+00   normaliser 9.0698e+03   wall row is 0.30% of the normaliser and 0.03% of the imbalance   by sum|b| it would read 1.1239e-01
 100 omega  reported 5.5164e-05   |b-A phi| 3.8997e+05   normaliser 7.0692e+09   wall row is 53.69% of the normaliser and 70.37% of the imbalance   by sum|b| it would read 2.0103e-04
 300 k      reported 1.6122e-05   |b-A phi| 4.6872e-01   normaliser 2.9073e+04   wall row is 0.34% of the normaliser and 0.02% of the imbalance   by sum|b| it would read 4.9636e-03
 300 omega  reported 8.1738e-07   |b-A phi| 5.4631e+03   normaliser 6.6836e+09   wall row is 56.90% of the normaliser and 66.20% of the imbalance   by sum|b| it would read 2.8113e-06
 400 omega  reported 3.1030e-07   |b-A phi| 2.0742e+03   normaliser 6.6847e+09   wall row is 56.89% of the normaliser and 65.32% of the imbalance   by sum|b| it would read 1.0674e-06
```

and, over 300 iterations of the same case, which equation is actually the worst:

```
    Ux       0.3%
    Uy      73.0%
    mass     0.0%
    k       26.7%
    omega    0.0%
  median omega residual 1.774e-05 vs median momentum residual 1.028e-04 (ratio 1.73e-01)
```

**Magnitude.** The `omega` residual is never the binding constraint (0.0% of
iterations) and runs at 0.17 of the momentum residual. Whether that is because
`omega` genuinely converges fastest or because its normaliser is inflated by a
prescribed row cannot be separated from the reported number: normalising the same
imbalance by `sum|b|` instead gives 3.4 times more for `omega` and 310 times more
for `k`. Nothing in a reported coefficient changes, but the run-stopping
criterion is a maximum over five numbers that do not measure the same thing, and
one of them is largely measuring its own boundary condition.

`k` is clean — its wall row contributes 0.3% — so this is specifically an `omega`
issue and specifically a consequence of the identity substitution, which is
otherwise the right thing to do.

**The correct formulation.** Exclude the substituted rows from both sums:

    imbalance = |b - A phi| on rows that are solved for
    N         = the same restriction of the existing normaliser

i.e. mask the wall row in `Coefficients.residual` when `fixed_wall` was applied,
exactly as `_fix_wall_row` already knows which row that is. That reports the
residual of the system actually being solved, which was the stated intent of the
Stage-0 fix (README item 3) and is not quite what it achieved. Separately, state
in `Residuals` that the five figures are each normalised against their own
equation and that `worst` is a convenience, not a norm.

**References.** Jasak, H. (1996), *Error Analysis and Estimation for the Finite
Volume Method*, §3.9, for the normalisation. Ferziger & Perić (2002), §5.3, on
residual scaling and stopping criteria.

**Confidence.** CONFIRMED — measured directly on the coefficients the solver
assembles.

**Cost.** A mask argument to `Coefficients.residual` and one call site in
`sst._solve_and_clip`. Local.

---

## Examined and found sound

This section is a deliverable. Each item was checked against a source or a
measurement and holds; the point is that the ground does not need covering again.
Three of them contradict a premise the brief put to me, and those are marked.

### S1 — The scheme is globally conservative and the fluxes telescope exactly

Faces are stored once and shared with opposite sign by the two cells either side
(`metrics.cell_face_areas`, `operators.divergence`), so the sum of the cell
divergences must reduce to the boundary flux with no interior residue. Tested
with random face fluxes on the NACA mesh:

```
  sum of divergence over all cells  -4.393832e+00
  net boundary flux                 -4.393832e+00
  difference                        7.461e-14
```

Exact to rounding. Conservation is not in question anywhere in this report.

### S2 — Subtracting the continuity imbalance from the diagonal *restores* boundedness — it does not break it

**This contradicts the brief.** The brief asks whether
`coefficients.centre -= divergence(flux_i, flux_j)` in `add_convection` "can drive
`a_P` below the sum of the neighbour coefficients on a cell with large positive
imbalance". The algebra says the opposite, and so does the measurement.

For the upwind operator, with `F_f` the outward face fluxes,

    a_P = sum_f max(F_f, 0),    a_nb = -max(-F_f, 0),
    a_P - sum |a_nb| = sum_f max(F_f,0) - sum_f max(-F_f,0) = sum_f F_f,

which is exactly the divergence. So *before* the subtraction the row's Scarborough
slack equals the cell's mass imbalance and is negative wherever the cell is a net
sink; *after* subtracting the divergence the slack is identically zero and the
operator is on the boundedness limit everywhere. Measured with random fluxes on
the NACA mesh:

```
  without the subtraction  min(a_P - sum|a_nb|) -8.563e+00   rows below zero 10655 / 21360
  with the subtraction     min(a_P - sum|a_nb|) -1.332e-15   rows below zero 0 / 21360
```

(the row count uses a `1e-12` tolerance, so the `-1.3e-15` minimum is rounding.)
Half the mesh violates the criterion without the term and none does with it. The
term is also consistent with the pressure correction that follows: both remove the
same `phi_P sum_f F_f`, and it vanishes at convergence in the discrete sense —
`sum_f F_f` is precisely the quantity `simple.iterate` drives to zero and reports
as `continuity`, not merely a continuous-level argument. The docstring's account
of why it is there (freestream `k` growing to 570 times its inlet value) is
correct.

### S3 — The least-squares gradient is exact for a linear field, including on both boundary rows

The brief asks whether replacing the missing cell neighbour by the boundary face
still reproduces a linear field on the wall and far-field rows. It does, because
the stencil uses the true geometric offset to the face centre and the true
difference to the face value:

```
  cylinder  max|grad-G| field 2.376e-14   wall row 2.376e-14   far row 3.553e-15
  naca      max|grad-G| field 7.620e-12   wall row 7.620e-12   far row 7.216e-15
```

for `G = (1.7, -0.9)`. The choice of inverse-square weighting and the rejection of
Green–Gauss are both argued correctly in the docstring; the Green–Gauss failure
mode described (skewness error entering divided by the cell volume, `|S|/V ~ 1e5`
on a boundary-layer cell) is real.

The separate question of what a *zero-gradient* condition asserts on that stencil
is F13, and is a different matter.

### S4 — The limiter argument is the standard van Leer ratio, and the scheme is the standard TVD form

`_face_correction` forms `r = 2 (grad_U . d_{U->f}) / (phi_D - phi_U)`. That is the
standard gradient-based reconstruction of the upstream-of-upstream node: on a
uniform mesh `phi_U - phi_UU = grad_U . (x_U - x_UU) = 2 grad_U . d_{U->f}`, so
`r` is exactly `(phi_U - phi_UU)/(phi_D - phi_U)`, the classical ratio. The
limiter `(r + |r|)/(1 + |r|)` is van Leer's, and the face value
`phi_U + 0.5 psi(r) (phi_D - phi_U)` is the standard TVD form. Both are as
Darwish & Moukalled's unified NVF/TVD framework prescribes for a virtual upwind
node.

Strict TVD is a one-dimensional, uniform-mesh property and this mesh is neither,
so the scheme is not formally TVD here — but it is bounded in the NVD sense
(the face value is confined between `phi_U` and `phi_D` by construction, since
`0 <= psi <= 2` and the correction is `0.5 psi (phi_D - phi_U)`), which is the
property that matters for keeping `k` and `omega` positive. The MMS measures it at
better than first order and below second, which is exactly what a limiter should
give. Nothing here changes a number, and saying so is the point: this was one of
the brief's questions and the answer is that it is right.

### S5 — Patankar under-relaxation is implemented correctly, including the order of operations

```python
        self.centre /= factor
        self.source += (1.0 - factor) * self.centre * field
```

The source uses the *already divided* diagonal, so the added term is
`(1-alpha) (a_P/alpha) phi_old = ((1-alpha)/alpha) a_P phi_old`, which is what
the docstring claims and what Patankar specifies. Doing it in the other order
would be wrong by a factor of `alpha`. The fixed point of the relaxed equation is
identical to the unrelaxed one. F5 is not a defect in this routine; it is a
defect in what `momentum()` hands to `face_fluxes` afterwards.

### S6 — The far-field pressure-correction coupling is sign-consistent between the matrix and the flux update

The brief asks for this to be verified independently rather than accepted as
fixed. Derived: for an outflow face holding `p' = 0`, the outward flux correction
is `F' = -rho D g (0 - p'_P) = +rho D g p'_P`, so the cell's corrected divergence
gains `+rho D g p'_P`; the matrix must therefore carry `+rho D g` on that
diagonal. `pressure_correction` adds exactly
`self._far_field_coupling(flux_j, diagonal)` to `coefficients.centre[:, -1]`, and
`apply_correction` adds exactly `self._far_field_coupling(...) * correction[:, -1]`
to `state.flux_j[:, -1]`, from the same function. The signs agree, the magnitudes
are the same expression, and the physical direction is right: a cell at higher
pressure than the boundary pushes more mass out. The Dirichlet value passed to the
correction gradient, `np.where(fixed, 0.0, correction[:, -1])`, is consistent with
`p' = 0` on fixed faces and zero-gradient elsewhere. Sound.

Also sound, and worth stating because it looks like an omission: the
pressure-correction matrix carries no non-orthogonal cross term. That is correct
and standard — `p' -> 0` at convergence, so omitting part of its Laplacian
changes the convergence rate and not the fixed point. The same omission in
`face_fluxes` is F4 precisely because the flux definition does *not* vanish at
convergence.

### S7 — Every SST constant is the published SST-2003 set

Checked line by line against the TMR page fetched 2026-09-07:

| code | value | SST-2003 |
|---|---|---|
| `BETA_STAR` | 0.09 | 0.09 |
| `A1` | 0.31 | 0.31 |
| `KAPPA` | 0.41 | 0.41 |
| `SIGMA_K1, SIGMA_W1, BETA_1` | 0.85, 0.5, 0.075 | 0.85, 0.5, 0.075 |
| `SIGMA_K2, SIGMA_W2, BETA_2` | 1.0, 0.856, 0.0828 | 1.0, 0.856, 0.0828 |
| `GAMMA_1, GAMMA_2` | 5/9, 0.44 | 5/9, 0.44 |
| `PRODUCTION_LIMIT` | 10.0 | 10 |

The docstring's note that `gamma` is derivable as
`beta/beta* - sigma_w kappa²/sqrt(beta*)` (giving 0.5532 and 0.4404) and that the
published rounded values are used instead is correct and is the right choice —
the TMR lists 5/9 and 0.44 as the *defining* constants of SST-2003. The plan's
note that Esch & Menter quote the derived pair while Menter, Kuntz & Langtry quote
the rounded one is also correct.

### S8 — The F1 blending function's cross-diffusion floor matches the published form exactly

**This contradicts the brief**, which suspected that the code "floors a quantity
that has already been divided by `omega`" where the paper floors something else.
The paper divides by `omega` too. Published:

    CD_kw = max( 2 rho sigma_w2 (1/omega) grad k . grad omega , 1e-10 )
    arg3  = 4 rho sigma_w2 k / (CD_kw d²)

Code (`sst._blending_f1`):

```python
        cross = np.maximum(
            2.0 * self.fluid.density * SIGMA_W2 * np.sum(grad_k * grad_omega, axis=-1)
            / omega,
            1e-10,
        )
        free_shear = 4.0 * self.fluid.density * SIGMA_W2 * k / (cross * distance**2)
```

Identical, including the `1e-10` floor, which is the SST-2003 value (Menter 1994
used `1e-20`). Dimensionally consistent, and the floored quantity is the same one
the paper floors. Sound.

### S9 — The F2 blending function and the eddy viscosity use the strain-rate invariant, so the variant claimed is the variant implemented

`_eddy_viscosity` uses `max(A1 * omega, strain * F2)` where
`base.strain_rate` returns `sqrt(2 S_ij S_ij)` — verified algebraically for two
dimensions as `2[(du/dx)² + (dv/dy)²] + (du/dy + dv/dx)²`. That is the 2003
revision; the 1994 model used the vorticity magnitude. `base.vorticity` exists and
is deliberately unused, with a docstring saying so. `_blending_f2` matches the
published `tanh(max(2 sqrt(k)/(beta* omega d), 500 nu/(d² omega))²)`. Sound —
the code implements SST-2003, not SST-V or SST-Vm.

### S10 — The `k` production form and limiter are Menter's

`P_k = min(mu_t S², 10 beta* rho k omega)`. Menter (2003) eq. (5) gives
`P_k = mu_t (du_i/dx_j)(du_i/dx_j + du_j/dx_i)`, which for incompressible flow is
`mu_t . 2 S_ij S_ij = mu_t S²` with `S` as defined above. The limiter constant is
10, matching SST-2003 (SST-1994 used 20). The Stage 0 argument for abandoning
Kato–Launder is correct: `Omega` is identically zero on a stagnation streamline by
symmetry, so `mu_t S Omega` puts a line of zero production along it, and the
limiter Menter specifies for the stagnation anomaly then never activates.

The lag in `state.eddy_viscosity` (the relaxed, previous-iteration value) is
benign at convergence — `_eddy_viscosity` is idempotent and
`relax_eddy_viscosity` is a plain relaxation, so at the fixed point the relaxed
and updated values coincide. It biases the transient, which is what a relaxation
is for.

### S11 — The `omega` logarithmic wall branch is Esch & Menter's

`omega_log = u_tau / (sqrt(beta*) kappa y1)`, coded as
`friction / (np.sqrt(_BETA_STAR) * _KAPPA * distance)` with
`sqrt(0.09) = 0.3`, matching the paper's `u_tau/(0.3 kappa y)`. The blend
`sqrt(omega_vis² + omega_log²)` is theirs, and the argument for why it needs no
switch (`1/y²` against `1/y`) is right. The `k` zero-flux condition is what Esch
& Menter specify for the automatic treatment. (Note for Stage 7: the TMR's
`SST-2003` verification cases use `k_wall = 0` on a wall-resolved mesh, so a TMR
comparison must be run down the low-Re path, not this one.)

### S12 — The fourth-power friction-velocity blend is accurate where it matters, and is not a systematic over-prediction

**This contradicts the brief**, which asserts that "the fourth-power blend always
exceeds the larger of its branches" implies "a systematic over-prediction of
`tau_w` even on a wall-resolved mesh" and asks for it to be quantified at
`y+ = 0.5, 1, 5`. The first clause is true and the second does not follow.

The code's fixed-point iteration was run against a true composite profile
(Spalding 1961, with the same `kappa = 0.41`, `C = 5.2` pairing the code uses),
recovering `u_tau` from `(U1, y1)` alone:

```
    y+     U+(true)    u_tau found     error     viscous br.  log br.
     0.1     0.1000     1.000000     -0.000%     1.00000    0.01923
     0.5     0.5000     1.000012     +0.001%     0.99999    0.09615
     1.0     0.9998     1.000266     +0.027%     0.99992    0.19225
     2.0     1.9974     1.001100     +0.110%     0.99934    0.28975
     5.0     4.8751     1.007774     +0.777%     0.98743    0.53313
    11.0     8.8316     1.011937     +1.194%     0.89603    0.79725
    30.0    12.7865     0.997291     -0.271%     0.65285    0.94792
   100.0    16.2588     0.996745     -0.326%     0.40322    0.98993
   300.0    19.0711     0.999067     -0.093%     0.25213    0.99800
```

The blend is not systematically high: it peaks at **+1.19% at y+ 11**, in the
middle of the buffer layer where no formulation is right, and it is *low* by 0.1
to 0.3% throughout the log layer. At `y+ = 1` the error is `+0.027%` and at
`y+ = 0.5` it is `+0.001%`. Repeating with the linear profile `U+ = y+`, which is
what a wall-resolved mesh actually carries below `y+ ~ 5`, gives `+0.0021%`,
`+0.034%` and `+2.13%` at `y+ = 0.5, 1, 5`.

Every one of those is far inside the accuracy the project claims (its own Stage 3
table records a 5.19% `Cd` spread across `y+ 0.5-5` from mesh effects alone). The
`_Y_PLUS_FLOOR = 1.0` freeze on the logarithmic branch is confirmed harmless: at
`y+ <= 1` the log branch is 0.19 of the viscous one and the fourth power reduces
its contribution to `0.19^4 = 0.0013`.

### S13 — The wall-row production override is the standard wall-function production, not an invention

The brief asks for this to be judged against Launder & Spalding (1974) and
Chieng & Launder (1980), and specifically whether the *pointwise* value the code
uses is a legitimate substitute for the *cell-averaged* one those papers use.

The code sets `dU/dy = min(u_tau²/nu, u_tau/(kappa y1))` and forms
`P_k = mu_t (dU/dy)²`. In the log layer, with the mixing-length eddy viscosity
`mu_t = rho u_tau kappa y1`,

    P_k = rho u_tau kappa y1 . (u_tau / (kappa y1))² = rho u_tau³ / (kappa y1)
        = tau_w . dU/dy ,

which is exactly the Launder–Spalding wall-cell production. It is standard
machinery, as the docstring claims.

Against Chieng & Launder's cell average: integrating `P = rho u_tau³/(kappa y)`
from the sublayer edge `y_v` to the cell face `y_n = 2 y1`,

    P_avg = (rho u_tau³ / (kappa y_n)) ln(y_n / y_v),
    P_point(y1) = 2 rho u_tau³ / (kappa y_n),
    P_avg / P_point = (1/2) ln(y_n / y_v).

With `y_v+ = 11`: at `y+ = 30` the ratio is `0.85`, at `y+ = 40` it is `1.00`, and
at `y+ = 100` it is `1.45`. So the pointwise form the code uses is within about
50% of the cell-averaged one across the whole range Stage 3 claims, crossing it
near `y+ 40`. That is a defensible approximation, and it is two orders smaller
than the factor of `(kappa U+)² ~ 30` error it replaced.

It also has a property Chieng & Launder's form does not: `min(u_tau²/nu, ...)`
reduces *exactly* to the resolved gradient `U1/y1` in the sublayer, so a
wall-resolved mesh is untouched. Measured by the project's own test at
`median(corrected/resolved) = 1.00` at y+ 1 and `< 0.4` at y+ 30. Verdict: a
legitimate equivalent, correctly reasoned, correctly tested.

### S14 — The effective wall viscosity multiplying the cross term does not distort the wall shear

The brief asks whether amplifying the non-orthogonal cross term by the same
`mu_wall = tau_w y1 / U1` that carries the orthogonal flux is intended and
correct. It is harmless, for a reason worth stating: `T = S - g d` satisfies
`T . S = 0`, so `T` lies *in* the face plane and `grad phi . T` picks the
**tangential** derivative of velocity, which near a wall is smaller than the
normal derivative by roughly the ratio of the wall spacing to the chord. Both
halves of the flux carry the same `mu_wall`, so the amplification is uniform and
the balance between them is unchanged whatever `mu_wall` is. Measured on the
NACA 2412 at y+ 1:

```
  mu_wall / mu:  min 1.000  median 1.001  max 1.005
  |cross term| / |orthogonal term| on the wall faces:
    mean 1.483e-03   median 2.558e-09   p99 5.286e-02   max 1.664e-01
```

One caveat rather than a finding: the cross term is 17% of the orthogonal one on
the worst wall face, so the shear the momentum equation receives there exceeds
`tau_w . A` by that much. The project's test
`test_the_wall_shear_is_what_the_momentum_equation_receives` checks only the
orthogonal half and would not see it. On a mesh whose wall faces were more skewed
this would deserve a finding; on these meshes (mean `1.5e-3`) it does not.

### S15 — The transpose viscous stress is algebraically correct

`div(mu grad(u)^T)_m = d/dx_n (mu du_n/dx_m) = (dmu/dx_n)(du_n/dx_m) + mu d(div u)/dx_m`,
and the second term vanishes for incompressible flow. The code returns

    ( mu_x u_x + mu_y v_x ,  mu_x u_y + mu_y v_y )

which is `sum_n (dmu/dx_n)(du_n/dx_m)` for `m = 0, 1` respectively. Verified index
by index. The argument for computing it this way rather than as a face sum (no
boundary treatment needed, no chance of transposing the `j` halves) is sound.

One inconsistency, noted rather than raised as a finding because it is
unmeasurable at y+ 1: the wall value passed to the `mu` gradient is the molecular
viscosity, while the wall *flux* is carried by `mu_wall`. On a y+ 30 mesh those
differ by two orders and the transpose term's wall gradient would be built from
the wrong one. Worth a line in the docstring if the wall model is ever the
default on coarse meshes.

### S16 — The residual normalisation is the standard finite-volume one

`|A phi - A phi_bar| + |b - A phi_bar|` summed over the field is exactly the
normalisation factor used throughout the finite-volume literature and in
OpenFOAM. The brief asks whether it is standard; it is, and it is not a local
invention. Its behaviour when the field mean is near zero and when the operator
has a near-null mode is fine (see F17 for both). Its behaviour on an
identity-substituted row is not, and that is F17.

`Residuals.worst` excluding the pressure residual is defensible and correctly
argued: that figure measures how well the inner linear solve reduced the `p'`
system, not whether the physics has settled, and the quantity that does measure
that — `continuity` — *is* included. Measured on the NACA 2412, the pressure
residual runs at `3.0e-03` while everything else is at `1e-05`, so including it
would put a floor under every run for no physical reason. Sound.

The continuity normalisation by `sum |flux_j[:, -1]|` is stable in practice: the
far-field split sits at 50.01% outflow on the converged cylinder and moves by
less than a percent through a run, so the reference is effectively the constant
`2 M_in`. Sound.

### S17 — The characteristic far-field split is correctly applied

Dirichlet velocity where `u . n < 0`, `p = 0` where `u . n > 0`, decided face by
face, with the diffusion terms masked to match (`far_field_active=inflow`) and —
importantly — *both* halves of the boundary diffusive flux masked, not just the
implicit one, so a zero-gradient face carries no non-orthogonal correction either.
That last detail is a real trap and the code has it right, with a comment saying
why. The choice is standard practice for incompressible external flow. The
*values* imposed are F2's problem, not the split's.

### S18 — The `p = 0` outflow condition is harmless outside the wake

The brief asks for the magnitude of the error from pinning `p = 0` on every
outflow face rather than at a single reference point. Measured on the converged
cylinder at `r/R = 80` (`r/D = 40`), against the potential-flow field a doublet
would produce there:

```
  potential-flow Cp there ranges -3.124e-04 .. +3.124e-04 (imposed as 0 on outflow)
  solver's Cp in the last cell row  -3.033e-04 .. +2.604e-02
```

The inviscid far-field pressure variation at 40 diameters is `+/- 3.1e-04 q`, so
imposing zero costs nothing there. The exception is the wake: the solver's own
last-row `Cp` reaches `+2.6e-02` where the wake crosses the boundary, and pinning
the face to zero suppresses that. That is a wake-truncation error, not a
uniform-pressure error, and it is one of the mechanisms behind F2's cylinder
result; the remedy is the same (move the boundary, or correct it analytically).
As a *reference-pressure* choice, `p = 0` on the outflow arc is sound.

### S19 — The hyperbolic marching formulation is the standard one

Orthogonality plus prescribed cell area is exactly the Steger–Chaussee pair for
two dimensions; the brief asks whether it is, and it is. The implementation
details that are easy to get wrong are all right here and each is argued in the
docstring: the Newton right-hand side collapses to `(f1, V + f2) + B r_old` with a
`+f2` (taking it as `-f2` marches inward); the cell width is evaluated at the
*mid* layer, not the outer one, which otherwise overstates the area by
`(1 + step/radius)`; and — the important one — the dissipation smooths the
marching *increment* rather than the position, because the discrete Laplacian of a
curved grid line is `O(R d_xi²)` and smoothing positions injects a source
proportional to the body's curvature. On a circle the marcher reproduces the exact
concentric answer, measured here at mean and peak non-orthogonality of `0.0000`
degrees.

`_blend_to_circle`'s claim that a polar construction cannot fold is correct: the
radii are monotone by construction and `_monotone_angles` enforces a monotone
angular sweep, so grid lines cannot cross. The `sweep = 2 pi` choice (rather than
estimating it from the last marched step) is right for the reason given.

### S20 — The laminar path is genuinely untouched by the turbulent constructs

`Case._wall_model` returns `None` for `model_name == "laminar"`, which propagates
to `PressureVelocityCoupling(wall_model=False)` (so `wall_viscosity` is never
called), to `post.compute_forces`, `post.surface_data` and
`post.separation_points` (so `wall_shear_stress` takes the `mu U1 / y1` branch),
and `Laminar.update` zeroes the eddy viscosity every iteration. The first-layer
sizing takes the Blasius route rather than the `y+` route. Traced through every
call site; the isolation is complete. The regression the handover records — "a
wall treatment leaking into laminar runs where it had no business being" — is
fixed, and the reason it mattered (the fourth-power blend returning slightly more
than molecular viscosity even where the viscous branch is exact) is quantified in
S12 at `+0.03%` at `y+ 1`.

### S21 — The finite-volume metrics are correct

Shoelace area and true area centroid computed from the same sum; face area
vectors formed by rotating the node-to-node edge, `+i` and `+j` consistently;
`wall_face_area` negating `face_j[:, 0]` so that pressure on the body pushes along
`-face_j`. The force integral's use of `faces.wall.area` (which already carries
that negation, via `build_faces`) is therefore correct: `sum(p * area)` is the
force the fluid exerts on the body, and a constant pressure offset integrates to
zero round a closed contour, which is why `p = 0` at the far field costs nothing
in `Cd`. The project's own test `test_uniform_pressure_exerts_no_net_force`
checks that. The sign convention for lift and drag is right; only the moment is
not (F10).

### S22 — The mesh quality metrics measure what they claim

Non-orthogonality as the angle between the face normal and the centroid-to-centroid
line, skewness as the offset between the face midpoint and where that line crosses
the face — both are the standard definitions and both were reproduced
independently here to the digit (`mean 3.932, peak 60.069` computed both ways).
The docstrings' explanation of why each matters is accurate, including that the
non-orthogonal correction grows as `tan` and overtakes the implicit part near 70
degrees.

The one reservation is a reporting one, recorded under F4: the mean over all faces
averages a near-perfect marched region with a much worse blended one and describes
neither, and the `>60 degrees` warning fraction reads 0.00% on a mesh where 2.92%
of faces are past 30 degrees.

---

## Known items re-examined

`docs/hardening-plan.md` lists four open items. Each was verified independently
rather than accepted. One of them does not reproduce.

### K1 — "The divergence monitor's false positive" — **does not reproduce on `main`**

**What the plan says.** A NACA 0012 at Re 2e6 with SST on factory defaults raises
`SolverDiverged` at iteration 367. It is not diverging: with the monitor disarmed
the residual peaks at `1.8e-01` near iteration 400 as the eddy-viscosity ratio
passes 100, then recovers monotonically to `2.8e-05` by iteration 1100 with
`Cd = 0.009487 +/- 2e-6` and `Cl = -8e-6`. The clause that fires is
`_MONITOR_LOST`, and the flaw is in what "best" means. This is described in
`README.md`, `docs/handover.md` and `docs/hardening-plan.md` as the first thing
to fix and as blocking the primary use case.

**What I found.** It does not happen. Run on `main` at `37fabd2`: NACA 0012,
`AIR_15C`, `U = 30 m/s` (Re 2.030e6), zero incidence, 240 surface points, target
`y+ = 1`, far field 40 chords, factory `Numerics()`, **divergence monitor armed**:

```
Run 1: factory defaults, monitor armed
  Re 2.030e+06   mesh 240x89   marched 50
    100 worst 7.451e-03  mut/mu 9.3    Cl +0.000788  Cd +0.0066491
    200 worst 2.696e-04  mut/mu 90.2   Cl -0.000016  Cd +0.0094467
    300 worst 7.026e-05  mut/mu 116.7  Cl -0.000119  Cd +0.0094931
    400 worst 4.275e-05  mut/mu 117.6  Cl -0.000116  Cd +0.0094904
    500 worst 4.080e-05  mut/mu 117.2  Cl -0.000137  Cd +0.0094891
    600 worst 4.754e-05  mut/mu 117.0  Cl -0.000081  Cd +0.0094860
    700 worst 3.548e-05  mut/mu 117.0  Cl -0.000128  Cd +0.0094902
    800 worst 3.251e-05  mut/mu 117.0  Cl -0.000069  Cd +0.0094848
    900 worst 3.071e-05  mut/mu 117.0  Cl -0.000126  Cd +0.0094888
   1000 worst 3.111e-05  mut/mu 117.0  Cl -0.000076  Cd +0.0094856
   1100 worst 3.451e-05  mut/mu 117.0  Cl -0.000130  Cd +0.0094903
   1200 worst 4.077e-05  mut/mu 117.0  Cl -0.000057  Cd +0.0094840
   1300 worst 3.586e-05  mut/mu 117.0  Cl -0.000158  Cd +0.0094929
   1400 worst 3.968e-05  mut/mu 117.0  Cl -0.000083  Cd +0.0094858
   1500 worst 3.510e-05  mut/mu 117.0  Cl -0.000111  Cd +0.0094875
   1600 worst 2.818e-05  mut/mu 117.0  Cl -0.000091  Cd +0.0094859
```

The run went the full 1600 iterations without raising anything. No exception. No
excursion. The residual falls monotonically through the region
where the plan reports a peak of `1.8e-01`; the largest value after iteration 100
is `7.45e-03`. The eddy-viscosity ratio crosses 100 between iterations 200 and
300 and settles at `117.0` — which is exactly the figure the plan reports for the
converged state — and nothing happens when it does.

The *converged state* the plan describes is reproduced almost exactly:
`Cd = 0.009489` with a spread of about `3e-6` over iterations 300 to 1100,
against the plan's `0.009487 +/- 2e-6`; `mu_t/mu = 117.0` against 117. The
*transient* is not reproduced at all.

**What I add, and what it means.**

1. **The blocker as documented is not a blocker today.** Either it was removed by
   something merged after the plan was written — the Stage 3 wall treatment
   (PR #6) is the obvious candidate, and `README.md` already credits "the
   `mu_t S^2` production correction and the wall treatment" with turning an
   earlier limit cycle into a recovery — or the case differs in a parameter none
   of the three documents records. Neither the angle of attack, the surface point
   count, the `y+` target nor the far-field ratio is stated anywhere for this
   case, and all four change the answer. **Before any work is done on the
   monitor, the case that motivates it should be re-measured and its full
   specification written down.** Three documents currently direct the next
   engineer at a defect that may not exist.

2. **The plan's diagnosis of the mechanism is sound as far as it goes**, and I
   would add one component it does not name. `DivergenceMonitor._history` is
   cleared only by `CflRamp` backing off, and `CflRamp` is inactive when
   `pseudo_transient` is `False`, which is the default. So on factory defaults
   the two comparison windows are never reset. At iteration 367 both windows
   would sit inside the excursion, which makes the `1.5x` median-rise test
   *harder* to trip, not easier — so if the plan's trace is accurate the climb
   between iterations 267–317 and 317–367 was genuinely steeper than 1.5x in the
   median. That is a real climb, and a monitor tuned to ignore it would also fail
   to catch the laminar cylinder at Re 2e6, which is the case Stage 2 was
   rebuilt around. The tension the plan identifies is real; the case it uses to
   demonstrate it is not currently demonstrating it.

3. **The physics question — is the excursion itself worth fixing? — has no
   support in anything I can reproduce.** The plan attributes it to the
   eddy-viscosity ratio passing 100. Two independent cases cross that threshold
   here and neither excurses: the NACA 2412 at 5 degrees passes 100 between
   iterations 100 and 200 and peaks at `mu_t/mu = 214` with a monotone residual
   falling to `9.9e-07`; the NACA 0012 above passes 100 between 200 and 300 and
   settles at 117, also monotone. Crossing `mu_t/mu = 100` is not sufficient to
   produce a residual excursion in this solver. If the excursion is real on some
   configuration, its cause is something else.

4. **The monitor is not close to firing, not merely short of it.** A second run
   with the monitor disarmed carried a shadow `DivergenceMonitor` alongside and
   logged the ratio to the run's own best at every hundredth iteration:

   ```
       100 worst 7.451e-03  best-so-far 6.170e-03  ratio 1.2  mut/mu 9.3
       300 worst 7.026e-05  best-so-far 6.702e-05  ratio 1.0  mut/mu 116.7
       500 worst 4.080e-05  best-so-far 2.784e-05  ratio 1.5  mut/mu 117.2
       600 worst 4.754e-05  best-so-far 2.708e-05  ratio 1.8  mut/mu 117.0
      1000 worst 3.111e-05  best-so-far 2.421e-05  ratio 1.3  mut/mu 117.0
      1200 worst 4.077e-05  best-so-far 2.421e-05  ratio 1.7  mut/mu 117.0
   ```

   `_MONITOR_LOST` requires a ratio of **100**; the run peaks at **1.8**.
   `_MONITOR_FLOOR` requires the residual to exceed `1e-3`; it is below that from
   about iteration 150 onward. The shadow monitor's own verdict over 1600
   iterations:

   ```
     a shadow monitor would first have tripped at iteration None
     peak residual 1.000e+00 at iteration 1
     mu_t/mu first exceeds 100 at iteration 217
     mu_t/mu peak 117.7 at iteration 351
     final Cl -0.0000913  Cd 0.00948589  Cdp 0.00190986  Cdf 0.00757603
   ```

   The eddy-viscosity ratio crosses 100 at iteration 217 and peaks at 351 —
   squarely inside the window the plan describes — and the residual there is
   `4e-05`. Two of the monitor's three gates are never approached, let alone
   crossed.

   Set the plan's residual table against the same windows from this run:

   | iterations | plan's median | measured median | measured peak |
   |---|---|---|---|
   | 300 – 500 | `2.68e-02` (peak `1.78e-01`) | `3.99e-05` | `1.01e-04` |
   | 500 – 800 | `2.53e-03` | `3.70e-05` | `5.14e-05` |
   | 800 – 1100 | `4.73e-05` | `3.44e-05` | `5.15e-05` |
   | 1100 – 1500 | `2.81e-05` | `3.60e-05` | `5.76e-05` |

   The last two rows agree. The first two are three and two orders apart. The run
   arrives at the plan's *late* behaviour almost immediately and never passes
   through the plan's *early* behaviour at all — which is exactly the signature
   of a transient that something has since removed, rather than of a different
   case.

5. **There is a real open item on this case, and it is not the monitor.** The
   residual stops falling at about `4e-05` around iteration 400 and then
   oscillates between `2.8e-05` and `4.8e-05` for the next 1200 iterations,
   against a default `tolerance = 1e-6`. The run would exhaust
   `max_iterations = 3000` rather than converge.
   `History.forces_are_steady` would call it settled — `Cd` moves only in the
   sixth decimal, `0.0094840` to `0.0094929` over 1200 iterations — but
   `Residuals.has_converged` never will. That plateau is the genuine unresolved
   behaviour on this case, it is a different problem from the one the plan
   describes, and it is worth diagnosing. From F17, the equation binding `worst`
   is almost always `Uy` or `k`, so that is where to look; the deferred
   correction's lagged source is the usual cause of a plateau at this level.

### K2 — The marched-to-analytic seam: confirmed, and the proposed remedy will not work as stated

**What the plan says.** "Still the worst region on an aerofoil mesh: the single
face at 60.07 degrees on the y+ 1 mesh is it. The handover happens in one layer,
so non-orthogonality jumps from about 3 degrees to the peak and back. Blend the
two constructions over several layers instead."

**What I found.** The jump is confirmed and is sharper than "about 3 degrees to
the peak":

```
    layer  51     2.669 deg
    layer  52    35.378 deg   <-- last marched face
    layer  53    62.149 deg   <-- first blended face
    layer  54    60.329 deg
    layer  55    57.918 deg
    ...
    layer  88     1.021 deg
```

(This sequence was measured on the *unrotated* NACA 2412, which marches 52
layers; the face-mean block below is the case `build_case` actually produces at
5 degrees incidence, which marches 54. The two-layer difference is the body
rotation changing the resampled surface, and it does not affect either
conclusion.)

But it is **not a single face**, and that is the part of the diagnosis I would
change. Taking face means rather than per-layer peaks:

```
  marched layers 54 of 89
  j-faces, marched region : mean 0.248  p99 5.159  peak 20.558 deg
  j-faces, blended region : mean 9.502  p99 49.082  peak 57.387 deg
  i-faces, blended region : mean 9.623  peak 60.069 deg
  all faces               : mean 3.932  peak 60.069  fraction > 30 deg 2.92%
```

The blended region is 35 of 89 layers — 39% of the cells — and it averages 9.5
degrees with a 99th percentile near 50. The "single face at 60.07 degrees" is the
worst point of a large bad region, not an isolated defect, and it is the region
F4's spurious flux lives in. The quality report's own warning fires only above 60
degrees, so it reports `0.00%` of faces on a mesh where 2.92% are past 30.

**What I add: the mechanism, measured, and why "blend over several layers" as
stated will not remove it.**

```
  angle(marched step, polar ray) at the handover layer:
    mean 36.232 deg   median 38.413 deg   p99 63.185 deg   peak 63.362 deg
  turn in grid-line direction across the seam: mean 36.159  peak 63.391 deg
```

The mismatch is a **direction** mismatch of 36 degrees on average. Now look at
what `_blend_to_circle` already blends. It relaxes the *angular distribution*
towards uniform with a smoothstep in log radius, over every blended layer — and
the docstring records that doing this correctly took the mean from 15 degrees to
4.5. What it does **not** blend is the radial placement:

```python
    fraction = np.concatenate(([0.0], np.cumsum(thicknesses) / thicknesses.sum()))
    r = r_inner[:, None] + (radius - r_inner[:, None]) * fraction[None, :]
```

Every blended layer steps outward along the ray from the body centroid. The first
one does so immediately, at 36 degrees to the direction the march arrived on. No
number of subsequent layers changes the first step, so spreading the *existing*
construction over more layers cannot remove the seam — the discontinuity is in
the marching direction, and the construction has no marching direction to blend.

**The correct construction**, which is the "blend the normals rather than only
the positions" option:

    n_march = normalise( x_m - x_{m-1} )        the direction the march arrived on
    e_r     = normalise( x_m - centre )         the polar ray
    s_j     = smoothstep over the first N blended layers
    w_j     = normalise( (1 - s_j) n_march + s_j e_r )
    x_{j+1} = x_j + t_j w_j

for `j < N`, reverting to the current polar placement thereafter, with `N` chosen
so that `36 deg / N` is comfortably below the marched region's own 5-degree 99th
percentile — `N ~ 8` to 10. The radii still increase monotonically and the
angular sweep is still monotone, so the construction's one great virtue — it
cannot fold — is preserved.

Elliptic smoothing with Steger–Sorenson control functions (Thompson, Thames &
Mastin) is the textbook alternative and would also work. I would not recommend it
here: it replaces a direct, unconditionally safe construction with an iterative
one carrying control parameters that have to be tuned per geometry, which is
precisely the trade the project has already declined once with adaptive
dissipation.

**Do F8 first.** Raising the Newton pass count carries the march ten layers
further out, where the marched layer is closer to circular and the direction
mismatch to be blended is smaller. It is a one-line change and it reduces the
problem this item has to solve.

### K3 — Wake refinement: confirmed, with the number the plan quotes

**What the plan says.** "An O-grid wraps the wake and falls below four cells per
diameter about two diameters downstream, so a shed vortex is smeared within a
couple of its own spacings. Stage 4 cannot work on that."

**What I found.** Confirmed, essentially exactly. Cells per reference length
along the downstream centreline, measured on the meshes `build_case` produces:

```
cylinder D=1: 180x53, marched 27
     x =   0.6   mean cell 0.0232   ->  43.1 cells per length
     x =   1.0   mean cell 0.0642   ->  15.6 cells per length
     x =   2.0   mean cell 0.2109   ->   4.7 cells per length
     x =   3.0   mean cell 0.3417   ->   2.9 cells per length
     x =   5.0   mean cell 0.6373   ->   1.6 cells per length

NACA 0012 c=1: 240x89, marched 50
     x =   0.6   mean cell 0.0111   ->  90.5 cells per length
     x =   1.0   mean cell 0.0077   -> 129.6 cells per length
     x =   2.0   mean cell 0.1362   ->   7.3 cells per length
     x =   3.0   mean cell 0.2752   ->   3.6 cells per length
     x =   5.0   mean cell 0.5521   ->   1.8 cells per length
```

The cylinder crosses four cells per diameter between `x/D = 2` and `3`, which is
what the plan says. (The NACA row at `x/c = 1.0` reading 129.6 is the
trailing-edge cluster, not wake resolution.)

**What I add: the target to design against.** "Four cells per diameter" is the
symptom; the design quantity is cells per vortex core. At the Stage 4 acceptance
condition of Re 100, `St = 0.164`, the streamwise vortex spacing is
`U/f = D/St = 6.1 D` and a Kármán core is roughly `0.5` to `1 D` across. Four
cells per diameter is therefore two to four cells across a core, which will be
diffused away within a couple of spacings by any scheme — the plan's conclusion
is right and the mechanism is numerical diffusion of the core, not of the street.
A defensible target is 20 cells per diameter maintained to `x/D = 10`, which is a
factor of 4 at `x/D = 2` rising to 12 at `x/D = 10`. That is what the refinement
has to deliver, and it is worth writing into Stage 6 as an acceptance criterion
so the work has a target rather than a direction.

### K4 — C-grid topology: not re-costed, but it also fixes something the plan does not claim for it

**What the plan says.** The structural fix for the wake; it removes the trailing
edge from the interior of the marched line; honest cost measured at 63 uses of
periodic `np.roll` across ten modules, and `StructuredMatrix`'s five-band
assumption breaks because a wake cut makes cell `(i, 0)` a neighbour of
`(Ni-1-i, 0)`, far apart in the `k = i*Nj + j` ordering. Keep the O-grid for bluff
bodies, where a circle meshes at 0.0 degrees non-orthogonality.

**What I found.** I did not re-cost the software change; that is outside this
audit's remit and the plan's figure is the kind that is measured rather than
estimated. Two things I can add from the physics side.

The claim that a circle meshes at exactly zero non-orthogonality is **confirmed**:
`mean 0.0000 peak 0.0000` degrees, measured. That is a genuine property and the
plan is right to keep the O-grid for bluff bodies.

But it is also the reason the validation gate is blind to F4, F13 and F14, all of
which are identically zero on an orthogonal mesh, and to F2's vortex term, which
is identically zero on a non-lifting body. **The gate's greatest strength — that
the cylinder mesh is perfect — is also the reason it cannot see four of the
findings in this report.** Whatever is decided about the C-grid, a second gate on
a non-orthogonal lifting case is needed independently, and it is cheap: the
NACA 2412 at 5 degrees used throughout this audit converges to `1e-6` in 1007
iterations and about a quarter of an hour on this machine (0.84 s per
iteration solo, 240x89).

The C-grid does one further thing the plan does not claim for it: by letting the
march run downstream instead of wrapping, it shrinks the polar-blended fraction of
the mesh, which is where 39% of the cells and essentially all of the
non-orthogonality currently live. That makes it a fix for F4's *severity* as well
as for the wake.

### K5 — "Systematically refined mesh families" (listed under Stage 6, needed by Stage 7)

Not one of the four, but on the same list, and this audit has made a start on it:
the three-mesh cylinder family under F11 is the first grid-convergence sequence
the project has run, and it produced the observed order of `Cd` (1.261) and the
first evidence that the wake length is not grid-converged. The families need to
come out of one specification — `MeshSettings` with an explicit `first_layer` and
a scaled `surface_points` is enough for the cylinder, as used here, but an
aerofoil family also needs the surface resampler's `min_spacing` to scale with it
or the refinement is not uniform.

---

## Open questions for the owner

These are decisions, not tasks. Each is a fork I could not settle from the code
and the literature alone, with what I would choose and why.

**1. The `omega` wall value: authority, or measurement?** F1 shows a 9% swing in
`Cd` from a constant that both Menter and Wilcox describe as one the answer
should be insensitive to. Taking the TMR's 60 on authority is defensible and is
one character. Establishing which value reproduces the NASA TMR flat-plate
(`2DZP`) skin-friction distribution is Stage 7 work brought forward by several
months. *I would bring it forward.* It is the single largest effect in this
report, it is in the primary use case, and a 9% `Cd` change adopted without a
verification case is exactly the kind of decision the handover's standing
instructions exist to prevent.

**2. The far field: correct it, or move it?** F2's error is `O(1/R)` and the
point-vortex correction reduces it to `O(1/R²)`. Moving the boundary instead is
free of modelling assumptions but expensive in cells and, at the measured
scaling, would need several hundred chords to reach 0.1% in `Cl`. *I would
implement the correction*, because Stage 7 needs the domain error to be small
compared with the discretisation error and at 40 chords it is not: on the
cylinder, where both were measured on the same case, the domain truncation is
`+1.32%` in `Cd` against a discretisation error of `−0.076%` — seventeen times
larger. A GCI computed on a mesh family at a fixed 40 chords would report the
smaller of the two and be silent about the larger.

**3. Does the validation gate pin digits, or pin bands with an uncertainty?**
F3 moves the gate's `Cd` from 1.5142 to 1.5164; F2's domain limit is 1.4945; the
Richardson limit at fixed domain is 1.5154; the wake length is not grid-converged
and extrapolates to 2.200 against a reported 2.1219. Every one of those is a
change to a number the handover calls "unchanged through every stage so far". The
project has committed to ASME V&V 20, which requires a result to be validated *at
a stated uncertainty*, not against a fixed percentage. *I would convert the gate
to report `Cd`, wake and separation each with a discretisation uncertainty from
a two- or three-mesh family*, and accept that the headline digits move once. The
alternative — keeping the digits — means the gate is pinning a number that three
separate first-order errors happen to produce.

**4. How far to take F4's consistency fix.** The flux definition must change, and
`apply_correction`'s compact operator must change with it or the two disagree
again — that is the failure `_far_field_coupling`'s docstring already records.
The open question is whether the *pressure-correction matrix* should also gain
the non-orthogonal term. It does not need it for correctness (`p' -> 0`), but on
a mesh with 39% of cells in the blended region it would improve the convergence
rate, at the cost of an extra lagged term and a possible loss of diagonal
dominance. *I would leave the matrix orthogonal-only* and revisit it only if the
outer iteration slows.

**5. Which relaxation-independence remedy (F5).** Building the Rhie-Chow mobility
from the unrelaxed diagonal is one line and removes the leading dependence. The
Choi/Pascau form removes it by construction and is the same machinery Stage 4
needs to make Rhie-Chow time-step-independent. *I would do the second*, because
otherwise it gets done twice.

**6. Freestream turbulence: TMR ambient values, or the sustaining terms (F7)?**
Changing the defaults is two numbers. The `SST-sust` sustaining terms are two
extra source terms and they remove the far-field-radius dependence of the ambient
level entirely, which matters for F2 and matters much more for the gamma
transition model Stage 5 calls "required, not optional". *I would implement
`SST-sust`*, and note that the TMR specifies it as a named variant so it is
verifiable rather than invented.

**7. Does the project want a second regression gate?** The cylinder is
orthogonal, non-lifting, laminar and run at one mesh density. It is structurally
incapable of detecting F1, F2's vortex term, F4, F6, F7, F13 or F14 — seven of
the seventeen findings. A NACA 2412 at 5 degrees with SST converges to `1e-6` in
1007 iterations, about a quarter of an hour here, and would see all of them.
*I would add it*, with a wider tolerance band
than the cylinder's, precisely because it is not a validated case — its job is to
detect change, not to certify accuracy.

**8. The moment convention (F10).** Flipping the sign is right if the intent is
comparability with published `Cm`. If the intent is "counter-clockwise positive
in the mesh frame", the quantity should be documented as such and not called
`Cm`. This is a decision about what the number is *for*, and it is the owner's.
The same applies to the default moment reference: the contour centroid is a
defensible choice, but published aerofoil moments are quoted about the quarter
chord.

**9. Should `enforce_global_mass_balance` exist (F9)?** The stated justification —
Poisson compatibility — does not apply, because the fixed-pressure outflow faces
make the matrix non-singular. The routine is doing something real but different:
it is imposing global conservation on an extrapolated outflow, and it is doing it
at convergence. *I would remove it* and enforce compatibility on the source only
in the case where no outflow face exists, which is the only case that needs it —
but this is a judgement about what the far field is *for* and the owner may want
the insurance. If it stays, it needs the diagnostic F9 describes, because at
present the one situation it silently declines to handle is the one it was
written for.

**10. What the divergence monitor should now be judged against (K1).** The case
that motivates the work does not reproduce. Before the monitor is touched, either
that case needs to be re-established with a full specification recorded, or the
item needs to be closed and the monitor left alone until a real false positive
appears. *I would re-measure first.* The plan's own standing instruction — "do
not claim a result before the run returns" — cuts both ways: the documented
failure is itself a claim that has now failed to reproduce twice in this audit's
two turbulent cases, and it is directing the next engineer's time.

**11. A documentation question, not a physics one.** `docs/handover.md` gives the
interpreter as `C:/AI_CFD_Analysis/.venv/Scripts/python.exe`, which does not
exist in this checkout (`.venv` at the repository root does); it says "252 tests,
about 100 seconds" where the run takes 260; `README.md` says 188 tests where
there are 252. None of that affects an answer, but the handover is the document
that tells the next person how to establish a baseline, and two of its three
commands are wrong.

---

## Bibliography

**Primary model definitions**

- NASA Turbulence Modeling Resource, "Menter Shear Stress Transport Model",
  `https://tmbwg.github.io/turbmodels/sst.html`, accessed 2026-09-07. (The old
  address `turbmodels.larc.nasa.gov` now redirects to a NASA landing page; this
  is the live location.) Used for: the SST-2003 definition, the omega-equation
  production erratum, the production limiter's scope, the wall boundary
  conditions, the farfield value bands, and the `SST-sust` variant.
- Menter, F. R. (1994), "Two-Equation Eddy-Viscosity Turbulence Models for
  Engineering Applications", *AIAA Journal* 32(8), 1598–1605. Blending functions,
  constants, and the wall `omega` condition in the appendix.
- Menter, F. R., Kuntz, M. & Langtry, R. (2003), "Ten Years of Industrial
  Experience with the SST Turbulence Model", in K. Hanjalić, Y. Nagano &
  M. Tummers (eds), *Turbulence, Heat and Mass Transfer 4*, Begell House,
  625–632. Equations (1) and (5); the source of the production erratum.
- Esch, T. & Menter, F. R. (2003), "Heat Transfer Predictions Based on
  Two-Equation Turbulence Models with Advanced Wall Treatment", *Turbulence, Heat
  and Mass Transfer 4* / IGTC 2003. Equations (15)–(18): the automatic wall
  treatment, the blended `omega`, the fourth-power friction velocity.
- Wilcox, D. C. (2006), *Turbulence Modeling for CFD*, 3rd ed., DCW Industries.
  §4.6.2 for the near-wall `omega` asymptote `6 nu/(beta_1 y²)`; §7.3.1 for the
  surface boundary condition and the insensitivity argument.
- Spalart, P. R. & Rumsey, C. L. (2007), "Effective Inflow Conditions for
  Turbulence Models in Aerodynamic Calculations", *AIAA Journal* 45(10),
  2544–2553. Ambient-value recommendations and the freestream decay analysis.
- Launder, B. E. & Spalding, D. B. (1974), "The Numerical Computation of
  Turbulent Flows", *Computer Methods in Applied Mechanics and Engineering* 3(2),
  269–289. Wall-function production `P_k = tau_w dU/dy`.
- Chieng, C. C. & Launder, B. E. (1980), "On the Calculation of Turbulent Heat
  Transport Downstream from an Abrupt Pipe Expansion", *Numerical Heat Transfer*
  3(2), 189–207. Cell-averaged wall-cell production.
- Spalding, D. B. (1961), "A Single Formula for the Law of the Wall", *Journal of
  Applied Mechanics* 28(3), 455–458. The composite profile used to test the
  friction-velocity blend.
- Bradshaw, P., Ferriss, D. H. & Atwell, N. P. (1967), "Calculation of Boundary
  Layer Development Using the Turbulent Energy Equation", *Journal of Fluid
  Mechanics* 28(3), 593–616. The shear-stress relation the `a1` limiter enforces.

**Discretisation and pressure–velocity coupling**

- Rhie, C. M. & Chow, W. L. (1983), "Numerical Study of the Turbulent Flow Past
  an Airfoil with Trailing Edge Separation", *AIAA Journal* 21(11), 1525–1532.
- Patankar, S. V. (1980), *Numerical Heat Transfer and Fluid Flow*, Hemisphere.
  §6.7 (outlet mass-flux correction), and the implicit under-relaxation form.
- Van Doormaal, J. P. & Raithby, G. D. (1984), "Enhancements of the SIMPLE Method
  for Predicting Incompressible Fluid Flows", *Numerical Heat Transfer* 7(2),
  147–163. SIMPLEC and the `alpha_p ~ 1 - alpha_u` pairing.
- Majumdar, S. (1988), "Role of Underrelaxation in Momentum Interpolation for
  Calculation of Flow with Nonstaggered Grids", *Numerical Heat Transfer* 13(1),
  125–132. The original statement of the relaxation dependence.
- Choi, S. K. (1999), "Note on the Use of Momentum Interpolation Method for
  Unsteady Flows", *Numerical Heat Transfer Part A* 36(5), 545–550.
- Yu, B., Tao, W.-Q., Wei, J.-J., Kawaguchi, Y., Tagawa, T. & Ozoe, H. (2002),
  "Discussion on Momentum Interpolation Method for Collocated Grids of
  Incompressible Flow", *Numerical Heat Transfer Part B* 42(2), 141–166.
- Cubero, A. & Fueyo, N. (2007), "A compact momentum interpolation procedure for
  unsteady flows and relaxation", *Numerical Heat Transfer Part B* 52(6),
  507–529.
- Pascau, A. (2011), "Cell face velocity alternatives in a structured colocated
  grid for the unsteady Navier–Stokes equations", *International Journal for
  Numerical Methods in Fluids* 65(7), 812–833.
- Jasak, H. (1996), *Error Analysis and Estimation for the Finite Volume Method
  with Applications to Fluid Flows*, PhD thesis, Imperial College London. Ch. 3
  for the over-relaxed decomposition, the deferred correction and the residual
  normalisation.
- Demirdžić, I. & Muzaferija, S. (1995), "Numerical method for coupled fluid
  flow, heat transfer and stress analysis using unstructured moving meshes with
  cells of arbitrary topology", *Computer Methods in Applied Mechanics and
  Engineering* 125(1–4), 235–255.
- Ferziger, J. H. & Perić, M. (2002), *Computational Methods for Fluid Dynamics*,
  3rd ed., Springer. §5.3, §7.1–7.2, §8.4, §8.6.
- Darwish, M. S. & Moukalled, F. (2003), "TVD schemes for unstructured grids",
  *International Journal of Heat and Mass Transfer* 46(4), 599–611. The unified
  NVF/TVD framework and the virtual upwind node.
- Sweby, P. K. (1984), "High Resolution Schemes Using Flux Limiters for
  Hyperbolic Conservation Laws", *SIAM Journal on Numerical Analysis* 21(5),
  995–1011.
- van Leer, B. (1974), "Towards the Ultimate Conservative Difference Scheme II",
  *Journal of Computational Physics* 14(4), 361–370. The limiter itself.

**Boundary conditions and the far field**

- Thomas, J. L. & Salas, M. D. (1986), "Far-Field Boundary Conditions for
  Transonic Lifting Solutions to the Euler Equations", *AIAA Journal* 24(7),
  1074–1080.
- Usab, W. J. & Murman, E. M. (1983), "Embedded Mesh Solutions of the Euler
  Equation Using a Multiple-Grid Method", AIAA Paper 83-1946.
- Vassberg, J. C. & Jameson, A. (2010), "In Pursuit of Grid Convergence for
  Two-Dimensional Euler Solutions", *Journal of Aircraft* 47(4), 1152–1166.
  NACA 0012 grid families to 150 chords, with and without the point-vortex
  far-field influence.

**Mesh generation**

- Steger, J. L. & Chaussee, D. S. (1980), "Generation of Body-Fitted Coordinates
  Using Hyperbolic Partial Differential Equations", *SIAM Journal on Scientific
  and Statistical Computing* 1(4), 431–437.
- Chan, W. M. & Steger, J. L. (1992), "Enhancements of a Three-Dimensional
  Hyperbolic Grid Generation Scheme", *Applied Mathematics and Computation* 51(2–3),
  181–205.
- Chan, W. M., Rogers, S. E., Nash, S. M., Buning, P. G. & Meakin, R.,
  *HYPGEN User's Manual*, NASA TM 108791.
- Thompson, J. F., Thames, F. C. & Mastin, C. W. (1974), "Automatic Numerical
  Generation of Body-Fitted Curvilinear Coordinate System for Field Containing
  Any Number of Arbitrary Two-Dimensional Bodies", *Journal of Computational
  Physics* 15(3), 299–319; and Thompson, Warsi & Mastin (1985), *Numerical Grid
  Generation: Foundations and Applications*, North-Holland, for the
  Steger–Sorenson control functions.

**Verification and validation**

- Roache, P. J. (1998), *Verification and Validation in Computational Science and
  Engineering*, Hermosa Publishers.
- Salari, K. & Knupp, P. (2000), *Code Verification by the Method of Manufactured
  Solutions*, Sandia National Laboratories report SAND2000-1444.
- Oberkampf, W. L. & Roy, C. J. (2010), *Verification and Validation in
  Scientific Computing*, Cambridge University Press.
- Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H. &
  Raad, P. E. (2008), "Procedure for Estimation and Reporting of Uncertainty Due
  to Discretization in CFD Applications", *Journal of Fluids Engineering* 130(7),
  078001.
- ASME (2009), *Standard for Verification and Validation in Computational Fluid
  Dynamics and Heat Transfer*, ASME V&V 20-2009.

**Benchmark data used for orientation**

- Tritton, D. J. (1959), "Experiments on the flow past a circular cylinder at low
  Reynolds numbers", *Journal of Fluid Mechanics* 6(4), 547–567.
- Dennis, S. C. R. & Chang, G.-Z. (1970), "Numerical solutions for steady flow
  past a circular cylinder at Reynolds numbers up to 100", *Journal of Fluid
  Mechanics* 42(3), 471–489.
- Coutanceau, M. & Bouard, R. (1977), "Experimental determination of the main
  features of the viscous flow in the wake of a circular cylinder in uniform
  translation. Part 1", *Journal of Fluid Mechanics* 79(2), 231–256.
- Fornberg, B. (1980), "A numerical study of steady viscous flow past a circular
  cylinder", *Journal of Fluid Mechanics* 98(4), 819–855.
- Abbott, I. H. & von Doenhoff, A. E. (1959), *Theory of Wing Sections*, Dover.
- Anderson, J. D. (2016), *Fundamentals of Aerodynamics*, 6th ed., McGraw-Hill,
  §1.5 and §4.3 for the moment sign convention.
- Schlichting, H. & Gersten, K. (2017), *Boundary-Layer Theory*, 9th ed.,
  Springer, §18.2 for the `1/7`-power skin-friction correlation.
- Simpson, R. L. (1989), "Turbulent Boundary-Layer Separation", *Annual Review of
  Fluid Mechanics* 21, 205–234.
