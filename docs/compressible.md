# Extending to compressible subsonic flow

The solver is incompressible. This note records what a compressible path would
have to change, and which seams were left in place for it.

## What is already prepared

**Density is a field, not a scalar.** `Fluid.density_field(shape)` returns an
array, and every flux expression multiplies by density rather than folding it
into a constant. Nothing in the discretisation assumes it is uniform.

**One scalar-transport assembler.** `operators.add_convection` and
`operators.add_diffusion` are generic in the transported quantity: momentum,
pressure correction, `k` and `omega` all go through them. An energy equation is
another caller, not another discretisation.

**Fluid properties sit behind `Fluid`.** Sutherland's law for viscosity and a
temperature-dependent conductivity are additions to that class rather than
changes to its callers.

## What would have to change

**An energy equation.** Total enthalpy or temperature, transported by the same
assembler, with viscous dissipation and pressure work as sources. This is the
smallest of the changes.

**An equation of state, and pressure re-entering continuity.** This is the real
work. In the incompressible pressure correction the coefficient of `p'` comes
entirely from the velocity correction, `rho * D * g`. Compressible SIMPLE adds a
term from `d(rho)/dp`, so that the pressure equation becomes convective-diffusive
rather than purely diffusive, and changes character as the Mach number rises.
`simple.PressureVelocityCoupling.pressure_correction` is where that term goes.

**Density upwinding in the face fluxes.** Face density has to be biased upstream
above roughly Mach 0.3, or the scheme goes unstable where it starts to matter.
`simple.PressureVelocityCoupling.face_fluxes` is the place.

**Characteristic far-field boundary conditions.** The current far field splits on
the sign of `u . n`, which is the right idea but the incompressible version of
it. Compressible flow needs the split made on the sign of the acoustic
characteristics, so that at subsonic inflow four quantities are imposed and one
extrapolated. `bc.Boundaries` is self-contained and is the only place affected.

**Shock capturing, above about Mach 0.7.** A limiter on the convective scheme,
and a sensor to switch it on. Below that it is not needed and the deferred
correction already in place is sufficient.

## What would not change

Geometry, meshing, the finite-volume metrics, the linear algebra, the turbulence
model, and the entire GUI. The mesh would want more resolution where shocks form,
but nothing about how it is generated.
