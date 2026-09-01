# Optional dependencies

## pyamg

`pyamg` provides algebraic multigrid, which would accelerate the
pressure-correction solve -- the most expensive part of a SIMPLE iteration, and
the one whose cost grows worst with mesh size.

It is not in `requirements.txt`. There is no cp314 wheel, so pip falls back to
building it, which needs MSVC and fails on a machine without the Visual C++ build
tools. Worse, that failure aborts the *whole* install: pip builds every wheel
before installing any of them, so one unbuildable package leaves the environment
with nothing in it at all.

If you have the build tools, `pip install pyamg` and the solver will pick it up
on its own. Otherwise the fallback is ILU-preconditioned BiCGSTAB, which is
what `linalg.incomplete_lu_preconditioner` provides and is perfectly serviceable
at these mesh sizes.

## numba

Not used. The solver is numpy-vectorised throughout and spends most of its time
inside scipy's sparse kernels, so a JIT would have little left to accelerate
without a rewrite of the assembly into explicit loops.
