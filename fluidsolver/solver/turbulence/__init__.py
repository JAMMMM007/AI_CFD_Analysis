"""Turbulence closures for the RANS momentum equations."""

from fluidsolver.solver.turbulence.base import TurbulenceModel
from fluidsolver.solver.turbulence.laminar import Laminar
from fluidsolver.solver.turbulence.sst import KOmegaSST

MODELS = {"laminar": Laminar, "k-omega-sst": KOmegaSST}

__all__ = ["TurbulenceModel", "Laminar", "KOmegaSST", "MODELS"]
