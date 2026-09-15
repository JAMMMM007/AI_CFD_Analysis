"""The grid-convergence machinery, tested against sequences whose answer is known.

A GCI is a number the project intends to publish beside every force coefficient,
so the arithmetic that produces it needs to be verified against something other
than itself. The strongest available check is a manufactured sequence: choose an
exact limit and an exact order, generate three values that obey them, and require
the procedure to recover both. That is exact rather than approximate, so the
tolerances here are numerical rather than physical.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.convergence import (
    SAFETY_FACTOR,
    analyse,
    representative_size,
)


def manufactured(limit: float, coefficient: float, order: float, sizes):
    """``f(h) = limit + coefficient h^order``, sampled at three sizes."""
    return [limit + coefficient * h**order for h in sizes]


class TestObservedOrder:
    @pytest.mark.parametrize("order", [1.0, 1.261, 2.0, 2.88])
    @pytest.mark.parametrize("ratio", [1.3, 1.5, 2.0])
    def test_a_manufactured_sequence_returns_its_own_order(self, order, ratio):
        """The whole point: the order is measured, never assumed."""
        sizes = [1.0, ratio, ratio**2]
        values = manufactured(1.515358, 0.02, order, sizes)
        result = analyse("Cd", values, sizes)
        assert result.order == pytest.approx(order, rel=1e-9)

    @pytest.mark.parametrize("order", [0.735, 1.261, 2.0])
    def test_the_extrapolated_limit_is_the_manufactured_one(self, order):
        sizes = [1.0, 1.5, 2.25]
        values = manufactured(2.2004, -0.05, order, sizes)
        result = analyse("wake", values, sizes)
        assert result.extrapolated == pytest.approx(2.2004, rel=1e-9)

    def test_it_reproduces_the_audit_s_independently_computed_cylinder_study(self):
        """The cross-check that this arithmetic is not merely self-consistent.

        The 2026-09-07 physics audit ran three systematically refined cylinder
        meshes from its own script and reported an observed order of 1.261 and a
        Richardson limit of 1.515358 for ``Cd``. Those values are used here as a
        fixture: the sequence is the audit's, the ratios are the ones it assumed
        (``r = 1.5``, hence sizes 1, 1.5, 2.25), and this module has to arrive at
        the same two numbers by a completely separate implementation.

        Note what is *not* being claimed. This checks the procedure, not the
        study. Whether the audit's family really refines by 1.5 in ``h`` is a
        different question, and :func:`cylinder_family` answers it differently --
        see its docstring.
        """
        result = analyse(
            "Cd", [1.514945209, 1.514669624, 1.514210039], [1.0, 1.5, 2.25]
        )
        assert result.order == pytest.approx(1.261, abs=5e-4)
        assert result.extrapolated == pytest.approx(1.515358, abs=5e-6)

    def test_assuming_second_order_understates_the_band(self):
        """The reason the order is measured, in the terms the reader will meet it.

        Take one fixed sequence -- the audit's cylinder study -- and compute the
        band twice: once at the order the data actually shows, and once at the
        formal order the scheme claims. The formal one is the narrower, so
        assuming it is not the conservative choice, and a code whose observed
        order is far from its formal order fails Roache's precondition for the
        extrapolation to mean anything in the first place.
        """
        values, sizes = [1.514945209, 1.514669624, 1.514210039], [1.0, 1.5, 2.25]
        result = analyse("Cd", values, sizes)
        relative = abs((values[0] - values[1]) / values[0])

        measured = SAFETY_FACTOR * relative / (1.5**result.order - 1.0)
        assumed = SAFETY_FACTOR * relative / (1.5**2.0 - 1.0)

        assert result.gci_fine == pytest.approx(measured)
        assert measured > assumed
        assert measured / assumed == pytest.approx(1.87, abs=0.02)

    def test_the_band_is_the_safety_factor_times_the_relative_difference(self):
        """The formula, written out once, so a transcription error cannot hide."""
        sizes = [1.0, 1.5, 2.25]
        values = manufactured(1.5, 0.03, 1.7, sizes)
        result = analyse("Cd", values, sizes)
        expected = (
            SAFETY_FACTOR
            * abs((values[0] - values[1]) / values[0])
            / (1.5**result.order - 1.0)
        )
        assert result.gci_fine == pytest.approx(expected, rel=1e-12)


class TestRefusals:
    def test_a_family_refined_too_little_is_refused(self):
        """Celik's own precondition, and the one this project's meshes fail.

        Below ``r = 1.3`` the difference between two solutions is comparable with
        the iteration error in them, and the extrapolation fits that instead of
        the discretisation. Refusing is the honest outcome: a GCI computed on
        such a family is a number with nothing behind it.
        """
        sizes = [1.0, 1.1, 1.21]
        with pytest.raises(ValueError, match="1.3"):
            analyse("Cd", manufactured(1.5, 0.02, 2.0, sizes), sizes)

    def test_meshes_given_in_the_wrong_order_are_refused(self):
        with pytest.raises(ValueError, match="fine to coarse"):
            analyse("Cd", [1.0, 1.1, 1.2], [2.25, 1.5, 1.0])

    def test_two_identical_finest_values_are_refused(self):
        with pytest.raises(ValueError, match="agree exactly"):
            analyse("Cd", [1.5, 1.5, 1.6], [1.0, 1.5, 2.25])


class TestNonMonotone:
    def test_an_oscillatory_sequence_is_flagged_rather_than_hidden(self):
        """It still returns a number, and the number should not be believed.

        A sequence whose successive differences change sign is not in the
        asymptotic range, and that is a finding about the mesh family rather than
        an inconvenience in the arithmetic. Suppressing it would hide exactly the
        case a reader most needs to be told about.
        """
        sizes = [1.0, 1.5, 2.25]
        result = analyse("Cd", [1.500, 1.510, 1.505], sizes)
        assert not result.monotone
        assert "OSCILLATORY" in str(result)

    def test_a_clean_sequence_is_not_flagged(self):
        sizes = [1.0, 1.5, 2.25]
        result = analyse("Cd", manufactured(1.5, 0.02, 1.5, sizes), sizes)
        assert result.monotone
        assert "OSCILLATORY" not in str(result)


class TestRepresentativeSize:
    def test_it_is_the_root_mean_cell_area(self):
        class Fake:
            volume = np.full((10, 4), 0.25)

        # 40 cells of area 0.25: sqrt(10 / 40) = 0.5.
        assert representative_size(Fake()) == pytest.approx(0.5)

    def test_halving_every_cell_edge_halves_it(self):
        class Coarse:
            volume = np.full((10, 4), 4.0e-4)

        class Fine:
            volume = np.full((20, 8), 1.0e-4)

        assert representative_size(Coarse()) / representative_size(Fine()) == (
            pytest.approx(2.0)
        )
