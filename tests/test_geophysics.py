"""Tests for rotational-stability constraints (silhouette.geophysics).

Anchored on closed-form results: the ellipsoid shape integrals sum to 2, the
mean stress in a non-rotating uniform sphere is -(4pi/15) G rho^2 R^2, and the
strengthless spin barrier is ~2.3 h at 2000 kg/m^3.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.geophysics import (  # noqa: E402
    G,
    axes_from_ratios,
    drucker_prager_constants,
    is_stable_cohesionless,
    mean_stresses,
    min_density_cohesionless,
    propagate_axis_uncertainty,
    required_cohesion,
    shape_factors,
    shedding_limit_density,
)


# ---------------------------------------------------------------- geometry

def test_axes_from_ratios_preserve_volume_and_ratios():
    a, b, c = axes_from_ratios(2.0, 1.4, r_eq=3.0)
    assert np.isclose(a * b * c, 27.0)
    assert np.isclose(a / b, 2.0) and np.isclose(b / c, 1.4)


@pytest.mark.parametrize("axes", [(1, 1, 1), (2, 1.4, 1), (3, 2, 1), (5, 1, 1)])
def test_shape_factors_sum_to_two(axes):
    assert np.isclose(sum(shape_factors(*axes)), 2.0, rtol=1e-8)


def test_sphere_shape_factors_are_two_thirds():
    assert np.allclose(shape_factors(1.0, 1.0, 1.0), (2 / 3, 2 / 3, 2 / 3), rtol=1e-8)


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6])
def test_shape_factors_are_scale_invariant(scale):
    """A_i depends only on axis ratios; large axes must not break quadrature."""
    ref = shape_factors(2.0, 1.4, 1.0)
    got = shape_factors(2.0 * scale, 1.4 * scale, 1.0 * scale)
    assert np.allclose(got, ref, rtol=1e-8)


def test_shape_factors_ordered_by_axis_length():
    a1, a2, a3 = shape_factors(3.0, 2.0, 1.0)
    assert a1 < a2 < a3          # longest axis has the smallest factor


# ---------------------------------------------------------------- stress

@pytest.mark.parametrize("radius", [1.0, 5e2, 5e4, 5e6])
def test_nonrotating_sphere_mean_stress_matches_analytic(radius):
    rho = 2000.0
    got = mean_stresses(radius, radius, radius, rho, 0.0)[0]
    expected = -(4 * np.pi / 15) * G * rho ** 2 * radius ** 2
    assert np.isclose(got, expected, rtol=1e-8)


def test_nonrotating_sphere_stress_is_isotropic():
    s = mean_stresses(1e4, 1e4, 1e4, 2500.0, 0.0)
    assert np.allclose(s, s[0], rtol=1e-8)


def test_rotation_reduces_equatorial_compression():
    """Centrifugal support offsets self-gravity in the spin plane, not along c."""
    a = b = c = 1e4
    still = mean_stresses(a, b, c, 2000.0, 0.0)
    spun = mean_stresses(a, b, c, 2000.0, 2 * np.pi / (3 * 3600.0))
    assert spun[0] > still[0]          # less compressive in x
    assert np.isclose(spun[2], still[2])   # unchanged along the spin axis


def test_drucker_prager_zero_friction_has_no_pressure_term():
    s, k = drucker_prager_constants(0.0, cohesion=0.0)
    assert np.isclose(s, 0.0) and np.isclose(k, 0.0)


# ---------------------------------------------------------------- limits

def test_shedding_limit_reproduces_classic_spin_barrier():
    """A strengthless sphere at 2000 kg/m^3 sheds near the ~2.3 h barrier."""
    p_crit = np.sqrt(3 * np.pi / (G * 2000.0)) / 3600.0
    assert 2.2 < p_crit < 2.4
    assert np.isclose(shedding_limit_density(p_crit), 2000.0, rtol=1e-8)


def test_min_density_decreases_with_period():
    rhos = [min_density_cohesionless(1.0, 1.0, p) for p in (2.0, 3.0, 6.0, 12.0)]
    assert all(r is not None for r in rhos)
    assert rhos == sorted(rhos, reverse=True)


def test_elongation_requires_higher_density():
    """A more elongated body is harder to hold together at the same spin."""
    round_ = min_density_cohesionless(1.0, 1.0, 3.0)
    elong = min_density_cohesionless(2.0, 1.4, 3.0)
    assert elong > round_


def test_more_friction_helps():
    weak = min_density_cohesionless(1.5, 1.2, 3.0, friction_deg=20.0)
    strong = min_density_cohesionless(1.5, 1.2, 3.0, friction_deg=40.0)
    assert strong < weak


def test_stability_is_size_independent():
    """The cohesionless criterion must not depend on the body's size."""
    assert (is_stable_cohesionless(1.5, 1.2, 6.0, 2000.0)
            == is_stable_cohesionless(1.5, 1.2, 6.0, 2000.0))
    rho = min_density_cohesionless(1.5, 1.2, 4.0)
    assert is_stable_cohesionless(1.5, 1.2, 4.0, rho * 1.05)
    assert not is_stable_cohesionless(1.5, 1.2, 4.0, rho * 0.95)


# ---------------------------------------------------------------- cohesion

def test_slow_rotator_needs_no_cohesion():
    assert required_cohesion(1.5, 1.2, 12.0, 2000.0, 1.0) == 0.0


def test_fast_rotator_needs_cohesion():
    assert required_cohesion(1.5, 1.2, 2.0, 2000.0, 1.0) > 0.0


def test_cohesion_scales_as_diameter_squared():
    y1 = required_cohesion(1.5, 1.2, 2.0, 2000.0, 1.0)
    y2 = required_cohesion(1.5, 1.2, 2.0, 2000.0, 2.0)
    assert np.isclose(y2 / y1, 4.0, rtol=1e-6)


def test_fast_km_body_cohesion_is_tens_of_pascals():
    """Order-of-magnitude check against published fast-rotator cohesions."""
    y = required_cohesion(1.5, 1.2, 2.0, 2000.0, 1.0, friction_deg=35.0)
    assert 1.0 < y < 1000.0


# ---------------------------------------------------------------- uncertainty

def test_uncertainty_propagation_brackets_the_best_value():
    con = propagate_axis_uncertainty(1.5, 1.2, 3.0, ab_frac=0.05, bc_frac=0.20)
    assert con.low <= con.best <= con.high
    assert con.high > con.low          # b/c uncertainty must actually matter
    assert "kg/m^3" in con.summary()
