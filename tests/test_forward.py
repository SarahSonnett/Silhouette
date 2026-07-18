"""Tests for the convex forward light-curve model (silhouette.forward).

The strongest check here is a cross-validation against the *independently
derived* analytical relations in :mod:`silhouette.model`: with geometric
scattering at zero phase, the forward model's light-curve amplitude must
reproduce ``amplitude_model(a/b, b/c, aspect)``.

These tests need only the Gaussian image (normals + areas), never the Minkowski
reconstruction, so they run in milliseconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.forward import (  # noqa: E402
    SCATTERING_LAWS,
    convex_brightness,
    convex_lightcurve,
    ecliptic_to_body_matrix,
    geometric,
    lambert,
    lommel_seeliger,
    ls_lambert,
    projected_area,
    rotation_phase,
)
from silhouette.model import amplitude_model  # noqa: E402
from silhouette.shapes import ConvexShape  # noqa: E402

A, B, C = 2.0, 1.4, 1.0
N_DIRS = 400


@pytest.fixture(scope="module")
def ellipsoid():
    return ConvexShape.from_ellipsoid(A, B, C, N_DIRS)


@pytest.fixture(scope="module")
def unit_sphere():
    return ConvexShape.from_ellipsoid(1.0, 1.0, 1.0, N_DIRS)


def _view(aspect_deg):
    """Unit observer direction at a given aspect from an ecliptic-north pole."""
    t = np.radians(aspect_deg)
    return np.array([np.sin(t), 0.0, np.cos(t)])


# ---------------------------------------------------------------- frames

@pytest.mark.parametrize("lam,bet", [(0, 90), (35, -67), (210, 12), (300, -5)])
@pytest.mark.parametrize("phi", [0.0, 1.3, 4.7])
def test_rotation_maps_pole_to_body_z(lam, bet, phi):
    p = np.array([np.cos(np.radians(bet)) * np.cos(np.radians(lam)),
                  np.cos(np.radians(bet)) * np.sin(np.radians(lam)),
                  np.sin(np.radians(bet))])
    assert np.allclose(ecliptic_to_body_matrix(lam, bet, phi) @ p, [0, 0, 1], atol=1e-12)


def test_rotation_matrix_is_orthonormal():
    r = ecliptic_to_body_matrix(123.0, -40.0, 2.1)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_rotation_phase_advances_one_turn_per_period():
    phi = rotation_phase([0.0, 0.5, 1.0], period=1.0, phi0=0.0, t0=0.0)
    assert np.allclose(phi, [0.0, np.pi, 2 * np.pi])


# ---------------------------------------------------------------- sphere

def test_sphere_projected_area_is_pi(unit_sphere):
    assert np.isclose(projected_area(unit_sphere, [0, 0, 1]), np.pi, rtol=2e-3)


def test_sphere_lightcurve_is_flat(unit_sphere):
    e = _view(60.0)
    lc = convex_lightcurve(unit_sphere, np.linspace(0, 1, 40), e, e,
                           pole_lon=0.0, pole_lat=90.0, period=1.0,
                           phot_func=geometric)
    assert lc.std() / lc.mean() < 5e-3        # a sphere cannot vary with rotation


# ---------------------------------------------------------------- geometry

@pytest.mark.parametrize("aspect", [90.0, 60.0, 30.0])
def test_projected_area_matches_analytic_ellipsoid(ellipsoid, aspect):
    t = np.radians(aspect)
    e = _view(aspect)
    phis = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    got = np.array([projected_area(ellipsoid, ecliptic_to_body_matrix(0.0, 90.0, p) @ e)
                    for p in phis])
    expected = np.pi * np.sqrt(
        C ** 2 * np.sin(t) ** 2 * (A ** 2 * np.sin(phis) ** 2 + B ** 2 * np.cos(phis) ** 2)
        + A ** 2 * B ** 2 * np.cos(t) ** 2)
    assert np.sqrt(np.mean(((got - expected) / expected) ** 2)) < 5e-3


# ---------------------------------------------------------------- cross-validation

@pytest.mark.parametrize("aspect", [90.0, 70.0, 50.0, 30.0])
def test_amplitude_matches_analytic_amplitude_model(ellipsoid, aspect):
    """Forward model vs the independent closed-form amplitude-aspect relation."""
    e = _view(aspect)
    lc = convex_lightcurve(ellipsoid, np.linspace(0, 1, 120, endpoint=False), e, e,
                           pole_lon=0.0, pole_lat=90.0, period=1.0,
                           phot_func=geometric)
    amp = -2.5 * np.log10(lc.min() / lc.max())
    assert np.isclose(amp, amplitude_model(A / B, B / C, np.radians(aspect)), atol=5e-3)


def test_pole_on_view_has_no_amplitude(ellipsoid):
    """Viewing down the spin axis shows the constant a*b face."""
    e = _view(0.0)
    lc = convex_lightcurve(ellipsoid, np.linspace(0, 1, 60, endpoint=False), e, e,
                           pole_lon=0.0, pole_lat=90.0, period=1.0, phot_func=geometric)
    assert lc.std() / lc.mean() < 5e-3


# ---------------------------------------------------------------- scattering laws

@pytest.mark.parametrize("name", sorted(SCATTERING_LAWS))
def test_scattering_laws_are_finite_and_nonnegative(name):
    f = SCATTERING_LAWS[name]
    mu0 = np.linspace(0.01, 1.0, 25)
    mu = np.linspace(1.0, 0.01, 25)
    out = f(mu0, mu, 0.1, None)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)


def test_ls_lambert_reduces_to_lommel_seeliger_when_c_zero():
    mu0 = np.linspace(0.05, 1.0, 20)
    mu = np.linspace(1.0, 0.05, 20)
    assert np.allclose(ls_lambert(mu0, mu, 0.0, 0.0), lommel_seeliger(mu0, mu, 0.0))


def test_lommel_seeliger_safe_at_zero_denominator():
    assert np.allclose(lommel_seeliger(np.zeros(3), np.zeros(3), 0.0), 0.0)


def test_unlit_or_hidden_facets_contribute_nothing(ellipsoid):
    """Sun behind the body: no facet is both visible and illuminated."""
    assert convex_brightness(ellipsoid, [0, 0, 1], [0, 0, -1], phot_func=lambert) == 0.0


def test_brightness_positive_for_normal_geometry(ellipsoid):
    for law in (geometric, lambert, lommel_seeliger, ls_lambert):
        assert convex_brightness(ellipsoid, [1, 0, 0], [1, 0, 0], phot_func=law) > 0
