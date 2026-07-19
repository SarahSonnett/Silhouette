"""Tests for convex light-curve inversion (silhouette.inversion).

The headline test is a synthetic round trip: generate light curves from a known
ellipsoid and spin state, add noise, then invert from a deliberately wrong
starting pole and check that both the pole and the DEEVE axis ratios come back.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.forward import convex_lightcurve, ls_lambert  # noqa: E402
from silhouette.inversion import (  # noqa: E402
    LightCurveObs,
    ellipsoid_coeffs,
    invert_convex,
    _optimal_scale,
)
from silhouette.shapes import (  # noqa: E402
    ConvexShape,
    ellipsoid_gaussian_image,
    fibonacci_sphere,
    real_sph_harm_basis,
)

TRUE_AXES = (2.0, 1.4, 1.0)
TRUE_POLE = (60.0, 35.0)
PERIOD = 0.25


def _dir(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _angular_sep(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.degrees(np.arccos(np.clip(
        np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


@pytest.fixture(scope="module")
def synthetic_lightcurves():
    rng = np.random.default_rng(3)
    truth = ConvexShape.from_ellipsoid(*TRUE_AXES, 250)
    lcs = []
    for k, lon in enumerate(np.linspace(0, 315, 7)):
        earth = _dir(lon, rng.uniform(-5, 5))
        sun = _dir(lon + rng.uniform(3, 15), rng.uniform(-5, 5))
        t = np.sort(rng.uniform(0, PERIOD, 35)) + k * 200.0
        flux = convex_lightcurve(truth, t, sun, earth, *TRUE_POLE, PERIOD,
                                 phi0=0.7, t0=0.0, phot_func=ls_lambert)
        sigma = 0.02 * flux.mean()
        lcs.append(LightCurveObs(t, flux + rng.normal(0, sigma, flux.size),
                                 np.full(flux.size, sigma), sun, earth, relative=True))
    return lcs


@pytest.fixture(scope="module")
def inversion(synthetic_lightcurves):
    # start from a pole ~22 deg away from the truth
    return invert_convex(synthetic_lightcurves, period=PERIOD,
                         pole0=(40.0, 20.0), phi0=0.0, lmax=4, n_normals=180)


# ---------------------------------------------------------------- helpers

def test_optimal_scale_recovers_known_factor():
    model = np.linspace(1.0, 2.0, 30)
    assert np.isclose(_optimal_scale(3.7 * model, model, np.ones(30)), 1.0 / 3.7, rtol=1e-9)
    assert np.isclose(_optimal_scale(model, 3.7 * model, np.ones(30)), 3.7, rtol=1e-9)


def test_ellipsoid_coeffs_reproduce_gaussian_image():
    lmax = 6
    normals, solid = fibonacci_sphere(400)
    basis = real_sph_harm_basis(lmax, normals)
    coeffs = ellipsoid_coeffs(2.0, 1.4, 1.0, lmax, normals, solid, basis)
    areas = np.exp(basis @ coeffs) * solid
    truth = ellipsoid_gaussian_image(normals, 2.0, 1.4, 1.0, solid)
    assert np.sqrt(np.mean(((areas - truth) / truth) ** 2)) < 0.05


def test_lightcurveobs_broadcasts_scalar_sigma():
    lc = LightCurveObs(np.linspace(0, 1, 10), np.ones(10), 0.05,
                       _dir(0, 0), _dir(10, 0))
    assert lc.sigma.shape == (10,) and np.allclose(lc.sigma, 0.05)
    assert len(lc) == 10


# ---------------------------------------------------------------- round trip

def test_inversion_converges(inversion):
    assert inversion.success
    assert inversion.redchi2 < 2.0        # data have 2% noise


def test_inversion_recovers_pole(inversion):
    sep = _angular_sep(inversion.pole_lon, inversion.pole_lat, *TRUE_POLE)
    assert sep < 10.0                     # started ~22 deg away


def test_inversion_recovers_axis_ratios(inversion):
    ab, bc = inversion.axis_ratios()
    a, b, c = TRUE_AXES
    assert np.isclose(ab, a / b, rtol=0.12)
    assert np.isclose(bc, b / c, rtol=0.12)


def test_inversion_shape_is_closed(inversion):
    """The fitted Gaussian image must describe a genuine closed convex body."""
    assert inversion.closure < 0.02


def test_inversion_areas_positive(inversion):
    assert np.all(inversion.shape.areas > 0)


def test_summary_mentions_key_quantities(inversion):
    text = inversion.summary()
    for token in ("pole", "chi^2", "DEEVE", "closure"):
        assert token in text


def test_relative_scale_is_profiled_out(synthetic_lightcurves):
    """Rescaling a relative light curve must not change the fit."""
    scaled = [LightCurveObs(lc.times, lc.flux * 17.0, lc.sigma * 17.0,
                            lc.sun, lc.earth, relative=True)
              for lc in synthetic_lightcurves]
    a = invert_convex(synthetic_lightcurves, PERIOD, pole0=(55.0, 30.0),
                      lmax=2, n_normals=120, max_nfev=60)
    b = invert_convex(scaled, PERIOD, pole0=(55.0, 30.0),
                      lmax=2, n_normals=120, max_nfev=60)
    assert np.isclose(a.redchi2, b.redchi2, rtol=1e-6)
    assert np.isclose(a.pole_lat, b.pole_lat, atol=1e-6)


# ---------------------------------------------------------------- multistart

def test_default_pole_grid_covers_sphere():
    from silhouette.inversion import default_pole_grid
    grid = default_pole_grid(n_lon=6)
    assert len(grid) == 30
    lats = {lat for _, lat in grid}
    assert min(lats) < -50 and max(lats) > 50      # both hemispheres sampled


def test_multistart_picks_lowest_chi2(synthetic_lightcurves):
    """Multistart must return the best of its starts and expose the rest."""
    from silhouette.inversion import invert_convex_multistart
    seeds = [(35.0, 10.0), (200.0, -50.0), (300.0, 70.0)]
    res = invert_convex_multistart(synthetic_lightcurves, PERIOD, pole_grid=seeds,
                                   n_workers=1, lmax=3, n_normals=120, max_nfev=120)
    assert len(res.candidates) == len(seeds)
    # candidates are sorted best-first and the winner matches the returned fit
    chis = [c[2] for c in res.candidates]
    assert chis == sorted(chis)
    assert np.isclose(res.redchi2, chis[0])


# ---------------------------------------------------------------- period scan

def test_period_grid_step_keeps_phase_coherent():
    """Step must satisfy dP = P^2 / (T * oversample)."""
    from silhouette.inversion import period_search_grid
    p, baseline, over = 0.25, 50.0, 4.0
    grid = period_search_grid(p, baseline, half_width_frac=0.01, oversample=over)
    assert np.isclose(np.diff(grid).max(), p ** 2 / (baseline * over), rtol=0.2)
    assert grid.min() < p < grid.max()


def test_period_scan_recovers_known_period():
    """Scanning inside the alias-free window must find the true period."""
    from silhouette.inversion import period_search_grid, scan_period
    rng = np.random.default_rng(7)
    truth = ConvexShape.from_ellipsoid(*TRUE_AXES, 200)
    lcs = []
    for k, lon in enumerate([0, 25, 50]):
        earth = _dir(lon, rng.uniform(-5, 5))
        sun = _dir(lon + 8, rng.uniform(-5, 5))
        t = np.sort(rng.uniform(0, PERIOD, 25)) + k * 25.0
        flux = convex_lightcurve(truth, t, sun, earth, *TRUE_POLE, PERIOD,
                                 phi0=0.7, t0=0.0, phot_func=ls_lambert)
        sigma = 0.015 * flux.mean()
        lcs.append(LightCurveObs(t, flux + rng.normal(0, sigma, flux.size),
                                 np.full(flux.size, sigma), sun, earth, relative=True))

    baseline = max(l.times.max() for l in lcs) - min(l.times.min() for l in lcs)
    alias = PERIOD ** 2 / baseline          # one-rotation alias spacing
    grid = period_search_grid(PERIOD * 1.0008, baseline,
                              half_width_frac=0.0018, oversample=8)
    assert (grid.max() - grid.min()) < 2 * alias      # stay inside the alias window

    res = scan_period(lcs, grid, pole_grid=[(35.0, 10.0), (200.0, -40.0)],
                      n_workers=1, lmax=3, n_normals=110, max_nfev=100)
    step = np.diff(grid).max()
    assert abs(res.best_period - PERIOD) <= 1.5 * step
    assert res.redchi2.shape == grid.shape
    assert np.nanmax(res.redchi2) > np.nanmin(res.redchi2)   # landscape has structure
