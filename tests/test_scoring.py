"""Tests for the solution scorer (silhouette.scoring / calibration / planning).

A tiny synthetic data set keeps these fast and single-threaded: a known
irregular body seen in four apparitions, inverted with a coarse shape
expansion from a handful of starting poles.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.calibration import (  # noqa: E402
    FEATURES,
    ScoreModel,
    classical_accept,
    fit_score_model,
    irregular_shape,
    random_geometry,
    random_truth,
    reliability,
    render,
    yield_curve,
)
from silhouette.forward import convex_lightcurve, ls_lambert  # noqa: E402
from silhouette.inversion import LightCurveObs, _sep_deg  # noqa: E402
from silhouette.planning import (  # noqa: E402
    phase_angle_deg,
    recommend_observations,
    shift_invariant_rms,
    solar_elongation_deg,
)
from silhouette.scoring import (  # noqa: E402
    PopulationPrior,
    _rebase_phi0,
    alias_periods,
    bootstrap_stability,
    candidate_features,
    combine_uncalibrated,
    data_features,
    find_candidates,
    likelihood_weights,
    longitude_coverage,
    model_flux,
    neff_factor,
    score_solutions,
    shape_proxies,
)
from silhouette.shapes import ConvexShape  # noqa: E402

PERIOD = 0.3
POLE = (40.0, 50.0)
FAST = dict(lmax=2, n_normals=80, max_nfev=80)
GRID = [(0.0, 45.0), (120.0, 45.0), (240.0, 45.0), (60.0, -45.0), (180.0, -45.0), (300.0, -45.0)]


def _dir(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


@pytest.fixture(scope="module")
def lcs():
    rng = np.random.default_rng(7)
    truth = irregular_shape(1.7, 1.2, 0.1, rng, n_normals=200)
    out = []
    for k, lon in enumerate((10.0, 100.0, 190.0, 280.0)):
        e, s = _dir(lon, 3.0), _dir(lon + 12.0, 1.0)
        t = k * 400.0 + np.sort(rng.uniform(0.0, PERIOD, 30))
        f = convex_lightcurve(truth, t, s, e, *POLE, PERIOD, phi0=0.4, t0=0.0,
                              phot_func=ls_lambert)
        sig = 0.01 * f.mean()
        out.append(LightCurveObs(t, f + rng.normal(0, sig, f.size), np.full(f.size, sig), s, e))
    return out


@pytest.fixture(scope="module")
def cands(lcs):
    c = find_candidates(lcs, [PERIOD], pole_grid=GRID, **FAST)
    noise = likelihood_weights(c, lcs)
    bootstrap_stability(c, lcs, n_boot=3, seed=1, boot_max_nfev=15, **FAST)
    candidate_features(c, lcs, noise_diag=noise)
    combine_uncalibrated(c)
    return c


# --- small pure functions ---------------------------------------------------

def test_alias_periods_spacing():
    p = alias_periods(0.25, 1000.0, n_alias=2)
    assert p.size == 5
    np.testing.assert_allclose(np.diff(p), 0.25 ** 2 / 1000.0, rtol=1e-9)
    assert 0.25 in np.round(p, 12)
    h = alias_periods(0.25, 1000.0, n_alias=0, harmonics=(0.5, 2.0))
    np.testing.assert_allclose(h, [0.125, 0.25, 0.5])


def test_shape_proxies_exact_for_ellipsoid():
    ab, bc = shape_proxies(ConvexShape.from_ellipsoid(2.0, 1.25, 1.0, 400))
    assert ab == pytest.approx(1.6, rel=0.01)
    assert bc == pytest.approx(1.25, rel=0.01)


def test_longitude_coverage():
    assert longitude_coverage(np.array([10.0])) == 0.0
    assert longitude_coverage(np.array([350.0, 10.0])) == pytest.approx(20.0)
    assert longitude_coverage(np.array([0.0, 90.0, 180.0, 270.0])) == pytest.approx(270.0)


def test_neff_factor_bounds():
    assert neff_factor(0.0) == 1.0
    assert neff_factor(-0.5) == 1.0
    assert neff_factor(0.9) == pytest.approx(0.1 / 1.9)
    assert neff_factor(0.999) == 0.05


def test_rebase_phi0_preserves_orientation():
    shape = ConvexShape.from_ellipsoid(1.8, 1.2, 1.0, 150)
    t = np.linspace(5.0, 5.3, 20)
    s, e = _dir(0, 0), _dir(15, 0)
    a = convex_lightcurve(shape, t, s, e, 30, 40, PERIOD, phi0=1.1, t0=0.0)
    b = convex_lightcurve(shape, t, s, e, 30, 40, PERIOD,
                          phi0=_rebase_phi0(1.1, PERIOD, 0.0, 5.0), t0=5.0)
    np.testing.assert_allclose(a, b, rtol=1e-9)


# --- scorer pipeline ----------------------------------------------------------

def test_true_pole_is_top_candidate(cands):
    best = max(cands, key=lambda c: c.probability)
    assert _sep_deg(best.pole_lon, best.pole_lat, *POLE) < 15.0
    assert best.like_weight > 0.5


def test_weights_and_fractions_normalised(cands):
    assert sum(c.like_weight for c in cands) == pytest.approx(1.0)
    assert sum(c.probability for c in cands) == pytest.approx(1.0)
    assert sum(c.boot_frac for c in cands) <= 1.0 + 1e-9
    assert all(0.0 <= c.boot_frac <= 1.0 for c in cands)


def test_every_feature_present_and_finite(cands):
    for c in cands:
        missing = [f for f in FEATURES if f not in c.features]
        assert not missing, missing
        assert all(np.isfinite(c.features[f]) for f in FEATURES)


def test_candidates_are_distinct_families(cands):
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if a.period == b.period:
                assert _sep_deg(a.pole_lon, a.pole_lat, b.pole_lon, b.pole_lat) >= 15.0


def test_prior_shifts_weights(lcs, cands):
    import copy
    c2 = copy.deepcopy(cands)
    # a prior that strongly disfavours the northern hemisphere
    likelihood_weights(c2, lcs, prior=PopulationPrior(
        pole_lat_logpdf=lambda b: -50.0 if b > 0 else 0.0))
    north = [c for c in c2 if c.pole_lat > 0]
    south = [c for c in c2 if c.pole_lat <= 0]
    if north and south:
        assert sum(c.like_weight for c in south) > sum(
            c.like_weight for c in cands if c.pole_lat <= 0)


def test_model_flux_matches_fit_chi2(lcs, cands):
    c = cands[0]
    models = model_flux(c.fit, lcs)
    chi = sum(float(np.sum(((lc.flux - m) / lc.sigma) ** 2)) for lc, m in zip(lcs, models))
    assert chi <= c.chi2 * 1.001 + 1e-9      # fit chi2 also carries the closure penalty


def test_data_features(lcs):
    d = data_features(lcs)
    assert d["n_app"] == 4
    assert d["n_lc"] == 4
    assert d["lon_coverage"] == pytest.approx(270.0, abs=1.0)
    assert 10.0 < d["phase_max"] < 14.0
    assert d["noise_frac"] == pytest.approx(0.01, rel=0.3)


def test_score_solutions_end_to_end(lcs):
    rep = score_solutions(lcs, [PERIOD], pole_grid=GRID[:3], n_boot=2,
                          boot_max_nfev=10, **FAST)
    assert not rep.calibrated
    probs = [c.probability for c in rep.candidates]
    assert probs == sorted(probs, reverse=True)
    assert "UNCALIBRATED" in rep.summary()


# --- calibration --------------------------------------------------------------

def test_irregular_shape_is_closed_and_positive():
    rng = np.random.default_rng(0)
    s = irregular_shape(2.0, 1.3, 0.25, rng)
    assert s.closure_residual() < 1e-3
    assert np.all(s.areas > 0)


def test_random_geometry_and_render():
    rng = np.random.default_rng(4)
    truth = random_truth(rng)
    plan = random_geometry(rng, truth.period)
    assert 1 <= plan.n_app <= 6 and plan.n_lc >= plan.n_app
    out = render(truth, plan, rng)
    assert len(out) == plan.n_lc
    alphas = np.concatenate([phase_angle_deg(lc.sun, lc.earth) for lc in out])
    assert alphas.max() <= plan.phase_max + 1.0


def test_score_model_fit_predict():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, len(FEATURES)))
    y = (X[:, FEATURES.index("boot_frac")] + 0.3 * rng.normal(size=n) > 0).astype(float)
    groups = np.repeat(np.arange(n // 4), 4)
    m = fit_score_model(X, y, groups, n_splits=3)
    p = m.predict_from_matrix(X)
    assert np.all((p >= 0) & (p <= 1))
    assert np.corrcoef(p, y)[0, 1] > 0.5
    assert isinstance(m, ScoreModel)


def test_classical_accept_rule():
    base = dict(cand_period=0.3, cand_lon=10.0, cand_lat=40.0, correct=1.0)
    rows = [dict(base, cand_redchi2=1.00),
            dict(base, cand_redchi2=1.05, cand_lon=190.0),          # mirror, close
            dict(base, cand_redchi2=1.50, cand_lon=90.0, cand_lat=-20.0)]
    assert classical_accept(rows, allow_mirror=False) is None
    assert classical_accept(rows, allow_mirror=True)["cand_redchi2"] == 1.00
    rows[1]["cand_lat"] = -40.0                                        # antipode, not mirror
    assert classical_accept(rows, allow_mirror=True) is None


def test_yield_curve_and_reliability():
    p = np.array([0.9, 0.8, 0.3, 0.1])
    ok = np.array([1, 0, 1, 0])
    thr, yl, fr = yield_curve(p, ok, 4, thresholds=np.array([0.0, 0.5, 0.85]))
    np.testing.assert_allclose(yl, [0.5, 0.25, 0.25])
    np.testing.assert_allclose(fr, [0.5, 0.5, 0.0])
    mp, fy, n = reliability(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(fy, [0.0, 1.0])


# --- planning -----------------------------------------------------------------

def test_geometry_angles():
    sun = np.array([[-2.5, 0.0, 0.0]])              # asteroid at 2.5 AU, opposition
    earth = np.array([[-1.5, 0.0, 0.0]])
    assert phase_angle_deg(sun, earth)[0] == pytest.approx(0.0, abs=1e-6)
    assert solar_elongation_deg(sun, earth)[0] == pytest.approx(180.0, abs=1e-6)


def test_shift_invariant_rms():
    x = np.sin(np.linspace(0, 2 * np.pi, 36, endpoint=False))
    assert shift_invariant_rms(x, np.roll(x, 7)) == pytest.approx(0.0, abs=1e-12)
    asym = np.array([0, 1, 2, 3, 0, 0], dtype=float)
    assert shift_invariant_rms(asym, asym[::-1]) > 0.0
    assert shift_invariant_rms(asym, asym[::-1], allow_reverse=True) == pytest.approx(0.0)


def test_recommender_ranks_and_filters(cands):
    lons = np.arange(0.0, 360.0, 30.0)
    earth = np.array([_dir(l, 5.0) * 1.5 for l in lons])
    sun = np.array([_dir(l + 10.0, 1.0) * 2.5 for l in lons])
    recs = recommend_observations(cands, lons, sun, earth, min_elongation_deg=0.0)
    assert len(recs) == lons.size
    scores = [round(r.p_true_after, 3) for r in recs]
    assert scores == sorted(scores, reverse=True)
    p = np.array([c.probability for c in cands])
    p = p / p.sum()
    assert all(np.sum(p * p) - 1e-9 <= r.p_true_after <= 1.0 + 1e-9 for r in recs)
    same = recommend_observations([cands[0], cands[0]], lons, sun, earth,
                                  probabilities=[0.5, 0.5], min_elongation_deg=0.0)
    assert max(r.expected_dchi2 for r in same) == pytest.approx(0.0, abs=1e-9)
    assert max(r.p_true_after for r in same) == pytest.approx(0.5)
    strict = recommend_observations(cands, lons, sun, earth, min_elongation_deg=179.0)
    assert len(strict) < lons.size


def test_spin_vector_pole_matches_rotation_sense():
    """Silhouette's pole is anti-parallel to the angular momentum; the helper flips it."""
    from silhouette.forward import ecliptic_to_body_matrix
    from silhouette.scoring import spin_vector_pole
    lon, lat = 30.0, 40.0
    xb = np.array([1.0, 0.0, 0.0])
    p1 = ecliptic_to_body_matrix(lon, lat, 0.0).T @ xb
    p2 = ecliptic_to_body_matrix(lon, lat, 0.1).T @ xb
    omega = np.cross(p1, p2)                     # direction of physical spin
    slon, slat = spin_vector_pole(lon, lat)
    assert np.dot(omega, _dir(slon, slat)) > 0
    assert np.dot(omega, _dir(lon, lat)) < 0


def test_simple_orbit_geometry_opposition():
    from silhouette.planning import simple_orbit_geometry
    sun, earth = simple_orbit_geometry([0.0], a_au=2.5, incl_deg=0.0, lon0_deg=0.0)
    np.testing.assert_allclose(np.linalg.norm(earth[0]), 1.5)
    assert solar_elongation_deg(sun, earth)[0] == pytest.approx(180.0)


# --- Squint/SpinDoc glue ---------------------------------------------------------

def test_split_nights_and_reduction():
    from silhouette.io import Photometry
    from silhouette.pipeline import AU_LIGHT_DAYS, lightcurves_from_photometry, split_nights
    t = np.concatenate([58700.1 + np.arange(12) * 0.01, 58701.1 + np.arange(10) * 0.01,
                        58702.1 + np.arange(3) * 0.01])
    n = t.size
    phot = Photometry(time=t, mag=np.full(n, 15.0), merr=np.full(n, 0.02),
                      rhelio=np.full(n, 2.0), delta=np.full(n, 1.0), alpha=np.full(n, 10.0))
    assert [len(i) for i in split_nights(t)] == [12, 10, 3]
    geo = (np.array([[-2.0, 0, 0], [-2.0, 0.1, 0]]), np.array([[-1.0, 0, 0], [-1.0, 0.1, 0]]))
    lcs = lightcurves_from_photometry(phot, geometry=geo, min_points=8)
    assert len(lcs) == 2                                    # 3-point night dropped
    np.testing.assert_allclose(lcs[0].flux, 10 ** (-0.4 * (15.0 - 5 * np.log10(2.0))))
    np.testing.assert_allclose(lcs[0].times, t[:12] - AU_LIGHT_DAYS)
    rel = lcs[0].sigma / lcs[0].flux
    np.testing.assert_allclose(rel, np.sqrt(0.02 ** 2 + 0.005 ** 2) * 0.4 * np.log(10))
    with pytest.raises(ValueError):
        lightcurves_from_photometry(phot)


def test_upgrade_legacy_s2_matches_new_rule():
    import pandas as pd
    from silhouette.calibration import upgrade_legacy_s2
    # one injection, best chi2_nu 0.5: old rule used s2=1, new uses 0.5
    df = pd.DataFrame({"inj": [0, 0], "redchi2_min": [0.5, 0.5],
                       "dchi2_scaled": [0.0, 2.0], "margin_scaled": [2.0, -2.0],
                       "like_weight": [0.73, 0.27], "boot_frac": [0.6, 0.4],
                       "p_uncal": [0.8, 0.2]})
    up = upgrade_legacy_s2(df)
    np.testing.assert_allclose(up["dchi2_scaled"], [0.0, 4.0])
    w = np.array([1.0, np.exp(-2.0)])
    np.testing.assert_allclose(up["like_weight"], w / w.sum())
    assert upgrade_legacy_s2(up) is up


def test_group_by_apparition_with_phase_correction():
    from silhouette.io import Photometry
    from silhouette.pipeline import lightcurves_from_photometry
    t = np.concatenate([58700.1 + np.arange(10) * 0.01, 58703.1 + np.arange(10) * 0.01,
                        58900.1 + np.arange(10) * 0.01])
    n = t.size
    alpha = np.where(t < 58800, 5.0, 15.0)
    phot = Photometry(time=t, mag=np.full(n, 15.0), merr=np.full(n, 0.02),
                      rhelio=np.full(n, 2.0), delta=np.full(n, 1.0), alpha=alpha)
    sun = np.array([[-2.0, 0, 0], [-2.0, 0.1, 0], [-2.0, 0.5, 0]])
    earth = np.array([[-1.0, 0, 0], [-1.0, 0.1, 0], [-1.0, 0.5, 0]])
    lcs = lightcurves_from_photometry(phot, geometry=(sun, earth), group="apparition",
                                      phase_G=0.15)
    assert [len(lc) for lc in lcs] == [20, 10]
    assert lcs[0].earth.shape == (20, 3)
    np.testing.assert_allclose(lcs[0].earth[10], earth[1])
    # phase correction brightens the higher-phase apparition more
    assert lcs[1].flux[0] > lcs[0].flux[0]
