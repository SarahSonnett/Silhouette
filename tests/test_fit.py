"""Round-trip recovery tests for silhouette.fit."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.apparitions import Apparition  # noqa: E402
from silhouette.fit import fit_shape  # noqa: E402
from silhouette.model import (  # noqa: E402
    aspect_angle, amplitude_model, mean_mag_model,
)


def _make_apparitions(ab, bc, plon, plat, n=10, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lons = np.linspace(0, 324, n)
    lats = rng.uniform(-5, 5, n)
    apps = []
    for i in range(n):
        th = aspect_angle(lons[i], lats[i], plon, plat)
        amp = float(amplitude_model(ab, bc, th) + rng.normal(0, noise))
        mm = float(mean_mag_model(ab, bc, th, zero_point=15.0) + rng.normal(0, noise))
        apps.append(Apparition(
            index=i, epoch_mid=58000.0 + 200 * i, n_points=40, span_days=2.0,
            amplitude=max(amp, 1e-3), amplitude_err=max(noise, 0.01),
            mean_mag=mm, mean_mag_err=max(noise, 0.01),
            alpha_mean=5.0, rhelio_mean=2.5, delta_mean=1.6,
            ecl_lon=float(lons[i]), ecl_lat=float(lats[i]), geom_source="file"))
    return apps


def test_noiseless_recovery():
    ab, bc, plon, plat = 1.6, 1.3, 60.0, 35.0
    apps = _make_apparitions(ab, bc, plon, plat, n=12, noise=0.0)
    fit = fit_shape(apps)
    assert np.isclose(fit.ab, ab, atol=0.05)
    assert np.isclose(fit.bc, bc, atol=0.1)
    # Pole recovered up to the prograde/retrograde mirror ambiguity.
    best = (fit.pole_lon, fit.pole_lat)
    mir = fit.mirror_pole
    ok = (np.isclose(best[0], plon, atol=8) and np.isclose(best[1], plat, atol=8)) or \
         (np.isclose(mir[0], plon, atol=8) and np.isclose(mir[1], plat, atol=8))
    assert ok


def test_single_apparition_lower_bound():
    apps = _make_apparitions(2.0, 1.4, 30.0, 20.0, n=12)[:1]
    fit = fit_shape(apps)
    assert fit.degenerate and fit.pole_lon is None
    assert fit.ab >= 1.0  # an a/b lower bound is reported


def test_mirror_is_reported():
    apps = _make_apparitions(1.5, 1.25, 100.0, -25.0, n=10)
    fit = fit_shape(apps)
    assert fit.mirror_pole is not None
