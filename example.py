"""End-to-end Silhouette demo with a known ground truth.

Synthesises a multi-apparition photometry file for an asteroid of *known* axis
ratios and pole (using SpotLight as the forward model when available, otherwise
the analytical relations), then runs the full Silhouette pipeline to recover
them — a self-checking round trip. Writes the synthetic data and the recovered
fit figure into ``docs/images/``.

Run:  python example.py
"""

from __future__ import annotations

import os

import numpy as np

from silhouette import (
    read_photometry, reduce_apparitions, resolve_geometry, fit_shape,
    save_summary, HAVE_SPOTLIGHT,
)
from silhouette._compat import HGfunction, spotlight
from silhouette.model import aspect_angle, amplitude_model, mean_mag_model, axes_from_ratios

OUTDIR = os.path.join(os.path.dirname(__file__), "docs", "images")

# ---- ground truth ----------------------------------------------------------
TRUE_AB = 1.6          # a/b
TRUE_BC = 1.3          # b/c
TRUE_POLE_LON = 60.0   # deg
TRUE_POLE_LAT = 35.0   # deg
PERIOD = 0.25          # days
H0 = 15.0              # absolute-ish zero point


def _synthesise(path, n_apparitions=8, pts_per_app=40, seed=1):
    """Write a synthetic multi-apparition photometry file with ecliptic columns."""
    rng = np.random.default_rng(seed)
    a, b, c = axes_from_ratios(TRUE_AB, TRUE_BC)
    rows = []
    base_mjd = 58000.0

    # Spread apparitions around the ecliptic; small spread in latitude.
    lons = np.linspace(0, 315, n_apparitions) + rng.uniform(-10, 10, n_apparitions)
    lats = rng.uniform(-6, 6, n_apparitions)

    for k in range(n_apparitions):
        lon, lat = float(lons[k] % 360), float(lats[k])
        theta = aspect_angle(lon, lat, TRUE_POLE_LON, TRUE_POLE_LAT)
        rh = rng.uniform(2.2, 3.1)
        delta = rh - rng.uniform(0.6, 1.1)
        alpha = rng.uniform(2.0, 18.0)
        dist_mod = 5.0 * np.log10(rh * delta)
        phase_dim = HGfunction(alpha, 0.0, 0.15)            # phase dimming
        app_mjd = base_mjd + k * 200.0

        # rotation-phase sampling over a couple of nights
        phases = np.sort(rng.uniform(0, 1, pts_per_app))
        times = app_mjd + phases * PERIOD + rng.integers(0, 3, pts_per_app)

        # Synthesise from the SAME geometric-scattering relations the fitter
        # assumes, so this is a true round trip. (SpotLight's Lambertian model
        # is used only for the figure mosaic, not the magnitudes — its
        # limb-darkening would bias a geometric-scattering amplitude fit.)
        amp = amplitude_model(TRUE_AB, TRUE_BC, theta)
        mean_off = mean_mag_model(TRUE_AB, TRUE_BC, theta, zero_point=0.0)
        # double-peaked rotation curve with peak-to-trough = amp
        mags = (H0 + mean_off + 0.5 * amp * np.sin(2.0 * 2.0 * np.pi * phases)
                + dist_mod + phase_dim)

        merr = np.full(pts_per_app, 0.02)
        mags = mags + rng.normal(0, 0.02, pts_per_app)
        for t, m, e in zip(times, mags, merr):
            rows.append((t, m, e, rh, delta, alpha, lon, lat))

    rows.sort()
    with open(path, "w") as f:
        f.write("MJD mag merr Rhelio Delta alpha ecl_lon ecl_lat\n")
        for t, m, e, rh, delta, alpha, lon, lat in rows:
            f.write(f"{t:.5f} {m:.4f} {e:.4f} {rh:.4f} {delta:.4f} "
                    f"{alpha:.3f} {lon:.4f} {lat:.4f}\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data_path = os.path.join(OUTDIR, "synthetic_photometry.txt")
    _synthesise(data_path)
    print(f"SpotLight available: {HAVE_SPOTLIGHT}")
    print(f"Wrote synthetic data -> {data_path}\n")

    phot = read_photometry(data_path)
    apps = reduce_apparitions(phot, period=PERIOD)
    resolve_geometry(apps, prefer_file=True)     # uses file ecliptic columns
    fit = fit_shape(apps)

    print(fit.summary())
    print(f"\nground truth: a:b={TRUE_AB}, b:c={TRUE_BC}, "
          f"pole=({TRUE_POLE_LON}, {TRUE_POLE_LAT})")

    out = os.path.join(OUTDIR, "fit_summary.png")
    save_summary(fit, out)
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
