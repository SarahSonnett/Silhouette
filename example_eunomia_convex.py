"""Convex light-curve inversion of real DAMIT photometry of (15) Eunomia.

Runs the Phase-1a convex inversion on the bundled 109 DAMIT light curves and
compares the recovered pole and DEEVE axis ratios with the DAMIT convex model
(λ ≈ 3°, β ≈ −67°, P = 6.082753 h; Kaasalainen et al. 2002, Hanuš et al. 2013).

Two practical points this example exists to demonstrate:

**Uncertainties.** DAMIT does not distribute per-point errors, and estimating
them from point-to-point scatter is actively harmful here: many archival curves
were digitised or smoothed from published figures, so consecutive points are
nearly identical and a scatter-based estimator collapses toward zero. Those
curves then carry absurd weight (fractional sigma ~1e-4) and dominate chi-squared.
A uniform, honest fractional uncertainty is used instead.

**Multi-modality.** Convex inversion has many local minima — most notably a
spurious near-spherical solution that matches the mean brightness but no
rotational structure. A single starting pole is unreliable; this uses a grid of
starts and keeps the best fit.

Run:  python example_eunomia_convex.py --n-workers 8
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402

from silhouette.damit import read_damit_lcs  # noqa: E402
from silhouette.inversion import (  # noqa: E402
    LightCurveObs,
    default_pole_grid,
    invert_convex_multistart,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "15_eunomia_damit_lcs.txt")

PERIOD_H = 6.082753          # DAMIT sidereal period
DAMIT_POLE = (3.0, -67.0)
FRAC_SIGMA = 0.02            # assumed uniform photometric uncertainty


def angular_sep(p1, p2):
    l1, b1 = np.radians(p1)
    l2, b2 = np.radians(p2)
    return np.degrees(np.arccos(np.clip(
        np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


def offset_to_damit(lon, lat):
    """Separation from the DAMIT pole, allowing for the mirror ambiguity."""
    return min(angular_sep((lon, lat), DAMIT_POLE),
               angular_sep(((lon + 180.0) % 360.0, -lat), DAMIT_POLE))


def main():
    ap = argparse.ArgumentParser(description="Convex inversion of (15) Eunomia")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
                    help="parallel pole starts (default leaves 4 cores free)")
    ap.add_argument("--lmax", type=int, default=4, help="shape expansion degree")
    ap.add_argument("--n-normals", type=int, default=180)
    args = ap.parse_args()

    curves = read_damit_lcs(DATA)
    dense = sorted([c for c in curves
                    if c.intensity.size >= 8 and c.span_days <= 1.5],
                   key=lambda c: c.epoch_mid)
    lcs = [LightCurveObs(c.jd, c.intensity, FRAC_SIGMA * c.intensity.mean(),
                         c.sun, c.earth, relative=not c.calibrated)
           for c in dense]

    span = (max(c.epoch_mid for c in dense) - min(c.epoch_mid for c in dense)) / 365.25
    print(f"{len(curves)} DAMIT curves -> {len(dense)} dense, "
          f"{sum(len(l) for l in lcs)} points over {span:.0f} yr")

    grid = default_pole_grid(n_lon=6)
    print(f"{len(grid)} starting poles on {args.n_workers} workers ...")

    t0 = time.time()
    res = invert_convex_multistart(
        lcs, period=PERIOD_H / 24.0, pole_grid=grid, n_workers=args.n_workers,
        lmax=args.lmax, n_normals=args.n_normals, max_nfev=250)
    print(f"done in {time.time() - t0:.0f}s\n")

    print(res.summary())
    ab, bc = res.axis_ratios()
    print(f"\nDAMIT model : pole ({DAMIT_POLE[0]:.0f}, {DAMIT_POLE[1]:.0f}) deg, "
          f"P = {PERIOD_H} h")
    print(f"Silhouette  : pole ({res.pole_lon:.1f}, {res.pole_lat:.1f}) deg  ->  "
          f"{offset_to_damit(res.pole_lon, res.pole_lat):.1f} deg from DAMIT "
          f"(mirror allowed)")
    print(f"              a:b = {ab:.3f}, b:c = {bc:.3f}")

    good = [c for c in res.candidates if c[2] < 2.0 * res.redchi2]
    print(f"\nmulti-modality: {len(res.candidates)} starts converged, "
          f"{len(good)} within 2x the best chi^2")
    print("top candidates (lon, lat, chi2_nu, offset from DAMIT):")
    for lon, lat, chi in res.candidates[:6]:
        print(f"   ({lon:6.1f}, {lat:6.1f})   {chi:7.2f}   {offset_to_damit(lon, lat):5.1f} deg")


if __name__ == "__main__":
    main()
