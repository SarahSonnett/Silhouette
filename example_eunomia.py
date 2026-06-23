"""Multi-apparition example on real DAMIT photometry of asteroid (15) Eunomia.

Uses the 109 light curves DAMIT collected for (15) Eunomia
(`data/15_eunomia_damit_lcs.txt`, id 108), spanning many apparitions over
decades. Each light curve carries the asteroid-centric Sun/Earth ecliptic
vectors, so the aspect geometry comes straight from the file — no ephemeris
lookup. The photometry is relative, so only the amplitude–aspect observable is
used (`use_meanmag=False`).

The recovered pole is compared with the DAMIT convex-inversion solution
(λ ≈ 3°, β ≈ −67°, P = 6.083 h; Kaasalainen et al. 2002, refined by
Hanuš et al. 2013), overplotted on the pole map.

Run:  python example_eunomia.py
"""

from __future__ import annotations

import os

import numpy as np

from silhouette import read_damit_lcs, damit_apparitions, fit_shape, save_summary
from silhouette.model import aspect_angle

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "15_eunomia_damit_lcs.txt")
PERIOD_H = 6.082753                 # DAMIT sidereal period (hours)
DAMIT_POLE = (3.0, -67.0)           # DAMIT pole (ecliptic lon, lat), degrees


def _angular_sep(p1, p2):
    l1, b1 = np.radians(p1); l2, b2 = np.radians(p2)
    return np.degrees(np.arccos(np.clip(
        np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


def main():
    curves = read_damit_lcs(DATA)
    apps = damit_apparitions(curves, period=PERIOD_H / 24.0)
    print(f"Read {len(curves)} DAMIT light curves -> {len(apps)} apparitions.")

    fit = fit_shape(apps, use_meanmag=False)
    print()
    print(fit.summary())

    # Compare to DAMIT (account for the prograde/retrograde mirror).
    best = fit.pole_lon, fit.pole_lat
    sep = min(_angular_sep(best, DAMIT_POLE), _angular_sep(fit.mirror_pole, DAMIT_POLE))
    print(f"\nDAMIT pole: λ={DAMIT_POLE[0]:.0f}°, β={DAMIT_POLE[1]:.0f}°  "
          f"(P={PERIOD_H:.3f} h)")
    print(f"closest Silhouette pole is {sep:.0f}° from DAMIT; "
          f"recovered axis ratios a:b={fit.ab:.2f}, b:c={fit.bc:.2f}")

    out = os.path.join(HERE, "docs", "images", "eunomia_fit.png")
    save_summary(fit, out, reference_pole=DAMIT_POLE)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
