"""Worked example on real photometry of asteroid (16152) 1999 YN12.

Uses the single-apparition r-band light curve in ``data/16152_2019_rp.txt``
(2019 Aug–Sep, ~430 points). Because this is one apparition, Silhouette can only
constrain an a/b LOWER BOUND from the rotation amplitude — the pole and b/c
require photometry from several apparitions at differing ecliptic geometry.

For reference, the DAMIT convex models of (16152) give a spin pole near
(λ, β) ≈ (115°, 63°) or (305°, 68°) with a sidereal period of 22.936 h, derived
from many apparitions of combined dense and sparse photometry
(https://damit.cuni.cz/projects/damit/?q=16152).

Run:  python example_16152.py
"""

from __future__ import annotations

import os

from silhouette import read_photometry, reduce_apparitions, fit_shape, save_summary

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "16152_2019_rp.txt")
PERIOD_H = 22.936          # DAMIT sidereal period (hours)


def main():
    phot = read_photometry(DATA, columns={"merr": "TmagFinalErr"}, object_name="16152")
    print(f"Loaded {len(phot)} points; ecliptic columns present: {phot.has_ecliptic}")

    apps = reduce_apparitions(phot, period=PERIOD_H / 24.0)
    print(f"Grouped into {len(apps)} apparition(s).")
    for a in apps:
        print(f"  span {a.span_days:.1f} d, amplitude {a.amplitude:.3f} "
              f"± {a.amplitude_err:.3f} mag")

    fit = fit_shape(apps)
    print()
    print(fit.summary())

    out = os.path.join(HERE, "docs", "images", "16152_fit.png")
    save_summary(fit, out)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
