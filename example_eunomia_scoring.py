"""Solution scorer on real DAMIT photometry of (15) Eunomia.

The Phase-1a example found two pole families at the same chi2_nu (0.94): one
3° from DAMIT and one 25° away, i.e. convex inversion alone could not choose.
This runs the scorer on the same data with the calibrated settings and asks
what probability it gives each family, and whether the DAMIT solution is the
one it favours.

Poles are spin angular-momentum directions (DAMIT convention since
2026-09-30), so they compare directly with DAMIT's (λ ≈ 3°, β ≈ −67°).

Two subsets are scored: all dense curves (the classical "enough data" case)
and the first two apparitions only (the case where a probability is most
useful).

Run:  python example_eunomia_scoring.py --n-workers 2
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from silhouette.calibration import FAST_GRID, FAST_INV, load_default_model  # noqa: E402
from silhouette.damit import read_damit_lcs  # noqa: E402
from silhouette.inversion import LightCurveObs, _sep_deg  # noqa: E402
from silhouette.scoring import (  # noqa: E402
    alias_periods, baseline_of, score_solutions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "15_eunomia_damit_lcs.txt")
PERIOD_H = 6.082753
DAMIT_POLE = (3.0, -67.0)
FRAC_SIGMA = 0.02            # uniform; never estimate from DAMIT point scatter


def load(max_apparitions=None, app_gap_days=60.0):
    curves = read_damit_lcs(DATA)
    dense = sorted([c for c in curves if c.intensity.size >= 8 and c.span_days <= 1.5],
                   key=lambda c: c.epoch_mid)
    if max_apparitions is not None:
        keep, n_app, last = [], 0, None
        for c in dense:
            if last is None or c.epoch_mid - last > app_gap_days:
                n_app += 1
            if n_app > max_apparitions:
                break
            keep.append(c)
            last = c.epoch_mid
        dense = keep
    return [LightCurveObs(c.jd, c.intensity, FRAC_SIGMA * c.intensity.mean(),
                          c.sun, c.earth, relative=not c.calibrated) for c in dense]


def run(label, lcs, args, model):
    periods = alias_periods(PERIOD_H / 24.0, baseline_of(lcs), n_alias=1)
    print(f"\n=== {label}: {len(lcs)} curves, {sum(len(l) for l in lcs)} points, "
          f"baseline {baseline_of(lcs) / 365.25:.1f} yr ===")
    t0 = time.time()
    rep = score_solutions(lcs, periods, pole_grid=FAST_GRID, n_workers=args.n_workers,
                          n_boot=args.n_boot, boot_max_nfev=30, model=model, **FAST_INV)
    print(f"scored in {time.time() - t0:.0f}s")
    print(rep.summary())
    rows = []
    for i, c in enumerate(rep.candidates):
        sv = (c.pole_lon, c.pole_lat)
        d = _sep_deg(*sv, *DAMIT_POLE)
        d_mirror = _sep_deg(*sv, DAMIT_POLE[0] + 180.0, DAMIT_POLE[1])
        on_p = abs(c.period * 24.0 - PERIOD_H) < 1e-6
        print(f"  #{i}: pole ({sv[0]:6.1f},{sv[1]:6.1f})  {d:5.1f} deg from DAMIT, "
              f"{d_mirror:5.1f} from DAMIT's mirror, DAMIT period: {on_p}  p={c.probability:.3f}")
        rows.append({"period_h": c.period * 24, "pole": list(sv), "sep_damit": d, "sep_damit_mirror": d_mirror,
                     "redchi2": c.redchi2, "like_weight": c.like_weight,
                     "boot_frac": c.boot_frac, "probability": c.probability})
    return {"label": label, "calibrated": rep.calibrated, "noise": rep.noise,
            "data": rep.data, "p_none": rep.p_none, "candidates": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=12)
    ap.add_argument("--subset", choices=["all", "two", "both"], default="both")
    args = ap.parse_args()
    model = load_default_model()
    print("calibration model:", "loaded" if model is not None else "not trained (uncalibrated)")
    out = []
    if args.subset in ("two", "both"):
        out.append(run("first two apparitions", load(max_apparitions=2), args, model))
    if args.subset in ("all", "both"):
        out.append(run("all dense curves", load(), args, model))
    os.makedirs(os.path.join(HERE, "results", "scorer"), exist_ok=True)
    path = os.path.join(HERE, "results", "scorer",
                        f"eunomia_scoring_{'cal' if model is not None else 'uncal'}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
