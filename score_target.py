"""Score competing spin/shape solutions for a real target, then plan the next night.

Squint → SpinDoc → here. Give it the calibrated photometry table (the same
file SpinDoc reads) and SpinDoc's period:

  python score_target.py data/16152_2019_rp.txt --target 16152 --period-h 22.936 \
      --n-workers 4 --plan-start 2027-01-01 --plan-stop 2029-12-31

It splits the table into nights, fetches Sun/Earth vectors from JPL Horizons,
tries the period and its one-rotation aliases (``--harmonics`` adds half/double
periods), scores every candidate pole family, and — with ``--plan-start`` /
``--plan-stop`` — ranks future dates by how much one more night would settle
the question. Probabilities are calibrated when ``silhouette/models/scorer_v1.pkl``
exists (train it with ``calibrate_scorer.py``), otherwise they are labelled
uncalibrated.

Poles are spin angular-momentum directions (DAMIT convention).
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
from silhouette.io import read_photometry  # noqa: E402
from silhouette.pipeline import lightcurves_from_photometry, spindoc_periods  # noqa: E402
from silhouette.planning import (  # noqa: E402
    format_recommendations, horizons_geometry, recommend_observations,
)
from silhouette.scoring import baseline_of, score_solutions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photometry", help="Squint/SpinDoc calibrated photometry table")
    ap.add_argument("--target", required=True, help="JPL Horizons designation")
    ap.add_argument("--period-h", type=float, required=True, help="SpinDoc period (hours)")
    ap.add_argument("--n-alias", type=int, default=1)
    ap.add_argument("--harmonics", type=float, nargs="*", default=[],
                    help="extra period multiples, e.g. 0.5 2")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--min-points", type=int, default=8)
    ap.add_argument("--group", choices=["night", "apparition"], default="night",
                    help="one relative curve per night, or per apparition "
                         "(calibrated photometry; needed for slow rotators)")
    ap.add_argument("--phase-G", type=float, default=None,
                    help="SpinDoc G: divide out the H-G phase function (with --group apparition)")
    ap.add_argument("--plan-start", default=None, help="first date to consider (YYYY-MM-DD)")
    ap.add_argument("--plan-stop", default=None)
    ap.add_argument("--plan-step", default="5d")
    ap.add_argument("--plan-sigma", type=float, default=0.02, help="mag per point next time")
    ap.add_argument("--plan-points", type=int, default=40, help="points in the next night")
    ap.add_argument("--out", default=None, help="JSON summary path")
    args = ap.parse_args()

    phot = read_photometry(args.photometry, object_name=args.target)
    lcs = lightcurves_from_photometry(phot, target=args.target, gap_hours=args.gap_hours,
                                      min_points=args.min_points, group=args.group,
                                      phase_G=args.phase_G)
    periods = spindoc_periods(args.period_h, baseline_of(lcs), n_alias=args.n_alias,
                              harmonics=args.harmonics)
    model = load_default_model()
    print(f"{len(lcs)} light curves ({args.group} grouping), {sum(len(l) for l in lcs)} points, baseline "
          f"{baseline_of(lcs):.1f} d; trial periods (h): {np.round(periods * 24, 5)}")
    print("calibration model:", "loaded" if model is not None else "none (uncalibrated)")

    t0 = time.time()
    rep = score_solutions(lcs, periods, pole_grid=FAST_GRID, n_workers=args.n_workers,
                          n_boot=args.n_boot, boot_max_nfev=30, model=model, **FAST_INV)
    print(f"scored in {time.time() - t0:.0f}s\n")
    print(rep.summary())

    summary = {"target": args.target, "calibrated": rep.calibrated, "noise": rep.noise,
               "data": rep.data, "p_none": rep.p_none,
               "candidates": [{"period_h": c.period * 24,
                               "pole": [c.pole_lon, c.pole_lat],
                               "redchi2": c.redchi2, "ab_proxy": c.elongation,
                               "bc_proxy": c.flattening, "like_weight": c.like_weight,
                               "boot_frac": c.boot_frac, "probability": c.probability}
                              for c in rep.candidates]}

    if args.plan_start and args.plan_stop and len(rep.candidates) > 1:
        jd, sun, earth = horizons_geometry(args.target, args.plan_start, args.plan_stop,
                                           args.plan_step)
        recs = recommend_observations(rep.candidates[:5], jd, sun, earth,
                                      n_points=args.plan_points, sigma_mag=args.plan_sigma,
                                      model_sigma_mag=rep.noise["model_rms_mag"])
        from astropy.time import Time
        fmt = lambda e: Time(e, format="jd").iso[:10]  # noqa: E731
        print(f"\nBest dates for one more night ({args.plan_points} pts at "
              f"{args.plan_sigma} mag, plus {rep.noise['model_rms_mag']:.3f} mag model error):")
        print(format_recommendations(recs, top=10, epoch_fmt=fmt))
        summary["recommendations"] = [{"date": fmt(r.epoch), "p_true_after": r.p_true_after,
                                       "expected_dchi2": r.expected_dchi2,
                                       "phase_deg": r.phase_deg,
                                       "elongation_deg": r.elongation_deg}
                                      for r in recs[:20]]

    out = args.out or os.path.splitext(os.path.basename(args.photometry))[0] + "_scores.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
