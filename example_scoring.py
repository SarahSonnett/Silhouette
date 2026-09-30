"""Solution-scorer demo on a deliberately ambiguous synthetic asteroid.

Two apparitions of a low-inclination main-belt body: the textbook recipe for
the (λ, β) ↔ (λ + 180°, β) mirror-pole degeneracy, and fewer than the ≥3
apparitions classical inversion asks for. The scorer is run exactly as it would
be on real data:

1. trial periods = the period and its one-rotation aliases (``alias_periods``)
   — on real data the centre would come from SpinDoc;
2. candidate pole families at each period (``find_candidates``);
3. likelihood weights, bootstrap agreement and, if a trained model is present
   (``silhouette/models/scorer_v1.pkl``), calibrated probabilities;
4. the recommender ranks the next three years of epochs by how strongly one
   night of photometry would separate the surviving candidates.

Run:  python example_scoring.py --n-workers 2
Writes docs/images/scoring_demo.png
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402

from silhouette.calibration import (  # noqa: E402
    FAST_INV, calibrated_setup, irregular_shape,
)
from silhouette.forward import convex_lightcurve, ls_lambert  # noqa: E402
from silhouette.inversion import LightCurveObs, _sep_deg  # noqa: E402
from silhouette.planning import (  # noqa: E402
    format_recommendations, predicted_rotation, recommend_observations,
    simple_orbit_geometry,
)
from silhouette.scoring import (  # noqa: E402
    alias_periods, baseline_of, model_flux, score_solutions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE_POLE = (70.0, 25.0)            # Silhouette convention
PERIOD = 7.3 / 24.0                 # days
ORBIT = dict(a_au=2.6, incl_deg=4.0, node_deg=30.0, lon0_deg=0.0)


def make_data(rng):
    truth = irregular_shape(1.6, 1.2, 0.12, rng)
    # two apparitions: nights around oppositions ~1.3 yr apart
    nights = [0.0, 6.0, 15.0, 470.0, 478.0, 490.0]
    sun, earth = simple_orbit_geometry(nights, **ORBIT)
    # shift the epoch origin so day 0 is an opposition
    lcs = []
    for t0, s, e in zip(nights, sun, earth):
        t = t0 + np.sort(rng.uniform(0.0, PERIOD, 40))
        f = convex_lightcurve(truth, t, s, e, *TRUE_POLE, PERIOD, phi0=0.9, t0=0.0,
                              phot_func=ls_lambert, arg=0.15)
        sig = 0.015 * f.mean()
        lcs.append(LightCurveObs(t, f + rng.normal(0, sig, f.size), np.full(f.size, sig),
                                 s, e, relative=True))
    return truth, lcs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    truth, lcs = make_data(rng)
    periods = alias_periods(PERIOD, baseline_of(lcs), n_alias=1)
    grid, model, version = calibrated_setup()
    print(f"{len(lcs)} light curves, {sum(len(l) for l in lcs)} points, 2 apparitions; "
          f"trial periods (h): {np.round(periods * 24, 5)}")
    print("calibration model:", f"{version} ({len(grid)} starts)" if model is not None
          else "not trained yet (uncalibrated)")

    t0 = time.time()
    rep = score_solutions(lcs, periods, pole_grid=grid, n_workers=args.n_workers,
                          n_boot=args.n_boot, boot_max_nfev=30, model=model, **FAST_INV)
    print(f"scored in {time.time() - t0:.0f}s\n")
    print(rep.summary())
    print(f"\ntruth: pole {TRUE_POLE}, P = {PERIOD * 24:.5f} h")
    for i, c in enumerate(rep.candidates):
        sep = _sep_deg(c.pole_lon, c.pole_lat, *TRUE_POLE)
        mir = _sep_deg(c.pole_lon, c.pole_lat, TRUE_POLE[0] + 180.0, TRUE_POLE[1])
        tag = "TRUE" if sep < 20 and abs(c.period - PERIOD) < 1e-9 else (
            "mirror" if mir < 20 else "")
        print(f"  #{i}: {sep:5.1f} deg from truth, {mir:5.1f} deg from its mirror  {tag}")

    # ---- recommender: next three years, every 5 days ------------------------
    future = np.arange(520.0, 520.0 + 3 * 365.25, 5.0)
    fsun, fearth = simple_orbit_geometry(future, **ORBIT)
    top = rep.candidates[:4]
    recs = recommend_observations(top, future, fsun, fearth, n_points=40,
                                  sigma_mag=0.015, model_sigma_mag=rep.noise["model_rms_mag"],
                                  min_elongation_deg=90.0)
    print("\nBest epochs to observe next (days since first night; one night, 40 pts, 0.015 mag):")
    print(format_recommendations(recs, top=6, epoch_fmt=lambda e: f"day {e:.0f}"))

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for i, c in enumerate(rep.candidates):
        ax.scatter(c.pole_lon, c.pole_lat, s=40 + 900 * c.probability, alpha=0.6,
                   color=f"C{i % 10}", edgecolor="k",
                   label=f"#{i}: p={c.probability:.2f}, P={c.period * 24:.4f} h")
    ax.plot(*TRUE_POLE, "k*", ms=16, label="truth")
    ax.plot((TRUE_POLE[0] + 180) % 360, TRUE_POLE[1], "kx", ms=12, mew=2, label="truth mirror")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("pole ecliptic longitude (deg)")
    ax.set_ylabel("pole ecliptic latitude (deg)")
    kind = "calibrated" if rep.calibrated else "uncalibrated"
    ax.set_title(f"Candidate poles, marker area ∝ {kind} probability")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=0.4)

    ax = axes[0, 1]
    lc = lcs[3]
    ph = ((lc.times - lc.times[0]) / PERIOD) % 1.0
    ax.errorbar(ph, lc.flux / lc.flux.mean(), lc.sigma / lc.flux.mean(), fmt="k.", ms=4,
                label="data (apparition 2, night 1)")
    for i, c in enumerate(rep.candidates[:2]):
        m = model_flux(c.fit, lcs)[3]
        o = np.argsort(ph)
        ax.plot(ph[o], m[o] / lc.flux.mean(), color=f"C{i}", lw=2,
                label=f"#{i} model (χ²ν={c.redchi2:.2f})")
    ax.set_xlabel("rotational phase")
    ax.set_ylabel("relative flux")
    ax.set_title("Both top candidates fit the existing data")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rec_by_epoch = {r.epoch: r.p_true_after for r in recs}
    ax.plot(future, [rec_by_epoch.get(e, np.nan) for e in future], "C0-")
    p_now = np.array([c.probability for c in top])
    p_now = p_now / p_now.sum()
    ax.axhline(float(np.sum(p_now ** 2)), color="0.5", ls=":", label="today (no new data)")
    best = recs[0]
    ax.axvline(best.epoch, color="C3", ls="--", label=f"best: day {best.epoch:.0f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("days since first observation")
    ax.set_ylabel("expected P(true solution) after one night")
    ax.set_title("Value of one more night (gaps: elongation < 90°)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    i_best = int(np.argmin(np.abs(future - best.epoch)))
    for i, c in enumerate(top[:2]):
        cv = predicted_rotation(c, fsun[i_best], fearth[i_best])
        ax.plot(np.linspace(0, 1, cv.size, endpoint=False), cv, color=f"C{i}", lw=2,
                label=f"#{i} predicted, amp {np.ptp(cv):.2f} mag")
    ax.invert_yaxis()
    ax.set_xlabel("rotational phase (arbitrary zero)")
    ax.set_ylabel("Δmag")
    ax.set_title(f"Predictions at day {best.epoch:.0f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "docs", "images", "scoring_demo.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
