"""How well must a light curve be sampled and measured to recover a shape?

Maps the joint effect of **rotational sampling** (points per rotation) and
**photometric precision** (fractional uncertainty) on the accuracy of a convex
inversion, using synthetic data with a known ground-truth shape and pole.

The two are expected to trade off against each other: coarse sampling can be
partly compensated by better photometry and vice versa, because what ultimately
matters is the total information content of the light curve — roughly, how well
the rotational Fourier structure is determined. This study measures that
trade-off rather than assuming it.

Each grid cell runs several noise realisations and reports the median error in
the DEEVE axis ratios and the pole direction.

Parallelism is bounded by ``--n-workers`` (default leaves several cores free),
and BLAS threading is pinned to one thread per worker so the job cannot
oversubscribe the machine.

Run:  python study_resolution_precision.py --n-workers 8
"""

from __future__ import annotations

import argparse
import os

# Pin BLAS threads before NumPy is imported; children inherit this.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

from silhouette.forward import convex_lightcurve, ls_lambert  # noqa: E402
from silhouette.inversion import (  # noqa: E402
    LightCurveObs, invert_convex_multistart, default_pole_grid,
)
from silhouette.shapes import ConvexShape  # noqa: E402

# ---- ground truth ----------------------------------------------------------
TRUE_AXES = (2.0, 1.4, 1.0)
TRUE_POLE = (60.0, 35.0)
PERIOD = 0.25
N_APPARITIONS = 6                 # held fixed to isolate sampling vs precision

# Pole multistart per cell. This must be dense: the basin containing the true
# solution is narrow, and a coarse grid simply never samples it (chi^2 ranks the
# true minimum correctly, so the failure is coverage, not discrimination). The
# grid is deliberately truth-agnostic -- seeding near the answer would flatter
# the results.
SEED_POLES = default_pole_grid(n_lon=5, lats=(-70.0, -35.0, 0.0, 35.0, 70.0))

SAMPLING = [6, 10, 15, 25, 40, 60]                    # points per rotation
PRECISION = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]    # fractional sigma

OUT_PNG = os.path.join(os.path.dirname(__file__), "docs", "images",
                       "resolution_precision.png")


def _dir(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _angular_sep(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    return np.degrees(np.arccos(np.clip(
        np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


def run_cell(args):
    """One (sampling, precision, realisation) trial -> error metrics."""
    n_pts, frac, seed = args
    rng = np.random.default_rng(seed)
    truth = ConvexShape.from_ellipsoid(*TRUE_AXES, 250)

    lcs = []
    for k, lon in enumerate(np.linspace(0, 300, N_APPARITIONS)):
        earth = _dir(lon + rng.uniform(-8, 8), rng.uniform(-6, 6))
        sun = _dir(lon + rng.uniform(3, 15), rng.uniform(-6, 6))
        t = np.sort(rng.uniform(0, PERIOD, n_pts)) + k * 200.0
        flux = convex_lightcurve(truth, t, sun, earth, *TRUE_POLE, PERIOD,
                                 phi0=0.7, t0=0.0, phot_func=ls_lambert)
        sigma = frac * flux.mean()
        lcs.append(LightCurveObs(t, flux + rng.normal(0, sigma, flux.size),
                                 np.full(flux.size, sigma), sun, earth, relative=True))

    # A small pole multistart, run serially inside the cell (parallelism lives at
    # the grid level). Without this, occasional convergence to the spurious
    # near-spherical minimum dominates the statistics and measures optimiser luck
    # rather than the information content of the data.
    try:
        res = invert_convex_multistart(
            lcs, period=PERIOD, pole_grid=SEED_POLES, n_workers=1,
            lmax=3, n_normals=120, max_nfev=200)
        ab, bc = res.axis_ratios()
    except Exception:
        return n_pts, frac, np.nan, np.nan, np.nan

    a, b, c = TRUE_AXES
    return (n_pts, frac,
            abs(ab - a / b) / (a / b) * 100.0,
            abs(bc - b / c) / (b / c) * 100.0,
            min(_angular_sep(res.pole_lon, res.pole_lat, *TRUE_POLE),
                _angular_sep((res.pole_lon + 180) % 360, -res.pole_lat, *TRUE_POLE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 4),
                    help="parallel workers (default leaves 4 cores free)")
    ap.add_argument("--n-real", type=int, default=5, help="noise realisations per cell")
    ap.add_argument("--quick", action="store_true", help="coarse 3x3 grid, 2 realisations")
    args = ap.parse_args()

    sampling, precision, n_real = SAMPLING, PRECISION, args.n_real
    if args.quick:
        sampling, precision, n_real = [10, 25, 60], [0.005, 0.02, 0.10], 2

    jobs = [(n, f, 1000 * i + 17 * j + r)
            for i, n in enumerate(sampling)
            for j, f in enumerate(precision)
            for r in range(n_real)]
    print(f"{len(jobs)} inversions on {args.n_workers} workers "
          f"({len(sampling)}x{len(precision)} grid, {n_real} realisations)")

    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futs = {pool.submit(run_cell, j): j for j in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)}")

    # ---- aggregate (median over realisations) ----
    grid_ab = np.full((len(sampling), len(precision)), np.nan)
    grid_bc = np.full_like(grid_ab, np.nan)
    grid_pole = np.full_like(grid_ab, np.nan)
    for i, n in enumerate(sampling):
        for j, f in enumerate(precision):
            cell = [r for r in results if r[0] == n and r[1] == f]
            if cell:
                grid_ab[i, j] = np.nanmedian([c[2] for c in cell])
                grid_bc[i, j] = np.nanmedian([c[3] for c in cell])
                grid_pole[i, j] = np.nanmedian([c[4] for c in cell])

    print("\nmedian |a/b| error (%)   rows = pts/rotation, cols = sigma")
    print("        " + "".join(f"{f*100:8.1f}%" for f in precision))
    for i, n in enumerate(sampling):
        print(f"  {n:4d}  " + "".join(f"{v:8.1f}" for v in grid_ab[i]))
    print("\nmedian pole error (deg)")
    print("        " + "".join(f"{f*100:8.1f}%" for f in precision))
    for i, n in enumerate(sampling):
        print(f"  {n:4d}  " + "".join(f"{v:8.1f}" for v in grid_pole[i]))

    _plot(sampling, precision, grid_ab, grid_bc, grid_pole)
    print(f"\nSaved {OUT_PNG}")


def _plot(sampling, precision, grid_ab, grid_bc, grid_pole):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    panels = [(grid_ab, "median |a:b| error (%)"),
              (grid_bc, "median |b:c| error (%)"),
              (grid_pole, "median pole error (deg)")]
    for ax, (grid, title) in zip(axes, panels):
        finite = grid[np.isfinite(grid) & (grid > 0)]
        norm = LogNorm(vmin=max(finite.min(), 1e-2), vmax=finite.max()) if finite.size else None
        im = ax.imshow(grid, origin="lower", cmap="viridis_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(precision)))
        ax.set_xticklabels([f"{f*100:g}%" for f in precision])
        ax.set_yticks(range(len(sampling)))
        ax.set_yticklabels(sampling)
        ax.set_xlabel("photometric precision (fractional $\\sigma$)")
        ax.set_ylabel("points per rotation")
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                            fontsize=11, color="w")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Silhouette convex inversion: shape accuracy vs sampling and photometric precision\n"
                 f"truth a:b:c = {TRUE_AXES[0]}:{TRUE_AXES[1]}:{TRUE_AXES[2]}, "
                 f"pole = {TRUE_POLE}, {N_APPARITIONS} apparitions", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
