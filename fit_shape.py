"""Command-line driver for an analytical Silhouette shape + pole fit.

Usage
-----
    python fit_shape.py --infile photometry.txt --period 0.2194 \
        --object 433 --outdir results

Reads a tabular photometry file, groups it into apparitions, resolves the
ecliptic geometry (from file columns or JPL Horizons), fits a:b, b:c and the
rotation pole, then writes a parameter summary and the multi-panel figure.
"""

from __future__ import annotations

import argparse
import os

from silhouette import (
    read_photometry,
    reduce_apparitions,
    resolve_geometry,
    fit_shape,
    save_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analytical asteroid shape + pole fit")
    p.add_argument("--infile", required=True, help="tabular photometry file")
    p.add_argument("--period", type=float, required=True,
                   help="rotation period (days)")
    p.add_argument("--object", default=None,
                   help="Horizons designation (used if file lacks ecliptic columns)")
    p.add_argument("--outdir", default="results")
    p.add_argument("--gap-days", type=float, default=60.0,
                   help="minimum epoch gap separating apparitions")
    p.add_argument("--order", type=int, default=2, help="Fourier order for amplitude")
    p.add_argument("--G", type=float, default=0.15, help="IAU phase-slope parameter")
    p.add_argument("--no-meanmag", action="store_true",
                   help="fit amplitude-aspect only (skip mean-magnitude method)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    phot = read_photometry(args.infile, object_name=args.object)
    apps = reduce_apparitions(phot, period=args.period,
                              gap_days=args.gap_days, fourier_order=args.order,
                              G=args.G)
    print(f"Grouped {len(phot)} points into {len(apps)} apparition(s).")

    resolve_geometry(apps, target=args.object)
    fit = fit_shape(apps, use_meanmag=not args.no_meanmag)

    print(fit.summary())

    txt = os.path.join(args.outdir, "BestFitParameters.txt")
    with open(txt, "w") as f:
        f.write(fit.summary() + "\n\n")
        f.write("idx  epoch_mid  N  ecl_lon  ecl_lat  amplitude  amp_err  "
                "mean_mag  mean_mag_err  geom\n")
        for a in fit.apparitions:
            f.write(f"{a.index}  {a.epoch_mid:.4f}  {a.n_points}  "
                    f"{a.ecl_lon}  {a.ecl_lat}  {a.amplitude:.4f}  "
                    f"{a.amplitude_err:.4f}  {a.mean_mag:.4f}  "
                    f"{a.mean_mag_err:.4f}  {a.geom_source}\n")

    png = os.path.join(args.outdir, "fit_summary.png")
    save_summary(fit, png)
    print(f"\nWrote {txt} and {png}")


if __name__ == "__main__":
    main()
