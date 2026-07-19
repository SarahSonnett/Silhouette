"""Figures for the rotational-stability constraints in :mod:`silhouette.geophysics`.

Three views of the same physics:

1. **Cohesion vs density** — how much cohesion the body needs as a function of
   assumed bulk density, with the cohesionless threshold marked. Needs a size.
2. **Spin-barrier context** — the period/density plane with the cohesionless
   stability boundary for this shape, and the classic strengthless-sphere
   barrier for reference.
3. **Shape sensitivity** — how the minimum cohesionless density moves as the
   axis ratios vary within their (large) uncertainties, since ``b/c`` from
   light-curve inversion is typically good only to ~20%.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geophysics import (
    min_density_cohesionless,
    required_cohesion,
    shedding_limit_density,
)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})


def plot_cohesion_vs_density(ab, bc, period_h, diameter_km,
                             friction_degs: Sequence[float] = (25.0, 35.0, 45.0),
                             rho_range: Tuple[float, float] = (500.0, 5000.0),
                             ax=None):
    """Required cohesion as a function of assumed bulk density."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))
    rhos = np.linspace(*rho_range, 200)
    for phi in friction_degs:
        y = [required_cohesion(ab, bc, period_h, r, diameter_km, phi) for r in rhos]
        ax.plot(rhos, y, lw=2, label=f"$\\phi$ = {phi:.0f}°")
        rmin = min_density_cohesionless(ab, bc, period_h, phi)
        if rmin is not None and rho_range[0] < rmin < rho_range[1]:
            ax.axvline(rmin, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("assumed bulk density (kg m$^{-3}$)")
    ax.set_ylabel("required cohesion (Pa)")
    ax.set_title(f"Cohesion needed at P = {period_h:.3f} h, D = {diameter_km:g} km\n"
                 f"a:b = {ab:.2f}, b:c = {bc:.2f}", fontsize=13)
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=0.0)          # cohesion cannot be negative
    ax.grid(alpha=0.3)
    ax.legend(title="friction angle")
    ax.text(0.98, 0.94, "zero ⇒ cohesionless\nrubble pile suffices",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, color="0.35")
    return ax.get_figure()


def plot_spin_barrier(ab, bc, friction_deg: float = 35.0,
                      period_range: Tuple[float, float] = (1.5, 12.0),
                      mark: Optional[Tuple[float, float]] = None, ax=None):
    """Minimum cohesionless density vs rotation period, with the classic barrier."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))
    periods = np.linspace(*period_range, 160)

    shape_curve = [min_density_cohesionless(ab, bc, p, friction_deg) for p in periods]
    shape_curve = [np.nan if v is None else v for v in shape_curve]
    sphere_curve = [min_density_cohesionless(1.0, 1.0, p, friction_deg) for p in periods]
    sphere_curve = [np.nan if v is None else v for v in sphere_curve]

    ax.plot(periods, shape_curve, lw=2.5, color="firebrick",
            label=f"this shape (a:b={ab:.2f}, b:c={bc:.2f})")
    ax.plot(periods, sphere_curve, lw=1.8, ls="--", color="steelblue",
            label="sphere, same friction")
    ax.plot(periods, [shedding_limit_density(p) for p in periods], lw=1.5, ls=":",
            color="0.35", label="strengthless sphere (shedding)")

    ax.fill_between(periods, shape_curve, 1e5, alpha=0.12, color="firebrick")
    ax.text(0.97, 0.90, "cohesionless\nrubble pile OK", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="firebrick")
    ax.text(0.05, 0.10, "needs cohesion\nor monolithic", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, color="0.3")

    if mark is not None:
        ax.plot(*mark, "*", ms=20, color="gold", markeredgecolor="k",
                zorder=5, label="this object")
    ax.set_xlabel("rotation period (h)")
    ax.set_ylabel("minimum cohesionless density (kg m$^{-3}$)")
    ax.set_yscale("log")
    ax.set_ylim(50, 2e4)
    ax.set_title(f"Rotational stability limit ($\\phi$ = {friction_deg:.0f}°)", fontsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10)
    return ax.get_figure()


def plot_shape_sensitivity(ab, bc, period_h, friction_deg: float = 35.0,
                           ab_frac: float = 0.05, bc_frac: float = 0.20, ax=None):
    """Minimum cohesionless density across the axis-ratio uncertainty box.

    Light-curve inversion constrains ``b/c`` far more weakly than ``a/b``, so the
    spread here is dominated by the vertical axis. This is the honest error bar
    on any density or strength statement.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))
    abs_ = np.linspace(ab * (1 - ab_frac), ab * (1 + ab_frac), 40)
    bcs = np.linspace(max(bc * (1 - bc_frac), 1.0), bc * (1 + bc_frac), 40)
    grid = np.empty((bcs.size, abs_.size))
    for i, y in enumerate(bcs):
        for j, x in enumerate(abs_):
            v = min_density_cohesionless(max(x, 1.0), max(y, 1.0), period_h, friction_deg)
            grid[i, j] = np.nan if v is None else v

    im = ax.pcolormesh(abs_, bcs, grid, cmap="magma_r", shading="auto")
    ax.get_figure().colorbar(im, ax=ax, label="min cohesionless density (kg m$^{-3}$)")
    ax.plot(ab, bc, "*", ms=18, color="cyan", markeredgecolor="k", label="best fit")
    ax.set_xlabel("a : b")
    ax.set_ylabel("b : c")
    ax.set_title(f"Sensitivity to axis ratios (P = {period_h:.3f} h)\n"
                 f"a:b ±{ab_frac*100:.0f}%, b:c ±{bc_frac*100:.0f}%", fontsize=13)
    ax.legend()
    return ax.get_figure()


def plot_strength_summary(ab, bc, period_h, diameter_km,
                          friction_deg: float = 35.0, path: Optional[str] = None):
    """Combined three-panel strength/density figure."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    plot_cohesion_vs_density(ab, bc, period_h, diameter_km, ax=axes[0])
    rho_min = min_density_cohesionless(ab, bc, period_h, friction_deg)
    plot_spin_barrier(ab, bc, friction_deg,
                      mark=(period_h, rho_min) if rho_min else None, ax=axes[1])
    plot_shape_sensitivity(ab, bc, period_h, friction_deg, ax=axes[2])
    fig.suptitle("Silhouette: what must this body be made of to survive its spin?",
                 fontsize=16)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig


__all__ = ["plot_cohesion_vs_density", "plot_spin_barrier",
           "plot_shape_sensitivity", "plot_strength_summary"]
