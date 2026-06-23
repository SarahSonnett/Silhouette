"""Multi-panel rendering of a Silhouette fit, in the spirit of SpotLight.

The headline figure (:func:`plot_summary`) echoes SpotLight's "combined" layout
— a mosaic of disk-resolved ellipsoid renderings across one rotation on top —
but the lower panels show the *inverse-problem* diagnostics: the amplitude- and
mean-magnitude-versus-aspect relations with the analytical fit overlaid, and a
pole probability map on the ecliptic sky.

When SpotLight is importable the mosaic uses its real triaxial renderer; if not,
a lightweight schematic silhouette is drawn instead so the figure always works.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from ._compat import HAVE_SPOTLIGHT, spotlight
from .fit import SilhouetteFit
from .model import (
    amplitude_model,
    aspect_angle,
    mean_mag_model,
)


# ---------------------------------------------------------------------------
# Ellipsoid rendering (SpotLight when available, schematic otherwise)
# ---------------------------------------------------------------------------

def _spotlight_images(axes, aspect_deg, phase_deg, resolution, n_pixels):
    """Disk-resolved images across one rotation via SpotLight."""
    obs_lat = 90.0 - aspect_deg
    results = spotlight.spotlight_lightcurve(
        axes=list(axes),
        sun_lat_deg=obs_lat,
        sun_dwlong_deg=float(phase_deg),
        obs_lat_deg=obs_lat,
        resolution=resolution,
        n_pixels=n_pixels,
        scaling=True,
    )
    return [(r.obs_wlong_deg, r.img) for r in results]


def _schematic_silhouette(ax, axes, aspect_deg, phase_deg):
    """Draw an approximate projected-ellipse silhouette (SpotLight fallback)."""
    a, b, c = axes
    th = np.radians(aspect_deg)
    ph = np.radians(phase_deg)
    width = 2.0 * np.sqrt((a * np.sin(ph)) ** 2 + (b * np.cos(ph)) ** 2)
    in_plane = np.sqrt((a * np.cos(ph)) ** 2 + (b * np.sin(ph)) ** 2)
    height = 2.0 * np.sqrt((in_plane * np.cos(th)) ** 2 + (c * np.sin(th)) ** 2)
    e = Ellipse((0, 0), width, height, facecolor="0.7", edgecolor="0.2", lw=0.8)
    ax.add_patch(e)
    lim = max(a, b) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_facecolor("black")


def plot_model_mosaic(
    fit: SilhouetteFit,
    aspect_deg: Optional[float] = None,
    phase_deg: float = 0.0,
    n_cols: int = 6,
    resolution: int = 18,
    n_pixels: int = 96,
) -> plt.Figure:
    """Mosaic of the best-fit ellipsoid through one rotation.

    Defaults to the median apparition aspect when ``aspect_deg`` is not given.
    """
    a, b, c = fit.axes
    if not np.isfinite(c):
        c = b  # single-apparition: c undetermined, show a prolate proxy
    axes = (a, b, c)

    if aspect_deg is None and fit.apparitions and fit.pole_lon is not None:
        th = np.degrees(aspect_angle(
            [ap.ecl_lon for ap in fit.apparitions],
            [ap.ecl_lat for ap in fit.apparitions],
            fit.pole_lon, fit.pole_lat))
        aspect_deg = float(np.median(th))
    if aspect_deg is None:
        aspect_deg = 90.0

    n_rows = (resolution + n_cols - 1) // n_cols
    fig, axarr = plt.subplots(n_rows, n_cols,
                              figsize=(2.3 * n_cols, 2.3 * n_rows), squeeze=False)

    if HAVE_SPOTLIGHT:
        frames = _spotlight_images(axes, aspect_deg, phase_deg, resolution, n_pixels)
        vmax = max(img.max() for _, img in frames) or 1.0
        for i, (wlong, img) in enumerate(frames):
            ax = axarr[i // n_cols][i % n_cols]
            ax.imshow(np.clip(img / vmax, 0, 1), cmap="gray", vmin=0, vmax=1,
                      origin="lower", interpolation="nearest")
            ax.set_title(f"{wlong:.0f}°", fontsize=6)
            ax.axis("off")
        n = len(frames)
    else:
        for i in range(resolution):
            ax = axarr[i // n_cols][i % n_cols]
            _schematic_silhouette(ax, axes, aspect_deg, i * 360.0 / resolution)
            ax.set_title(f"{i * 360.0 / resolution:.0f}°", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        n = resolution

    for i in range(n, n_rows * n_cols):
        axarr[i // n_cols][i % n_cols].axis("off")

    fig.suptitle(f"Best-fit ellipsoid  a:b:c = {a:.2f}:{b:.2f}:{c:.2f}"
                 f"   (aspect {aspect_deg:.0f}°)", fontsize=9)
    fig.tight_layout(pad=0.3)
    return fig


# ---------------------------------------------------------------------------
# Aspect-relation diagnostics
# ---------------------------------------------------------------------------

def plot_aspect_curves(fit: SilhouetteFit, ax_amp=None, ax_mag=None):
    """Amplitude- and mean-magnitude-versus-aspect, data + analytical model."""
    apps = fit.apparitions
    th = np.degrees(aspect_angle(
        [a.ecl_lon for a in apps], [a.ecl_lat for a in apps],
        fit.pole_lon, fit.pole_lat))
    amps = np.array([a.amplitude for a in apps])
    samps = np.array([a.amplitude_err for a in apps])
    means = np.array([a.mean_mag for a in apps])
    smeans = np.array([a.mean_mag_err for a in apps])

    grid = np.linspace(0, 180, 361)
    gth = np.radians(grid)

    if ax_amp is None or ax_mag is None:
        fig, (ax_amp, ax_mag) = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig = ax_amp.get_figure()

    ax_amp.errorbar(th, amps, yerr=samps, fmt="o", color="steelblue", capsize=3)
    ax_amp.plot(grid, amplitude_model(fit.ab, fit.bc, gth), "-", color="firebrick",
                label=f"a:b={fit.ab:.2f}, b:c={fit.bc:.2f}")
    ax_amp.set_xlabel("Aspect angle (deg)")
    ax_amp.set_ylabel("Amplitude (mag)")
    ax_amp.set_title("Amplitude–aspect")
    ax_amp.legend(fontsize=8)

    ax_mag.errorbar(th, means, yerr=smeans, fmt="o", color="seagreen", capsize=3)
    ax_mag.plot(grid, mean_mag_model(fit.ab, fit.bc, gth, fit.zero_point), "-",
                color="firebrick")
    ax_mag.set_xlabel("Aspect angle (deg)")
    ax_mag.set_ylabel("Mean reduced mag")
    ax_mag.set_ylim(ax_mag.get_ylim()[::-1])   # mag increases downward
    ax_mag.set_title("Mean magnitude–aspect")
    return fig


# ---------------------------------------------------------------------------
# Pole sky map
# ---------------------------------------------------------------------------

def plot_pole_map(fit: SilhouetteFit, ax=None, n_grid: int = 121):
    """chi^2 map over ecliptic (lon, lat) with the best + mirror poles marked."""
    apps = fit.apparitions
    lon = np.array([a.ecl_lon for a in apps])
    lat = np.array([a.ecl_lat for a in apps])
    amps = np.array([a.amplitude for a in apps])
    samps = np.array([a.amplitude_err for a in apps])
    means = np.array([a.mean_mag for a in apps])
    smeans = np.array([a.mean_mag_err for a in apps])

    L = np.linspace(0, 360, n_grid)
    B = np.linspace(-90, 90, n_grid // 2 + 1)
    chi = np.empty((B.size, L.size))
    for i, bb in enumerate(B):
        for j, ll in enumerate(L):
            th = aspect_angle(lon, lat, ll, bb)
            r = (amplitude_model(fit.ab, fit.bc, th) - amps) / samps
            rm = (mean_mag_model(fit.ab, fit.bc, th, fit.zero_point) - means) / smeans
            chi[i, j] = np.sum(r ** 2) + np.sum(rm ** 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    im = ax.pcolormesh(L, B, np.log10(chi), cmap="viridis_r", shading="auto")
    fig.colorbar(im, ax=ax, label="log10 chi^2")
    if fit.pole_lon is not None:
        ax.plot(fit.pole_lon, fit.pole_lat, "*", color="white", ms=16,
                markeredgecolor="k", label="best pole")
        if fit.mirror_pole is not None:
            ax.plot(fit.mirror_pole[0], fit.mirror_pole[1], "P", color="orange",
                    ms=11, markeredgecolor="k", label="mirror")
    ax.set_xlabel("Ecliptic longitude (deg)")
    ax.set_ylabel("Ecliptic latitude (deg)")
    ax.set_title("Pole solution map")
    ax.legend(fontsize=8, loc="lower right")
    return fig


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def plot_summary(fit: SilhouetteFit, resolution: int = 18, n_pixels: int = 96) -> plt.Figure:
    """One figure: ellipsoid mosaic on top, aspect curves + pole map below."""
    if fit.degenerate and fit.pole_lon is None:
        # Single apparition: mosaic + an a/b lower-bound annotation only.
        fig = plot_model_mosaic(fit, resolution=resolution, n_pixels=n_pixels)
        fig.text(0.5, 0.02,
                 f"Single apparition: a/b ≥ {fit.ab:.2f} (lower bound); "
                 f"pole and b/c undetermined",
                 ha="center", fontsize=10, color="firebrick")
        return fig

    a, b, c = fit.axes
    n_cols = 6
    n_img_rows = (resolution + n_cols - 1) // n_cols

    th_med = float(np.median(np.degrees(aspect_angle(
        [ap.ecl_lon for ap in fit.apparitions],
        [ap.ecl_lat for ap in fit.apparitions], fit.pole_lon, fit.pole_lat))))
    axes = (a, b, c if np.isfinite(c) else b)
    if HAVE_SPOTLIGHT:
        frames = _spotlight_images(axes, th_med, 0.0, resolution, n_pixels)
        vmax = max(img.max() for _, img in frames) or 1.0

    # Mosaic cells are sized ~square (width 14/n_cols), and the lower diagnostic
    # panels each get a generous fixed height.
    fig = plt.figure(figsize=(14, 2.4 * n_img_rows + 9))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[2.4 * n_img_rows, 4.2, 4.2],
                           hspace=0.4, wspace=0.25)
    mos = gridspec.GridSpecFromSubplotSpec(n_img_rows, n_cols,
                                           subplot_spec=gs[0, :], wspace=0.08, hspace=0.35)
    if HAVE_SPOTLIGHT:
        for i, (wlong, img) in enumerate(frames):
            ax = fig.add_subplot(mos[i // n_cols, i % n_cols])
            ax.imshow(np.clip(img / vmax, 0, 1), cmap="gray", vmin=0, vmax=1,
                      origin="lower", interpolation="nearest")
            ax.set_title(f"{wlong:.0f}°", fontsize=8)
            ax.axis("off")
    else:
        for i in range(resolution):
            ax = fig.add_subplot(mos[i // n_cols, i % n_cols])
            _schematic_silhouette(ax, axes, th_med, i * 360.0 / resolution)
            ax.set_title(f"{i * 360.0 / resolution:.0f}°", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    ax_amp = fig.add_subplot(gs[1, 0])
    ax_mag = fig.add_subplot(gs[1, 1])
    plot_aspect_curves(fit, ax_amp=ax_amp, ax_mag=ax_mag)

    ax_pole = fig.add_subplot(gs[2, :])
    plot_pole_map(fit, ax=ax_pole)

    fig.suptitle(
        f"Silhouette  —  a:b:c = {a:.2f}:{b:.2f}:{axes[2]:.2f}   "
        f"pole ({fit.pole_lon:.0f}°, {fit.pole_lat:.0f}°)   "
        f"χ²ᵥ={fit.redchi2:.2f}   N={fit.n_apparitions}",
        fontsize=11, y=0.99)
    return fig


def save_summary(fit: SilhouetteFit, path: str, dpi: int = 150, **kwargs) -> None:
    """Render and save the combined summary figure."""
    fig = plot_summary(fit, **kwargs)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_model_mosaic",
    "plot_aspect_curves",
    "plot_pole_map",
    "plot_summary",
    "save_summary",
]
