"""Group photometry into apparitions and reduce each to fit observables.

For each apparition (a cluster of epochs separated from its neighbours by a long
gap) Silhouette derives the two analytical observables used by the shape fit:

* **amplitude** ``A`` — peak-to-trough range of the rotation light curve, found
  by phasing at the rotation period and fitting a Fourier series
  (``spindoc.fourier``).
* **mean reduced magnitude** ``M`` — the apparition's average magnitude after
  reduction to unit heliocentric/geocentric distance and H-G phase correction.
  Its variation between apparitions encodes the changing projected area
  (the "magnitude method").

Uncertainties on both come from a light bootstrap so they feed naturally into
the weighted least-squares fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import curve_fit

from ._compat import HGfunction, fourier
from .io import Photometry


@dataclass
class Apparition:
    """One apparition's reduced light-curve observables."""

    index: int
    epoch_mid: float                 # MJD at apparition midpoint
    n_points: int
    span_days: float
    amplitude: float                 # mag, peak-to-trough
    amplitude_err: float
    mean_mag: float                  # reduced + phase-corrected, mag
    mean_mag_err: float
    alpha_mean: float                # mean solar phase angle, deg
    rhelio_mean: float
    delta_mean: float
    # Geometry (filled in by geometry.resolve_geometry)
    ecl_lon: Optional[float] = None  # observer-centric ecliptic longitude, deg
    ecl_lat: Optional[float] = None  # observer-centric ecliptic latitude, deg
    geom_source: Optional[str] = None


def group_apparitions(phot: Photometry, gap_days: float = 60.0) -> List[List[int]]:
    """Return lists of row indices, one list per apparition.

    Rows are sorted by epoch and split wherever the time gap to the next point
    exceeds ``gap_days``.
    """
    order = np.argsort(phot.time)
    groups: List[List[int]] = [[int(order[0])]]
    for prev, cur in zip(order[:-1], order[1:]):
        if phot.time[cur] - phot.time[prev] > gap_days:
            groups.append([])
        groups[-1].append(int(cur))
    return groups


def _fit_amplitude(
    phase: np.ndarray,
    mag: np.ndarray,
    merr: np.ndarray,
    order: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple:
    """Fit a Fourier series in rotational phase; return (amplitude, sigma)."""
    grid = np.linspace(0.0, 1.0, 361)

    def amp_from(p, m):
        # coeff layout for spindoc.fourier: [period, mean, phi1, A1, phi2, A2, ...]
        p0 = [1.0, float(np.median(m))]
        guess_amp = 0.5 * (np.max(m) - np.min(m))
        for k in range(order):
            p0 += [0.0, guess_amp / (k + 1)]
        try:
            popt, _ = curve_fit(fourier, p, m, p0=p0, maxfev=20000)
            model = fourier(grid, *popt)
            return float(np.max(model) - np.min(model))
        except Exception:
            return float(np.max(m) - np.min(m))

    amp = amp_from(phase, mag)

    boot = np.empty(n_boot)
    n = phase.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = amp_from(phase[idx], mag[idx] + rng.normal(0.0, merr[idx]))
    sigma = float(np.std(boot)) if n_boot > 1 else float(np.median(merr))
    sigma = max(sigma, 1e-3)
    return amp, sigma


def reduce_apparitions(
    phot: Photometry,
    period: float,
    gap_days: float = 60.0,
    fourier_order: int = 2,
    G: float = 0.15,
    min_points: int = 5,
    n_boot: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[Apparition]:
    """Reduce a :class:`Photometry` into a list of :class:`Apparition`.

    Parameters
    ----------
    period : float
        Rotation period in the same time units as ``phot.time`` (days).
    gap_days : float
        Minimum epoch gap separating apparitions.
    fourier_order : int
        Number of harmonics in the per-apparition Fourier fit (2 captures the
        usual double-peaked asteroid light curve).
    G : float
        IAU phase-slope parameter used for the cross-apparition phase
        correction. Only the relative correction matters, so ``H`` is arbitrary.
    min_points : int
        Apparitions with fewer points are skipped.
    """
    rng = rng or np.random.default_rng(0)
    apps: List[Apparition] = []

    for gi, idx in enumerate(group_apparitions(phot, gap_days)):
        if len(idx) < min_points:
            continue
        idx = np.asarray(idx)
        t = phot.time[idx]
        m = phot.mag[idx]
        e = phot.merr[idx]
        rh = phot.rhelio[idx]
        dl = phot.delta[idx]
        al = phot.alpha[idx]

        # Reduce to unit distance; phase-correct for the mean-magnitude method.
        m_red = m - 5.0 * np.log10(rh * dl)
        phase_dimming = HGfunction(al, 0.0, G) - 0.0   # = -2.5 log10(phi(alpha))
        m_corr = m_red - phase_dimming

        rot_phase = (t / period) % 1.0
        amplitude, amp_err = _fit_amplitude(rot_phase, m, e, fourier_order, n_boot, rng)

        # Robust mean reduced magnitude and its uncertainty.
        mean_mag = float(np.median(m_corr))
        mean_mag_err = float(1.4826 * np.median(np.abs(m_corr - mean_mag)) / np.sqrt(len(idx)))
        mean_mag_err = max(mean_mag_err, 1e-3)

        ecl_lon = ecl_lat = None
        if phot.has_ecliptic:
            ecl_lon = float(np.mean(phot.ecl_lon[idx]))
            ecl_lat = float(np.mean(phot.ecl_lat[idx]))

        apps.append(Apparition(
            index=gi,
            epoch_mid=float(np.mean(t)),
            n_points=len(idx),
            span_days=float(t.max() - t.min()),
            amplitude=amplitude,
            amplitude_err=amp_err,
            mean_mag=mean_mag,
            mean_mag_err=mean_mag_err,
            alpha_mean=float(np.mean(al)),
            rhelio_mean=float(np.mean(rh)),
            delta_mean=float(np.mean(dl)),
            ecl_lon=ecl_lon,
            ecl_lat=ecl_lat,
            geom_source="file" if ecl_lon is not None else None,
        ))
    return apps


__all__ = ["Apparition", "group_apparitions", "reduce_apparitions"]
