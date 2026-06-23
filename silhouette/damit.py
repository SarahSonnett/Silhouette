"""Reader for DAMIT light-curve data files (multi-apparition real photometry).

DAMIT (https://damit.cuni.cz) distributes, for each modelled asteroid, the
photometry used in its convex inversion. The plain-text ``lc`` export has the
structure::

    <number of light curves>
    <N_points> <0|1>            # 0 = relative, 1 = calibrated (reduced to unit dist.)
    <JD> <brightness> <Sx> <Sy> <Sz> <Ex> <Ey> <Ez>
    ...                          # one line per point
    <N_points> <0|1>            # next light curve
    ...

``JD`` is light-time corrected, ``brightness`` is intensity, and
``(Sx,Sy,Sz)`` / ``(Ex,Ey,Ez)`` are the asteroid-centric ecliptic Cartesian
vectors (AU) to the Sun and Earth. The Earth vector gives the observer's
ecliptic direction — exactly the aspect geometry Silhouette needs — so no
ephemeris lookup is required.

Two practical points drive the reduction here:

* **Relative scaling.** Each relative light curve has its own arbitrary zero
  point, so amplitudes must be measured *within* a single light curve, never by
  combining curves. We therefore take one amplitude per light curve and, per
  apparition, keep the largest (best rotational coverage) — the classic
  maximum-amplitude-per-apparition approach of the amplitude–aspect method.
* **Sparse blocks.** DAMIT exports also contain sparse survey photometry
  (few points spread over years). These are filtered out by requiring a dense
  block (``min_points`` within ``max_span_days``).

Because relative curves carry no absolute magnitude, the DAMIT path uses the
amplitude observable only (``fit_shape(..., use_meanmag=False)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .apparitions import Apparition


@dataclass
class DamitLightcurve:
    jd: np.ndarray
    intensity: np.ndarray
    calibrated: bool
    sun: np.ndarray            # mean asteroid-centric Sun vector (AU)
    earth: np.ndarray          # mean asteroid-centric Earth vector (AU)

    @property
    def epoch_mid(self) -> float:
        return float(self.jd.mean())

    @property
    def span_days(self) -> float:
        return float(self.jd.max() - self.jd.min())

    @property
    def amplitude(self) -> float:
        """Peak-to-trough magnitude range within this light curve."""
        m = -2.5 * np.log10(self.intensity)
        return float(m.max() - m.min())


def read_damit_lcs(path: str) -> List[DamitLightcurve]:
    """Parse a DAMIT plain-text light-curve export into a list of curves."""
    with open(path) as fh:
        tok = fh.read().split()
    pos = 0

    def take(n):
        nonlocal pos
        vals = tok[pos:pos + n]
        pos += n
        return vals

    n_lc = int(take(1)[0])
    curves: List[DamitLightcurve] = []
    for _ in range(n_lc):
        n_pts, cal = take(2)
        n_pts = int(n_pts)
        cal = bool(int(cal))
        jd = np.empty(n_pts)
        inten = np.empty(n_pts)
        sun = np.empty((n_pts, 3))
        earth = np.empty((n_pts, 3))
        for k in range(n_pts):
            row = np.array(take(8), dtype=float)
            jd[k], inten[k] = row[0], row[1]
            sun[k], earth[k] = row[2:5], row[5:8]
        curves.append(DamitLightcurve(jd, inten, cal, sun.mean(0), earth.mean(0)))
    return curves


def _ecliptic_lon_lat(vec: np.ndarray):
    """Ecliptic longitude/latitude (deg) of a Cartesian ecliptic vector."""
    x, y, z = vec
    lon = np.degrees(np.arctan2(y, x)) % 360.0
    lat = np.degrees(np.arcsin(z / np.linalg.norm(vec)))
    return float(lon), float(lat)


def damit_apparitions(
    curves: List[DamitLightcurve],
    period: float,
    gap_days: float = 60.0,
    min_points: int = 8,
    max_span_days: float = 1.5,
    amp_err_floor: float = 0.03,
    amp_err_frac: float = 0.08,
) -> List[Apparition]:
    """Reduce DAMIT light curves to one amplitude observable per apparition.

    Parameters
    ----------
    curves : list of DamitLightcurve
    period : float
        Sidereal rotation period in days (only used to label the reduction;
        amplitudes here are peak-to-trough within each dense curve).
    gap_days : float
        Minimum epoch gap separating apparitions.
    min_points, max_span_days : int, float
        Dense-curve filter: keep curves with at least ``min_points`` spanning
        less than ``max_span_days`` (excludes sparse survey blocks).
    amp_err_floor, amp_err_frac : float
        Per-apparition amplitude uncertainty is ``max(floor, frac * amplitude)``.
    """
    dense = [c for c in curves
             if c.intensity.size >= min_points and c.span_days <= max_span_days]
    if not dense:
        raise ValueError("no dense light curves passed the filter")
    dense.sort(key=lambda c: c.epoch_mid)

    # Group into apparitions by epoch gap.
    groups: List[List[DamitLightcurve]] = [[dense[0]]]
    for c in dense[1:]:
        if c.epoch_mid - groups[-1][-1].epoch_mid > gap_days:
            groups.append([])
        groups[-1].append(c)

    apps: List[Apparition] = []
    for gi, grp in enumerate(groups):
        best = max(grp, key=lambda c: c.amplitude)     # best rotational coverage
        lon, lat = _ecliptic_lon_lat(best.earth)
        rhelio = float(np.linalg.norm(best.sun))
        delta = float(np.linalg.norm(best.earth))
        cos_a = np.dot(best.sun, best.earth) / (rhelio * delta)
        alpha = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        apps.append(Apparition(
            index=gi,
            epoch_mid=best.epoch_mid,
            n_points=sum(c.intensity.size for c in grp),
            span_days=best.span_days,
            amplitude=best.amplitude,
            amplitude_err=max(amp_err_floor, amp_err_frac * best.amplitude),
            mean_mag=0.0,                # relative photometry: no absolute mag
            mean_mag_err=1.0,
            alpha_mean=alpha,
            rhelio_mean=rhelio,
            delta_mean=delta,
            ecl_lon=lon,
            ecl_lat=lat,
            geom_source="damit",
        ))
    return apps


__all__ = ["DamitLightcurve", "read_damit_lcs", "damit_apparitions"]
