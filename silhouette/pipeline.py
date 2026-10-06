"""Glue from Squint/SpinDoc photometry to the solution scorer.

The intended chain is

    Squint (images → calibrated magnitudes)
      → SpinDoc (period, H–G, Fourier fit)
      → Silhouette scorer (candidate spin/shape solutions with probabilities)
      → recommender (when to observe next).

:func:`lightcurves_from_photometry` turns a Squint/SpinDoc-style table (read
with :func:`silhouette.io.read_photometry`) into per-night
:class:`~silhouette.inversion.LightCurveObs` objects: magnitudes are reduced to
unit distances and converted to intensities, epochs are light-time corrected,
and asteroid-centric ecliptic Sun/Earth vectors come from JPL Horizons (one
query per night). Each night is treated as relative photometry by default —
zero points drift between nights far more than the convex model can absorb.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._compat import HGfunction
from .inversion import LightCurveObs
from .io import Photometry
from .scoring import alias_periods

AU_LIGHT_DAYS = 0.005775518331      # light travel time for 1 AU, in days


def split_nights(times_days: np.ndarray, gap_hours: float = 6.0) -> List[np.ndarray]:
    """Index arrays of runs separated by gaps longer than ``gap_hours``."""
    order = np.argsort(times_days)
    t = times_days[order]
    breaks = np.where(np.diff(t) > gap_hours / 24.0)[0] + 1
    return [order[s] for s in np.split(np.arange(t.size), breaks)]


def horizons_vectors_at(target: str, jd: Sequence[float],
                        location: str = "500@399") -> Tuple[np.ndarray, np.ndarray]:
    """Asteroid-centric ecliptic Sun and observer vectors (AU) at given JDs."""
    from astroquery.jplhorizons import Horizons

    jd = [float(x) for x in jd]
    helio = Horizons(id=target, location="@sun", epochs=jd,
                     id_type="smallbody").vectors(refplane="ecliptic")
    geo = Horizons(id=target, location=location, epochs=jd,
                   id_type="smallbody").vectors(refplane="ecliptic")
    r_h = np.column_stack([np.asarray(helio[k], dtype=float) for k in ("x", "y", "z")])
    r_g = np.column_stack([np.asarray(geo[k], dtype=float) for k in ("x", "y", "z")])
    return -r_h, -r_g


def lightcurves_from_photometry(
    phot: Photometry,
    target: Optional[str] = None,
    geometry: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    gap_hours: float = 6.0,
    min_points: int = 8,
    location: str = "500@399",
    sigma_floor_mag: float = 0.005,
    group: str = "night",
    phase_G: Optional[float] = None,
    app_gap_days: float = 60.0,
) -> List[LightCurveObs]:
    """Per-night relative light curves with geometry, ready for the scorer.

    Parameters
    ----------
    phot : a :class:`~silhouette.io.Photometry` (Squint/SpinDoc table; MJD times)
    target : Horizons designation used to fetch Sun/Earth vectors per night
    geometry : alternatively, ``(sun_vecs, earth_vecs)`` with one row per night
        (asteroid-centric ecliptic, AU) — e.g. for offline use
    gap_hours : gap that separates nights
    min_points : nights with fewer points are dropped
    location : Horizons observer code (default geocentre; use the site code
        for close NEAs)
    sigma_floor_mag : added in quadrature to the magnitude errors
    group : ``"night"`` (default) makes every night its own relative curve;
        ``"apparition"`` joins the nights of an apparition into one relative
        curve (a single free scale), which is what makes long-period rotators
        usable — a 23-h rotator never completes a rotation in one night, and
        per-night zero points would throw away how the nights connect. Only
        sensible for *calibrated* photometry (e.g. Squint's TmagCorr).
    phase_G : with ``group="apparition"``, divide out the IAU H–G phase
        function (SpinDoc's G) so phase-angle darkening across the apparition
        is not mistaken for shape. Approximate: the scattering law's own
        geometric phase dependence is still in the model.
    app_gap_days : gap separating apparitions when ``group="apparition"``
    """
    if group not in ("night", "apparition"):
        raise ValueError("group must be 'night' or 'apparition'")
    nights = [idx for idx in split_nights(phot.time, gap_hours) if idx.size >= min_points]
    if not nights:
        raise ValueError(f"no night has >= {min_points} points")
    mid_mjd = np.array([float(np.mean(phot.time[idx])) for idx in nights])
    if geometry is not None:
        sun, earth = (np.atleast_2d(np.asarray(g, dtype=float)) for g in geometry)
        if sun.shape[0] != len(nights):
            raise ValueError(f"geometry has {sun.shape[0]} rows but there are "
                             f"{len(nights)} nights")
    elif target is not None:
        sun, earth = horizons_vectors_at(target, mid_mjd + 2400000.5, location=location)
    else:
        raise ValueError("need either target= (for JPL Horizons) or geometry=(sun, earth)")

    def reduce(idx):
        # reduce to unit distances (and optionally zero phase), then to intensity
        mred = phot.mag[idx] - 5.0 * np.log10(phot.rhelio[idx] * phot.delta[idx])
        if phase_G is not None:
            mred = mred - HGfunction(phot.alpha[idx], 0.0, phase_G)
        flux = 10.0 ** (-0.4 * mred)
        merr = np.sqrt(phot.merr[idx] ** 2 + sigma_floor_mag ** 2)
        sigma = flux * merr * 0.4 * np.log(10.0)
        # light-time correction (DAMIT convention: epochs at the asteroid)
        t = phot.time[idx] - phot.delta[idx] * AU_LIGHT_DAYS
        return t, flux, sigma

    if group == "night":
        out = []
        for idx, s, e in zip(nights, sun, earth):
            t, flux, sigma = reduce(idx)
            o = np.argsort(t)
            out.append(LightCurveObs(t[o], flux[o], sigma[o], s, e, relative=True))
        return out

    # one relative curve per apparition, with per-point geometry (each point
    # takes its night's vectors)
    breaks = np.where(np.diff(mid_mjd) > app_gap_days)[0] + 1
    out = []
    for block in np.split(np.arange(len(nights)), breaks):
        idx = np.concatenate([nights[k] for k in block])
        svec = np.vstack([np.repeat(sun[k][None, :], nights[k].size, axis=0) for k in block])
        evec = np.vstack([np.repeat(earth[k][None, :], nights[k].size, axis=0) for k in block])
        t, flux, sigma = reduce(idx)
        o = np.argsort(t)
        out.append(LightCurveObs(t[o], flux[o], sigma[o], svec[o], evec[o], relative=True))
    return out


def spindoc_periods(period_hours: float, baseline_days: float, n_alias: int = 1,
                    harmonics: Sequence[float] = ()) -> np.ndarray:
    """Trial periods (days) around a SpinDoc period: the period and its aliases.

    ``harmonics=(0.5, 2.0)`` adds the half/double period — worth including when
    SpinDoc's Fourier-order choice between single- and double-peaked solutions
    was marginal.
    """
    return alias_periods(period_hours / 24.0, baseline_days, n_alias=n_alias,
                         harmonics=harmonics)


__all__ = ["split_nights", "horizons_vectors_at", "lightcurves_from_photometry",
           "spindoc_periods", "AU_LIGHT_DAYS"]
