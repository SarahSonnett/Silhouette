"""Analytical shape + pole fit from reduced apparition observables.

Given a list of :class:`~silhouette.apparitions.Apparition` (each with an
amplitude, a mean magnitude, and an ecliptic direction), this performs a
weighted nonlinear least-squares fit over

    p = a/b,  q = b/c,  pole longitude lam_p,  pole latitude beta_p,
    and a mean-magnitude zero point M0

jointly matching the amplitude-aspect and mean-magnitude observables. A grid of
pole starting points avoids local minima. Degeneracies are handled explicitly:

* 1 apparition  -> report an ``a/b`` lower bound only; pole and ``b/c`` undetermined.
* 2 apparitions -> a fit is attempted but flagged as weakly constrained.
* The prograde/retrograde mirror pole is always reported alongside the best one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares

from .apparitions import Apparition
from .model import (
    ab_lower_bound,
    amplitude_model,
    aspect_angle,
    axes_from_ratios,
    mean_mag_model,
    mirror_pole,
)


@dataclass
class PoleSolution:
    pole_lon: float
    pole_lat: float
    cost: float


@dataclass
class SilhouetteFit:
    """Result of an analytical shape + pole fit."""

    ab: float                                  # a/b
    bc: float                                  # b/c
    axes: tuple                                # (a, b, c), b = 1
    pole_lon: Optional[float]
    pole_lat: Optional[float]
    mirror_pole: Optional[tuple]               # (lon, lat) degenerate solution
    zero_point: float
    perr: dict = field(default_factory=dict)   # 1-sigma uncertainties by name
    candidates: List[PoleSolution] = field(default_factory=list)
    n_apparitions: int = 0
    redchi2: float = float("nan")
    degenerate: bool = False
    used_meanmag: bool = True                  # was the mean-magnitude observable fit?
    notes: List[str] = field(default_factory=list)
    apparitions: List[Apparition] = field(default_factory=list)

    def summary(self) -> str:
        import math

        def err(key):
            v = self.perr.get(key)
            return v if (v is not None and math.isfinite(v)) else None

        lines = [
            "Silhouette shape + pole fit",
            f"  apparitions used : {self.n_apparitions}",
            f"  a:b = {self.ab:.3f}" + (f" +/- {err('ab'):.3f}" if err('ab') else ""),
        ]
        if self.degenerate and self.pole_lon is None:
            lines.append(f"  b:c = undetermined (single apparition)")
            lines.append(f"  a:b is a LOWER BOUND (equatorial aspect assumed)")
        else:
            lines.append(f"  b:c = {self.bc:.3f}" + (f" +/- {err('bc'):.3f}" if err('bc') else ""))
            lines.append(f"  pole (lon, lat) = ({self.pole_lon:.1f}, {self.pole_lat:.1f}) deg"
                         + (f"  +/- ({err('pole_lon'):.1f}, {err('pole_lat'):.1f})"
                            if err('pole_lat') and err('pole_lon') else ""))
            if self.mirror_pole is not None:
                lines.append(f"  mirror pole     = ({self.mirror_pole[0]:.1f}, "
                             f"{self.mirror_pole[1]:.1f}) deg [degenerate]")
            lines.append(f"  reduced chi^2   = {self.redchi2:.2f}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _residuals(params, lon, lat, A, sA, M, sM, use_meanmag):
    p, q, lam_p, beta_p, m0 = params
    theta = aspect_angle(lon, lat, lam_p, beta_p)
    res = [(amplitude_model(p, q, theta) - A) / sA]
    if use_meanmag:
        res.append((mean_mag_model(p, q, theta, zero_point=m0) - M) / sM)
    return np.concatenate(res)


def _covariance(result, dof):
    """1-sigma parameter errors from a least_squares result."""
    J = result.jac
    cost = 2.0 * result.cost
    try:
        cov = np.linalg.inv(J.T @ J) * (cost / max(dof, 1))
        return np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.nan)


def fit_shape(
    apparitions: List[Apparition],
    use_meanmag: bool = True,
    p_bounds=(1.0, 6.0),
    q_bounds=(1.0, 6.0),
    pole_lon_grid=None,
    pole_lat_grid=None,
) -> SilhouetteFit:
    """Fit axis ratios and pole from reduced apparitions.

    Parameters
    ----------
    apparitions : list of Apparition
        Must have ecliptic geometry resolved (see ``geometry.resolve_geometry``).
    use_meanmag : bool
        Include the mean-magnitude observable in the fit (recommended).
    p_bounds, q_bounds : tuple
        Bounds on ``a/b`` and ``b/c``.
    pole_lon_grid, pole_lat_grid : array-like, optional
        Starting points for the multi-start pole search.
    """
    n = len(apparitions)
    if n == 0:
        raise ValueError("no apparitions to fit")

    amps = np.array([a.amplitude for a in apparitions])
    samps = np.array([a.amplitude_err for a in apparitions])

    # --- Single apparition: amplitude lower bound only -----------------------
    if n == 1:
        lb = ab_lower_bound(amps[0])
        return SilhouetteFit(
            ab=lb, bc=float("nan"), axes=(lb, 1.0, float("nan")),
            pole_lon=None, pole_lat=None, mirror_pole=None, zero_point=0.0,
            n_apparitions=1, degenerate=True, used_meanmag=False,
            notes=["single apparition: a/b is a lower bound at equatorial aspect; "
                   "pole and b/c are undetermined"],
            apparitions=list(apparitions),
        )

    for a in apparitions:
        if a.ecl_lon is None or a.ecl_lat is None:
            raise ValueError(
                "apparition geometry not resolved; call geometry.resolve_geometry first"
            )

    lon = np.array([a.ecl_lon for a in apparitions])
    lat = np.array([a.ecl_lat for a in apparitions])
    means = np.array([a.mean_mag for a in apparitions])
    smeans = np.array([a.mean_mag_err for a in apparitions])
    m0_guess = float(np.median(means))

    if pole_lon_grid is None:
        pole_lon_grid = np.arange(0.0, 360.0, 45.0)
    if pole_lat_grid is None:
        pole_lat_grid = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])

    lo = [p_bounds[0], q_bounds[0], 0.0, -90.0, -np.inf]
    hi = [p_bounds[1], q_bounds[1], 360.0, 90.0, np.inf]

    best = None
    cand: List[PoleSolution] = []
    p0_amp = float(np.clip(ab_lower_bound(amps.max()), *p_bounds))

    for lam0 in pole_lon_grid:
        for bet0 in pole_lat_grid:
            x0 = [p0_amp, 1.5, float(lam0), float(bet0), m0_guess]
            try:
                r = least_squares(
                    _residuals, x0, bounds=(lo, hi), method="trf",
                    args=(lon, lat, amps, samps, means, smeans, use_meanmag),
                    max_nfev=5000,
                )
            except Exception:
                continue
            cand.append(PoleSolution(r.x[2] % 360.0, r.x[3], float(r.cost)))
            if best is None or r.cost < best.cost:
                best = r

    # Distinct pole minima (dedup by ~10 deg), best first.
    cand.sort(key=lambda s: s.cost)
    distinct: List[PoleSolution] = []
    for s in cand:
        if not any(abs(((s.pole_lon - d.pole_lon + 180) % 360) - 180) < 10
                   and abs(s.pole_lat - d.pole_lat) < 10 for d in distinct):
            distinct.append(s)

    p, q, lam_p, beta_p, m0 = best.x
    n_obs = len(amps) * (2 if use_meanmag else 1)
    n_par = 5 if use_meanmag else 4
    dof = max(n_obs - n_par, 1)
    redchi2 = float(2.0 * best.cost / dof)
    perr_arr = _covariance(best, dof)
    perr = {
        "ab": float(perr_arr[0]),
        "bc": float(abs(perr_arr[1] / max(q, 1e-6) ** 2)),  # d(1/q) propagation note
        "pole_lon": float(perr_arr[2]),
        "pole_lat": float(perr_arr[3]),
    }
    # b/c = q directly; report its own sigma.
    perr["bc"] = float(perr_arr[1])

    notes = []
    degenerate = False
    if n == 2:
        degenerate = True
        notes.append("only 2 apparitions: pole/b:c weakly constrained")
    if n < 4:
        notes.append("fewer than 4 apparitions: solution may be non-unique; "
                     "check candidate poles")

    return SilhouetteFit(
        ab=float(p),
        bc=float(q),
        axes=axes_from_ratios(p, q),
        pole_lon=float(lam_p % 360.0),
        pole_lat=float(beta_p),
        mirror_pole=mirror_pole(lam_p % 360.0, beta_p),
        zero_point=float(m0),
        perr=perr,
        candidates=distinct[:6],
        n_apparitions=n,
        redchi2=redchi2,
        degenerate=degenerate,
        used_meanmag=use_meanmag,
        notes=notes,
        apparitions=list(apparitions),
    )


__all__ = ["SilhouetteFit", "PoleSolution", "fit_shape"]
