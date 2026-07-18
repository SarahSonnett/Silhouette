"""Convex light-curve inversion (Phase 1a).

Fits a convex shape — as spherical-harmonic coefficients of the log Gaussian
image — together with the spin pole and rotation phase, to observed brightness
time series with error bars. This is the step beyond the analytical
amplitude–aspect fit in :mod:`silhouette.fit`: it uses every photometric point
rather than one amplitude per apparition, so departures from a perfect
ellipsoid can be absorbed by the extra shape freedom.

Parameterisation
----------------
``a_i = exp(Σ_k c_k Y_k(n_i)) · dω_i`` keeps every area weight positive for any
real coefficients. The ``l = 0`` coefficient is held fixed: it only sets the
overall size, which is degenerate with the photometric scale (all-relative
photometry carries no absolute scale, and only axis *ratios* matter downstream).

Relative light curves
---------------------
Each relative light curve has its own arbitrary scale. Rather than fit those as
free parameters, the optimal scale is **profiled out** analytically at every
evaluation (a weighted linear least-squares scalar), which removes one nuisance
parameter per light curve.

Closure
-------
A set of area weights describes a genuine closed convex body only if
``Σ a_i n_i = 0``. That is not automatic under the exponential parameterisation,
so a penalty term enforces it.

The rotation period is an **input** here; scanning it is Phase 1b (and is the
only part of the pipeline that wants multiple cores).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .forward import DEFAULT_LAW, convex_lightcurve
from .shapes import (
    ConvexShape,
    ellipsoid_gaussian_image,
    fibonacci_sphere,
    real_sph_harm_basis,
)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass
class LightCurveObs:
    """One observed light curve with its viewing/illumination geometry.

    ``sun`` and ``earth`` are asteroid-centric **ecliptic** vectors, either one
    per point or a single mean vector for the curve. ``relative=True`` means the
    flux scale is arbitrary and will be profiled out.
    """

    times: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    sun: np.ndarray
    earth: np.ndarray
    relative: bool = True

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        self.sun = np.atleast_2d(np.asarray(self.sun, dtype=float))
        self.earth = np.atleast_2d(np.asarray(self.earth, dtype=float))
        if self.sigma.size == 1:
            self.sigma = np.full(self.times.size, float(self.sigma))

    def __len__(self) -> int:
        return int(self.times.size)


@dataclass
class InversionResult:
    """Outcome of a convex inversion."""

    shape: ConvexShape
    coeffs: np.ndarray
    lmax: int
    pole_lon: float
    pole_lat: float
    phi0: float
    period: float
    chi2: float
    redchi2: float
    n_data: int
    n_params: int
    closure: float
    success: bool
    message: str = ""
    scales: List[float] = field(default_factory=list)
    # (lon, lat, redchi2) for every start of a multistart run, best first
    candidates: List[Tuple[float, float, float]] = field(default_factory=list)

    def axis_ratios(self) -> Tuple[float, float]:
        """``(a/b, b/c)`` of the DEEVE — the handoff to the geophysics module."""
        return self.shape.axis_ratios()

    def summary(self) -> str:
        ab, bc = self.axis_ratios()
        return "\n".join([
            "Silhouette convex inversion",
            f"  light-curve points : {self.n_data}   free parameters : {self.n_params}",
            f"  pole (lon, lat)    : ({self.pole_lon:.1f}, {self.pole_lat:.1f}) deg",
            f"  period             : {self.period:.6f} (fixed)",
            f"  reduced chi^2      : {self.redchi2:.3f}",
            f"  DEEVE a:b, b:c     : {ab:.3f}, {bc:.3f}",
            f"  closure residual   : {self.closure:.2e}",
        ])


# ---------------------------------------------------------------------------
# Coefficient helpers
# ---------------------------------------------------------------------------

def sh_project(values: np.ndarray, basis: np.ndarray, solid_angles: np.ndarray) -> np.ndarray:
    """Project a function sampled on the sphere onto the real-SH basis."""
    return (basis * solid_angles[:, None]).T @ values


def ellipsoid_coeffs(a: float, b: float, c: float, lmax: int,
                     normals: np.ndarray, solid_angles: np.ndarray,
                     basis: Optional[np.ndarray] = None) -> np.ndarray:
    """SH coefficients of ``log`` area density for a triaxial ellipsoid seed."""
    if basis is None:
        basis = real_sph_harm_basis(lmax, normals)
    areas = ellipsoid_gaussian_image(normals, a, b, c, solid_angles)
    return sh_project(np.log(areas / solid_angles), basis, solid_angles)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------

def _optimal_scale(model: np.ndarray, flux: np.ndarray, sigma: np.ndarray) -> float:
    """Weighted least-squares scale minimising ``||(s·model − flux)/sigma||``."""
    w = 1.0 / np.square(sigma)
    denom = float(np.sum(w * model * model))
    if denom <= 0.0:
        return 1.0
    return float(np.sum(w * flux * model) / denom)


def invert_convex(
    lightcurves: Sequence[LightCurveObs],
    period: float,
    pole0: Tuple[float, float] = (0.0, 0.0),
    phi0: float = 0.0,
    lmax: int = 4,
    n_normals: int = 200,
    seed_axes: Tuple[float, float, float] = (1.3, 1.1, 1.0),
    phot_func: Callable = DEFAULT_LAW,
    phot_arg=None,
    closure_weight: float = 50.0,
    max_nfev: int = 400,
    verbose: int = 0,
) -> InversionResult:
    """Fit a convex shape + pole + rotation phase to observed light curves.

    Parameters
    ----------
    lightcurves : sequence of LightCurveObs
    period : rotation period, in the time units of the observations (held fixed)
    pole0 : initial ``(lon, lat)`` in degrees — e.g. from
        :func:`silhouette.fit.fit_shape` or a published solution
    lmax : spherical-harmonic degree of the shape expansion
    n_normals : number of Gaussian-image directions
    seed_axes : axis ratios used to build the starting shape
    closure_weight : strength of the ``Σ a_i n_i = 0`` penalty
    """
    normals, solid = fibonacci_sphere(n_normals)
    basis = real_sph_harm_basis(lmax, normals)
    n_coef = basis.shape[1]

    c_seed = ellipsoid_coeffs(*seed_axes, lmax, normals, solid, basis)
    c0_fixed = float(c_seed[0])                      # size is degenerate: hold it

    t0 = float(min(np.min(lc.times) for lc in lightcurves))
    n_data = int(sum(len(lc) for lc in lightcurves))

    def build_shape(free_coeffs):
        coeffs = np.concatenate([[c0_fixed], free_coeffs])
        return ConvexShape(normals=normals,
                           areas=np.exp(basis @ coeffs) * solid), coeffs

    def unpack(p):
        return p[:n_coef - 1], p[n_coef - 1], p[n_coef], p[n_coef + 1]

    def residuals(p):
        free_c, plon, plat, ph0 = unpack(p)
        shape, _ = build_shape(free_c)
        out = []
        for lc in lightcurves:
            model = convex_lightcurve(shape, lc.times, lc.sun, lc.earth,
                                      plon, plat, period, phi0=ph0, t0=t0,
                                      phot_func=phot_func, arg=phot_arg)
            scale = _optimal_scale(model, lc.flux, lc.sigma) if lc.relative else 1.0
            out.append((scale * model - lc.flux) / lc.sigma)
        closure = (shape.areas @ normals) / shape.areas.sum()
        out.append(closure_weight * np.sqrt(n_data) * closure)
        return np.concatenate(out)

    p0 = np.concatenate([c_seed[1:], [pole0[0], pole0[1], phi0]])
    lo = np.concatenate([np.full(n_coef - 1, -np.inf), [-np.inf, -90.0, -np.inf]])
    hi = np.concatenate([np.full(n_coef - 1, np.inf), [np.inf, 90.0, np.inf]])

    res = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                        max_nfev=max_nfev, verbose=verbose)

    free_c, plon, plat, ph0 = unpack(res.x)
    shape, coeffs = build_shape(free_c)

    scales = []
    for lc in lightcurves:
        model = convex_lightcurve(shape, lc.times, lc.sun, lc.earth,
                                  plon, plat, period, phi0=ph0, t0=t0,
                                  phot_func=phot_func, arg=phot_arg)
        scales.append(_optimal_scale(model, lc.flux, lc.sigma) if lc.relative else 1.0)

    n_params = n_coef - 1 + 3
    chi2 = float(2.0 * res.cost)
    dof = max(n_data - n_params, 1)
    return InversionResult(
        shape=shape, coeffs=coeffs, lmax=lmax,
        pole_lon=float(plon % 360.0), pole_lat=float(plat), phi0=float(ph0),
        period=float(period), chi2=chi2, redchi2=chi2 / dof,
        n_data=n_data, n_params=n_params,
        closure=shape.closure_residual(),
        success=bool(res.success), message=str(res.message), scales=scales,
    )


def default_pole_grid(n_lon: int = 6, lats: Sequence[float] = (-70, -35, 0, 35, 70)):
    """A coarse grid of starting poles covering the ecliptic sphere."""
    return [(float(lon), float(lat))
            for lat in lats
            for lon in np.linspace(0.0, 360.0, n_lon, endpoint=False)]


def _multistart_worker(payload):
    """Module-level helper so starts can be dispatched to worker processes."""
    lightcurves, period, seed, kwargs = payload
    try:
        return invert_convex(lightcurves, period=period, pole0=seed, **kwargs)
    except Exception:
        return None


def invert_convex_multistart(
    lightcurves: Sequence[LightCurveObs],
    period: float,
    pole_grid: Optional[Sequence[Tuple[float, float]]] = None,
    n_workers: int = 1,
    **kwargs,
) -> InversionResult:
    """Run :func:`invert_convex` from many starting poles and keep the best fit.

    Convex inversion is strongly **multi-modal** — a single start can converge to
    a spurious minimum (most commonly a near-spherical shape that fits the mean
    brightness but no rotational structure). Scanning a grid of starting poles
    and keeping the lowest chi-squared is the standard remedy.

    Starts are independent, so this is the natural place to spend cores.
    ``n_workers=1`` (the default) runs serially and never oversubscribes; raise
    it to parallelise. BLAS threading is pinned to one thread per worker so a
    pool of N workers uses N cores, not N x cores.

    The returned result carries every distinct solution in ``.candidates`` so the
    degree of multi-modality is visible rather than hidden.

    .. note::
       With ``n_workers > 1`` this uses ``multiprocessing``, which on macOS
       starts workers by *spawn*. The calling code must therefore live in an
       importable module guarded by ``if __name__ == "__main__":`` — running it
       from a heredoc or an interactive ``python -`` session raises
       ``BrokenProcessPool``. Use ``n_workers=1`` in those contexts.
    """
    grid = list(pole_grid) if pole_grid is not None else default_pole_grid()

    if n_workers > 1:
        # Children re-import NumPy under 'spawn'; pin their BLAS threads.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        from concurrent.futures import ProcessPoolExecutor
        payloads = [(list(lightcurves), period, seed, kwargs) for seed in grid]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_multistart_worker, payloads))
    else:
        results = [_multistart_worker((list(lightcurves), period, s, kwargs))
                   for s in grid]

    good = [r for r in results if r is not None and np.isfinite(r.redchi2)]
    if not good:
        raise RuntimeError("every starting pole failed to converge")

    good.sort(key=lambda r: r.redchi2)
    best = good[0]
    best.candidates = [(r.pole_lon, r.pole_lat, r.redchi2) for r in good]
    return best


__all__ = ["LightCurveObs", "InversionResult", "invert_convex",
           "invert_convex_multistart", "default_pole_grid",
           "ellipsoid_coeffs", "sh_project"]
