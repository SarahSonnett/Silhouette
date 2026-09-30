"""Solution scorer: probabilities for competing spin/shape solutions.

Convex inversion rarely returns *one* answer. Period aliases, the
``(λ, β) ↔ (λ + 180°, β)`` mirror pole, the spin-sense antipode and plain
local minima all produce candidate solutions whose chi-squared values differ by
little. The classical practice (DAMIT, Ďurech et al.) is to accept a solution as
"unique" when it beats every rival by a hand-set chi-squared margin, and to
reject everything else. That is an accept/reject rule, not a probability, and it
throws away the partial information in the rejected cases.

This module turns a set of candidate solutions into a ranked list with a
probability attached to each, in three layers:

1. **Likelihood layer** (:func:`likelihood_weights`) — relative likelihoods from
   chi-squared with an honest noise model: the error bars are rescaled so the
   best fit has ``chi2_nu = 1`` (unmodelled systematics, or over-generous
   assumed errors, are taken to be shared by every candidate), and the effective number of independent points is
   reduced for residual autocorrelation within each light curve (correlated
   residuals make raw Δχ² far too decisive). Optional population priors on pole
   latitude and elongation enter here.
2. **Stability layer** (:func:`bootstrap_stability`) — whole light curves are
   resampled with replacement, every candidate is refit from a warm start, and
   the fraction of resamples each candidate wins is recorded. This follows the
   bootstrap-agreement idea of Ďurech et al. (2022) for ATLAS periods, extended
   from periods to pole families.
3. **Calibration layer** (:mod:`silhouette.calibration`) — a classifier trained
   on injected synthetic asteroids maps the features computed here (Δχ²,
   bootstrap agreement, geometry coverage, noise, ...) to a *calibrated*
   ``P(candidate is correct)``. Without a trained model, the scorer reports the
   uncalibrated likelihood weights and says so.

The candidate list itself comes from :func:`find_candidates`, which runs the
existing pole multistart at each trial period (typically a SpinDoc period plus
its one-rotation aliases, see :func:`alias_periods`) and clusters the converged
starts into pole families.

Pole convention
---------------
Poles here are in Silhouette's own convention (the same one
:func:`silhouette.forward.convex_lightcurve` uses). See ``docs/scoring.md`` for
the note on how that convention relates to DAMIT's spin-vector poles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forward import DEFAULT_LAW, convex_lightcurve
from .inversion import (
    InversionResult,
    LightCurveObs,
    _optimal_scale,
    _sep_deg,
    default_pole_grid,
    invert_convex,
)
from .shapes import ConvexShape

def spin_vector_pole(lon: float, lat: float) -> Tuple[float, float]:
    """Convert a Silhouette pole to the angular-momentum (DAMIT-style) pole.

    :func:`silhouette.forward.ecliptic_to_body_matrix` applies ``R_z(+φ)`` to
    ecliptic vectors, which means the *body* turns by ``−φ`` about the stated
    pole: Silhouette's pole points **opposite** to the spin angular momentum.
    DAMIT (and the literature) quote the angular-momentum direction, i.e. the
    antipode ``(λ + 180°, −β)``. Both describe the same physical rotation, so
    comparisons with DAMIT must go through this conversion. (Found 2026-09-29;
    it is why the Eunomia example matched DAMIT only "with mirror allowed".)
    """
    return float((lon + 180.0) % 360.0), float(-lat)


# ---------------------------------------------------------------------------
# Candidate container
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One distinct spin/shape solution competing to explain the data."""

    period: float
    pole_lon: float
    pole_lat: float
    chi2: float
    redchi2: float
    n_starts: int                   # starts that converged into this family
    n_starts_total: int             # starts attempted at this period
    fit: InversionResult = field(repr=False)
    elongation: float = np.nan      # cheap a/b proxy (projected-area ratio)
    flattening: float = np.nan      # cheap b/c proxy
    log_like: float = np.nan
    like_weight: float = np.nan
    boot_frac: float = np.nan
    probability: float = np.nan
    calibrated: bool = False
    features: Dict[str, float] = field(default_factory=dict, repr=False)

    def label(self) -> str:
        return (f"P={self.period:.7f}  pole=({self.pole_lon:6.1f},{self.pole_lat:6.1f})"
                f"  a/b~{self.elongation:.2f}")


# ---------------------------------------------------------------------------
# Cheap shape proxies
# ---------------------------------------------------------------------------

def projected_areas(shape: ConvexShape, dirs: np.ndarray) -> np.ndarray:
    """Projected area of a convex body along each unit direction (body frame).

    For a closed convex body ``A(v) = ½ Σ a_i |n_i · v|`` — no Minkowski solve
    needed, so this is instantaneous.
    """
    return 0.5 * np.abs(shape.normals @ np.atleast_2d(dirs).T).T @ shape.areas


def shape_proxies(shape: ConvexShape, n_dirs: int = 72) -> Tuple[float, float]:
    """``(a/b, b/c)`` proxies from projected areas, without the Minkowski solve.

    For a triaxial ellipsoid spinning about ``c`` the equatorial projected area
    runs from ``πbc`` to ``πac`` (ratio ``a/b``) and the pole-on area is ``πab``
    (ratio to the maximum equatorial area ``b/c``), so these reproduce the
    ellipsoid values exactly and approximate the DEEVE ratios for general
    convex shapes. The full :meth:`ConvexShape.axis_ratios` costs seconds per
    call; these cost microseconds, which matters inside the calibration loop.
    """
    th = np.linspace(0.0, np.pi, n_dirs, endpoint=False)
    eq = projected_areas(shape, np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)]))
    pole_on = float(projected_areas(shape, np.array([[0.0, 0.0, 1.0]]))[0])
    return float(eq.max() / eq.min()), float(pole_on / eq.max())


# ---------------------------------------------------------------------------
# Trial periods
# ---------------------------------------------------------------------------

def alias_periods(period: float, baseline: float, n_alias: int = 1,
                  harmonics: Sequence[float] = ()) -> np.ndarray:
    """A period and its one-rotation aliases, ``P ± k·P²/T`` for ``k ≤ n_alias``.

    Over a baseline ``T`` two periods differing by ``P²/T`` differ by exactly
    one rotation, so both phase the data equally well — the classic alias the
    Phase-1b scan found 108 s from the truth. Feed a SpinDoc period (or a
    :func:`silhouette.inversion.scan_period` minimum) through this to get the
    trial periods the scorer should weigh against each other.

    ``harmonics`` optionally adds multiples such as ``(0.5, 2.0)`` for the
    single- vs double-peaked ambiguity.
    """
    step = period ** 2 / baseline
    out = [period + k * step for k in range(-n_alias, n_alias + 1)]
    out += [period * h for h in harmonics]
    out = np.array(sorted(set(round(p, 12) for p in out if p > 0)))
    return out


def baseline_of(lightcurves: Sequence[LightCurveObs]) -> float:
    t = np.concatenate([lc.times for lc in lightcurves])
    return float(t.max() - t.min())


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def _pin_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _start_worker(payload):
    lightcurves, period, seed, kwargs = payload
    try:
        return period, invert_convex(lightcurves, period=period, pole0=seed, **kwargs)
    except Exception:
        return period, None


def _run_pool(func, payloads, n_workers):
    if n_workers > 1 and len(payloads) > 1:
        _pin_threads()
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(func, payloads))
    return [func(p) for p in payloads]


def find_candidates(
    lightcurves: Sequence[LightCurveObs],
    periods: Sequence[float],
    pole_grid: Optional[Sequence[Tuple[float, float]]] = None,
    n_workers: int = 1,
    pole_tol_deg: float = 15.0,
    max_candidates: int = 8,
    chi2_ratio_cut: float = 3.0,
    **inv_kwargs,
) -> List[Candidate]:
    """Run the pole multistart at every trial period and collect distinct solutions.

    Every (period, starting pole) pair is an independent inversion, so they are
    dispatched to one pool of ``n_workers`` processes. Converged starts at each
    period are clustered into pole families (best chi-squared first, within
    ``pole_tol_deg``); each family becomes a :class:`Candidate` carrying its
    best fit. Families worse than ``chi2_ratio_cut × best chi2_nu`` are dropped
    and at most ``max_candidates`` are kept, best first.

    ``inv_kwargs`` go to :func:`silhouette.inversion.invert_convex` (``lmax``,
    ``n_normals``, ``max_nfev``, ``phot_func``, ...).

    As with the other multiprocess helpers, ``n_workers > 1`` needs the caller
    to be an importable module with a ``__main__`` guard.
    """
    grid = list(pole_grid) if pole_grid is not None else default_pole_grid()
    periods = [float(p) for p in periods]
    payloads = [(list(lightcurves), p, seed, inv_kwargs) for p in periods for seed in grid]
    out = _run_pool(_start_worker, payloads, n_workers)

    candidates: List[Candidate] = []
    for p in periods:
        fits = [r for q, r in out if q == p and r is not None and np.isfinite(r.redchi2)]
        fits.sort(key=lambda r: r.redchi2)
        fams: List[List[InversionResult]] = []
        for r in fits:
            for fam in fams:
                if _sep_deg(r.pole_lon, r.pole_lat, fam[0].pole_lon, fam[0].pole_lat) < pole_tol_deg:
                    fam.append(r)
                    break
            else:
                fams.append([r])
        for fam in fams:
            best = fam[0]
            ab, bc = shape_proxies(best.shape)
            candidates.append(Candidate(
                period=p, pole_lon=best.pole_lon, pole_lat=best.pole_lat,
                chi2=best.chi2, redchi2=best.redchi2, n_starts=len(fam),
                n_starts_total=len(grid), fit=best, elongation=ab, flattening=bc))

    if not candidates:
        raise RuntimeError("no start converged at any trial period")
    candidates.sort(key=lambda c: c.chi2)
    best_red = candidates[0].redchi2
    candidates = [c for c in candidates if c.redchi2 <= chi2_ratio_cut * best_red]
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Layer 1: likelihood weights
# ---------------------------------------------------------------------------

def model_flux(fit: InversionResult, lightcurves: Sequence[LightCurveObs],
               phot_func: Callable = DEFAULT_LAW, phot_arg=None,
               t0: Optional[float] = None) -> List[np.ndarray]:
    """Model intensities of a fit for each light curve (relative scales profiled)."""
    t0 = float(min(np.min(lc.times) for lc in lightcurves)) if t0 is None else t0
    out = []
    for lc in lightcurves:
        m = convex_lightcurve(fit.shape, lc.times, lc.sun, lc.earth,
                              fit.pole_lon, fit.pole_lat, fit.period,
                              phi0=fit.phi0, t0=t0, phot_func=phot_func, arg=phot_arg)
        s = _optimal_scale(m, lc.flux, lc.sigma) if lc.relative else 1.0
        out.append(s * m)
    return out


def residual_autocorrelation(fit: InversionResult,
                             lightcurves: Sequence[LightCurveObs],
                             **model_kw) -> float:
    """Pooled lag-1 autocorrelation of normalised residuals within light curves.

    Model systematics show up as runs of same-sign residuals along each curve.
    Such points are not independent, so Δχ² computed as if they were overstates
    the evidence between candidates.
    """
    num = den = 0.0
    for lc, m in zip(lightcurves, model_flux(fit, lightcurves, **model_kw)):
        order = np.argsort(lc.times)
        r = ((lc.flux - m) / lc.sigma)[order]
        if r.size < 3:
            continue
        r = r - r.mean()
        num += float(np.sum(r[1:] * r[:-1]))
        den += float(np.sum(r * r))
    return num / den if den > 0 else 0.0


def neff_factor(rho: float) -> float:
    """Effective-sample-size factor ``(1 − ρ)/(1 + ρ)`` for AR(1)-like residuals."""
    return float(np.clip((1.0 - rho) / (1.0 + rho), 0.05, 1.0))


@dataclass
class PopulationPrior:
    """Optional population priors, as log-densities (``None`` = flat).

    ``pole_lat_logpdf(lat_deg)`` — e.g. a LEADER/PyLEADER pole-latitude
    distribution (β folded or not); ``elongation_logpdf(ab)`` — an a/b
    distribution. Candidates are points, so an isotropic pole prior is simply
    flat here.
    """

    pole_lat_logpdf: Optional[Callable[[float], float]] = None
    elongation_logpdf: Optional[Callable[[float], float]] = None

    def __call__(self, c: Candidate) -> float:
        lp = 0.0
        if self.pole_lat_logpdf is not None:
            lp += float(self.pole_lat_logpdf(c.pole_lat))
        if self.elongation_logpdf is not None:
            lp += float(self.elongation_logpdf(c.elongation))
        return lp


def noise_scale(redchi2_min: float) -> float:
    """Error-bar rescaling ``s² = χ²_ν,min`` (floored at 0.05).

    Scaling the errors so the best candidate has ``χ²_ν = 1`` works in *both*
    directions: inflating underestimated errors (unmodelled systematics) and
    shrinking overestimated ones — e.g. the uniform 2 % assumed for DAMIT
    curves, where ``χ²_ν ≈ 0.2`` would otherwise flatten every Δχ² fivefold.
    It is the same normalisation that makes the classical ``χ²/χ²_min``
    uniqueness threshold scale-free.
    """
    return float(max(redchi2_min, 0.05))


def likelihood_weights(candidates: Sequence[Candidate],
                       lightcurves: Sequence[LightCurveObs],
                       prior: Optional[Callable[[Candidate], float]] = None,
                       **model_kw) -> Dict[str, float]:
    """Fill ``log_like`` and ``like_weight`` on every candidate (in place).

    ``log L_i = −½ · f_eff · (χ²_i − χ²_min) / s²`` + BIC-style parameter
    penalty + log prior, with ``s² = χ²_ν,min`` (:func:`noise_scale`: errors
    rescaled so the best model has ``χ²_ν = 1``) and ``f_eff`` from the best fit's residual
    autocorrelation. Weights are the normalised ``exp(log L)``; they sum to one
    over the candidates *found*, so they cannot express "none of these" — the
    calibration layer can.

    Returns the noise-model diagnostics.
    """
    best = min(candidates, key=lambda c: c.chi2)
    s2 = noise_scale(best.redchi2)
    rho = residual_autocorrelation(best.fit, lightcurves, **model_kw)
    f = neff_factor(rho)
    n = best.fit.n_data
    for c in candidates:
        ll = -0.5 * f * (c.chi2 - best.chi2) / s2
        ll += -0.5 * (c.fit.n_params - best.fit.n_params) * np.log(max(n * f, 2.0))
        if prior is not None:
            ll += prior(c)
        c.log_like = float(ll)
    lls = np.array([c.log_like for c in candidates])
    w = np.exp(lls - lls.max())
    w /= w.sum()
    for c, wi in zip(candidates, w):
        c.like_weight = float(wi)
    return {"s2": s2, "rho": rho, "neff_factor": f}


# ---------------------------------------------------------------------------
# Layer 2: bootstrap stability
# ---------------------------------------------------------------------------

def _rebase_phi0(phi0: float, period: float, t0_old: float, t0_new: float) -> float:
    """Rotation phase at a new reference epoch (same physical orientation)."""
    return float(phi0 + 2.0 * np.pi * (t0_new - t0_old) / period)


def _resample(lightcurves: Sequence[LightCurveObs], rng: np.random.Generator):
    """Whole-light-curve bootstrap; falls back to within-curve points if < 3 curves.

    Points inside a curve share systematics (zero point, seeing, model misfit),
    so the light curve — not the point — is the natural resampling unit.
    """
    n = len(lightcurves)
    if n >= 3:
        return [lightcurves[i] for i in rng.integers(0, n, n)]
    out = []
    for lc in lightcurves:
        idx = np.sort(rng.integers(0, len(lc), len(lc)))
        sun = lc.sun if lc.sun.shape[0] == 1 else lc.sun[idx]
        earth = lc.earth if lc.earth.shape[0] == 1 else lc.earth[idx]
        out.append(LightCurveObs(lc.times[idx], lc.flux[idx], lc.sigma[idx],
                                 sun, earth, relative=lc.relative))
    return out


def _boot_worker(payload):
    sample, cands, t0_orig, kwargs = payload
    t0_new = float(min(np.min(lc.times) for lc in sample))
    chis = []
    for period, lon, lat, phi0, coeffs, lmax, n_norm in cands:
        try:
            r = invert_convex(sample, period=period, pole0=(lon, lat),
                              phi0=_rebase_phi0(phi0, period, t0_orig, t0_new),
                              lmax=lmax, n_normals=n_norm, coeffs0=coeffs, **kwargs)
            chis.append(r.chi2)
        except Exception:
            chis.append(np.inf)
    return chis


def bootstrap_stability(candidates: Sequence[Candidate],
                        lightcurves: Sequence[LightCurveObs],
                        n_boot: int = 20,
                        seed: int = 0,
                        boot_max_nfev: int = 40,
                        n_workers: int = 1,
                        **inv_kwargs) -> np.ndarray:
    """Fraction of bootstrap resamples each candidate wins (fills ``boot_frac``).

    Each resample refits *every* candidate from a warm start (its own shape,
    pole and phase), with a small ``boot_max_nfev`` so candidates stay in their own
    basins, then awards the resample to the lowest chi-squared. A solution that
    only wins because of one or two light curves loses this vote.

    Returns the ``(n_boot, n_candidates)`` chi-squared matrix.
    """
    rng = np.random.default_rng(seed)
    t0 = float(min(np.min(lc.times) for lc in lightcurves))
    inv_kwargs = dict(inv_kwargs)
    inv_kwargs.pop("lmax", None)
    inv_kwargs.pop("n_normals", None)
    inv_kwargs["max_nfev"] = boot_max_nfev
    cands = [(c.period, c.pole_lon, c.pole_lat, c.fit.phi0, c.fit.coeffs,
              c.fit.lmax, c.fit.shape.normals.shape[0]) for c in candidates]
    payloads = [(_resample(lightcurves, rng), cands, t0, inv_kwargs) for _ in range(n_boot)]
    chi = np.array(_run_pool(_boot_worker, payloads, n_workers), dtype=float)
    wins = np.zeros(len(candidates))
    for row in chi:
        if np.all(~np.isfinite(row)):
            continue
        wins[int(np.nanargmin(row))] += 1
    frac = wins / max(n_boot, 1)
    for c, fr in zip(candidates, frac):
        c.boot_frac = float(fr)
    return chi


# ---------------------------------------------------------------------------
# Features for the calibration layer
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _lonlat(v):
    v = _unit(v)
    return (np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0,
            np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))))


def longitude_coverage(lons_deg: np.ndarray) -> float:
    """Degrees of ecliptic longitude spanned: ``360 − largest circular gap``."""
    lons = np.sort(np.asarray(lons_deg) % 360.0)
    if lons.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([lons, [lons[0] + 360.0]]))
    return float(360.0 - gaps.max())


def data_features(lightcurves: Sequence[LightCurveObs],
                  app_gap_days: float = 60.0) -> Dict[str, float]:
    """Candidate-independent descriptors of the data set (times in days)."""
    mids = np.sort([float(np.mean(lc.times)) for lc in lightcurves])
    n_app = 1 + int(np.sum(np.diff(mids) > app_gap_days)) if mids.size else 0
    e_lon, e_lat = _lonlat(np.vstack([_unit(lc.earth).mean(axis=0) for lc in lightcurves]))
    alphas = np.concatenate([
        np.degrees(np.arccos(np.clip(np.sum(_unit(lc.sun) * _unit(lc.earth), axis=1), -1, 1)))
        for lc in lightcurves])
    noise = np.concatenate([lc.sigma / np.maximum(np.abs(lc.flux), 1e-12) for lc in lightcurves])
    amps = [2.5 * np.log10(np.percentile(lc.flux, 95) / max(np.percentile(lc.flux, 5), 1e-12))
            for lc in lightcurves if len(lc) >= 5]
    return {
        "n_points": float(sum(len(lc) for lc in lightcurves)),
        "n_lc": float(len(lightcurves)),
        "n_app": float(n_app),
        "lon_coverage": longitude_coverage(e_lon),
        "lat_span": float(np.ptp(e_lat)) if e_lat.size else 0.0,
        "phase_max": float(alphas.max()),
        "phase_min": float(alphas.min()),
        "noise_frac": float(np.median(noise)),
        "amp_mag": float(np.median(amps)) if amps else 0.0,
        "baseline_days": baseline_of(lightcurves),
    }


def candidate_features(candidates: Sequence[Candidate],
                       lightcurves: Sequence[LightCurveObs],
                       noise_diag: Optional[Dict[str, float]] = None,
                       data_feats: Optional[Dict[str, float]] = None) -> None:
    """Fill ``features`` on every candidate (in place).

    Combines the data descriptors with per-candidate evidence (scaled Δχ²,
    likelihood weight, bootstrap agreement, basin size) and structure
    (|β|, shape proxies, whether its mirror/antipode is also a candidate, its
    period rank). These are the inputs of the calibration classifier.
    """
    df = data_feats if data_feats is not None else data_features(lightcurves)
    nd = noise_diag or {}
    best = min(candidates, key=lambda c: c.chi2)
    s2 = nd.get("s2", noise_scale(best.redchi2))
    f = nd.get("neff_factor", 1.0)
    period_best = {}
    for c in candidates:
        period_best[c.period] = min(period_best.get(c.period, np.inf), c.chi2)
    p_order = sorted(period_best, key=period_best.get)
    ranked = sorted(candidates, key=lambda c: c.chi2)
    span = [max(float(np.ptp(lc.times)), 0.0) for lc in lightcurves]
    for c in candidates:
        mirror = any(o is not c and o.period == c.period and
                     _sep_deg(o.pole_lon, o.pole_lat, c.pole_lon + 180.0, c.pole_lat) < 25.0
                     for o in candidates)
        anti = any(o is not c and o.period == c.period and
                   _sep_deg(o.pole_lon, o.pole_lat, c.pole_lon + 180.0, -c.pole_lat) < 25.0
                   for o in candidates)
        others = [o for o in candidates if o is not c]
        best_other = min((o.chi2 for o in others), default=np.inf)
        feats = dict(df)
        feats.update({
            "dchi2_scaled": f * (c.chi2 - best.chi2) / s2,
            "margin_scaled": f * (best_other - c.chi2) / s2 if np.isfinite(best_other) else 50.0,
            "like_weight": c.like_weight,
            "boot_frac": c.boot_frac,
            "rank": float(ranked.index(c)),
            "period_rank": float(p_order.index(c.period)),
            "n_candidates": float(len(candidates)),
            "n_periods": float(len(p_order)),
            "basin_frac": c.n_starts / max(c.n_starts_total, 1),
            "redchi2_min": best.redchi2,
            "redchi2_ratio": c.redchi2 / best.redchi2,
            "neff_factor": f,
            "abs_lat": abs(c.pole_lat),
            "elongation": c.elongation,
            "flattening": c.flattening,
            "has_mirror": float(mirror),
            "has_antipode": float(anti),
            "sep_from_best": _sep_deg(c.pole_lon, c.pole_lat, best.pole_lon, best.pole_lat),
            "rot_coverage": float(np.median([min(s / c.period, 1.5) for s in span])),
        })
        c.features = {k: float(v) for k, v in feats.items()}


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    """Ranked candidates with probabilities, plus the diagnostics behind them."""

    candidates: List[Candidate]
    noise: Dict[str, float]
    data: Dict[str, float]
    calibrated: bool
    p_none: float = np.nan   # calibrated P(no candidate is correct), if available

    def summary(self) -> str:
        kind = ("calibrated P(correct)" if self.calibrated
                else "UNCALIBRATED likelihood x bootstrap weight")
        lines = [
            "Silhouette solution scores",
            f"  data: {int(self.data['n_points'])} pts, {int(self.data['n_lc'])} curves, "
            f"{int(self.data['n_app'])} apparitions, ecl-lon coverage "
            f"{self.data['lon_coverage']:.0f} deg, phase {self.data['phase_min']:.0f}-"
            f"{self.data['phase_max']:.0f} deg",
            f"  noise model: s^2={self.noise['s2']:.2f}, residual rho={self.noise['rho']:.2f}"
            f" -> N_eff factor {self.noise['neff_factor']:.2f}",
            f"  probability column: {kind}",
            f"  {'#':>2s} {'period':>12s} {'pole (lon, lat)':>17s} {'chi2_nu':>8s} "
            f"{'a/b~':>5s} {'b/c~':>5s} {'L-wt':>6s} {'boot':>5s} {'prob':>6s}",
        ]
        for i, c in enumerate(self.candidates):
            lines.append(
                f"  {i:2d} {c.period:12.7f} ({c.pole_lon:6.1f},{c.pole_lat:6.1f}) "
                f"{c.redchi2:8.3f} {c.elongation:5.2f} {c.flattening:5.2f} "
                f"{c.like_weight:6.3f} {c.boot_frac:5.2f} {c.probability:6.3f}")
        if self.calibrated and np.isfinite(self.p_none):
            lines.append(f"  P(none of these is correct) ~ {self.p_none:.2f}")
        return "\n".join(lines)


def combine_uncalibrated(candidates: Sequence[Candidate]) -> None:
    """Fallback probability: likelihood weight tempered by bootstrap agreement.

    ``p ∝ like_weight · (boot_frac + 1/n_boot-ish floor)`` renormalised. It is a
    sensible ranking, but not calibrated — use a trained
    :class:`silhouette.calibration.ScoreModel` for probabilities you can quote.
    """
    lw = np.array([c.like_weight for c in candidates])
    bf = np.array([c.boot_frac if np.isfinite(c.boot_frac) else 1.0 for c in candidates])
    p = lw * (bf + 0.05)
    p = p / p.sum() if p.sum() > 0 else np.full(len(candidates), 1.0 / len(candidates))
    for c, pi in zip(candidates, p):
        c.probability = float(pi)
        c.calibrated = False


def score_solutions(
    lightcurves: Sequence[LightCurveObs],
    periods: Sequence[float],
    pole_grid: Optional[Sequence[Tuple[float, float]]] = None,
    n_workers: int = 1,
    n_boot: int = 20,
    boot_max_nfev: int = 40,
    prior: Optional[Callable[[Candidate], float]] = None,
    model=None,
    seed: int = 0,
    find_kwargs: Optional[dict] = None,
    **inv_kwargs,
) -> ScoreReport:
    """Find candidate solutions and attach a probability to each.

    Parameters
    ----------
    lightcurves : the photometry (e.g. Squint output reduced to intensities,
        with Sun/Earth vectors)
    periods : trial periods — typically ``alias_periods(P_spindoc, baseline)``
    pole_grid, n_workers : passed to :func:`find_candidates`
    n_boot : bootstrap resamples for the stability layer (0 to skip)
    prior : optional :class:`PopulationPrior` (or any ``Candidate -> log p``)
    model : optional trained :class:`silhouette.calibration.ScoreModel`; without
        one the probabilities are the uncalibrated fallback
    inv_kwargs : forwarded to the inversions (``lmax``, ``n_normals``,
        ``max_nfev``, ``phot_func``, ``phot_arg``)
    """
    model_kw = {k: inv_kwargs[k] for k in ("phot_func", "phot_arg") if k in inv_kwargs}
    cands = find_candidates(lightcurves, periods, pole_grid=pole_grid,
                            n_workers=n_workers, **(find_kwargs or {}), **inv_kwargs)
    noise = likelihood_weights(cands, lightcurves, prior=prior, **model_kw)
    if n_boot > 0 and len(cands) > 1:
        bootstrap_stability(cands, lightcurves, n_boot=n_boot, seed=seed,
                            boot_max_nfev=boot_max_nfev, n_workers=n_workers, **inv_kwargs)
    else:
        for c in cands:
            c.boot_frac = 1.0 if c is cands[0] else 0.0
    dfeat = data_features(lightcurves)
    candidate_features(cands, lightcurves, noise_diag=noise, data_feats=dfeat)

    p_none = np.nan
    if model is not None:
        probs = model.predict_proba(cands)
        for c, p in zip(cands, probs):
            c.probability = float(p)
            c.calibrated = True
        # families are >= pole_tol apart, so "candidate i is correct" events are
        # (nearly) mutually exclusive and the leftover mass is "none of them"
        p_none = float(np.clip(1.0 - np.sum(probs), 0.0, 1.0))
    else:
        combine_uncalibrated(cands)
    cands.sort(key=lambda c: -c.probability)
    return ScoreReport(candidates=cands, noise=noise, data=dfeat,
                       calibrated=model is not None, p_none=p_none)


__all__ = [
    "Candidate", "PopulationPrior", "ScoreReport", "spin_vector_pole",
    "projected_areas", "shape_proxies", "alias_periods", "baseline_of",
    "find_candidates", "likelihood_weights", "residual_autocorrelation",
    "neff_factor", "noise_scale", "bootstrap_stability", "data_features", "candidate_features",
    "longitude_coverage", "combine_uncalibrated", "score_solutions", "model_flux",
]
