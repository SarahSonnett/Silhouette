"""Next-observation recommender: when would new data break the degeneracy?

Once the scorer has a short list of surviving solutions, the most useful
question is not "which is right?" but "when should I observe to find out?".
Competing solutions (mirror poles, antipodes, aliases, different elongations)
predict *different* light curves at future viewing geometries. The epochs where
those predictions diverge most are where one night of photometry does the most
work, and choosing them well is how to get a unique solution from fewer than
the canonical ≥3 apparitions.

For each future epoch the recommender predicts one full rotation for every
candidate, converts to magnitudes, removes the mean (relative photometry) and
compares candidates pairwise after the best circular phase shift — rotational
phase cannot be propagated reliably years ahead, so only the phase-invariant
light-curve *shape* and *amplitude* count. The pairwise difference becomes the
Δχ²_ij one night of ``n_points`` at ``sigma_mag`` would deliver.

Epochs are ranked by the **expected probability of the true solution after
that night**: if candidate ``i`` is true, the updated weight of each rival
``j`` is ``p_j exp(−Δχ²_ij / 2)``, so

    E[p_true] = Σ_i p_i · p_i / Σ_j p_j exp(−Δχ²_ij / 2).

This is 0–1, rewards separating the *probable* candidates, and cannot be
dominated by one improbable but very different rival. It equals the current
``Σ p_i²`` when an epoch separates nothing and approaches 1 when it separates
everything. The probability-weighted mean Δχ² is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .forward import DEFAULT_LAW, convex_lightcurve


def _unit(v):
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def phase_angle_deg(sun, earth) -> np.ndarray:
    """Solar phase angle (deg) from asteroid-centric Sun and observer vectors."""
    return np.degrees(np.arccos(np.clip(np.sum(_unit(sun) * _unit(earth), axis=1), -1, 1)))


def solar_elongation_deg(sun, earth) -> np.ndarray:
    """Sun–observer–asteroid angle (deg); needs true vectors (AU), not unit ones.

    With asteroid-centric vectors ``S`` (to Sun) and ``E`` (to observer), the
    observer sees the asteroid along ``−E`` and the Sun along ``S − E``.
    """
    s = np.atleast_2d(np.asarray(sun, dtype=float))
    e = np.atleast_2d(np.asarray(earth, dtype=float))
    to_ast = _unit(-e)
    to_sun = _unit(s - e)
    return np.degrees(np.arccos(np.clip(np.sum(to_ast * to_sun, axis=1), -1, 1)))


def predicted_rotation(candidate, sun, earth, n_phase: int = 72,
                       phot_func=DEFAULT_LAW, phot_arg=None) -> np.ndarray:
    """One rotation of the candidate's model light curve (mag, mean removed)."""
    fit = candidate.fit
    t = np.linspace(0.0, fit.period, n_phase, endpoint=False)
    f = convex_lightcurve(fit.shape, t, sun, earth, fit.pole_lon, fit.pole_lat,
                          fit.period, phi0=0.0, t0=0.0, phot_func=phot_func, arg=phot_arg)
    m = -2.5 * np.log10(np.maximum(f, 1e-30))
    return m - m.mean()


def shift_invariant_rms(a: np.ndarray, b: np.ndarray, allow_reverse: bool = False) -> float:
    """Minimum RMS difference over circular phase shifts (optionally time reversal).

    Only the phase offset is unknowable years ahead. An asymmetric profile
    (slow rise, fast fall) and its time reversal *are* distinguishable within
    one night, so reversal is not forgiven by default; ``allow_reverse=True``
    gives a deliberately conservative score.
    """
    best = np.inf
    seqs = [b, b[::-1]] if allow_reverse else [b]
    for bb in seqs:
        for k in range(a.size):
            best = min(best, float(np.sqrt(np.mean((a - np.roll(bb, k)) ** 2))))
    return best


@dataclass
class EpochScore:
    epoch: float
    phase_deg: float
    elongation_deg: float
    p_true_after: float            # expected posterior prob. of the true candidate
    expected_dchi2: float          # probability-weighted mean over candidate pairs
    top_pair_dchi2: float          # between the two most probable candidates
    amplitudes: List[float] = field(default_factory=list)   # mag, per candidate


def recommend_observations(
    candidates: Sequence,
    epochs: Sequence[float],
    sun_vecs: np.ndarray,
    earth_vecs: np.ndarray,
    probabilities: Optional[Sequence[float]] = None,
    n_points: int = 40,
    sigma_mag: float = 0.02,
    model_sigma_mag: float = 0.0,
    min_elongation_deg: float = 60.0,
    max_phase_deg: Optional[float] = None,
    n_phase: int = 72,
) -> List[EpochScore]:
    """Rank future epochs by how well one night would separate the candidates.

    Parameters
    ----------
    candidates : scorer candidates (anything with ``.fit`` and ``.probability``)
    epochs, sun_vecs, earth_vecs : future epochs and asteroid-centric ecliptic
        vectors in AU (e.g. from :func:`horizons_geometry`)
    probabilities : override the candidates' own ``probability`` values
    n_points, sigma_mag : what one night of data would look like
    model_sigma_mag : how well the candidates reproduce the *existing* data
        (``ScoreReport.noise['model_rms_mag']``); added in quadrature, because
        predictions are no better than the fits they come from
    min_elongation_deg, max_phase_deg : observability cuts (elongation needs
        true AU vectors; if the vectors are unit length it is skipped)

    Returns epochs sorted best first (by ``p_true_after``, then mean Δχ²).
    """
    sun_vecs = np.atleast_2d(np.asarray(sun_vecs, dtype=float))
    earth_vecs = np.atleast_2d(np.asarray(earth_vecs, dtype=float))
    p = np.asarray(probabilities if probabilities is not None
                   else [c.probability for c in candidates], dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = p / p.sum() if p.sum() > 0 else np.full(len(candidates), 1.0 / len(candidates))
    order = np.argsort(p)[::-1]
    i1, i2 = (int(order[0]), int(order[1])) if len(order) > 1 else (0, 0)

    alpha = phase_angle_deg(sun_vecs, earth_vecs)
    unit_like = np.allclose(np.linalg.norm(earth_vecs, axis=1), 1.0, atol=1e-6)
    elong = (np.full(alpha.size, np.nan) if unit_like
             else solar_elongation_deg(sun_vecs, earth_vecs))

    sig2 = sigma_mag ** 2 + model_sigma_mag ** 2
    out: List[EpochScore] = []
    for ep, s, e, a, el in zip(epochs, sun_vecs, earth_vecs, alpha, elong):
        if np.isfinite(el) and el < min_elongation_deg:
            continue
        if max_phase_deg is not None and a > max_phase_deg:
            continue
        curves = [predicted_rotation(c, s, e, n_phase=n_phase) for c in candidates]
        n = len(curves)
        dchi = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = shift_invariant_rms(curves[i], curves[j])
                dchi[i, j] = dchi[j, i] = n_points * d * d / sig2
        w = np.outer(p, p)
        iu = np.triu_indices(n, 1)
        den = float(w[iu].sum())
        top = float(dchi[i1, i2]) if n > 1 else 0.0
        p_after = float(np.sum(p * p / (np.exp(-0.5 * dchi) @ p)))
        out.append(EpochScore(epoch=float(ep), phase_deg=float(a),
                              elongation_deg=float(el), p_true_after=p_after,
                              expected_dchi2=float((w * dchi)[iu].sum()) / den if den > 0 else 0.0,
                              top_pair_dchi2=top,
                              amplitudes=[float(np.ptp(cv)) for cv in curves]))
    out.sort(key=lambda r: (-round(r.p_true_after, 3), -r.expected_dchi2))
    return out


def horizons_geometry(target: str, start: str, stop: str, step: str = "5d"
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asteroid-centric ecliptic Sun and Earth vectors (AU) from JPL Horizons.

    Returns ``(jd, sun_vecs, earth_vecs)``. Needs ``astroquery`` and network
    access. ``start``/``stop`` are Horizons date strings (``"2027-01-01"``).
    """
    from astroquery.jplhorizons import Horizons

    epochs = {"start": start, "stop": stop, "step": step}
    helio = Horizons(id=target, location="@sun", epochs=epochs,
                     id_type="smallbody").vectors(refplane="ecliptic")
    geo = Horizons(id=target, location="500@399", epochs=epochs,
                   id_type="smallbody").vectors(refplane="ecliptic")
    jd = np.asarray(helio["datetime_jd"], dtype=float)
    r_h = np.column_stack([np.asarray(helio[k], dtype=float) for k in ("x", "y", "z")])
    r_g = np.column_stack([np.asarray(geo[k], dtype=float) for k in ("x", "y", "z")])
    return jd, -r_h, -r_g


def simple_orbit_geometry(epochs_days: Sequence[float], a_au: float = 2.7,
                          incl_deg: float = 8.0, node_deg: float = 0.0,
                          lon0_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate asteroid-centric Sun/Earth vectors (AU) for circular orbits.

    Earth on a 1 AU circle in the ecliptic; the asteroid on an ``a_au`` circle
    inclined by ``incl_deg`` about the line of nodes ``node_deg``; ``lon0_deg``
    is the asteroid's longitude in its orbit at day 0. Good enough for demos,
    tests and rough planning — use :func:`horizons_geometry` for real targets.
    """
    t = np.asarray(epochs_days, dtype=float)
    th_e = 2.0 * np.pi * t / 365.25
    r_e = np.column_stack([np.cos(th_e), np.sin(th_e), np.zeros_like(t)])
    th_a = np.radians(lon0_deg) + 2.0 * np.pi * t / (365.25 * a_au ** 1.5)
    x = a_au * np.cos(th_a)
    y = a_au * np.sin(th_a)
    i, om = np.radians(incl_deg), np.radians(node_deg)
    # rotate the orbit plane: inclination about the x axis, then the node about z
    xi, yi, zi = x, y * np.cos(i), y * np.sin(i)
    r_a = np.column_stack([xi * np.cos(om) - yi * np.sin(om),
                           xi * np.sin(om) + yi * np.cos(om), zi])
    return -r_a, r_e - r_a


def format_recommendations(recs: Sequence[EpochScore], top: int = 10,
                           epoch_fmt=lambda e: f"{e:.1f}") -> str:
    lines = [f"{'epoch':>14s} {'phase':>6s} {'elong':>6s} {'E[p_true]':>9s} {'E[dchi2]':>9s} "
             f"{'top-2 dchi2':>11s}  amplitudes (mag)"]
    for r in recs[:top]:
        amps = " ".join(f"{a:.2f}" for a in r.amplitudes)
        el = f"{r.elongation_deg:6.0f}" if np.isfinite(r.elongation_deg) else "     -"
        lines.append(f"{epoch_fmt(r.epoch):>14s} {r.phase_deg:6.1f} {el} "
                     f"{r.p_true_after:9.3f} {r.expected_dchi2:9.1f} "
                     f"{r.top_pair_dchi2:11.1f}  {amps}")
    return "\n".join(lines)


__all__ = ["phase_angle_deg", "solar_elongation_deg", "predicted_rotation",
           "shift_invariant_rms", "EpochScore", "recommend_observations",
           "horizons_geometry", "simple_orbit_geometry", "format_recommendations"]
