"""Rotational-stability constraints on density and cohesion (Phase 2).

Given a shape (DEEVE axis ratios) and a spin period, ask what a body must be
made of to survive its own rotation. Following Holsapple (2001, 2004, 2007), the
body is treated as a homogeneous, self-gravitating triaxial ellipsoid of a
Drucker–Prager (cohesive Mohr–Coulomb) material.

Method
------
For a uniformly rotating homogeneous ellipsoid the internal stress field is
known in closed form. Its **volume-averaged** normal stresses are

    <s_xx> = -(1/5) rho a^2 (2 pi G rho A_1 - omega^2)
    <s_yy> = -(1/5) rho b^2 (2 pi G rho A_2 - omega^2)
    <s_zz> = -(1/5) rho c^2 (2 pi G rho A_3)

(compression negative; spin about the short ``c`` axis), where ``A_i`` are the
standard ellipsoid shape integrals satisfying ``A_1 + A_2 + A_3 = 2``. Applying
the Drucker–Prager criterion to these averages gives a limit analysis: the body
is stable while

    sqrt(J_2) <= k - s * I_1

with ``I_1`` the first stress invariant, ``J_2`` the second deviatoric invariant,
and ``(s, k)`` fixed by the friction angle and cohesion.

Two regimes, deliberately separated
-----------------------------------
* **Cohesionless stability is size-independent.** Every stress term scales as
  ``a^2``, so the zero-cohesion condition depends only on shape, friction angle,
  and the dimensionless spin ``omega^2 / (G rho)``. That yields a *minimum bulk
  density* for a strengthless rubble pile from the light curve alone — no
  diameter needed.
* **Cohesion in pascals does need a size**, because a stress scales as
  ``rho^2 G R^2``. Supply a diameter (e.g. from WISE/NEATM) for that.

Uncertainty
-----------
Light-curve inversion determines ``b/c`` far more poorly than ``a/b`` (the
c-axis is constrained only through aspect diversity; ~20% errors are typical
even with a well-determined pole). The functions here therefore accept ranges
and provide :func:`propagate_axis_uncertainty`, so results are reported as
intervals rather than misleadingly precise point values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

G = 6.67430e-11          # gravitational constant, m^3 kg^-1 s^-2


# ---------------------------------------------------------------------------
# Ellipsoid geometry
# ---------------------------------------------------------------------------

def axes_from_ratios(ab: float, bc: float, r_eq: float = 1.0):
    """Semi-axes ``(a, b, c)`` from ``a/b`` and ``b/c`` at equivalent radius ``r_eq``.

    Scaled so ``a·b·c = r_eq³`` (equal volume to a sphere of radius ``r_eq``).
    """
    c = r_eq / (ab * bc * bc) ** (1.0 / 3.0)
    b = bc * c
    a = ab * b
    return a, b, c


def shape_factors(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Ellipsoid shape integrals ``A_i`` (dimensionless, summing to 2).

    ``A_i = a b c ∫_0^inf du / [(a_i² + u) sqrt((a²+u)(b²+u)(c²+u))]``

    The integral is scale-invariant (substituting ``u -> λ²u`` leaves it
    unchanged), so the axes are normalised to order unity first. This is not
    cosmetic: with axes in metres the integrand decays on a scale of ``a²``
    (~10⁹ for a 50 km body) and adaptive quadrature silently returns a result
    that is orders of magnitude too small.
    """
    scale_len = (a * b * c) ** (1.0 / 3.0)
    an, bn, cn = a / scale_len, b / scale_len, c / scale_len

    def integrand(u, ai):
        return 1.0 / ((ai * ai + u)
                      * np.sqrt((an * an + u) * (bn * bn + u) * (cn * cn + u)))

    prod = an * bn * cn
    out = []
    for ai in (an, bn, cn):
        val, _ = quad(integrand, 0.0, np.inf, args=(ai,), limit=200)
        out.append(prod * val)
    return tuple(out)


# ---------------------------------------------------------------------------
# Stress state
# ---------------------------------------------------------------------------

def mean_stresses(a, b, c, rho: float, omega: float):
    """Volume-averaged normal stresses ``(s_xx, s_yy, s_zz)`` in Pa.

    Compression is negative. Spin ``omega`` (rad/s) is about the short ``c`` axis.
    """
    a1, a2, a3 = shape_factors(a, b, c)
    k = rho / 5.0
    sxx = -k * a * a * (2.0 * np.pi * G * rho * a1 - omega ** 2)
    syy = -k * b * b * (2.0 * np.pi * G * rho * a2 - omega ** 2)
    szz = -k * c * c * (2.0 * np.pi * G * rho * a3)
    return sxx, syy, szz


def _invariants(sxx, syy, szz):
    i1 = sxx + syy + szz
    m = i1 / 3.0
    j2 = 0.5 * ((sxx - m) ** 2 + (syy - m) ** 2 + (szz - m) ** 2)
    return i1, j2


def drucker_prager_constants(friction_deg: float, cohesion: float = 0.0):
    """``(s, k)`` matching Mohr–Coulomb at the compressive meridian."""
    phi = np.radians(friction_deg)
    denom = np.sqrt(3.0) * (3.0 - np.sin(phi))
    s = 2.0 * np.sin(phi) / denom
    k = 6.0 * cohesion * np.cos(phi) / denom
    return s, k


# ---------------------------------------------------------------------------
# Stability and required strength
# ---------------------------------------------------------------------------

def required_cohesion(ab: float, bc: float, period_h: float, rho: float,
                      diameter_km: float, friction_deg: float = 35.0) -> float:
    """Minimum cohesion (Pa) needed for stability. Zero if none is required.

    ``diameter_km`` is the volume-equivalent diameter; cohesion scales as its
    square, so this is where an absolute size is required.
    """
    r_eq = 0.5 * diameter_km * 1e3
    a, b, c = axes_from_ratios(ab, bc, r_eq)
    omega = 2.0 * np.pi / (period_h * 3600.0)
    i1, j2 = _invariants(*mean_stresses(a, b, c, rho, omega))
    s, _ = drucker_prager_constants(friction_deg)
    # stable when sqrt(J2) <= k - s*I1 ; solve the deficit for the cohesion term
    deficit = np.sqrt(j2) + s * i1
    if deficit <= 0.0:
        return 0.0
    phi = np.radians(friction_deg)
    return float(deficit * np.sqrt(3.0) * (3.0 - np.sin(phi)) / (6.0 * np.cos(phi)))


def is_stable_cohesionless(ab: float, bc: float, period_h: float, rho: float,
                           friction_deg: float = 35.0) -> bool:
    """Can this shape/spin/density hold together with **no** cohesion?

    Size-independent: every stress term scales identically with the body's size.
    """
    a, b, c = axes_from_ratios(ab, bc, 1.0)          # scale-free
    omega = 2.0 * np.pi / (period_h * 3600.0)
    i1, j2 = _invariants(*mean_stresses(a, b, c, rho, omega))
    s, _ = drucker_prager_constants(friction_deg)
    return bool(np.sqrt(j2) <= -s * i1)


def min_density_cohesionless(ab: float, bc: float, period_h: float,
                             friction_deg: float = 35.0,
                             rho_bounds: Tuple[float, float] = (10.0, 2.0e4)
                             ) -> Optional[float]:
    """Smallest bulk density (kg/m³) that survives this spin with no cohesion.

    This is the headline light-curve-only result: if the answer exceeds any
    plausible density for the taxonomic type, the body **must** have cohesion or
    be monolithic. Returns ``None`` if even the upper bound is unstable.
    """
    lo, hi = rho_bounds
    if is_stable_cohesionless(ab, bc, period_h, lo, friction_deg):
        return float(lo)
    if not is_stable_cohesionless(ab, bc, period_h, hi, friction_deg):
        return None

    def f(rho):
        a, b, c = axes_from_ratios(ab, bc, 1.0)
        omega = 2.0 * np.pi / (period_h * 3600.0)
        i1, j2 = _invariants(*mean_stresses(a, b, c, rho, omega))
        s, _ = drucker_prager_constants(friction_deg)
        return np.sqrt(j2) + s * i1          # < 0 once stable

    return float(brentq(f, lo, hi, xtol=1e-3))


def shedding_limit_density(period_h: float) -> float:
    """Classic spin-barrier density for a strengthless **sphere**: ``3π/(G P²)``.

    A surface particle at the equator stays attached only above this density.
    Provided for context alongside the shape-aware Drucker–Prager limits.
    """
    p = period_h * 3600.0
    return float(3.0 * np.pi / (G * p * p))


# ---------------------------------------------------------------------------
# Uncertainty propagation
# ---------------------------------------------------------------------------

@dataclass
class DensityConstraint:
    """Minimum cohesionless density over a range of plausible shapes."""

    best: Optional[float]
    low: Optional[float]
    high: Optional[float]
    friction_deg: float
    period_h: float

    def summary(self) -> str:
        def fmt(v):
            return "unstable at any density" if v is None else f"{v:.0f} kg/m^3"
        return (f"minimum cohesionless density (phi={self.friction_deg:.0f} deg, "
                f"P={self.period_h:.4f} h): {fmt(self.best)} "
                f"[range {fmt(self.low)} - {fmt(self.high)}]")


def propagate_axis_uncertainty(ab: float, bc: float, period_h: float,
                               ab_frac: float = 0.05, bc_frac: float = 0.20,
                               friction_deg: float = 35.0) -> DensityConstraint:
    """Minimum cohesionless density with axis-ratio uncertainties folded in.

    Defaults reflect what light-curve inversion actually delivers: ``a/b`` to a
    few percent, ``b/c`` only to ~20%. The extremes of the axis-ratio box are
    scanned, since the dependence is monotonic in neither ratio in general.
    """
    vals = []
    for fa in (1.0 - ab_frac, 1.0, 1.0 + ab_frac):
        for fb in (1.0 - bc_frac, 1.0, 1.0 + bc_frac):
            rho = min_density_cohesionless(max(ab * fa, 1.0), max(bc * fb, 1.0),
                                           period_h, friction_deg)
            vals.append(rho)
    best = min_density_cohesionless(ab, bc, period_h, friction_deg)
    finite = [v for v in vals if v is not None]
    return DensityConstraint(
        best=best,
        low=min(finite) if finite else None,
        high=max(finite) if finite else None,
        friction_deg=friction_deg, period_h=period_h)


__all__ = [
    "G", "axes_from_ratios", "shape_factors", "mean_stresses",
    "drucker_prager_constants", "required_cohesion", "is_stable_cohesionless",
    "min_density_cohesionless", "shedding_limit_density",
    "DensityConstraint", "propagate_axis_uncertainty",
]
