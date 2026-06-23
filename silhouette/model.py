"""Analytical triaxial-ellipsoid light-curve relations.

A rotating triaxial ellipsoid with semi-axes ``a >= b >= c`` (spinning about the
short ``c`` axis) is observed at *aspect angle* ``theta`` — the angle between the
line of sight and the spin axis. Under geometric scattering the disk-integrated
brightness is proportional to the projected area, which gives two closed-form
observables:

Amplitude (peak-to-trough, mag)::

    A(theta) = 2.5 log10(a/b)
             - 1.25 log10( (a^2 cos^2 th + c^2 sin^2 th)
                          / (b^2 cos^2 th + c^2 sin^2 th) )

Mean magnitude — from the rotation-averaged projected area
``<S(theta)>``, which brightens toward pole-on (the full a*b face) and fades
toward equator-on. Only the *relative* variation with aspect carries shape
information, so an additive zero point absorbs the absolute scale.

All axis ratios are parameterised as ``p = a/b >= 1`` and ``q = b/c >= 1`` with
``b == 1`` chosen as the working unit (so ``a = p``, ``c = 1/q``).
"""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Rotation samples used to average the projected area over one spin.
_PHI = np.linspace(0.0, 2.0 * np.pi, 181)


def aspect_angle(ecl_lon, ecl_lat, pole_lon, pole_lat):
    """Aspect angle (radians) between the line of sight and the spin axis.

    ``cos(theta) = sin(beta) sin(beta_p) + cos(beta) cos(beta_p) cos(lam - lam_p)``

    All inputs are in degrees; arrays broadcast elementwise.
    """
    b = np.radians(ecl_lat)
    bp = np.radians(pole_lat)
    dlon = np.radians(np.asarray(ecl_lon) - pole_lon)
    cos_t = np.sin(b) * np.sin(bp) + np.cos(b) * np.cos(bp) * np.cos(dlon)
    return np.arccos(np.clip(cos_t, -1.0, 1.0))


def axes_from_ratios(p: float, q: float):
    """Return semi-axes ``(a, b, c)`` for ratios ``p = a/b`` and ``q = b/c`` (b=1)."""
    return float(p), 1.0, 1.0 / float(q)


def amplitude_model(p, q, theta):
    """Light-curve amplitude (mag) for ratios ``p, q`` at aspect ``theta`` (rad)."""
    a, b, c = float(p), 1.0, 1.0 / float(q)
    ct2 = np.cos(theta) ** 2
    st2 = np.sin(theta) ** 2
    num = a * a * ct2 + c * c * st2
    den = b * b * ct2 + c * c * st2
    return 2.5 * np.log10(a / b) - 1.25 * np.log10(num / den)


def mean_projected_area(p, q, theta):
    """Rotation-averaged projected area ``<S>`` (units of ``pi``, b=1).

    The instantaneous projected area of the ellipsoid at aspect ``theta`` and
    rotation phase ``phi`` is

        S = pi * sqrt( c^2 sin^2(th) (a^2 sin^2(phi) + b^2 cos^2(phi))
                       + a^2 b^2 cos^2(th) )

    averaged here over a full rotation.
    """
    a, b, c = float(p), 1.0, 1.0 / float(q)
    theta = np.asarray(theta, dtype=float)
    st2 = np.sin(theta) ** 2
    ct2 = np.cos(theta) ** 2
    phi = _PHI
    # shape (theta, phi) via broadcasting
    inner = (a * a * np.sin(phi) ** 2 + b * b * np.cos(phi) ** 2)
    s = np.sqrt(c * c * st2[..., None] * inner + a * a * b * b * ct2[..., None])
    return _trapz(s, phi, axis=-1) / (2.0 * np.pi)


def mean_mag_model(p, q, theta, zero_point=0.0):
    """Relative mean magnitude for ratios ``p, q`` at aspect ``theta`` (rad).

    ``M = zero_point - 2.5 log10(<S(theta)>)``. The zero point is a free
    nuisance parameter in the fit that absorbs the absolute brightness scale.
    """
    return zero_point - 2.5 * np.log10(mean_projected_area(p, q, theta))


def ab_lower_bound(amplitude: float) -> float:
    """Minimum ``a/b`` consistent with an observed amplitude (equatorial aspect).

    Largest amplitude occurs at ``theta = 90 deg`` where ``A = 2.5 log10(a/b)``,
    so any single amplitude implies ``a/b >= 10^(A/2.5)``.
    """
    return float(10.0 ** (amplitude / 2.5))


def mirror_pole(pole_lon: float, pole_lat: float):
    """Return the indistinguishable mirror pole ``(lon+180 mod 360, -lat)``.

    Amplitude and mean-magnitude depend only on ``sin^2`` / ``cos^2`` of the
    aspect, so ``(lam_p, beta_p)`` and ``(lam_p+180, -beta_p)`` are exactly
    degenerate without epoch/timing information.
    """
    return (pole_lon + 180.0) % 360.0, -pole_lat


__all__ = [
    "aspect_angle",
    "axes_from_ratios",
    "amplitude_model",
    "mean_projected_area",
    "mean_mag_model",
    "ab_lower_bound",
    "mirror_pole",
]
