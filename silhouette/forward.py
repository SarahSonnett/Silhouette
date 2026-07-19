"""Forward light-curve model for convex shapes.

For a **convex** body the disk-integrated brightness is a plain weighted sum
over the Gaussian image — no ray tracing, z-buffer, or visibility test is
needed, because convexity guarantees a surface element is visible iff its normal
faces the observer and illuminated iff it faces the Sun::

    L = Σ_i a_i · S(μ0_i, μ_i, α)      over facets with μ_i > 0 and μ0_i > 0

with ``μ_i = n_i·Ê`` (cos emission), ``μ0_i = n_i·Ŝ`` (cos incidence) evaluated
in the **body-fixed frame**. This makes the forward model cheap (two
matrix-vector products per epoch) and smooth in the shape parameters.

Scattering convention
---------------------
Functions here take SpotLight's signature ``f(mu0, mu, alpha, arg)`` so laws are
interchangeable *in form*, but Silhouette's kernels include the **μ projection
factor**, because it sums over facets of true area ``a_i`` rather than over
image pixels (where projection is already baked into the pixel grid). So

* Silhouette ``lommel_seeliger`` = ``μ μ0 / (μ + μ0)``   (facet flux per unit area)
* SpotLight ``lommel_seeliger``  = ``μ0 μ / (μ0 + μ)``   (per-pixel I/F)

which coincide, whereas Silhouette's ``lambert`` = ``μ μ0`` carries an extra μ
relative to SpotLight's ``lambertian`` = ``μ0``. Keep this in mind when porting
a law between the two packages.

The default is Lommel–Seeliger + Lambert, the standard combination for convex
light-curve inversion (Kaasalainen & Torppa 2001).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .shapes import ConvexShape


# ---------------------------------------------------------------------------
# Scattering laws  —  f(mu0, mu, alpha, arg) -> facet flux per unit area
# ---------------------------------------------------------------------------

def geometric(mu0, mu, alpha=0.0, arg=None):
    """Pure projected area (no limb darkening): ``S = μ``.

    Summing this over illuminated, visible facets returns the body's projected
    area exactly, which is the assumption behind the analytical amplitude–aspect
    relations in :mod:`silhouette.model`. Useful for cross-validation.
    """
    return mu


def lambert(mu0, mu, alpha=0.0, arg=None):
    """Lambertian facet flux: ``S = μ μ0``."""
    return mu * mu0


def lommel_seeliger(mu0, mu, alpha=0.0, arg=None):
    """Lommel–Seeliger facet flux: ``S = μ μ0 / (μ + μ0)`` (0 where μ+μ0 = 0)."""
    denom = mu + mu0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0.0, mu * mu0 / denom, 0.0)


def ls_lambert(mu0, mu, alpha=0.0, arg=0.1):
    """Lommel–Seeliger + Lambert: ``S = μμ0/(μ+μ0) + c·μμ0``.

    ``arg`` is the Lambert weight ``c`` (Kaasalainen's convex-inversion default
    is a small value, ~0.1). This is the package default.
    """
    c = 0.1 if arg is None else float(arg)
    return lommel_seeliger(mu0, mu, alpha) + c * lambert(mu0, mu, alpha)


SCATTERING_LAWS = {
    "geometric": geometric,
    "lambert": lambert,
    "lommel_seeliger": lommel_seeliger,
    "ls_lambert": ls_lambert,
}

DEFAULT_LAW = ls_lambert


def phase_function(alpha, a0: float = 0.5, d: float = 0.1, k: float = -0.005):
    """Empirical phase function ``f(α) = a0·exp(−α/d) + k·α + 1``.

    The standard single-parameter-set form used alongside the LS+Lambert law in
    convex inversion; ``α`` in radians. Returns 1.0 for the default of no
    correction if ``a0 = k = 0``.
    """
    return a0 * np.exp(-alpha / d) + k * alpha + 1.0


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def ecliptic_to_body_matrix(pole_lon: float, pole_lat: float, phi: float) -> np.ndarray:
    """Rotation taking ecliptic vectors into the body-fixed frame.

    ``R = R_z(φ) · R_y(β_p − π/2) · R_z(−λ_p)``, which maps the spin-pole
    direction ``(cosβ cosλ, cosβ sinλ, sinβ)`` onto the body ``+z`` axis for any
    rotation phase ``φ``.
    """
    return _rz(phi) @ _ry(np.radians(pole_lat) - np.pi / 2.0) @ _rz(-np.radians(pole_lon))


def rotation_phase(times, period: float, phi0: float = 0.0, t0: Optional[float] = None):
    """Rotation phase (radians) at ``times`` for a given period (same time units)."""
    times = np.asarray(times, dtype=float)
    t0 = float(times[0]) if t0 is None else float(t0)
    return phi0 + 2.0 * np.pi * (times - t0) / period


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def convex_brightness(shape: ConvexShape, sun_body, earth_body,
                      phot_func: Callable = DEFAULT_LAW, arg=None,
                      alpha: float = 0.0) -> float:
    """Disk-integrated brightness for one geometry, in the **body frame**.

    ``sun_body`` / ``earth_body`` are directions from the body toward the Sun and
    observer, expressed in body-fixed coordinates (they are normalised here).
    """
    s = _unit(sun_body)
    e = _unit(earth_body)
    mu0 = shape.normals @ s
    mu = shape.normals @ e
    lit = (mu > 0.0) & (mu0 > 0.0)
    if not np.any(lit):
        return 0.0
    contrib = phot_func(mu0[lit], mu[lit], alpha, arg)
    return float(np.sum(shape.areas[lit] * contrib))


def convex_lightcurve(shape: ConvexShape, times, sun_vecs, earth_vecs,
                      pole_lon: float, pole_lat: float, period: float,
                      phi0: float = 0.0, t0: Optional[float] = None,
                      phot_func: Callable = DEFAULT_LAW, arg=None,
                      phase_func: Optional[Callable] = None) -> np.ndarray:
    """Model intensities at ``times`` for the given spin state and geometry.

    Parameters
    ----------
    shape : ConvexShape
    times : (N,) epochs (light-time corrected), same units as ``period``
    sun_vecs, earth_vecs : (N, 3) asteroid-centric **ecliptic** vectors toward
        the Sun and observer (magnitudes ignored; DAMIT files supply these
        directly)
    pole_lon, pole_lat : spin pole ecliptic longitude/latitude, degrees
    period : rotation period, in the units of ``times``
    phi0, t0 : rotation phase zero point and its epoch
    phot_func, arg : scattering law and its parameter
    phase_func : optional callable applied to the solar phase angle
    """
    times = np.asarray(times, dtype=float)
    sun_vecs = np.atleast_2d(np.asarray(sun_vecs, dtype=float))
    earth_vecs = np.atleast_2d(np.asarray(earth_vecs, dtype=float))
    if sun_vecs.shape[0] == 1 and times.size > 1:
        sun_vecs = np.repeat(sun_vecs, times.size, axis=0)
    if earth_vecs.shape[0] == 1 and times.size > 1:
        earth_vecs = np.repeat(earth_vecs, times.size, axis=0)

    s_hat = _unit(sun_vecs)
    e_hat = _unit(earth_vecs)
    alphas = np.arccos(np.clip(np.sum(s_hat * e_hat, axis=1), -1.0, 1.0))
    phis = rotation_phase(times, period, phi0, t0)

    # Vectorised over epochs: the pole part of the rotation is constant, so
    # apply it once and fold the per-epoch spin R_z(phi) in analytically.
    pole_rot = _ry(np.radians(pole_lat) - np.pi / 2.0) @ _rz(-np.radians(pole_lon))
    cos_p, sin_p = np.cos(phis), np.sin(phis)

    def _spin(vecs):
        u = vecs @ pole_rot.T
        return np.column_stack([u[:, 0] * cos_p - u[:, 1] * sin_p,
                                u[:, 0] * sin_p + u[:, 1] * cos_p,
                                u[:, 2]])

    mu0 = shape.normals @ _spin(s_hat).T          # (n_normals, n_epochs)
    mu = shape.normals @ _spin(e_hat).T
    lit = (mu > 0.0) & (mu0 > 0.0)

    contrib = phot_func(np.where(lit, mu0, 0.0), np.where(lit, mu, 0.0),
                        alphas[None, :], arg)
    out = np.sum(shape.areas[:, None] * np.where(lit, contrib, 0.0), axis=0)

    if phase_func is not None:
        out = out * phase_func(alphas)
    return out


def projected_area(shape: ConvexShape, earth_body) -> float:
    """Projected area of a convex body along ``earth_body`` (body frame).

    Uses the convex identity ``A_proj = Σ_{μ>0} a_i μ_i``; equivalently
    ``½ Σ a_i |μ_i|``.
    """
    mu = shape.normals @ _unit(earth_body)
    return float(np.sum(shape.areas[mu > 0.0] * mu[mu > 0.0]))


__all__ = [
    "geometric", "lambert", "lommel_seeliger", "ls_lambert",
    "SCATTERING_LAWS", "DEFAULT_LAW", "phase_function",
    "ecliptic_to_body_matrix", "rotation_phase",
    "convex_brightness", "convex_lightcurve", "projected_area",
]
