"""Shape representations for light-curve inversion.

The convex representation follows Kaasalainen & Torppa (2001, Icarus 153, 24).
A convex body is described by its **Gaussian image**: for every unit normal
direction ``n_i`` on a discretised sphere, an area weight ``a_i > 0`` giving the
surface area whose outward normal points along ``n_i``. Two properties make this
the natural inversion variable:

* **Positivity is built in.** Writing ``a_i = exp(Σ c_lm Y_lm(n_i)) dω_i`` keeps
  every weight positive for any real coefficients, which removes the
  ill-posedness of the inverse problem.
* **Brightness is a plain weighted sum.** For a *convex* body a facet is visible
  iff ``n·E > 0`` and illuminated iff ``n·S > 0`` — no ray tracing or z-buffer
  is needed. Disk-integrated brightness is just
  ``Σ_i a_i · scattering(μ_i, μ0_i)`` over those facets. (The scattering law
  itself lives in ``forward.py``; this module stays purely geometric.)

Recovering an actual polyhedron from the Gaussian image is the **Minkowski
problem**, solved here variationally. That polyhedron gives the volume and
inertia tensor, and hence the **DEEVE** (dynamically-equivalent equal-volume
ellipsoid) whose axis ratios feed the rotational-stability geophysics.

Note on convexity: a convex model cannot represent concavities and therefore
slightly *overestimates* volume, biasing DEEVE ratios rounder. This is inherent
to the method and should be reported alongside results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import lpmv


# ---------------------------------------------------------------------------
# Sphere discretisation
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` near-uniformly spaced unit vectors and their solid angles.

    Returns ``(normals, solid_angles)`` with ``normals`` shape ``(n, 3)`` and
    every solid angle equal to ``4π/n``.
    """
    if n < 4:
        raise ValueError("need at least 4 directions to bound a volume")
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n                      # uniform in z -> equal-area bands
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i       # golden-angle spiral
    normals = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    solid = np.full(n, 4.0 * np.pi / n)
    return normals, solid


# ---------------------------------------------------------------------------
# Real spherical harmonics
# ---------------------------------------------------------------------------

def sh_degree_order(lmax: int) -> List[Tuple[int, int]]:
    """``[(l, m), ...]`` for a real SH expansion up to ``lmax`` (m negative = sine)."""
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


def real_sph_harm_basis(lmax: int, normals: np.ndarray) -> np.ndarray:
    """Design matrix of real spherical harmonics evaluated at ``normals``.

    Returns shape ``(len(normals), (lmax+1)**2)``, column order given by
    :func:`sh_degree_order`.
    """
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    ct = np.clip(z, -1.0, 1.0)
    phi = np.arctan2(y, x)

    cols = []
    for l, m in sh_degree_order(lmax):
        am = abs(m)
        # scipy's lpmv includes the Condon-Shortley phase.
        p = lpmv(am, l, ct)
        norm = np.sqrt((2 * l + 1) / (4 * np.pi)
                       * _factorial_ratio(l - am, l + am))
        if m == 0:
            cols.append(norm * p)
        elif m > 0:
            cols.append(np.sqrt(2.0) * norm * p * np.cos(am * phi))
        else:
            cols.append(np.sqrt(2.0) * norm * p * np.sin(am * phi))
    return np.column_stack(cols)


def _factorial_ratio(a: int, b: int) -> float:
    """``a! / b!`` computed stably for the SH normalisation."""
    from math import lgamma, exp
    return exp(lgamma(a + 1) - lgamma(b + 1))


# ---------------------------------------------------------------------------
# Analytic Gaussian image of an ellipsoid (reference / initialisation)
# ---------------------------------------------------------------------------

def ellipsoid_support(normals: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Support function ``h(n) = sqrt(a²n_x² + b²n_y² + c²n_z²)`` of an ellipsoid."""
    return np.sqrt((normals ** 2) @ np.array([a * a, b * b, c * c]))


def ellipsoid_gaussian_image(normals: np.ndarray, a: float, b: float, c: float,
                             solid_angles: np.ndarray) -> np.ndarray:
    """Discretised Gaussian image (area weights) of a triaxial ellipsoid.

    The reciprocal Gaussian curvature of an ellipsoid as a function of surface
    normal is ``1/K = (abc)² / h(n)⁴``; multiplying by the solid angle of each
    direction gives its share of surface area. (Sphere check: ``1/K = R²`` and
    the weights sum to ``4πR²``.)
    """
    h = ellipsoid_support(normals, a, b, c)
    return (a * b * c) ** 2 / h ** 4 * solid_angles


# ---------------------------------------------------------------------------
# Polytope geometry from support values
# ---------------------------------------------------------------------------

def polytope_from_support(normals: np.ndarray, h: np.ndarray):
    """Build ``P(h) = {x : n_i·x ≤ h_i}`` and return geometry.

    Returns ``(vertices, hull, volume, facet_areas)`` where ``facet_areas[i]`` is
    the area of the face lying on plane ``i`` (zero if that plane is redundant).
    Requires ``h > 0`` so the origin is interior.
    """
    halfspaces = np.hstack([normals, -h[:, None]])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    pts = np.asarray(hs.intersections)
    hull = ConvexHull(pts)

    # Attribute each hull triangle's area to the input plane it lies on, matched
    # by outward normal (robust against vertex-coincidence tolerances).
    owner = np.argmax(hull.equations[:, :3] @ normals.T, axis=1)
    areas = np.zeros(len(normals))
    for j, simplex in enumerate(hull.simplices):
        p0, p1, p2 = pts[simplex]
        areas[owner[j]] += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

    return pts, hull, float(hull.volume), areas


def minkowski_support(normals: np.ndarray, areas: np.ndarray,
                      h0: Optional[np.ndarray] = None,
                      maxiter: int = 400) -> np.ndarray:
    """Solve the Minkowski problem: support values ``h`` reproducing ``areas``.

    Solved as a **convex program**: minimise the linear objective ``Σ a_i h_i``
    subject to ``V(h)^(1/3) ≥ 1``. By the Brunn–Minkowski inequality ``V^(1/3)``
    is concave in ``h``, so the feasible set is convex and the minimiser is the
    sought body (Minkowski's first inequality attains equality only for
    homothets). Using ``∂V/∂h_i = A_i`` supplies exact constraint gradients.

    Two gauges are fixed afterwards:

    * **Scale** — ``h → t·h`` scales facet areas by ``t²``; ``t`` is chosen by
      least squares so the areas match ``areas``.
    * **Translation** — the Minkowski solution is unique only up to translation,
      so the body is recentred on its centre of mass (``h → h − n·d``). This
      matters physically: the inertia tensor must be taken about the centre of
      mass, not an arbitrary origin.
    """
    n = len(normals)
    if h0 is None:
        radius = np.sqrt(areas.sum() / (4.0 * np.pi))   # area-matched sphere
        h0 = np.full(n, radius)
        try:
            _, _, v0, _ = polytope_from_support(normals, h0)
            h0 = h0 / v0 ** (1.0 / 3.0)                 # start on the constraint
        except Exception:
            pass

    cache: dict = {}

    def geom(h):
        key = h.tobytes()
        if key not in cache:
            try:
                cache[key] = polytope_from_support(normals, h)[2:]
            except Exception:
                cache[key] = (1e-12, np.zeros(n))
        return cache[key]

    res = minimize(
        lambda h: float(areas @ h), h0, jac=lambda h: areas, method="SLSQP",
        constraints=[{
            "type": "ineq",
            "fun": lambda h: geom(h)[0] ** (1.0 / 3.0) - 1.0,
            "jac": lambda h: geom(h)[1] / (3.0 * geom(h)[0] ** (2.0 / 3.0)),
        }],
        bounds=[(1e-9, None)] * n,
        options={"maxiter": maxiter, "ftol": 1e-12},
    )

    h = np.clip(res.x, 1e-9, None)

    # Gauge 1: scale so the facet areas match the target areas.
    _, _, _, A = polytope_from_support(normals, h)
    denom = float(A @ A)
    if denom > 0:
        h = h * np.sqrt(max(float(areas @ A) / denom, 1e-12))

    # Gauge 2: recentre on the centre of mass.
    pts, hull, _, _ = polytope_from_support(normals, h)
    _, centroid, _ = _tetra_moments(pts, hull)
    return h - normals @ centroid


# ---------------------------------------------------------------------------
# Volume / inertia / DEEVE
# ---------------------------------------------------------------------------

def _tetra_moments(pts: np.ndarray, hull: ConvexHull):
    """``(volume, centroid, second-moment matrix ∫ x xᵀ dV)`` for unit density.

    Decomposes the polyhedron into tetrahedra from the origin (valid because the
    origin is interior) and uses the exact per-tetrahedron formulae
    ``∫ x xᵀ dV = (V/20)(S Sᵀ + Σ v_i v_iᵀ)`` and centroid ``(Σ v_i)/4``.
    Moments are taken about the origin; callers shift to the centre of mass.
    """
    vol = 0.0
    first = np.zeros(3)
    cov = np.zeros((3, 3))
    for simplex in hull.simplices:
        p0, p1, p2 = pts[simplex]
        d = np.linalg.det(np.array([p0, p1, p2]))
        if d < 0:                                   # orient outward
            p1, p2 = p2, p1
            d = -d
        vt = d / 6.0
        if vt <= 0:
            continue
        s = p0 + p1 + p2                            # origin contributes nothing
        cov += (vt / 20.0) * (np.outer(s, s)
                              + np.outer(p0, p0) + np.outer(p1, p1) + np.outer(p2, p2))
        first += vt * s / 4.0                       # tetra centroid = (0+p0+p1+p2)/4
        vol += vt
    centroid = first / vol if vol > 0 else np.zeros(3)
    return vol, centroid, cov


def inertia_from_polyhedron(pts: np.ndarray, hull: ConvexHull):
    """``(volume, inertia_tensor)`` for unit density, **about the centre of mass**.

    The parallel-axis theorem removes any residual offset of the polyhedron from
    the origin, so the returned tensor (and hence the DEEVE) is independent of
    where the Minkowski solution happened to place the body.
    """
    vol, centroid, cov = _tetra_moments(pts, hull)
    inertia = np.trace(cov) * np.eye(3) - cov            # about the origin
    d = centroid
    shift = vol * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return vol, inertia - shift                          # about the centre of mass


def deeve_axes(volume: float, inertia: np.ndarray) -> Tuple[float, float, float]:
    """Semi-axes ``(a ≥ b ≥ c)`` of the dynamically-equivalent equal-volume ellipsoid.

    A uniform ellipsoid of mass ``M`` has principal moments
    ``(M/5)(b²+c²) ≤ (M/5)(a²+c²) ≤ (M/5)(a²+b²)``; inverting those for the
    body's own principal moments gives equivalent axes, which are then scaled
    so the ellipsoid volume equals the body's volume.
    """
    mass = volume                                    # unit density
    i1, i2, i3 = np.sort(np.linalg.eigvalsh(inertia))
    k = 5.0 / (2.0 * mass)
    a2 = k * (i2 + i3 - i1)
    b2 = k * (i1 + i3 - i2)
    c2 = k * (i1 + i2 - i3)
    a, b, c = (np.sqrt(max(v, 0.0)) for v in (a2, b2, c2))
    ell_vol = 4.0 / 3.0 * np.pi * a * b * c
    if ell_vol > 0:
        scale = (volume / ell_vol) ** (1.0 / 3.0)
        a, b, c = a * scale, b * scale, c * scale
    return float(a), float(b), float(c)


# ---------------------------------------------------------------------------
# Convex shape
# ---------------------------------------------------------------------------

@dataclass
class ConvexShape:
    """A convex body stored as a discretised Gaussian image.

    Attributes
    ----------
    normals : (N, 3) unit outward normal directions
    areas   : (N,) positive surface-area weight for each direction
    """

    normals: np.ndarray
    areas: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_sh(cls, coeffs: np.ndarray, lmax: int, normals: np.ndarray,
                solid_angles: np.ndarray, basis: Optional[np.ndarray] = None) -> "ConvexShape":
        """Build from real-SH coefficients of ``log`` area density (positivity by construction)."""
        if basis is None:
            basis = real_sph_harm_basis(lmax, normals)
        return cls(normals=normals, areas=np.exp(basis @ coeffs) * solid_angles)

    @classmethod
    def from_ellipsoid(cls, a: float, b: float, c: float, n_normals: int = 300) -> "ConvexShape":
        """Build the convex representation of a triaxial ellipsoid (exact Gaussian image)."""
        normals, solid = fibonacci_sphere(n_normals)
        return cls(normals=normals, areas=ellipsoid_gaussian_image(normals, a, b, c, solid))

    # -- diagnostics -------------------------------------------------------
    @property
    def total_area(self) -> float:
        return float(self.areas.sum())

    def closure_residual(self) -> float:
        """``|Σ a_i n_i| / Σ a_i`` — zero for a valid closed convex body."""
        return float(np.linalg.norm(self.areas @ self.normals) / self.areas.sum())

    # -- reconstruction ----------------------------------------------------
    def support(self, **kw) -> np.ndarray:
        """Support values solving the Minkowski problem (cached)."""
        if "h" not in self._cache:
            self._cache["h"] = minkowski_support(self.normals, self.areas, **kw)
        return self._cache["h"]

    def polyhedron(self):
        """``(vertices, hull)`` of the reconstructed convex polyhedron (cached)."""
        if "poly" not in self._cache:
            pts, hull, _, _ = polytope_from_support(self.normals, self.support())
            self._cache["poly"] = (pts, hull)
        return self._cache["poly"]

    def volume(self) -> float:
        return self._mass_properties()[0]

    def inertia_tensor(self) -> np.ndarray:
        return self._mass_properties()[1]

    def _mass_properties(self):
        if "mass" not in self._cache:
            pts, hull = self.polyhedron()
            self._cache["mass"] = inertia_from_polyhedron(pts, hull)
        return self._cache["mass"]

    def deeve(self) -> Tuple[float, float, float]:
        """Semi-axes ``(a ≥ b ≥ c)`` of the equivalent ellipsoid."""
        vol, inertia = self._mass_properties()
        return deeve_axes(vol, inertia)

    def axis_ratios(self) -> Tuple[float, float]:
        """``(a/b, b/c)`` from the DEEVE — the inputs to the geophysics module."""
        a, b, c = self.deeve()
        return a / b, b / c


__all__ = [
    "fibonacci_sphere",
    "sh_degree_order",
    "real_sph_harm_basis",
    "ellipsoid_support",
    "ellipsoid_gaussian_image",
    "polytope_from_support",
    "minkowski_support",
    "inertia_from_polyhedron",
    "deeve_axes",
    "ConvexShape",
]
