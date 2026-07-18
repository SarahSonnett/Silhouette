"""Tests for the convex shape representation (silhouette.shapes).

The triaxial ellipsoid is the reference case throughout: its Gaussian image and
support function are known analytically, so a round trip
``ellipsoid -> Gaussian image -> Minkowski solve -> polyhedron -> DEEVE``
must return the original axis ratios.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.shapes import (  # noqa: E402
    ConvexShape,
    deeve_axes,
    ellipsoid_gaussian_image,
    ellipsoid_support,
    fibonacci_sphere,
    inertia_from_polyhedron,
    minkowski_support,
    polytope_from_support,
    real_sph_harm_basis,
    sh_degree_order,
)

N_DIRS = 150          # keep the Minkowski solves quick in CI

# Minkowski solves dominate the runtime, and ConvexShape caches its own
# reconstruction — so share one instance per shape across the module.
TRIAXIAL = (2.0, 1.4, 1.0)


@pytest.fixture(scope="module")
def triaxial():
    return ConvexShape.from_ellipsoid(*TRIAXIAL, N_DIRS)


@pytest.fixture(scope="module")
def sphere():
    return ConvexShape.from_ellipsoid(1.5, 1.5, 1.5, N_DIRS)


# ---------------------------------------------------------------- discretisation

def test_fibonacci_sphere_is_unit_and_closed():
    n, w = fibonacci_sphere(200)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.isclose(w.sum(), 4 * np.pi)
    # near-uniform coverage -> directions sum to ~zero
    assert np.linalg.norm(n.mean(axis=0)) < 0.05


# ---------------------------------------------------------------- harmonics

def test_sh_basis_shape_and_orthonormality():
    lmax = 4
    normals, w = fibonacci_sphere(3000)
    basis = real_sph_harm_basis(lmax, normals)
    assert basis.shape == (3000, (lmax + 1) ** 2)
    assert len(sh_degree_order(lmax)) == (lmax + 1) ** 2
    # <Y_i, Y_j> = delta_ij under the equal-solid-angle quadrature
    gram = (basis * w[:, None]).T @ basis
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=0.05)


# ---------------------------------------------------------------- Gaussian image

def test_sphere_gaussian_image_total_area():
    r = 1.7
    normals, w = fibonacci_sphere(500)
    areas = ellipsoid_gaussian_image(normals, r, r, r, w)
    assert np.isclose(areas.sum(), 4 * np.pi * r ** 2, rtol=1e-10)
    assert np.allclose(areas, areas[0])        # isotropic


def test_gaussian_image_positive_and_closed(triaxial):
    shape = triaxial
    assert np.all(shape.areas > 0)
    # a valid convex body satisfies sum(a_i n_i) = 0
    assert shape.closure_residual() < 5e-3


# ---------------------------------------------------------------- Minkowski

def test_minkowski_recovers_ellipsoid_support():
    a, b, c = 2.0, 1.4, 1.0
    normals, w = fibonacci_sphere(N_DIRS)
    areas = ellipsoid_gaussian_image(normals, a, b, c, w)
    h = minkowski_support(normals, areas)
    h_true = ellipsoid_support(normals, a, b, c)
    rel = np.sqrt(np.mean(((h - h_true) / h_true) ** 2))
    assert rel < 0.05        # recentred solution matches the analytic support


def test_polytope_volume_close_to_ellipsoid(triaxial):
    a, b, c = TRIAXIAL
    shape = triaxial
    expected = 4.0 / 3.0 * np.pi * a * b * c
    # discretisation makes the polytope slightly smaller than the smooth body
    np.testing.assert_allclose(shape.volume(), expected, rtol=0.05)


# ---------------------------------------------------------------- DEEVE

def test_deeve_of_sphere_is_isotropic(sphere):
    ab, bc = sphere.axis_ratios()
    assert abs(ab - 1.0) < 0.05
    assert abs(bc - 1.0) < 0.05


def test_deeve_recovers_triaxial_ratios(triaxial):
    a, b, c = TRIAXIAL
    ab, bc = triaxial.axis_ratios()
    assert np.isclose(ab, a / b, rtol=0.06)
    assert np.isclose(bc, b / c, rtol=0.06)


def test_deeve_recovers_flat_body():
    a, b, c = 3.0, 2.0, 1.0
    shape = ConvexShape.from_ellipsoid(a, b, c, N_DIRS)
    ab, bc = shape.axis_ratios()
    assert np.isclose(ab, a / b, rtol=0.08)
    assert np.isclose(bc, b / c, rtol=0.08)


def test_deeve_axes_exact_for_analytic_ellipsoid_inertia():
    """deeve_axes must invert the uniform-ellipsoid inertia formulae exactly."""
    a, b, c = 3.0, 2.0, 1.0
    vol = 4.0 / 3.0 * np.pi * a * b * c
    m = vol                                        # unit density
    inertia = np.diag([m / 5 * (b * b + c * c),
                       m / 5 * (a * a + c * c),
                       m / 5 * (a * a + b * b)])
    got = deeve_axes(vol, inertia)
    assert np.allclose(got, (a, b, c), rtol=1e-10)


def test_inertia_is_translation_invariant():
    """Inertia is taken about the centre of mass, so shifting the body is a no-op."""
    a, b, c = 2.0, 1.4, 1.0
    normals, w = fibonacci_sphere(N_DIRS)
    areas = ellipsoid_gaussian_image(normals, a, b, c, w)
    h = minkowski_support(normals, areas)
    pts, hull, _, _ = polytope_from_support(normals, h)
    v1, i1 = inertia_from_polyhedron(pts, hull)
    v2, i2 = inertia_from_polyhedron(pts + np.array([0.7, -0.3, 0.45]), hull)
    assert np.isclose(v1, v2, rtol=1e-8)
    assert np.allclose(np.sort(np.linalg.eigvalsh(i1)),
                       np.sort(np.linalg.eigvalsh(i2)), rtol=1e-6)


# ---------------------------------------------------------------- SH construction

def test_from_sh_constant_coefficient_is_a_sphere():
    lmax = 2
    normals, w = fibonacci_sphere(N_DIRS)
    coeffs = np.zeros((lmax + 1) ** 2)
    coeffs[0] = 1.0                                # Y_00 only -> isotropic
    shape = ConvexShape.from_sh(coeffs, lmax, normals, w)
    assert np.all(shape.areas > 0)
    assert np.allclose(shape.areas, shape.areas[0])
    ab, bc = shape.axis_ratios()
    assert abs(ab - 1.0) < 0.05 and abs(bc - 1.0) < 0.05
