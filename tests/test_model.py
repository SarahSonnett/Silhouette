"""Unit tests for the analytical relations in silhouette.model."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.model import (  # noqa: E402
    aspect_angle, amplitude_model, mean_projected_area, ab_lower_bound, mirror_pole,
)


def test_amplitude_at_equator_is_pure_axis_ratio():
    # At equatorial aspect (theta = 90 deg) amplitude reduces to 2.5 log10(a/b).
    p = 1.7
    amp = amplitude_model(p, 1.3, np.radians(90.0))
    assert np.isclose(amp, 2.5 * np.log10(p), atol=1e-6)


def test_amplitude_vanishes_pole_on():
    # Pole-on (theta = 0) shows the constant a*b face -> zero amplitude.
    assert np.isclose(amplitude_model(2.0, 1.5, 0.0), 0.0, atol=1e-9)


def test_amplitude_monotonic_in_aspect():
    th = np.radians(np.linspace(0, 90, 50))
    amp = amplitude_model(1.8, 1.4, th)
    assert np.all(np.diff(amp) >= -1e-9)


def test_ab_lower_bound_roundtrip():
    p = 2.3
    amp = 2.5 * np.log10(p)
    assert np.isclose(ab_lower_bound(amp), p, atol=1e-9)


def test_aspect_angle_known_geometry():
    # Pole at the ecliptic pole (lat=90): aspect = 90 - target latitude.
    th = np.degrees(aspect_angle(123.0, 20.0, 0.0, 90.0))
    assert np.isclose(th, 70.0, atol=1e-6)


def test_mean_area_brighter_pole_on():
    # Full a*b face (pole-on) is larger than the equatorial mean for a>b>c.
    pole_on = mean_projected_area(1.8, 1.5, 0.0)
    equator = mean_projected_area(1.8, 1.5, np.radians(90.0))
    assert pole_on > equator


def test_mirror_pole():
    lon, lat = mirror_pole(60.0, 35.0)
    assert np.isclose(lon, 240.0) and np.isclose(lat, -35.0)
