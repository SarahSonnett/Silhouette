"""Integration test on real multi-apparition DAMIT photometry of (15) Eunomia.

The bundled file (``data/15_eunomia_damit_lcs.txt``) holds 109 relative light
curves with embedded Sun/Earth geometry. The amplitude-only fit should recover
a clearly elongated, retrograde, high-latitude spin consistent with the DAMIT
convex model (λ≈3°, β≈−67°), up to the mirror ambiguity and the ~30° accuracy
of the analytical ellipsoid method.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette import read_damit_lcs, damit_apparitions, fit_shape  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "15_eunomia_damit_lcs.txt")
PERIOD_DAYS = 6.082753 / 24.0
DAMIT_POLE = (3.0, -67.0)


def _sep(p1, p2):
    l1, b1 = np.radians(p1); l2, b2 = np.radians(p2)
    return np.degrees(np.arccos(np.clip(
        np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


def test_reads_damit_format():
    curves = read_damit_lcs(DATA)
    assert len(curves) == 109
    assert curves[0].intensity.size == curves[0].jd.size > 0
    assert curves[0].earth.shape == (3,)


def test_eunomia_recovers_elongated_retrograde_pole():
    curves = read_damit_lcs(DATA)
    apps = damit_apparitions(curves, period=PERIOD_DAYS)
    assert len(apps) >= 10                       # genuinely multi-apparition
    fit = fit_shape(apps, use_meanmag=False)
    assert not fit.used_meanmag
    assert 1.3 < fit.ab < 2.2                    # clearly elongated, like Eunomia
    # Pole within ~35 deg of the DAMIT solution (up to the mirror ambiguity).
    sep = min(_sep((fit.pole_lon, fit.pole_lat), DAMIT_POLE),
              _sep(fit.mirror_pole, DAMIT_POLE))
    assert sep < 35.0
