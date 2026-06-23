"""Integration test on the bundled real photometry of asteroid (16152).

The dataset (``data/16152_2019_rp.txt``) is a single 2019 apparition, so the
expected, scientifically correct outcome is an a/b lower bound with the pole and
b/c flagged undetermined — not a full pole solution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette import read_photometry, reduce_apparitions, fit_shape  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "16152_2019_rp.txt")
PERIOD_DAYS = 22.936 / 24.0


def test_reads_spindoc_format():
    phot = read_photometry(DATA, columns={"merr": "TmagFinalErr"}, object_name="16152")
    assert len(phot) > 400
    assert not phot.has_ecliptic           # no ecliptic columns in this file


def test_single_apparition_lower_bound():
    phot = read_photometry(DATA, columns={"merr": "TmagFinalErr"})
    apps = reduce_apparitions(phot, period=PERIOD_DAYS)
    assert len(apps) == 1                   # ~50 day span -> one apparition
    fit = fit_shape(apps)
    assert fit.degenerate and fit.pole_lon is None
    # Amplitude ~0.4 mag implies a clearly elongated body.
    assert fit.ab > 1.2
