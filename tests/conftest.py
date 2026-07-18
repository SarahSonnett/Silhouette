"""Test-session guards.

Silhouette is often developed on a machine that is simultaneously running other
compute (e.g. parallel PyLEADER jobs). NumPy/SciPy link against threaded BLAS
back-ends (Accelerate/OpenBLAS/MKL) that will otherwise spawn one thread per
core and contend with those jobs.

These variables must be set *before* NumPy is imported, so this module pins them
at collection time and imports NumPy immediately to lock the choice in.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy  # noqa: E402,F401  (imported here to freeze the thread settings)
