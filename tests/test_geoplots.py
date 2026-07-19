"""Smoke tests for the strength/density figures (silhouette.geoplots).

These check that each panel builds and carries the expected axis labelling; the
underlying physics is tested in test_geophysics.py.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette.geoplots import (  # noqa: E402
    plot_cohesion_vs_density,
    plot_shape_sensitivity,
    plot_spin_barrier,
    plot_strength_summary,
)


def test_cohesion_panel_builds():
    fig = plot_cohesion_vs_density(1.45, 1.2, 2.6, 1.0, friction_degs=(35.0,))
    ax = fig.axes[0]
    assert "cohesion" in ax.get_ylabel().lower()
    assert ax.get_ylim()[0] >= 0.0          # cohesion is non-negative
    plt.close(fig)


def test_spin_barrier_panel_builds():
    fig = plot_spin_barrier(1.45, 1.2, 35.0, mark=(2.6, 2500.0))
    ax = fig.axes[0]
    assert "period" in ax.get_xlabel().lower()
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_shape_sensitivity_panel_builds():
    fig = plot_shape_sensitivity(1.45, 1.2, 2.6)
    ax = fig.axes[0]
    assert "a : b" in ax.get_xlabel()
    plt.close(fig)


def test_summary_figure_has_three_panels(tmp_path):
    out = tmp_path / "strength.png"
    fig = plot_strength_summary(1.45, 1.2, 2.6, 1.0, path=str(out))
    assert out.exists() and out.stat().st_size > 1000
    del fig
