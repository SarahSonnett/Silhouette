"""Silhouette — analytical asteroid shape & pole fitting from light-curve photometry.

The inverse of SpotLight: ingest tabular asteroid photometry, reduce it into
per-apparition observables, and analytically fit the triaxial axis ratios
``a:b`` and ``b:c`` together with the rotation-pole ecliptic longitude/latitude,
using the amplitude-aspect and mean-magnitude relations. Results render as a
multi-panel figure echoing SpotLight's combined output.

Quick start
-----------
>>> from silhouette import (read_photometry, reduce_apparitions,
...                         resolve_geometry, fit_shape, save_summary)
>>> phot = read_photometry("photometry.txt", object_name="433")
>>> apps = reduce_apparitions(phot, period=0.2194)
>>> resolve_geometry(apps, target="433")          # file columns or Horizons
>>> fit = fit_shape(apps)
>>> print(fit.summary())
>>> save_summary(fit, "docs/images/fit_summary.png")
"""

from .io import Photometry, read_photometry
from .apparitions import Apparition, group_apparitions, reduce_apparitions
from .damit import DamitLightcurve, read_damit_lcs, damit_apparitions
from .geometry import resolve_geometry, fetch_horizons_ecliptic
from .model import (
    aspect_angle,
    amplitude_model,
    mean_mag_model,
    mean_projected_area,
    ab_lower_bound,
    mirror_pole,
    axes_from_ratios,
)
from .fit import SilhouetteFit, PoleSolution, fit_shape
from .plotting import (
    plot_model_mosaic,
    plot_aspect_curves,
    plot_pole_map,
    plot_summary,
    save_summary,
)
from ._compat import HAVE_SPOTLIGHT, HAVE_SPINDOC

__all__ = [
    "Photometry", "read_photometry",
    "Apparition", "group_apparitions", "reduce_apparitions",
    "DamitLightcurve", "read_damit_lcs", "damit_apparitions",
    "resolve_geometry", "fetch_horizons_ecliptic",
    "aspect_angle", "amplitude_model", "mean_mag_model", "mean_projected_area",
    "ab_lower_bound", "mirror_pole", "axes_from_ratios",
    "SilhouetteFit", "PoleSolution", "fit_shape",
    "plot_model_mosaic", "plot_aspect_curves", "plot_pole_map",
    "plot_summary", "save_summary",
    "HAVE_SPOTLIGHT", "HAVE_SPINDOC",
]
