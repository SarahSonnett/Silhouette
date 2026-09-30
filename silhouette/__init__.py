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
from .shapes import (
    ConvexShape,
    fibonacci_sphere,
    real_sph_harm_basis,
    ellipsoid_gaussian_image,
    minkowski_support,
    deeve_axes,
)
from .inversion import (
    LightCurveObs, InversionResult, invert_convex,
    invert_convex_multistart, cluster_pole_families, PoleFamily, default_pole_grid,
    PeriodScanResult, period_search_grid, scan_period,
)
from .geophysics import (
    min_density_cohesionless, required_cohesion, is_stable_cohesionless,
    shedding_limit_density, propagate_axis_uncertainty, DensityConstraint,
)
from .geoplots import (
    plot_cohesion_vs_density, plot_spin_barrier, plot_shape_sensitivity,
    plot_strength_summary,
)
from .forward import (
    convex_brightness,
    convex_lightcurve,
    projected_area,
    ecliptic_to_body_matrix,
    SCATTERING_LAWS,
    geometric,
    lambert,
    lommel_seeliger,
    ls_lambert,
)
from .plotting import (
    plot_model_mosaic,
    plot_aspect_curves,
    plot_pole_map,
    plot_summary,
    save_summary,
)
from .scoring import (
    Candidate,
    PopulationPrior,
    ScoreReport,
    alias_periods,
    find_candidates,
    principal_axis_logprior,
    score_solutions,
    spin_vector_pole,
)
from .calibration import ScoreModel, load_default_model
from .planning import horizons_geometry, recommend_observations, simple_orbit_geometry
from .pipeline import lightcurves_from_photometry, spindoc_periods
from ._compat import HAVE_SPOTLIGHT, HAVE_SPINDOC

__all__ = [
    "Photometry", "read_photometry",
    "Apparition", "group_apparitions", "reduce_apparitions",
    "DamitLightcurve", "read_damit_lcs", "damit_apparitions",
    "resolve_geometry", "fetch_horizons_ecliptic",
    "aspect_angle", "amplitude_model", "mean_mag_model", "mean_projected_area",
    "ab_lower_bound", "mirror_pole", "axes_from_ratios",
    "SilhouetteFit", "PoleSolution", "fit_shape",
    # shape inversion (Phase 1a)
    "ConvexShape", "fibonacci_sphere", "real_sph_harm_basis",
    "ellipsoid_gaussian_image", "minkowski_support", "deeve_axes",
    "convex_brightness", "convex_lightcurve", "projected_area",
    "ecliptic_to_body_matrix", "SCATTERING_LAWS",
    "geometric", "lambert", "lommel_seeliger", "ls_lambert",
    "LightCurveObs", "InversionResult", "invert_convex",
    "invert_convex_multistart", "cluster_pole_families", "PoleFamily",
    "default_pole_grid",
    # geophysics (Phase 2)
    "min_density_cohesionless", "required_cohesion", "is_stable_cohesionless",
    "shedding_limit_density", "propagate_axis_uncertainty", "DensityConstraint",
    "plot_cohesion_vs_density", "plot_spin_barrier", "plot_shape_sensitivity",
    "plot_strength_summary",
    "PeriodScanResult", "period_search_grid", "scan_period",
    "plot_model_mosaic", "plot_aspect_curves", "plot_pole_map",
    "plot_summary", "save_summary",
    # solution scorer
    "Candidate", "PopulationPrior", "ScoreReport", "alias_periods", "find_candidates",
    "principal_axis_logprior", "score_solutions", "spin_vector_pole",
    "ScoreModel", "load_default_model",
    "horizons_geometry", "recommend_observations", "simple_orbit_geometry",
    "lightcurves_from_photometry", "spindoc_periods",
    "HAVE_SPOTLIGHT", "HAVE_SPINDOC",
]
