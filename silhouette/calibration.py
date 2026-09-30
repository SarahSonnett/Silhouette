"""Calibration layer for the solution scorer: injection-recovery + classifier.

A likelihood weight is only a probability if the model and noise assumptions
hold exactly; in practice they don't (irregular shapes, the wrong scattering
law, correlated residuals, too few starting poles). The honest fix is
empirical: inject synthetic asteroids whose truth is known, run the *identical*
pipeline, and learn how often a candidate with a given set of features is
actually right.

* :func:`random_truth` / :func:`random_geometry` draw a body (irregular convex
  shape, isotropic pole, period, scattering mismatch) and a survey-like set of
  light curves (1–6 apparitions, sparse-to-dense rotational coverage, 0.5–4 %
  noise, varied phase angles).
* :func:`inject_one` renders the truth, runs
  :func:`silhouette.scoring.find_candidates` → likelihood → bootstrap →
  features, and labels each candidate ``correct`` when it has the true period
  and lies within ``tol_deg`` of the true pole.
* :func:`run_injections` farms injections out to ``n_workers`` processes and
  appends rows to a CSV as they finish, so an interrupted run keeps its data.
* :class:`ScoreModel` is a gradient-boosted classifier with isotonic
  calibration, trained with injection-grouped cross-validation (candidates from
  the same injection never straddle train and test).

The metric that matters for spin catalogues is **yield at a fixed false-solution
rate** — how many objects get a pole you can trust — not shape error. See
:func:`yield_curve` and ``calibrate_scorer.py``.
"""

from __future__ import annotations

import csv
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .forward import convex_lightcurve, ls_lambert
from .inversion import LightCurveObs, _sep_deg, default_pole_grid
from .scoring import (
    alias_periods,
    baseline_of,
    bootstrap_stability,
    candidate_features,
    combine_uncalibrated,
    data_features,
    find_candidates,
    likelihood_weights,
)
from .shapes import ConvexShape, fibonacci_sphere, real_sph_harm_basis

# Features the classifier sees. Kept explicit so a saved model and new data
# can never silently disagree on column order.
FEATURES = [
    "n_points", "n_lc", "n_app", "lon_coverage", "lat_span", "phase_max",
    "phase_min", "noise_frac", "amp_mag",
    "dchi2_scaled", "margin_scaled", "like_weight", "boot_frac", "rank",
    "period_rank", "n_candidates", "n_periods", "basin_frac", "redchi2_min",
    "redchi2_ratio", "neff_factor", "abs_lat", "elongation", "flattening",
    "has_mirror", "has_antipode", "sep_from_best", "rot_coverage",
]

# Fast pipeline settings used inside the calibration loop. Real targets can use
# richer settings, but the model is only calibrated for settings it was trained
# with — keep production runs close to these or retrain.
FAST_INV = dict(lmax=3, n_normals=150, max_nfev=150)
FAST_GRID = default_pole_grid(n_lon=5, lats=(-60.0, -20.0, 20.0, 60.0))


# ---------------------------------------------------------------------------
# Synthetic truth
# ---------------------------------------------------------------------------

def _dir(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def irregular_shape(ab: float, bc: float, roughness: float,
                    rng: np.random.Generator, n_normals: int = 300,
                    lmax: int = 5) -> ConvexShape:
    """Triaxial ellipsoid with a random low-order convex perturbation.

    The log Gaussian image of the ellipsoid gets random real-SH terms of degree
    2..lmax (amplitude ∝ ``roughness``, decaying with degree), then the l = 1
    closure error is removed iteratively so the body stays closed.
    """
    b = bc
    a = ab * b
    base = ConvexShape.from_ellipsoid(a, b, 1.0, n_normals)
    normals, solid = fibonacci_sphere(n_normals)
    basis = real_sph_harm_basis(lmax, normals)
    coeffs = np.zeros(basis.shape[1])
    k = 0
    for ell in range(lmax + 1):
        for _m in range(2 * ell + 1):
            if ell >= 2:
                coeffs[k] = rng.normal(0.0, roughness / ell)
            k += 1
    areas = base.areas * np.exp(basis @ coeffs)
    for _ in range(8):                           # restore closure  Σ a n = 0
        c = (areas @ normals) / areas.sum()
        areas = areas * np.clip(1.0 - 3.0 * (normals @ c), 0.2, None)
    return ConvexShape(normals=normals, areas=areas)


@dataclass
class Truth:
    ab: float
    bc: float
    roughness: float
    pole_lon: float
    pole_lat: float
    period: float          # days
    phi0: float
    lambert_c: float       # scattering used to render (fit assumes 0.1)
    shape: ConvexShape = field(repr=False)


def random_truth(rng: np.random.Generator) -> Truth:
    ab = float(np.exp(rng.uniform(np.log(1.05), np.log(2.5))))
    bc = float(rng.uniform(1.0, 1.5))
    rough = float(rng.uniform(0.0, 0.25))
    z = rng.uniform(-1.0, 1.0)                   # isotropic pole
    lat = float(np.degrees(np.arcsin(z)))
    lon = float(rng.uniform(0.0, 360.0))
    period_h = float(np.exp(rng.uniform(np.log(3.0), np.log(20.0))))
    return Truth(ab=ab, bc=bc, roughness=rough, pole_lon=lon, pole_lat=lat,
                 period=period_h / 24.0, phi0=float(rng.uniform(0, 2 * np.pi)),
                 lambert_c=float(rng.uniform(0.0, 0.3)),
                 shape=irregular_shape(ab, bc, rough, rng))


@dataclass
class GeometryPlan:
    n_app: int
    n_lc: int
    pts_per_lc: int
    coverage: float          # fraction of a rotation each curve spans
    noise_frac: float
    phase_max: float
    epochs: List[float]
    earth: List[np.ndarray] = field(repr=False)
    sun: List[np.ndarray] = field(repr=False)


def random_geometry(rng: np.random.Generator, period: float) -> GeometryPlan:
    """A survey-like observing plan in asteroid-centric ecliptic vectors (days)."""
    n_app = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.2, 0.2, 0.2, 0.15, 0.1]))
    incl = float(rng.uniform(0.0, 20.0))
    high_phase = rng.random() < 0.15             # NEA-like geometry
    alpha_max = float(rng.uniform(30.0, 60.0) if high_phase else rng.uniform(8.0, 28.0))
    pts = int(rng.integers(12, 61))
    cover = float(rng.uniform(0.4, 1.3))
    noise = float(np.exp(rng.uniform(np.log(0.005), np.log(0.04))))
    epochs, earth, sun = [], [], []
    t_app = 0.0
    for _ in range(n_app):
        lon_app = float(rng.uniform(0.0, 360.0))
        lat_app = float(rng.uniform(-incl, incl))
        n_lc = int(rng.integers(1, 5))
        for j in range(n_lc):
            dt = float(rng.uniform(0.0, 40.0))
            alpha = float(rng.uniform(2.0, alpha_max))
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            e = _dir(lon_app + 0.3 * dt, lat_app)
            s = _dir(lon_app + 0.3 * dt + sgn * alpha, 0.3 * lat_app)
            epochs.append(t_app + dt)
            earth.append(e)
            sun.append(s)
        t_app += float(rng.uniform(350.0, 800.0))
    order = np.argsort(epochs)
    return GeometryPlan(n_app=n_app, n_lc=len(epochs), pts_per_lc=pts, coverage=cover,
                        noise_frac=noise, phase_max=alpha_max,
                        epochs=[epochs[i] for i in order],
                        earth=[earth[i] for i in order], sun=[sun[i] for i in order])


def render(truth: Truth, plan: GeometryPlan, rng: np.random.Generator) -> List[LightCurveObs]:
    lcs = []
    for ep, e, s in zip(plan.epochs, plan.earth, plan.sun):
        span = plan.coverage * truth.period
        t = ep + np.sort(rng.uniform(0.0, span, plan.pts_per_lc))
        f = convex_lightcurve(truth.shape, t, s, e, truth.pole_lon, truth.pole_lat,
                              truth.period, phi0=truth.phi0, t0=0.0,
                              phot_func=ls_lambert, arg=truth.lambert_c)
        sig = plan.noise_frac * float(np.mean(f))
        lcs.append(LightCurveObs(t, f + rng.normal(0.0, sig, f.size),
                                 np.full(f.size, sig), s, e, relative=True))
    return lcs


# ---------------------------------------------------------------------------
# One injection
# ---------------------------------------------------------------------------

def inject_one(seed: int, tol_deg: float = 20.0, n_boot: int = 10,
               boot_max_nfev: int = 30, n_alias: int = 1) -> List[Dict[str, float]]:
    """Run the full scorer pipeline on one synthetic asteroid; one row per candidate."""
    rng = np.random.default_rng(seed)
    truth = random_truth(rng)
    plan = random_geometry(rng, truth.period)
    lcs = render(truth, plan, rng)
    periods = alias_periods(truth.period, max(baseline_of(lcs), truth.period), n_alias=n_alias)
    t_start = time.time()
    cands = find_candidates(lcs, periods, pole_grid=FAST_GRID, n_workers=1, **FAST_INV)
    noise = likelihood_weights(cands, lcs)
    if len(cands) > 1 and n_boot > 0:
        bootstrap_stability(cands, lcs, n_boot=n_boot, seed=seed, boot_max_nfev=boot_max_nfev,
                            n_workers=1, **FAST_INV)
    else:
        for c in cands:
            c.boot_frac = 1.0
    candidate_features(cands, lcs, noise_diag=noise, data_feats=data_features(lcs))
    combine_uncalibrated(cands)
    elapsed = time.time() - t_start

    rows = []
    for c in cands:
        sep = _sep_deg(c.pole_lon, c.pole_lat, truth.pole_lon, truth.pole_lat)
        sep_anti = _sep_deg(c.pole_lon, c.pole_lat, truth.pole_lon + 180.0, -truth.pole_lat)
        sep_mirror = _sep_deg(c.pole_lon, c.pole_lat, truth.pole_lon + 180.0, truth.pole_lat)
        right_p = abs(c.period - truth.period) < 1e-6 * truth.period
        row = dict(c.features)
        row.update({
            "inj": float(seed), "cand_period": c.period, "cand_lon": c.pole_lon,
            "cand_lat": c.pole_lat, "cand_redchi2": c.redchi2,
            "p_uncal": c.probability,
            "true_period": truth.period, "true_lon": truth.pole_lon,
            "true_lat": truth.pole_lat, "true_ab": truth.ab, "true_bc": truth.bc,
            "true_rough": truth.roughness, "true_lambert_c": truth.lambert_c,
            "sep_true": sep, "sep_antipode": sep_anti, "sep_mirror": sep_mirror,
            "right_period": float(right_p),
            "plan_pts": float(plan.pts_per_lc), "plan_cover": plan.coverage,
            "plan_noise": plan.noise_frac, "plan_alpha_max": plan.phase_max,
            "correct": float(right_p and sep < tol_deg),
            "elapsed_s": elapsed,
            "s2": noise["s2"],
        })
        rows.append(row)
    return rows


def upgrade_legacy_s2(df):
    """Convert rows made with the old ``s² = max(1, χ²_ν,min)`` rule to ``s² = χ²_ν,min``.

    The first calibration run (2026-09-29) used the floored rule. Rows without
    an ``s2`` column are converted exactly: the scaled Δχ² features are
    rescaled by ``s²_old / s²_new`` and the likelihood weights (and the
    uncalibrated probability built from them) are recomputed per injection.
    Returns a new DataFrame.
    """
    import pandas as pd

    if "s2" in df.columns:
        return df
    out = df.copy()
    r = out["redchi2_min"].astype(float)
    factor = np.maximum(1.0, r) / np.maximum(r, 0.05)
    out["dchi2_scaled"] = out["dchi2_scaled"] * factor
    sentinel = out["margin_scaled"] == 50.0
    out.loc[~sentinel, "margin_scaled"] = (out["margin_scaled"] * factor)[~sentinel]
    ll = -0.5 * out["dchi2_scaled"]
    ll = ll - ll.groupby(out["inj"]).transform("max")
    w = np.exp(ll)
    out["like_weight"] = w / w.groupby(out["inj"]).transform("sum")
    pu = out["like_weight"] * (out["boot_frac"] + 0.05)
    out["p_uncal"] = pu / pu.groupby(out["inj"]).transform("sum")
    out["s2"] = np.maximum(r, 0.05)
    return pd.DataFrame(out)


def _inject_safe(seed):
    try:
        return seed, inject_one(seed), None
    except Exception as exc:  # keep the run alive; record the failure
        return seed, [], repr(exc)


def run_injections(seeds: Sequence[int], out_csv: str, n_workers: int = 4,
                   log_every: int = 10, deadline: Optional[float] = None) -> int:
    """Run injections in parallel, appending rows to ``out_csv`` as they finish.

    Seeds already present in the CSV are skipped, so the run can be resumed.
    ``deadline`` (a ``time.time()`` value) stops the run cleanly: pending
    injections are cancelled and the ones in flight are allowed to finish.
    Returns the number of new injections completed.
    """
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            for row in csv.DictReader(fh):
                done.add(int(float(row["inj"])))
    todo = [int(s) for s in seeds if int(s) not in done]
    if not todo:
        return 0
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    header = None
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            header = next(csv.reader(fh), None)
    t0 = time.time()
    n_done = 0
    fail_log = out_csv + ".failures.txt"
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_inject_safe, s) for s in todo]
        stopped = False
        for fut in as_completed(futs):
            if deadline is not None and not stopped and time.time() > deadline:
                n_cancel = sum(f.cancel() for f in futs)
                print(f"[{time.strftime('%H:%M:%S')}] deadline reached; cancelled "
                      f"{n_cancel} pending injections", flush=True)
                stopped = True
            if fut.cancelled():
                continue
            seed, rows, err = fut.result()
            n_done += 1
            if err is not None:
                with open(fail_log, "a") as fh:
                    fh.write(f"{seed}\t{err}\n")
            if rows:
                if header is None:
                    header = list(rows[0].keys())
                    with open(out_csv, "w", newline="") as fh:
                        csv.writer(fh).writerow(header)
                with open(out_csv, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                    for r in rows:
                        w.writerow(r)
            if n_done % log_every == 0:
                rate = n_done / (time.time() - t0) * 3600.0
                print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(todo)} injections "
                      f"({rate:.0f}/h)", flush=True)
    return n_done


# ---------------------------------------------------------------------------
# The calibrated classifier
# ---------------------------------------------------------------------------

@dataclass
class ScoreModel:
    """Gradient-boosted P(correct | features) with isotonic calibration."""

    classifier: object
    isotonic: object
    features: List[str]
    meta: Dict[str, object] = field(default_factory=dict)

    def predict_from_matrix(self, X: np.ndarray) -> np.ndarray:
        raw = self.classifier.predict_proba(X)[:, 1]
        return np.clip(self.isotonic.predict(raw), 0.0, 1.0)

    def predict_proba(self, candidates) -> np.ndarray:
        X = np.array([[c.features[f] for f in self.features] for c in candidates], dtype=float)
        return self.predict_from_matrix(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "ScoreModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "scorer_v1.pkl")


def load_default_model() -> Optional[ScoreModel]:
    """The bundled calibration model, or ``None`` if it has not been trained."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        return ScoreModel.load(DEFAULT_MODEL_PATH)
    return None


def _make_classifier(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=0.1, random_state=seed)


def fit_score_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = 5, seed: int = 0,
                    features: Sequence[str] = FEATURES) -> ScoreModel:
    """Fit the classifier and an isotonic map from injection-grouped OOF scores.

    Out-of-fold predictions come from ``GroupKFold`` over injections, so the
    isotonic calibration never sees a score the classifier was trained on.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    oof = np.full(y.size, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = _make_classifier(seed).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    clf = _make_classifier(seed).fit(X, y)
    import sklearn
    return ScoreModel(classifier=clf, isotonic=iso, features=list(features),
                      meta={"n_rows": int(y.size), "n_injections": int(np.unique(groups).size),
                            "sklearn": sklearn.__version__,
                            "trained": time.strftime("%Y-%m-%d %H:%M")})


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """``(bin_mean_pred, bin_frac_correct, bin_count)`` for a reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mp, fy, n = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mp.append(float(p[m].mean()))
        fy.append(float(y[m].mean()))
        n.append(int(m.sum()))
    return np.array(mp), np.array(fy), np.array(n)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def classical_accept(group_rows: List[Dict[str, float]], margin: float = 0.10,
                     allow_mirror: bool = True) -> Optional[Dict[str, float]]:
    """DAMIT-style uniqueness rule applied to one injection's candidates.

    Accept the best candidate when every rival (other period, or other pole at
    the same period) has ``chi2_nu`` at least ``margin`` higher. With
    ``allow_mirror`` a rival that is the ``(λ+180°, β)`` mirror at the same
    period does not block acceptance (DAMIT reports both). Returns the accepted
    row or ``None``.
    """
    rows = sorted(group_rows, key=lambda r: r["cand_redchi2"])
    best = rows[0]
    for r in rows[1:]:
        if r["cand_redchi2"] >= (1.0 + margin) * best["cand_redchi2"]:
            continue
        if (allow_mirror and r["cand_period"] == best["cand_period"] and
                _sep_deg(r["cand_lon"], r["cand_lat"], best["cand_lon"] + 180.0,
                         best["cand_lat"]) < 25.0):
            continue
        return None
    return best


def yield_curve(p_best: np.ndarray, correct_best: np.ndarray, n_total: int,
                thresholds: Optional[np.ndarray] = None):
    """Accept an object when its top candidate has ``p ≥ threshold``.

    Returns ``(thresholds, yield_frac, false_rate)`` where ``yield_frac`` is
    correct acceptances / all objects and ``false_rate`` is wrong / accepted.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.99, 100)
    yl, fr = [], []
    for t in thresholds:
        acc = p_best >= t
        n_acc = int(acc.sum())
        yl.append(float(np.sum(correct_best[acc])) / max(n_total, 1))
        fr.append(float(np.sum(1 - correct_best[acc])) / n_acc if n_acc else np.nan)
    return thresholds, np.array(yl), np.array(fr)


__all__ = [
    "FEATURES", "FAST_INV", "FAST_GRID", "Truth", "GeometryPlan", "ScoreModel",
    "irregular_shape", "random_truth", "random_geometry", "render", "inject_one",
    "run_injections", "upgrade_legacy_s2", "fit_score_model", "load_default_model", "DEFAULT_MODEL_PATH",
    "reliability", "brier", "classical_accept", "yield_curve",
]
