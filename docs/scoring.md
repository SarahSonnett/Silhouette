# Solution scorer: probabilities for competing spin and shape solutions

*Built 2026-09-29/30. Modules: `silhouette/scoring.py`, `silhouette/calibration.py`,
`silhouette/planning.py`, `silhouette/pipeline.py`. Drivers: `score_target.py`,
`calibrate_scorer.py`, `example_scoring.py`, `example_eunomia_scoring.py`.*

## Why

Convex inversion rarely returns one answer. Several things produce candidate
solutions whose χ² values differ very little:

- period aliases (P ± P²/T);
- the (λ, β) ↔ (λ + 180°, β) mirror pole;
- the spin-sense antipode;
- ordinary local minima.

The classical practice (DAMIT; Ďurech et al.) accepts a solution as "unique"
when it beats every rival by a hand-set χ² margin, typically 10%, and
discards everything else. That is an accept/reject rule, not a probability. It
also throws away the partial information in rejected objects, which is most of
any sparse survey: Gaia DR3 gave unique models for about 14% of attempted
asteroids, ATLAS for about 3%.

The scorer keeps every plausible solution and attaches a probability to each.
It does not need the pole to be known in advance.

## Where it sits

```
Squint (images → calibrated mags)
  → SpinDoc (period, H–G, Fourier; bootstrap period error)
  → silhouette.pipeline.lightcurves_from_photometry  (nights, reduction, Horizons geometry)
  → silhouette.scoring.score_solutions               (candidates + probabilities)
  → silhouette.planning.recommend_observations       (when to observe next)
```

`score_target.py` runs the whole chain from a SpinDoc-format photometry file and a
SpinDoc period.

## The three layers

1. **Candidates** (`find_candidates`). The trial periods are the SpinDoc period
   and its one-rotation aliases (`alias_periods`, optionally with half and
   double harmonics). At each trial period the pole multistart runs, and the
   converged starts are clustered into pole families (15° tolerance, best χ²
   first). Each family becomes a `Candidate` that carries its best fit, the
   share of starts that fell into its basin, and cheap a/b and b/c proxies. The
   proxies are projected-area ratios, which are exact for ellipsoids and cost
   microseconds instead of the seconds a Minkowski solve takes.

2. **Likelihood weights** (`likelihood_weights`). The weight is
   `log L = −½ · f_eff · (χ²_i − χ²_min)/s²`, plus a BIC-style penalty on the
   parameter count and an optional population prior.
   - The noise scale is `s² = χ²_ν,min` (`noise_scale`): errors are rescaled so
     the best candidate has χ²_ν = 1. This works in both directions. It
     inflates underestimated errors, and it shrinks the uniform 2% assumed for
     DAMIT curves, where χ²_ν ≈ 0.2 would otherwise flatten every Δχ² fivefold.
   - `f_eff = (1−ρ)/(1+ρ)` comes from the lag-1 autocorrelation of the best
     fit's residuals within each light curve. Correlated residuals mean fewer
     independent points, and a raw Δχ² is far too decisive without this
     correction.
   - `PopulationPrior` takes log-densities for pole latitude and a/b, for
     example from PyLEADER/LEADER distributions.

3. **Bootstrap agreement** (`bootstrap_stability`). Whole light curves are
   resampled with replacement; the resampling unit is the curve because points
   within a curve share their systematics. Every candidate is refit from a warm
   start with a small iteration budget so it stays in its own basin, and each
   resample is awarded to the lowest χ². `boot_frac` is each candidate's share
   of the wins. This generalises the ATLAS period bootstrap of Ďurech et al.
   (2022) from periods to pole families.

**Calibration** (`calibration.py`). A likelihood weight is a probability only
when the model and noise assumptions hold exactly. They don't: shapes are
irregular, the scattering law is wrong, residuals are correlated, and the start
grid can miss a basin. So the probabilities are calibrated empirically:

- Synthetic asteroids are injected with a known truth:
  - irregular convex shapes: a/b from 1.05 to 2.5, b/c from 1.0 to 1.5, random
    SH roughness;
  - isotropic poles;
  - periods of 3–20 h;
  - a Lambert weight for rendering drawn from 0–0.3, while the fit assumes 0.1.
- They are observed with survey-like plans:
  - 1–6 apparitions, 1–4 nights each;
  - 12–60 points covering 0.4–1.3 rotations;
  - 0.5–4% noise;
  - phase angles up to 8–28°, or up to 60° for 15% of bodies.
- The identical pipeline runs on each injection.
- A candidate is labelled *correct* if it has the true period and lies within
  20° of the true pole.
- A gradient-boosted classifier (`HistGradientBoostingClassifier`) maps the
  28 features in `calibration.FEATURES` (data descriptors plus per-candidate
  evidence) to P(correct).
- Isotonic calibration is fitted on out-of-fold scores from
  injection-grouped CV, so candidates of one injection never straddle a split.
- 20% of injections are held out entirely for the evaluation below.

The calibrated probabilities do not have to sum to 1 over the candidates.
`ScoreReport.p_none = 1 − Σp` is the probability that the truth is not among
them.

The model is only calibrated for the settings it was trained with
(`calibration.FAST_INV` = lmax 3, 150 normals, 150 function evaluations, and
the 20-start `FAST_GRID`). `score_target.py` and the examples use those
settings. Use different settings, and you should retrain.

## The observation recommender

For each future epoch, `recommend_observations` works as follows:

1. Predict one full rotation for every surviving candidate.
2. Remove the mean (relative photometry) and compare candidates pairwise
   after the best circular phase shift, because the rotational phase cannot
   be propagated years ahead.
3. Convert the difference to the Δχ²_ij that one night (`n_points` at
   `sigma_mag`) would deliver.

Epochs are ranked by the expected probability of the true solution after
that night:

`E[p_true] = Σ_i p_i² / Σ_j p_j exp(−Δχ²_ij/2)`

This equals today's `Σp²` if the epoch separates nothing and approaches 1 if it
separates everything. It rewards separating the *probable* candidates.
Observability cuts apply: solar elongation (default ≥60°) and an optional
maximum phase angle. Geometry comes from `planning.horizons_geometry`, or from
`simple_orbit_geometry` (circular orbits) for demos.

This answers the practical form of "can we do better than ≥3 apparitions?".
The requirement is really geometry diversity. Choosing the next apparition
for its discriminating power can settle a solution that three arbitrary
apparitions might not.

## Pole convention: important

`forward.ecliptic_to_body_matrix` applies R_z(+φ) to ecliptic vectors, so the
body turns by −φ about the stated pole. **Silhouette's pole points opposite to
the spin angular momentum.** DAMIT and the literature quote the
angular-momentum direction, which is the antipode (λ + 180°, −β). Both describe
the same physical rotation. Use `scoring.spin_vector_pole` before comparing
with DAMIT; `score_target.py` prints both conventions.

This is why `example_eunomia_convex.py` matched DAMIT only "with mirror
allowed". Its "mirror" was (λ + 180°, −β), exactly this flip, not the true
(λ + 180°, β) mirror ambiguity. The core convention has **not** been changed.
That needs a decision, because flipping the sign of φ in `forward.py` changes
the meaning of every stored pole and the Eunomia and 16152 examples.

## Results

*(filled in from the overnight calibration run; see `results/scorer/metrics.json`)*
