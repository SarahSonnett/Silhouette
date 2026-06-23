# Silhouette — Analytical Asteroid Shape & Pole Fitter

**Silhouette** takes tabular asteroid light-curve photometry and analytically
fits the triaxial axis ratios **a:b** and **b:c** together with the rotation
**pole** ecliptic longitude and latitude. It is, in effect, the *inverse* of
[SpotLight](https://github.com/SarahSonnett/SpotLight): where SpotLight renders a synthetic light curve from a
known ellipsoid and viewing geometry, Silhouette recovers the ellipsoid and pole
from observed brightness variations — and renders the result in a multi-panel
figure modelled on SpotLight's combined output.

The fit is **analytical**: it uses closed-form ellipsoid relations
(amplitude–aspect and the mean-magnitude/projected-area method) and a light
nonlinear least-squares solve — no convex inversion or shape facets.

![Silhouette fit summary](docs/images/fit_summary.png)

---

## How it works

For a triaxial ellipsoid (`a ≥ b ≥ c`, spinning about `c`) observed at **aspect
angle** `θ` (the angle between the line of sight and the spin axis):

- **Amplitude:**
  `A(θ) = 2.5·log₁₀(a/b) − 1.25·log₁₀[(a²cos²θ + c²sin²θ)/(b²cos²θ + c²sin²θ)]`
- **Aspect from a candidate pole:**
  `cos θ = sin β·sin βₚ + cos β·cos βₚ·cos(λ − λₚ)`
- **Mean magnitude:** from the rotation-averaged projected area, which brightens
  toward pole-on (the full `a·b` face) and fades toward equator-on. Its variation
  between apparitions breaks the `b/c` + pole degeneracy that amplitude alone
  cannot.

Silhouette groups the photometry into apparitions, reduces each to an amplitude
and a mean reduced magnitude, then fits `(a/b, b/c, λₚ, βₚ)` jointly by weighted
least squares over a grid of pole starting points.

The amplitude–aspect and magnitude–aspect (mean-magnitude) relations and their
simultaneous solution for a triaxial-ellipsoid pole and shape follow
Michałowski (1993) [[1]](#references), building on the amplitude–magnitude method
of Zappalà & Knežević (1984) [[2]](#references) and the pole/shape methodology
reviewed in Magnusson et al. (1989) [[3]](#references); the photometric
reduction uses the IAU H–G phase function of Bowell et al. (1989)
[[4]](#references).

### What is and isn't recoverable

| Apparitions | Result |
|-------------|--------|
| **≥ 4**, spread in ecliptic longitude | Full `a:b`, `b:c`, and pole `(λ, β)` |
| **2–3** | Fit attempted, flagged as weakly constrained |
| **1** | `a/b` **lower bound** only (equatorial aspect assumed); pole and `b/c` undetermined |

The amplitude and mean-magnitude observables depend only on `sin²θ`/`cos²θ`, so
the prograde/retrograde **mirror pole** `(λₚ+180°, −βₚ)` is exactly degenerate
and is always reported alongside the best solution. Breaking it requires
epoch/timing information, which Silhouette does not currently use.

---

## Reuse of sibling repositories

Silhouette imports two of its siblings when they are on the path, and falls back
to a vendored minimal copy otherwise (so it always runs standalone):

- **[SpotLight](https://github.com/SarahSonnett/SpotLight)** — the forward triaxial renderer, used to draw the
  best-fit ellipsoid mosaic.
- **[SpinDoc](https://github.com/SarahSonnett/SpinDoc)** — the Fourier light-curve model (`fourier`) and IAU
  H–G phase function (`HGfunction`), used during per-apparition reduction.

Check `silhouette.HAVE_SPOTLIGHT` / `silhouette.HAVE_SPINDOC` to see what was
found.

---

## Installation

```bash
git clone git@github.com:SarahSonnett/Silhouette.git
cd Silhouette
pip install -r requirements.txt
```

`astroquery` is optional — it is needed only when geometry is fetched from JPL
Horizons rather than supplied in the input file.

---

## Input format

A whitespace- or comma-delimited table with a one-line header. Column names are
auto-recognised from a broad alias set; required canonical fields are
`time` (MJD or JD), `mag`, `merr`, `rhelio`, `delta`, `alpha`. Optional
`ecl_lon`/`ecl_lat` (observer-centric ecliptic coordinates of the target)
make the file fully self-contained; otherwise Silhouette fetches them from
Horizons. The SpinDoc-style calibrated photometry file
(`Frame Rhelio Delta alpha … MJD TmagCorr … TmagFinalErr`) is read directly.

```
MJD        mag      merr   Rhelio  Delta   alpha  ecl_lon  ecl_lat
58000.123  18.421   0.020  2.71    1.78    6.3    34.21    -1.05
...
```

---

## Quick start

```python
from silhouette import (read_photometry, reduce_apparitions,
                        resolve_geometry, fit_shape, save_summary)

phot = read_photometry("photometry.txt", object_name="433")
apps = reduce_apparitions(phot, period=0.2194)     # rotation period in days
resolve_geometry(apps, target="433")               # file columns, else Horizons
fit  = fit_shape(apps)

print(fit.summary())
save_summary(fit, "fit_summary.png")
```

### Command line

```bash
python fit_shape.py --infile photometry.txt --period 0.2194 \
    --object 433 --outdir results
```

Writes `results/BestFitParameters.txt` and `results/fit_summary.png`.

---

## Worked example: asteroid (16152)

The repo bundles a real single-apparition r-band light curve of **(16152)** in
[`data/16152_2019_rp.txt`](data/16152_2019_rp.txt) (≈430 points, 2019 Aug–Sep;
the same calibrated photometry used by [SpinDoc](https://github.com/SarahSonnett/SpinDoc)).

```bash
python example_16152.py
```

```
Loaded 434 points; ecliptic columns present: False
Grouped into 1 apparition(s).
  span 50.8 d, amplitude 0.425 ± 0.019 mag

Silhouette shape + pole fit
  apparitions used : 1
  a:b = 1.479
  b:c = undetermined (single apparition)
  a:b is a LOWER BOUND (equatorial aspect assumed)
```

Because this is a **single apparition**, Silhouette returns only an `a/b ≥ 1.48`
lower bound — the pole and `b/c` are not recoverable from one viewing geometry.
This is the correct, honest result, and it is consistent with an elongated body:
the [DAMIT](https://damit.cuni.cz/projects/damit/?q=16152) convex models of
(16152) give a spin pole near `(λ, β) ≈ (115°, 63°)` or `(305°, 68°)` with a
22.936 h period, but those were derived from **many** apparitions of combined
dense and sparse photometry. Reproducing a full Silhouette pole + `b/c` solution
likewise requires multi-apparition coverage.

### Full-capability demo (synthetic, multi-apparition)

To exercise the complete pole + shape solution and produce the headline figure
above:

```bash
python example.py
```

This synthesises a multi-apparition data set for a *known* ellipsoid and pole
(`a:b=1.6, b:c=1.3, pole=(60°, 35°)`) and recovers them as a self-check.

---

## Output figure

The summary figure mirrors SpotLight's combined layout:

1. **Ellipsoid mosaic** — the best-fit shape rendered through one rotation (via
   SpotLight when available).
2. **Amplitude–aspect** — observed amplitudes with the analytical model curve.
3. **Mean magnitude–aspect** — the projected-area brightness variation and fit.
4. **Pole solution map** — χ² over the ecliptic sky, with the best pole and its
   degenerate mirror marked.

---

## Caveats

- The amplitude–aspect method assumes **geometric scattering** (brightness ∝
  projected area). Real surfaces (and non-geometric scattering laws such as
  SpotLight's default Lambertian) introduce amplitude/aspect deviations; treat
  recovered axis ratios as model-dependent.
- Robust pole determination needs several apparitions well spread in ecliptic
  longitude; sparse coverage yields non-unique solutions — inspect the candidate
  poles and the pole map.

---

## Testing

```bash
python -m pytest tests/
```

---

## References

1. Michałowski, T. (1993). *Poles, shapes, senses of rotation, and sidereal
   periods of asteroids.* **Icarus** 106, 563–572.
   [doi:10.1006/icar.1993.1193](https://doi.org/10.1006/icar.1993.1193)
   — amplitude–aspect and magnitude–aspect relations for a triaxial ellipsoid,
   solved simultaneously for pole and shape.
2. Zappalà, V., & Knežević, Z. (1984). *Rotation axes of asteroids: Results for
   14 objects.* **Icarus** 59, 436–455.
   [doi:10.1016/0019-1035(84)90112-X](https://doi.org/10.1016/0019-1035(84)90112-X)
   — the (improved) amplitude–magnitude method combining light-curve amplitude
   and mean brightness.
3. Magnusson, P., Barucci, M. A., Drummond, J. D., et al. (1989). *Determination
   of pole orientations and shapes of asteroids.* In **Asteroids II**
   (R. P. Binzel, T. Gehrels, M. S. Matthews, eds.), pp. 66–97. Univ. of Arizona
   Press. [bibcode:1989aste.conf...66M](https://ui.adsabs.harvard.edu/abs/1989aste.conf...66M)
   — review of photometric pole/shape determination methods.
4. Bowell, E., Hapke, B., Domingue, D., et al. (1989). *Application of
   photometric models to asteroids.* In **Asteroids II**, pp. 524–556. Univ. of
   Arizona Press.
   [bibcode:1989aste.conf..524B](https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B)
   — the IAU H–G magnitude system used for phase correction.

---

*Author: S. Sonnett. Part of an asteroid photometry toolset alongside SpotLight,
SpinDoc, and WISETrails.*
