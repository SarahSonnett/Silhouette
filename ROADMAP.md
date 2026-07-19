# Silhouette Development Roadmap
### Realistic shape inversion → density & strength constraints

*Status: **Phases 1a, 1b and 2 are built and tested** (103 tests green). This
document is kept as the design record; see the status table below for what
landed and what changed along the way. Remaining open questions are flagged
**[DECIDE]**.*

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **1a** | convex forward model + fit at known period | ✅ `shapes.py`, `forward.py`, `inversion.py` |
| **1b** | parallel period scan | ✅ `scan_period`, `period_search_grid` |
| **2** | DEEVE → density & cohesion limits | ✅ `geophysics.py`, `geoplots.py` |
| **3** | integration, docs, plots | ◐ README/docs done; unified `analyze()` still open |

**Decisions resolved by building:**

- *Shape model* → **convex** (Kaasalainen–Torppa), as leaned toward here.
- *Renderer* → **none needed**. Convexity makes brightness a weighted sum over
  the Gaussian image, so no facet renderer or z-buffer was required. The
  planned `forward.py` became ~30 lines of linear algebra.
- *Scattering* → left **pluggable**, defaulting to Lommel–Seeliger + Lambert.
- *Uncertainties* → **uniform fractional σ** for archival data; deriving σ from
  point-to-point scatter is actively harmful on digitised DAMIT curves.
- *Period* → **input by default**, with `scan_period` for discovery.

**Lessons that changed the design:**

- Convex inversion is strongly **multi-modal**; a pole multistart is mandatory,
  and results must be reported as **pole families**, not a single winner.
- **Seed coverage beats fit effort** — `max_nfev` 200 vs 600 gave identical
  results, while 6 vs 25 starts decided success. Spend cores on starts.
- **`b/c` is ~20% uncertain even with a perfect pole**, so the geophysics takes
  ranges, not point values.

---

## 1. Vision

Grow Silhouette from an **analytical ellipsoid estimator** into a **light-curve
shape inverter with a geophysical payoff**:

1. Fit a *realistic* shape model — allowing imperfect / non-ellipsoidal bodies —
   to an input light curve (a model light curve, or data with error bars).
2. Use the recovered shape **plus** the spin state to constrain the **bulk
   density** and **cohesion / tensile strength** the asteroid would need to hold
   its shape together at its observed spin rate.

The current amplitude–aspect ellipsoid fit stays in the package as the **fast
initializer** (and a standalone quick-look tool); the new inversion refines it.

---

## 2. Where Silhouette is today (baseline)

| Module | Role |
|--------|------|
| `io.py` | flexible tabular photometry reader |
| `damit.py` | DAMIT multi-apparition light-curve reader (geometry embedded) |
| `apparitions.py` | epoch grouping, Fourier amplitude + mean-mag reduction |
| `geometry.py` | ecliptic aspect geometry (file columns or Horizons) |
| `model.py` | closed-form amplitude–aspect & mean-magnitude relations |
| `fit.py` | analytical `a:b, b:c, pole` least-squares (with mirror, degeneracy handling) |
| `plotting.py` | SpotLight-style multi-panel figure |

Reuses **SpotLight** (ellipsoid renderer) and **SpinDoc** (Fourier / H–G / period).
Worked examples: (15) Eunomia (DAMIT, multi-apparition), (16152) (single
apparition), synthetic self-check.

**Gap:** the fit only uses *one amplitude per apparition*. It never looks at the
actual brightness-vs-time curve, so it cannot see shape detail beyond a triaxial
ellipsoid, and cannot exploit calibrated absolute photometry fully.

---

## 3. Target architecture (new modules)

```
shapes.py       shape parameterizations, each exposing: facets (vertices+faces),
                volume, inertia tensor, and a DEEVE (equal-volume ellipsoid).
                  - Ellipsoid      (wraps current model)
                  - Convex          (Gaussian-image / support-function, Phase 1)
                  - [optional later] Cellinoid, SphericalHarmonic star-shape
forward.py      render brightness(rotation phase) for a shape at a given
                Sun/Earth geometry through a scattering law; returns model
                intensities directly comparable to observed light curves.
inversion.py    fit {shape params, pole λ,β, period P, scattering, per-LC
                scale/zero-point} to observed brightness(t) with σ, minimizing
                χ². Seeded by the amplitude–aspect fit. n_workers-bounded.
geophysics.py   DEEVE → Holsapple/Drucker–Prager rotational-stability limits:
                min density for cohesionless stability (light-curve only) and
                required cohesion vs density (needs a diameter).
```

Plotting extends with: 3-D shape render, per-apparition data-vs-model light-curve
overlays, and a cohesion–density constraint diagram.

---

## 4. Shape-model choice **[DECIDE]**

| Option | Pros | Cons | Effort |
|--------|------|------|--------|
| **Convex inversion** (Kaasalainen–Torppa) — *current lean* | DAMIT-grade; directly comparable to DAMIT; robust, well-posed via positive-definite parameterization; gives volume + inertia for geophysics | biggest build; can't represent concavities (slight volume overestimate) | High |
| Cellinoid (asymmetric octant ellipsoid) | lightweight; keeps interpretable axis ratios; fast | limited realism; still essentially ellipsoidal | Moderate |
| Spherical-harmonic star-shape | general non-convex bumps | needs full non-convex renderer + regularization; can be unstable | High |

**Working assumption for this doc:** Option 1 (convex). The other two can be
layered in later behind the same `shapes.py` interface if desired.

Renderer approach **[DECIDE]**: leaning **native facet renderer** inside
Silhouette (z-buffer + scattering, handles any shape), keeping SpotLight for the
ellipsoid mosaics. Alternative: extend SpotLight upstream (more reuse, tighter
coupling).

---

## 5. The convex-inversion method (technical reference)

Following Kaasalainen & Torppa (2001) [[1]](#references) and Kaasalainen,
Torppa & Muinonen (2001) [[2]](#references):

- **Parameterization.** The convex shape is described by its **Gaussian image**:
  the surface area element as a function of the surface-normal direction on the
  unit sphere, `G(θ,φ)`. Expand `log G` in spherical harmonics — the log keeps
  `G > 0`, which removes the ill-posedness and *guarantees a valid convex body*.
  Typical expansion degree ℓ ≈ 6–8 (~50–80 coefficients).
- **Forward brightness.** Disk-integrated brightness = Σ over surface facets of
  `albedo · S(μ, μ₀, α) · (projected area)`, with facets visible & illuminated
  (`μ, μ₀ > 0`). With the Gaussian-image form this is an integral over the unit
  sphere, cheap to evaluate.
- **Scattering law `S`** **[DECIDE default]**: Lommel–Seeliger + Lambert
  (`S = μμ₀/(μ+μ₀) + c·μμ₀`), the convex-inversion standard; `c` and the
  single-parameter phase function fit alongside shape.
- **Optimization.** Levenberg–Marquardt over {SH coefficients, pole (λ,β),
  period P, initial phase, scattering params, per-relative-LC scale}. Relative
  LCs contribute shape/relative info; calibrated LCs additionally pin absolute
  brightness.
- **Period.** Very tightly constrained → a *fine* grid of trial periods is
  needed; each trial is an independent LM fit ⇒ the **parallelizable** step.
- **Polyhedron reconstruction.** Solve the Minkowski problem (from the Gaussian
  image / facet normals + areas) to get the vertex model, for rendering, volume,
  and inertia.

Validation target: invert Eunomia's bundled DAMIT light curves and compare the
recovered convex model, pole, period, and DEEVE ratios against the DAMIT model.

---

## 6. Compute & parallelism (the core-budget question)

- A **single shape+pole fit with a known period** is single-threaded and light
  (seconds). Since period is usually known (SpinDoc / DAMIT), this is the common
  case and barely loads the machine.
- The **period scan** is the only heavy, parallel part. Expose **`n_workers`**
  as a first-class parameter across the period scan, pole grid, and bootstrap
  error bars, with a **conservative default**, so Silhouette can be pinned to a
  couple of cores and leave 8–10 free for other work.
- No implicit thread pools: cap BLAS threads too (`OMP_NUM_THREADS` etc.) so a
  background run never contends for cores it wasn't given.

---

## 7. Geophysics (the payoff)

From the fitted shape's **inertia tensor** → principal axes → **DEEVE**
(dynamically-equivalent equal-volume ellipsoid) effective `a:b:c`. Then apply the
Holsapple / Drucker–Prager rotational-stability analysis
(Holsapple 2001, 2004, 2007 [[3]](#references),[[4]](#references),[[5]](#references)):

- **Minimum density for cohesionless stability** — *light-curve only,
  size-independent.* Depends on shape ratios, spin ω, and friction angle φ
  through the dimensionless spin `ω̃ = ω/√(Gρ)`. Gives the smallest bulk density
  at which the body could be a strengthless rubble pile; if implausibly high, the
  body **must** have cohesion or be monolithic. *This is the clean headline.*
- **Required cohesion vs density** — needs an absolute **diameter** (wire in from
  the WISE/NEATM diameters in Simmer / PyLEADER). Curve of minimum cohesion (Pa)
  vs assumed density, marking the "must have strength" regime
  (cf. Scheeres et al. 2010; Sánchez & Scheeres 2014; Rozitis et al. 2014).
- **Friction-angle sensitivity** — φ (≈25–40°) is the main assumed unknown; show
  the limit band across that range.
- **Spin-barrier context** — place the object on the classic spin-period vs
  size/density diagram (Harris 1996; Pravec & Harris 2000).

**Caveats to surface in outputs:** uniform-density assumption; convex models
overestimate volume (bias effective ratios rounder); DEEVE reduction is the
practical bridge to the ellipsoid strength formulas, not a full FEM stress solve.

---

## 8. Phased plan

**Phase 1a — Convex forward model + fit, known period.**
`shapes.Convex`, `forward.py` (facet renderer + LS+Lambert), `inversion.py`
fitting SH coeffs + pole + scattering at a fixed period. Deliverable: invert
Eunomia at the DAMIT period, compare shape/pole to DAMIT. *Single-core, cheap.*

**Phase 1b — Parallel period scan.**
Add the trial-period grid with `n_workers` bounding. Deliverable: recover the
period from scratch on a synthetic case; core-capped so it stays background-safe.

**Phase 2 — Geophysics.**
`geophysics.py` (DEEVE + Holsapple limits), diameter input hook, constraint
plots. Deliverable: density/cohesion constraints for Eunomia and a fast-rotator
example.

**Phase 3 — Integration & docs.**
Unified `analyze()` pipeline (data → shape → spin → geophysics), plotting,
README, examples, tests.

---

## 9. Open decisions **[DECIDE]**

Resolved items are listed in the status section above. Still open:

1. **Non-convex shapes.** Convex models cannot represent concavities and
   overestimate volume. Worth adding a Cellinoid or SH star-shape behind the same
   `shapes.py` interface if concavity matters for the geophysics.
2. **Scattering parameters.** Currently the law is fixed during a fit; the
   Lambert weight `c` and a phase function could be fitted alongside the shape.
3. **Uncertainties on the fitted parameters.** Only χ² spread across pole
   families is reported. Bootstrap over light curves, or LM covariance, would
   give real error bars on `a/b`, `b/c`, and the pole.
4. **Aspect coverage as a third study axis.** The sampling/precision grid holds
   apparitions fixed at 6; aspect diversity is likely the dominant control on
   `b/c` and deserves its own study.
5. **Diameter provenance** for cohesion in pascals — wire in WISE/NEATM
   diameters from Simmer/PyLEADER rather than passing them by hand.
6. **Unified `analyze()`** pipeline (data → shape → spin → geophysics) and a
   period-scan CLI.
7. **Period aliasing.** `scan_period` finds the best period within a window, but
   periods separated by `P²/T` are genuine one-rotation aliases; alias
   identification/reporting is not yet automated.

---

## 10. References

1. Kaasalainen, M., & Torppa, J. (2001). *Optimization methods for asteroid
   lightcurve inversion. I. Shape determination.* **Icarus** 153, 24–36.
   [2001Icar..153...24K](https://ui.adsabs.harvard.edu/abs/2001Icar..153...24K)
2. Kaasalainen, M., Torppa, J., & Muinonen, K. (2001). *…II. The complete inverse
   problem.* **Icarus** 153, 37–51.
   [2001Icar..153...37K](https://ui.adsabs.harvard.edu/abs/2001Icar..153...37K)
3. Holsapple, K. A. (2001). *Equilibrium configurations of solid cohesionless
   bodies.* **Icarus** 154, 432–448.
   [2001Icar..154..432H](https://ui.adsabs.harvard.edu/abs/2001Icar..154..432H)
4. Holsapple, K. A. (2004). *Equilibrium figures of spinning bodies with
   self-gravity.* **Icarus** 172, 272–303.
   [2004Icar..172..272H](https://ui.adsabs.harvard.edu/abs/2004Icar..172..272H)
5. Holsapple, K. A. (2007). *Spin limits of Solar System bodies: From the small
   fast-rotators to 2003 EL61.* **Icarus** 187, 500–509.
   [2007Icar..187..500H](https://ui.adsabs.harvard.edu/abs/2007Icar..187..500H)
6. Scheeres, D. J., Hartzell, C. M., Sánchez, P., & Swift, M. (2010). *Scaling
   forces to asteroid surfaces: The role of cohesion.* **Icarus** 210, 968–984.
   [2010Icar..210..968S](https://ui.adsabs.harvard.edu/abs/2010Icar..210..968S)
7. Sánchez, P., & Scheeres, D. J. (2014). *The strength of regolith and rubble
   pile asteroids.* **Meteoritics & Planetary Science** 49, 788–811.
   [2014M&PS...49..788S](https://ui.adsabs.harvard.edu/abs/2014M%26PS...49..788S)
8. Rozitis, B., MacLennan, E., & Emery, J. P. (2014). *Cohesive forces prevent
   the rotational breakup of rubble-pile asteroid (29075) 1950 DA.* **Nature**
   512, 174–176. [2014Natur.512..174R](https://ui.adsabs.harvard.edu/abs/2014Natur.512..174R)
9. Pravec, P., & Harris, A. W. (2000). *Fast and slow rotation of asteroids.*
   **Icarus** 148, 12–20.
   [2000Icar..148...12P](https://ui.adsabs.harvard.edu/abs/2000Icar..148...12P)
10. Ďurech, J., Sidorin, V., & Kaasalainen, M. (2010). *DAMIT: a database of
    asteroid models.* **A&A** 513, A46.
    [doi:10.1051/0004-6361/200912693](https://doi.org/10.1051/0004-6361/200912693)

*See the main [README](README.md#references) for the amplitude–aspect method
references underlying the current baseline.*
