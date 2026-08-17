# Changelog

All notable changes to CONIFER are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2026-08-17

Deploys the stronger mean learner the adequacy gate actually selects, and re-synchronizes the release
metadata. The default `fit()` and the public API are unchanged.

### Fixed
- **`fit_gated` now deploys the gradient-boosted (`HistGradientBoosting`) mean when the learner-adequacy
  contest selects the flexible learner**, instead of fitting the weaker random-feature `'ml'` model. The
  out-of-fold contest already ranked ridge against the boosted tree; the deployed fit now matches the
  learner that won it, so a region with a genuine nonlinear canopy–size link gets the model the gate chose.
  Where the contest selects `linear` (as on the regeneration-dominated Idaho and southern cruises), the
  deployed estimate is unchanged.

### Changed
- Release metadata re-synchronized: `CITATION.cff` advanced to the current version (it had drifted at
  0.3.0 through two releases), and the README acronym and version banner updated to the featured **Extremal**
  conformal set and the consolidated **v0.3** release line.

## [0.3.2] - 2026-08-13

Corrects the multivariate conformal set so its coverage guarantee holds in the plot-scarce regime, and
re-expands the CONIFER acronym to match the reported set. The default prediction set (the L∞ / max-score
band) and the public API are unchanged.

### Changed
- **`conformalize(mode='min_vol')` now estimates the ellipsoid shape on a disjoint calibration sub-fold**
  (new default `_mv_disjoint=True`). Estimating the shape in-sample on the same residuals it scores breaks
  score exchangeability and under-covers; the disjoint fold restores finite-sample coverage. The legacy
  in-sample shape is retained (`_mv_disjoint=False`) for ablation only. This changes the `min_vol` set's
  output; the default `maxscore` (L∞) band is unaffected.
- **`conformalize_holdout` averages the ellipsoid shape across repeated splits**, so the reported
  `(shape, threshold)` pair is mutually consistent.
- **Documentation now presents the L∞ / max-score band as the reported set**, with the minimum-volume
  ellipsoid as a data-rich-calibration variant that under-covers when stands per stratum are few
  (README, methodology, calibration docstrings).
- **Acronym: CONIFER now expands to "…Fay–Herriot with Extremal conformal Regions"** (was "Ellipsoidal"),
  matching the featured L∞ set. The package name (`conifer-sae`) is unchanged.

## [0.3.1] - 2026-08-13

Adds two **auxiliary-information adequacy gates** that make covariate use *no-harm*: the estimate with
covariates is never worse than the covariate-free estimate, while gains are kept where the auxiliary data
carry signal. Opt-in via `conifer.fit_gated`; default `fit()` behaviour is unchanged.

### Added
- **`conifer.fit_gated(...)`** + the **`conifer.adequacy`** module, with two data-driven gates:
  - **Learner-adequacy gate** — chooses the mean learner (`linear` ridge vs `ml` boosted-tree) by a 5-fold
    out-of-fold R² contest, so the flexible learner is used only where it generalizes. No sample-size
    threshold: at small n the boosted-tree OOF R² collapses and the linear mean is selected automatically.
  - **Covariate-adequacy gate** — caps the covariate mean by its out-of-fold trust
    `rho = OOF R²(X → fullest-cruised target)`; blends `s = rho_eff·s_cov + (1−rho_eff)·s_0cov` with
    `rho_eff = rho if rho ≥ rho_floor else 0` (default floor `0.30`, selected on the held-out battery). This
    stops the Fay–Herriot mean from over-leaning on weak covariates when the direct estimate is noisy.
  - Helpers `conifer.choose_learner(...)`, `conifer.covariate_adequacy(...)`; result `GatedResult` exposing
    `s_hat_`, `rho`, `rho_eff`, `learner`, and the underlying `est_cov_` / `est_0cov_` (for conformalization).

### Validated
- No-harm across four regions × {sparse, rich} (real engine): strictly no-harm on the Aitchison distribution
  **shape** in all eight cells (improves six), no-harm on combined log-count in seven of eight. A weak-covariate
  region (Mississippi, +17% ungated harm) becomes **exact no-harm** — the gate detects the weak signal
  (trust 0.21 < floor) and defers — while covariate gains are preserved where the signal is real (South
  Carolina, Idaho, Arkansas). Reproduced by `tests/test_adequacy.py`.

## [0.3.0] - 2026-08-11

Adds an **optional data-rich deferral** that closes the one honest gap in v0.2: in the data-rich regime the
design-direct estimator became reliable, but v0.2 kept over-shrinking toward the model and lost accuracy to it.
Default `fit()` behaviour is unchanged — the deferral only activates when you pass per-stand plot support, so
every previously reported v0.2 number is identical.

### Added
- **Support-aware reduce-to-direct gate** (`DiameterDistribution`, params `cv_defer=True`, `defer_c=8.0`,
  `defer_a=1.0`). When `fit(..., plots=<per-stand plot arrays>, direct_dens=<design-direct density>)` is
  supplied, the deployed estimate is blended toward the design-direct density by
  `w_ik = [k_i/(k_i+c)] · [n_ik/(n_ik+a)]`: the first factor grows the deferral as a stand's plot support grows
  (`c` = the plot count at which model and direct get equal weight); the second is an add-one support prior that
  defers a class only where the direct actually has tally, so structural-zero / low-tally classes keep the
  hurdle instead of chasing a noisy or zero direct. Exposes `.defer_w_` (m×K weights) and `.defer_c_`.
- `c=8` was selected on the held-out plasmode simulation; `a=1` is the canonical add-one constant (the
  simulation has no structural zeros, so it cannot calibrate `a` — the prior encodes the forestry reality the
  simulation omits).

### Fixed
- **Data-rich over-shrinkage.** With the gate, per-region log-density RMSE vs full-cruise held-out truth
  improves in every data-rich regime (Mississippi 0.275→0.260, Arkansas 0.461→0.363, South Carolina
  0.608→0.600 — now at the design-direct / mature-FH frontier) while every data-poor regime is preserved
  (largest change +0.002, at noise level). Validated on both the held-out plasmode (improves at every plot
  count) and the three real regions.

### Notes
- Backward compatible: omit `plots` → exactly v0.2. The gate reduces to v0.2 as `a→∞` and to the design-direct
  as `k→∞` in supported classes.
- An earlier held-out-plot cross-validation variant (STACK) was implemented and rejected: it won on the
  simulation but failed to transfer to the real regions (over-deferred at low plot count; its CV signal
  decoupled from truth).

## [0.2.9] - 2026-08-06

Maintenance release. Version bump only; no change to the estimator, the calibration, the API, or any
reported number since 0.2.8.

### Changed
- Version synchronized to 0.2.9 across `pyproject.toml`, `conifer/__init__.py`, and `CITATION.cff`.

## [0.2.8] - 2026-07-28

Docs and packaging only. No change to the estimator, the calibration, or any reported number.

### Added
- **A 60-second, zero-code tour** at the top of the README: an inline highlight loop
  (`docs/media/conifer-money.gif`) linked to the full screencast (`docs/media/conifer-demo.mp4`,
  with a poster), walking `pip install "conifer-sae[app]"` → `conifer-studio` → Load the demo
  cruise → the CONIFER-vs-direct result → measured coverage. Built from the real retuned demo
  output, not a mock-up.
- A **Preview Release** notice at the top of the README (visible on the PyPI project page), noting
  that APIs may change before the first stable 1.0.

## [0.2.7] - 2026-07-27

The demo cruise, retuned. `conifer.demo` now generates a silviculturally plausible, *stocked*
Inland-Northwest mixed-conifer cruise, and the workflow it showcases finally demonstrates the
method's actual advantage. **No change to the estimator, the calibration, or any reported number
from the real studies** - this release is the synthetic demo generator, the app copy, and the
tests only. The 0.2.6 UI/safety release is untouched.

### Changed
- **The demo was essentially bare ground; now it is a stocked forest.** The old generator was
  calibrated to a sparse figure (~19-25 stems/acre, ~4 ft2/ac basal area) that the project's own
  audited Idaho analysis had already flagged as a units error - a per-plot mean read as a per-acre
  density. The retuned generator produces a typical (median) stand of ~250 stems/acre, ~115 ft2/ac
  basal area and a ~10" quadratic mean diameter, with stands ranging from dense small-tree
  regeneration to open, large-tree old growth.
- **The demo now shows the small-area gain honestly.** The previous demo could not: its synthetic
  covariates were scrambled by a large random projection, so the class shares carried no learnable
  signal and CONIFER had nothing to borrow (it ran 10-20% *worse* than the direct estimator). The
  covariates are now driven by the same latent stand structure as the diameter distribution - as
  real LiDAR metrics are - so there is genuine strength to borrow, and the debiased-ML mean earns
  its place.
- **New `regime` argument to `make_cruise`** (`"sparse"` default, or `"rich"`), with a matching
  control in the Stand Structure Studio. In the sparse regime (2-3 plots per stand) CONIFER beats
  the design-direct estimator by ~10-15% in merchantable-class RMSE; in the rich regime (~20 plots
  per stand) it correctly converges to the direct estimate rather than beating it. Both stories are
  now visible from the app.
- **Smaller demo plots.** `DEMO_PLOT_AREA` is now 0.05 acre - a realistic small fixed plot, as in
  the real St. Joe cruise - so the sparse regime is genuinely thin and the gain is real. The app
  reads this constant, so its plot-size default follows automatically.
- **Honest copy throughout.** The `conifer.demo` docstring, `REAL_TARGETS` (kept as an alias of the
  new `DEMO_TARGETS`), and the app's demo description no longer imply the demo is calibrated to
  reused real inventory data. They state plainly that it is synthetic, and that CONIFER's accuracy
  claims rest on the real St. Joe and southern studies, not on this demo.

### Fixed
- The demo's per-acre summaries (TPA, basal area, QMD) are no longer an order of magnitude too low,
  so they read as a believable working forest rather than non-forest.
- The app smoke test's plausible-scale guard was pinned to the old ~24 TPA demo; it now bounds the
  stocked demo's density band while still catching an order-of-magnitude plot-design mismatch.

## [0.2.5] - 2026-07-27

Documentation and packaging only. No change to the estimator, the calibration, or any reported
number; the API is identical to 0.2.4.

### Changed
- The PyPI version badge now reads the live version from shields.io rather than a hardcoded
  string, so it can no longer fall a release behind. (0.2.4 shipped a README still badged 0.2.3.)
- `CITATION.cff` is synced to the released version (was left at 0.2.3 through the 0.2.4 release).
- The getting-started vignette was re-executed against this release, so its printed version
  banner and outputs match the shipped package rather than the 0.2.0 run it carried.

## [0.2.4] - 2026-07-27

### Changed
- **Stand Structure Studio: visual redesign.** The app's presentation layer was rebuilt — a
  generated CONIFER mark in two palettes (light for the dark hero, green for light surfaces),
  a topographic motif, a typographic scale (Fraunces / Inter / JetBrains Mono), and reusable
  `eyebrow` and `step` elements so the workflow reads as numbered stages rather than a wall of
  controls. `apps/forester/_preview.html` renders the styling statically, so a layout change
  can be eyeballed without launching Streamlit.
- **No change to the estimator, the calibration, or any reported number.** This release is
  presentation only; `conifer._engine`, `calibration`, and `report` are untouched.

## [0.2.3] - 2026-07-27

### Changed
- CI: moved GitHub Actions off the deprecated Node 20 runtime (checkout / setup-python / artifact actions bumped one major). No package code changes.

### Fixed
- **The map failed to draw for every metric.** `Could not draw the map: 'QMD (in)'`, and the
  same for total stems per acre and for uncertainty. `attach_estimates` slugs joined column
  names so they survive a shapefile or GeoPackage write — `QMD (in)` becomes `qmd_in`, since
  shapefiles cap field names at ten characters and reject parentheses. That is correct for a
  file on disk and wrong for a frame that is only plotted, and `plot_map` went on to look up
  the readable name, raising `KeyError` every time. `plot_map` now passes `gis_safe=False`;
  exports are untouched and still get safe names.

  It slipped through because the map only runs when a stand polygon layer is loaded, and no
  test loaded one. Two tests now cover it — one asserting every metric draws, one asserting
  that anything written to disk still gets slugged — so the two requirements cannot drift
  apart again.

## [0.2.2] - 2026-07-27

### Fixed
- **The Stand Structure Studio produced estimates ~117x too high on the demo cruise.**
  `conifer.demo` generates a fixed-area cruise at 0.2-acre plots, matching the real Idaho
  design, but the app's plot-design control still defaulted to variable-radius (prism/BAF)
  at 0.1 acres. Loading the demo and pressing Run therefore applied prism expansion factors
  to fixed-area tallies. Measured against the demo's known truth: RMSE 1402.57 and a mean of
  2854.8 TPA, against 1.20 and 25.0 TPA once corrected, with the truth at 24.4. Nothing was
  raised — the run completed and produced a confident report.
- **Structural fix, not just a corrected default.** `conifer.demo` now exports
  `DEMO_DESIGN`, `DEMO_PLOT_AREA`, `DEMO_BAF` and `DEMO_N_STANDS`, and the app reads its
  widget defaults from them. The two drifted apart precisely because the app hardcoded a
  guess about what the demo generates; now it cannot. A regression test pins the constants
  to the cruise `make_cruise` actually produces, and fails if the direct estimate and the
  known truth are more than a factor of two apart.

### Changed
- The app's interface text now matches the voice of the stand report and the vignette.
  Terminology is unchanged — the reader is a forester who can fact-check a p-value. The
  help text on the plot-id field now states the consequence plainly: without plot
  identifiers the intervals fall back to a method known to under-cover, so they will look
  reassuringly tight and be wrong.

## [0.2.1] - 2026-07-27

Documentation only. No change to the estimator, the package code, or any published API;
the 0.2.0 wheel is functionally identical to this one.

### Fixed
- **Documented `python -m conifer.studio` as the launch that does not depend on `PATH`.**
  On Windows, pip installs console scripts into `%APPDATA%\Python\Python3xx\Scripts`,
  which is frequently absent from `PATH` — so `conifer-studio` fails with
  `CommandNotFoundException` even though it installed correctly. `python -m conifer.studio`
  takes the same arguments and behaves identically on every platform. This is released as a
  patch because PyPI renders `README.md` on the project page, and someone installing from
  PyPI is exactly the person who hits this.

## [0.2.0] - 2026-07-26

### Changed
- **Compositional sampling covariance now vanishes with sample size (Fay–Herriot consistency fix).**
  A design-based Monte-Carlo study surfaced an over-shrinkage failure mode: on weak-covariate-signal
  populations the point estimate did **not** converge to the design-direct estimate as plot tallies
  grew (RMSE stayed on a model-anchored floor while the direct RMSE fell), and the reported `s_var`
  grew with sample size. Root cause: the analytic Dirichlet-multinomial covariance used a globally
  estimated overdispersion φ from `estimate_phi`, which conflates genuine **between-stand**
  compositional signal with **within-stand** overdispersion, driving φ small so `kappa=(nᵢ+φ)/(nᵢ(1+φ))`
  approached a non-vanishing floor `1/(1+φ)`. The default sampling covariance is now the multinomial
  `kappa=1/nᵢ`, which vanishes as tallies accumulate — the EBLUP now converges to the direct estimator
  in the data-rich regime while **retaining the sparse-regime borrowing gain** (verified: young-plantation
  k=1 unchanged at −39% vs direct; plasmode AR/MS now within ±0.03 of direct at all k, was +0.3 to +0.6).

### Added
- `DiameterDistribution(di_overdispersion=...)` — `False` (default) uses the new vanishing multinomial
  covariance; `True` restores the v0.1 Dirichlet-multinomial covariance for exact reproducibility.
  Genuine plot-level overdispersion, when identified from replicates, should be passed via `D_ext`.

- **A forester-facing layer, so the package can be used without writing a pipeline.** None of it
  touches the estimator; it is I/O, calibration bookkeeping, and reporting around the same engine.
  - `conifer.io` — `from_treelist()` turns a tree list (stand, plot, DBH) plus stand-level
    covariates into a validated `Inventory`. Handles fixed-area and variable-radius (prism/BAF)
    cruises, wiring the design-based `D_ext` for the latter; FIA 1"/2" class-break presets; units;
    shapefile/GeoPackage in (`read_stands`) and out (`attach_estimates`, with GIS-legal field
    names). **Stand identifiers are carried end to end** — previously a mis-sorted input silently
    produced wrong estimates for every stand with no warning. `validate()` reports problems in
    language a forester can act on.
  - `conifer.calibration` — `conformalize_holdout()` calibrates conformal prediction sets **without
    a known truth**, which no real inventory has. See "Fixed" below for why the obvious approach
    fails. `coverage_check()` measures realised coverage on the scale of the guarantee being made.
  - `conifer.report` — `stand_table`, `summary_table`, `class_summary`, `comparison_table`
    (design-direct beside CONIFER), `narrative()` (plain-language, every figure injected from the
    fit rather than generated), `to_excel()`, `to_html()`.
  - `conifer.plots` — figures with real DBH-class labels and forestry units: distribution,
    field-only-vs-CONIFER, coverage badge, data-gain, and a stand-polygon choropleth.
  - `conifer.demo` — a reproducible synthetic cruise with a known truth, for tests and demos.
  - `apps/forester/` — a Streamlit application ("Stand Structure Studio"): upload a tree list, get
    maps, stand tables, an Excel workbook and a printable stand report.
- `conifer.report.data_gain(est)` — per-stand γ = Su/(Su+Dᵢ), the EBLUP's own shrinkage weight,
  reported as "% of this estimate from this stand's own plots". Validated to rise with plot count
  (corr 0.88; 0.042 at one plot → 0.264 at nine).

### Fixed
- **Conformal sets are now calibrated correctly when the truth is unknown.** Calibrating against the
  design-direct estimate of the *same* plots used to fit is invalid: because `s_hat` is a shrinkage
  *of* `s_dir`, the residual `s_hat − s_dir = (1−w)(s_model − s_dir)` is mechanically smaller than
  the true error. Measured against a known truth this under-covered at **68% for a nominal 90% set**.
  `conformalize_holdout()` instead splits each stand's *plots* in two, fits on one half and
  calibrates against the other, which is independent by construction and errs wide rather than
  narrow. The broken approach is retained as `conformalize_naive()`, documented and covered by a
  regression test, so the failure stays reproducible.
- **Reported intervals are per-class by default, not the joint simplex set.** The joint band paid a
  full K-class multiplicity price for a simultaneous guarantee that few operational questions need,
  which made the reported interval ~29× the estimate. A per-class interval answers the question a
  forester actually asks and is ~3.4× at 0.953 measured per-class coverage. `joint=True` still gives
  the simultaneous claim, and every output states which guarantee it is making.
- Prediction bounds are floored at zero in the reporting layer; the engine's standardized score scale
  could otherwise return a negative lower bound on stem density.
- `calibration._fit_kwargs` reads the estimator's constructor by introspection instead of a
  hand-maintained list. A hardcoded list had gone stale, so the holdout refit was silently running a
  different model configuration than the one being reported. Guarded by a test.

## [0.1.3] - 2026-07-26

### Added
- **Getting-started vignette** (`docs/vignettes/conifer-getting-started.ipynb`, executed, plus a
  rendered `.html`): an `sae`-style, end-to-end walkthrough on fully reproducible synthetic
