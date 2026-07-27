# Changelog

All notable changes to CONIFER are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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