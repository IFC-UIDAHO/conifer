# Changelog

All notable changes to CONIFER are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-26

Initial public release.

### Added
- `conifer.DiameterDistribution` — compositional area-level Fay–Herriot estimator of the forest
  diameter distribution, with a cross-fitted one-step-debiased machine-learned mean and design-aware
  conformal prediction sets on the simplex. Reduces exactly to classical Fay–Herriot when the mean
  is linear.
- `conifer.CompositionalFH` — the reusable engine (target-agnostic core; alias of the estimator today).
- `conifer.SpeciesComposition` — placeholder for the planned v0.3 species-shares estimator.
- `conifer.plots.plot_estimate` — estimate-with-conformal-interval diagnostic plot.
- `conifer fit` command-line interface (`--counts / --area / --aux / --out`).
- Continuous integration on Python 3.9–3.12; smoke and naming tests.

### Notes
- Developed and validated on the St. Joe (Idaho) study (see the accompanying project report).
- The estimator was previously named `StemDensityClassSAE`; that name is retained as a
  backward-compatible alias.

[Unreleased]: https://github.com/IFC-UIDAHO/conifer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IFC-UIDAHO/conifer/releases/tag/v0.1.0
