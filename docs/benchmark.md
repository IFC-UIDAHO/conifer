# Benchmark

A short, self-contained, reproducible benchmark ships with the package at
[`examples/benchmark.ipynb`](https://github.com/IFC-UIDAHO/conifer/blob/main/examples/benchmark.ipynb). It
runs in a few seconds and reproduces the headline pattern on a synthetic population with a **known truth**.

## What it shows

1. **CONIFER wins in the data-poor (sparse) regime** and converges to the design-direct estimate as plots
   accumulate — the v0.2 [sampling-covariance refinement](refinement-v0.2.md).
2. **The design-aware conformal set is calibrated** near the nominal 0.90, while the analytic Gaussian
   interval under-covers.

## Run it

```bash
pip install conifer-sae
jupyter notebook examples/benchmark.ipynb        # or: jupyter lab
```

The notebook generates a 200-stand population from the compositional Fay–Herriot model, samples $k$ plots per
stand across $k \in \{1,2,3,5,8,15\}$, and scores CONIFER against the design-direct estimator and a k-NN
baseline on log-density RMSE, then measures joint conformal coverage against the known truth.

## The full study

The single-notebook benchmark is a small illustration. The full design-based Monte-Carlo — **eight forest
archetypes plus three plasmode populations built from real 3D-NAIP covariates**, the complete competitor
slate (spatial FH, BART-FH, SAEforest, Dirichlet, Weibull, k-NN, global mean), and every metric (RMSE, tail
RMSE, bias, conformal and Gaussian coverage, MSE calibration, benchmarking) across the sampling gradient —
lives in `PSAE_work/simulation/` (`run_simulation.py`, `make_figures.py`; see `SIMULATION_DESIGN.md` and
`LESSON_LEARNED_v0.2.md`).
