# CONIFER

**Compositional, design-aware small-area estimation of the forest diameter distribution, with conformal
prediction sets.**

CONIFER estimates stems-per-acre by diameter class for small areas (stands) by borrowing strength from
auxiliary data (LiDAR/NAIP canopy metrics) through a compositional area-level Fay–Herriot model with a
cross-fitted machine-learning mean, a structural-zero hurdle, and design-aware conformal prediction sets. It
is built for the regime real inventories live in: **few plots per stand, near-empty large-diameter tails, and
a need for honest uncertainty.**

```python
import conifer
est = conifer.DiameterDistribution().fit(counts, area_eff, X,
                                         total_logN=logN, var_logN=varN)
est.s_hat_          # stems/acre by DBH class, per stand
est.conformalize(s_truth_cal, cal_idx, joint=True).predict_interval()
```

## Why it exists

- **Wins where small-area estimation is meant to help** — sparse, data-poor stands — and
  **converges to the design-direct estimate when data are plentiful** (Fay–Herriot consistency; the optional
  v0.3 support-aware deferral gate closes the residual data-rich gap on real regions).
- **Honest uncertainty**: the analytic Gaussian interval under-covers on degenerate compositions; the
  design-aware conformal set is calibrated.
- **Validated on four real regions** (Idaho + Arkansas, Mississippi, South Carolina) and a design-based
  Monte-Carlo with known truth.

## Where to go next

- [Installation](installation.md) — `pip install conifer-sae`
- [Methodology](methodology.md) — the model, the estimator, the guarantees
- [The v0.2 refinement](refinement-v0.2.md) — the sampling-covariance fix a simulation surfaced
- [The v0.3 deferral](deferral-v0.3.md) — the support-aware reduce-to-direct gate for the data-rich regime
- [Benchmark](benchmark.md) — a runnable known-truth benchmark
- **Getting-started walkthrough** — `docs/vignettes/conifer-getting-started.ipynb` (rendered `.html` alongside)
- [Citation](citation.md)

!!! note "Status"
    CONIFER is released as open-source software; the methodology manuscript is in preparation. See
    [Citation](citation.md).
