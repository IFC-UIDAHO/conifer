<!--
  Drop-in section for the GitHub README and the PyPI long-description.
  Suggested placement: right after the "Validation" section.
  Set the version number where marked. Current PyPI is conifer-sae 0.2.8;
  this note is written for the next release (e.g. v0.2.9).
-->

## Estimator versions — what the region results were computed on

The four field validations (Idaho St. Joe, and the southern Arkansas / Mississippi /
South Carolina ownerships) were run and reported on **CONIFER v0.1**. The design-based
simulation that accompanies them then surfaced — and fixed — a real failure mode, and that
fix shipped as **v0.2**, the current default. This note states plainly which results sit on
which version, because one region moves materially between them.

### The v0.2 fix (shipped as the default)

On weak-signal, small-sample populations the v0.1 estimator over-shrank: it did not converge
to the direct estimate as plots accumulated. The cause was a non-vanishing floor in the
analytic Dirichlet–multinomial sampling covariance — the global overdispersion `φ`, estimated
from **cross-stand** share variance, mistook genuine between-stand signal for within-stand
overdispersion, so the covariance settled on `κ → 1/(1+φ)` instead of vanishing. v0.2 restores
the multinomial covariance `κ = 1/nᵢ`, which vanishes with tallies and restores Fay–Herriot
consistency: the estimator now **provably converges to the direct estimate as data accumulate,
while keeping the sparse-regime gain** (shipped as `di_overdispersion=False`, the default; the
v0.1 behaviour is still available via `di_overdispersion=True` for reproducibility). A
design-based external covariance `D_ext` bypasses the internal covariance entirely and was
never affected — which is why prism / variable-radius cruises were immune.

### What actually changes between v0.1 and v0.2

The floor only bites where signal is weak and the sample is small. Re-running CONIFER on each
region's own data (identical seeds, each region's density convention):

| Region | CONIFER v0.1 | CONIFER v0.2 | change | verdict |
|---|---|---|---|---|
| Arkansas (449 stands) | 1.441 | 1.395 | −3.2% | stable — result unchanged |
| **Mississippi (46 stands)** | 0.728 | 0.599 | **−17.7%** | **the sparse-regime result flips** |
| South Carolina (33 stands) | 2.246 | 2.220 | −1.2% | stable — result unchanged |

*(Idaho, the strong-signal origin region, is expected to be similarly stable but has not yet
been formally re-certified on v0.2.)*

### The Mississippi re-test (current state)

Mississippi was reported as an honest **counter-case**: at k = 2–4 plots per stand the
well-sampled direct estimate beat CONIFER by +14% on v0.1. Re-measured on **v0.2** with an
**independent held-out-truth** design — fit on `k` plots, score against the mean of the
*held-out* plots, so the estimate and the truth share no data — the result becomes a textbook
small-area crossover:

| plots retained per stand, k | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| CONIFER − Direct (% of Direct RMSE) | **−15%** | **−10%** | **−7%** | **−4%** | +1% | +6% |

CONIFER wins where the data are thin and defers to the direct estimate as plots accumulate —
exactly the deferral this README documents. Paired bootstrap intervals exclude zero at every
k except the crossover (k ≈ 6); simple-random and systematic coarse-grid subsampling agree
within one percentage point; and both CONIFER and a plain linear-mean Fay–Herriot crush the
global-mean null, so the gain is model-based borrowing, not shrinkage to a constant. The
plasmode population in the simulation — built from Mississippi's own LiDAR covariates —
predicted this: it moved 0.722 → 0.452 across the same v0.1 → v0.2 fix, and the real MS v0.1
field number is 0.722.

### Honest scope

This re-test is **Mississippi-specific** and recent; Arkansas, South Carolina and Idaho have
not yet been re-certified on v0.2 (their results move ≤ ~3% and are not expected to change
qualitatively, but that re-run is pending). The v0.1 numbers were correct for the estimator
that produced them — the region reports simply predate the shipped fix. A methodology
manuscript and updated region reports documenting the v0.2 re-test are in preparation.

---

<!-- CHANGELOG entry to add for the release -->

## Changelog — v0.2.9  <!-- set the exact version -->

- **Docs:** clarified which field validations were computed on v0.1 vs v0.2, and added the
  Mississippi held-out-truth re-test showing the v0.2 sparse-regime crossover (see the
  "Estimator versions" section).
- No API or numerical change to the estimator itself in this release; `di_overdispersion=False`
  (v0.2) remains the shipped default, with `True` available for v0.1 reproducibility.
