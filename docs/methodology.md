# Methodology

CONIFER targets the vector of stems-per-acre by DBH class, $\mathbf s_i \in \mathbb R_{\ge 0}^K$, for each
small area (stand) $i$. It factors that vector into a **total** and a **composition** and models each with an
area-level Fay–Herriot estimator that borrows strength from auxiliary covariates.

$$
\mathbf s_i \;=\; N_i \,\boldsymbol\pi_i, \qquad N_i=\text{stems/acre (total)},\quad
\boldsymbol\pi_i=\text{class shares on the simplex}.
$$

## The estimator

1. **Composition (compositional area-level FH).** The class shares are mapped to additive-log-ratio (ALR)
   coordinates $\mathbf y_i$ with an **analytic sampling covariance** $D_i$ derived from the plot count
   likelihood (valid for tiny tallies — no bootstrap). A cross-fitted regularized ML mean
   $\hat{\mathbf m}(\mathbf x_i)$ (random-feature ridge ensemble, optionally boosted trees), constructed to
   **nest the linear Fay–Herriot**, provides the synthetic mean; a global–local correlated random effect
   $\Sigma_u$ shrinks toward it. The EBLUP is
   $\hat{\boldsymbol\theta}_i=\hat{\mathbf m}_i+G_i(\mathbf y_i-\hat{\mathbf m}_i)$ with gain
   $G_i=\Sigma_u(\Sigma_u+D_i)^{-1}$.
2. **Total (log-FH).** $\log N_i$ is estimated by a univariate Fay–Herriot on the log scale.
3. **Structural-zero hurdle.** A logistic presence model, blended with the empirical presence by tally size
   $n/(n+8)$, handles classes that are genuinely absent (no sawtimber in a regen stand) — the regime where a
   plain Gaussian FH degrades.
4. **Uncertainty.** A second-order, ML-aware MSE ($g_1+g_3+g_4+g_D$) is delta-transformed back to
   stems/acre. On top of it, **design-aware conformal prediction sets** (per-class marginal, joint $L_\infty$,
   or a minimum-volume ellipsoid) give finite-sample, group-valid coverage where the analytic interval does
   not.
5. **Design/model combination.** An optional **region-calibrated adequacy gate** decides, per stand, how much
   to trust the direct estimate versus the model — heuristic (`gate`) or risk-optimal self-tuning (`sure`).

The estimator **reduces to the linear multivariate Fay–Herriot** when the ML mean is linear and the ML-mean
variance term vanishes, so classical FH is a special case, not a competitor.

## What it guarantees, and what it does not

- The conformal sets are **finite-sample valid** under exchangeability within a calibration group; they are
  the honest uncertainty statement, and the package labels which guarantee (marginal vs joint) each output
  makes.
- The point estimator **wins in the data-poor regime** and, since v0.2, **converges to the design-direct
  estimate as plots accumulate** — see [The v0.2 refinement](refinement-v0.2.md).
- It does **not** manufacture signal that the covariates do not carry: on homogeneous, plot-rich populations
  the design-direct estimate is already near-optimal and CONIFER correctly defers to it.

The full derivation, MSE components, and novelty positioning are in the methodology manuscript (in
preparation) and the reference implementation `conifer.DiameterDistribution` (engine modules `stemclass.py`,
`gcounts.py`, `sampling_cov.py`, `uncertainty.py`).
