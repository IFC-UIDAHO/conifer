# The v0.2 refinement — the sampling-covariance floor

CONIFER v0.2 corrects a real over-shrinkage failure mode that a design-based Monte-Carlo study surfaced. It
is reported here as a result, not concealed — and it is the reason the estimator now satisfies Fay–Herriot
consistency.

## The symptom

On weak-covariate-signal populations (the "plasmode" populations built from real LiDAR/NAIP covariates), the
point estimator **did not converge to the design-direct estimate as plot tallies grew**: its RMSE stayed on a
model-anchored floor while the direct estimator's RMSE fell, and the reported per-class variance *grew* with
sample size instead of shrinking.

## The cause

The compositional sampling covariance used a Dirichlet–multinomial form
$\kappa=(n_i+\phi)/(n_i(1+\phi))$. The overdispersion $\phi$ was estimated **globally** from the
**cross-stand** share variance — which is dominated by genuine **between-stand** compositional signal. That
signal was misread as **within-stand** overdispersion, driving $\phi$ small so that

$$
\kappa \;\longrightarrow\; \frac{1}{1+\phi} \quad(\text{a non-vanishing floor, as } n_i\to\infty).
$$

With $D_i$ floored, the EBLUP gain $G_i=\Sigma_u(\Sigma_u+D_i)^{-1}$ can never approach the identity, so the
estimate stays anchored to the (biased, weak-signal) covariate mean no matter how many plots arrive.

## The fix

Default to the **multinomial** sampling covariance $\kappa=1/n_i$, which **vanishes** as tallies accumulate.
The EBLUP gain then $\to I$ and the estimate resolves onto the design-direct estimate in the data-rich regime
— the Fay–Herriot consistency the model should have — **while preserving the sparse-regime borrowing gain**
(the two covariances nearly coincide at small $n_i$).

```python
conifer.DiameterDistribution(di_overdispersion=False)   # default (v0.2, vanishing multinomial)
conifer.DiameterDistribution(di_overdispersion=True)    # legacy v0.1 Dirichlet-multinomial, for reproducibility
```

Genuine plot-level overdispersion, when it is actually identified from replicate plots, should be supplied
through the design-based `D_ext` argument rather than inferred from cross-stand variance.

## Two independent confirmations

- **Preserved where it should be, corrected where it should be.** On the strongest archetype the sparse-regime
  win is unchanged (young-plantation, $k{=}1$: −39% vs direct); on the weak-signal plasmode populations the
  data-rich gap closes from +0.3…+0.6 log-RMSE to within ±0.03 of direct at every $k$ (e.g. plasmode
  Mississippi, real run: CONIFER **wins** at $k{=}2$, 0.573 vs 0.600, and tracks Direct within ~0.02–0.03
  thereafter).
- **Mechanism isolated.** Supplying a design-based `D_ext` (which bypasses the internal covariance entirely)
  removes the failure **even in legacy mode** — so the floor, not the Dirichlet–multinomial model, is
  demonstrably the cause. This also explains why prism / variable-radius cruises, which always supply a
  design-based `D_ext`, were never affected.

See the `CHANGELOG` `[0.2.0]` entry and `PSAE_work/simulation/LESSON_LEARNED_v0.2.md` for the full write-up.

> **Successor:** the real-region data-rich gap this refinement leaves open is closed by the optional
> v0.3 [support-aware deferral gate](deferral-v0.3.md).
