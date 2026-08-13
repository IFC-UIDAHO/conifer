# The v0.3 deferral — the support-aware reduce-to-direct gate

CONIFER v0.2 fixed the sampling-covariance floor so the estimator **converges to the design-direct estimate
as plots accumulate** on the simulation. On the real regions, though, a residual **data-rich over-shrinkage**
remained: where the design-direct had become reliable, the estimator still leaned too far on the model and lost
accuracy to it (e.g. Arkansas data-rich log-density RMSE 0.461 vs the direct's 0.344). v0.3 closes that gap with
an **optional** deferral that is off by default and changes no previously reported v0.2 number.

## What it does

When you pass a stand's per-plot support to `fit`, the deployed estimate is blended toward the design-direct
**density** `D` class by class:

$$\hat s^{(v3)}_{ik} = (1-w_{ik})\,\hat s^{(v2)}_{ik} + w_{ik}\,D_{ik},
\qquad w_{ik} = \underbrace{\frac{k_i}{k_i+c}}_{\text{adequacy}} \cdot \underbrace{\frac{n_{ik}}{n_{ik}+a}}_{\text{support prior}}$$

- **Adequacy** `k_i/(k_i+c)` — the deferral grows with the stand's plot count `k_i`; `c` is the plot count at
  which model and direct get equal weight. `c = 8` ("equal weight at 8 plots") was selected on the held-out
  plasmode simulation, where the deferral direction helps at every plot count; the real regions confirm it
  preserves the data-poor regime (w ≈ 0.27 at 3 plots). The simulation's clean direct cannot calibrate the
  data-poor safety, so `c` is a design choice, not a fit parameter.
- **Support prior** `n_{ik}/(n_{ik}+a)` — a class is deferred to the direct only where the direct actually has
  tally `n_{ik}`. `a = 1` is the canonical add-one constant. This keeps the v0.2 **hurdle** on structural-zero /
  low-tally classes instead of chasing a noisy or zero direct. The simulation has no structural zeros and so
  cannot calibrate `a`; the prior encodes the forestry reality the simulation omits.

It blends the **density** (total and shares together), because on real data the per-class expansion varies, so
reducing toward the direct requires the actual design-direct density, not a count-share reconstruction.

Limits: reduces to v0.2 when no plot support is supplied (`a → ∞` likewise); reduces to the design-direct as
`k → ∞` in supported classes.

## Why it is honest

v0.3 does **not** beat the best mature Fay–Herriot in the data-rich regime — it **reaches the design-direct /
mature-FH frontier** there while keeping v0.2's decisive data-poor advantage. Net, it is no-harm across every
real region × regime cell tested:

| region · regime | v0.2 | v0.3 | Direct | best FH |
|---|---|---|---|---|
| Mississippi · sparse | 0.565 | 0.559 | 0.729 | 0.617 (sae.prop) |
| Mississippi · rich | 0.275 | **0.260** | 0.227 | 0.234 (cpag051) |
| Arkansas · sparse | 0.772 | 0.765 | 1.069 | 0.897 (cpag051) |
| Arkansas · rich | 0.461 | **0.363** | 0.344 | 0.326 (cpag051) |
| South Carolina · sparse | 0.913 | 0.915 | 1.361 | 1.245 (cpag051) |
| South Carolina · rich | 0.608 | **0.600** | 0.794 | 0.717 (msae) |

(log-density RMSE vs full-cruise held-out truth; worst cell SC-sparse +0.002, at noise level.) On the held-out
plasmode the gate improves at every plot count. South Carolina is the discriminating case: its structural-zero
tail makes the raw direct unreliable *even when rich*, so the support prior keeps the hurdle there — a uniform
plot-count-only gate broke SC-rich to 0.653; the support gate fixes it to 0.600.

An earlier held-out-plot cross-validation variant (STACK) was implemented and **rejected**: it won on the
simulation but failed to transfer to the real regions — it over-deferred at low plot count and its CV signal
decoupled from truth. The real-region test was decisive; the simulation alone would have shipped a broken fix.

## How to use it

```python
est = model.fit(
    counts, area_eff, X, groups=groups, total_logN=logN, var_logN=vlogN,
    plots=plots,          # list of per-stand plot arrays (supplies k_i and n_ik)
    direct_dens=D,        # the design-direct class density to defer toward
)
est.defer_w_              # (stands x classes) deferral weights actually applied
```

Parameters live on the estimator: `DiameterDistribution(..., cv_defer=True, defer_c=8.0, defer_a=1.0)`.

See the `CHANGELOG` `[0.3.0]` entry for the full record.
