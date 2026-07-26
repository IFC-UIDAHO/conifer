"""CONIFER quickstart — fit the diameter-distribution estimator on synthetic data.

    python examples/quickstart.py
"""
import numpy as np

import conifer

rng = np.random.default_rng(0)
m, K, p = 60, 5, 4                       # 60 areas, 5 DBH classes, 4 covariates

# latent size structure driven by the covariates -> class shares -> observed counts
X = rng.normal(size=(m, p))
shares = np.exp(X @ rng.normal(size=(p, K)))
shares /= shares.sum(1, keepdims=True)
area_eff = rng.uniform(0.5, 2.0, m)
counts = rng.poisson(shares * (area_eff[:, None] * 200)).astype(float)

# --- fit ---
est = conifer.DiameterDistribution(seed=0).fit(counts, area_eff, X)
print("CONIFER", conifer.__version__)
print("estimator:", type(est).__name__, "(engine:", conifer.CompositionalFH.__name__ + ")")
print("mean stem density by DBH class:", np.round(est.s_hat_.mean(0), 1))

# --- design-aware conformal set on a held-out calibration split ---
cal = np.arange(m // 2)                   # first half as calibration (illustrative)
s_truth = counts / area_eff[:, None]      # stand-in "truth" for the demo
est.conformalize(s_truth, cal, joint=True, alpha=0.10)
lo, hi = est.predict_interval(joint=True)
print("joint 90% set half-width (area 0):", np.round(((hi - lo) / 2)[0], 1))
