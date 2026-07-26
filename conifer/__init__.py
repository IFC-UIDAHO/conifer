"""
CONIFER - COmpositional Nonlinear-debiased Inference, Fay-Herriot with Ellipsoidal conformal Regions.

Design-aware small-area estimation of forest structure as *distributions*, not just totals. The
first release ships the diameter-distribution estimator: a compositional area-level Fay-Herriot
with a cross-fitted, one-step-debiased machine-learned mean and design-aware conformal prediction
sets on the simplex. It reduces exactly to classical Fay-Herriot when the mean is linear.

Quickstart
----------
>>> import conifer
>>> est = conifer.DiameterDistribution(seed=0).fit(counts, area_eff, X)
>>> s_hat = est.s_hat_                            # stem density by DBH class
>>> est.conformalize(s_cal, cal_idx, joint=True, alpha=0.10)
>>> lo, hi = est.predict_interval(joint=True)     # valid joint prediction set

Names
-----
DiameterDistribution : the estimator you fit (formerly ``StemDensityClassSAE``).
CompositionalFH      : the reusable engine underneath it (the same object today).
SpeciesComposition   : planned v0.3 sibling, on the same engine.
"""
from .estimators import DiameterDistribution, CompositionalFH, SpeciesComposition

__version__ = "0.1.1"
__all__ = ["DiameterDistribution", "CompositionalFH", "SpeciesComposition", "__version__"]
