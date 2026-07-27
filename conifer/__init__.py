"""
CONIFER - COmpositional Nonlinear-debiased Inference, Fay-Herriot with Ellipsoidal conformal Regions.

Design-aware small-area estimation of forest structure as *distributions*, not just totals.
The engine is a compositional area-level Fay-Herriot with a cross-fitted, one-step-debiased
machine-learned mean and design-aware conformal prediction sets on the simplex. It reduces
exactly to classical Fay-Herriot when the mean is linear.

Two ways in
-----------
**From a tree list** (the forester's path - labels, units and plot design handled for you)::

    import conifer
    inv = conifer.from_treelist(trees, stand_col="STAND", plot_col="PLOT",
                                dbh_col="DBH_IN", baf=20, aux=lidar_metrics)
    est = inv.fit()
    conifer.conformalize_direct(est)              # no known truth needed
    cov = conifer.coverage_check(est)             # the number that earns trust
    conifer.report.to_html(est, "stand_report.html", coverage=cov)

**From matrices** (the statistician's path - unchanged since v0.1)::

    est = conifer.DiameterDistribution(seed=0).fit(counts, area_eff, X)
    est.conformalize(s_truth_cal, cal_idx, joint=True, alpha=0.10)
    lo, hi = est.predict_interval(joint=True)

Names
-----
DiameterDistribution : the estimator you fit (formerly ``StemDensityClassSAE``).
CompositionalFH      : the reusable engine underneath it (the same object today).
SpeciesComposition   : planned v0.3 sibling, on the same engine.
"""
from .estimators import DiameterDistribution, CompositionalFH, SpeciesComposition
from .io import (
    Inventory,
    Issue,
    from_treelist,
    from_matrices,
    read_stands,
    attach_estimates,
    FIA_1IN_BREAKS,
    FIA_2IN_BREAKS,
    DEFAULT_BREAKS_IN,
    DEFAULT_BREAKS_CM,
)
from .calibration import (
    conformalize_holdout,
    conformalize_naive,
    conformalize_direct,
    coverage_check,
)
from . import report, plots, io, calibration, demo

__version__ = "0.2.0"
__all__ = [
    "DiameterDistribution", "CompositionalFH", "SpeciesComposition",
    "Inventory", "Issue", "from_treelist", "from_matrices", "read_stands", "attach_estimates",
    "conformalize_holdout", "conformalize_naive", "conformalize_direct", "coverage_check",
    "report", "plots", "io", "calibration", "demo",
    "FIA_1IN_BREAKS", "FIA_2IN_BREAKS", "DEFAULT_BREAKS_IN", "DEFAULT_BREAKS_CM",
    "__version__",
]
