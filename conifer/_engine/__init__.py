"""
mvml_sae — Multivariate / Multi-output ML-based Small Area Estimation
====================================================================

Reference implementation of the PSAE "Multi-output ML-SAE Integration"
(PSAE_02-05-26_UofI deck) for stand-level diameter-distribution recovery.

Two estimation paths, identical I/O (DBH-class proportions in, recovered
proportions + Johnson's S_B parameters + uncertainty out):

  Path A  (deck)   : RF synthetic mean  ->  multivariate robust EBLUP (REBLUP)
                     on ALR-transformed DBH-class proportions  ->  bootstrap MSE.
                     `path_a_rf_reblup.RFReblupSAE`

  Path B  (novel)  : joint Bayesian multivariate Random-Weight-NN Fay-Herriot
                     (Gibbs; matrix-normal B, IW Sigma_u) -> native posterior
                     credible intervals.  Direct multi-output extension of
                     Parker's `nlfh::fit_fh_rnn`.
                     `path_b_bayes_rnnfh.BayesianMVRnnFH`

Shared layers: composition (ALR), data (synthetic stands + D_i),
jsb_recovery (Johnson's S_B), benchmark (RMSE by stand type).

See README.md for the math and the nlfh/msae mapping.
"""

from .composition import alr, alr_inv, proportions_from_counts, dbh_class_edges
from .data import generate_stand_data, StandData
from .path_a_rf_reblup import RFReblupSAE
from .path_b_bayes_rnnfh import BayesianMVRnnFH
from .jsb_recovery import recover_johnson_sb, johnson_sb_cdf
from .benchmark import rmse_by_stand_type, compare_methods

__all__ = [
    "alr", "alr_inv", "proportions_from_counts", "dbh_class_edges",
    "generate_stand_data", "StandData",
    "RFReblupSAE", "BayesianMVRnnFH",
    "recover_johnson_sb", "johnson_sb_cdf",
    "rmse_by_stand_type", "compare_methods",
]

__version__ = "0.1.0"

# Distribution-Valued SAE (DV-SAE) — new flagship distribution-valued estimator
from . import distributional  # noqa: E402,F401

from . import forestsim  # noqa: E402,F401
from . import gcounts  # noqa: E402,F401
from . import spatial  # noqa: E402,F401
