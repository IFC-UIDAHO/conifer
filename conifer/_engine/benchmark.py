"""
benchmark.py — RMSE-by-stand-type evaluation reproducing the deck's slide 7.

Computes RMSE of recovered ALR means (and of proportions) against the truth,
broken out by stand type, and relative RMSE (%) of each ML method vs the linear
multivariate-FH baseline (negative % = improvement, matching slide 7's
-35% / -26% / ~parity story).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .composition import alr_inv


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_by_stand_type(theta_hat, theta_true, types, scale="alr", edges=None):
    """Return dict {stand_type: rmse, 'overall': rmse}."""
    theta_hat = np.asarray(theta_hat); theta_true = np.asarray(theta_true)
    if scale == "proportion":
        a = alr_inv(theta_hat); b = alr_inv(theta_true)
    else:
        a = theta_hat; b = theta_true
    out = {}
    for t in np.unique(types):
        mask = types == t
        out[t] = _rmse(a[mask], b[mask])
    out["overall"] = _rmse(a, b)
    return out


def compare_methods(results: dict, theta_true, types, baseline="Linear FH",
                    scale="alr"):
    """results: {method_name: theta_hat (m,q)}.
    Returns (rmse_df, relative_df) where relative is % change vs baseline."""
    rows = {name: rmse_by_stand_type(th, theta_true, types, scale=scale)
            for name, th in results.items()}
    rmse_df = pd.DataFrame(rows).T          # methods x stand-types
    base = rmse_df.loc[baseline]
    rel = (rmse_df - base) / base * 100.0   # % change vs baseline
    rel_df = rel.round(1)
    return rmse_df.round(4), rel_df
