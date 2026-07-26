"""Diagnostic plots for CONIFER estimates.

v0.1 ships the core estimate-with-interval plot. The fuller diagnostic family developed
in the St. Joe study (PDF recovery by archetype, coverage-vs-width, competitor shootout,
QMD diagnostics) will be folded in as standard package figures in later releases.
"""
from __future__ import annotations
import numpy as np


def plot_estimate(est, area_index: int = 0, ax=None, joint: bool = True, alpha: float = 0.10):
    """Bar chart of the estimated diameter distribution for one area, with the
    conformal prediction interval as whiskers (call ``conformalize`` first for intervals).

    Parameters
    ----------
    est : fitted DiameterDistribution
    area_index : which area (row) to plot
    joint : use the joint (simultaneous) set if available, else per-class
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    s = np.asarray(est.s_hat_)[area_index]
    K = s.shape[0]
    x = np.arange(K)
    ax.bar(x, s, color="#3a7d44", alpha=0.85, label="estimate")
    try:
        lo, hi = est.predict_interval(joint=joint, alpha=alpha)
        lo, hi = np.asarray(lo)[area_index], np.asarray(hi)[area_index]
        ax.errorbar(x, s, yerr=[s - lo, hi - s], fmt="none", ecolor="#1b3a26", capsize=3,
                    label=f"conformal set ({int((1-alpha)*100)}%)")
        ax.legend(fontsize=8, frameon=False)
    except Exception:
        pass  # not conformalized yet — just show the point estimate
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k+1}" for k in range(K)])
    ax.set_xlabel("DBH class")
    ax.set_ylabel("stem density (per acre)")
    ax.set_title(f"CONIFER — area {area_index}")
    return ax


__all__ = ["plot_estimate"]
