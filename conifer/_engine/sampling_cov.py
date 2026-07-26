"""
sampling_cov.py — estimating / smoothing the q x q area sampling covariance D_i.

In the area-level FH measurement model  y_i = theta_i + e_i,  e_i ~ N_q(0, D_i),
the D_i are assumed *known*. In practice they must be estimated from the survey
(few plots per stand), which is the main practical hurdle noted in the deck
(slide 8) and our notes. We estimate D_i as the covariance of the per-plot ALR
vectors divided by the number of plots, then smooth it for stability with few
plots: ridge toward a pooled / diagonal target and enforce positive-definiteness.
"""
from __future__ import annotations
import numpy as np


def _nearest_pd(A: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Symmetrize and clip eigenvalues to make A positive-definite."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, jitter, None)
    return (V * w) @ V.T


def estimate_Di(plot_alr: np.ndarray, shrink: float = 0.3,
                pooled_target: np.ndarray | None = None) -> np.ndarray:
    """Estimate one area's sampling covariance from its per-plot ALR vectors.

    plot_alr : (n_plots, q)
    shrink   : weight in [0,1] toward the smoothing target (Ledoit-Wolf style).
    pooled_target : optional q x q target (e.g. cross-stand pooled D); if None a
                    diagonal target built from this stand's variances is used.
    Returns q x q PD matrix.
    """
    plot_alr = np.atleast_2d(np.asarray(plot_alr, dtype=float))
    n, q = plot_alr.shape
    if n < 2:
        # cannot estimate covariance from <2 plots; fall back to target/identity
        base = pooled_target if pooled_target is not None else np.eye(q)
        return _nearest_pd(base.copy())
    S = np.cov(plot_alr, rowvar=False, ddof=1)        # plot-level covariance
    S = np.atleast_2d(S)
    D_raw = S / n                                      # variance of the mean
    target = pooled_target if pooled_target is not None else np.diag(np.diag(D_raw))
    D = (1.0 - shrink) * D_raw + shrink * target
    return _nearest_pd(D)


def pooled_Di(list_plot_alr: list[np.ndarray]) -> np.ndarray:
    """Build a pooled diagonal-ish target by averaging per-stand raw D_i."""
    mats = []
    q = None
    for pa in list_plot_alr:
        pa = np.atleast_2d(pa)
        if pa.shape[0] >= 2:
            S = np.cov(pa, rowvar=False, ddof=1) / pa.shape[0]
            mats.append(np.atleast_2d(S))
            q = S.shape[0]
    if not mats:
        return None
    return _nearest_pd(np.mean(mats, axis=0))


def design_Di_from_plots(plot_stand, plot_spa, m, K, shrink=0.4, eps=0.5):
    """DESIGN-BASED sampling covariance of the ALR direct estimate, from plot replicates.
    Works for ANY plot design (incl. variable-radius/prism) because it uses the per-plot
    EXPANSION-WEIGHTED stems/acre, not a multinomial count assumption.
      plot_spa : (P,K) per-plot stems/acre by class ; plot_stand : (P,) stand index of each plot.
    For stands with >=2 plots: D_i = among-plot Cov(ALR)/n_i, shrunk toward a pooled GVF target.
    For 0-1 plot stands: pooled per-plot covariance / max(n_i,1) (borrow strength).
    """
    import numpy as np
    q = K - 1
    sp = np.clip(np.asarray(plot_spa, float), 0, None) + eps
    sh = sp / sp.sum(1, keepdims=True)
    palr = np.log(sh[:, :-1] / sh[:, -1:])            # per-plot ALR (ref = last class)
    Ds = np.zeros((m, q, q)); ncnt = np.zeros(m, int); perplot = []
    for i in range(m):
        sel = plot_stand == i; ni = int(sel.sum()); ncnt[i] = ni
        if ni >= 2:
            C = np.cov(palr[sel].T, ddof=1)
            if C.shape == (q, q):
                Ds[i] = _nearest_pd(C / ni); perplot.append(C)
    pool = _nearest_pd(np.mean(perplot, axis=0)) if perplot else np.eye(q)
    for i in range(m):
        ni = max(ncnt[i], 1); target = pool / ni
        Ds[i] = (1 - shrink) * Ds[i] + shrink * target if ncnt[i] >= 2 else target
    return Ds
