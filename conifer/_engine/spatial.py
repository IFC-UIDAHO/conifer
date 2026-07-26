"""spatial.py — Spatial DV-SAE: a spatial Fay-Herriot on the functional-PCA scores.

Adds a spatially-correlated random effect so stands borrow strength from NEIGHBORS,
not just from covariates and their own plots -- the regime where it matters most is
sparse / zero-plot stands (federal/FIA). Random effect covariance is separable
space x score: u ~ N(0, A_k * C(rho)) per score k, C(rho)_ij = exp(-d_ij/rho)
(exponential / Matern-1/2). This is the explicit, runnable form of the amortized
spatial-RE prior that the `vmsae` VAE encoder approximates at scale.
"""
from __future__ import annotations
import numpy as np


def exp_kernel(coords, rho):
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return np.exp(-d / max(rho, 1e-6))


def _profile(y, d, X, C):
    """GLS beta + profile-likelihood line search over process variance A."""
    best = None
    for A in np.geomspace(1e-3, 8.0, 18):
        V = A * C + np.diag(d)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            continue
        Vi = np.linalg.inv(V); XtVi = X.T @ Vi
        b = np.linalg.solve(XtVi @ X + 1e-8 * np.eye(X.shape[1]), XtVi @ y)
        r = y - X @ b
        ll = -np.sum(np.log(np.diag(L))) - 0.5 * r @ Vi @ r
        if best is None or ll > best[0]:
            best = (ll, A, b, Vi)
    return best


def spatial_fh_score(y, d, X, coords, rho_grid=None):
    """Spatial FH for one (FPCA) score. y (m,), d (m,) sampling var, X (m,p),
    coords (m,2). Returns theta (m,), var (m,), params."""
    if rho_grid is None:
        ext = np.ptp(coords, 0).mean(); rho_grid = ext * np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    best = None
    for rho in rho_grid:
        C = exp_kernel(coords, rho); res = _profile(y, d, X, C)
        if res is None: continue
        ll, A, b, Vi = res
        if best is None or ll > best[0]: best = (ll, rho, A, b, Vi, C)
    ll, rho, A, b, Vi, C = best
    r = y - X @ b
    theta = X @ b + A * (C @ (Vi @ r))
    M = A * C; postcov = M - M @ Vi @ M
    var = np.clip(np.diag(postcov), 1e-9, None)
    return theta, var, dict(rho=float(rho), A=float(A))


def spatial_fh_scores(scores, Dsc_diag, X, coords, rho_grid=None):
    """Apply spatial FH independently to each FPCA score column."""
    scores = np.asarray(scores, float); m, r = scores.shape
    Xd = np.hstack([np.ones((m, 1)), np.asarray(X, float)])
    Th = np.empty_like(scores); Var = np.empty_like(scores); params = []
    for k in range(r):
        th, v, p = spatial_fh_score(scores[:, k], Dsc_diag[:, k], Xd, coords, rho_grid)
        Th[:, k] = th; Var[:, k] = v; params.append(p)
    return Th, Var, params
