"""distributional.py — Distribution-Valued Small Area Estimation (DV-SAE).

The award-target contribution (Direction B): make the ENTIRE diameter distribution
the area-level response, in a COHERENT geometry, and shrink each stand's noisy
empirical distribution toward an ML-predicted synthetic distribution.

Two coherent representations (both guarantee a valid output distribution):
  * Wasserstein / quantile space  -- W2 distance on the line = L2 on quantile
    functions Q(tau). FH shrinkage becomes the OT geodesic (convex combination of
    quantile functions); a convex combo of monotone functions is monotone, so the
    estimate is ALWAYS a valid quantile function. Coherence is free.
  * Bayes-Hilbert space  -- clr-transformed densities live in L2_0; FH there, then
    inverse-clr (softmax) returns a valid density (>=0, integrates to 1).

Unifying engine: functional PCA reduces either representation to r scores; a
multivariate Fay-Herriot on the scores (correlated random effects Sigma_u) borrows
strength across the WHOLE shape; reconstruction + back-map gives a coherent
distribution. FPCA scores are the optimal low-dim functionals (vs ad-hoc QMD/
percentiles), so DV-SAE subsumes the index approach.
"""
from __future__ import annotations
import numpy as np


# ============================ representations ============================
def pdf_to_cdf(pdf, grid):
    pdf = np.clip(np.asarray(pdf, float), 0, None)
    inc = 0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid)
    c = np.concatenate([[0.0], np.cumsum(inc)])
    return c / c[-1] if c[-1] > 0 else c


def cdf_to_quantile(cdf, grid, taus):
    # invert a monotone CDF at probability levels taus
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 1))
    return np.interp(taus, cdf, grid, left=grid[0], right=grid[-1])


def pdf_to_quantile(pdf, grid, taus):
    return cdf_to_quantile(pdf_to_cdf(pdf, grid), grid, taus)


def samples_to_quantile(x, taus):
    x = np.sort(np.asarray(x, float))
    if x.size == 0:
        return np.full(taus.shape, np.nan)
    pp = (np.arange(1, x.size + 1) - 0.5) / x.size   # Hazen plotting positions
    return np.interp(taus, pp, x, left=x[0], right=x[-1])


def quantile_to_pdf(Q, grid, taus):
    """Map a quantile function back to a density on `grid` (CDF = inverse of Q)."""
    Qm = np.maximum.accumulate(np.asarray(Q, float))
    cdf = np.interp(grid, Qm, taus, left=0.0, right=1.0)
    pdf = np.gradient(cdf, grid)
    pdf = np.clip(pdf, 0, None)
    a = np.trapz(pdf, grid)
    return pdf / a if a > 0 else pdf


def quantile_to_cdf(Q, grid, taus):
    Qm = np.maximum.accumulate(np.asarray(Q, float))
    return np.interp(grid, Qm, taus, left=0.0, right=1.0)


# ============================ distances ============================
def w2(Q1, Q2, taus):
    """2-Wasserstein distance between two distributions via their quantile fns."""
    return float(np.sqrt(np.trapz((np.asarray(Q1) - np.asarray(Q2)) ** 2, taus)))


def w1(Q1, Q2, taus):
    return float(np.trapz(np.abs(np.asarray(Q1) - np.asarray(Q2)), taus))


def l1_cdf(cdf1, cdf2, grid):
    """L1 distance between CDFs on the diameter grid (Cramer / W1 in x-space)."""
    return float(np.trapz(np.abs(np.asarray(cdf1) - np.asarray(cdf2)), grid))


# ============================ Bayes-Hilbert (clr) ============================
def clr(pdf, grid, eps=1e-8):
    f = np.clip(np.asarray(pdf, float), eps, None)
    lg = np.log(f)
    T = grid[-1] - grid[0]
    return lg - np.trapz(lg, grid) / T


def clr_inv(g, grid):
    g = np.asarray(g, float)
    g = g - g.max()
    f = np.exp(g)
    a = np.trapz(f, grid)
    return f / a if a > 0 else f


# ============================ functional PCA ============================
def fpca(R, w, n_comp=4):
    """Weighted functional PCA.
    R: (m, T) representation rows; w: (T,) integration weights (e.g. dtau or dx).
    Returns mean (T,), comps (n_comp, T), scores (m, n_comp), explained (n_comp,).
    Scores are <R-mean, phi_k>_w so reconstruction is mean + scores @ comps.
    """
    R = np.asarray(R, float)
    mean = R.mean(0)
    Rc = R - mean
    sw = np.sqrt(np.clip(w, 1e-12, None))
    U, S, Vt = np.linalg.svd(Rc * sw, full_matrices=False)
    n_comp = min(n_comp, Vt.shape[0])
    comps = Vt[:n_comp] / sw                       # eigenfunctions on raw scale
    scores = Rc @ (comps * w).T                     # weighted inner products
    expl = (S[:n_comp] ** 2) / (S ** 2).sum()
    return mean, comps, scores, expl


def fpca_reconstruct(mean, comps, scores):
    return mean + np.asarray(scores) @ comps


def project_scores(R, mean, comps, w):
    return (np.asarray(R) - mean) @ (comps * w).T


# ============================ multivariate FH (EM-EBLUP) ============================
def _npd(A, jit=1e-8):
    A = 0.5 * (A + A.T)
    wv, V = np.linalg.eigh(A)
    return (V * np.clip(wv, jit, None)) @ V.T


def mv_fh_em(H, Y, D, iters=60):
    """Multivariate Fay-Herriot via EM. H mean-basis (m,p); Y (m,r); D (m,r,r).
    Returns B (p,r), Sigma_u (r,r)."""
    H = np.asarray(H, float); Y = np.asarray(Y, float); D = np.asarray(D, float)
    m, p = H.shape
    G0 = H.T @ H + 1e-6 * np.eye(p)
    B = np.linalg.solve(G0, H.T @ Y)
    Su = _npd(np.cov((Y - H @ B).T, ddof=1) / 2)
    for _ in range(iters):
        Vi = np.linalg.inv(Su[None] + D)
        G = np.einsum("qr,mrs->mqs", Su, Vi)
        Th = H @ B + np.einsum("mqs,ms->mq", G, Y - H @ B)
        B = np.linalg.solve(G0, H.T @ Th)
        Vp = Su[None] - np.einsum("mqs,sr->mqr", G, Su)
        Su = _npd(((Th - H @ B).T @ (Th - H @ B) + Vp.sum(0)) / m)
    return B, Su


def mv_fh_eblup(H, Y, D, B, Su):
    Vi = np.linalg.inv(Su[None] + D)
    G = np.einsum("qr,mrs->mqs", Su, Vi)
    Th = H @ B + np.einsum("mqs,ms->mq", G, Y - H @ B)
    Vp = Su[None] - np.einsum("mqs,sr->mqr", G, Su)
    return Th, Vp


# ============================ Wasserstein-geodesic FH ============================
def wasserstein_fh(Q_dir, Q_ml, DW, taus):
    """Distribution-valued FH shrinkage in 2-Wasserstein space.

    Model in quantile coordinates: Q_i^true = Q_i^ML + proc (W2-variance A);
    Q_dir = Q_i^true + sampling noise (per-stand W2-variance DW_i). The W2-MSE-
    optimal estimator is the OT geodesic (convex combo of quantile functions):
        Q_hat_i = (1-w_i) Q_ml_i + w_i Q_dir_i,   w_i = A / (A + DW_i),
    with A estimated by a Wasserstein moment equation. Output is monotone =>
    a valid quantile function (coherent by construction).
    """
    Q_dir = np.asarray(Q_dir, float); Q_ml = np.asarray(Q_ml, float)
    DW = np.asarray(DW, float)
    resid2 = np.trapz((Q_dir - Q_ml) ** 2, taus, axis=1)   # ||Qdir-Qml||^2
    A = max(float(resid2.mean() - DW.mean()), 1e-9)         # process W2-variance
    w = A / (A + DW)
    Qhat = (1 - w)[:, None] * Q_ml + w[:, None] * Q_dir
    Qhat = np.maximum.accumulate(Qhat, axis=1)              # enforce monotone
    mse_W = (A * DW) / (A + DW)                              # FH W2-MSE per stand
    return Qhat, w, A, mse_W


# ============================ weighted empirical distribution ============================
def weighted_quantile(values, weights, taus):
    """TPA-weighted empirical quantile function (for tree lists)."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    if v.size == 0:
        return np.full(np.shape(taus), np.nan)
    o = np.argsort(v); v = v[o]; w = np.clip(w[o], 0, None)
    cw = np.cumsum(w); 
    if cw[-1] <= 0:
        return np.full(np.shape(taus), v[-1])
    pp = (cw - 0.5 * w) / cw[-1]                 # weighted Hazen positions
    return np.interp(taus, pp, v, left=v[0], right=v[-1])


# ============================ isotonic (monotone) projection ============================
def isotonic(y):
    """Pool-Adjacent-Violators: L2 projection onto the monotone non-decreasing cone.
    Returns the unique closest non-decreasing vector to y (Brunk 1958)."""
    y = np.asarray(y, float).copy()
    n = y.size
    val = y.copy(); wgt = np.ones(n); idx = list(range(n + 1))  # block boundaries
    lvl = list(y); wt = [1.0] * n; bounds = [[i] for i in range(n)]
    # simple O(n) PAVA
    out_val = []; out_w = []
    for i in range(n):
        out_val.append(y[i]); out_w.append(1.0)
        while len(out_val) > 1 and out_val[-2] > out_val[-1]:
            v2 = out_val.pop(); w2 = out_w.pop()
            v1 = out_val.pop(); w1 = out_w.pop()
            out_val.append((v1 * w1 + v2 * w2) / (w1 + w2)); out_w.append(w1 + w2)
    res = np.empty(n); k = 0
    for v, w in zip(out_val, out_w):
        c = int(round(w))
        res[k:k + c] = v; k += c
    return res


def wasserstein_fh_hetero(Q_dir, Q_ml, DW_tau, taus):
    """HETEROSCEDASTIC per-tau Wasserstein-FH: a separate shrinkage weight at each
    probability level, w_i(tau)=A(tau)/(A(tau)+DW_i(tau)), then ISOTONIC projection
    back onto the monotone cone to restore coherence.

    DW_tau: (m, T) per-stand, per-tau sampling variance of the direct quantile.
    A(tau) (process variance at level tau) by the moment identity
       E_i[(Qdir-Qml)^2](tau) = A(tau) + mean_i DW_i(tau).
    Returns Qhat (m,T), w (m,T), A_tau (T,). Coherent by Proposition 5.
    """
    Q_dir = np.asarray(Q_dir, float); Q_ml = np.asarray(Q_ml, float)
    DW_tau = np.asarray(DW_tau, float)
    A_tau = np.maximum((Q_dir - Q_ml).var(0) - DW_tau.mean(0), 1e-9)
    w = A_tau[None, :] / (A_tau[None, :] + DW_tau)
    raw = (1 - w) * Q_ml + w * Q_dir
    Qhat = np.array([isotonic(raw[i]) for i in range(raw.shape[0])])   # coherence
    return Qhat, w, A_tau
