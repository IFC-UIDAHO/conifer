"""gcounts.py — Generalized (count-valued) Distribution SAE for stems-per-acre by
diameter class: the actual operational tree-list deliverable.

The DV-SAE flagship targets the *shape* of the diameter distribution. The tree list
also needs the *absolute* stems/acre per class -- a vector of overdispersed COUNTS
with structural zeros (no sawtimber in a regen stand). This module gives the
generalized area-level model:

  total density   N_i = stems/acre  -> log-FH (Gaussian on log scale)
  composition     p_i (class shares, simplex) -> DIRICHLET-MULTINOMIAL area-level FH
                  with an ANALYTIC sampling covariance on the ALR scale (no bootstrap;
                  valid for tiny plot tallies), correlated random effects Sigma_u
  presence        pi_ik (structural zero hurdle) -> logistic FH  [optional]
  deliverable     s_ik = N_i * p_ik   (stems/acre by class), MSE by the delta method

Why "generalized": the earlier class-FH put a Gaussian on log1p(density) with a
bootstrap D_i. Here the count likelihood supplies the sampling covariance in closed
form (overdispersion phi), is valid when a plot tallies only a handful of trees, and
the hurdle handles structural zeros -- the regimes where Gaussian FH degrades.
"""
from __future__ import annotations
import numpy as np
from .sampling_cov import _nearest_pd
from .distributional import mv_fh_em, mv_fh_eblup


def estimate_phi(counts):
    """Global Dirichlet-multinomial overdispersion phi by a moment match of the
    observed share variance to the multinomial baseline (phi->inf == multinomial)."""
    counts = np.asarray(counts, float); n = counts.sum(1, keepdims=True)
    p = counts / np.clip(n, 1, None); pbar = p.mean(0)
    # observed vs multinomial variance of shares, averaged over classes with mass
    num, den = 0.0, 0.0
    for k in range(p.shape[1]):
        if pbar[k] <= 1e-4 or pbar[k] >= 1 - 1e-4: continue
        v_obs = p[:, k].var(); v_mn = (pbar[k] * (1 - pbar[k]) * (1.0/np.clip(n,1,None)).mean())
        num += v_obs; den += v_mn
    r = max(num / max(den, 1e-9), 1.0)        # variance inflation factor >=1
    nbar = float(n.mean())
    # r = (nbar+phi)/(nbar(1+phi))*nbar = (nbar+phi)/(1+phi)  -> solve for phi
    phi = max((nbar - r) / max(r - 1.0, 1e-6), 0.5)
    return float(phi)


def alr_shares_and_Di(counts, phi=None, ref=-1, alpha=0.5, overdispersion=True):
    """ALR of class shares + ANALYTIC sampling covariance D_i.
    counts: (m,K) integer tallies. Returns y (m,K-1), D (m,K-1,K-1), p_smooth (m,K).

    overdispersion : bool (default True, legacy v0.1 behavior)
        True  -> Dirichlet-multinomial sampling covariance, kappa=(nᵢ+φ)/(nᵢ(1+φ)).
                 NOTE (v0.2 lesson): with φ estimated globally by ``estimate_phi`` from the
                 CROSS-STAND share variance, real between-stand compositional *signal* is
                 mistaken for within-stand overdispersion, so φ is driven small and kappa
                 approaches a NON-VANISHING floor 1/(1+φ). The EBLUP then never converges to
                 the design-direct estimate as plot tallies grow (over-shrinkage in the
                 data-rich regime; verified on the plasmode simulations).
        False -> multinomial sampling covariance, kappa=1/nᵢ, which VANISHES as tallies
                 accumulate — restoring Fay-Herriot consistency (EBLUP -> direct as D_i -> 0)
                 while preserving the sparse-regime model-borrowing gain. Genuine plot-level
                 overdispersion, when identified from replicates, should be supplied via the
                 estimator's ``D_ext`` argument rather than inferred from cross-stand variance.
    """
    counts = np.asarray(counts, float); m, K = counts.shape
    n = counts.sum(1, keepdims=True)
    p = (counts + alpha) / (n + K * alpha)               # additive-smoothed shares
    if phi is None: phi = estimate_phi(counts)
    ref = ref % K; keep = [k for k in range(K) if k != ref]
    y = (np.log(p) - np.log(p[:, [ref]]))[:, keep]       # ALR
    q = K - 1; D = np.empty((m, q, q))
    for i in range(m):
        ni = max(float(n[i, 0]),1.0)
        kappa = ((ni + phi) / (ni * (1.0 + phi))) if overdispersion else (1.0 / ni)   # DM floor vs vanishing multinomial
        pi = p[i]; Cov_p = kappa * (np.diag(pi) - np.outer(pi, pi))
        # Jacobian of ALR wrt p: dy_j/dp_l = [l==j]/p_j - [l==ref]/p_ref
        J = np.zeros((q, K))
        for jj, j in enumerate(keep):
            J[jj, j] += 1.0 / pi[j]; J[jj, ref] += -1.0 / pi[ref]
        D[i] = _nearest_pd(J @ Cov_p @ J.T + 1e-9 * np.eye(q))
    return y, D, p


def _alr_inv(y, ref=-1):
    y = np.asarray(y, float); q = y.shape[1]; K = q + 1; ref = ref % K
    keep = [k for k in range(K) if k != ref]
    full = np.zeros((y.shape[0], K)); full[:, keep] = y
    full -= full.max(1, keepdims=True); e = np.exp(full)
    return e / e.sum(1, keepdims=True)


def log_total_fh(logN, var_logN, X, iters=80):
    """Univariate FH on log total density. Returns theta (m,), mse (m,)."""
    y = np.asarray(logN, float); d = np.asarray(var_logN, float)
    A = max(y.var() - d.mean(), 1e-4)
    for _ in range(iters):
        W = 1.0 / (A + d); XtWX = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        b = np.linalg.solve(XtWX, X.T @ (W * y)); r = y - X @ b
        s = -0.5 * W.sum() + 0.5 * np.sum(r * r * W * W)
        A = max(A + s / (0.5 * np.sum(W * W)), 1e-6)
    W = 1.0 / (A + d); b = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]), X.T @ (W * y))
    g = A / (A + d); th = X @ b + g * (y - X @ b); mse = A * d / (A + d)
    return th, mse


class GeneralizedCountsSAE:
    """stems/acre by class = total(log-FH) x composition(Dirichlet-multinomial FH),
    with an optional presence hurdle for structural zeros."""
    def __init__(self, hurdle=True):
        self.hurdle = hurdle

    def fit(self, counts, perdacre_total, var_logtotal, X):
        counts = np.asarray(counts, float); m, K = counts.shape
        Xd = np.hstack([np.ones((m, 1)), X]) if X.ndim == 2 else X
        # composition
        y, D, psm = alr_shares_and_Di(counts)
        B, Su = mv_fh_em(Xd, y, D); th_c, Vp = mv_fh_eblup(Xd, y, D, B, Su)
        p_hat = _alr_inv(th_c)
        # presence hurdle (logistic FH-lite: shrink empirical presence toward covariate model)
        if self.hurdle:
            present = (counts > 0).astype(float)
            # simple ridge-logit on X for presence prob, blended with empirical by tally size
            from numpy.linalg import lstsq
            P = np.empty_like(present)
            for k in range(K):
                z = np.clip(present[:, k], 1e-3, 1 - 1e-3); lo = np.log(z / (1 - z))
                bk = lstsq(Xd, lo, rcond=None)[0]; P[:, k] = 1 / (1 + np.exp(-Xd @ bk))
            n = counts.sum(1, keepdims=True); wgt = n / (n + 8.0)        # trust data more if many trees
            pres = wgt * present + (1 - wgt) * P
            p_hat = p_hat * pres; p_hat = p_hat / p_hat.sum(1, keepdims=True)
        # total
        th_N, mseN = log_total_fh(perdacre_total, var_logtotal, Xd)
        N_hat = np.exp(th_N)
        s_hat = N_hat[:, None] * p_hat                                    # stems/acre by class
        self.p_hat_, self.N_hat_, self.s_hat_, self.Su_, self.phi_ = p_hat, N_hat, s_hat, Su, estimate_phi(counts)
        self.mseN_ = mseN; self.Vp_comp_ = Vp
        return self


# ============================ count log-rate FH (the working generalized model) ============================
def count_lograte_fh(counts, area_eff, X, ridge_total=True):
    """Generalized area-level FH for stems/acre by class using the COUNT likelihood's
    ANALYTIC sampling covariance (no bootstrap), valid for tiny tallies + structural zeros.

    counts   : (m,K) integer class tallies from the cruise
    area_eff : (m,) effective sampled area per stand = n_plots * plot_area (acres)
    X        : (m,p) covariates (intercept added inside)
    Model: per-acre rate r_ik = c_ik/area_eff_i ~ Poisson-rate; on the log scale,
      y_ik = log(r_ik + offset),  Var(log r_ik) ~ 1/c_ik (delta method)  -> diagonal D_i
      (independent Poisson class tallies); correlated random effects Sigma_u borrow
      strength across classes. Returns s_hat (m,K) stems/acre by class, plus Vp, Su.
    """
    counts = np.asarray(counts, float); m, K = counts.shape
    ae = np.asarray(area_eff, float)[:, None]
    rate = counts / ae
    off = 0.5 / ae                                    # offset so zeros are finite
    y = np.log(rate + off)
    Xd = np.hstack([np.ones((m, 1)), np.asarray(X, float)])
    D = np.empty((m, K, K))
    for i in range(m):
        v = 1.0 / (counts[i] + 0.5)                   # delta-method Poisson var of log-rate
        D[i] = _nearest_pd(np.diag(v) + 1e-9 * np.eye(K))
    B, Su = mv_fh_em(Xd, y, D); th, Vp = mv_fh_eblup(Xd, y, D, B, Su)
    s_hat = np.clip(np.exp(th) - off, 0, None)
    return dict(s_hat=s_hat, theta=th, Su=Su, Vp=Vp, D=D, y=y, off=off)
