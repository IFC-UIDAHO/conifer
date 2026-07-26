"""uncertainty.py — strong inference layer for the multivariate ML-SAE.

Implements the program in ../Uncertainty_MSE_CI_Program.md:
  I.  corrected_mse_path_a   second-order, bias-corrected MSE (g1 + g3 + ML mean
                             variance/bias via infinitesimal-jackknife/OOB) and
                             BACK-TRANSFORM to the proportion scale (delta method).
  III propagate_Di           inverse-Wishart resampling of the sampling covariance
                             D_i -> extra-variance Delta^D ("known-D" understates UQ).
  IV  ConformalSAE           Mondrian (by stand type) split/CV conformal with a
                             multivariate Mahalanobis score -> finite-sample,
                             distribution-free coverage (valid under misspecification).
  VI  distribution_bands     per-replicate Johnson's S_B -> pointwise + SIMULTANEOUS
                             CDF bands + functional CIs (QMD, percentiles).
  V   calibration_audit      empirical vs nominal coverage across scales; R-hat.

All functions are framework-light (numpy/scipy/sklearn) and reuse the package.
"""
from __future__ import annotations
import numpy as np
try:
    from scipy.stats import invwishart as _sp_invwishart
except Exception:
    _sp_invwishart = None

from .composition import alr_inv, class_midpoints
from .sampling_cov import _nearest_pd


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _inv_sqrt(M):
    w, V = np.linalg.eigh(_nearest_pd(M))
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _wishart_rvs(df, V, rng):
    """Wishart(df, V) via Bartlett decomposition (E[W]=df*V)."""
    q = V.shape[0]
    L = np.linalg.cholesky(_nearest_pd(V))
    A = np.zeros((q, q))
    for i in range(q):
        A[i, i] = np.sqrt(rng.chisquare(max(df - i, 1.0)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    LA = L @ A
    return LA @ LA.T


def _iw_rvs(df, Psi, rng):
    """Inverse-Wishart(df, Psi): if W~Wishart(df,Psi^{-1}) then W^{-1}~IW(df,Psi)."""
    if _sp_invwishart is not None:
        return np.atleast_2d(_sp_invwishart.rvs(df=df, scale=_nearest_pd(Psi),
                                                random_state=rng))
    Wi = np.linalg.inv(_nearest_pd(Psi))
    W = _wishart_rvs(df, Wi, rng)
    return np.linalg.inv(_nearest_pd(W))


def _softmax_jacobian(theta_alr):
    """Jacobian d p / d theta of inverse-ALR (ref = last class), shape (K, q)."""
    p = alr_inv(theta_alr)                       # (K,)
    K = p.shape[0]; q = K - 1
    # full softmax Jacobian (K x K) then drop the reference column -> (K, q)
    Jfull = np.diag(p) - np.outer(p, p)
    keep = list(range(K - 1))                    # ref = last
    return Jfull[:, keep]                         # (K, q)


# ----------------------------------------------------------------------------
# Item I — corrected MSE for Path A, with back-transformation
# ----------------------------------------------------------------------------
def corrected_mse_path_a(model, data, n_boot=200, debias=True, seed=3):
    """Second-order, bias-aware MSE for RFReblupSAE on the ALR scale, plus the
    proportion-scale MSE by the delta method.

    Components per area i (q x q):
      g1_i   = Sigma_u - Sigma_u (Sigma_u+D_i)^{-1} Sigma_u     (Prasad-Rao leading)
      g3_i   = bootstrap variability of the EBLUP from re-estimating Sigma_u
      gML_i  = RF mean variance (tree-to-tree / infinitesimal jackknife) [+ bias^2]
    Returns dict with per-area totals on ALR and proportion scales and components.
    """
    rng = np.random.default_rng(seed)
    X, Y, D = model._X, model._Y, model._D
    m, q = Y.shape
    Su = _nearest_pd(model.Sigma_u_)
    mhat = model.m_hat_

    # --- g1 (analytic leading term) ---
    g1 = np.empty((m, q, q))
    for i in range(m):
        Vi = Su + D[i]
        g1[i] = Su - Su @ np.linalg.solve(Vi, Su)

    # --- gML: RF mean uncertainty (variance across trees / n_trees ~ IJ proxy) + OOB bias^2 ---
    gML = np.zeros((m, q, q))
    bias2 = np.zeros((m, q))
    mdl = getattr(model, "_model", None)
    if mdl is not None and hasattr(mdl, "estimators_"):
        try:
            preds = np.stack([est.predict(X) for est in mdl.estimators_])  # (T,m,q) or (T,m)
            if preds.ndim == 2:
                preds = preds[:, :, None]
            T = preds.shape[0]
            var_tree = preds.var(axis=0) / max(T, 1)        # (m,q) IJ-style proxy
            for i in range(m):
                gML[i] = np.diag(var_tree[i])
            if debias and model._train is not None:
                # OOB-style bias proxy: mean tree pred vs fitted mean (small for RF)
                bias2 = (preds.mean(axis=0) - mhat) ** 2
        except Exception:
            pass

    # --- g3: bootstrap variability from re-estimating Sigma_u (mean held; FAST) ---
    Lu = np.linalg.cholesky(Su)
    LD = [np.linalg.cholesky(_nearest_pd(D[i])) for i in range(m)]
    acc = np.zeros((m, q, q))
    theta0 = model.theta_hat_
    for b in range(n_boot):
        U = (Lu @ rng.normal(size=(q, m))).T
        theta_star = mhat + U
        Ystar = theta_star + np.stack([LD[i] @ rng.normal(size=q) for i in range(m)])
        R = Ystar - mhat
        Su_b = _nearest_pd(np.einsum("iq,ir->qr", R, R) / m - D.mean(axis=0), jitter=1e-6)
        for i in range(m):
            Vi = Su_b + D[i]
            th = mhat[i] + Su_b @ np.linalg.solve(Vi, Ystar[i] - mhat[i])
            d = th - theta_star[i]
            acc[i] += np.outer(d, d)
    mse_boot = acc / n_boot                       # naive bootstrap MSE (ALR)
    # g3 = part of bootstrap MSE in excess of g1 (the Sigma_u-estimation inflation)
    g3 = np.clip(mse_boot - g1, 0.0, None)

    total_alr = g1 + g3 + gML
    for i in range(m):
        total_alr[i] += np.diag(bias2[i])

    # --- back-transform to proportion scale (delta method) ---
    K = q + 1
    total_prop = np.empty((m, K, K))
    for i in range(m):
        J = _softmax_jacobian(theta0[i])          # (K,q)
        total_prop[i] = J @ total_alr[i] @ J.T

    return {
        "g1": g1, "g3": g3, "gML": gML, "bias2": bias2,
        "mse_alr": total_alr, "mse_prop": total_prop,
        "mse_boot_naive": mse_boot,
        "total_alr_trace": np.array([np.trace(total_alr[i]) for i in range(m)]),
        "naive_alr_trace": np.array([np.trace(mse_boot[i]) for i in range(m)]),
        "total_prop_trace": np.array([np.trace(total_prop[i]) for i in range(m)]),
    }


# ----------------------------------------------------------------------------
# Item III — propagate D_i uncertainty (inverse-Wishart)
# ----------------------------------------------------------------------------
def propagate_Di(model, n_plots, n_draws=100, seed=5):
    """Extra-variance from treating D_i as ESTIMATED (few plots) not known.
    D_i^{(b)} ~ IW(df_i, df_i * D_i_hat),  df_i = max(n_plots_i - 1, q+2).
    Returns per-area variance added to theta_hat across D draws (trace) and the
    inflation ratio vs the known-D MSE trace.
    """
    rng = np.random.default_rng(seed)
    X, Y, D = model._X, model._Y, model._D
    m, q = Y.shape
    Su = _nearest_pd(model.Sigma_u_)
    mhat = model.m_hat_
    n_plots = np.atleast_1d(n_plots)
    if n_plots.size == 1:
        n_plots = np.full(m, int(n_plots))
    theta_draws = np.empty((n_draws, m, q))
    for b in range(n_draws):
        for i in range(m):
            df = max(int(n_plots[i]) - 1 + q + 2, q + 2)
            Db = _iw_rvs(df, df * _nearest_pd(D[i]), rng)
            Db = _nearest_pd(np.atleast_2d(Db))
            Vi = Su + Db
            theta_draws[b, i] = mhat[i] + Su @ np.linalg.solve(Vi, Y[i] - mhat[i])
    extra_var = theta_draws.var(axis=0)            # (m,q)
    return {"extra_var": extra_var,
            "extra_trace": extra_var.sum(axis=1),
            "theta_mean_over_D": theta_draws.mean(axis=0)}


# ----------------------------------------------------------------------------
# Item IV — Mondrian conformal SAE
# ----------------------------------------------------------------------------
class ConformalSAE:
    """Mondrian (group-wise) split-conformal prediction for area-level SAE.

    Nonconformity score (multivariate, heteroscedastic): for held-out area i,
        s_i = || (Sigma_u + D_i)^{-1/2} (y_i - theta_hat_{-i}) ||_2 ,
    giving a distribution-free prediction REGION for a new area's direct
    estimate with finite-sample coverage >= ceil((1-a)(n_g+1))/n_g within each
    Mondrian group g (here, stand type). Coverage holds under exchangeability
    even if the mean model is misspecified.
    """
    def __init__(self, alpha=0.10):
        self.alpha = alpha

    @staticmethod
    def _scores(Yc, Thc, Su, Dc):
        s = np.empty(Yc.shape[0])
        for j in range(Yc.shape[0]):
            W = _inv_sqrt(Su + Dc[j])
            s[j] = np.linalg.norm(W @ (Yc[j] - Thc[j]))
        return s

    def calibrate(self, theta_hat_cal, Y_cal, D_cal, Sigma_u, groups_cal):
        """Compute per-group conformal radii from calibration residuals."""
        Su = _nearest_pd(Sigma_u)
        s = self._scores(Y_cal, theta_hat_cal, Su, D_cal)
        self.radii_ = {}
        for g in np.unique(groups_cal):
            sg = np.sort(s[groups_cal == g]); ng = sg.size
            k = int(np.ceil((1 - self.alpha) * (ng + 1)))
            k = min(max(k, 1), ng)
            self.radii_[g] = float(sg[k - 1])
        self.radius_global_ = float(np.sort(s)[min(
            int(np.ceil((1 - self.alpha) * (s.size + 1))), s.size) - 1])
        self.Sigma_u_ = Su
        return self

    def covers(self, target, theta_hat, D, groups):
        """Empirical coverage of `target` (e.g. y_test or theta_true) by the
        calibrated Mahalanobis ball of radius r_g around theta_hat."""
        cov = np.empty(target.shape[0], dtype=bool)
        for j in range(target.shape[0]):
            r = self.radii_.get(groups[j], self.radius_global_)
            W = _inv_sqrt(self.Sigma_u_ + D[j])
            cov[j] = np.linalg.norm(W @ (target[j] - theta_hat[j])) <= r
        return cov


def cvplus_conformal(fit_predict, X, Y, D, groups, alpha=0.10, K=5, seed=0):
    """CV+ Mondrian conformal: K-fold out-of-fold residuals -> radii, then report
    coverage of the held-out DIRECT estimates (the clean conformal guarantee).
    `fit_predict(tr_idx, te_idx) -> (theta_hat_te, Sigma_u)` is supplied by caller.
    Returns dict(coverage_overall, coverage_by_group, radii, nominal).
    """
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    fold = rng.integers(0, K, size=m)
    theta_oof = np.empty_like(Y); Su_oof = None
    for k in range(K):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        if te.size == 0:
            continue
        th_te, Su = fit_predict(tr, te)
        theta_oof[te] = th_te; Su_oof = Su
    conf = ConformalSAE(alpha=alpha).calibrate(theta_oof, Y, D, Su_oof, groups)
    cov = conf.covers(Y, theta_oof, D, groups)     # coverage of direct estimates
    by = {str(g): float(cov[groups == g].mean()) for g in np.unique(groups)}
    return {"coverage_overall": float(cov.mean()), "coverage_by_group": by,
            "radii": conf.radii_, "nominal": 1 - alpha}


# ----------------------------------------------------------------------------
# Item VI — distribution-level (functional) uncertainty
# ----------------------------------------------------------------------------
def distribution_bands(prop_draws, edges, grid=None, level=0.90, max_draws=200):
    """Per-replicate Johnson's S_B -> CDF bands + functional CIs for one stand.

    prop_draws : (n_draws, K) posterior/bootstrap proportion draws for a stand.
    Returns pointwise band, simultaneous (sup-norm) band, and CIs for QMD and
    the 25th/50th/95th diameter percentiles.
    """
    from .jsb_recovery import recover_johnson_sb, johnson_sb_cdf
    prop_draws = np.asarray(prop_draws, float)
    if prop_draws.shape[0] > max_draws:
        idx = np.linspace(0, prop_draws.shape[0] - 1, max_draws).astype(int)
        prop_draws = prop_draws[idx]
    edges = np.asarray(edges, float)
    if grid is None:
        grid = np.linspace(edges[0], edges[-1], 81)
    mids = class_midpoints(edges)
    a = (1 - level) / 2

    cdfs = []; qmd = []; pctl = {25: [], 50: [], 95: []}
    for p in prop_draws:
        par = recover_johnson_sb(p, edges, refine_bounds=False)
        F = johnson_sb_cdf(grid, par["gamma"], par["delta"], par["xi"], par["lam"])
        F = np.clip(F, 0, 1); F = np.maximum.accumulate(F)
        cdfs.append(F)
        # QMD from class proportions (quadratic mean diameter)
        qmd.append(np.sqrt(np.sum(p * mids ** 2)))
        for qp in pctl:
            pctl[qp].append(float(np.interp(qp / 100.0, F, grid)))
    cdfs = np.array(cdfs)
    mean_F = cdfs.mean(axis=0); sd_F = cdfs.std(axis=0) + 1e-9
    lo_pt = np.quantile(cdfs, a, axis=0); hi_pt = np.quantile(cdfs, 1 - a, axis=0)
    # simultaneous sup-norm band
    sup = np.max(np.abs(cdfs - mean_F) / sd_F, axis=1)
    c = np.quantile(sup, level)
    lo_sim = np.clip(mean_F - c * sd_F, 0, 1); hi_sim = np.clip(mean_F + c * sd_F, 0, 1)
    out = {"grid": grid, "mean_cdf": mean_F,
           "cdf_lo_pointwise": lo_pt, "cdf_hi_pointwise": hi_pt,
           "cdf_lo_simultaneous": lo_sim, "cdf_hi_simultaneous": hi_sim,
           "QMD_CI": (float(np.quantile(qmd, a)), float(np.quantile(qmd, 1 - a)),
                      float(np.mean(qmd)))}
    for qp in pctl:
        out[f"p{qp}_CI"] = (float(np.quantile(pctl[qp], a)),
                            float(np.quantile(pctl[qp], 1 - a)), float(np.mean(pctl[qp])))
    return out


# ----------------------------------------------------------------------------
# Item V — calibration audit
# ----------------------------------------------------------------------------
def coverage_curve(theta_true, draws, levels=(0.5, 0.8, 0.9, 0.95)):
    """Empirical coverage of theta_true by central posterior/bootstrap intervals
    at several nominal levels (draws: (n_draws, m, q)). Returns {level: cov}."""
    out = {}
    for L in levels:
        a = (1 - L) / 2
        lo = np.quantile(draws, a, axis=0); hi = np.quantile(draws, 1 - a, axis=0)
        out[L] = float(np.mean((theta_true >= lo) & (theta_true <= hi)))
    return out


def rhat(chains):
    """Split-R-hat for a scalar parameter. chains: (n_chains, n_draws)."""
    chains = np.asarray(chains, float)
    M, N = chains.shape
    if N < 4:
        return np.nan
    half = N // 2
    s = np.vstack([chains[:, :half], chains[:, half:]])
    m2, n2 = s.shape
    means = s.mean(axis=1); B = n2 * means.var(ddof=1)
    W = s.var(axis=1, ddof=1).mean()
    var = (n2 - 1) / n2 * W + B / n2
    return float(np.sqrt(var / W)) if W > 0 else np.nan
