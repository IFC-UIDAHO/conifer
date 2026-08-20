"""
conifer.adequacy — v0.3.1 auxiliary-information *adequacy gates*.

Two data-driven gates that make covariate (auxiliary) use **no-harm**: the estimate
with covariates is never worse than the covariate-free estimate, while gains are kept
where the auxiliary data carry signal.

GATE A — learner adequacy
    Choose the mean learner ('linear' ridge vs 'ml' boosted-tree) by a 5-fold
    out-of-fold R² contest on X -> adequacy target. The flexible learner is used
    only where it actually generalizes; at small n its OOF R² collapses (predicts
    held-out areas worse than the mean) and the linear mean is selected
    automatically. There is **no sample-size threshold** — "small" reveals itself.

GATE B — covariate adequacy
    rho = clip(OOF R²(X -> adequacy target), 0, 1) is the covariate *trust*.
    The covariate mean is capped:  s = rho_eff * s_cov + (1 - rho_eff) * s_0cov,
    rho_eff = rho if rho >= rho_floor else 0  (defer fully to the covariate-free
    estimate below the floor). This stops the Fay–Herriot mean from over-leaning
    on weak covariates when the direct estimate is noisy (gamma small), which is
    exactly how weak auxiliaries hurt.

The **adequacy target** should be the *fullest cruised* data available (near-truth):
operationally, the well-cruised stands on which you calibrate covariate trust before
predicting data-poor ones. If omitted, the direct implied by ``counts`` is used, which
is optimistic in the data-poor regime (a warning is emitted) — because covariates can
fit a noisy direct's plot-selection artifacts without predicting truth.

Validated no-harm (4 regions x {sparse,rich}, real engine): strictly no-harm on the
Aitchison distribution **shape** in all 8 cells (improves 6), no-harm on combined
log-count in 7/8. Turns a weak-covariate region (Mississippi, +17% ungated harm) into
**exact no-harm** while preserving gains where covariates carry signal (South Carolina,
Idaho, Arkansas). See ``_recompute/v04_covgate/gate_results.csv`` in the study repo.
"""
from __future__ import annotations
import warnings
import numpy as np

__all__ = ["fit_gated", "GatedResult", "choose_learner", "covariate_adequacy", "reference_reliability"]

DEFAULT_RHO_FLOOR = 0.30  # selected on the held-out battery (clean gap: weak 0.20 vs signal 0.34+)


def _clr(P):
    # normalize to shares, then floor shares at 1e-6 (matches the Aitchison scoring convention;
    # for structurally-degenerate zero classes this must NOT use a raw-count floor).
    P = np.asarray(P, float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    P = np.clip(P, 1e-6, None)
    g = np.exp(np.mean(np.log(P), 1, keepdims=True))
    return np.log(P / g)


def _standardize(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / np.clip(X.std(0), 1e-9, None)


def _oof_r2(Xs, Y, make_est, cv=5, seed=0):
    from sklearn.model_selection import KFold
    kf = KFold(cv, shuffle=True, random_state=seed)
    pred = np.zeros_like(Y)
    for tr, te in kf.split(Xs):
        for j in range(Y.shape[1]):
            m = make_est(); m.fit(Xs[tr], Y[tr, j]); pred[te, j] = m.predict(Xs[te])
    sse = ((Y - pred) ** 2).sum(); sst = max(((Y - Y.mean(0)) ** 2).sum(), 1e-12)
    return float(1.0 - sse / sst)


def _contest(Xs, adequacy_target, cv=5, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    Y = _clr(adequacy_target); Y = Y - Y.mean(0)
    r2_lin = _oof_r2(Xs, Y, lambda: Ridge(alpha=1.0), cv, seed)
    r2_ml = _oof_r2(Xs, Y, lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=3), cv, seed)
    learner = "linear" if r2_lin >= r2_ml else "ml"
    rho = float(np.clip(max(r2_lin, r2_ml), 0.0, 1.0))
    return learner, rho, r2_lin, r2_ml


def reference_reliability(adequacy_target, adequacy_n):
    """v0.3.5 — reliability lambda of the clr(reference) target under multinomial
    within-stand sampling noise.

    The Gate A/B contest scores out-of-fold prediction of a NOISY reference (the
    fullest cruise available, not truth). Observed OOF R-squared is therefore
    attenuated: R2_obs ~= lambda * R2_true with lambda = var(signal)/var(observed)
    (classical errors-in-variables). This estimates lambda from multinomial theory:
    per stand i with n_i reference trees and floored shares p_ic,
        var(log p_hat_ic) ~= (1 - p_ic) / (n_i * p_ic),
    mapped to the clr scale by the centering matrix G = I - J/K, and compared to the
    empirical variance of the column-centred clr target (matching the contest metric).

    Parameters
    ----------
    adequacy_target : (m, K) reference class counts/densities (shares are taken).
    adequacy_n      : (m,) total reference tree count per stand.

    Returns lambda in (0, 1].
    """
    P = np.asarray(adequacy_target, float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    P = np.clip(P, 1e-6, None)
    n = np.clip(np.asarray(adequacy_n, float).reshape(-1), 1.0, None)
    m, K = P.shape
    # half-count floor for the variance formula: a class with ~zero expected tally carries
    # ~no clr signal, and (1-p)/(n p) at a raw 1e-6 share floor would explode the estimate
    Pv = np.clip(P, 0.5 / n[:, None], 1.0)
    v_log = (1.0 - Pv) / (n[:, None] * Pv)                     # var of log p_hat, per cell
    G = np.eye(K) - np.ones((K, K)) / K
    G2 = G ** 2                                                # diag of G diag(v) G'
    v_clr = v_log @ G2.T
    noise = float(np.mean(v_clr))
    Y = _clr(P); Y = Y - Y.mean(0)
    total = max(float(Y.var()), 1e-12)
    lam = 1.0 - noise / total
    return float(np.clip(lam, 1e-3, 1.0))


def choose_learner(X, adequacy_target, cv=5, seed=0):
    """GATE A + trust: returns (learner, rho, r2_linear, r2_ml) from the OOF-CV contest."""
    return _contest(_standardize(X), adequacy_target, cv, seed)


def covariate_adequacy(X, adequacy_target, cv=5, seed=0):
    """GATE B trust only: rho in [0,1], the out-of-fold covariate signal (max of ridge/tree OOF R²)."""
    return _contest(_standardize(X), adequacy_target, cv, seed)[1]


class GatedResult:
    """Result of :func:`fit_gated`. ``s_hat_`` is the gated point estimate (m x K stems/area).
    ``est_cov_`` / ``est_0cov_`` are the underlying fitted ``DiameterDistribution`` objects —
    use them for conformalization. ``rho`` is the covariate trust, ``rho_eff`` the applied
    (floored) weight, ``learner`` the mean chosen by Gate A."""
    def __init__(self, s_hat_, rho, rho_eff, learner, r2_linear, r2_ml, est_cov_, est_0cov_,
                 deployed_mode_=None):
        self.s_hat_ = s_hat_
        self.rho = float(rho); self.rho_eff = float(rho_eff); self.learner = learner
        self.r2_linear = float(r2_linear); self.r2_ml = float(r2_ml)
        self.est_cov_ = est_cov_; self.est_0cov_ = est_0cov_
        #: the mean_mode actually fitted for the deployed covariate arm ('linear' or 'bart');
        #: 'linear' when the covariate-free fallback was returned. v0.3.4.
        self.deployed_mode_ = deployed_mode_ if deployed_mode_ is not None else (
            "linear" if learner == "linear" else "bart")
    @property
    def used_covariates(self): return self.rho_eff > 0.0
    def __repr__(self):
        tag = f"learner={self.learner!r}, rho={self.rho:.3f}->{self.rho_eff:.3f}"
        return f"GatedResult({tag}, used_covariates={self.used_covariates})"


def _ramp_weight(rho, rho_floor, rho_ramp):
    """v0.3.6 — smooth gate weight. Hard step (rho_ramp=0) or a linear ramp of
    half-width rho_ramp centred on rho_floor: 0 below floor-ramp, 1 above floor+ramp.
    Removes the knife-edge where trust epsilon-below the floor discarded covariates
    that were demonstrably useful (pilot: rho=0.29 vs floor 0.30 cost 0.04 RMSE)."""
    if rho_ramp <= 0.0:
        return 1.0 if rho >= rho_floor else 0.0
    return float(np.clip((rho - (rho_floor - rho_ramp)) / (2.0 * rho_ramp), 0.0, 1.0))


DEFAULT_RHO_RAMP = 0.15


def fit_gated(counts, area_eff, X, *, groups=None, total_logN=None, var_logN=None,
              adequacy_target=None, adequacy_n=None, adequacy_reliability=None,
              rho_floor=DEFAULT_RHO_FLOOR, rho_ramp=DEFAULT_RHO_RAMP, cv=5, seed=0,
              fit_kwargs=None, **dd_kwargs):
    """Fit CONIFER with the v0.3.1 adequacy gates (see module docstring).

    Parameters mirror ``DiameterDistribution.fit`` plus:
      adequacy_target : (m,K) near-truth class counts/shares to calibrate covariate trust
                        (the fullest cruised data). If None, uses ``counts`` (optimistic; warns).
      rho_floor       : defer fully to the covariate-free estimate when trust < floor (default 0.30).
      fit_kwargs      : dict of extra keyword arguments forwarded to BOTH underlying
                        ``DiameterDistribution.fit`` calls (v0.3.4) — e.g.
                        ``dict(D_ext=..., plots=..., direct_dens=...)`` to enable the
                        design-based ALR covariance and the support-aware defer gate,
                        which were previously unreachable through ``fit_gated``.
      adequacy_n      : (m,) total reference tree count per stand (v0.3.5). Enables the
                        errors-in-variables reliability correction: the observed contest
                        R² is attenuated by the reference's own sampling noise, so the
                        trust is corrected as rho = clip(rho_obs / lambda, 0, 1) with
                        lambda from :func:`reference_reliability`. The learner CHOICE is
                        unaffected (both learners attenuate equally).
      adequacy_reliability : float in (0,1], overrides the internal lambda estimate
                        (takes precedence over ``adequacy_n``). Default None: if neither
                        is given, lambda = 1 and behaviour is identical to v0.3.4.
      rho_ramp        : half-width of the smooth gate ramp (v0.3.6, default 0.15). The
                        covariate weight ramps linearly from 0 at ``rho_floor - rho_ramp``
                        to full trust at ``rho_floor + rho_ramp`` (midpoint = the classic
                        floor). ``rho_ramp=0`` restores the pre-0.3.6 hard cutoff exactly.
    ``mean_mode`` in ``dd_kwargs`` is ignored — the learner is chosen by Gate A.
    Returns a :class:`GatedResult` (``.deployed_mode_`` gives the fitted covariate arm;
    ``.rho_raw`` the uncorrected trust; ``.reliability_`` the lambda applied).
    """
    from .estimators import DiameterDistribution
    counts = np.asarray(counts, float); m, K = counts.shape
    Xs = _standardize(X)
    if adequacy_target is None:
        warnings.warn("fit_gated: no adequacy_target; using the direct implied by counts, which is "
                      "optimistic in the data-poor regime. Pass the fullest cruised data as adequacy_target.",
                      stacklevel=2)
        adequacy_target = counts
    dd_kwargs.pop("mean_mode", None)
    learner, rho_raw, r2_lin, r2_ml = _contest(Xs, adequacy_target, cv, seed)
    # --- v0.3.5 reliability correction (errors-in-variables disattenuation) ---
    if adequacy_reliability is not None:
        lam = float(np.clip(adequacy_reliability, 1e-3, 1.0))
    elif adequacy_n is not None:
        lam = reference_reliability(adequacy_target, adequacy_n)
    else:
        lam = 1.0
    rho = float(np.clip(rho_raw / lam, 0.0, 1.0))
    rho_eff = rho * _ramp_weight(rho, rho_floor, rho_ramp)
    fkw = dict(groups=groups, total_logN=total_logN, var_logN=var_logN)
    if fit_kwargs:
        fkw.update(dict(fit_kwargs))
    est0 = DiameterDistribution(seed=seed, mean_mode="linear", **dd_kwargs).fit(counts, area_eff, np.zeros((m, 1)), **fkw)
    if rho_eff <= 0.0:
        g = GatedResult(est0.s_hat_, rho, 0.0, learner, r2_lin, r2_ml, est0, est0,
                        deployed_mode_="linear")
        g.rho_raw = float(rho_raw); g.reliability_ = float(lam)
        return g
    # Deploy the SAME learner the contest used: the contest ranks ridge vs HistGradientBoosting, so when
    # it selects the flexible learner, fit the gradient-boosted mean ('bart' -> HGB) rather than the weaker
    # random-feature 'ml' model. Falls back to random features automatically if HGB is unavailable.
    _fit_mode = "bart" if learner == "ml" else learner
    estc = DiameterDistribution(seed=seed, mean_mode=_fit_mode, **dd_kwargs).fit(counts, area_eff, Xs, **fkw)
    s = rho_eff * estc.s_hat_ + (1.0 - rho_eff) * est0.s_hat_
    g = GatedResult(s, rho, rho_eff, learner, r2_lin, r2_ml, estc, est0,
                    deployed_mode_=_fit_mode)
    g.rho_raw = float(rho_raw); g.reliability_ = float(lam)
    return g
