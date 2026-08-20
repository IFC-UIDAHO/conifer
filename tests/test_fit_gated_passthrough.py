"""v0.3.4: fit_gated fit_kwargs passthrough + deployed_mode_ exposure."""
import warnings
import numpy as np
import pytest
import conifer


def _toy(m=80, K=4, seed=3, nonlinear=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, 4))
    drv = np.sin(2.2 * X[:, 0]) + X[:, 0] * X[:, 1] if nonlinear else X[:, 0]
    eta = np.array([1.5, 1.0, 0.2, -0.8]) + np.outer(2.0 * drv, np.array([0.8, 0.2, -0.4, -0.6]))
    eta += rng.normal(0, 0.2, (m, K))
    P = np.exp(eta); P /= P.sum(1, keepdims=True)
    N = np.exp(5.0 + 0.2 * X[:, 0] + rng.normal(0, 0.2, m))
    nplots = 3
    plots = np.zeros((m, nplots, K))
    for i in range(m):
        for p in range(nplots):
            t = rng.poisson(12)
            plots[i, p] = rng.multinomial(t, P[i])
    counts = plots.sum(1)
    direct = counts / np.clip(counts.sum(1, keepdims=True), 1, None) * N[:, None]
    lN = np.log(np.clip(direct.sum(1), 1e-3, None))
    vN = np.full(m, 0.05)
    target = N[:, None] * P  # near-truth adequacy target
    return counts, X, lN, vN, plots, direct, target


def test_backward_compatible():
    counts, X, lN, vN, *_ , target = _toy()
    g = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target, seed=0)
    assert g.s_hat_.shape == counts.shape
    assert g.learner in ("linear", "ml")
    assert g.deployed_mode_ in ("linear", "bart")


def test_fit_kwargs_reach_the_engine():
    counts, X, lN, vN, plots, direct, target = _toy(seed=5)
    base = conifer.fit_gated(counts, np.ones(len(counts)), X,
                             total_logN=lN, var_logN=vN, adequacy_target=target, seed=0)
    via = conifer.fit_gated(counts, np.ones(len(counts)), X,
                            total_logN=lN, var_logN=vN, adequacy_target=target, seed=0,
                            fit_kwargs=dict(plots=plots, direct_dens=direct))
    # the support-aware defer gate must have moved the estimate
    assert not np.allclose(base.s_hat_, via.s_hat_), "plots/direct_dens did not reach fit()"


def test_bad_fit_kwarg_raises():
    counts, X, lN, vN, *_, target = _toy()
    with pytest.raises(TypeError):
        conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target,
                          fit_kwargs=dict(not_a_real_kwarg=1))


def test_deployed_mode_matches_learner():
    counts, X, lN, vN, *_, target = _toy(seed=11, nonlinear=True)
    g = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target, seed=0)
    if not g.used_covariates:
        # below the rho floor the covariate-free linear fallback is deployed regardless of learner
        assert g.deployed_mode_ == "linear"
    elif g.learner == "ml":
        assert g.deployed_mode_ == "bart"
    else:
        assert g.deployed_mode_ == "linear"


def test_mean_mode_still_ignored():
    counts, X, lN, vN, *_, target = _toy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = conifer.fit_gated(counts, np.ones(len(counts)), X,
                              total_logN=lN, var_logN=vN, adequacy_target=target,
                              mean_mode="lograte")
    assert g.deployed_mode_ in ("linear", "bart")
