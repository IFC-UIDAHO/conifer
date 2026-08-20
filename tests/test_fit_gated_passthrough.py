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


# ---------------- v0.3.5 reliability-corrected trust ----------------

def test_reliability_identity_when_absent():
    counts, X, lN, vN, *_, target = _toy(seed=21)
    a = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target, seed=0)
    b = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target, seed=0,
                          adequacy_reliability=1.0)
    assert np.allclose(a.s_hat_, b.s_hat_)
    assert a.reliability_ == 1.0 and abs(a.rho - a.rho_raw) < 1e-12


def test_reference_reliability_bounds_and_monotonicity():
    counts, X, lN, vN, *_, target = _toy(seed=22)
    lam_small = conifer.reference_reliability(target, np.full(len(target), 5.0))
    lam_big = conifer.reference_reliability(target, np.full(len(target), 500.0))
    assert 0.0 < lam_small <= 1.0 and 0.0 < lam_big <= 1.0
    assert lam_big > lam_small  # more reference trees -> more reliable target


def test_correction_raises_rho_not_learner():
    counts, X, lN, vN, *_, target = _toy(seed=23)
    raw = conifer.fit_gated(counts, np.ones(len(counts)), X,
                            total_logN=lN, var_logN=vN, adequacy_target=target, seed=0)
    cor = conifer.fit_gated(counts, np.ones(len(counts)), X,
                            total_logN=lN, var_logN=vN, adequacy_target=target, seed=0,
                            adequacy_reliability=0.5)
    assert cor.rho >= raw.rho - 1e-12          # disattenuation can only raise (until clipped)
    assert cor.rho <= 1.0
    assert cor.learner == raw.learner          # Gate A ranking invariant
    assert abs(cor.rho - min(1.0, raw.rho_raw / 0.5)) < 1e-9


def test_adequacy_n_route_runs():
    counts, X, lN, vN, *_, target = _toy(seed=24)
    n_ref = target.sum(1) / target.sum(1).mean() * 60.0
    g = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target, seed=0,
                          adequacy_n=n_ref)
    assert 0.0 < g.reliability_ <= 1.0
    assert g.rho >= g.rho_raw - 1e-12


# ---------------- v0.3.6 smooth gate ramp ----------------

def test_ramp_continuity_and_endpoints():
    from conifer.adequacy import _ramp_weight
    f, w = 0.30, 0.15
    assert _ramp_weight(0.10, f, w) == 0.0          # junk fully rejected
    assert _ramp_weight(0.50, f, w) == 1.0          # strong trust untouched
    assert abs(_ramp_weight(0.30, f, w) - 0.5) < 1e-12
    a, b = _ramp_weight(0.299, f, w), _ramp_weight(0.301, f, w)
    assert abs(a - b) < 0.01                        # no cliff
    xs = [i / 100 for i in range(0, 101)]
    ws = [_ramp_weight(x, f, w) for x in xs]
    assert all(w2 >= w1 - 1e-12 for w1, w2 in zip(ws, ws[1:]))  # monotone


def test_ramp_zero_reproduces_hard_floor():
    counts, X, lN, vN, *_, target = _toy(seed=31)
    hard = conifer.fit_gated(counts, np.ones(len(counts)), X,
                             total_logN=lN, var_logN=vN, adequacy_target=target,
                             seed=0, rho_ramp=0.0)
    assert hard.rho_eff in (0.0, hard.rho)          # step behaviour


def test_ramp_partial_weight_between_bounds():
    counts, X, lN, vN, *_, target = _toy(seed=32)
    g = conifer.fit_gated(counts, np.ones(len(counts)), X,
                          total_logN=lN, var_logN=vN, adequacy_target=target,
                          seed=0, adequacy_reliability=0.5)
    if 0.15 < g.rho < 0.45:
        assert 0.0 < g.rho_eff < g.rho              # genuinely partial
