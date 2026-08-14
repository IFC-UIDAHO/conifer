"""Tests for the v0.3.1 auxiliary-information adequacy gates (conifer.adequacy)."""
import numpy as np
import conifer


def _synth(m=48, K=3, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, 3))
    beta = rng.normal(size=(3, K))
    lin = signal * (X @ beta) + rng.normal(scale=0.3, size=(m, K))
    lin -= lin.max(1, keepdims=True)
    shares = np.exp(lin); shares /= shares.sum(1, keepdims=True)
    totals = rng.integers(80, 400, m).astype(float)
    truth = shares * totals[:, None]
    counts = rng.poisson(np.clip(truth, 0.1, None)).astype(float)
    lN = np.log(np.clip(counts.sum(1), 1, None)); vN = np.full(m, 0.05)
    return counts, truth, X, lN, vN


def test_version_and_api():
    v = conifer.__version__
    assert isinstance(v, str) and v.split(".")[0].isdigit(), f"unexpected __version__: {v!r}"
    assert hasattr(conifer, "fit_gated") and hasattr(conifer, "GatedResult")


def test_no_harm_defers_on_useless_covariates():
    counts, truth, X, lN, vN = _synth(m=48, signal=0.0, seed=1)
    m = counts.shape[0]
    r = conifer.fit_gated(counts, np.ones(m), X, total_logN=lN, var_logN=vN,
                          adequacy_target=truth, boot_g3=0, di_overdispersion=False)
    assert isinstance(r, conifer.GatedResult)
    assert r.s_hat_.shape == counts.shape
    assert r.rho_eff == 0.0                              # useless covariates -> floored out
    assert np.allclose(r.s_hat_, r.est_0cov_.s_hat_)     # exact no-harm: defers to covariate-free


def test_learner_contest_picks_linear_at_small_n():
    counts, truth, X, lN, vN = _synth(m=30, signal=1.5, seed=2)  # linear signal, small n
    learner, rho, r2_lin, r2_ml = conifer.choose_learner(X, truth)
    assert learner == "linear"                           # trees can't generalize at n=30
    assert 0.0 <= rho <= 1.0


def test_covariate_adequacy_range():
    counts, truth, X, lN, vN = _synth(m=60, signal=1.5, seed=3)
    assert 0.0 <= conifer.covariate_adequacy(X, truth) <= 1.0
