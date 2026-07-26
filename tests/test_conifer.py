"""Smoke + naming tests for the CONIFER first release."""
import numpy as np

import conifer


def test_version_and_names():
    assert isinstance(conifer.__version__, str) and conifer.__version__[0].isdigit()  # a version string, not a pinned release
    # the engine and the estimator are one object today, exposed under two names
    assert conifer.CompositionalFH is conifer.DiameterDistribution


def test_backcompat_alias():
    # the pre-CONIFER class name still resolves to the same class
    from conifer._engine.stemclass import StemDensityClassSAE
    assert StemDensityClassSAE is conifer.DiameterDistribution


def test_fit_smoke():
    rng = np.random.default_rng(0)
    m, K, p = 50, 5, 4
    X = rng.normal(size=(m, p))
    shares = np.exp(X @ rng.normal(size=(p, K)))
    shares /= shares.sum(1, keepdims=True)
    area = rng.uniform(0.5, 2.0, m)
    counts = rng.poisson(shares * (area[:, None] * 200)).astype(float)

    est = conifer.DiameterDistribution(seed=0).fit(counts, area, X)
    assert est.s_hat_.shape == (m, K)
    assert np.all(est.s_hat_ >= 0)


def test_species_composition_is_stubbed():
    import pytest

    with pytest.raises(NotImplementedError):
        conifer.SpeciesComposition()
