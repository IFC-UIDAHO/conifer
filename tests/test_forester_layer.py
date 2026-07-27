"""Tests for the forester-facing layer (io / calibration / report / plots).

The important one is ``test_naive_calibration_undercovers_holdout_does_not``. It locks in a
real bug that was found and fixed during development: calibrating conformal sets against the
design-direct estimate of the *same* plots used to fit produces sets that are too narrow,
because CONIFER's estimate is a shrinkage of that very target. If someone later "simplifies"
conformalize_holdout back into conformalize_naive, this test fails.
"""
import numpy as np
import pandas as pd
import pytest

import conifer


@pytest.fixture(scope="module")
def cruise():
    return conifer.demo.make_cruise(n_stands=140, seed=11)


@pytest.fixture(scope="module")
def inv(cruise):
    trees, aux, _stands, _truth = cruise
    return conifer.from_treelist(trees, stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
                                 baf=20, aux=aux, aux_stand_col="STAND",
                                 group_col="STAND_TYPE")


# --- io ---------------------------------------------------------------------
def test_treelist_roundtrip_shapes(inv):
    assert inv.counts.shape == (inv.m, inv.K)
    assert inv.area_eff.shape == (inv.m,)
    assert inv.X.shape[0] == inv.m
    assert len(inv.stand_ids) == inv.m
    assert inv.D_ext is not None, "a prism cruise must get a design-based sampling covariance"


def test_class_labels_are_human(inv):
    assert all("in" in lab for lab in inv.class_labels)
    assert not any(lab.startswith("C") and lab[1:].isdigit() for lab in inv.class_labels)


def test_split_plots_partitions_exactly(inv):
    a, b, elig = inv.split_plots(seed=3)
    assert a.plot_counts.shape[0] + b.plot_counts.shape[0] == inv.plot_counts.shape[0]
    assert elig.size > 0
    # every eligible stand appears in BOTH halves - that is what makes calibration valid
    for i in elig[:20]:
        assert (a.plot_stand == i).sum() >= 1
        assert (b.plot_stand == i).sum() >= 1


def test_validation_catches_nan_covariates(inv):
    bad = conifer.from_matrices(inv.counts, inv.area_eff, inv.X.copy())
    bad.X[0, 0] = np.nan
    issues = bad.validate()
    assert any(i.code == "aux_nan" and i.level == "error" for i in issues)


def test_validation_catches_row_mismatch(inv):
    bad = conifer.from_matrices(inv.counts, inv.area_eff[:-1], inv.X)
    assert any(i.code == "row_mismatch" for i in bad.validate())


def test_missing_plot_col_is_flagged(cruise):
    trees, aux, _s, _t = cruise
    i2 = conifer.from_treelist(trees, stand_col="STAND", dbh_col="DBH_IN", baf=20,
                               aux=aux, aux_stand_col="STAND")
    assert any(i.code == "no_plot_ids" for i in i2.issues)


# --- calibration: the invariant that matters --------------------------------
@pytest.mark.slow
def test_naive_calibration_undercovers_holdout_does_not(inv, cruise):
    """Naive calibration must under-cover; the holdout construction must not.

    Measured against the KNOWN truth, which only a simulation has.
    """
    _t, _a, _s, truth = cruise
    T = truth.set_index("STAND").reindex(inv.stand_ids).to_numpy(float)
    est = inv.fit()
    alpha = 0.10

    conifer.conformalize_holdout(est, alpha=alpha, joint=True, reps=3)
    cov_holdout = float(np.mean(est.joint_covered(T)))

    conifer.conformalize_naive(est, alpha=alpha, joint=True)
    cov_naive = float(np.mean(est.joint_covered(T)))

    assert cov_holdout >= 1 - alpha - 0.03, (
        f"holdout calibration under-covered: {cov_holdout:.3f} < {1-alpha:.2f}")
    assert cov_naive < cov_holdout, (
        "naive calibration should be strictly worse than holdout; if this ever passes, "
        "re-check whether the estimator still shrinks toward the direct estimate")


def test_fit_kwargs_covers_every_constructor_arg():
    """Every kwarg that changes the fit must be copied into the holdout refit.

    A kwarg missing here means the calibration refit runs a different model from the one
    being reported - silently. ``di_overdispersion`` (v0.2) is the live example.
    """
    import inspect
    from conifer.calibration import _fit_kwargs

    sig = inspect.signature(conifer.DiameterDistribution.__init__)
    ctor = {p for p in sig.parameters if p not in ("self", "seed")}
    est = conifer.DiameterDistribution(seed=0)
    copied = set(_fit_kwargs(est))
    missing = {k for k in ctor if hasattr(est, k)} - copied
    assert not missing, (
        f"these constructor kwargs change the fit but are not copied into the holdout "
        f"refit: {sorted(missing)}. Add them to conifer/calibration.py::_fit_kwargs.")


# --- report -----------------------------------------------------------------
def test_tables_are_labelled_and_indexed_by_stand(inv):
    from conifer import report as R

    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=True, reps=2)
    for tab in (R.summary_table(est), R.stand_table(est), R.comparison_table(est)):
        assert tab.index.name == "stand"
        assert list(tab.index) == list(inv.stand_ids)
    cs = R.class_summary(est)
    assert cs.index.name == "DBH class"


def test_narrative_contains_no_placeholder(inv):
    from conifer import report as R

    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=True, reps=2)
    text = " ".join(R.narrative(est))
    assert "nan" not in text.lower()
    assert "{" not in text and "}" not in text


def test_qmd_and_basal_area_are_sane(inv):
    from conifer import report as R

    est = inv.fit()
    mid = inv.midpoints
    qmd = R.quadratic_mean_diameter(est.s_hat_, mid)
    ba = R.basal_area(est.s_hat_, mid, "in")
    assert np.all(qmd >= mid.min()) and np.all(qmd <= mid.max())
    assert np.all(ba > 0)


# --- the trust display -------------------------------------------------------
def test_data_gain_is_meaningful_and_tracks_effort(inv):
    """Gamma must exist, sit in (0,1), and rise with plot count.

    This is the number the report shows a forester as "% from this stand's own plots". It
    replaced ``w_adeq_``, which is all zeros under default settings and therefore read as
    "0% of this came from your field data" - false and alarming. If gamma ever stops
    tracking effort, the trust display has become decoration.
    """
    from conifer.report import data_gain

    est = inv.fit()
    g = data_gain(est)
    assert g is not None, "no data-gain available; the trust display would silently vanish"
    assert g.shape == (inv.m,)
    assert np.all((g > 0) & (g < 1))
    r = np.corrcoef(g, inv.n_plots)[0, 1]
    assert r > 0.4, f"data gain should rise with plot count, got corr={r:.2f}"


def test_no_zero_percent_field_data_claim(inv):
    """The report must never tell a forester 0% of the estimate came from their plots."""
    import re

    from conifer import report as R

    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=True, reps=2)
    tab = R.summary_table(est)
    own = [c for c in tab.columns if "own plots" in c or "stand's plots" in c]
    if own:
        assert (tab[own[0]] > 0).all(), "a stand is credited 0% of its own field data"
    # word-boundary match: plain "0% of" also occurs inside "20% of all stems"
    text = " ".join(R.narrative(est))
    assert not re.search(r"(?<![1-9])\b0%", text), f"a literal 0% claim leaked into: {text[:200]}"


def test_gis_export_field_names_are_legal(inv):
    """Joined columns must survive a shapefile/GeoPackage write (10 chars, no % or superscripts)."""
    import re
    from conifer import report as R
    from conifer.io import _gis_safe_columns

    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=True, reps=2)
    cols = list(_gis_safe_columns(R.summary_table(est).reset_index(drop=True)).columns)
    assert len(cols) == len(set(cols)), f"duplicate field names after slugging: {cols}"
    for c in cols:
        assert len(c) <= 10 and re.fullmatch(r"[0-9a-z_]+", c), f"illegal field name {c!r}"


# --- the interval that gets reported -----------------------------------------
@pytest.mark.slow
def test_marginal_interval_is_valid_and_much_narrower(inv, cruise):
    """The default (per-class) interval must hold its own guarantee AND be far narrower.

    This is the fix that took the reported interval from ~29x the estimate to ~3x. The joint
    L-infinity band pays a full K-class multiplicity price for a simultaneous promise almost
    no operational question needs; a forester asking "how many 10-15 inch stems?" is asking a
    marginal question. Both are honest, they just claim different things — so each is checked
    against *its own* scale here, never the other's.
    """
    _t, _a, _s, truth = cruise
    T = truth.set_index("STAND").reindex(inv.stand_ids).to_numpy(float)
    est = inv.fit()
    alpha = 0.10

    conifer.conformalize_holdout(est, alpha=alpha, joint=False, reps=3)
    lo, hi = est.predict_interval(joint=False, alpha=alpha)
    lo = np.clip(lo, 0, None)
    marginal_cov = float(np.mean((T >= lo) & (T <= hi)))
    marginal_w = float(np.median((hi - lo) / np.clip(est.s_hat_, 1e-9, None)))

    conifer.conformalize_holdout(est, alpha=alpha, joint=True, reps=3)
    jlo, jhi = est.predict_interval(joint=True, alpha=alpha)
    jlo = np.clip(jlo, 0, None)
    joint_cov = float(np.mean(((T >= jlo) & (T <= jhi)).all(1)))
    joint_w = float(np.median((jhi - jlo) / np.clip(est.s_hat_, 1e-9, None)))

    assert marginal_cov >= 1 - alpha - 0.03, (
        f"per-class interval under-covers per-class: {marginal_cov:.3f}")
    assert joint_cov >= 1 - alpha - 0.03, (
        f"joint set under-covers jointly: {joint_cov:.3f}")
    assert marginal_w < joint_w, (
        "the per-class interval should be narrower than the joint set; if it is not, the "
        "multiplicity correction is not doing anything and the default should be revisited")


def test_coverage_check_measures_the_right_scale(inv):
    """A marginal interval must not be scored against the joint yardstick.

    Doing so understates it badly and makes a perfectly valid interval look broken — which is
    what the code did before ``_measure`` existed.
    """
    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=False, reps=2)
    marg = conifer.coverage_check(est, alpha=0.10, joint=False, reps=4)
    assert marg["joint"] is False
    assert "diameter class contained" in marg["summary"]
    assert marg["empirical"] >= 0.85
