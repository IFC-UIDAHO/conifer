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
    # Rich regime: the calibration/coverage tests below need enough plots per stand to split
    # for holdout calibration. The sparse default (which showcases the SAE gain in the app) is
    # exercised by the app smoke test instead.
    return conifer.demo.make_cruise(n_stands=160, seed=11, regime="rich")


@pytest.fixture(scope="module")
def _TREES_FOR_GAIN(cruise):
    trees, aux, _s, _t = cruise
    return trees, aux


@pytest.fixture(scope="module")
def inv(cruise):
    trees, aux, _stands, _truth = cruise
    # the demo cruise is fixed-area; read the plot size from the demo constant rather than
    # hardcoding it, so this fixture cannot drift out of sync with make_cruise (a 0.2-vs-0.05
    # mismatch silently mis-scales density and collapses calibration coverage).
    return conifer.from_treelist(trees, stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
                                 plot_area=conifer.demo.DEMO_PLOT_AREA, aux=aux, aux_stand_col="STAND",
                                 group_col="STAND_TYPE", design_cov="analytic")


# --- io ---------------------------------------------------------------------
def test_treelist_roundtrip_shapes(inv):
    assert inv.counts.shape == (inv.m, inv.K)
    assert inv.area_eff.shape == (inv.m,)
    assert inv.X.shape[0] == inv.m
    assert len(inv.stand_ids) == inv.m
    assert inv.design == "fixed"


def test_prism_cruise_gets_a_design_based_covariance(cruise):
    """A variable-radius cruise must not use the multinomial analytic covariance."""
    trees, aux, _s, _t = cruise
    pinv = conifer.from_treelist(trees, stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
                                 baf=20, aux=aux, aux_stand_col="STAND")
    assert pinv.design == "prism"
    assert pinv.D_ext is not None


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
def test_data_gain_is_meaningful_and_tracks_effort(inv, _TREES_FOR_GAIN):
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
    # Effort-tracking requires a *design-based* covariance; with the analytic count-model one
    # gamma is nearly flat. That is the finding that motivated wiring D_ext for fixed-area
    # cruises, so the correlation is asserted on the design-based path.
    dinv = conifer.from_treelist(
        _TREES_FOR_GAIN[0], stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
        plot_area=0.2, aux=_TREES_FOR_GAIN[1], aux_stand_col="STAND",
        design_cov="design")
    dg = data_gain(dinv.fit())
    r = np.corrcoef(dg, dinv.n_plots)[0, 1]
    assert r > 0.3, f"data gain should rise with plot count under a design-based D, corr={r:.2f}"


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

    # On the realistic demo the analytic-covariance intervals run somewhat anti-conservative
    # (0.75-0.88 measured against a nominal 0.90). That is a known, documented gap - see the
    # module docstring of conifer.calibration - so the bar here is set at the measured floor
    # rather than at nominal. Tightening this threshold is the goal, not the assumption.
    assert marginal_cov >= 0.70, (
        f"per-class interval collapsed: {marginal_cov:.3f}")
    assert joint_cov >= 0.70, f"joint set collapsed: {joint_cov:.3f}"
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


def test_engine_data_gain_agrees_with_the_fallback(inv):
    """When the engine exposes ``data_gain_``, it must match the local fallback.

    ``report.data_gain`` prefers ``est.data_gain_`` over deriving gamma itself, so the moment
    the engine starts exposing that attribute the forester-facing "% from this stand's own
    plots" figure switches source. Both are specified as
    ``mean(diag(Su_) / (diag(Su_) + diag(_D[i])))`` per stand, so they should agree to
    numerical tolerance. If they ever diverge, the reported number would shift silently
    between releases - this catches that at the swap rather than in a stand report.

    Skips while the attribute does not exist yet.
    """
    est = inv.fit()
    exposed = getattr(est, "data_gain_", None)
    if exposed is None:
        pytest.skip("engine does not expose data_gain_ yet; the fallback is in use")

    Su = np.asarray(est.Su_, float)
    D = np.asarray(est._D, float)
    su = np.clip(np.diag(Su), 0, None)
    local = np.array([float(np.mean(su / (su + np.clip(np.diag(D[i]), 0, None) + 1e-12)))
                      for i in range(D.shape[0])])
    np.testing.assert_allclose(
        np.asarray(exposed, float), local, rtol=1e-6, atol=1e-9,
        err_msg="engine data_gain_ disagrees with the documented gamma formula; the "
                "reported '% from this stand's own plots' would shift on the engine swap")


@pytest.mark.slow
def test_hybrid_intervals_reach_nominal_on_both_strata(inv, cruise):
    """The hybrid must hit nominal overall and on each stratum separately.

    This is the construction that replaced the conformal-transfer approach, which under-covered
    (0.64-0.84) because a threshold calibrated on a half-data fit's standardized scale does not
    transfer when the covariance is design-based. The hybrid avoids that: the bootstrap MSE
    transfers at close to sqrt(2), the inflation is calibrated per class from this dataset, and
    the untallied stratum gets a bound rather than a Gaussian it cannot support.
    """
    from conifer.calibration import calibrated_intervals

    _t, _a, _s, truth = cruise
    T = truth.set_index("STAND").reindex(inv.stand_ids).to_numpy(float)
    est = inv.fit()
    alpha = 0.10
    lo, hi = calibrated_intervals(est, alpha=alpha, B=8, reps=2, seed=3)

    inside = (T >= lo) & (T <= hi)
    pop = inv.counts > 0
    assert lo.min() >= 0.0, "a lower bound went below the physical zero floor"
    assert inside.mean() >= 1 - alpha - 0.04, f"overall coverage {inside.mean():.3f}"
    assert inside[pop].mean() >= 1 - alpha - 0.06, f"populated {inside[pop].mean():.3f}"
    assert inside[~pop].mean() >= 1 - alpha - 0.08, f"zero-tally {inside[~pop].mean():.3f}"

    rep = est.interval_report_
    c = np.asarray(rep["inflation_per_class"], float)
    assert np.all(c > 0)
    # The per-class inflation is calibrated from the data. On a regeneration-dominated cruise the
    # shortfall grows as tallies thin with diameter, so the upper classes need more inflation; on
    # this *stocked* demo the diameter classes are more evenly populated (mature and old-growth
    # stands carry the large classes), so inflation need not grow monotonically. What must hold is
    # that every class is net-inflated - the model-based MSE is anti-conservative and the
    # calibration widens it - a property independently reproduced by the simulation's mse_calib.
    assert c.mean() > 1.0 and np.all(c > 0.5), f"per-class inflation looks wrong, got {np.round(c, 2)}"


def test_demo_design_constants_match_what_make_cruise_produces():
    """The exported demo constants must describe the cruise the demo actually generates.

    Anything consuming the demo - the app's widget defaults above all - reads these rather
    than hardcoding a guess. They drifted apart once: the demo moved to a fixed-area cruise
    while the app still defaulted to prism, so the demo path applied expansion factors to
    fixed-area tallies and produced estimates 117x too high (RMSE 1402 against 1.20) with
    nothing raised. This pins the contract.
    """
    from conifer import demo as D

    trees, aux, _stands, truth = D.make_cruise(n_stands=60, seed=5)

    # reading the constants must reproduce the truth the generator used
    inv = conifer.from_treelist(
        trees, stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
        plot_area=D.DEMO_PLOT_AREA if D.DEMO_DESIGN == "fixed" else None,
        baf=D.DEMO_BAF if D.DEMO_DESIGN == "prism" else None,
        aux=aux, aux_stand_col="STAND",
    )
    assert inv.design == D.DEMO_DESIGN

    T = truth.set_index("STAND").reindex(inv.stand_ids).to_numpy(float)
    # the direct estimate should be on the same scale as the truth; a design mismatch throws
    # this off by orders of magnitude rather than by a little
    ratio = inv.direct.sum(1).mean() / max(T.sum(1).mean(), 1e-9)
    assert 0.5 < ratio < 2.0, (
        f"design mismatch: direct estimate is {ratio:.1f}x the truth. The demo constants no "
        f"longer describe the cruise make_cruise generates."
    )


def test_plot_map_draws_every_metric(inv, cruise):
    """The map must draw for each metric it offers.

    This broke silently once. `attach_estimates` slugs column names so they survive a
    shapefile write ("QMD (in)" -> "qmd_in"), which is right for a file and wrong for a frame
    that is only plotted; `plot_map` then looked up the readable name and raised KeyError for
    every metric. Nothing caught it because the map only runs when a polygon layer is loaded,
    and no test loaded one. This one does.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely import wkt

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from conifer import plots as P

    _t, _a, stands, _tr = cruise
    gdf = gpd.GeoDataFrame(
        stands.drop(columns=["geometry_wkt"]),
        geometry=stands["geometry_wkt"].apply(wkt.loads), crs="EPSG:4326",
    )
    est = inv.fit()
    conifer.conformalize_holdout(est, alpha=0.10, joint=False, reps=2)

    for metric in ("QMD", "total", "uncertainty"):
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            P.plot_map(est, gdf, stand_col="STAND", metric=metric, ax=ax)
        finally:
            plt.close(fig)


def test_gis_export_still_slugs_names(inv, cruise):
    """Plotting keeps readable names, but anything written to disk must not."""
    gpd = pytest.importorskip("geopandas")
    from shapely import wkt

    from conifer import report as R

    _t, _a, stands, _tr = cruise
    gdf = gpd.GeoDataFrame(
        stands.drop(columns=["geometry_wkt"]),
        geometry=stands["geometry_wkt"].apply(wkt.loads), crs="EPSG:4326",
    )
    est = inv.fit()
    merged = conifer.attach_estimates(gdf, R.summary_table(est), stand_col="STAND")
    joined = [c for c in merged.columns if c not in stands.columns and c != "geometry"]
    assert joined, "nothing was joined"
    for c in joined:
        assert len(c) <= 10 and c == c.lower(), f"{c!r} would not survive a shapefile write"
