"""conifer.io - get a forester's data into CONIFER without writing a pipeline.

CONIFER's engine takes three row-aligned matrices (``counts``, ``area_eff``, ``X``).
A forester has a *tree list* and a *stand layer*. This module is the bridge, and it is
deliberately opinionated about the two things that silently ruin an estimate:

1. **Row alignment.** Everything here carries ``stand_ids`` end to end. You never sort
   three arrays by hand again.
2. **Plot design.** Fixed-area and variable-radius (prism/BAF) cruises need *different*
   sampling covariances. ``from_treelist`` detects which you have and wires the correct
   one, including the design-based ``D_ext`` override for prism cruises.

Everything in this module is I/O and bookkeeping. It does not touch the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Inventory",
    "Issue",
    "from_treelist",
    "from_matrices",
    "read_stands",
    "attach_estimates",
    "FIA_2IN_BREAKS",
    "FIA_1IN_BREAKS",
    "DEFAULT_BREAKS_IN",
    "DEFAULT_BREAKS_CM",
]

# ---------------------------------------------------------------------------
# Conventional DBH class breaks. Edges are inclusive-low, exclusive-high.
# ---------------------------------------------------------------------------
DEFAULT_BREAKS_IN = np.array([1, 5, 9, 13, 17, 21, 25], float)   # six 4" classes
FIA_2IN_BREAKS = np.arange(1.0, 29.1, 2.0)                        # 2" classes, 1-29"
FIA_1IN_BREAKS = np.arange(1.0, 21.1, 1.0)                        # 1" classes, 1-21"
DEFAULT_BREAKS_CM = np.array([2.5, 12.5, 22.5, 32.5, 42.5, 52.5, 62.5], float)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    """One data problem, in words a forester can act on."""

    level: str        # "error" | "warning" | "note"
    code: str
    message: str
    fix: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        tag = {"error": "ERROR", "warning": "WARNING", "note": "note"}[self.level]
        s = f"[{tag}] {self.message}"
        return s + (f"\n         -> {self.fix}" if self.fix else "")


def _issues_frame(issues: Sequence[Issue]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"level": i.level, "code": i.code, "message": i.message, "suggested fix": i.fix}
         for i in issues],
        columns=["level", "code", "message", "suggested fix"],
    )


def _fmt(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------
@dataclass
class Inventory:
    """Everything CONIFER needs, aligned, labelled, and unit-aware.

    Attributes
    ----------
    counts : (m, K) float
        Class tallies per stand. For a fixed-area cruise these are the raw tree tallies
        (the multinomial sampling model the engine's analytic ``D_i`` assumes). For a prism
        cruise these are *effective* counts: the raw tally total redistributed across
        classes by the expansion-weighted composition, so that ``counts / area_eff``
        reproduces the design-unbiased stems/acre while ``counts.sum(1)`` still carries the
        real field effort.
    area_eff : (m,) float
        Effective area the stand's sample represents, in acres (hectares in metric).
    X : (m, p) float
        Auxiliary covariates, row-aligned to ``counts``.
    stand_ids : (m,) object
        The forester's own stand identifiers. Carried through every output.
    edges : (K+1,) float
        DBH class edges, in ``dbh_units``.
    D_ext : (m, K-1, K-1) float or None
        Design-based sampling covariance, set for prism cruises. ``None`` lets the engine
        use its analytic covariance, which is correct for fixed-area plots.
    """

    counts: np.ndarray
    area_eff: np.ndarray
    X: np.ndarray
    stand_ids: np.ndarray
    edges: np.ndarray
    groups: np.ndarray | None = None
    D_ext: np.ndarray | None = None
    aux_names: list = field(default_factory=list)
    n_plots: np.ndarray | None = None
    plot_spa: np.ndarray | None = None
    plot_stand: np.ndarray | None = None
    plot_counts: np.ndarray | None = None
    plot_area: float = 1.0
    design: str = "fixed"
    dbh_units: str = "in"
    area_units: str = "acre"
    issues: list = field(default_factory=list)

    # -- derived --------------------------------------------------------
    @property
    def m(self) -> int:
        return int(self.counts.shape[0])

    @property
    def K(self) -> int:
        return int(self.counts.shape[1])

    @property
    def midpoints(self) -> np.ndarray:
        """Class midpoint diameters, in ``dbh_units``."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def class_labels(self) -> list:
        """Human labels, e.g. ``'5-9 in'`` - never ``'C2'``."""
        u = self.dbh_units
        return [f"{_fmt(self.edges[k])}-{_fmt(self.edges[k + 1])} {u}" for k in range(self.K)]

    @property
    def direct(self) -> np.ndarray:
        """The design-direct estimate, stems per unit area, (m, K).

        This is what the forester would report from the cruise alone. CONIFER's whole claim
        is that it beats this; showing the two side by side is the argument.
        """
        return self.counts / np.clip(self.area_eff, 1e-12, None)[:, None]

    # -- validation -----------------------------------------------------
    def validate(self, raise_on_error: bool = False) -> list:
        """Check the data before it reaches the estimator. Returns the issue list."""
        iss = []
        c, a, X = self.counts, self.area_eff, self.X
        m, K = c.shape

        if not (c.shape[0] == a.shape[0] == X.shape[0] == len(self.stand_ids)):
            iss.append(Issue("error", "row_mismatch",
                             f"Row counts disagree: counts={c.shape[0]}, area={a.shape[0]}, "
                             f"covariates={X.shape[0]}, stand ids={len(self.stand_ids)}.",
                             "Every input must have one row per stand, in the same order."))
        if np.any(~np.isfinite(X)):
            n_bad = int(np.sum(~np.isfinite(X).all(1)))
            iss.append(Issue("error", "aux_nan",
                             f"{n_bad} stand(s) have a missing or infinite covariate value.",
                             "Drop those stands, or fill the gap (e.g. the region mean) before "
                             "fitting. CONIFER will otherwise return NaN without telling you."))
        if np.any(c < 0):
            iss.append(Issue("error", "negative_counts",
                             "Some class counts are negative.",
                             "Check the tree list for bad DBH or expansion-factor values."))
        if np.any(a <= 0):
            iss.append(Issue("error", "nonpositive_area",
                             f"{int(np.sum(a <= 0))} stand(s) have zero or negative effective area.",
                             "A stand with no sampled area cannot be estimated. Remove it, or "
                             "supply its plot area."))

        empty_stands = np.where(c.sum(1) <= 0)[0]
        if empty_stands.size:
            iss.append(Issue("warning", "empty_stand",
                             f"{empty_stands.size} stand(s) tallied zero trees "
                             f"(e.g. {list(np.asarray(self.stand_ids)[empty_stands][:3])}).",
                             "Their estimate will be driven entirely by the model, not by field "
                             "data. That is legitimate borrowing, but flag it in the report."))
        empty_classes = np.where(c.sum(0) <= 0)[0]
        if empty_classes.size:
            labs = [self.class_labels[k] for k in empty_classes]
            iss.append(Issue("warning", "empty_class",
                             f"No trees anywhere in class(es): {', '.join(labs)}.",
                             "CONIFER's hurdle model will push these to near-zero but not exactly "
                             "zero. Consider merging the empty tail into the class below it."))
        if m < 30:
            iss.append(Issue("warning", "few_areas",
                             f"Only {m} stands. Small-area estimation borrows strength *across* areas.",
                             "Below ~30 areas the between-area covariance is poorly estimated and "
                             "CONIFER over-shrinks toward the mean. Treat intervals as indicative."))
        if self.n_plots is not None:
            thin = int(np.sum(np.asarray(self.n_plots) < 2))
            if thin:
                iss.append(Issue("note", "thin_stands",
                                 f"{thin} stand(s) have fewer than 2 plots.",
                                 "This is exactly the case SAE exists for - expect a wide interval "
                                 "there. Note these stands cannot join the conformal calibration, "
                                 "which needs two independent halves of a stand's plots."))
        if X.shape[1] > max(3, m // 5):
            iss.append(Issue("warning", "many_covariates",
                             f"{X.shape[1]} covariates for {m} stands.",
                             "Consider trimming to the strongest predictors; the ML mean is "
                             "cross-fitted but still benefits from a leaner design matrix."))

        self.issues = iss
        if raise_on_error:
            errs = [i for i in iss if i.level == "error"]
            if errs:
                raise ValueError("CONIFER input validation failed:\n"
                                 + "\n".join(str(e) for e in errs))
        return iss

    def issue_table(self) -> pd.DataFrame:
        return _issues_frame(self.issues if self.issues else self.validate())

    # -- fitting --------------------------------------------------------
    def fit(self, **kwargs):
        """Validate, then fit ``DiameterDistribution`` with everything wired correctly.

        Any keyword is passed to the estimator constructor. The design-based ``D_ext``
        (prism cruises) and the Mondrian ``groups`` are supplied automatically, and the
        stand ids, class labels and units are attached to the fitted object so every
        downstream table and figure is human-readable.
        """
        from . import DiameterDistribution

        self.validate(raise_on_error=True)
        seed = kwargs.pop("seed", 0)
        est = DiameterDistribution(seed=seed, **kwargs)
        est.fit(self.counts, self.area_eff, self.X, groups=self.groups, D_ext=self.D_ext)
        est.stand_ids_ = np.asarray(self.stand_ids)
        est.edges_ = np.asarray(self.edges, float)
        est.class_labels_ = self.class_labels
        est.dbh_units_ = self.dbh_units
        est.area_units_ = self.area_units
        est.inventory_ = self
        return est

    # -- plot-level surgery (what makes honest calibration possible) ----
    def subset_plots(self, keep) -> "Inventory":
        """A new :class:`Inventory` built from a subset of the plots.

        This is what makes honest conformal calibration possible without knowing the truth:
        fit on one half of each stand's plots and calibrate against the *other* half's
        direct estimate, which is independent of the fit. See :mod:`conifer.calibration`.

        ``keep`` is a boolean mask, or an index array, over plots.
        """
        if self.plot_counts is None or self.plot_stand is None:
            raise ValueError("This Inventory has no plot-level detail. Build it with "
                             "`from_treelist(..., plot_col=...)` to enable plot splitting.")
        keep = np.asarray(keep)
        if keep.dtype != bool:
            mask = np.zeros(self.plot_counts.shape[0], bool)
            mask[keep] = True
            keep = mask
        m, K = self.m, self.K
        ps = self.plot_stand[keep]
        pc = self.plot_counts[keep]
        spa = self.plot_spa[keep]
        n_new = np.bincount(ps, minlength=m).astype(float)
        tally = np.zeros((m, K))
        np.add.at(tally, ps, pc)
        wsum = np.zeros((m, K))
        np.add.at(wsum, ps, spa)

        nz = np.clip(n_new, 1.0, None)
        D_ext = None
        if self.design == "fixed":
            counts = tally
            area_eff = nz * self.plot_area
            # Inherit the parent's covariance choice. If the parent used a design-based D and
            # the halves silently fell back to the analytic one, every calibration computed
            # from a split would be measuring a different estimator than the one reported -
            # the same mismatch that made bootstrap_mse need a design_D flag.
            if self.D_ext is not None and ps.size:
                from ._engine.sampling_cov import design_Di_from_plots
                D_ext = design_Di_from_plots(ps, spa, m, K)
        else:
            dens = wsum / nz[:, None]
            tot = tally.sum(1)
            comp = dens / np.clip(dens.sum(1, keepdims=True), 1e-12, None)
            counts = comp * tot[:, None]
            area_eff = np.where(dens.sum(1) > 0,
                                tot / np.clip(dens.sum(1), 1e-12, None), 1.0)
            area_eff = np.clip(area_eff, 1e-6, None)
            if ps.size:
                from ._engine.sampling_cov import design_Di_from_plots
                D_ext = design_Di_from_plots(ps, spa, m, K)

        return Inventory(
            counts=counts, area_eff=area_eff, X=self.X, stand_ids=self.stand_ids,
            edges=self.edges, groups=self.groups, D_ext=D_ext, aux_names=self.aux_names,
            n_plots=n_new, plot_spa=spa, plot_stand=ps, plot_counts=pc,
            plot_area=self.plot_area, design=self.design,
            dbh_units=self.dbh_units, area_units=self.area_units,
        )

    def split_plots(self, frac: float = 0.5, seed: int = 0, min_plots: int = 2):
        """Split each stand's plots into two independent halves.

        Returns ``(inv_a, inv_b, eligible)``. ``eligible`` indexes the stands that had at
        least ``min_plots`` plots and therefore appear in *both* halves. Stands below that
        threshold keep all their plots in ``inv_a`` - so the fit does not lose them - and
        are excluded from ``eligible``.
        """
        if self.plot_stand is None:
            raise ValueError("Plot-level detail is required. Use `from_treelist(..., plot_col=...)`.")
        rng = np.random.default_rng(seed)
        P = self.plot_stand.shape[0]
        a = np.ones(P, bool)
        elig = []
        for i in range(self.m):
            idx = np.where(self.plot_stand == i)[0]
            if idx.size >= min_plots:
                idx = idx.copy()
                rng.shuffle(idx)
                nb = max(1, int(round((1 - frac) * idx.size)))
                a[idx[:nb]] = False
                elig.append(i)
        return self.subset_plots(a), self.subset_plots(~a), np.asarray(elig, int)

    def direct_variance(self) -> np.ndarray:
        """Sampling variance of the design-direct estimate, per stand x class.

        Estimated empirically from the among-plot spread of expansion-weighted density, so
        it is valid for any plot design. Stands with fewer than two plots borrow the pooled
        per-plot variance. Used by :func:`conifer.calibration.conformalize_holdout` to
        remove the *known* noise the calibration target carries.
        """
        if self.plot_spa is None or self.plot_stand is None:
            raise ValueError("Plot-level detail is required to estimate the direct-estimator "
                             "variance. Use `from_treelist(..., plot_col=...)`.")
        m, K = self.m, self.K
        v = np.zeros((m, K))
        per_plot = []
        for i in range(m):
            sel = self.plot_stand == i
            n = int(sel.sum())
            if n >= 2:
                pv = self.plot_spa[sel].var(axis=0, ddof=1)
                v[i] = pv / n
                per_plot.append(pv)
        pooled = np.mean(per_plot, axis=0) if per_plot else np.clip(self.direct.mean(0), 1e-9, None)
        for i in range(m):
            n = int((self.plot_stand == i).sum())
            if n < 2:
                v[i] = pooled / max(n, 1)
        return np.clip(v, 1e-12, None)

    # -- display --------------------------------------------------------
    def counts_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.counts, index=pd.Index(self.stand_ids, name="stand"),
                            columns=self.class_labels)

    def direct_frame(self) -> pd.DataFrame:
        unit = "TPA" if self.area_units == "acre" else "TPH"
        return pd.DataFrame(self.direct, index=pd.Index(self.stand_ids, name="stand"),
                            columns=[f"{c} ({unit})" for c in self.class_labels])

    def describe(self) -> pd.DataFrame:
        """One-screen summary a forester can sanity-check before fitting."""
        area_label = ("effective sampled area" if self.design == "prism"
                      else "sampled area")
        rows = [
            ("Stands (small areas)", f"{self.m}"),
            ("DBH classes", f"{self.K}  ({self.class_labels[0]} ... {self.class_labels[-1]})"),
            ("Covariates", f"{self.X.shape[1]}"
             + (f"  ({', '.join(self.aux_names[:4])}...)" if self.aux_names else "")),
            ("Plot design", {"fixed": "fixed-area plots",
                             "prism": "variable-radius (prism / BAF)"}[self.design]),
            ("Total trees tallied", f"{int(self.counts.sum()):,}"),
            ("Plots per stand (median)",
             f"{np.median(self.n_plots):.0f}" if self.n_plots is not None else "not supplied"),
            ("Stands with 2+ plots",
             f"{int(np.sum(np.asarray(self.n_plots) >= 2))}" if self.n_plots is not None else "unknown"),
            (area_label.capitalize(), f"{self.area_eff.sum():,.2f} {self.area_units}s"),
            ("Sampling covariance", "design-based, from plot replicates" if self.D_ext is not None
             else "analytic, from the count model"),
        ]
        return pd.DataFrame(rows, columns=["", "value"]).set_index("")


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------
def from_treelist(
    trees: pd.DataFrame,
    *,
    stand_col: str,
    dbh_col: str,
    plot_col: str | None = None,
    breaks="default",
    design: str = "auto",
    plot_area=None,
    baf=None,
    tpa_col: str | None = None,
    species_col: str | None = None,
    aux: pd.DataFrame | None = None,
    aux_stand_col: str | None = None,
    group_col: str | None = None,
    dbh_units: str = "in",
    area_units: str = "acre",
    min_dbh: float | None = None,
    design_cov: str = "auto",
) -> Inventory:
    """Turn a tree list into a CONIFER-ready :class:`Inventory`.

    Parameters
    ----------
    trees : DataFrame
        One row per tallied tree. Must have a stand column and a DBH column.
    stand_col, dbh_col, plot_col : str
        Column names. ``plot_col`` is optional but strongly recommended: with it CONIFER
        can compute a *design-based* sampling covariance from plot replicates, and can
        calibrate its prediction sets honestly by splitting a stand's plots in two.
    breaks : sequence or {'default', 'fia_2in', 'fia_1in', 'metric'}
        DBH class edges in ``dbh_units``. ``'default'`` is six 4-inch classes, 1-25".
    design : {'auto', 'fixed', 'prism'}
        ``'fixed'`` - fixed-area plots; supply ``plot_area`` (acres per plot).
        ``'prism'`` - variable-radius/BAF; supply ``baf`` or a per-tree ``tpa_col``.
        ``'auto'``  - inferred from which of those you supplied.
    plot_area : float or str
        Acres per plot (a number), or the name of a column holding it.
    baf : float or str
        Basal area factor (ft^2/acre), or a column name. Each tree's trees-per-acre
        expansion is then ``BAF / (0.005454 * DBH^2)``.
    tpa_col : str
        Per-tree expansion factor, if you already have one. Overrides ``baf``.
    aux : DataFrame
        Stand-level auxiliary covariates (LiDAR metrics, spectral indices, terrain...).
        Must contain ``aux_stand_col`` (defaults to ``stand_col``); every other numeric
        column becomes a covariate.
    group_col : str
        A column in ``aux`` giving a stratum/type label, used as CONIFER's Mondrian group
        so conformal sets are calibrated *within* stand type.
    min_dbh : float
        Drop trees below this DBH before binning (a merchantability threshold).
    design_cov : {'auto', 'design', 'analytic'}
        Which sampling covariance to give the estimator. ``'design'`` builds it empirically
        from the between-plot spread, which is valid for any plot design and makes the
        reported data gain respond to sampling effort. ``'analytic'`` uses the engine's
        count-model covariance. ``'auto'`` (default) uses the design-based one whenever
        enough stands carry plot replicates, and falls back to analytic otherwise.

    Notes
    -----
    **Why the plot design matters.** CONIFER's default sampling covariance is analytic and
    assumes tallies from a fixed-area plot. On a prism cruise that assumption is wrong -
    each tree carries a DBH-dependent expansion factor. With ``design='prism'`` this
    function instead builds the covariance empirically from the among-plot spread of
    expansion-weighted stems/acre (``D_ext``), which is valid for *any* plot design, and
    passes effective counts that keep the direct estimate design-unbiased while preserving
    the real field effort in ``counts.sum(1)``.
    """
    if stand_col not in trees.columns:
        raise KeyError(f"stand column {stand_col!r} is not in the tree list. "
                       f"Columns: {list(trees.columns)}")
    if dbh_col not in trees.columns:
        raise KeyError(f"DBH column {dbh_col!r} is not in the tree list. "
                       f"Columns: {list(trees.columns)}")

    df = trees.copy()
    df[dbh_col] = pd.to_numeric(df[dbh_col], errors="coerce")
    n_before = len(df)
    df = df[np.isfinite(df[dbh_col])]
    dropped_nan = n_before - len(df)
    if min_dbh is not None:
        df = df[df[dbh_col] >= float(min_dbh)]

    edges = _resolve_breaks(breaks, dbh_units)
    K = len(edges) - 1
    design = _resolve_design(design, plot_area, baf, tpa_col)

    # --- per-tree expansion ----------------------------------------------
    if tpa_col is not None and tpa_col in df.columns:
        expf = pd.to_numeric(df[tpa_col], errors="coerce").fillna(0.0).to_numpy(float)
    elif design == "prism":
        b = df[baf].to_numpy(float) if isinstance(baf, str) else float(baf)
        expf = b / (0.005454 * np.clip(df[dbh_col].to_numpy(float), 1e-6, None) ** 2)
    else:
        pa = (df[plot_area].to_numpy(float) if isinstance(plot_area, str)
              else float(plot_area if plot_area else 1.0))
        expf = 1.0 / np.clip(pa, 1e-12, None) * np.ones(len(df))

    # --- bin --------------------------------------------------------------
    kidx = np.digitize(df[dbh_col].to_numpy(float), edges) - 1
    in_range = (kidx >= 0) & (kidx < K)
    dropped_range = int((~in_range).sum())
    df = df.loc[in_range].copy()
    kidx = kidx[in_range]
    expf = expf[in_range]
    df["_k"] = kidx
    df["_expf"] = expf

    stands = pd.Index(pd.unique(df[stand_col]), name="stand")
    if aux is not None:
        acol = aux_stand_col or stand_col
        if acol not in aux.columns:
            raise KeyError(f"{acol!r} is not a column in the auxiliary table. "
                           f"Columns: {list(aux.columns)}")
        stands = pd.Index([s for s in stands if s in set(aux[acol])], name="stand")
    sidx = {s: i for i, s in enumerate(stands)}
    m = len(stands)
    if m == 0:
        raise ValueError("No stands survive after joining the tree list to the auxiliary "
                         "table. Check that the stand identifiers match exactly - a common "
                         "cause is one side being text and the other numeric, or trailing "
                         "whitespace.")
    df = df[df[stand_col].isin(set(stands))]
    si = df[stand_col].map(sidx).to_numpy(int)
    kk = df["_k"].to_numpy(int)
    ee = df["_expf"].to_numpy(float)

    tally = np.zeros((m, K))
    np.add.at(tally, (si, kk), 1.0)
    wsum = np.zeros((m, K))
    np.add.at(wsum, (si, kk), ee)

    # --- plots ------------------------------------------------------------
    plot_spa = plot_stand = plot_counts = None
    if plot_col is not None and plot_col in df.columns:
        pkey = df[stand_col].astype(str) + "||" + df[plot_col].astype(str)
        plots = pd.Index(pd.unique(pkey))
        pidx = {p: i for i, p in enumerate(plots)}
        pi = pkey.map(pidx).to_numpy(int)
        P = len(plots)
        plot_spa = np.zeros((P, K))
        np.add.at(plot_spa, (pi, kk), ee)
        plot_counts = np.zeros((P, K))
        np.add.at(plot_counts, (pi, kk), 1.0)
        plot_stand = np.zeros(P, int)
        plot_stand[pi] = si
        n_plots = np.bincount(plot_stand, minlength=m).astype(float)
    else:
        n_plots = np.ones(m)
    n_plots = np.clip(n_plots, 1.0, None)

    # --- counts + effective area, by design -------------------------------
    D_ext = None
    if design == "fixed":
        pa_val = (float(np.mean(pd.to_numeric(trees[plot_area], errors="coerce")))
                  if isinstance(plot_area, str) else float(plot_area))
        counts = tally
        area_eff = n_plots * pa_val
        # A design-based sampling covariance is available for a fixed-area cruise too,
        # whenever stands carry plot replicates - and it is better behaved than the analytic
        # count-model one, which assumes a multinomial tally and was the source of the v0.2
        # over-shrinkage. It also makes the reported data gain track sampling effort, which
        # the analytic covariance does not.
        if design_cov in ("auto", "design") and plot_stand is not None:
            n_repl = int(np.sum(n_plots >= 2))
            if design_cov == "design" or n_repl >= max(8, 0.25 * m):
                from ._engine.sampling_cov import design_Di_from_plots
                D_ext = design_Di_from_plots(plot_stand, plot_spa, m, K)
    else:
        pa_val = 1.0
        dens = wsum / n_plots[:, None]
        tot_tally = tally.sum(1)
        comp = dens / np.clip(dens.sum(1, keepdims=True), 1e-12, None)
        counts = comp * tot_tally[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            area_eff = np.where(dens.sum(1) > 0,
                                tot_tally / np.clip(dens.sum(1), 1e-12, None), 1.0)
        area_eff = np.clip(area_eff, 1e-6, None)
        if plot_spa is not None and design_cov != "analytic":
            from ._engine.sampling_cov import design_Di_from_plots
            D_ext = design_Di_from_plots(plot_stand, plot_spa, m, K)

    # --- covariates --------------------------------------------------------
    groups = None
    if aux is not None:
        acol = aux_stand_col or stand_col
        A = aux.drop_duplicates(subset=[acol]).set_index(acol).reindex(stands)
        if group_col and group_col in A.columns:
            groups = pd.Categorical(A[group_col]).codes.astype(int)
            A = A.drop(columns=[group_col])
        num = A.select_dtypes(include=[np.number])
        aux_names = list(num.columns)
        X = num.to_numpy(float)
        if X.shape[1] == 0:
            raise ValueError("The auxiliary table has no numeric columns to use as "
                             "covariates. CONIFER needs at least one predictor to borrow "
                             "strength from.")
    else:
        X = np.column_stack([np.ones(m), np.log1p(tally.sum(1))])
        aux_names = ["intercept", "log_tally"]

    inv = Inventory(
        counts=np.asarray(counts, float),
        area_eff=np.asarray(area_eff, float),
        X=np.asarray(X, float),
        stand_ids=np.asarray(stands),
        edges=edges,
        groups=groups,
        D_ext=D_ext,
        aux_names=aux_names,
        n_plots=n_plots,
        plot_spa=plot_spa,
        plot_stand=plot_stand,
        plot_counts=plot_counts,
        plot_area=pa_val,
        design=design,
        dbh_units=dbh_units,
        area_units=area_units,
    )
    inv.validate()
    if dropped_nan:
        inv.issues.insert(0, Issue("note", "dropped_nan_dbh",
                                   f"Dropped {dropped_nan} tree(s) with a missing or "
                                   "non-numeric DBH.", ""))
    if dropped_range:
        inv.issues.insert(0, Issue("note", "dropped_out_of_range",
                                   f"Dropped {dropped_range} tree(s) whose DBH fell outside "
                                   f"{_fmt(edges[0])}-{_fmt(edges[-1])} {dbh_units}.",
                                   "Widen `breaks` if those trees should be included."))
    if aux is None:
        inv.issues.insert(0, Issue("warning", "no_covariates",
                                   "No auxiliary data supplied.",
                                   "Small-area estimation borrows strength through covariates. "
                                   "Without LiDAR, spectral or terrain predictors CONIFER "
                                   "cannot do much more than the direct estimate."))
    if plot_col is None:
        inv.issues.insert(0, Issue("warning", "no_plot_ids",
                                   "No plot identifiers supplied.",
                                   "Without them the prediction sets cannot be calibrated "
                                   "honestly (the fit and the calibration target would share "
                                   "the same data, which makes intervals too narrow). Supply "
                                   "`plot_col` if your tree list has a plot or point number."))
    return inv


def from_matrices(counts, area_eff, X, *, stand_ids=None, edges=None, groups=None,
                  aux_names=None, dbh_units="in", area_units="acre") -> Inventory:
    """Wrap arrays you already have in an :class:`Inventory` (adds labels and validation)."""
    counts = np.atleast_2d(np.asarray(counts, float))
    m, K = counts.shape
    if edges is not None:
        edges = np.asarray(edges, float)
    else:
        edges = _resolve_breaks("default", dbh_units)
    if len(edges) != K + 1:
        edges = np.arange(K + 1, dtype=float)
    ids = stand_ids if stand_ids is not None else [f"stand_{i + 1}" for i in range(m)]
    inv = Inventory(
        counts=counts,
        area_eff=np.ravel(np.asarray(area_eff, float)),
        X=np.atleast_2d(np.asarray(X, float)),
        stand_ids=np.asarray(ids),
        edges=edges,
        groups=None if groups is None else np.asarray(groups),
        aux_names=list(aux_names or []),
        dbh_units=dbh_units,
        area_units=area_units,
    )
    inv.validate()
    return inv


# ---------------------------------------------------------------------------
# Geospatial
# ---------------------------------------------------------------------------
def read_stands(path, *, stand_col: str, layer: str | None = None):
    """Read a stand polygon layer (shapefile / GeoPackage / GeoJSON) as a GeoDataFrame.

    Reprojects to EPSG:4326 for web mapping, and checks that ``stand_col`` exists and is
    unique. Requires ``geopandas``.
    """
    try:
        import geopandas as gpd
    except ImportError as e:  # pragma: no cover
        raise ImportError("read_stands needs geopandas:  pip install 'conifer-sae[geo]'") from e
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if stand_col not in gdf.columns:
        raise KeyError(f"{stand_col!r} is not a column in the stand layer. "
                       f"Found: {list(gdf.columns)}")
    dup = int(gdf[stand_col].duplicated().sum())
    if dup:
        raise ValueError(f"{dup} duplicated stand id(s) in {stand_col!r}. Each polygon needs a "
                         "unique id - dissolve multipart stands first.")
    if gdf.crs is None:
        raise ValueError("The stand layer has no CRS defined. Set one before using it "
                         "(a .prj file alongside the shapefile, or gdf.set_crs(...)).")
    return gdf.to_crs(4326)


_GIS_RENAME = {
    "total TPA": "tot_tpa", "total TPH": "tot_tph",
    "total TPA low": "tpa_lo", "total TPA high": "tpa_hi",
    "total TPH low": "tph_lo", "total TPH high": "tph_hi",
    "interval width (% of estimate)": "ci_pct",
    "% from field data": "pct_field", "% borrowed from model": "pct_model",
    "plots": "n_plots",
}


def _gis_safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns so they survive a GeoPackage / shapefile write.

    Shapefiles truncate field names to 10 characters and OGR rejects several characters we
    use for readability (``%``, ``(``, ``/``, ``²``). Left alone, a write either fails or
    silently produces duplicate truncated names, which is worse. Readable names are mapped
    to short, documented ones; anything unmapped is slugged and de-duplicated.
    """
    import re

    out, seen = [], {}
    for c in df.columns:
        name = _GIS_RENAME.get(c)
        if name is None:
            name = str(c).replace("²", "2").replace("%", "pct")
            name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()[:10] or "col"
        if name in seen:
            seen[name] += 1
            name = f"{name[:8]}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    df = df.copy()
    df.columns = out
    return df


def attach_estimates(gdf, table: pd.DataFrame, *, stand_col: str, how: str = "left",
                     gis_safe: bool = True):
    """Join a CONIFER results table onto the stand polygons.

    Returns a GeoDataFrame ready to write straight back out as a shapefile or GeoPackage.
    With ``gis_safe`` the joined columns are renamed to short, OGR-legal field names and the
    redundant join key is dropped, so ``.to_file(...)`` succeeds instead of erroring on a
    duplicate or illegal field.
    """
    t = table.reset_index()
    key = "stand" if "stand" in t.columns else t.columns[0]
    vals = t.drop(columns=[key])
    if gis_safe:
        vals = _gis_safe_columns(vals)
    t = pd.concat([t[[key]], vals], axis=1)

    out = gdf.merge(t, left_on=stand_col, right_on=key, how=how)
    if key != stand_col and key in out.columns:
        out = out.drop(columns=[key])          # the join key is already in `stand_col`
    if vals.shape[1]:
        missing = int(out[vals.columns[0]].isna().sum())
        if missing:
            import warnings
            warnings.warn(f"{missing} polygon(s) got no estimate - the stand ids did not match. "
                          "Check for type mismatches (101 vs '101') or trailing whitespace.")
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_breaks(breaks, dbh_units: str) -> np.ndarray:
    if isinstance(breaks, str):
        table = {
            "default": DEFAULT_BREAKS_IN if dbh_units == "in" else DEFAULT_BREAKS_CM,
            "fia_2in": FIA_2IN_BREAKS,
            "fia_1in": FIA_1IN_BREAKS,
            "metric": DEFAULT_BREAKS_CM,
        }
        if breaks not in table:
            raise ValueError(f"Unknown breaks preset {breaks!r}. "
                             f"Options: {list(table)}, or a list of edges.")
        return np.asarray(table[breaks], float)
    e = np.asarray(list(breaks), float)
    if e.ndim != 1 or e.size < 3:
        raise ValueError("`breaks` needs at least 3 edges (2 classes), ascending.")
    if np.any(np.diff(e) <= 0):
        raise ValueError("`breaks` must be strictly increasing.")
    return e


def _resolve_design(design, plot_area, baf, tpa_col) -> str:
    if design in ("fixed", "prism"):
        return design
    if design != "auto":
        raise ValueError("design must be 'auto', 'fixed', or 'prism'.")
    if baf is not None or tpa_col is not None:
        return "prism"
    if plot_area is not None:
        return "fixed"
    raise ValueError(
        "Cannot infer the plot design. Supply one of:\n"
        "  plot_area=<acres per plot>   for a fixed-area cruise\n"
        "  baf=<basal area factor>      for a prism / variable-radius cruise\n"
        "  tpa_col='<column>'           if each tree already carries its expansion factor"
    )
