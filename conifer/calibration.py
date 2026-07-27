"""conifer.calibration - conformal calibration when you do NOT know the truth.

``DiameterDistribution.conformalize`` needs ``s_truth_cal``: the true stem density at a set
of calibration areas. In a simulation you have it. **In a real inventory you never do**,
which made the package's headline guarantee unreachable in practice. This module closes
that gap, and it is worth being precise about why the obvious shortcut fails.

The wrong way (and why)
-----------------------
It is tempting to calibrate against the design-direct estimate ``s_dir``, since it is
design-unbiased for the truth. **That does not work**, because CONIFER's estimate is itself
a shrinkage of ``s_dir``:

    s_hat = w * s_dir + (1 - w) * s_model

so ``s_hat - s_dir = (1 - w)(s_model - s_dir)`` is mechanically *smaller* than the real
error ``s_hat - s_true``. Calibrating on it produces sets that are too narrow. Measured on
simulated data where the truth is known, this under-covers badly (~68% for a nominal 90%
set). :func:`conformalize_naive` implements it only so the failure is documented and
testable; do not use it for anything you report.

The right way
-------------
Split each stand's **plots** in two. Fit on half; calibrate against the direct estimate
computed from the *other* half. The calibration target is then statistically independent of
the fit, and the residual decomposes cleanly:

    E|s_hat_A - s_dir_B|^2  =  E|s_hat_A - s_true|^2  +  Var(s_dir_B)

Both extra terms push the same way, so the resulting set is **conservative** - it covers at
least the stated rate:

* ``Var(s_dir_B)`` - the holdout half carries its own sampling noise, which inflates the score;
* ``s_hat_A`` is fit on half the plots, so it is worse than the estimate you actually report.

**The price.** Conservative means wide. Measured on the demo cruise (median 3 plots/stand,
truth known), a nominal 90% joint set covers 99.4% and is ~2.7x wider than an oracle set
calibrated on the truth itself; at nominal 80% it covers 90.5% at ~1.5x. The inflation is
``Var(s_dir_B)``, and on a thin cruise the holdout half is a single plot, so it dominates.
Tuning ``frac`` does not fix this (0.25/0.34/0.50 all land near 2.7x), and raising
``min_plots`` makes it worse by shrinking the calibration set.

Two efficiency fixes were tried and **rejected**, recorded here so they are not retried:

1. *Moment-matched deflation* of the known ``Var(s_dir_B)``. Over-corrected to 67% coverage -
   worse than the naive method it was meant to improve on. The reason is instructive: the
   deflation factor is ``sqrt(1 + Var(s_dir_B) / s_var_)``, so it leans on the model variance
   ``s_var_`` being well calibrated. That is precisely what conformal prediction exists to
   repair, which makes the correction circular. Using a better-estimated ``Var(s_dir_B)``
   (pooled per-plot variance rather than the 1-plot holdout) does not rescue it - the ratio
   is dominated by ``s_var_``, not by the numerator.
2. *Per-class marginal sets instead of the joint band*, to avoid the multiplicity price
   across K classes. Tempting - on one simulated cruise it held 94% joint coverage at less
   than a third of the width. It does not replicate: across three cruises joint coverage was
   0.75 at nominal 90% and 0.42 at nominal 80%. Marginal calibration does not give joint
   validity, and a single favourable seed made it look as though it might.

What *does* help, modestly: at the 90% level the **minimum-volume simplex ellipsoid**
(``mode='min_vol'``) is about 27% tighter than the L-infinity band at equal or better joint
coverage (0.983 vs 0.976 over three cruises). At 80% the ordering reverses. Evidence so far
is three simulated cruises, which is not enough to change the default on.

Closing the remaining ~2x gap without giving up validity is open work. Until then this errs
wide, on purpose, and says so.

:func:`conformalize_holdout` is that construction and is the default. Everything here calls
the validated engine's own ``conformalize``; none of it reimplements conformal machinery.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "conformalize_holdout", "conformalize_naive", "coverage_check", "CalibrationReport",
    "conformalize_direct",
]

_MV = ("min_vol", "minvol", "ellipsoid")


class CalibrationReport(dict):
    """Result of a calibration, carrying a plain-language verdict in ``.summary``."""

    def __repr__(self) -> str:  # pragma: no cover - display only
        return self.get("summary", dict.__repr__(self))

    @property
    def summary(self) -> str:
        return self.get("summary", "")


def _inv_of(est, inventory=None):
    inv = inventory if inventory is not None else getattr(est, "inventory_", None)
    if inv is None:
        raise ValueError(
            "Calibration needs the Inventory the estimator was fit from.\n"
            "Fit through `inventory.fit()` (recommended), or pass inventory=... explicitly."
        )
    return inv


def _fit_kwargs(est):
    """Reconstruct the estimator's configuration so the holdout refit matches it.

    Read by **introspecting the constructor** rather than from a hand-maintained list. Every
    kwarg that changes the fit must be carried into the holdout refit; if one is missed, the
    refit silently runs a *different* model from the one being reported and the calibrated
    threshold is transferred from the wrong estimator. A hardcoded list goes stale the moment
    someone adds a kwarg - ``di_overdispersion`` in v0.2 was exactly that - so this reads the
    signature instead and picks up new settings automatically.

    ``seed`` is excluded because the caller sets it explicitly.
    """
    import inspect

    try:
        params = inspect.signature(type(est).__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic estimators
        return {}
    return {name: getattr(est, name)
            for name, p in params.items()
            if name not in ("self", "seed")
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            and hasattr(est, name)}


def _copy_threshold(src, dst, joint, mode):
    """Move a calibrated conformal threshold from the holdout fit onto the reported fit.

    The thresholds are on the standardized score scale (residual divided by the fit's own
    standard error), which is what makes the transfer meaningful across two fits of
    different precision.
    """
    dst._conf_joint = bool(joint)
    dst._conf_mode = mode
    dst._mv_geom = getattr(src, "_mv_geom", "count")
    if joint and mode in _MV:
        dst._mv_Sig = getattr(src, "_mv_Sig", {})
        dst._mv_Sinv = getattr(src, "_mv_Sinv", {})
        dst._mv_tau = getattr(src, "_mv_tau", {})
    elif joint:
        dst.conf_joint_ = dict(getattr(src, "conf_joint_", {}))
    else:
        dst.conf_fac_ = {g: np.asarray(v) for g, v in getattr(src, "conf_fac_", {}).items()}


def conformalize_holdout(
    est,
    *,
    inventory=None,
    alpha: float = 0.10,
    joint: bool = False,
    mode: str = "maxscore",
    geom: str = "count",
    frac: float = 0.5,
    reps: int = 5,
    min_plots: int = 2,
    seed: int = 0,
    verbose: bool = False,
):
    """Calibrate CONIFER's prediction sets with **no known truth**, honestly.

    Splits each stand's plots into a fit half and a holdout half, fits on the first, and
    calibrates against the second half's design-direct estimate. Because the two halves are
    independent, the calibration is valid; because the holdout is noisy and the fit is
    data-starved, the resulting set is **conservative** rather than optimistic.

    Parameters
    ----------
    est : fitted DiameterDistribution
        The estimator you intend to report. It is calibrated in place.
    alpha : float
        Miscoverage; ``0.10`` gives a 90% set.
    joint : bool
        ``True`` for a set valid simultaneously across all DBH classes.
    mode : {'maxscore', 'min_vol'}
        L-infinity band (robust, thin calibration data) or minimum-volume simplex
        ellipsoid (tighter, needs more calibration stands).
    frac : float
        Share of each stand's plots held out for calibration.
    reps : int
        Number of independent plot splits to average the threshold over. More is steadier;
        each rep costs one refit.
    min_plots : int
        A stand needs this many plots to be split, so only these stands calibrate.
    Returns
    -------
    CalibrationReport

    Notes
    -----
    Requires plot-level detail, i.e. an inventory built with
    ``from_treelist(..., plot_col=...)``. Falls back to :func:`conformalize_naive` with a
    loud warning if plot identifiers were never supplied.
    """
    inv = _inv_of(est, inventory)
    if inv.plot_stand is None or inv.plot_counts is None:
        import warnings
        warnings.warn(
            "No plot identifiers in this inventory, so the fit and the calibration target "
            "cannot be made independent. Falling back to the naive calibration, which is "
            "known to produce sets that are TOO NARROW. Supply `plot_col` to fix this.",
            stacklevel=2,
        )
        return conformalize_naive(est, inventory=inv, alpha=alpha, joint=joint, mode=mode,
                                  geom=geom, seed=seed)

    cfg = _fit_kwargs(est)
    thresholds, n_cal_used = [], []
    for r in range(max(1, reps)):
        inv_a, inv_b, elig = inv.split_plots(frac=frac, seed=seed + r, min_plots=min_plots)
        if elig.size < 8:
            break
        est_a = inv_a.fit(seed=getattr(est, "seed", 0), **cfg)
        target = inv_b.direct
        est_a.conformalize(target, elig, joint=joint, alpha=alpha, mode=mode, geom=geom)
        thresholds.append(est_a)
        n_cal_used.append(int(elig.size))
        if verbose:
            print(f"  split {r+1}/{reps}: calibrated on {elig.size} stands")

    if not thresholds:
        import warnings
        warnings.warn(
            f"Fewer than 8 stands have the {min_plots}+ plots needed to be split, so an "
            "independent calibration is not possible. Falling back to the naive calibration, "
            "which produces sets that are TOO NARROW. Report intervals with that caveat.",
            stacklevel=2,
        )
        return conformalize_naive(est, inventory=inv, alpha=alpha, joint=joint, mode=mode,
                                  geom=geom, seed=seed)

    # average the thresholds across splits, then transfer onto the reported fit
    ref = thresholds[0]
    if joint and mode in _MV:
        keys = set().union(*[set(t._mv_tau) for t in thresholds])
        ref._mv_tau = {k: float(np.mean([t._mv_tau[k] for t in thresholds if k in t._mv_tau]))
                       for k in keys}
    elif joint:
        keys = set().union(*[set(t.conf_joint_) for t in thresholds])
        ref.conf_joint_ = {k: float(np.mean([t.conf_joint_[k] for t in thresholds
                                             if k in t.conf_joint_])) for k in keys}
    else:
        keys = set().union(*[set(t.conf_fac_) for t in thresholds])
        ref.conf_fac_ = {k: np.mean([t.conf_fac_[k] for t in thresholds if k in t.conf_fac_],
                                    axis=0) for k in keys}
    _copy_threshold(ref, est, joint, mode)

    est._cal_kind_ = (f"plot-holdout conformal ({len(thresholds)} split"
                      f"{'s' if len(thresholds) != 1 else ''}, no known truth required"
                      + ")")
    est._cal_alpha_ = alpha
    est._cal_conservative_ = True

    nominal = int(round((1 - alpha) * 100))
    return CalibrationReport(
        summary=(
            f"Calibrated a {nominal}% {'simultaneous' if joint else 'per-class'} prediction "
            f"set using {int(np.mean(n_cal_used))} stands, without needing to know the truth: "
            f"CONIFER was fit on half of each stand's plots and scored against the other half. "
            "Because those halves are independent - and the holdout half is itself noisy - "
            "the resulting set errs wide rather than narrow: on a thin cruise expect it to "
            "cover well above the stated rate, and to be correspondingly wide."
        ),
        method="plot-holdout",
        n_calibration=int(np.mean(n_cal_used)),
        reps=len(thresholds),
        alpha=alpha,
        nominal=nominal / 100,
        joint=joint,
        mode=mode,
        distribution_free=True,
        conservative=True,
    )


def conformalize_naive(est, *, inventory=None, alpha=0.10, joint=True, mode="maxscore",
                       geom="count", min_plots=2, cal_frac=0.5, seed=0):
    """Calibrate against the design-direct estimate of the *same* plots. **Under-covers.**

    Documented and kept only so the failure is reproducible and testable. ``s_hat`` is a
    shrinkage of ``s_dir``, so the residual understates the true error and the sets come out
    too narrow. Use :func:`conformalize_holdout` for anything you report.
    """
    inv = _inv_of(est, inventory)
    s_dir = inv.direct
    m = s_dir.shape[0]
    n_plots = np.ones(m) if inv.n_plots is None else np.asarray(inv.n_plots, float)
    eligible = np.where(n_plots >= min_plots)[0]
    if eligible.size < 10:
        eligible = np.argsort(-n_plots)[: max(10, m // 2)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eligible)
    cal_idx = perm[: max(5, int(round(cal_frac * perm.size)))]
    est.conformalize(s_dir, cal_idx, joint=joint, alpha=alpha, mode=mode, geom=geom)
    est._cal_kind_ = "naive direct calibration (KNOWN TO UNDER-COVER)"
    est._cal_alpha_ = alpha
    est._cal_conservative_ = False
    nominal = int(round((1 - alpha) * 100))
    return CalibrationReport(
        summary=(f"Calibrated a nominal {nominal}% set against the direct estimate of the same "
                 f"plots used to fit. These sets are known to be TOO NARROW - the true coverage "
                 f"is materially below {nominal}%. Supply plot identifiers and use "
                 f"conformalize_holdout instead."),
        method="naive",
        n_calibration=int(cal_idx.size),
        alpha=alpha,
        nominal=nominal / 100,
        joint=joint,
        mode=mode,
        distribution_free=False,
        conservative=False,
        warning="under-covers",
    )


# Back-compatible name; now routed to the correct construction.
conformalize_direct = conformalize_holdout


def _measure(est, target, idx, joint, alpha):
    """Coverage on the scale of the guarantee being made.

    ``joint=True`` asks whether *every* class was contained at once; ``joint=False`` asks
    whether a class picked at random was contained. Reporting the joint number for a
    marginal interval would understate it badly and make a valid interval look broken.
    """
    lo, hi = est.predict_interval(joint=joint, alpha=alpha)
    lo = np.clip(np.asarray(lo, float), 0.0, None)
    hi = np.asarray(hi, float)
    inside = (target >= lo) & (target <= hi)
    hit = float(np.mean(inside[idx].all(1))) if joint else float(np.mean(inside[idx]))
    return hit, float(np.mean((hi - lo)[idx]))


def coverage_check(est, *, inventory=None, alpha=0.10, joint=False, mode="maxscore",
                   frac=0.5, reps=12, min_plots=2, seed=0, truth=None):
    """Measure how often the prediction set actually contains the field value.

    This is the number that earns a forester's trust, and it is computed the only honest way
    available without a known truth: repeatedly refit on half of each stand's plots, calibrate
    on part of the holdout, and test on the rest of the holdout - so the tested value never
    touched the fit or the calibration.

    Pass ``truth`` (an ``(m, K)`` array) to check against a real known truth instead; that is
    only possible in simulation, and it is what the package's own tests use.
    """
    inv = _inv_of(est, inventory)
    nominal = 1 - alpha

    if truth is not None:
        truth = np.asarray(truth, float)
        elig = np.where(np.asarray(inv.n_plots, float) >= min_plots)[0] \
            if inv.n_plots is not None else np.arange(inv.m)
        hits, widths = [], []
        for r in range(reps):
            p = np.random.default_rng(seed + r).permutation(elig)
            cut = max(4, p.size // 2)
            est.conformalize(truth, p[:cut], joint=joint, alpha=alpha, mode=mode)
            h, w = _measure(est, truth, p[cut:], joint, alpha)
            hits.append(h)
            widths.append(w)
        cov = float(np.mean(hits))
        note = "Measured against a known truth (simulation only)."
    else:
        if inv.plot_stand is None:
            raise ValueError("An honest coverage check needs plot identifiers. Rebuild the "
                             "inventory with `from_treelist(..., plot_col=...)`, or pass "
                             "truth=... if you are running a simulation.")
        cfg = _fit_kwargs(est)
        hits, widths = [], []
        for r in range(reps):
            inv_a, inv_b, elig = inv.split_plots(frac=frac, seed=seed + 500 + r,
                                                 min_plots=min_plots)
            if elig.size < 8:
                continue
            est_a = inv_a.fit(seed=getattr(est, "seed", 0), **cfg)
            target = inv_b.direct
            p = np.random.default_rng(seed + r).permutation(elig)
            cut = max(4, p.size // 2)
            est_a.conformalize(target, p[:cut], joint=joint, alpha=alpha, mode=mode)
            h, w = _measure(est_a, target, p[cut:], joint, alpha)
            hits.append(h)
            widths.append(w)
        if not hits:
            raise ValueError(f"Too few stands have {min_plots}+ plots to run a coverage check.")
        cov = float(np.mean(hits))
        note = ("Measured by holding out half of each stand's plots, so the tested value never "
                "informed the fit or the calibration. The held-out value is itself a noisy "
                "estimate of the truth, which makes this a conservative reading.")

    ok = cov >= nominal - 0.03
    scope = ("prediction set contained every diameter class at once" if joint
             else "interval for a given diameter class contained the held-out field value")
    verdict = (
        f"Across {len(hits)} independent splits, the {int(nominal*100)}% {scope} "
        f"{cov*100:.0f}% of the time"
    )
    verdict += (". That meets the stated level - the interval can be reported as advertised."
                if ok else
                f", short of the {int(nominal*100)}% target. Widen alpha, add calibration "
                "stands, or report these intervals as indicative only.")

    return CalibrationReport(
        summary=verdict,
        empirical=cov,
        nominal=nominal,
        joint=joint,
        meets_nominal=bool(ok),
        mean_width=float(np.mean(widths)) if widths else float("nan"),
        reps=len(hits),
        note=note,
    )
