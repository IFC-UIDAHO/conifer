"""conifer.plots - figures a forester reads, not diagnostics a statistician reads.

Every figure here uses real DBH class labels and forestry units. The two that matter most
for adoption are :func:`plot_comparison` (field-only vs CONIFER, both with error bars - the
argument for the model makes itself) and :func:`plot_coverage` (does the stated interval
actually hold up).
"""
from __future__ import annotations

import base64
import io

import numpy as np

__all__ = [
    "plot_estimate", "plot_distribution", "plot_comparison", "plot_coverage",
    "plot_borrowing", "plot_map", "fig_to_uri", "PALETTE",
]

PALETTE = {
    "green": "#2D6A4F",
    "light": "#95d5b2",
    "dark": "#1b3a26",
    "sand": "#f7f5f0",
    "line": "#dfe6e1",
    "muted": "#5d6d64",
    "direct": "#b08968",
    "warn": "#b45309",
}


def _meta(est):
    from .report import _meta as _m
    return _m(est)


def _iv(est, joint=True, alpha=0.10):
    """Prediction bounds, floored at zero (see conifer.report._intervals)."""
    from .report import _intervals
    return _intervals(est, joint=joint, alpha=alpha)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["line"])
    ax.grid(axis="y", color=PALETTE["line"], lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    return ax


# ---------------------------------------------------------------------------
def plot_distribution(est, stand=0, *, ax=None, joint=False, alpha=0.10, show_direct=True):
    """One stand's diameter distribution, with the calibrated prediction set as whiskers.

    If the estimator carries its :class:`~conifer.io.Inventory`, the field-only (direct)
    estimate is overlaid as hollow bars so the forester can see what the model changed.
    """
    import matplotlib.pyplot as plt

    M = _meta(est)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.8))
    i = int(stand) if isinstance(stand, (int, np.integer)) else int(np.where(M["ids"] == stand)[0][0])
    s = np.asarray(est.s_hat_, float)[i]
    x = np.arange(len(s))
    lo, hi = _iv(est, joint, alpha)

    inv = getattr(est, "inventory_", None)
    if show_direct and inv is not None:
        ax.bar(x, inv.direct[i], width=0.74, facecolor="none", edgecolor=PALETTE["direct"],
               lw=1.6, ls="--", label="field-only (this stand's plots)", zorder=2)
    ax.bar(x, s, width=0.62, color=PALETTE["green"], label="CONIFER estimate", zorder=3)
    if lo is not None:
        ax.errorbar(x, s, yerr=[np.maximum(s - lo[i], 0), np.maximum(hi[i] - s, 0)], fmt="none",
                    ecolor=PALETTE["dark"], capsize=4, lw=1.4, zorder=4,
                    label=f"{int((1-alpha)*100)}% interval"
                          + (" (all classes at once)" if joint else " (per class)"))
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(M["labels"], rotation=0, fontsize=8.5)
    ax.set_xlabel(f"DBH class ({M['dbh_units']})", fontsize=10)
    ax.set_ylabel(f"stems per {M['area_units']}", fontsize=10)
    ax.set_title(f"Stand {M['ids'][i]} - estimated diameter distribution",
                 fontsize=11.5, color=PALETTE["dark"], weight="semibold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
    return ax


def plot_estimate(est, area_index: int = 0, ax=None, joint: bool = True, alpha: float = 0.10):
    """Back-compatible alias of :func:`plot_distribution` (kept from v0.1)."""
    return plot_distribution(est, stand=area_index, ax=ax, joint=joint, alpha=alpha)


def plot_comparison(est, *, ax=None, joint=False, alpha=0.10, metric="total"):
    """Field-only vs CONIFER across all stands - the trust plot.

    Each stand is a point: its field-only estimate on x, CONIFER's on y, coloured by how
    many plots the stand had. Points on the 1:1 line mean the model changed nothing;
    departures should concentrate in thinly sampled stands, and that is exactly the claim
    small-area estimation makes. The vertical bars are CONIFER's prediction set.
    """
    import matplotlib.pyplot as plt

    inv = getattr(est, "inventory_", None)
    if inv is None:
        raise ValueError("plot_comparison needs the Inventory (fit via `inventory.fit()`).")
    M = _meta(est)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.4))

    s = np.asarray(est.s_hat_, float)
    d = inv.direct
    lo, hi = _iv(est, joint, alpha)
    if metric == "qmd":
        from .report import quadratic_mean_diameter as _q
        yv, xv = _q(s, M["midpoints"]), _q(d, M["midpoints"])
        lab = f"QMD ({M['dbh_units']})"
        ylo = yhi = None
    else:
        yv, xv = s.sum(1), d.sum(1)
        lab = f"total stems per {M['area_units']}"
        ylo, yhi = (lo.sum(1), hi.sum(1)) if lo is not None else (None, None)

    n = inv.n_plots if inv.n_plots is not None else np.ones(len(xv))
    if ylo is not None:
        ax.vlines(xv, ylo, yhi, color=PALETTE["light"], lw=1.4, zorder=1,
                  label=f"{int((1-alpha)*100)}% prediction set")
    sc = ax.scatter(xv, yv, c=n, cmap="YlGn", vmin=1, s=42, edgecolor=PALETTE["dark"],
                    lw=0.6, zorder=3)
    lim = [0, float(max(np.nanmax(xv), np.nanmax(yv))) * 1.06]
    ax.plot(lim, lim, ls="--", color=PALETTE["muted"], lw=1.2, zorder=2, label="1:1 (no change)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("plots in stand", fontsize=9, color=PALETTE["muted"])
    cb.ax.tick_params(labelsize=8, colors=PALETTE["muted"])
    _style(ax)
    ax.set_xlabel(f"field-only estimate - {lab}", fontsize=10)
    ax.set_ylabel(f"CONIFER estimate - {lab}", fontsize=10)
    ax.set_title("Where the model changed the answer", fontsize=11.5,
                 color=PALETTE["dark"], weight="semibold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return ax


def plot_coverage(report, *, ax=None):
    """The coverage badge as a figure: measured vs stated interval performance."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 1.9))
    emp = float(report.get("empirical", np.nan))
    nom = float(report.get("nominal", 0.9))
    ok = bool(report.get("meets_nominal", emp >= nom - 0.03))
    ax.barh([0], [emp], color=PALETTE["green"] if ok else PALETTE["warn"], height=0.42, zorder=3)
    ax.axvline(nom, color=PALETTE["dark"], ls="--", lw=1.6, zorder=4)
    ax.text(nom, 0.42, f"  stated {nom*100:.0f}%", va="bottom", ha="left",
            fontsize=9, color=PALETTE["dark"])
    ax.text(emp, 0, f" {emp*100:.0f}% measured ", va="center",
            ha="right" if emp > nom * 0.55 else "left", fontsize=11, weight="bold",
            color="white" if emp > nom * 0.55 else PALETTE["dark"], zorder=5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.7)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.set_title("How often the prediction set actually contained the field value",
                 fontsize=11, color=PALETTE["dark"], weight="semibold", loc="left")
    return ax


def plot_borrowing(est, *, ax=None):
    """How much of each stand's estimate is its own data vs borrowed strength."""
    import matplotlib.pyplot as plt

    from .report import data_gain

    g = data_gain(est)
    if g is None:
        raise ValueError("This fit does not expose the data-gain weights needed for this plot.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.4))
    own = 100 * np.asarray(g, float)
    order = np.argsort(own)
    x = np.arange(len(own))
    ax.fill_between(x, 0, own[order], color=PALETTE["green"], label="from this stand's own plots")
    ax.fill_between(x, own[order], 100, color=PALETTE["light"],
                    label="borrowed from similar stands")
    ax.set_xlim(0, max(len(own) - 1, 1))
    ax.set_ylim(0, 100)
    _style(ax)
    ax.set_xlabel("stands, sorted by how much field data they contribute", fontsize=10)
    ax.set_ylabel("share of the estimate (%)", fontsize=10)
    ax.set_title("Where each estimate comes from", fontsize=11.5,
                 color=PALETTE["dark"], weight="semibold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return ax


def plot_map(est, gdf, *, stand_col, metric="QMD", ax=None, alpha=0.10, joint=False,
             show_uncertainty=False, cmap="YlGn"):
    """Choropleth of a stand-level result over the stand polygons.

    ``metric`` is any column produced by :func:`conifer.report.summary_table`, or one of
    the shortcuts ``'QMD'``, ``'total'``, ``'uncertainty'``.
    """
    import matplotlib.pyplot as plt

    from .io import attach_estimates
    from .report import summary_table

    tab = summary_table(est, alpha=alpha, joint=joint)
    short = {
        "QMD": [c for c in tab.columns if c.startswith("QMD")],
        "total": [c for c in tab.columns
                  if c.startswith("total") and "low" not in c and "high" not in c],
        "uncertainty": [c for c in tab.columns if "interval width" in c],
    }
    if show_uncertainty:
        metric = "uncertainty"
    col = short.get(metric, [metric])
    if not col or col[0] not in tab.columns:
        raise KeyError(f"{metric!r} is not available. Options: {list(tab.columns)}")
    col = col[0]

    merged = attach_estimates(gdf, tab[[col]], stand_col=stand_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 6.2))
    merged.plot(column=col, ax=ax, cmap="OrRd" if metric == "uncertainty" else cmap,
                legend=True, edgecolor="white", lw=0.5,
                missing_kwds={"color": PALETTE["sand"], "edgecolor": PALETTE["line"],
                              "hatch": "///", "label": "no estimate"})
    ax.set_axis_off()
    ax.set_title(col if metric != "uncertainty" else "Where the estimate is least certain",
                 fontsize=11.5, color=PALETTE["dark"], weight="semibold", loc="left")
    return ax


def fig_to_uri(fig, dpi=150) -> str:
    """Encode a matplotlib figure as a base64 data URI for embedding in an HTML report."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
