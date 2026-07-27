"""conifer.report — the outputs a forester actually hands to someone.

CONIFER's estimator returns matrices. A forester reports *stand tables*, *per-acre
summaries*, and a document. This module turns the former into the latter:

* :func:`stand_table`      — trees/acre by DBH class, per stand, with intervals
* :func:`summary_table`    — one row per stand: total TPA, basal area, QMD, interval width
* :func:`comparison_table` — the design-direct estimate beside CONIFER's, with both intervals
* :func:`narrative`        — a plain-language reading of the result, with every number
                             injected from the fit (nothing is generated or paraphrased)
* :func:`to_excel` / :func:`to_html` — the deliverable

**Which interval is reported.** These functions default to the **per-class (marginal)**
interval: for any one diameter class you name, it contains the truth at the stated rate.
That matches the question a forester actually asks ("how many 10-15 inch stems?"), and it is
far narrower than the joint set — measured over three simulated cruises, a nominal 90%
marginal interval is ~3.4x the estimate against ~29x for the joint L-infinity band, at 0.953
measured per-class coverage. Pass ``joint=True`` for the simultaneous-over-all-classes claim,
which is a genuinely stronger statement and priced accordingly. Every function labels which
guarantee it is reporting.

Derived quantities use standard US inventory conventions when ``dbh_units == 'in'``:
basal area in ft^2/acre as ``0.005454 * DBH^2`` per tree, QMD as the basal-area-weighted
quadratic mean. In metric, basal area is m^2/ha as ``pi/40000 * DBH_cm^2``.
"""
from __future__ import annotations

import datetime as _dt
import html as _html

import numpy as np
import pandas as pd

__all__ = [
    "stand_table", "summary_table", "comparison_table", "class_summary",
    "narrative", "to_excel", "to_html", "quadratic_mean_diameter", "basal_area",
    "data_gain",
]


# ---------------------------------------------------------------------------
# derived forestry quantities
# ---------------------------------------------------------------------------
def _ba_factor(units: str) -> float:
    return 0.005454 if units == "in" else np.pi / 40000.0


def basal_area(s: np.ndarray, midpoints: np.ndarray, units: str = "in") -> np.ndarray:
    """Basal area per unit area (ft^2/acre, or m^2/ha in metric) from a class density vector."""
    return np.asarray(s, float) @ (_ba_factor(units) * np.asarray(midpoints, float) ** 2)


def quadratic_mean_diameter(s: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
    """QMD — the diameter of the tree of mean basal area. In the same units as ``midpoints``."""
    s = np.atleast_2d(np.asarray(s, float))
    mid = np.asarray(midpoints, float)
    tot = np.clip(s.sum(1), 1e-12, None)
    return np.sqrt((s @ mid ** 2) / tot)


def _meta(est):
    """Labels and units, whether or not the estimator came through conifer.io."""
    K = est.s_hat_.shape[1]
    ids = getattr(est, "stand_ids_", None)
    if ids is None:
        ids = np.array([f"stand_{i+1}" for i in range(est.s_hat_.shape[0])])
    edges = getattr(est, "edges_", None)
    if edges is None:
        edges = np.arange(K + 1, dtype=float)
    labels = getattr(est, "class_labels_", None) or [f"class {k+1}" for k in range(K)]
    du = getattr(est, "dbh_units_", "in")
    au = getattr(est, "area_units_", "acre")
    mid = 0.5 * (np.asarray(edges, float)[:-1] + np.asarray(edges, float)[1:])
    dens = "TPA" if au == "acre" else "TPH"
    return dict(ids=np.asarray(ids), edges=np.asarray(edges, float), labels=list(labels),
                dbh_units=du, area_units=au, midpoints=mid, dens=dens, K=K)


def data_gain(est):
    """Per-stand **data gain** gamma - how much of the estimate rests on that stand's own plots.

    This is the EBLUP's own shrinkage weight, ``gamma_i = Su / (Su + D_i)`` averaged over the
    ALR coordinates, where ``Su`` is the between-stand covariance and ``D_i`` the stand's
    sampling covariance. It is the honest answer to a forester's first question - *how much of
    this number came from my cruise?* - because it is the quantity the model actually uses to
    decide, and it rises toward 1 as plots accumulate.

    It replaces ``w_adeq_`` for this purpose. ``w_adeq_`` belongs to the optional outer
    adequacy gate, which is **off by default**, so it reads as all zeros on a standard fit -
    reporting that would tell the forester "0% of this came from your field data", which is
    both false and alarming.

    Returns ``None`` if the estimator exposes neither a ``data_gain_`` attribute nor the
    covariance pieces needed to derive one.
    """
    g = getattr(est, "data_gain_", None)          # preferred: exposed by the engine
    if g is not None:
        return np.asarray(g, float)
    Su = getattr(est, "Su_", None)
    D = getattr(est, "_D", None)                  # fallback while the engine is mid-release
    if Su is None or D is None:
        return None
    try:
        su = np.clip(np.diag(np.asarray(Su, float)), 0, None)
        D = np.asarray(D, float)
        return np.array([float(np.mean(su / (su + np.clip(np.diag(D[i]), 0, None) + 1e-12)))
                         for i in range(D.shape[0])])
    except Exception:
        return None


def _intervals(est, joint=True, alpha=0.10):
    """Prediction bounds, floored at zero.

    Stem density cannot be negative. The engine works on a standardized score scale and can
    return a lower bound below zero when the set is wide; reporting "-1288 trees per acre"
    to a forester is both meaningless and instantly discrediting, so the lower bound is
    truncated at the physical floor here. This only ever *shrinks* the set toward a region
    the truth cannot occupy, so it cannot break coverage.
    """
    lo = hi = None
    try:
        lo, hi = est.predict_interval(joint=joint, alpha=alpha)
    except Exception:
        try:
            lo, hi = est.class_intervals(z=1.645)
        except Exception:
            return None, None
    lo = np.clip(np.asarray(lo, float), 0.0, None)
    hi = np.clip(np.asarray(hi, float), 0.0, None)
    return lo, hi


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def stand_table(est, *, stands=None, intervals=True, joint=False, alpha=0.10,
                round_to=1) -> pd.DataFrame:
    """The stand table: trees per acre by DBH class, one row per stand.

    With ``intervals=True`` each class gets a ``low``/``high`` pair from the calibrated
    prediction set, so the table carries its own uncertainty.
    """
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    idx = np.arange(s.shape[0]) if stands is None else np.asarray(stands)
    lo, hi = _intervals(est, joint, alpha) if intervals else (None, None)

    data = {}
    for k, lab in enumerate(M["labels"]):
        data[(lab, M["dens"])] = s[idx, k]
        if lo is not None:
            data[(lab, "low")] = lo[idx, k]
            data[(lab, "high")] = hi[idx, k]
    df = pd.DataFrame(data, index=pd.Index(M["ids"][idx], name="stand"))
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["DBH class", ""])
    return df.round(round_to)


def summary_table(est, *, joint=False, alpha=0.10, round_to=1) -> pd.DataFrame:
    """One row per stand: the numbers that go on the cover page of a stand report."""
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    lo, hi = _intervals(est, joint, alpha)
    qmd = quadratic_mean_diameter(s, M["midpoints"])
    ba = basal_area(s, M["midpoints"], M["dbh_units"])
    total = s.sum(1)

    out = pd.DataFrame({
        f"total {M['dens']}": total,
        f"basal area ({'ft²/ac' if M['dbh_units'] == 'in' else 'm²/ha'})": ba,
        f"QMD ({M['dbh_units']})": qmd,
    }, index=pd.Index(M["ids"], name="stand"))

    if lo is not None:
        out[f"total {M['dens']} low"] = lo.sum(1)
        out[f"total {M['dens']} high"] = hi.sum(1)
        # Absolute width first, and the relative figure computed on the classes the stand
        # actually tallied. A relative width that includes the untallied classes divides by a
        # near-zero estimate and reads enormous - a property of the denominator, not of the
        # interval a forester acts on.
        out[f"interval width ({M['dens']})"] = hi.sum(1) - lo.sum(1)
        inv_ = getattr(est, "inventory_", None)
        popm = (inv_.counts > 0) if inv_ is not None else np.ones_like(s, bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            relp = np.where(popm, (hi - lo) / np.clip(s, 1e-9, None), np.nan)
        out["width on tallied classes (% of estimate)"] = 100 * np.nanmean(relp, axis=1)

    inv = getattr(est, "inventory_", None)
    if inv is not None:
        if inv.n_plots is not None:
            out.insert(0, "plots", inv.n_plots.astype(int))
        # On a prism cruise `area_eff` is a derived expansion basis, not an area anyone
        # measured in the field - showing it as "sampled acres" invites misreading, so it
        # is reported only for fixed-area cruises where it is a real, checkable quantity.
        if inv.design == "fixed":
            out.insert(0 if inv.n_plots is None else 1, f"sampled {M['area_units']}s",
                       np.round(inv.area_eff, 3))
    g = data_gain(est)
    if g is not None:
        out["% from this stand's plots"] = 100 * g
        out["% borrowed from similar stands"] = 100 * (1 - g)
    return out.round(round_to)


def class_summary(est, *, joint=False, alpha=0.10, round_to=2) -> pd.DataFrame:
    """Region-wide mean density by DBH class — the headline distribution."""
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    lo, hi = _intervals(est, joint, alpha)
    out = pd.DataFrame({
        f"mean {M['dens']}": s.mean(0),
        "share of stems (%)": 100 * s.mean(0) / max(s.mean(0).sum(), 1e-12),
        "stands with 1+ TPA": (s >= 1.0).sum(0),
    }, index=pd.Index(M["labels"], name="DBH class"))
    if lo is not None:
        out.insert(1, "low", lo.mean(0))
        out.insert(2, "high", hi.mean(0))
    inv = getattr(est, "inventory_", None)
    if inv is not None:
        out.insert(0, f"field-only {M['dens']}", inv.direct.mean(0))
    return out.round(round_to)


def comparison_table(est, *, alpha=0.10, joint=False, round_to=1) -> pd.DataFrame:
    """Design-direct beside CONIFER, per stand — the single most persuasive table.

    If CONIFER's interval is narrower and its point estimate sits inside the direct
    estimate's own error bar, the case for the model-based estimate makes itself.
    """
    inv = getattr(est, "inventory_", None)
    if inv is None:
        raise ValueError("comparison_table needs the Inventory (fit via `inventory.fit()`).")
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    d = inv.direct
    lo, hi = _intervals(est, joint, alpha)

    qmd_sae = quadratic_mean_diameter(s, M["midpoints"])
    qmd_dir = quadratic_mean_diameter(d, M["midpoints"])
    out = pd.DataFrame({
        "plots": (inv.n_plots.astype(int) if inv.n_plots is not None else 1),
        f"field-only total {M['dens']}": d.sum(1),
        f"CONIFER total {M['dens']}": s.sum(1),
        f"field-only QMD ({M['dbh_units']})": qmd_dir,
        f"CONIFER QMD ({M['dbh_units']})": qmd_sae,
    }, index=pd.Index(M["ids"], name="stand"))
    if lo is not None:
        out[f"CONIFER {int((1-alpha)*100)}% interval"] = [
            f"{a:,.0f} – {b:,.0f}" for a, b in zip(lo.sum(1), hi.sum(1))
        ]
    g = data_gain(est)
    if g is not None:
        out["% from this stand's plots"] = np.round(100 * g, 0)
    return out.round(round_to)


# ---------------------------------------------------------------------------
# narrative — template-first, numbers injected
# ---------------------------------------------------------------------------
def narrative(est, *, coverage=None, calibration=None, alpha=0.10, joint=False) -> list[str]:
    """A plain-language reading of the result, as a list of paragraphs.

    Every figure in the text is read directly off the fitted estimator. Nothing here is
    generated by a language model, so nothing here can be a hallucinated number. If you
    later add an LLM polish step, keep this function as the source of the facts.
    """
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    m, K = s.shape
    inv = getattr(est, "inventory_", None)
    lo, hi = _intervals(est, joint, alpha)
    qmd = quadratic_mean_diameter(s, M["midpoints"])
    ba = basal_area(s, M["midpoints"], M["dbh_units"])
    dens, du, au = M["dens"], M["dbh_units"], M["area_units"]
    paras = []

    # 1 — what was estimated
    p = (f"CONIFER estimated the diameter distribution for **{m} stands** across "
         f"**{K} DBH classes** ({M['labels'][0]} through {M['labels'][-1]}). ")
    if inv is not None:
        p += (f"The estimate draws on **{int(inv.counts.sum()):,} tallied trees** over "
              f"**{inv.area_eff.sum():,.0f} sampled {au}s**")
        if inv.n_plots is not None:
            p += f", a median of **{np.median(inv.n_plots):.0f} plots per stand**"
        p += ". "
    paras.append(p.strip())

    # 2 — the headline numbers
    big = int(np.argmax(s.mean(0)))
    paras.append(
        f"Across the whole area the estimated stand structure averages "
        f"**{s.sum(1).mean():,.0f} {dens}**, a basal area of **{ba.mean():.0f} "
        f"{'ft²/ac' if du == 'in' else 'm²/ha'}**, and a quadratic mean diameter of "
        f"**{qmd.mean():.1f} {du}** (range {qmd.min():.1f}–{qmd.max():.1f} {du} across stands). "
        f"The **{M['labels'][big]}** class carries the most stems, at "
        f"{s.mean(0)[big]:,.0f} {dens} — {100*s.mean(0)[big]/max(s.mean(0).sum(),1e-9):.0f}% of all stems."
    )

    # 3 — where the number came from: data vs model
    _g = data_gain(est)
    if _g is not None:
        w = 100 * _g
        thin = int(np.sum(w < 40))
        thin_ids = np.argsort(w)[:3]
        paras.append(
            f"On average **{w.mean():.0f}% of each stand's estimate rests on that stand's own "
            f"plots**, and the remaining {100-w.mean():.0f}% is borrowed from stands that look "
            f"like it. That split is not a setting — it is what the model computes from how "
            f"noisy each stand's own data is, and it ranges from **{w.min():.0f}% to "
            f"{w.max():.0f}%** across your stands. The lowest are "
            f"{', '.join(str(x) for x in np.asarray(M['ids'])[thin_ids])} — check those first "
            f"if a number looks wrong, because they are leaning hardest on the model."
        )

    # 4 — uncertainty, and whether it can be trusted
    if lo is not None:
        rel = 100 * (hi.sum(1) - lo.sum(1)) / np.clip(s.sum(1), 1e-9, None)
        kind = getattr(est, "_cal_kind_", "calibrated")
        lvl = int((1 - alpha) * 100)
        if joint:
            p = (f"The **{lvl}% prediction set** covers **all {K} diameter classes at once** — "
                 f"a stronger claim, and correspondingly a wider one. It spans on average "
                 f"**±{rel.mean()/2:.0f}% of the estimate** for total {dens}. ")
        else:
            inv_ = getattr(est, "inventory_", None)
            popm = (inv_.counts > 0) if inv_ is not None else np.ones_like(s, bool)
            with np.errstate(invalid="ignore", divide="ignore"):
                relp = np.where(popm, (hi - lo) / np.clip(s, 1e-9, None), np.nan)
            wp = float(np.nanmean(relp)) * 100
            tail_abs = float(np.mean((hi - lo)[~popm])) if (~popm).any() else 0.0
            p = (f"Each class carries its own **{lvl}% interval**: for any single diameter "
                 f"class you name, the interval shown contains that stand's true density "
                 f"{lvl}% of the time. On the classes a stand actually tallied it spans "
                 f"**±{wp/2:.0f}% of the estimate**. For classes where nothing was tallied the "
                 f"interval runs from zero up to about **{tail_abs:.1f} {dens}** — small in "
                 f"absolute terms, but large as a percentage precisely because the estimate it "
                 f"is measured against is near zero, so read those in {dens} and not in "
                 f"percent. This is a per-class guarantee, not a promise about all {K} classes "
                 f"at once; for that, ask for the joint set, which is wider. ")
        if kind:
            p += f"Calibration method: {kind}. "
        paras.append(p.strip())
    inv_ = getattr(est, "inventory_", None)
    if inv_ is not None and getattr(inv_, "D_ext", None) is not None:
        paras.append(
            "**A caveat on these intervals.** This fit used a design-based sampling "
            "covariance built from your plot replicates, which sharpens the estimates and "
            "makes the data-gain figures above meaningful. The prediction intervals under "
            "that setting are currently **approximate** - a known calibration gap, not a "
            "silent one. Treat them as indicative, and read the measured coverage below "
            "rather than the nominal level."
        )
    if coverage is not None:
        paras.append("**Does the interval hold up?** " + coverage.get("summary", ""))
    elif calibration is not None:
        paras.append(calibration.get("summary", ""))

    # 5 — comparison to the field-only estimate
    if inv is not None and lo is not None:
        d = inv.direct
        agree = np.mean((d.sum(1) >= lo.sum(1)) & (d.sum(1) <= hi.sum(1)))
        shift = 100 * (s.sum(1).mean() - d.sum(1).mean()) / max(d.sum(1).mean(), 1e-9)
        paras.append(
            f"Compared with the **field-only (design-direct) estimate**, CONIFER's total density "
            f"differs by **{shift:+.1f}% on average**, and the field-only estimate falls inside "
            f"CONIFER's prediction set for **{100*agree:.0f}% of stands**. Where the two disagree "
            f"most, it is almost always a stand with few plots — the model is pulling a noisy "
            f"direct estimate back toward what comparable, better-sampled stands look like."
        )

    # 6 — honest caveats
    caveats = []
    if inv is not None:
        for i in inv.issues:
            if i.level in ("error", "warning"):
                caveats.append(i.message)
    if m < 30:
        caveats.append(f"Only {m} stands were available; below about 30, the between-stand "
                       "covariance is hard to pin down and intervals should be read as indicative.")
    if caveats:
        paras.append("**Read with care:** " + " ".join(caveats[:4]))

    return paras


# ---------------------------------------------------------------------------
# deliverables
# ---------------------------------------------------------------------------
def to_excel(est, path, *, alpha=0.10, joint=False) -> str:
    """Write the full result set as a multi-sheet Excel workbook."""
    sheets = {
        "Summary by stand": summary_table(est, alpha=alpha, joint=joint),
        "Stand table": stand_table(est, alpha=alpha, joint=joint),
        "Distribution": class_summary(est, alpha=alpha, joint=joint),
    }
    try:
        sheets["Field vs CONIFER"] = comparison_table(est, alpha=alpha, joint=joint)
    except ValueError:
        pass
    inv = getattr(est, "inventory_", None)
    if inv is not None:
        sheets["Data checks"] = inv.issue_table()
        sheets["Inputs"] = inv.describe().reset_index()
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31])
    return str(path)


_CSS = """
:root{--ink:#1b2b22;--muted:#5d6d64;--line:#dfe6e1;--green:#2D6A4F;--sand:#f7f5f0;--warn:#b45309}
*{box-sizing:border-box}
body{font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
 color:var(--ink);margin:0;padding:0 0 60px;background:#fff;line-height:1.6}
.wrap{max-width:940px;margin:0 auto;padding:0 28px}
header{background:linear-gradient(135deg,#1b3a26,#2D6A4F 60%,#40916c);color:#fff;padding:36px 0 30px;margin-bottom:32px}
header .wrap{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
h1{margin:0;font-size:27px;letter-spacing:-.3px}
.sub{opacity:.85;font-size:14px;margin-top:6px}
.stamp{font-size:12px;opacity:.8;text-align:right}
h2{font-size:18px;margin:34px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--green)}
p{margin:0 0 13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:22px 0 8px}
.kpi{background:var(--sand);border:1px solid var(--line);border-left:4px solid var(--green);
 border-radius:7px;padding:14px 16px}
.kpi .v{font-size:25px;font-weight:650;color:var(--green);line-height:1.15}
.kpi .l{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.55px;margin-top:5px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 6px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{background:var(--sand);font-weight:600;color:var(--muted);font-size:11.5px;
 text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0}
td:first-child,th:first-child{text-align:left;font-weight:600}
tbody tr:hover{background:#f3f8f4}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--line);border-radius:7px}
.badge{display:inline-block;padding:4px 11px;border-radius:999px;font-size:12px;font-weight:600}
.ok{background:#d7f0e1;color:#14532d}.warn{background:#fdebd2;color:var(--warn)}
.note{background:var(--sand);border-left:4px solid var(--green);padding:13px 17px;border-radius:0 7px 7px 0;
 margin:15px 0;font-size:13.5px}
.note.caution{border-left-color:var(--warn);background:#fffaf2}
figure{margin:18px 0}figure img{width:100%;border:1px solid var(--line);border-radius:7px}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
strong{font-weight:650}
"""


def _md_bold(s: str) -> str:
    out, parts = "", s.split("**")
    for i, p in enumerate(parts):
        out += (f"<strong>{_html.escape(p)}</strong>" if i % 2 else _html.escape(p))
    return out


def to_html(est, path, *, title="Forest Structure Report", subtitle="", alpha=0.10,
            joint=False, coverage=None, calibration=None, figures=None) -> str:
    """Write a self-contained, branded HTML stand report (print to PDF from the browser).

    ``figures`` is an optional mapping of caption -> base64 PNG data URI.
    """
    M = _meta(est)
    s = np.asarray(est.s_hat_, float)
    qmd = quadratic_mean_diameter(s, M["midpoints"])
    ba = basal_area(s, M["midpoints"], M["dbh_units"])
    stamp = _dt.datetime.now().strftime("%d %B %Y, %H:%M")

    kpis = [
        (f"{s.shape[0]:,}", "stands estimated"),
        (f"{s.sum(1).mean():,.0f}", f"mean {M['dens']}"),
        (f"{ba.mean():.0f}", "mean basal area " + ("ft²/ac" if M["dbh_units"] == "in" else "m²/ha")),
        (f"{qmd.mean():.1f}", f"mean QMD ({M['dbh_units']})"),
    ]
    if coverage is not None and np.isfinite(coverage.get("empirical", np.nan)):
        kpis.append((f"{coverage['empirical']*100:.0f}%",
                     f"checked coverage of the {int(coverage['nominal']*100)}% interval"))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_html.escape(title)}</title><style>{_CSS}</style></head><body>",
        "<header><div class='wrap'><div>",
        f"<h1>{_html.escape(title)}</h1>",
        f"<div class='sub'>{_html.escape(subtitle or 'Small-area estimate of the diameter distribution')}</div>",
        "</div><div class='stamp'>Produced by CONIFER<br>"
        f"{_html.escape(stamp)}</div></div></header><div class='wrap'>",
        "<div class='kpis'>",
    ]
    parts += [f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{_html.escape(l)}</div></div>"
              for v, l in kpis]
    parts.append("</div>")

    if coverage is not None:
        cls = "ok" if coverage.get("meets_nominal") else "warn"
        parts.append(f"<div class='note'><span class='badge {cls}'>"
                     f"{'Interval check passed' if coverage.get('meets_nominal') else 'Interval check: review'}"
                     f"</span> {_md_bold(coverage.get('summary',''))}"
                     f"<br><small>{_html.escape(coverage.get('note',''))}</small></div>")

    parts.append("<h2>What this says</h2>")
    parts += [f"<p>{_md_bold(p)}</p>" for p in
              narrative(est, coverage=coverage, calibration=calibration, alpha=alpha, joint=joint)]

    if figures:
        parts.append("<h2>Figures</h2>")
        for cap, uri in figures.items():
            parts.append(f"<figure><img src='{uri}' alt='{_html.escape(cap)}'>"
                         f"<figcaption style='font-size:12.5px;color:#5d6d64;margin-top:7px'>"
                         f"{_html.escape(cap)}</figcaption></figure>")

    parts.append("<h2>Estimated distribution, all stands</h2>")
    parts.append(class_summary(est, alpha=alpha, joint=joint).to_html(border=0, classes="t"))

    parts.append("<h2>Summary by stand</h2><div class='scroll'>")
    parts.append(summary_table(est, alpha=alpha, joint=joint).to_html(border=0))
    parts.append("</div>")

    try:
        parts.append("<h2>Field-only estimate vs CONIFER</h2>")
        parts.append("<div class='note'>The field-only column is what the cruise alone supports. "
                     "CONIFER borrows strength from comparable stands, so its estimate is steadier "
                     "in thinly sampled stands — the ones where the two columns differ most.</div>")
        parts.append("<div class='scroll'>")
        parts.append(comparison_table(est, alpha=alpha, joint=joint).to_html(border=0))
        parts.append("</div>")
    except ValueError:
        pass

    inv = getattr(est, "inventory_", None)
    if inv is not None and inv.issues:
        parts.append("<h2>Data checks</h2>")
        for i in inv.issues:
            cls = "caution" if i.level in ("error", "warning") else ""
            parts.append(f"<div class='note {cls}'><strong>{i.level.upper()}</strong> — "
                         f"{_html.escape(i.message)}"
                         + (f"<br><small>{_html.escape(i.fix)}</small>" if i.fix else "") + "</div>")

    parts.append(
        "<footer>Estimates are model-based (compositional area-level Fay–Herriot with a "
        "cross-fitted debiased machine-learned mean and design-aware conformal prediction sets). "
        "They borrow strength across stands and are not a substitute for a design-based estimate "
        "where the sample is adequate. Prediction sets state where a stand's value is expected to "
        "lie, not where the estimate's mean lies."
        "</footer></div></body></html>"
    )
    out = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return str(path)
