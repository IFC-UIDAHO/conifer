"""CONIFER - Stand Structure Studio.

A no-code front end for foresters: drop in a tree list, get maps, stand tables, honest
prediction intervals, and a printable report. Nothing here reimplements the method; it is
a guided path through conifer.io -> fit -> conifer.calibration -> conifer.report.

    streamlit run app.py
"""
from __future__ import annotations

import io as _io
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import conifer
from conifer import plots as cplots
from conifer import report as crep

st.set_page_config(page_title="CONIFER — Stand Structure Studio",
                   page_icon="🌲", layout="wide", initial_sidebar_state="expanded")

GREEN, DARK, SAND, LINE, MUTED = "#2D6A4F", "#1b3a26", "#f7f5f0", "#dfe6e1", "#5d6d64"

st.markdown(f"""<style>
.block-container{{padding-top:2.1rem;max-width:1380px}}
h1,h2,h3{{color:{DARK}}}
.hero{{background:linear-gradient(135deg,#1b3a26,{GREEN} 62%,#40916c);color:#fff;
 padding:26px 30px;border-radius:13px;margin-bottom:20px}}
.hero h1{{color:#fff;margin:0;font-size:29px;letter-spacing:-.4px}}
.hero p{{margin:7px 0 0;opacity:.9;font-size:15px}}
.step{{background:{SAND};border-left:4px solid {GREEN};padding:12px 17px;border-radius:0 8px 8px 0;
 margin-bottom:14px;font-size:14px}}
div[data-testid="stMetric"]{{background:{SAND};border:1px solid {LINE};border-left:4px solid {GREEN};
 padding:14px 16px;border-radius:8px}}
div[data-testid="stMetricValue"]{{color:{GREEN};font-size:26px}}
.stTabs [data-baseweb="tab"]{{font-size:15px;padding:9px 17px}}
.plain{{font-size:15.5px;line-height:1.65}}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class='hero'><h1>🌲 CONIFER — Stand Structure Studio</h1>
<p>Diameter distributions for every stand you have — including the ones with two plots and a
prayer — and intervals that hold up when someone checks them.</p></div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_table(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    for sep in (",", ";", "\t", "|"):
        try:
            f.seek(0)
            df = pd.read_csv(f, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    f.seek(0)
    return pd.read_csv(f)


def _guess(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for cand in cands:
        for k, orig in low.items():
            if k == cand or k.replace("_", "") == cand.replace("_", ""):
                return orig
    for cand in cands:
        for k, orig in low.items():
            if cand in k:
                return orig
    return None


def _fig(f):
    st.pyplot(f, use_container_width=True)
    plt.close(f)


def _dl(label, data, fname, mime, key=None):
    st.download_button(label, data=data, file_name=fname, mime=mime,
                       use_container_width=True, key=key)


@st.cache_data(show_spinner=False)
def _demo():
    return conifer.demo.make_cruise(n_stands=180, seed=7)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
S = st.session_state
S.setdefault("trees", None)
S.setdefault("aux", None)
S.setdefault("gdf", None)
S.setdefault("fitted", None)


# ---------------------------------------------------------------------------
# sidebar — data in
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Your data")

    if st.button("🎲  Load the demo cruise", use_container_width=True,
                 help="200 stands on a fixed-area cruise, with LiDAR metrics and stand "
                      "polygons — shaped to match a real Idaho inventory, so you can walk the "
                      "whole workflow before putting your own data anywhere near it."):
        trees, aux, stands, truth = _demo()
        S.trees, S.aux, S.truth_df = trees, aux, truth
        try:
            import geopandas as gpd
            from shapely import wkt
            S.gdf = gpd.GeoDataFrame(stands.drop(columns=["geometry_wkt"]),
                                     geometry=stands["geometry_wkt"].apply(wkt.loads),
                                     crs="EPSG:4326")
        except Exception:
            S.gdf = None
        S.fitted = None
        st.success("Demo cruise loaded.")

    st.caption("or upload your own —")
    up_trees = st.file_uploader("Tree list  ·  one row per tallied tree", type=["csv", "xlsx", "xls"])
    up_aux = st.file_uploader("Stand metrics  ·  one row per stand", type=["csv", "xlsx", "xls"])
    up_geo = st.file_uploader("Stand polygons  ·  optional", type=["gpkg", "geojson", "json", "zip"])

    if up_trees is not None:
        S.trees = _read_table(up_trees)
        S.fitted = None
    if up_aux is not None:
        S.aux = _read_table(up_aux)
        S.fitted = None
    if up_geo is not None:
        try:
            import geopandas as gpd
            S.gdf = gpd.read_file(up_geo)
        except Exception as e:
            st.error(f"Could not read that layer: {e}")

trees, aux = S.trees, S.aux

if trees is None:
    st.markdown("<div class='step'><b>Start here.</b> Load the demo cruise from the sidebar to "
                "walk the whole thing on realistic data, or upload your own tree list. Either "
                "way it runs on this machine and nothing leaves it — worth knowing before you "
                "point it at a client's inventory.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### What you bring\nA **tree list** — stand, plot, DBH. That alone is "
                    "enough to run. Stand-level **LiDAR or spectral metrics** are what give the "
                    "model something to borrow from, so without them it cannot do much more "
                    "than your cruise already does. **Stand polygons** turn the results into maps.")
    with c2:
        st.markdown("#### What it does\nBins DBH, works out the right sampling covariance for "
                    "your plot design, fits the small-area model, then calibrates the prediction "
                    "intervals **against your own held-out plots** — no known truth required, "
                    "which is just as well, because no inventory has one.")
    with c3:
        st.markdown("#### What you get\nStand and stock tables, per-acre summaries, QMD and "
                    "basal area, maps of both structure and uncertainty, an Excel workbook and a "
                    "printable stand report. Plus a measured figure for how often the intervals "
                    "actually contained the answer, which is the part worth showing a client.")
    st.stop()


# ---------------------------------------------------------------------------
# sidebar — columns + settings
# ---------------------------------------------------------------------------
tc = list(trees.columns)
with st.sidebar:
    st.header("2 · Which columns?")
    stand_col = st.selectbox("Stand id", tc, index=tc.index(_guess(tc, "stand", "stand_id", "unit")) if _guess(tc, "stand", "stand_id", "unit") else 0)
    dbh_col = st.selectbox("DBH", tc, index=tc.index(_guess(tc, "dbh_in", "dbh", "diameter")) if _guess(tc, "dbh_in", "dbh", "diameter") else 0)
    _p = _guess(tc, "plot", "plot_id", "point")
    plot_col = st.selectbox("Plot id  ·  worth finding", ["(none)"] + tc,
                            index=(tc.index(_p) + 1) if _p else 0,
                            help="Plot identifiers are what let CONIFER split each stand's "
                                 "plots in two and calibrate its intervals against the half it "
                                 "did not fit on. Without them it falls back to a method that is "
                                 "known to under-cover — the intervals will look reassuringly "
                                 "tight and be wrong. If your tree list has a plot or point "
                                 "number anywhere, use it.")
    plot_col = None if plot_col == "(none)" else plot_col

    st.header("3 · Your cruise")
    design = st.radio("Plot design", ["Fixed-area plots", "Variable-radius (prism / BAF)"],
                      help="This decides how CONIFER estimates each stand's sampling "
                           "covariance, and the two designs need genuinely different "
                           "treatment — a prism selects trees in proportion to basal area, so "
                           "every tree carries its own expansion factor. Getting it wrong "
                           "does not raise an error; it quietly gives you the wrong numbers.")
    if design.startswith("Variable"):
        baf = st.number_input("Basal area factor (ft²/ac)", 5.0, 100.0, 20.0, 5.0)
        plot_area, dsg = None, "prism"
    else:
        # 0.2 ac is the demo cruise's plot size, so Load demo -> Run is correct out of the box
        plot_area = st.number_input("Plot size (acres)", 0.005, 2.0, 0.2, 0.005, format="%.3f")
        baf, dsg = None, "fixed"

    brk = st.selectbox("DBH classes", ["Six 4-inch classes (1–25 in)",
                                       "FIA 2-inch classes (1–29 in)",
                                       "FIA 1-inch classes (1–21 in)"])
    breaks = {"Six 4-inch classes (1–25 in)": "default",
              "FIA 2-inch classes (1–29 in)": "fia_2in",
              "FIA 1-inch classes (1–21 in)": "fia_1in"}[brk]
    min_dbh = st.number_input("Ignore trees below (in)", 0.0, 20.0, 0.0, 0.5)

    st.header("4 · Uncertainty")
    conf = st.select_slider("Interval level", [80, 90, 95], value=90)
    alpha = 1 - conf / 100
    scope = st.radio(
        "What should the interval promise?",
        ["Each diameter class (recommended)", "All classes at once"],
        help="Per-class answers the question people actually ask — how many 10-15 inch "
             "stems? — and is several times narrower. The joint set promises that every class "
             "is contained at once, which is a stronger claim and priced accordingly. Both are "
             "honestly calibrated; they simply promise different things, and the output says "
             "which one you are looking at.")
    joint = scope.startswith("All")
    mode = "maxscore"

    acols = list(aux.columns) if aux is not None else []
    aux_stand = group_col = None
    if aux is not None:
        st.header("5 · Stand metrics")
        g = _guess(acols, "stand", "stand_id")
        aux_stand = st.selectbox("Stand id in metrics", acols,
                                 index=acols.index(g) if g else 0)
        gt = _guess(acols, "stand_type", "type", "stratum", "forest_type")
        group_col = st.selectbox("Stratify by  ·  optional", ["(none)"] + acols,
                                 index=(acols.index(gt) + 1) if gt else 0,
                                 help="Calibrates the intervals within each stand type rather "
                                      "than pooling across all of them — worth using if your "
                                      "types really do behave differently.")
        group_col = None if group_col == "(none)" else group_col

    run = st.button("▶  Run the estimate", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# build inventory
# ---------------------------------------------------------------------------
try:
    inv = conifer.from_treelist(
        trees, stand_col=stand_col, dbh_col=dbh_col, plot_col=plot_col,
        breaks=breaks, design=dsg, plot_area=plot_area, baf=baf,
        aux=aux, aux_stand_col=aux_stand, group_col=group_col,
        min_dbh=(min_dbh or None),
    )
except Exception as e:
    st.error(f"**Could not read that data.**\n\n{e}")
    st.stop()

st.subheader("What CONIFER is working with")
c1, c2 = st.columns([1.05, 1])
with c1:
    st.dataframe(inv.describe(), use_container_width=True)
with c2:
    issues = inv.issue_table()
    if len(issues):
        st.markdown("**Data checks**")
        st.dataframe(issues, use_container_width=True, hide_index=True, height=min(300, 60 + 38 * len(issues)))
    else:
        st.success("Nothing looks wrong with the input.")

errs = [i for i in inv.issues if i.level == "error"]
if errs:
    for e in errs:
        st.error(f"**{e.message}**\n\n{e.fix}")
    st.stop()

if run or S.fitted is not None:
    if run or S.fitted is None:
        try:
            with st.spinner("Fitting the small-area model…"):
                est = inv.fit()
            with st.spinner("Calibrating prediction sets (no known truth needed)…"):
                cal = conifer.conformalize_holdout(est, alpha=alpha, joint=joint, mode=mode, reps=4)
            with st.spinner("Checking whether those intervals actually hold up…"):
                try:
                    cov = conifer.coverage_check(est, alpha=alpha, joint=joint, mode=mode, reps=8)
                except Exception:
                    cov = None
            S.fitted = (est, cal, cov, alpha, mode, joint)
        except Exception as e:
            st.error("**The fit failed.**\n\n```\n" + "".join(
                traceback.format_exception_only(type(e), e)) + "```")
            with st.expander("Full detail"):
                st.code(traceback.format_exc())
            st.stop()

    est, cal, cov, alpha, mode, joint = S.fitted
    M = crep._meta(est)
    s = np.asarray(est.s_hat_, float)
    qmd = crep.quadratic_mean_diameter(s, M["midpoints"])
    ba = crep.basal_area(s, M["midpoints"], M["dbh_units"])

    st.divider()
    k = st.columns(5)
    k[0].metric("Stands estimated", f"{s.shape[0]:,}")
    k[1].metric(f"Mean {M['dens']}", f"{s.sum(1).mean():,.0f}")
    k[2].metric("Mean basal area", f"{ba.mean():.0f}",
                help="ft²/ac" if M["dbh_units"] == "in" else "m²/ha")
    k[3].metric(f"Mean QMD ({M['dbh_units']})", f"{qmd.mean():.1f}")
    if cov is not None:
        k[4].metric(f"{int((1-alpha)*100)}% {'joint' if joint else 'per-class'} — measured",
                    f"{cov['empirical']*100:.0f}%",
                    delta="holds up" if cov["meets_nominal"] else "below target",
                    delta_color="normal" if cov["meets_nominal"] else "inverse")

    if cov is not None:
        (st.success if cov["meets_nominal"] else st.warning)(cov["summary"])
        st.caption(cov["note"])

    tabs = st.tabs(["📖  In plain words", "📊  Tables", "📈  Figures",
                    "🗺️  Map", "🔍  Does it hold up?", "⬇️  Take it away"])

    # ---- plain words --------------------------------------------------
    with tabs[0]:
        st.markdown("<div class='plain'>", unsafe_allow_html=True)
        for p in crep.narrative(est, coverage=cov, calibration=cal, alpha=alpha, joint=joint):
            st.markdown(p)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("How CONIFER arrived at this — in four sentences"):
            st.markdown("""
1. For each stand it starts from the **direct estimate** of the diameter-class shares and
   how noisy that estimate is, computed from your plot data and plot design.
2. It also **predicts** those shares from your stand metrics with a flexible, cross-fitted
   machine-learned model, and removes that model's first-order bias.
3. It then **blends** the two, giving more weight to the direct estimate where the stand
   has enough plots and more weight to the model where it does not.
4. Finally it wraps the result in a **conformal prediction set** calibrated on held-out
   plots, so the stated interval is checked against data rather than assumed.
""")

    # ---- tables -------------------------------------------------------
    with tabs[1]:
        st.markdown("##### Estimated distribution across all stands")
        st.dataframe(crep.class_summary(est, alpha=alpha, joint=joint), use_container_width=True)
        st.markdown("##### Summary by stand")
        st.dataframe(crep.summary_table(est, alpha=alpha, joint=joint), use_container_width=True, height=330)
        st.markdown("##### Stand table — trees per acre by DBH class")
        st.dataframe(crep.stand_table(est, alpha=alpha, joint=joint), use_container_width=True, height=330)

    # ---- figures ------------------------------------------------------
    with tabs[2]:
        ids = list(M["ids"])
        pick = st.selectbox("Stand", ids, index=0)
        f, ax = plt.subplots(figsize=(9.5, 4.0))
        cplots.plot_distribution(est, stand=pick, ax=ax, alpha=alpha, joint=joint)
        _fig(f)
        c1, c2 = st.columns(2)
        with c1:
            f, ax = plt.subplots(figsize=(6.2, 5.4))
            cplots.plot_comparison(est, ax=ax, alpha=alpha, joint=joint)
            _fig(f)
            st.caption("Points off the 1:1 line are stands where the model moved the answer. "
                       "They should be the pale ones — thin cruises. If a well-plotted stand "
                       "has drifted a long way, that is worth a second look.")
        with c2:
            try:
                f, ax = plt.subplots(figsize=(6.4, 3.5))
                cplots.plot_borrowing(est, ax=ax)
                _fig(f)
                st.caption("Dark green is what the stand's own plots contributed; pale green "
                           "is what came from stands with similar structure.")
            except Exception:
                pass

    # ---- map ----------------------------------------------------------
    with tabs[3]:
        if S.gdf is None:
            st.info("Upload a stand polygon layer (GeoPackage, GeoJSON, or a zipped shapefile) "
                    "in the sidebar to map these results.")
        else:
            gcols = list(S.gdf.columns)
            gk = _guess(gcols, stand_col, "stand", "stand_id")
            sc = st.selectbox("Stand id in the polygon layer", gcols,
                              index=gcols.index(gk) if gk else 0)
            metric = st.radio("Show", ["QMD", "total", "uncertainty"], horizontal=True,
                              format_func=lambda x: {"QMD": "Quadratic mean diameter",
                                                     "total": "Total stems per acre",
                                                     "uncertainty": "Where the estimate is least certain"}[x])
            try:
                f, ax = plt.subplots(figsize=(9, 7))
                cplots.plot_map(est, S.gdf, stand_col=sc, metric=metric, ax=ax, alpha=alpha, joint=joint)
                _fig(f)
                if metric == "uncertainty":
                    st.caption("Darker stands carry the widest intervals relative to their "
                               "estimate. If you have budget for more plots next season, this "
                               "map is where to spend it.")
            except Exception as e:
                st.error(f"Could not draw the map: {e}")

    # ---- trust --------------------------------------------------------
    with tabs[4]:
        st.markdown("##### Do the intervals hold up?")
        if cov is not None:
            f, ax = plt.subplots(figsize=(8, 2.0))
            cplots.plot_coverage(cov, ax=ax)
            _fig(f)
            st.markdown(f"**{cov['summary']}**")
            st.caption(cov["note"])
        st.markdown("##### How this was calibrated")
        st.info(cal["summary"])
        st.markdown("##### Field-only estimate vs CONIFER, stand by stand")
        st.caption("The direct column is what your cruise supports on its own. Where the two "
                   "part company, check the plot count first — thin stands are where the model "
                   "is supposed to move the answer, and where you should question it if it moved "
                   "the wrong way.")
        try:
            st.dataframe(crep.comparison_table(est, alpha=alpha, joint=joint), use_container_width=True, height=380)
        except Exception as e:
            st.warning(str(e))

    # ---- download -----------------------------------------------------
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Excel workbook")
            st.caption("Summary by stand, full stand table, the distribution, the field-vs-CONIFER "
                       "comparison, and the data checks — one sheet each.")
            buf = _io.BytesIO()
            crep.to_excel(est, buf, alpha=alpha, joint=joint)
            _dl("⬇  conifer_results.xlsx", buf.getvalue(), "conifer_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("##### Stand report")
            st.caption("A self-contained page with the figures and the plain-language reading. "
                       "Open it and print to PDF.")
            figs = {}
            try:
                f, ax = plt.subplots(figsize=(7.6, 3.9))
                cplots.plot_distribution(est, stand=0, ax=ax, alpha=alpha, joint=joint)
                figs["Diameter distribution of the first stand, with its prediction set."] = cplots.fig_to_uri(f)
                plt.close(f)
                f, ax = plt.subplots(figsize=(6.4, 5.4))
                cplots.plot_comparison(est, ax=ax, alpha=alpha, joint=joint)
                figs["Field-only estimate against CONIFER's, coloured by plot count."] = cplots.fig_to_uri(f)
                plt.close(f)
                if cov is not None:
                    f, ax = plt.subplots(figsize=(7.6, 2.0))
                    cplots.plot_coverage(cov, ax=ax)
                    figs["Measured coverage of the stated interval."] = cplots.fig_to_uri(f)
                    plt.close(f)
            except Exception:
                pass
            import tempfile, os
            tmp = os.path.join(tempfile.mkdtemp(), "report.html")
            crep.to_html(est, tmp, title="Forest Structure Report",
                         subtitle=f"{s.shape[0]} stands · CONIFER small-area estimate",
                         alpha=alpha, joint=joint, coverage=cov, calibration=cal, figures=figs)
            _dl("⬇  stand_report.html", open(tmp, "rb").read(), "stand_report.html", "text/html")

        with c2:
            st.markdown("##### Plain CSV")
            _dl("⬇  summary_by_stand.csv",
                crep.summary_table(est, alpha=alpha, joint=joint).to_csv().encode(),
                "summary_by_stand.csv", "text/csv", key="c1")
            _dl("⬇  stand_table.csv",
                crep.stand_table(est, alpha=alpha, joint=joint).to_csv().encode(),
                "stand_table.csv", "text/csv", key="c2")

            st.markdown("##### Back into your GIS")
            if S.gdf is None:
                st.caption("Upload a polygon layer to get a GeoPackage with the estimates "
                           "joined on, ready to open in QGIS or ArcGIS Pro.")
            else:
                gcols = list(S.gdf.columns)
                gk = _guess(gcols, stand_col, "stand", "stand_id")
                sc2 = st.selectbox("Stand id in the polygon layer", gcols,
                                   index=gcols.index(gk) if gk else 0, key="gexp")
                try:
                    import tempfile, os
                    merged = conifer.attach_estimates(
                        S.gdf, crep.summary_table(est, alpha=alpha, joint=joint), stand_col=sc2)
                    gp = os.path.join(tempfile.mkdtemp(), "conifer_stands.gpkg")
                    merged.to_file(gp, driver="GPKG", layer="conifer")
                    _dl("⬇  conifer_stands.gpkg", open(gp, "rb").read(),
                        "conifer_stands.gpkg", "application/geopackage+sqlite3", key="g1")
                except Exception as e:
                    st.warning(f"Could not build the GeoPackage: {e}")
else:
    st.info("Everything above checks out. Press **Run the estimate** in the sidebar.")

st.divider()
st.caption("CONIFER estimates are model-based: they borrow strength across stands and are not "
           "a substitute for a design-based estimate where the sample is adequate. Prediction "
           "sets state where a stand's value is expected to lie, not where the estimate's mean lies.")
