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

st.set_page_config(
    page_title="CONIFER — Stand Structure Studio",
    page_icon="🌲", layout="wide", initial_sidebar_state="expanded",
    menu_items={"about": "CONIFER — COmpositional Nonlinear-debiased Inference, Fay–Herriot with "
                         "Ellipsoidal conformal Regions. Robust · Compositional · Confident."},
)

ACCENT = "#2D6A4F"

# In-app figures sit on the same white surface as the cards around them.
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E3E3E3", "text.color": "#171717",
    "axes.labelcolor": "#4D4D4D", "xtick.color": "#8F8F8F", "ytick.color": "#8F8F8F",
    "font.size": 10.5,
})

# ---------------------------------------------------------------------------
# design system — neutral-light SaaS: near-white canvas, one restrained accent,
# borders as 1px shadow rings, weights capped at 600, mono for figures.
# ---------------------------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root{
  --bg:#FAFAF9; --surface:#FFFFFF; --surface-2:#FBFBFA;
  --ink:#18181B; --ink-2:#52525B; --muted:#6B6B73;
  --hairline:#ECECEA; --ring:0 0 0 1px rgba(24,24,27,.07);
  --accent:#2D6A4F; --accent-600:#245A42; --accent-tint:#EAF3EC;
  --r:8px; --r-lg:12px;
}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)}
[data-testid="stHeader"]{background:transparent!important;height:0}
[data-testid="stToolbar"],#MainMenu,footer{display:none!important}
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1300px}
html,body,.stApp,[data-testid="stSidebar"],[data-testid="stAppViewContainer"]{
  font-family:'Inter',system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
h1,h2,h3,h4,h5{font-family:'Inter',system-ui,sans-serif!important;color:var(--ink)!important;font-weight:600;letter-spacing:-.3px}
h2{font-size:20px}h3{font-size:17px}h4,h5{font-size:15px}
.stApp p,.stMarkdown,.stApp li{color:var(--ink-2);font-size:14.5px;line-height:1.6}
.stApp a{color:var(--accent);text-decoration:none}
.stApp a:hover{text-decoration:underline}
::selection{background:var(--accent-tint)}
hr{border:none;border-top:1px solid var(--hairline);margin:1.2rem 0}

/* config rail (sidebar) */
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--hairline)}
[data-testid="stSidebar"] .block-container{padding-top:1.1rem}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:var(--ink-2)}
.stCaption,[data-testid="stCaptionContainer"],small{color:var(--muted)!important}

/* inputs */
label,[data-testid="stWidgetLabel"] p{color:var(--ink-2)!important;font-weight:500;font-size:13px}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],
[data-testid="stNumberInput"] input,[data-baseweb="select"] input{
  background:var(--surface)!important;border-color:var(--hairline)!important;color:var(--ink)!important;border-radius:8px!important}
[data-baseweb="select"]>div:hover{border-color:#D4D4D0!important}
input,textarea,[role="combobox"],[data-baseweb="select"] *{color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--surface-2)!important;border:1px dashed #DBDBD6!important;border-radius:10px}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted)!important}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{color:var(--ink-2)!important}
[data-baseweb="slider"] [role="slider"]{background:var(--accent)!important;border-color:var(--accent)!important}
[data-baseweb="slider"] [data-testid="stTickBar"]{background:transparent}

/* buttons — one filled primary, everything else quiet */
.stButton>button,.stDownloadButton>button{
  border-radius:8px;border:1px solid var(--hairline);background:var(--surface);color:var(--ink);
  font-family:'Inter';font-weight:500;font-size:13.5px;padding:8px 14px;transition:all .14s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#D4D4D0;background:var(--surface-2)}
.stButton>button[kind="primary"]{background:var(--accent);border:1px solid var(--accent);color:#fff!important;
  font-weight:600;box-shadow:0 1px 2px rgba(45,106,79,.28)}
.stButton>button[kind="primary"]:hover{background:var(--accent-600);border-color:var(--accent-600)}
.stButton>button p,.stButton>button div,.stDownloadButton>button p,.stDownloadButton>button div{color:inherit!important}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{outline:none;box-shadow:0 0 0 2px var(--bg),0 0 0 4px rgba(45,106,79,.4)}

/* tabs — quiet underline */
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--hairline)}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none;padding:10px 2px;margin-right:22px;
  color:var(--muted);font-family:'Inter';font-weight:500;font-size:13.5px}
.stTabs [data-baseweb="tab"]:hover{color:var(--ink)}
.stTabs [aria-selected="true"]{color:var(--ink)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent)!important;height:2px}
.stTabs [data-baseweb="tab-border"]{display:none}

/* data + surfaces */
[data-testid="stDataFrame"]{border:1px solid var(--hairline);border-radius:10px;overflow:hidden}
[data-testid="stAlert"]{border-radius:10px;border:1px solid var(--hairline);box-shadow:none}
[data-testid="stExpander"]{border:1px solid var(--hairline);border-radius:10px;background:var(--surface)}
[data-testid="stExpander"] summary{color:var(--ink-2);font-weight:500}

/* top bar (app shell) */
.cf-top{display:flex;align-items:center;gap:12px;padding:2px 2px 14px;margin-bottom:4px;border-bottom:1px solid var(--hairline)}
.cf-top .mk{width:30px;height:30px;display:grid;place-items:center;background:transparent;flex:0 0 auto}
.cf-top .mk svg{width:30px;height:30px}
.cf-wm{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:17px;color:var(--ink);letter-spacing:-.5px}
.cf-sub{color:var(--muted);font-size:12.5px;border-left:1px solid var(--hairline);padding-left:12px}
.cf-sp{margin-left:auto}
.cf-pill{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-2);background:var(--surface-2);
  border:1px solid var(--hairline);border-radius:999px;padding:4px 11px;white-space:nowrap}
.cf-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);display:inline-block;margin-right:7px;vertical-align:middle}

/* section label in the rail */
.cf-sec{font-size:11px;font-weight:600;letter-spacing:.7px;text-transform:uppercase;color:var(--muted);
  margin:22px 0 8px;display:flex;align-items:center;gap:8px}
.cf-sec::before{content:"";width:12px;height:2px;background:var(--accent);border-radius:2px}

/* results section heading */
.cf-h{display:flex;align-items:baseline;gap:10px;margin:24px 0 12px}
.cf-h .t{font-size:15px;font-weight:600;color:var(--ink)}
.cf-h .d{font-size:13px;color:var(--muted)}

/* KPI cards */
.cf-kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:2px 0 6px}
.cf-kpi{background:var(--surface);box-shadow:var(--ring);border-radius:12px;padding:15px 16px}
.cf-kpi .l{font-size:11px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--muted)}
.cf-kpi .v{font-family:'JetBrains Mono',monospace;font-size:25px;font-weight:700;color:var(--ink);
  letter-spacing:-1px;line-height:1.15;margin-top:7px}
.cf-kpi .s{font-size:12px;color:var(--muted);margin-top:4px}
.cf-kpi .s.good{color:var(--accent)}
.cf-kpi .s.bad{color:#B4472A}
.cf-kpi.hl{box-shadow:0 0 0 1px rgba(45,106,79,.35);background:var(--accent-tint)}
[data-testid="stMetric"]{background:var(--surface);box-shadow:var(--ring);border-radius:12px;padding:14px 16px}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:11px!important;font-weight:600;letter-spacing:.4px;text-transform:uppercase}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--ink)!important;font-size:24px;letter-spacing:-1px}
[data-testid="stMetricDelta"]{font-size:12px;font-weight:600}

/* empty-state onboarding */
.cf-empty{text-align:center;max-width:600px;margin:40px auto 14px}
.cf-empty h1{font-size:29px;font-weight:600;letter-spacing:-.6px;margin:0 0 10px;color:var(--ink)}
.cf-empty p{color:var(--ink-2);font-size:15px;line-height:1.6;margin:0 auto;max-width:500px}
.cf-feat{background:var(--surface);box-shadow:var(--ring);border-radius:12px;padding:18px;height:100%}
.cf-feat .k{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent);font-weight:600;letter-spacing:.5px}
.cf-feat h4{margin:8px 0 6px;font-size:14.5px}
.cf-feat p{color:var(--muted);font-size:13px;line-height:1.55;margin:0}
.plain p{font-size:15px;line-height:1.7;color:var(--ink-2)}
.cf-kicker{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
.cf-mono{font-family:'JetBrains Mono',monospace;color:var(--ink);font-weight:700;letter-spacing:-.5px}
[data-testid="stFileUploaderDropzoneInstructions"]{display:none!important}
[data-testid="stFileUploaderDropzone"]{padding:8px 10px!important;min-height:0!important}
[data-testid="stFileUploaderDropzone"]>button,[data-testid="stFileUploaderDropzone"] button{margin:0!important;width:100%}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
_MARK = ('<svg viewBox="0 0 120 120" fill="none" stroke-linecap="square">'
         '<path d="M14 100 L60 58 L106 100" stroke="#2D6A4F" stroke-width="8"/>'
         '<path d="M22 78 L60 43 L98 78" stroke="#3E8C68" stroke-width="8"/>'
         '<path d="M30 56 L60 28 L90 56" stroke="#52B788" stroke-width="8"/>'
         '<path d="M40 36 L60 18 L80 36" stroke="#C77D3A" stroke-width="8"/></svg>')

_DOT = {"idle": "#B4B4BB", "ready": "#C77D3A", "ok": "#2D6A4F"}


def render_top(text, tone="idle"):
    top_slot.markdown(
        f"<div class='cf-top'><span class='mk'>{_MARK}</span>"
        f"<span class='cf-wm'>conifer</span>"
        f"<span class='cf-sub'>Stand Structure Studio</span>"
        f"<span class='cf-sp'></span>"
        f"<span class='cf-pill'>v{conifer.__version__}</span>"
        f"<span class='cf-pill' style='margin-left:8px'><span class='cf-dot' style='background:{_DOT.get(tone)}'></span>"
        f"{text}</span></div>",
        unsafe_allow_html=True)


def sec(label):
    st.markdown(f"<div class='cf-sec'>{label}</div>", unsafe_allow_html=True)


def rhead(title, desc=""):
    st.markdown(f"<div class='cf-h'><span class='t'>{title}</span>"
                f"<span class='d'>{desc}</span></div>", unsafe_allow_html=True)


def kpi_row(items):
    cells = ""
    for it in items:
        hl = " hl" if it.get("hl") else ""
        cells += (f"<div class='cf-kpi{hl}'><div class='l'>{it['l']}</div>"
                  f"<div class='v'>{it['v']}</div>"
                  f"<div class='s {it.get('scls','')}'>{it.get('s','')}</div></div>")
    st.markdown(f"<div class='cf-kpis'>{cells}</div>", unsafe_allow_html=True)


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
def _demo(regime="sparse"):
    return conifer.demo.make_cruise(seed=7, regime=regime)


S = st.session_state
S.setdefault("trees", None)
S.setdefault("aux", None)
S.setdefault("gdf", None)
S.setdefault("fitted", None)


def load_demo(regime="sparse"):
    trees, aux, stands, truth = _demo(regime)
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


# ---------------------------------------------------------------------------
# top bar
# ---------------------------------------------------------------------------
top_slot = st.empty()


# ---------------------------------------------------------------------------
# config rail — data in
# ---------------------------------------------------------------------------
with st.sidebar:
    sec("Data")
    demo_regime_label = st.radio(
        "Demo sampling regime",
        ["Sparse — shows the small-area gain", "Data-rich — converges to direct"],
        help="Only affects the demo cruise. Sparse gives 2–3 plots per stand — the thin-sample "
             "regime where a direct per-stand estimate is too noisy and CONIFER beats it by "
             "borrowing across stands. Data-rich gives ~20 plots per stand, where the direct "
             "estimate is already good and CONIFER converges to it rather than beating it.")
    demo_regime = "sparse" if demo_regime_label.startswith("Sparse") else "rich"
    if st.button("Use the demo cruise", use_container_width=True,
                 help="200 stands on a fixed-area cruise with LiDAR metrics and stand polygons — "
                      "a synthetic but silviculturally realistic Inland-Northwest cruise. Runs the "
                      "whole workflow on realistic data before you point it at your own."):
        load_demo(demo_regime)
        st.rerun()
    up_trees = st.file_uploader("Tree list — one row per tallied tree", type=["csv", "xlsx", "xls"],
                                help="One row per tallied tree: stand id, DBH, and ideally a plot id.")
    st.caption("CSV, XLSX or XLS")
    up_aux = st.file_uploader("Stand metrics — one row per stand", type=["csv", "xlsx", "xls"],
                              help="LiDAR or spectral summaries per stand — what the model borrows strength from.")
    st.caption("CSV, XLSX or XLS")
    up_geo = st.file_uploader("Stand polygons — optional", type=["gpkg", "geojson", "json", "zip"],
                              help="Only used to draw maps and export a GeoPackage to QGIS/ArcGIS. Every "
                                   "table, chart and interval is computed without geometry.")
    st.caption("GeoPackage, GeoJSON or zipped SHP — only for maps + GIS export")

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


# ---------------------------------------------------------------------------
# empty state — onboarding
# ---------------------------------------------------------------------------
if trees is None:
    render_top("No data loaded", "idle")
    st.markdown(
        "<div class='cf-empty'>"
        "<div class='cf-kicker'>Robust · Compositional · Confident</div>"
        "<h1>Diameter distributions that hold up.</h1>"
        "<p><span class='cf-mono'>CONIFER</span> — COmpositional Nonlinear-debiased Inference, "
        "Fay–Herriot with Ellipsoidal conformal Regions. Load the demo cruise to walk the whole "
        "workflow on realistic data; nothing leaves this machine.</p></div>",
        unsafe_allow_html=True)
    _l, _c, _r = st.columns([1, 1.1, 1])
    with _c:
        if st.button("Load the demo cruise", type="primary", use_container_width=True, key="empty_demo"):
            load_demo(demo_regime)
            st.rerun()
        st.caption("or upload a tree list in the left panel.")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    feats = [
        ("INPUT", "What you bring",
         "A tree list — stand, plot, DBH. Stand-level LiDAR or spectral metrics give the model "
         "something to borrow from; stand polygons turn results into maps."),
        ("METHOD", "What it does",
         "Bins DBH, derives the right sampling covariance for your plot design, fits the small-area "
         "model, then calibrates intervals against your own held-out plots."),
        ("OUTPUT", "What you get",
         "Stand and stock tables, per-acre summaries, QMD and basal area, maps, an Excel workbook, "
         "a printable report — and a measured coverage figure worth showing a client."),
    ]
    for col, (k, h, b) in zip((f1, f2, f3), feats):
        col.markdown(f"<div class='cf-feat'><div class='k'>{k}</div><h4>{h}</h4><p>{b}</p></div>",
                     unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# config rail — model setup
# ---------------------------------------------------------------------------
render_top("Ready to run", "ready")
tc = list(trees.columns)
with st.sidebar:
    sec("Columns")
    stand_col = st.selectbox("Stand id", tc, index=tc.index(_guess(tc, "stand", "stand_id", "unit")) if _guess(tc, "stand", "stand_id", "unit") else 0)
    dbh_col = st.selectbox("DBH", tc, index=tc.index(_guess(tc, "dbh_in", "dbh", "diameter")) if _guess(tc, "dbh_in", "dbh", "diameter") else 0)
    _p = _guess(tc, "plot", "plot_id", "point")
    plot_col = st.selectbox("Plot id — worth finding", ["(none)"] + tc,
                            index=(tc.index(_p) + 1) if _p else 0,
                            help="Plot identifiers let CONIFER split each stand's plots in two and "
                                 "calibrate its intervals against the half it did not fit on. Without "
                                 "them it falls back to a method known to under-cover — intervals look "
                                 "reassuringly tight and are wrong.")
    plot_col = None if plot_col == "(none)" else plot_col

    sec("Cruise")
    _opts = ["Fixed-area plots", "Variable-radius (prism / BAF)"]
    _default_design = 0 if conifer.demo.DEMO_DESIGN == "fixed" else 1
    design = st.radio("Plot design", _opts, index=_default_design,
                      help="Sets how each stand's sampling covariance is estimated. A prism selects "
                           "trees in proportion to basal area, so every tree carries its own expansion "
                           "factor. Getting this wrong does not raise an error; it quietly returns the "
                           "wrong numbers.")
    if design.startswith("Variable"):
        baf = st.number_input("Basal area factor (ft²/ac)", 5.0, 100.0,
                              float(conifer.demo.DEMO_BAF), 5.0)
        plot_area, dsg = None, "prism"
    else:
        plot_area = st.number_input("Plot size (acres)", 0.005, 2.0,
                                    float(conifer.demo.DEMO_PLOT_AREA), 0.005, format="%.3f")
        baf, dsg = None, "fixed"

    brk = st.selectbox("DBH classes", ["Six 4-inch classes (1–25 in)",
                                       "FIA 2-inch classes (1–29 in)",
                                       "FIA 1-inch classes (1–21 in)"])
    breaks = {"Six 4-inch classes (1–25 in)": "default",
              "FIA 2-inch classes (1–29 in)": "fia_2in",
              "FIA 1-inch classes (1–21 in)": "fia_1in"}[brk]
    min_dbh = st.number_input("Ignore trees below (in)", 0.0, 20.0, 0.0, 0.5)

    sec("Uncertainty")
    conf = st.select_slider("Interval level", [80, 90, 95], value=90)
    alpha = 1 - conf / 100
    scope = st.radio(
        "What should the interval promise?",
        ["Each diameter class (recommended)", "All classes at once"],
        help="Per-class answers the question people actually ask — how many 10–15 inch stems? — and "
             "is several times narrower. The joint set promises every class is contained at once, a "
             "stronger claim and priced accordingly. Both are honestly calibrated.")
    joint = scope.startswith("All")
    mode = "maxscore"

    acols = list(aux.columns) if aux is not None else []
    aux_stand = group_col = None
    if aux is not None:
        sec("Stand metrics")
        g = _guess(acols, "stand", "stand_id")
        aux_stand = st.selectbox("Stand id in metrics", acols,
                                 index=acols.index(g) if g else 0)
        gt = _guess(acols, "stand_type", "type", "stratum", "forest_type")
        group_col = st.selectbox("Stratify by — optional", ["(none)"] + acols,
                                 index=(acols.index(gt) + 1) if gt else 0,
                                 help="Calibrates intervals within each stand type rather than pooling "
                                      "across all of them — worth using if your types really behave "
                                      "differently.")
        group_col = None if group_col == "(none)" else group_col

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run = st.button("Run model", type="primary", use_container_width=True)
    st.caption("Runs on this machine · nothing leaves it.")


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

rhead("Input review", "What CONIFER read from your data, and anything that looks off.")
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
        import warnings as _warnings
        try:
            with _warnings.catch_warnings(record=True) as _wlist:
                _warnings.simplefilter("always")
                with st.spinner("Fitting the small-area model…"):
                    est = inv.fit()
                with st.spinner("Calibrating prediction sets (no known truth needed)…"):
                    cal = conifer.conformalize_holdout(est, alpha=alpha, joint=joint, mode=mode, reps=4)
                with st.spinner("Checking whether those intervals actually hold up…"):
                    try:
                        cov = conifer.coverage_check(est, alpha=alpha, joint=joint, mode=mode, reps=8)
                    except Exception:
                        cov = None
            S.warns = [str(w.message) for w in _wlist]
            S.fitted = (est, cal, cov, alpha, mode, joint)
        except Exception as e:
            st.error("**The fit failed.**\n\n```\n" + "".join(
                traceback.format_exception_only(type(e), e)) + "```")
            with st.expander("Full detail"):
                st.code(traceback.format_exc())
            st.stop()

    est, cal, cov, alpha, mode, joint = S.fitted
    warns = S.get("warns", [])
    M = crep._meta(est)
    s = np.asarray(est.s_hat_, float)
    render_top(f"Fitted · {s.shape[0]:,} stands", "ok")
    qmd = crep.quadratic_mean_diameter(s, M["midpoints"])
    ba = crep.basal_area(s, M["midpoints"], M["dbh_units"])
    ba_units = "ft²/ac" if M["dbh_units"] == "in" else "m²/ha"

    # Surface what the library warns about; these go to the server console otherwise and the
    # forester in the browser never sees the naive fallback that under-covers.
    if isinstance(cal, dict) and cal.get("method") == "naive":
        st.error(
            "**These intervals under-cover — do not quote them at the stated level.** Calibration fell "
            "back to a method known to produce prediction sets that are too narrow, so the true coverage "
            "is materially below the level shown. The usual cause is a missing plot / point ID column — "
            "set one in the **Columns** panel — but it can also happen when too few stands have repeat "
            "plots to hold out on.")
    for _w in warns:
        st.warning(_w)

    if dsg == "prism":
        st.info(
            "**Prism cruise:** expect the smallest-diameter class to sit **below** the field-only "
            "estimate — that is the method working, not a miss. A prism tallies small trees with a very "
            "large per-tree expansion, so the field-only small-class estimate is high-variance and "
            "inflated, and CONIFER shrinks it toward what comparable stands support.")

    rhead("At a glance", "Headline numbers across every stand in the run.")
    k = st.columns(5)
    k[0].metric("Stands estimated", f"{s.shape[0]:,}")
    k[1].metric(f"Mean {M['dens']}", f"{s.sum(1).mean():,.0f}", help="per stand, per acre")
    k[2].metric("Mean basal area", f"{ba.mean():.0f}", help=ba_units)
    k[3].metric(f"Mean QMD ({M['dbh_units']})", f"{qmd.mean():.1f}")
    if cov is not None:
        k[4].metric(f"{int((1-alpha)*100)}% {'joint' if joint else 'per-class'} — measured",
                    f"{cov['empirical']*100:.0f}%",
                    delta="holds up" if cov["meets_nominal"] else "below target",
                    delta_color="normal" if cov["meets_nominal"] else "inverse")
    else:
        k[4].metric("Coverage", "—", help="needs plot ids")

    if cov is not None:
        (st.success if cov["meets_nominal"] else st.warning)(cov["summary"])
        st.caption(cov["note"])

    tabs = st.tabs(["Summary", "Tables", "Charts", "Map", "Coverage", "Export"])

    # ---- summary ------------------------------------------------------
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

    # ---- charts -------------------------------------------------------
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
                       "They should be the pale ones — thin cruises. A well-plotted stand that has "
                       "drifted a long way is worth a second look.")
        with c2:
            try:
                f, ax = plt.subplots(figsize=(6.4, 3.5))
                cplots.plot_borrowing(est, ax=ax)
                _fig(f)
                st.caption("Dark green is what the stand's own plots contributed; pale green is what "
                           "came from stands with similar structure.")
            except Exception:
                pass

    # ---- map ----------------------------------------------------------
    with tabs[3]:
        if S.gdf is None:
            st.info("Upload a stand polygon layer (GeoPackage, GeoJSON, or a zipped shapefile) "
                    "in the left panel to map these results.")
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
                    st.caption("Darker stands carry the widest intervals relative to their estimate. "
                               "If you have budget for more plots next season, this map is where to spend it.")
            except Exception as e:
                st.error(f"Could not draw the map: {e}")

    # ---- coverage -----------------------------------------------------
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
        st.caption("The direct column is what your cruise supports on its own. Where the two part "
                   "company, check the plot count first — thin stands are where the model is supposed "
                   "to move the answer, and where you should question it if it moved the wrong way.")
        try:
            st.dataframe(crep.comparison_table(est, alpha=alpha, joint=joint), use_container_width=True, height=380)
        except Exception as e:
            st.warning(str(e))

    # ---- export -------------------------------------------------------
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Excel workbook")
            st.caption("Summary by stand, full stand table, the distribution, the field-vs-CONIFER "
                       "comparison, and the data checks — one sheet each.")
            buf = _io.BytesIO()
            crep.to_excel(est, buf, alpha=alpha, joint=joint)
            _dl("Download conifer_results.xlsx", buf.getvalue(), "conifer_results.xlsx",
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
                         subtitle=f"{s.shape[0]} stands · CONIFER v{conifer.__version__} small-area estimate",
                         alpha=alpha, joint=joint, coverage=cov, calibration=cal, figures=figs)
            _dl("Download stand_report.html", open(tmp, "rb").read(), "stand_report.html", "text/html")

        with c2:
            st.markdown("##### Plain CSV")
            _dl("Download summary_by_stand.csv",
                crep.summary_table(est, alpha=alpha, joint=joint).to_csv().encode(),
                "summary_by_stand.csv", "text/csv", key="c1")
            _dl("Download stand_table.csv",
                crep.stand_table(est, alpha=alpha, joint=joint).to_csv().encode(),
                "stand_table.csv", "text/csv", key="c2")

            st.markdown("##### Back into your GIS")
            if S.gdf is None:
                st.caption("Upload a polygon layer to get a GeoPackage with the estimates joined on, "
                           "ready to open in QGIS or ArcGIS Pro.")
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
                    _dl("Download conifer_stands.gpkg", open(gp, "rb").read(),
                        "conifer_stands.gpkg", "application/geopackage+sqlite3", key="g1")
                except Exception as e:
                    st.warning(f"Could not build the GeoPackage: {e}")
else:
    st.info("Everything above checks out. Press **Run model** in the left panel.")
