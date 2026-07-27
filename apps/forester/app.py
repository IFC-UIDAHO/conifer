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
    menu_items={"about": "CONIFER — small-area estimation for forest inventory. "
                         "Fay–Herriot with nonlinear debiasing and conformal prediction sets."},
)

# Brand tokens (kept for any Python-side use / matplotlib harmony)
GREEN, DARK, SAND, LINE, MUTED, AMBER = "#2D6A4F", "#15281C", "#f7f5f0", "#dfe6e1", "#5d6d64", "#C77D3A"

# In-app figures share the paper surface so charts sit flush in their cards.
plt.rcParams.update({
    "figure.facecolor": "#FBFAF6", "axes.facecolor": "#FBFAF6",
    "axes.edgecolor": "#D8D2C2", "text.color": "#15281C",
    "axes.labelcolor": "#3B4A40", "xtick.color": "#5d6d64", "ytick.color": "#5d6d64",
    "font.size": 10.5,
})

# ---------------------------------------------------------------------------
# design system  ·  a paper-forward, editorial theme built on CONIFER's own
# topographic contour mark.  All colours are declared once as CSS variables so
# the app reads as one designed surface rather than default Streamlit chrome.
# ---------------------------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root{
  --paper:#F4F2EA; --paper-2:#FBFAF6; --card:#FCFBF7;
  --ink:#15281C; --ink-2:#3B4A40; --muted:#7C8A80;
  --line:#E6E1D5; --line-strong:#D8D2C2;
  --green-900:#15281C; --green-700:#1b3a26; --green:#2D6A4F; --green-500:#3E8C68; --green-400:#52B788;
  --amber:#C77D3A; --amber-tint:#F0DFC9;
  --shadow:0 18px 40px -24px rgba(21,40,28,.45);
}

/* ---- base surfaces (force the light, warm paper theme) ---- */
.stApp,[data-testid="stAppViewContainer"]{background:var(--paper)!important;color:var(--ink)}
[data-testid="stHeader"]{background:transparent!important;height:0}
[data-testid="stToolbar"],#MainMenu,footer{display:none!important}
.block-container{padding-top:2.0rem;padding-bottom:3rem;max-width:1320px}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{
  font-family:'Inter',system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
::selection{background:var(--amber);color:#fff}
.stApp a{color:var(--green);text-decoration:none;border-bottom:1px solid var(--amber-tint)}
.stApp a:hover{border-bottom-color:var(--amber)}

h1,h2,h3,h4{font-family:'Fraunces',Georgia,serif!important;color:var(--ink)!important;letter-spacing:-.2px}
h2{font-size:23px;font-weight:600;margin-top:.2rem}
h3{font-size:19px;font-weight:600}
h4,h5{font-size:16px;font-weight:600}
.stApp p,.stApp li,.stMarkdown{color:var(--ink-2);font-size:15px;line-height:1.62}
hr{border:none;border-top:1px solid var(--line);margin:1.5rem 0}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:var(--paper-2)!important;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stCaption{color:var(--ink-2)}
[data-testid="stCaptionContainer"],.stCaption,small{color:var(--muted)!important}

/* ---- inputs: warm, quiet, consistent ---- */
label,[data-testid="stWidgetLabel"] p{color:var(--ink-2)!important;font-weight:500;font-size:13.5px}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],
[data-testid="stNumberInput"] input,[data-baseweb="select"] input{
  background:var(--paper-2)!important;border-color:var(--line-strong)!important;color:var(--ink)!important;border-radius:9px!important}
[data-baseweb="select"]>div:hover{border-color:var(--green)!important}
input,textarea,[role="combobox"],[data-baseweb="select"] *{color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--paper-2)!important;border:1.5px dashed var(--line-strong)!important;border-radius:12px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--green)!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink-2)!important}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{color:var(--ink-2)!important}
[data-baseweb="slider"] [role="slider"]{background:var(--green)!important;border-color:var(--green)!important}

/* ---- buttons ---- */
.stButton>button,.stDownloadButton>button{
  border-radius:10px;border:1px solid var(--line-strong);background:var(--paper-2);color:var(--ink);
  font-family:'Inter';font-weight:600;font-size:14px;padding:9px 15px;transition:all .16s ease}
.stButton>button:hover{border-color:var(--green);color:var(--green);transform:translateY(-1px);box-shadow:0 6px 14px -8px rgba(45,106,79,.5)}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#2D6A4F 0%,#3E8C68 100%);color:#fff!important;border:none;box-shadow:0 10px 22px -12px rgba(45,106,79,.9)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.07);transform:translateY(-1px)}
.stDownloadButton>button:hover{border-color:var(--amber);color:var(--amber)}

/* ---- metrics as accented cards ---- */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:16px 18px 14px;position:relative;overflow:hidden;box-shadow:0 1px 2px rgba(20,40,28,.04)}
[data-testid="stMetric"]::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--green),var(--green-400))}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--ink)!important;font-size:25px;line-height:1.1}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:12.5px;font-weight:500;letter-spacing:.2px}
[data-testid="stMetricDelta"]{font-size:12.5px;font-weight:600}

/* ---- tabs: editorial underline, no pills ---- */
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line);background:transparent}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none;padding:11px 2px;margin-right:26px;
  color:var(--muted);font-family:'Inter';font-weight:600;font-size:13.5px;letter-spacing:.2px}
.stTabs [data-baseweb="tab"]:hover{color:var(--ink)}
.stTabs [aria-selected="true"]{color:var(--ink)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--amber)!important;height:2.5px;border-radius:3px}
.stTabs [data-baseweb="tab-border"]{display:none}

/* ---- data + alerts ---- */
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stAlert"],[role="alert"]{border-radius:12px;border:1px solid var(--line)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--paper-2)}
[data-testid="stExpander"] summary{color:var(--ink-2);font-weight:600}

/* ---- hero ---- */
.cf-hero{position:relative;overflow:hidden;border-radius:20px;padding:34px 40px;margin:2px 0 8px;
  background:radial-gradient(130% 150% at 86% -30%,#2f6a4d 0%,#1b3a26 45%,#15281C 100%);
  box-shadow:var(--shadow)}
.cf-topo{position:absolute;top:0;right:0;height:100%;width:56%;opacity:.9;pointer-events:none}
.cf-hero-inner{position:relative;display:flex;align-items:center;gap:26px}
.cf-hero-mark{flex:0 0 auto;width:66px;height:66px;display:grid;place-items:center;
  border:1px solid rgba(244,242,234,.22);border-radius:16px;background:rgba(244,242,234,.05)}
.cf-hero-mark svg{width:40px;height:40px}
.cf-wordmark{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:38px;color:#F4F2EA;
  letter-spacing:-1.5px;line-height:1}
.cf-hero-sub{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:2px;text-transform:uppercase;
  color:var(--green-400);margin:8px 0 6px;font-weight:600}
.cf-hero-tag{color:rgba(244,242,234,.82);font-size:15px;line-height:1.55;max-width:660px;margin:0 0 14px}
.cf-chips{display:flex;flex-wrap:wrap;gap:8px}
.cf-chip{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.6px;color:rgba(244,242,234,.85);
  border:1px solid rgba(244,242,234,.28);border-radius:999px;padding:5px 12px;background:rgba(244,242,234,.04)}
.cf-chip--amber{color:#E9B486;border-color:rgba(199,125,58,.55);background:rgba(199,125,58,.1)}

/* ---- section eyebrow ---- */
.cf-eyebrow{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:11px;
  font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:var(--green);margin:26px 0 2px}
.cf-tick{width:20px;height:2px;background:var(--amber);display:inline-block;border-radius:2px}
.cf-eyebrow-sub{color:var(--muted);font-size:13.5px;margin:3px 0 12px}

/* ---- intro cards ---- */
.cf-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px 22px 20px;height:100%;
  position:relative;overflow:hidden;transition:transform .16s ease,box-shadow .16s ease}
.cf-card::after{content:"";position:absolute;top:0;left:0;width:38px;height:3px;background:var(--amber);border-radius:0 0 3px 0}
.cf-card:hover{transform:translateY(-3px);box-shadow:0 16px 30px -20px rgba(21,40,28,.35)}
.cf-card-n{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--amber);font-weight:700;letter-spacing:1.5px}
.cf-card h4{margin:.45rem 0 .55rem;font-size:18px}
.cf-card p{color:var(--ink-2);font-size:14px;line-height:1.62;margin:0}
.cf-card b{color:var(--ink)}

/* ---- sidebar step headers ---- */
.cf-step{display:flex;align-items:center;gap:10px;margin:20px 0 6px}
.cf-step-n{width:25px;height:25px;border-radius:8px;background:var(--green);color:#fff;font-family:'JetBrains Mono',monospace;
  font-weight:700;font-size:13px;display:grid;place-items:center;box-shadow:0 3px 8px -3px rgba(45,106,79,.7)}
.cf-step-t{font-family:'Fraunces',serif;font-weight:600;font-size:16px;color:var(--ink)}
.cf-brand{display:flex;align-items:center;gap:11px;padding:2px 2px 14px;margin-bottom:6px;border-bottom:1px solid var(--line)}
.cf-brand svg{width:30px;height:30px}
.cf-brand-t{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:19px;color:var(--ink);letter-spacing:-1px}
.cf-brand-s{font-size:10.5px;color:var(--muted);letter-spacing:.4px}

/* ---- footer ---- */
.cf-footer{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:8px;padding-top:16px;
  border-top:1px solid var(--line);color:var(--muted);font-size:12.5px;line-height:1.5}
.cf-footer .m{font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700;letter-spacing:-.5px}
.plain{font-size:15.5px;line-height:1.7}
.plain p{font-size:15.5px;line-height:1.7;color:var(--ink-2)}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# brand marks + topographic texture (CONIFER's contour motif, drawn inline)
# ---------------------------------------------------------------------------
def _mark(colors):
    """The CONIFER contour mark — four nested chevrons, top one accented."""
    c = colors
    return (f'<svg viewBox="0 0 120 120" fill="none" stroke-linecap="square">'
            f'<path d="M14 100 L60 58 L106 100" stroke="{c[0]}" stroke-width="7"/>'
            f'<path d="M22 78 L60 43 L98 78" stroke="{c[1]}" stroke-width="7"/>'
            f'<path d="M30 56 L60 28 L90 56" stroke="{c[2]}" stroke-width="7"/>'
            f'<path d="M40 36 L60 18 L80 36" stroke="{c[3]}" stroke-width="7"/></svg>')

MARK_LIGHT = _mark(["#F4F2EA", "#F4F2EA", "#F4F2EA", "#C77D3A"])   # for the dark hero
MARK_COLOR = _mark(["#2D6A4F", "#3E8C68", "#52B788", "#C77D3A"])   # for light surfaces


def _topo_peaks():
    """A faint topographic field — nested contour peaks built from the chevron motif."""
    def peak(cx, apex_y, n, step, spread, y_gain):
        segs = []
        for i in range(n):
            half = (i + 1) * step * spread / 2
            drop = (i + 1) * step * y_gain
            ys = apex_y + drop
            ya = ys - step * 0.9
            segs.append(f'<path d="M{cx-half:.0f} {ys:.0f} L{cx:.0f} {ya:.0f} L{cx+half:.0f} {ys:.0f}"/>')
        return "".join(segs)
    big = peak(360, -18, 17, 17, 2.15, 0.60)
    small = peak(150, 66, 12, 15, 2.0, 0.62)
    return (f'<svg class="cf-topo" viewBox="0 0 480 300" preserveAspectRatio="xMaxYMid slice">'
            f'<g fill="none" stroke="#F4F2EA" stroke-width="1.2" stroke-linecap="square" opacity="0.12">{big}{small}</g>'
            f'<g fill="none" stroke="#C77D3A" stroke-width="1.3" stroke-linecap="square" opacity="0.22">'
            f'{peak(360,-18,4,17,2.15,0.60)}</g></svg>')


def eyebrow(label, sub=None):
    html = f"<div class='cf-eyebrow'><span class='cf-tick'></span>{label}</div>"
    if sub:
        html += f"<div class='cf-eyebrow-sub'>{sub}</div>"
    st.markdown(html, unsafe_allow_html=True)


def step(n, title):
    st.markdown(f"<div class='cf-step'><span class='cf-step-n'>{n}</span>"
                f"<span class='cf-step-t'>{title}</span></div>", unsafe_allow_html=True)


st.markdown(
    f"<div class='cf-hero'>{_topo_peaks()}"
    f"<div class='cf-hero-inner'>"
    f"<div class='cf-hero-mark'>{MARK_LIGHT}</div>"
    f"<div class='cf-hero-copy'>"
    f"<div class='cf-wordmark'>conifer</div>"
    f"<div class='cf-hero-sub'>Stand Structure Studio</div>"
    f"<p class='cf-hero-tag'>Diameter distributions for every stand you have — including the ones "
    f"with two plots and a prayer — and intervals that hold up when someone checks them.</p>"
    f"<div class='cf-chips'>"
    f"<span class='cf-chip'>Runs on this machine</span>"
    f"<span class='cf-chip'>Conformal prediction sets</span>"
    f"<span class='cf-chip cf-chip--amber'>Fay–Herriot · debiased</span>"
    f"</div></div></div></div>", unsafe_allow_html=True)


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
    return conifer.demo.make_cruise(seed=7)


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
    st.markdown(
        f"<div class='cf-brand'>{MARK_COLOR}<div>"
        f"<div class='cf-brand-t'>conifer</div>"
        f"<div class='cf-brand-s'>SMALL-AREA ESTIMATION</div></div></div>",
        unsafe_allow_html=True)
    step(1, "Your data")

    if st.button("Load the demo cruise", use_container_width=True,
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
    eyebrow("Start here", "Load the demo cruise from the sidebar to walk the whole workflow on "
            "realistic Idaho-shaped data — or upload your own tree list. Either way it runs on "
            "this machine and nothing leaves it, worth knowing before you point it at a client's "
            "inventory.")
    cards = [
        ("01 · INPUT", "What you bring",
         "A <b>tree list</b> — stand, plot, DBH. That alone is enough to run. Stand-level "
         "<b>LiDAR or spectral metrics</b> are what give the model something to borrow from, so "
         "without them it can't do much more than your cruise already does. <b>Stand polygons</b> "
         "turn the results into maps."),
        ("02 · METHOD", "What it does",
         "Bins DBH, works out the right sampling covariance for your plot design, fits the "
         "small-area model, then calibrates the prediction intervals <b>against your own held-out "
         "plots</b> — no known truth required, which is just as well, because no inventory has one."),
        ("03 · OUTPUT", "What you get",
         "Stand and stock tables, per-acre summaries, QMD and basal area, maps of structure and "
         "uncertainty, an Excel workbook and a printable stand report. Plus a measured figure for "
         "how often the intervals actually contained the answer — the part worth showing a client."),
    ]
    cols = st.columns(3, gap="medium")
    for col, (tag, head, body) in zip(cols, cards):
        col.markdown(
            f"<div class='cf-card'><div class='cf-card-n'>{tag}</div>"
            f"<h4>{head}</h4><p>{body}</p></div>", unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# sidebar — columns + settings
# ---------------------------------------------------------------------------
tc = list(trees.columns)
with st.sidebar:
    step(2, "Which columns?")
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

    step(3, "Your cruise")
    # Defaults follow the demo cruise, read from the package so the two cannot drift apart
    _opts = ["Fixed-area plots", "Variable-radius (prism / BAF)"]
    _default_design = 0 if conifer.demo.DEMO_DESIGN == "fixed" else 1
    design = st.radio("Plot design", _opts, index=_default_design,
                      help="This decides how CONIFER estimates each stand's sampling "
                           "covariance, and the two designs need genuinely different "
                           "treatment — a prism selects trees in proportion to basal area, so "
                           "every tree carries its own expansion factor. Getting it wrong "
                           "does not raise an error; it quietly gives you the wrong numbers.")
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

    step(4, "Uncertainty")
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
        step(5, "Stand metrics")
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

    run = st.button("Run the estimate", type="primary", use_container_width=True)


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

eyebrow("Input review", "What CONIFER read from your data, and anything that looks off.")
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

    eyebrow("At a glance", "Headline numbers across every stand in the run.")
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

    tabs = st.tabs(["In plain words", "Tables", "Figures",
                    "Map", "Does it hold up?", "Take it away"])

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

st.markdown(
    "<div class='cf-footer'><div style='max-width:820px'>CONIFER estimates are model-based: they "
    "borrow strength across stands and are not a substitute for a design-based estimate where the "
    "sample is adequate. Prediction sets state where a stand's value is expected to lie, not where "
    "the estimate's mean lies.</div>"
    "<div style='text-align:right;white-space:nowrap'><span class='m'>conifer</span> · "
    "small-area estimation<br>Fay–Herriot · nonlinear debiasing · conformal</div></div>",
    unsafe_allow_html=True)
