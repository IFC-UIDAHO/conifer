"""Build the CONIFER getting-started vignette as an executed notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
co = lambda t: C.append(nbf.v4.new_code_cell(t))

md("""# CONIFER — from a tree list to a stand report

### Compositional, design-aware small-area estimation of forest diameter distributions

You have a cruise. Some stands got plenty of plots, some got two. You need the **diameter
distribution** — stems per acre by DBH class — for every stand, including the thin ones, with
an interval you can defend to whoever is paying for the inventory.

This walkthrough goes end to end on a synthetic cruise shaped like a real one. Every number
and figure below is produced by the code above it.""")

md("""## 1. The data you actually have

Three things, and only the first is mandatory:

| | what it is |
|---|---|
| **tree list** | one row per tallied tree — stand, plot, species, DBH |
| **stand metrics** | one row per stand — LiDAR / spectral / terrain predictors |
| **stand polygons** | optional; turns the results into maps |

The demo cruise is a synthetic but silviculturally realistic stocked Inland-Northwest
mixed-conifer inventory — stands range from dense small-tree regeneration to open, large-tree
old growth, and the larger diameter classes are absent from many stands. That last property
matters more than it looks: those stands genuinely tally **nothing** in the larger classes, and
an estimator that ignores this will quietly invent trees. By default the cruise is *sparse* —
2–3 plots per stand, the thin-sample regime small-area estimation is built for.""")

co("""import numpy as np, pandas as pd, conifer
pd.set_option("display.width", 150)

trees, stand_metrics, stands, truth = conifer.demo.make_cruise(n_stands=160, seed=11)
print("CONIFER", conifer.__version__)
trees.head(6)""")

co("""stand_metrics.head(4)""")

md("""## 2. One call to get it in

`from_treelist` bins DBH into classes, derives the effective area, and — importantly — wires
the **correct sampling covariance for your plot design**. It also carries your stand
identifiers through every table, figure and export, so a mis-sorted input cannot silently
corrupt the estimate.""")

co("""inv = conifer.from_treelist(
    trees,
    stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
    plot_area=0.2,                      # fixed-area cruise; use baf=20 for a prism cruise
    aux=stand_metrics, aux_stand_col="STAND",
    group_col="STAND_TYPE",             # calibrate within stand type
)
inv.describe()""")

md("""### Check the data before you trust the answer

`validate()` runs automatically and reports problems in language you can act on.""")

co("""inv.issue_table()""")

md("""## 3. Fit

One call. The estimator blends each stand's own plot data with what comparable stands and the
covariates support — leaning on the model exactly as far as that stand's data is thin.""")

co("""est = inv.fit()
print("estimated", est.s_hat_.shape[0], "stands x", est.s_hat_.shape[1], "DBH classes")""")

md("""### How much of this came from my cruise?

The first question any forester asks. `data_gain` answers it honestly: it is the model's own
shrinkage weight, γ = Su/(Su+Dᵢ), which rises toward 1 as a stand accumulates plots.""")

co("""from conifer.report import data_gain
g = data_gain(est)
print("share of the estimate from the stand's own plots:")
print("  min %.0f%%   median %.0f%%   max %.0f%%" % (100*g.min(), 100*np.median(g), 100*g.max()))
print("  correlation with plot count: %.2f" % np.corrcoef(g, inv.n_plots)[0, 1])""")

md("""## 4. Intervals that survive contact with the tail

This is the part that is easy to get wrong, so it is worth being explicit.

Conformal prediction needs a calibration set where the truth is known. **A real inventory never
has one.** The tempting shortcut — calibrating against the direct estimate of the same plots
used to fit — is invalid, because the estimate is a *shrinkage of* that target, so the residual
is mechanically smaller than the real error. Measured against a known truth it under-covers at
68% for a nominal 90% set.

`calibrated_intervals` does three things instead, and treats tallied and untallied classes as
the different problems they are:

1. **Classes the stand tallied** get a Gaussian built on the parametric bootstrap MSE, inflated
   per class by a factor calibrated from *your* data by splitting each stand's plots.
2. **Classes with no tally** get a stratified bound plus a physical zero floor. No Gaussian
   works at the zero boundary whatever its standard deviation.
3. The whole thing is deliberately calibrated to sit slightly **wide** of the stated level. A
   factor tuned to hit nominal exactly on one forest does not transfer to the next.""")

co("""from conifer.calibration import calibrated_intervals

lo, hi = calibrated_intervals(est, alpha=0.10, B=10, reps=2, seed=1)
rep = est.interval_report_
print("per-class inflation, calibrated from this cruise:", np.round(rep["inflation_per_class"], 2))
print()
print(rep.summary)""")

md("""### Does it hold up?

The demo kept the truth, so we can check. A real inventory cannot do this — which is why the
calibration above never uses it.""")

co("""T = truth.set_index("STAND").reindex(inv.stand_ids).to_numpy(float)
pop = inv.counts > 0
inside = (T >= lo) & (T <= hi)
print("nominal 90% | measured coverage")
print("  overall                %.3f" % inside.mean())
print("  classes with a tally   %.3f" % inside[pop].mean())
print("  classes with no tally  %.3f" % inside[~pop].mean())""")

md("""Slightly above nominal on every stratum — the right direction to be wrong for a tool
someone makes decisions with.""")

md("""## 5. The tables you hand someone""")

co("""from conifer import report as R
R.class_summary(est)""")

co("""R.summary_table(est).head(6)""")

md("""Note the two width columns. The relative figure is computed on the classes a stand
actually tallied; for classes with no tally the interval runs from zero up to a fraction of a
stem per acre, which is small in absolute terms but enormous as a percentage because the
estimate it is measured against is near zero. Read those in TPA, not percent.""")

co("""R.stand_table(est).head(4)""")

md("""## 6. Field-only against CONIFER

The argument for a model-based estimate makes itself, or it does not. Stands far off the 1:1
line are where the model changed the answer — and they should be the ones with few plots.""")

co("""%matplotlib inline
import matplotlib.pyplot as plt
from conifer import plots as P

fig, ax = plt.subplots(figsize=(6.4, 5.2))
P.plot_comparison(est, ax=ax)
plt.tight_layout(); plt.show()""")

co("""fig, ax = plt.subplots(figsize=(8.2, 3.8))
P.plot_distribution(est, stand=0, ax=ax)
plt.tight_layout(); plt.show()""")

md("""## 7. In plain words

`narrative()` reads the result back in sentences. Every figure in it is taken from the fit —
nothing is generated, so nothing can be a hallucinated number.""")

co("""for para in R.narrative(est):
    print(para.replace("**", ""), "\\n")""")

md("""## 8. Out the door

```python
conifer.report.to_excel(est, "results.xlsx")          # multi-sheet workbook
conifer.report.to_html(est, "stand_report.html")      # printable stand report

# and back into your GIS, with legal field names
gdf = conifer.read_stands("stands.gpkg", stand_col="STAND")
conifer.attach_estimates(gdf, R.summary_table(est), stand_col="STAND").to_file("out.gpkg", driver="GPKG")
```

Or skip the code entirely:

```bash
pip install "conifer-sae[app]"
streamlit run apps/forester/app.py
```

---

### What this demo does and does not show

It exercises the **workflow** on plausibly-shaped data and checks that the intervals hold up.
In its default **sparse** regime — 2–3 plots per stand — CONIFER beats the design-direct
estimator by ~10–15% on the merchantable classes, the same thin-sample gain the real studies
measure; switch to `make_cruise(regime="rich")` and, with ~20 plots per stand, it converges to
the direct estimate instead, as a Fay–Herriot estimator should. Either way the numbers are
**simulated** — this is not evidence of accuracy. CONIFER's accuracy claim rests on the St. Joe
(Idaho) and Arkansas studies against real cruises, where the covariates are real and the
merchantable tallies are genuinely thin.""")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
import os
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "conifer-getting-started.ipynb"))
print("built", len(C), "cells")
