<p align="center">
  <img src="https://raw.githubusercontent.com/IFC-UIDAHO/conifer/main/assets/banner.png" alt="CONIFER — compositional, design-aware small-area estimation of forest diameter distributions" width="100%">
</p>

<p align="center">
  <b>CO</b>mpositional <b>N</b>onlinear-debiased <b>I</b>nference, <b>F</b>ay–Herriot with <b>E</b>llipsoidal conformal <b>R</b>egions<br>
  <em>Robust. Compositional. Confident.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/conifer-sae/"><img src="https://img.shields.io/badge/pypi-v0.2.3-orange" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9--3.12-blue" alt="Python 3.9-3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2D6A4F.svg" alt="License: MIT"></a>
  <a href="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml"><img src="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

Design-aware small-area estimation of forest structure as *distributions*, not just totals.
CONIFER estimates the **diameter distribution** — stem density split across DBH classes — for
small forest areas where the field sample is too thin for a reliable direct estimate, and it
attaches an honest, *checked* statement of uncertainty.

Under the hood it is a **compositional area-level Fay–Herriot** estimator with a cross-fitted,
one-step-**debiased machine-learned mean** and **design-aware conformal prediction sets on the
simplex**. It reduces exactly to classical Fay–Herriot when the mean is linear.

```bash
pip install conifer-sae
```

## Start from your cruise data

You have a tree list and, ideally, stand-level remote-sensing metrics. That is all CONIFER needs.

```python
import conifer

inv = conifer.from_treelist(
    trees,                       # one row per tallied tree
    stand_col="STAND", plot_col="PLOT", dbh_col="DBH_IN",
    plot_area=0.2,               # fixed-area cruise; use baf=20 for a prism cruise
    aux=stand_metrics,           # LiDAR / spectral / terrain, one row per stand
    aux_stand_col="STAND",
)
print(inv.describe())            # what CONIFER sees
print(inv.issue_table())         # problems, in language you can act on

est = inv.fit()

# prediction intervals, calibrated from your own plots — no known truth needed
lo, hi = conifer.calibration.calibrated_intervals(est, alpha=0.10)
print(est.interval_report_.summary)

cov = conifer.coverage_check(est)                 # and check that they actually hold up
print(cov.summary)
# "Across 10 independent splits, the 90% interval for a given diameter class contained
#  the held-out field value 91% of the time. That meets the stated level."

conifer.report.to_html(est, "stand_report.html", coverage=cov)
conifer.report.to_excel(est, "results.xlsx")
```

`from_treelist` bins DBH, derives effective area, and wires the **correct sampling covariance for
your plot design** — a prism cruise gets a design-based covariance built from the plot replicates,
because the multinomial assumption behind the analytic one does not hold when each tree carries a
DBH-dependent expansion factor. Stand identifiers are carried through every table, figure and
export, so a mis-sorted input can no longer silently corrupt the estimate.

Metric units, FIA 1″/2″ class breaks, stand polygons in and out (`read_stands`,
`attach_estimates`), and a `min_dbh` merchantability threshold are all supported.

## No code at all

```bash
pip install "conifer-sae[app]"
conifer-studio
```

If your shell reports `conifer-studio: command not found` (common on Windows, where pip's
scripts folder is often not on `PATH`), this does the same thing and never depends on `PATH`:

```bash
python -m conifer.studio
```

Upload a tree list — or press **Load the demo cruise** — and get stand tables, maps, an Excel
workbook and a printable stand report. Everything runs on the machine you start it on, so
proprietary inventory never leaves your network. See [`apps/forester/`](apps/forester/).

## About the uncertainty

Two things here are worth stating plainly, because both are easy to get wrong.

**Calibration without a known truth.** Conformal prediction needs a calibration set where the truth
is known. A real inventory never has one. Calibrating against the design-direct estimate of the
*same* plots looks reasonable and is **invalid**: CONIFER's estimate is a shrinkage *of* that
estimate, so the residual is mechanically smaller than the true error — measured against a known
truth it under-covers at 68% for a nominal 90% set. `conformalize_holdout()` instead splits each
stand's *plots* in two, fits on one half and calibrates against the other. Independent by
construction, and it errs wide rather than narrow.

**Per-class by default.** For any one diameter class you name, the reported interval contains the
truth at the stated rate. That is the question a forester actually asks, and it is several times
narrower than a set guaranteed to contain *all* classes at once. Pass `joint=True` for the
simultaneous claim; every table and figure states which of the two it is reporting.

**Tallied and untallied classes are different problems.** `calibrated_intervals()` builds a
Gaussian interval on the parametric bootstrap MSE for classes a stand actually tallied, inflated
per class by a factor calibrated from your own plot splits, and gives classes with no tally a
stratified bound and a physical zero floor instead. No Gaussian works at the zero boundary
whatever its standard deviation. Measured against a known truth on a cruise shaped like the real
St. Joe inventory, at a nominal 90%:

| | measured coverage |
|---|---|
| overall | **0.946** |
| classes the cruise tallied | 0.969 |
| classes with no tally | 0.927 |

Deliberately a little above nominal, not on it: a calibration tuned to hit 90% exactly on one
forest will not hold on the next. On the classes anyone acts on the interval runs about ±90% of
the estimate. Read the untallied classes in stems per acre rather than percent — the denominator
there is near zero, so a percentage reads alarmingly and means very little.

Both claims are checked, not asserted: `coverage_check()` measures realised coverage on held-out
plots and reports it in a sentence you can put in front of a client.

## Or start from matrices

The v0.1 interface is unchanged, for when the data is already aligned:

```python
import conifer
est = conifer.DiameterDistribution(seed=0).fit(counts, area_eff, X)
est.conformalize(s_truth_cal, cal_idx, joint=True, alpha=0.10)
lo, hi = est.predict_interval(joint=True)
est.benchmark(totals, var_totals=var_totals)      # coherence with FIA class totals
```

Run the minimal worked example:

```bash
python examples/quickstart.py
```

## The step-by-step vignette

An executed walkthrough of the path above — the tree list you have, `from_treelist` and its data
checks, the fit, *how much of each estimate came from that stand's own plots*, calibrated
intervals with a coverage check, the tables, your cruise beside CONIFER, and the plain-language
narrative. Every number and figure is produced by the code above it, on a synthetic cruise
calibrated to the measured shape of the real St. Joe inventory.

- **Notebook** (renders on GitHub): [`docs/vignettes/conifer-getting-started.ipynb`](https://github.com/IFC-UIDAHO/conifer/blob/main/docs/vignettes/conifer-getting-started.ipynb)
- **Rendered HTML**: [open the executed vignette](https://raw.githack.com/IFC-UIDAHO/conifer/main/docs/vignettes/conifer-getting-started.html)

```bash
pip install conifer-sae
jupyter notebook docs/vignettes/conifer-getting-started.ipynb
```

## The names

CONIFER is a **package**; you build **estimators** on a shared **engine**.

| Name | What it is |
|------|------------|
| `conifer.DiameterDistribution` | the estimator you fit (formerly `StemDensityClassSAE`) |
| `conifer.CompositionalFH` | the reusable engine underneath it — debiased-ML mean + FH + conformal simplex sets |
| `conifer.SpeciesComposition` | planned v0.3 sibling: species shares on the same engine |
| `conifer.io` | tree lists, stand polygons, validation — everything upstream of the fit |
| `conifer.calibration` | conformal calibration when the truth is unknown |
| `conifer.report` | stand tables, plain-language narrative, Excel and HTML deliverables |

A new region is a **run, not a name** — `DiameterDistribution(spatial=True, regen_aware=True)`
toggles capabilities; you don't fork the package per state.

## Why it exists (what nothing else does)

Small-area estimation and conformal prediction are both mature — but not together, and not on the
simplex, and not for forestry:

- **emdi / sae** (R) give area-level FH with parametric MSE, but no compositional target and no
  distribution-free prediction sets.
- **MAPIE / crepes** (Python) give conformal prediction, but no small-area borrowing and no simplex
  geometry.
- **rFIA / FIESTA** give design-based forest estimates, but do not model or borrow strength.

CONIFER sits in that gap: **compositional SAE + design-aware conformal sets on the simplex,
forestry-native** — and it consumes design-based FIA estimates as its benchmark rather than
competing with them.

## Reading the output on a prism cruise

On a variable-radius (prism/BAF) cruise, expect CONIFER's estimate in the **smallest** diameter
class to sit well below the field-only estimate. That is the method working, not a miss: a prism
selects trees with probability proportional to basal area, so a single tallied small tree carries a
very large trees-per-acre expansion, and only a minority of stands catch one at all. The field-only
small-tree estimate is therefore high-variance and inflated in exactly those stands, and CONIFER
shrinks it toward what comparable stands and the covariates support. The `% from this stand's own
plots` column tells you how far that shrinkage went for each stand.

## Command line

Point it at three aligned CSV matrices:

```bash
conifer fit --counts counts.csv --area area.csv --aux aux.csv --out s_hat.csv
```

## Validation

The estimator was developed and stress-tested on the St. Joe (Idaho) cruise. On the merchantable
diameter distribution it significantly beats the direct estimator and a broad competitor slate —
kNN, Weibull, MERF, SAEforest, BART-FH, Dirichlet-multinomial, KBAABB and a multivariate
Fay–Herriot — while remaining coherent with the design-based FIA totals it rolls up to. A
design-based Monte-Carlo study on two plasmode populations reproduces the same ordering
independently, and finds conformal coverage holding at 0.90–0.91 against a known truth across
sampling intensities.

Two things this project has been careful to state rather than bury:

- **CONIFER defers where it should.** As a stand accumulates plots, a Fay–Herriot estimator is
  supposed to converge to the direct estimate rather than beat it, and it does. The gain is in
  genuinely thin samples — which is what small-area estimation is for.
- **The bundled demo is not evidence of accuracy.** It is calibrated to the *shape* of a real
  cruise so the workflow and the intervals can be exercised honestly, but at ~20 plots per stand
  it sits in the data-rich regime and its synthetic covariates carry less signal than real LiDAR.
  `conifer.demo`'s own docstring says so. The accuracy claim rests on the Idaho and Arkansas
  studies against real cruises.

## Citing this work

A methodology manuscript describing the estimator is **under review**; this README will be updated
with the citation once it is available. Until then, cite the software:

```bibtex
@software{poolakkal_conifer,
  author  = {Poolakkal, Jaslam},
  title   = {{CONIFER}: compositional, design-aware small-area estimation of
             forest diameter distributions},
  url     = {https://github.com/IFC-UIDAHO/conifer},
  note    = {Methodology manuscript under review}
}
```

`CITATION.cff` in this repository carries the machine-readable version, which GitHub renders as a
*Cite this repository* button.

## Funding

This work was supported by the NCASI Foundation on a certain project funded by the USDA Forest
Service, Rocky Mountain Research Station, through the
[Partnership for Small Area Estimation](https://www.ncasifoundation.org/projects/partnership-for-small-area-estimation/).

## License

MIT. See [`LICENSE`](LICENSE).
