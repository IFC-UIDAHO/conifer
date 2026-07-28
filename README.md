<p align="center">
  <img src="https://raw.githubusercontent.com/IFC-UIDAHO/conifer/main/assets/banner.png" alt="CONIFER — compositional, design-aware small-area estimation of forest diameter distributions" width="100%">
</p>

<p align="center">
  <b>CO</b>mpositional <b>N</b>onlinear-debiased <b>I</b>nference, <b>F</b>ay–Herriot with <b>E</b>llipsoidal conformal <b>R</b>egions<br>
  <em>Robust. Compositional. Confident.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/conifer-sae/"><img src="https://img.shields.io/pypi/v/conifer-sae?color=orange&label=pypi" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9--3.12-blue" alt="Python 3.9-3.12">
  <a href="https://github.com/IFC-UIDAHO/conifer/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2D6A4F.svg" alt="License: MIT"></a>
  <a href="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml"><img src="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

> ⚠️ **Preview Release**
>
> CONIFER is under active development. APIs may change before the first stable 1.0 release.

Design-aware small-area estimation of forest structure as *distributions*, not just totals.
CONIFER estimates the **diameter distribution** — stem density split across DBH classes — for
small forest areas where the field sample is too thin for a reliable direct estimate, and it
attaches an honest, *checked* statement of uncertainty.

<p align="center">
  <a href="https://youtu.be/IR-qZSbJnhM" title="Watch the 60-second, zero-code tour">
    <img src="https://raw.githubusercontent.com/IFC-UIDAHO/conifer/main/docs/media/conifer-money.gif" alt="CONIFER vs a field-only estimate on a 2-plot stand" width="72%">
  </a>
  <br>
  <em>▶ <a href="https://youtu.be/IR-qZSbJnhM">Watch the 60-second, zero-code tour</a> — from <code>pip install</code> to a checked stand report, no code.</em>
</p>

For a stand *i* the target is the whole stand table `s_i = N_i · p_i`: a **total** density `N_i`
(stems per acre) times a **composition** `p_i` — the shares across DBH classes, a point on the
simplex. That is three hard problems at once, and no prior small-area method handles them together:

- the response is **compositional** (shares that sum to one, with a nonlinear link to canopy structure),
- the large-diameter tail is **structurally zero** (many stands genuinely tally *nothing* above 15–20″), and
- the stands you care about are **data-poor** (a handful of plots, sometimes none).

Under the hood CONIFER is a **compositional area-level Fay–Herriot** estimator with a cross-fitted,
one-step-**debiased machine-learned mean**, a **zero-robust hurdle** for the empty tail, an analytic
**design-based sampling covariance**, and **design-aware conformal prediction sets on the simplex**.
It **reduces exactly to classical Fay–Herriot** when the mean is linear (verified ratio 1.0002), so
the familiar estimator is a special case, not a competitor.

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
workbook and a printable stand report, with a per-stand *"% from your own plots"* readout so you can
see how far each estimate was borrowed. Everything runs on the machine you start it on, so
proprietary inventory never leaves your network. See [`apps/forester/`](https://github.com/IFC-UIDAHO/conifer/tree/main/apps/forester).

## Validated across four regions and a simulation

CONIFER was not tuned on one dataset. The full pipeline was developed on the Idaho origin region and
then **independently replicated, refit from scratch, on three Southeastern ownerships** — a
deliberate test of whether it generalizes rather than overfits where it was born. A design-based
Monte-Carlo simulation with a *known truth* backs the empirical ordering and hardened the estimator.

| Region | Forest type | Stands | Field vs CONIFER (sparse) | Notes |
|---|---|---|---|---|
| **St. Joe, Idaho** (origin) | Inland-NW mixed conifer | 375 | **+18%** better than direct | 3D-NAIP, FIA-coherent, external transfer test |
| **Arkansas** (AOI1) | Loblolly pine plantation | 449 | **+14%** better than direct | wins the competitor shootout; produced the adequacy gate |
| **Mississippi** (AOI3) | Loblolly pine plantation | 46 | direct wins (small, plot-rich) | the *honest counter-case*; still wins on distribution shape |
| **South Carolina** (AOI2) | Loblolly pine plantation | 33 | **+18%** better than direct | degenerate tail crashed three competitors; CONIFER ran on all |

**St. Joe, Idaho — the origin study.** ~50,000 trees on 375 industrial stands, wall-to-wall 3D-NAIP
canopy metrics, benchmarked against design-based FIA. In the sparse regime (1–2 plots per stand)
CONIFER cuts log-density RMSE from the direct estimator's **1.517 to 1.239 (+18%,** Holm-corrected
p ≈ 0), and for never-sampled stands from 2.017 to **1.412 (+30%)**. On merchantable stems (≥ 5″) the
quadratic-mean-diameter EBLUP beats direct by **+20%**. It leads a broad competitor slate — kNN,
Weibull, MERF, SAEforest, BART-FH, a Dirichlet-multinomial, a spatial Fay–Herriot, KBAABB, and a
multivariate FH on raw densities — and against that raw multivariate FH the **compositional** treatment
is decisive (1.229 vs 1.475, p < 0.001), i.e. the simplex geometry, not just "more covariates," is the
source of the gain. The estimate stays coherent with the design-based FIA totals it rolls up to
(461 vs 524 stems/acre, ≥ 2″ composition Aitchison distance 0.36). An external transfer to Moscow
Mountain — a different ownership 50 miles away — is reported honestly as *moderate* (Hellinger 0.30),
which is the limit a canopy-surface sensor should have.

**The Southeastern regions — a different forest, refit from scratch.** Arkansas (Bradley–Drew, 96%
loblolly), Mississippi (Meridian, mature + young-clearcut loblolly) and South Carolina (Greenwood,
young loblolly with a near-empty large-diameter tail) are FIA-benchmarked loblolly-pine plantations —
ecologically the opposite of Inland-NW conifer. Refit on each, CONIFER **wins or ties the
real-competitor shootout** (first in Arkansas and South Carolina, a close second on RMSE in
Mississippi while winning on distribution *shape*), delivers valid conformal coverage everywhere
(0.90–0.94), and in South Carolina simply **kept running on the degenerate tail that made BART-FH,
hierarchical BART-FH and a Dirichlet-multinomial crash** — "the crashes are the finding." Mississippi
is included precisely because it is the case where the direct estimate wins: a small, plot-rich,
homogeneous population where small-area borrowing has little to add, and CONIFER *does not
manufacture an advantage the sample size doesn't support*.

**What is intrinsic vs what you recalibrate.** Across all three Southern regions the single
load-bearing capability is the **zero-robust compositional hurdle** — removing it inflates error by
**+117% / +160% / +308%**, by far the largest effect of any component, and it is *species-agnostic*:
it is exactly what a degenerate large-diameter tail needs, whether the tail is Idaho conifer or
Carolina pine. What must be **calibrated per region, never transferred**, is the adequacy-gate
threshold (τ\* = 1 / 3 / 5 across the three regions) and the plot-density convention. Bolting on
spatial or regeneration structure did *not* help (neutral to harmful); CONIFER borrows strength
through covariates and the gate, not through region-specific machinery. That is the honest reason the
method travels: the transferable core is geometry, not an Idaho-shaped default model.

**The simulation.** Eleven populations — eight forest archetypes spanning young plantation, mixed
conifer, uneven-aged reverse-J, bimodal and old-growth, plus three *plasmode* populations built from
the **real** Arkansas/Mississippi/South Carolina LiDAR covariate matrices — sampled at 1–25 plots per
stand over 200 replicates against a known truth. CONIFER has the lowest log-density RMSE in **all
eight archetypes** (by 0.11–0.27), and its conformal joint coverage holds at **0.90–0.91 across every
sampling intensity** where the analytic Gaussian interval collapses on the zero tally tail. The
simulation also surfaced and fixed a real failure mode (see *Honest limitations*): the v0.2 estimator
now provably **converges to the direct estimate as plots accumulate**, keeping the sparse-regime gain
while deferring correctly when the data are rich.

## About the uncertainty

Two things here are worth stating plainly, because both are easy to get wrong.

**Calibration without a known truth.** Conformal prediction needs a calibration set where the truth
is known. A real inventory never has one. Calibrating against the design-direct estimate of the
*same* plots looks reasonable and is **invalid**: CONIFER's estimate is a shrinkage *of* that
estimate, so the residual is mechanically smaller than the true error — measured against a known
truth it under-covers (0.68 for a nominal 0.90 set; and a naive Gaussian box under-covers *jointly* at
0.475 against nominal 0.95). `conformalize_holdout()` instead splits each stand's *plots* in two, fits
on one half and calibrates against the other. Independent by construction, and it errs wide rather
than narrow.

**Per-class by default, joint on request.** For any one diameter class you name, the reported interval
contains the truth at the stated rate — the question a forester actually asks, and several times
narrower than a set guaranteed to contain *all* classes at once. Pass `joint=True` for the
simultaneous claim; every table and figure states which of the two it is reporting.

**The set is checked against a known truth, and it holds.** Measured on the St. Joe study, the
design-aware minimum-volume conformal set achieves **joint five-class coverage 0.946 at a nominal
0.95** — versus **0.475** for a Gaussian box on the same data — and an independent audit reproduced it
at 0.944. It is **20% tighter** than that Gaussian box at matched validity, and the zero-robust
log-ratio shrinks it by a further ~80% at unchanged coverage. Coverage holds *conditionally* too
(SD 0.032 across canopy strata), not just on average. Across the three Southeastern regions the joint
conformal coverage is **0.897–0.935** (target 0.90), while the analytic Gaussian box on a common joint
basis collapses to 0.33–0.60 — so the conformal set is *necessary*, not merely preferable.

**Why conformal, not the analytic MSE.** The analytic (plug-in or double-bootstrap) MSE is honestly
**anti-conservative — it recovers only about half of the empirical error**, most severely for
regeneration. So CONIFER reports the *conformal* set as its operational uncertainty statement, and
`coverage_check()` measures realised coverage on held-out plots and hands it back in a sentence you
can put in front of a client. On the classes anyone acts on, the interval runs about ±90% of the
estimate; read the near-empty tail classes in stems per acre, not percent, where the denominator is
near zero and a percentage reads alarmingly and means little.

## Geographic and forest-type scope

CONIFER carries the University of Idaho / Intermountain Forestry Cooperative provenance, and it is
fair to ask whether it is therefore locked to Pacific/Inland-Northwest conifers. It is not — but the
distinction matters, so here it is plainly:

- **It is a *method* you refit on your own cruise, not a frozen pretrained model.** Every region above
  was fit from scratch on its own field data. There are no Idaho-baked coefficients shipped as a
  default that would mis-fire elsewhere.
- **The validated envelope** is Inland-Northwest **mixed conifer** *and* Southeastern **loblolly-pine
  plantations** — two very different systems. The capability that carries the method (the zero-robust
  compositional geometry) is species-agnostic.
- **What you recalibrate locally** is the adequacy-gate threshold and the plot-density convention
  (both are per-dataset and audited every time), plus, of course, the fit itself against your plots.
- **What is not yet validated:** **hardwood and mixed-hardwood / bottomland systems.** All Southeastern
  validation is pine plantation; no hardwood field data has been run. Treat hardwood as unproven — not
  claimed to work, not claimed to fail.

In short: the honest warning is "recalibrate the gate and audit your density convention per region,"
not "this only works on Douglas-fir."

## About the estimator (a little deeper)

- **Debiased-ML mean.** The synthetic mean is a cross-fitted ensemble (random-feature ridge + a
  gradient-boosted residual correction) with a one-step (Riesz) debiasing applied to the out-of-fold
  mean — the discrete analogue of Neyman orthogonality. It shrinks toward the *out-of-fold* mean, which
  is what stops a flexible learner from quietly leaking the field estimate it is meant to improve on
  (debiasing lifts first-order coverage from 0.835 to 0.910). Honest caveat: the cross-fitted mean
  buys a *first-order* orthogonal MSE, not a genuine second-order expansion.
- **Total × composition.** The total `N_i` (a univariate log-scale FH) and the shares `p_i` (a
  compositional FH in additive-log-ratio space) are estimated as two coupled models and recombined by
  the delta method, with a structural-zero hurdle for class presence and additive benchmarking to the
  design total.
- **Design-aware conformal.** Split / Mondrian (group-conditional) conformal on isometric-log-ratio
  residuals, scored by the design-and-model covariance, with the minimum-volume ellipsoid of
  Braun et al. (2025); basis-equivariant on the simplex. Informative sampling is handled by a weighted
  conformal product weight (covariate-shift ratio × design weight).

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
narrative. Every number and figure is produced by the code above it, on a synthetic but
silviculturally realistic stocked cruise.

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

Small-area estimation and conformal prediction are both mature — but not together, not on the
simplex, and not for forestry. CONIFER is, to our knowledge, the **first estimator to unify in one
engine**: (i) a compositional area-level Fay–Herriot model for the full DBH-class stem-density vector,
(ii) a cross-fitted, one-step-debiased ML mean that provably nests the linear FH mean, (iii) a
zero-robust hurdle for the empty tail, (iv) design-aware minimum-volume conformal sets on the simplex,
and (v) a region-calibrated adequacy gate — validated across four regions.

- **emdi / sae** (R) give area-level FH with parametric MSE, but no compositional target and no
  distribution-free prediction sets.
- **MAPIE / crepes** (Python) give conformal prediction, but no small-area borrowing and no simplex
  geometry.
- **rFIA / FIESTA** give design-based forest estimates, but do not model or borrow strength — CONIFER
  *consumes* these as its benchmark rather than competing with them.
- The nearest prior art — Esteban et al. (2020) compositional FH, Georgakis et al. (2025) multivariate
  FH for volume/basal-area/height, Amaral et al. (2025) Dirichlet-HDR conformal, White et al. (2025)
  KBAABB — each supplies one ingredient; none targets the diameter distribution with valid,
  finite-sample, set-valued uncertainty.

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

## Honest limitations

This project states its limits rather than burying them.

- **CONIFER defers where it should.** As a stand accumulates plots, a Fay–Herriot estimator is
  supposed to converge to the direct estimate rather than beat it — and, since v0.2, it provably does.
  The gain is in genuinely thin samples, which is what small-area estimation is *for*; on well-sampled
  stands (as in Mississippi) the direct estimate can win, and CONIFER's adequacy gate is there to hand
  back to it.
- **Surface sensors are blind to the understory.** Photogrammetric (3D-NAIP) canopy metrics describe
  the canopy top, not the sub-canopy regeneration that dominates stem counts. This is the mechanism
  behind the 0–2″ class errors and the moderate cross-ownership transfer, and it is why canopy-
  penetrating LiDAR or a few local calibration plots are the right remedy for a new ownership.
- **Geography adds little here.** The estimator ties a simpler spatial Fay–Herriot within Monte-Carlo
  noise and its residuals are spatially white, so a spatial term is not warranted on these data — a
  parsimony result, and a ceiling on what geography alone can add.
- **A set-efficiency gap, largely closed.** A Dirichlet-HDR conformal set (Amaral et al. 2025) is
  tighter in principle; the zero-robust log-ratio closed ~80% of that gap, leaving a residual that is a
  modeling choice (an equivariant ellipsoid vs a density level-set), not a defect.
- **Where it loses:** the direct estimate wins the earth-mover (1-Wasserstein) *shape* metric, because
  CONIFER optimizes simplex geometry, not diameter-axis transport; and hardwood systems are unvalidated.
- **The bundled demo is synthetic.** Its default *sparse* regime (2–3 plots per stand) shows CONIFER
  beating the direct estimate — the same thin-sample gain the real studies measure — and its *rich*
  regime shows it converging to direct; but the numbers are simulated. `conifer.demo`'s own docstring
  says so. The accuracy claims rest on the four real studies above, not on the demo.

## Citing this work

A methodology manuscript describing the estimator is in preparation; this README will be updated
with the citation once it is available. Until then, cite the software:

```bibtex
@software{poolakkal_conifer,
  author  = {Poolakkal, Jaslam},
  title   = {{CONIFER}: compositional, design-aware small-area estimation of
             forest diameter distributions},
  url     = {https://github.com/IFC-UIDAHO/conifer},
  note    = {Methodology manuscript in preparation}
}
```

`CITATION.cff` in this repository carries the machine-readable version, which GitHub renders as a
*Cite this repository* button.

## Funding

Supported by the NCASI Foundation through the
[Partnership for Small Area Estimation](https://www.ncasifoundation.org/projects/partnership-for-small-area-estimation/),
under *Robust small-area estimation strategies for developing accurate stand-level diameter
distributions* (PI: Jaslam Poolakkal, University of Idaho), funded by the USDA Forest Service,
Rocky Mountain Research Station.

## License

MIT. See [`LICENSE`](https://github.com/IFC-UIDAHO/conifer/blob/main/LICENSE).
