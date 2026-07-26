<p align="center">
  <img src="assets/banner.png" alt="CONIFER — compositional, design-aware small-area estimation of forest diameter distributions" width="100%">
</p>

<p align="center">
  <b>CO</b>mpositional <b>N</b>onlinear-debiased <b>I</b>nference, <b>F</b>ay–Herriot with <b>E</b>llipsoidal conformal <b>R</b>egions<br>
  <em>Robust. Compositional. Confident.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/conifer-sae/"><img src="https://img.shields.io/pypi/v/conifer-sae.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/pypi/pyversions/conifer-sae.svg" alt="Python versions">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2D6A4F.svg" alt="License: MIT"></a>
  <a href="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml"><img src="https://github.com/IFC-UIDAHO/conifer/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

Design-aware small-area estimation of forest structure as *distributions*, not just totals. The
first release estimates the **diameter distribution** — stem density split across DBH classes — for
small forest areas where the field sample is too thin for a reliable direct estimate, and it comes
with an honest, *valid* statement of uncertainty.

Under the hood it is a **compositional area-level Fay–Herriot** estimator with a cross-fitted,
one-step-**debiased machine-learned mean** and **design-aware conformal prediction sets on the
simplex**. It reduces exactly to classical Fay–Herriot when the mean is linear.

```bash
pip install conifer-sae
```

## Quickstart

```python
import numpy as np, conifer

# counts: m areas x K DBH classes ; area_eff: effective sampled area ; X: covariates
est = conifer.DiameterDistribution(seed=0).fit(counts, area_eff, X)
s_hat = est.s_hat_                                   # stem density by DBH class

# design-aware conformal prediction set (valid joint coverage on the simplex)
est.conformalize(s_truth_cal, cal_idx, joint=True, alpha=0.10)
lo, hi = est.predict_interval(joint=True)

# coherence with a design-based benchmark (e.g. FIA class totals)
est.benchmark(totals, var_totals=var_totals)
```

Run the worked example:

```bash
python examples/quickstart.py
```

## The names

CONIFER is a **package**; you build **estimators** on a shared **engine**.

| Name | What it is |
|------|------------|
| `conifer.DiameterDistribution` | the estimator you fit (formerly `StemDensityClassSAE`) |
| `conifer.CompositionalFH` | the reusable engine underneath it — debiased-ML mean + FH + conformal simplex sets |
| `conifer.SpeciesComposition` | planned v0.3 sibling: species shares on the same engine |

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

## Command line

Point it at three aligned CSV matrices:

```bash
conifer fit --counts counts.csv --area area.csv --aux aux.csv --out s_hat.csv
```

## Validation

The estimator was developed and stress-tested in the St. Joe (Idaho) study: on the merchantable
diameter distribution it significantly beats the direct estimator and a broad competitor slate
(kNN, Weibull, MERF, SAEforest, BART-FH, Dirichlet-multinomial, KBAABB, a multivariate FH), is
coherent with FIA (design-consistent), and its conformal set is the only uncertainty statement with
tested valid joint coverage (0.944). With the zero-robust log-ratio default the joint set is also
efficient (~80% tighter at unchanged coverage). It reduces to classical Fay–Herriot (verified ratio
≈ 1.0002).

## Roadmap

- **v0.1** — `DiameterDistribution`: debiased-ML compositional FH + conformal simplex sets + benchmarking (this release)
- **v0.2** — capability flags: `spatial=True` (spatial random effect), `regen_aware=True` (understory sub-model)
- **v0.3** — `SpeciesComposition` (species shares) on the same engine
- **v0.4** — non-compositional metrics (volume, basal area, biomass) via a `MultivariateFH` sibling core
- **v1.0** — R front door via `reticulate` (one engine, `library(conifer)` for the FIA/forestry audience)

## Citing

If you use CONIFER, please cite it — see [`CITATION.cff`](CITATION.cff), or use GitHub's
"Cite this repository" button. Method framing: *a debiased-ML compositional Fay–Herriot with
design-aware simplex conformal prediction sets for forest diameter distributions.*

## License

MIT — see [LICENSE](LICENSE). Developed at the University of Idaho ([IFC-UIDAHO](https://github.com/IFC-UIDAHO)).

---

*The engine in `conifer/_engine/` is a vendored copy of the validated PSAE research code; fixes flow
from the research source of record (see `conifer/_engine/__provenance__.txt`). Brand assets (contour
style) are in [`assets/`](assets/).*
