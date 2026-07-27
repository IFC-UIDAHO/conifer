# CONIFER vignettes

Worked, runnable walkthroughs of the package.

| Vignette | Open | Rendered |
|----------|------|----------|
| **Getting started** — fit → conformal sets → coverage → benchmarking, end to end on reproducible synthetic data | [`conifer-getting-started.ipynb`](conifer-getting-started.ipynb) | [HTML](https://raw.githack.com/IFC-UIDAHO/conifer/main/docs/vignettes/conifer-getting-started.html) |

Prefer to watch first? A ~36-second captioned screencast of this exact workflow is in
[`../media/conifer-demo.mp4`](../media/conifer-demo.mp4).

## Run locally

```bash
pip install conifer-sae
jupyter notebook docs/vignettes/conifer-getting-started.ipynb
```

Every notebook here is executed in CI-style top-to-bottom order; all numbers and figures are produced
by the code in the cell above them, and the data generators are seeded so results reproduce exactly.
