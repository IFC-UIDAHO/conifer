# CONIFER vignettes

| Vignette | Open | Rendered |
|----------|------|----------|
| **Getting started** — tree list → fit → calibrated intervals → stand report | [`conifer-getting-started.ipynb`](conifer-getting-started.ipynb) | [HTML](https://raw.githack.com/IFC-UIDAHO/conifer/main/docs/vignettes/conifer-getting-started.html) |

Prefer to watch first? A ~36-second captioned screencast is in [`../media/conifer-demo.mp4`](../media/conifer-demo.mp4).

## Run it yourself

```bash
pip install conifer-sae
jupyter notebook docs/vignettes/conifer-getting-started.ipynb
```

Every number and figure in the vignette is produced by the code above it, on a synthetic
cruise calibrated to the measured shape of the real St. Joe (Idaho) inventory. The data
generators are seeded, so results reproduce exactly.

## Rebuilding

The notebook is generated and executed from source rather than hand-edited, so the prose and
the code stay in one place and the outputs can never drift from the cells that produced them:

```bash
python docs/vignettes/build_vignette.py
jupyter nbconvert --to notebook --execute --inplace docs/vignettes/conifer-getting-started.ipynb
jupyter nbconvert --to html --template lab docs/vignettes/conifer-getting-started.ipynb
```
