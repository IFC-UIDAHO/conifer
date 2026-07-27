# CONIFER — Stand Structure Studio

A no-code front end for the CONIFER small-area estimator, built for foresters.

**What it does.** Drop in a tree list (and, if you have them, stand-level remote-sensing
metrics and a stand polygon layer). It bins DBH, wires the right sampling covariance for
your plot design, fits CONIFER, calibrates honest prediction sets, checks that those
intervals actually hold up, and hands back stand tables, maps, an Excel workbook, and a
printable stand report.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints. Click **Load the demo cruise** to see the whole workflow with
no data of your own.

## Deploying for a group

Data never leaves the machine the app runs on, so for proprietary inventory the right
deployment is a server you control:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

behind an nginx reverse proxy on a university or agency server. Streamlit Community Cloud
and Hugging Face Spaces both work for demos, but uploads transit their servers — do not
use them for client inventory.

## What the app expects

| File | Required | Columns |
|------|----------|---------|
| Tree list | yes | a stand id, a DBH, and (strongly recommended) a plot id |
| Stand metrics | no, but SAE needs it | a stand id + numeric predictors (LiDAR, spectral, terrain) |
| Stand polygons | no | GeoPackage/shapefile/GeoJSON with a matching stand id |

Without stand metrics there is nothing to borrow strength from, and CONIFER degenerates
toward the field-only estimate. The app says so rather than quietly producing a number.
