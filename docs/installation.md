# Installation

CONIFER is on PyPI as **`conifer-sae`** and imports as `conifer`.

```bash
pip install conifer-sae
```

Requires Python ≥ 3.9. Core dependencies are NumPy, SciPy, scikit-learn, pandas, and matplotlib.

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `geo` | geopandas, shapely, pyproj | reading/writing stand polygons (shapefile / GeoPackage) |
| `report` | openpyxl | Excel report export |
| `app` | streamlit, folium (+ `geo`, `report`) | the forester Streamlit app |
| `bart` | xgboost | the optional boosted-tree mean |
| `test` | pytest | running the test suite |

```bash
pip install "conifer-sae[geo,report]"      # forester I/O + Excel reports
pip install "conifer-sae[app]"             # everything the Streamlit app needs
```

## From source (development)

```bash
git clone https://github.com/IFC-UIDAHO/conifer
cd conifer
pip install -e ".[test]"
pytest
```

Verify the install:

```bash
python -c "import conifer; print(conifer.DiameterDistribution().di_overdispersion)"   # -> False (v0.2 default)
```
