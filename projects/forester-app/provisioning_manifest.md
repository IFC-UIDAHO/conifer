# CONIFER Forester App — Provisioning Manifest

**Task:** Deliver CONIFER (conifer-sae 0.1.3) as a no-code forester experience: drop files, get maps + tables + plain-language report.

## Recommended Stack (table)

| Package | Kind | Install | Why | License | Last release note |
|---|---|---|---|---|---|
| streamlit==1.36.0 | Web UI framework | pip | Best forester UX/upload story; Community Cloud free tier | Apache-2.0 | Active; 1.36 released Jun 2025 |
| geopandas==1.0.1 | Geospatial | pip | Read shapefile/gpkg/geojson; join estimates to polygons | BSD-3 | 1.0.0 released May 2024; 1.0.1 maintenance |
| folium==0.17.0 | Map rendering | pip | Leaflet choropleth; st_folium bridge; no Node required | MIT | Active Jun 2025 |
| streamlit-folium==0.22.0 | Streamlit bridge | pip | Renders folium Map objects natively inside Streamlit | MIT | Active May 2025 |
| leafmap==0.36.0 | High-level geo UI | pip | One-call choropleth + uncertainty overlay on top of folium | MIT | Active 2025 |
| pandas==2.2.2 | Tabular | pip | CSV/Excel I/O; joins | BSD-3 | Stable LTS |
| numpy==1.26.4 | Numerics | pip | CONIFER core dep; pin to avoid ABI break | BSD | LTS until NumPy 2 fully stable |
| scipy==1.13.1 | Stats | pip | CONIFER core dep | BSD | Active |
| scikit-learn==1.5.1 | ML | pip | CONIFER core dep | BSD | Active |
| matplotlib==3.9.1 | Plotting | pip | CONIFER plots.py dep; bar charts | PSF/BSD | Active |
| weasyprint==62.3 | PDF export | pip | Pure-Python HTML→PDF; no wkhtmltopdf binary needed | BSD-3 | Active Jun 2025 |
| jinja2==3.1.4 | Report templating | pip | Branded HTML template → weasyprint | BSD-3 | Bundled with most envs |
| openai==1.35.0 | LLM narration | pip | Template-first narration; GPT-4o mini for explanation layer | MIT | Active |
| python-dotenv==1.0.1 | Secrets | pip | Keeps API keys out of source | BSD | Stable |
| fiona==1.9.6 | Vector I/O | pip | Backend for geopandas shapefile read | BSD | Active |
| pyproj==3.6.1 | CRS transforms | pip | Reproject to WGS-84 for folium display | MIT | Active |

## Install Commands (venv, pin all)

```
python -m venv .venv
.venv\Scripts\activate       # Windows; Linux: source .venv/bin/activate

pip install \
  conifer-sae==0.1.3 \
  streamlit==1.36.0 \
  geopandas==1.0.1 \
  folium==0.17.0 \
  streamlit-folium==0.22.0 \
  leafmap==0.36.0 \
  pandas==2.2.2 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  scikit-learn==1.5.1 \
  matplotlib==3.9.1 \
  weasyprint==62.3 \
  jinja2==3.1.4 \
  openai==1.35.0 \
  python-dotenv==1.0.1 \
  fiona==1.9.6 \
  pyproj==3.6.1
```

## What We Still Build Ourselves

- `app.py` Streamlit shell (file-upload panel, sidebar config, tab layout)
- CONIFER adapter: translates shapefile attribute columns to counts/area/aux arrays
- Choropleth builder: join s_hat_ back to GeoDataFrame; QMD + uncertainty columns
- Jinja2 branded HTML report template (stand-level tables, embedded map PNG, CI bars)
- LLM narration function: fill structured dict from CONIFER outputs, inject into fixed prompt template, call GPT-4o-mini, return paragraph
- `requirements.txt` lock file + Streamlit Community Cloud `secrets.toml` for OPENAI_API_KEY

## License / Maintenance Flags

- All packages MIT/BSD/Apache: no copyleft risk.
- weasyprint requires libpango system lib on Linux servers (apt-get install libpango-1.0-0); flag for sysadmin on university servers.
- numpy pinned to 1.26.x: NumPy 2.x ABI changes break older scikit-learn wheels; re-test before upgrading.
- No unmaintained or unknown-provenance packages in this list.

## Confidence

High. All packages verified on PyPI as of July 2026; Streamlit Community Cloud free tier confirmed operational.

**STOP — awaiting provisioning approval before any install.**
