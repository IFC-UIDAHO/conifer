"""conifer.demo — a realistic synthetic cruise, so the app has something to open on.

This is *not* part of the method. It generates the three files a forester would actually
bring — a tree list, a stand-level auxiliary table, and a stand polygon layer — from a
known truth, so that coverage can be checked honestly in demos and tests.

    trees, aux, stands, truth = conifer.demo.make_cruise()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_cruise", "write_demo_files"]

_TYPES = np.array(["even-aged plantation", "uneven-aged mixed conifer", "legacy / old growth"])
_SPP = np.array(["Douglas-fir", "grand fir", "western larch", "ponderosa pine", "western redcedar"])
_AUX = ["lidar_p95_ht", "lidar_cover", "lidar_rumple", "canopy_relief", "ndvi", "slope", "twi", "elev"]


def make_cruise(n_stands=180, seed=7, breaks=None, design="prism", baf=20.0,
                plot_area=0.1, min_plots=1, max_plots=9):
    """Simulate a cruise with a known truth.

    Returns
    -------
    trees : DataFrame   one row per tallied tree (STAND, PLOT, SPECIES, DBH_IN)
    aux   : DataFrame   one row per stand (STAND, STAND_TYPE, + 8 remote-sensing metrics)
    stands: DataFrame   one row per stand with a WKT polygon (write to .gpkg/.shp)
    truth : DataFrame   the true stems/acre by class — a real inventory never has this
    """
    from .io import DEFAULT_BREAKS_IN

    edges = np.asarray(DEFAULT_BREAKS_IN if breaks is None else breaks, float)
    K = len(edges) - 1
    mid = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(seed)
    m = n_stands

    # --- latent structure: covariates drive a nonlinear composition -------
    X = rng.normal(size=(m, len(_AUX)))
    stype = rng.integers(0, 3, m)
    B1 = rng.normal(scale=0.55, size=(len(_AUX), K - 1))
    B2 = rng.normal(scale=0.25, size=(len(_AUX), K - 1))
    eta = X @ B1 + (X ** 2 - 1.0) @ B2
    # stand type tilts the distribution: plantation -> small, legacy -> large
    tilt = np.array([-0.85, 0.0, 0.95])[stype][:, None] * np.linspace(-1.1, 1.4, K - 1)[None, :]
    eta = eta + tilt + rng.normal(scale=0.25, size=eta.shape)
    p = np.exp(np.column_stack([eta, np.zeros(m)]))
    p /= p.sum(1, keepdims=True)

    # total density: denser in plantations, sparser in legacy stands
    logN = 5.55 - 0.42 * stype + 0.28 * X[:, 0] + rng.normal(scale=0.22, size=m)
    N = np.exp(logN)                       # true stems per acre
    s_true = N[:, None] * p

    # --- the cruise: unequal plot effort across stands --------------------
    n_plots = rng.integers(min_plots, max_plots + 1, m)
    # thin stands are the realistic pain point: skew effort
    n_plots = np.where(rng.random(m) < 0.28, rng.integers(1, 3, m), n_plots)

    rows = []
    for i in range(m):
        for j in range(int(n_plots[i])):
            if design == "prism":
                # expected tally on a BAF plot ~ BA/acre / (mean tree BA factor)
                ba_ac = float(s_true[i] @ (0.005454 * mid ** 2))
                lam = max(ba_ac / baf, 0.4)
                ntree = rng.poisson(lam)
                if ntree == 0:
                    continue
                # prism selects with probability proportional to basal area
                w = s_true[i] * mid ** 2
                w = w / w.sum()
            else:
                lam = max(float(s_true[i].sum()) * plot_area, 0.4)
                ntree = rng.poisson(lam)
                if ntree == 0:
                    continue
                w = s_true[i] / s_true[i].sum()
            ks = rng.choice(K, size=ntree, p=w)
            dbh = rng.uniform(edges[ks], edges[ks + 1])
            spp = rng.choice(_SPP, size=ntree,
                             p=[0.38, 0.21, 0.16, 0.17, 0.08])
            for d, sp in zip(dbh, spp):
                rows.append((f"S{i+1:04d}", f"P{j+1}", sp, round(float(d), 1)))

    trees = pd.DataFrame(rows, columns=["STAND", "PLOT", "SPECIES", "DBH_IN"])

    # --- auxiliary table: noisy observations of the latent covariates -----
    Xobs = X + rng.normal(scale=0.18, size=X.shape)
    aux = pd.DataFrame(Xobs, columns=_AUX)
    aux.insert(0, "STAND", [f"S{i+1:04d}" for i in range(m)])
    aux.insert(1, "STAND_TYPE", _TYPES[stype])
    # give the metrics plausible ranges so the table reads like real LiDAR output
    aux["lidar_p95_ht"] = np.round(24 + 11 * aux["lidar_p95_ht"], 1).clip(4, 190)
    aux["lidar_cover"] = np.round(0.62 + 0.16 * aux["lidar_cover"], 3).clip(0.02, 1.0)
    aux["lidar_rumple"] = np.round(3.1 + 0.9 * aux["lidar_rumple"], 2).clip(1.0, None)
    aux["canopy_relief"] = np.round(0.48 + 0.09 * aux["canopy_relief"], 3).clip(0.02, 0.99)
    aux["ndvi"] = np.round(0.71 + 0.08 * aux["ndvi"], 3).clip(0.05, 0.99)
    aux["slope"] = np.round(22 + 11 * aux["slope"], 1).clip(0, 78)
    aux["twi"] = np.round(7.4 + 1.9 * aux["twi"], 2)
    aux["elev"] = np.round(1180 + 260 * aux["elev"], 0).clip(150, 3200)

    # --- stand polygons on a tidy grid (WGS84, somewhere in north Idaho) --
    cols = int(np.ceil(np.sqrt(m)))
    lon0, lat0, step = -116.35, 47.05, 0.0125
    polys, cx, cy = [], [], []
    for i in range(m):
        r, c = divmod(i, cols)
        x0, y0 = lon0 + c * step, lat0 + r * step
        jx, jy = 0.0018 * rng.standard_normal(2)
        pad = step * 0.055
        polys.append(
            f"POLYGON (({x0+pad+jx} {y0+pad+jy}, {x0+step-pad+jx} {y0+pad+jy}, "
            f"{x0+step-pad+jx} {y0+step-pad+jy}, {x0+pad+jx} {y0+step-pad+jy}, "
            f"{x0+pad+jx} {y0+pad+jy}))"
        )
        cx.append(x0 + step / 2)
        cy.append(y0 + step / 2)
    stands = pd.DataFrame({
        "STAND": [f"S{i+1:04d}" for i in range(m)],
        "STAND_TYPE": _TYPES[stype],
        "ACRES": np.round(rng.uniform(18, 140, m), 1),
        "lon": np.round(cx, 6), "lat": np.round(cy, 6),
        "geometry_wkt": polys,
    })

    truth = pd.DataFrame(s_true, columns=[f"{edges[k]:g}-{edges[k+1]:g} in" for k in range(K)])
    truth.insert(0, "STAND", [f"S{i+1:04d}" for i in range(m)])

    return trees, aux, stands, truth


def write_demo_files(outdir=".", **kwargs):
    """Write the demo cruise to disk as the three files a forester would upload."""
    import os

    trees, aux, stands, truth = make_cruise(**kwargs)
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for name, df in [("demo_treelist.csv", trees), ("demo_stand_metrics.csv", aux),
                     ("demo_truth.csv", truth)]:
        p = os.path.join(outdir, name)
        df.to_csv(p, index=False)
        paths[name] = p
    try:
        import geopandas as gpd
        from shapely import wkt

        g = gpd.GeoDataFrame(stands.drop(columns=["geometry_wkt"]),
                             geometry=stands["geometry_wkt"].apply(wkt.loads), crs="EPSG:4326")
        p = os.path.join(outdir, "demo_stands.gpkg")
        g.to_file(p, driver="GPKG", layer="stands")
        paths["demo_stands.gpkg"] = p
    except ImportError:
        p = os.path.join(outdir, "demo_stands.csv")
        stands.to_csv(p, index=False)
        paths["demo_stands.csv"] = p
    return paths
