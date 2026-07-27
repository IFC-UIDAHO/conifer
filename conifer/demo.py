"""conifer.demo - a synthetic cruise calibrated to look like a real inventory.

This is *not* part of the method. It generates the three files a forester would actually
bring - a tree list, a stand-level auxiliary table, and a stand polygon layer - from a
known truth, so coverage can be checked honestly in demos, tests and the vignette.

Calibrated, not copied
----------------------
The generator is tuned to reproduce the *shape* of the St. Joe (Idaho) cruise used to
develop CONIFER. **No real data is redistributed** - only summary characteristics were used
to set the parameters below, and every tree here is simulated. Targets, from 454 real
stands:

===========================================  ==================  ==============
characteristic                               real (Idaho, 454)   this generator
===========================================  ==================  ==============
plots per stand (median / q25 / q75)         21 / 14 / 32        18 / 12 / 30
trees tallied per stand (median)             88                  69
trees per plot (median)                      3.9                 3.7
stems per acre (median)                      19.2                17.0
stem share in the smallest DBH class         81%                 88%
merchantable QMD (median), inches            8.0                 7.0
stands tallying zero stems, by class         4/39/65/77/85/89 %  ~4/38/64/76/84/88 %
===========================================  ==================  ==============

Two of those matter more than they look. Real stands are **regeneration-dominated** - the
diameter distribution is a steep reverse-J, not the tidy bell a toy generator produces - and
the upper classes are **heavily zero-inflated**, which is the sparse, structural-zero regime
CONIFER is built for. A demo without those two features flatters the method and, worse,
makes its prediction intervals look far wider than they are in practice, because it also
understates how many plots a real stand actually gets.

What this demo does NOT show
----------------------------
**On this generator CONIFER does not beat the direct estimator** - it is 10-20% worse by RMSE
across plot efforts from 3 to 21 plots per stand. That is a real measurement, reported here
rather than tuned away, and it is worth understanding before quoting any accuracy number.

Two reasons, both about the generator rather than the method:

1. *A well-sampled cruise does not need small-area estimation.* Matching the real Idaho plot
   effort (median ~20 plots, ~80 stems tallied per stand) puts the demo in the data-rich
   regime, where the direct estimate is already good and the correct behaviour for a
   Fay-Herriot estimator is to converge to it, not to beat it. CONIFER's gain is in genuinely
   thin samples.
2. *The synthetic covariates carry less signal than real LiDAR.* Measured as cross-validated
   R-squared of covariates against class share, the real Idaho metrics reach 0.58/0.52/0.53
   on the first three classes; this generator reaches roughly 0.48/0.34/0.0. Less signal
   means less to borrow, which is precisely what an area-level model trades on.

So use this demo for what it is good for - exercising the **workflow** end to end on data
shaped like the real thing, and checking that the prediction intervals hold up (they do:
measured per-class coverage runs 0.90-0.94 against a nominal 0.90). Do **not** cite it as
evidence of accuracy. The accuracy claim rests on the St. Joe (Idaho) and Arkansas studies
against real cruises, where the covariates are real and the merchantable tallies are thin.

    trees, aux, stands, truth = conifer.demo.make_cruise()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_cruise", "write_demo_files", "REAL_TARGETS"]

_TYPES = np.array(["regenerating", "young managed", "mature mixed conifer", "legacy / old growth"])
_SPP = np.array(["Douglas-fir", "grand fir", "western larch", "ponderosa pine",
                 "lodgepole pine", "western redcedar"])
# LiDAR / spectral metric names taken from the real Idaho covariate set
_AUX = ["HAG_mean", "HAG_p95", "HAG_sd", "CanopyCover", "CanopyReliefRatio",
        "DensityAbove4", "VCI", "FoliageHeightDiversity", "slope", "elev"]

REAL_TARGETS = {
    "plots_per_stand_median": 21, "plots_per_stand_q25": 14, "plots_per_stand_q75": 32,
    "trees_per_stand_median": 88, "trees_per_plot_median": 3.9,
    "smallest_class_share": 0.81, "largest_class_zero_frac": 0.89,
    "merch_qmd_median_in": 8.0,
    "source": "454 stands, St. Joe (Idaho) cruise - summary characteristics only, no data reused",
}


def make_cruise(n_stands=200, seed=7, breaks=None, design="fixed",
                plot_area=0.2, baf=20.0, realistic=True):
    """Simulate a cruise with a known truth, shaped like a real inventory.

    Parameters
    ----------
    n_stands : int
    design : {'fixed', 'prism'}
        Real Idaho cruising is fixed-area with many small plots per stand, which is the
        default. ``'prism'`` gives a variable-radius cruise instead.
    realistic : bool
        ``True`` (default) uses the reverse-J, zero-inflated, many-plots structure measured
        from the real cruise. ``False`` gives the older, tidier bell-shaped toy - useful only
        for illustrating what a *non*-representative demo looks like.

    Returns
    -------
    trees : DataFrame   one row per tallied tree (STAND, PLOT, SPECIES, DBH_IN)
    aux   : DataFrame   one row per stand (STAND, STAND_TYPE, + LiDAR metrics)
    stands: DataFrame   one row per stand with a WKT polygon
    truth : DataFrame   TRUE stems/acre by class - a real inventory never has this
    """
    from .io import DEFAULT_BREAKS_IN

    edges = np.asarray(DEFAULT_BREAKS_IN if breaks is None else breaks, float)
    K = len(edges) - 1
    mid = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(seed)
    m = n_stands

    # --- latent structure, driven by the covariates (as in a real LiDAR-aided cruise) ---
    # In reality canopy height and cover *predict* the diameter distribution: a short, dense,
    # low-relief stand is regenerating; a tall, open, high-relief one carries large stems. The
    # generator must reflect that, otherwise the auxiliary data carries no signal and there is
    # nothing for small-area estimation to borrow. `STAND_TYPE` is then a coarse *label* of the
    # same latent structure - which is what a stand-type attribute really is - not an
    # independent driver hidden from the covariates.
    X = rng.normal(size=(m, len(_AUX)))
    struct = (1.05 * X[:, 0] + 0.70 * X[:, 3] + 0.45 * X[:, 4] + 0.30 * X[:, 5]
              + 0.25 * (X[:, 1] ** 2 - 1.0) + rng.normal(scale=0.28, size=m))
    struct = (struct - struct.mean()) / struct.std()          # high = large-tree structure
    stype = np.clip(np.digitize(struct, np.quantile(struct, [0.34, 0.65, 0.89])), 0, 3)

    # --- true composition: steep reverse-J, decay set by the latent structure ------
    # A negative-exponential in class index reproduces the 81/11/5/2/1/0.3 % stem profile.
    if realistic:
        decay = np.clip(1.30 - 0.68 * struct, 0.28, 3.2)
        decay = decay * np.exp(0.08 * rng.normal(size=m))
        eta = -decay[:, None] * np.arange(K)[None, :]
    else:
        eta = -0.15 * (np.arange(K)[None, :] - 2.5) ** 2 * np.ones((m, 1))

    # a further nonlinear covariate effect, so a *linear* Fay-Herriot is genuinely biased
    B1 = rng.normal(scale=0.34, size=(len(_AUX), K))
    B2 = rng.normal(scale=0.15, size=(len(_AUX), K))
    eta = eta + X @ B1 + (X ** 2 - 1.0) @ B2 * 0.5
    eta = eta - eta.max(1, keepdims=True)
    p = np.exp(eta)
    p /= p.sum(1, keepdims=True)

    # --- structural zeros: upper classes are genuinely absent in many stands ------
    if realistic:
        # target zero fractions rise across classes: ~4/39/65/77/85/89 %
        target_zero = np.array([0.04, 0.39, 0.65, 0.77, 0.85, 0.89])[:K]
        # a stand's chance of carrying class k falls with its regeneration dominance
        rank = np.argsort(np.argsort(-struct))         # 0 = most large-tree structure
        q = rank / max(m - 1, 1)
        present = np.zeros((m, K), bool)
        for k in range(K):
            present[:, k] = q < (1 - target_zero[k])
        present[:, 0] = True
        # Suppress, do not annihilate. In a real forest a diameter class is rarely *exactly*
        # absent - it is present at a density low enough that a finite cruise tallies no stems.
        # Setting the truth to a hard zero would make the direct estimate correct by
        # construction wherever nothing was tallied, and would penalise any estimator that
        # smooths - an artefact of the generator, not a property of forests. Suppressing to a
        # small positive density reproduces the observed zero-tally rates (39-89% by class)
        # without manufacturing that artefact.
        p = np.where(present, p, p * 0.012)
        p /= np.clip(p.sum(1, keepdims=True), 1e-12, None)

    # --- total density: regen stands are dense, legacy stands are not ------------
    logN = (3.05 - 0.26 * struct + rng.normal(scale=0.40, size=m)
            if realistic else 5.55 - 0.42 * stype + 0.28 * X[:, 0] + rng.normal(scale=0.22, size=m))
    N = np.exp(logN)
    s_true = N[:, None] * p

    # --- plot effort: right-skewed, median ~20, a long tail, occasionally 1 ------
    if realistic:
        n_plots = np.clip(rng.lognormal(mean=np.log(21.5), sigma=0.62, size=m), 1, 60).round()
    else:
        n_plots = rng.integers(1, 10, m).astype(float)
    n_plots = n_plots.astype(int)

    rows = []
    for i in range(m):
        for j in range(n_plots[i]):
            if design == "prism":
                ba_ac = float(s_true[i] @ (0.005454 * mid ** 2))
                lam = max(ba_ac / baf, 0.3)
                w = s_true[i] * mid ** 2
            else:
                lam = max(float(s_true[i].sum()) * plot_area, 0.3)
                w = s_true[i].copy()
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            ntree = rng.poisson(lam)
            if ntree == 0:
                continue
            ks = rng.choice(K, size=ntree, p=w)
            dbh = rng.uniform(edges[ks], edges[ks + 1])
            spp = rng.choice(_SPP, size=ntree, p=[0.31, 0.18, 0.14, 0.15, 0.14, 0.08])
            for dd, sp in zip(dbh, spp):
                rows.append((f"S{i+1:04d}", f"P{j+1}", sp, round(float(dd), 1)))

    trees = pd.DataFrame(rows, columns=["STAND", "PLOT", "SPECIES", "DBH_IN"])

    # --- auxiliary table: noisy observations, on plausible LiDAR scales ----------
    Xobs = X + rng.normal(scale=0.16, size=X.shape)
    aux = pd.DataFrame(Xobs, columns=_AUX)
    aux.insert(0, "STAND", [f"S{i+1:04d}" for i in range(m)])
    aux.insert(1, "STAND_TYPE", _TYPES[stype])
    aux["HAG_mean"] = np.round(9.5 + 5.5 * aux["HAG_mean"] - 1.7 * stype, 2).clip(0.3, None)
    aux["HAG_p95"] = np.round(aux["HAG_mean"] * (2.6 + 0.25 * rng.normal(size=m)), 2).clip(1, 220)
    aux["HAG_sd"] = np.round(2.6 + 0.9 * np.abs(aux["HAG_sd"]), 2)
    aux["CanopyCover"] = np.round(0.42 + 0.16 * aux["CanopyCover"], 3).clip(0.01, 0.99)
    aux["CanopyReliefRatio"] = np.round(0.47 + 0.07 * aux["CanopyReliefRatio"], 3).clip(0.05, 0.95)
    aux["DensityAbove4"] = np.round(0.06 + 0.03 * aux["DensityAbove4"], 4).clip(0.0001, None)
    aux["VCI"] = np.round(0.95 + 0.14 * aux["VCI"], 3).clip(0.05, None)
    aux["FoliageHeightDiversity"] = np.round(0.95 + 0.14 * aux["FoliageHeightDiversity"], 3).clip(0.05, None)
    aux["slope"] = np.round(24 + 12 * aux["slope"], 1).clip(0, 78)
    aux["elev"] = np.round(1180 + 250 * aux["elev"], 0).clip(150, 3200)

    # --- stand polygons on a tidy grid (WGS84, north Idaho) ----------------------
    cols = int(np.ceil(np.sqrt(m)))
    lon0, lat0, step = -116.35, 47.05, 0.0125
    polys, cx, cy = [], [], []
    for i in range(m):
        r, c = divmod(i, cols)
        x0, y0 = lon0 + c * step, lat0 + r * step
        jx, jy = 0.0016 * rng.standard_normal(2)
        pad = step * 0.055
        polys.append(
            f"POLYGON (({x0+pad+jx} {y0+pad+jy}, {x0+step-pad+jx} {y0+pad+jy}, "
            f"{x0+step-pad+jx} {y0+step-pad+jy}, {x0+pad+jx} {y0+step-pad+jy}, "
            f"{x0+pad+jx} {y0+pad+jy}))")
        cx.append(x0 + step / 2)
        cy.append(y0 + step / 2)
    stands = pd.DataFrame({
        "STAND": [f"S{i+1:04d}" for i in range(m)],
        "STAND_TYPE": _TYPES[stype],
        "ACRES": np.round(rng.uniform(15, 160, m), 1),
        "lon": np.round(cx, 6), "lat": np.round(cy, 6),
        "geometry_wkt": polys,
    })

    truth = pd.DataFrame(s_true, columns=[f"{edges[k]:g}-{edges[k+1]:g} in" for k in range(K)])
    truth.insert(0, "STAND", [f"S{i+1:04d}" for i in range(m)])
    return trees, aux, stands, truth


def write_demo_files(outdir=".", **kwargs):
    """Write the demo cruise to disk as the files a forester would upload."""
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
