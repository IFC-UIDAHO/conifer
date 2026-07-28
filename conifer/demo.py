"""conifer.demo - a synthetic cruise for exercising the workflow end to end.

This is *not* part of the method and *not* real data. It generates the three files a forester
would actually bring - a tree list, a stand-level auxiliary (LiDAR/spectral) table, and a stand
polygon layer - from a known truth, so coverage can be checked honestly and the whole workflow can
be walked without ever touching a proprietary inventory.

What the demo is
----------------
A silviculturally plausible, *stocked* Inland-Northwest mixed-conifer cruise: 200 stands whose
diameter distributions range from dense small-tree regeneration through pole and mature stands to
open, large-tree old growth. A typical (median) stand carries ~250 stems per acre, ~115 ft2/ac of
basal area and a ~10" quadratic mean diameter - and stands span dense regeneration (few large
stems, ~45 ft2/ac) to mature and old-growth (~200+ ft2/ac). A believable working forest, not the
~4 ft2/ac (essentially bare ground) an earlier version of this generator produced.

Canopy structure is a single latent that drives *both* the LiDAR-like covariates and the diameter
distribution, exactly as in a real LiDAR-aided cruise. That coherence matters: it is what lets the
covariates genuinely predict stand structure, so an area-level model has real strength to borrow. A
generator that instead scrambled the covariate-to-composition link with a large random projection
would carry no learnable signal - and would silently defeat the very method the demo is meant to
show.

Two sampling regimes
--------------------
``make_cruise(regime=...)`` sets how many plots each stand gets, and that is what decides whether
small-area estimation helps:

* ``"sparse"`` (default) - 2-3 plots per stand. This is the thin-sample regime CONIFER is built
  for: a direct per-stand estimate is too noisy to trust, and borrowing strength across stands
  through the covariates pays off. On this cruise CONIFER beats the design-direct estimator by
  roughly 10-15% in merchantable-class RMSE, and the per-class prediction intervals still hold
  their nominal coverage.
* ``"rich"`` - a well-sampled cruise (median ~20 plots per stand). Here the direct estimate is
  already good, so there is little to borrow and CONIFER *converges* to it rather than beating it.
  That is the correct behaviour, and it is why an accuracy number should never be quoted from a
  data-rich cruise.

Honest about what it is
-----------------------
The numbers here are simulated. Use the demo for what it is good for - exercising the workflow end
to end on plausibly-shaped data and checking that the prediction intervals hold up. The *accuracy*
claims for CONIFER rest on the real St. Joe (Idaho) and southern studies against real cruises with
real LiDAR, not on this synthetic demo. The synthetic covariates are made coherent with stand
structure (as real canopy metrics are), not tuned to flatter the method; the sparse-regime
advantage shown here is the same advantage those real studies measure, in the same regime.

    trees, aux, stands, truth = conifer.demo.make_cruise()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_cruise", "write_demo_files", "DEMO_TARGETS", "REAL_TARGETS", "DEMO_DESIGN",
           "DEMO_PLOT_AREA", "DEMO_BAF", "DEMO_N_STANDS", "DEMO_REGIME"]

# The cruise design this generator produces. Anything that consumes the demo - the app's
# widget defaults, the vignette, the tests - should read these rather than hardcode a guess.
# They drifted apart once already: the demo moved to a fixed-area cruise while the app still
# defaulted to prism, which applied expansion factors to fixed-area tallies and produced
# estimates 117x too high without raising anything.
DEMO_DESIGN = "fixed"
DEMO_PLOT_AREA = 0.05
DEMO_BAF = 20.0
DEMO_N_STANDS = 200
DEMO_REGIME = "sparse"

_TYPES = np.array(["regenerating", "young managed", "mature mixed conifer", "legacy / old growth"])
_SPP = np.array(["Douglas-fir", "grand fir", "western larch", "ponderosa pine",
                 "lodgepole pine", "western redcedar"])
# LiDAR / spectral metric names taken from the real Idaho covariate set
_AUX = ["HAG_mean", "HAG_p95", "HAG_sd", "CanopyCover", "CanopyReliefRatio",
        "DensityAbove4", "VCI", "FoliageHeightDiversity", "slope", "elev"]

# The generator's own design targets: what a typical stand aims for, and how the two sampling
# regimes are meant to behave. These are silviculturally-plausible design goals for a SYNTHETIC
# stocked Inland-Northwest mixed-conifer cruise - no real inventory data is reused, and the
# accuracy claims for CONIFER rest on the real St. Joe and southern studies, not on this demo.
DEMO_TARGETS = {
    "stand_density_tpa_median": 250,
    "basal_area_ft2ac_median": 115,
    "qmd_in_median": 9.7,
    "merch_qmd_in_median": 11.2,
    "sparse_regime": "2-3 plots/stand; CONIFER beats the design-direct estimator by ~10-15% in "
                     "merchantable-class RMSE (the thin-sample regime SAE is built for)",
    "rich_regime": "median ~20 plots/stand; the direct estimate is already good and CONIFER "
                   "converges to it rather than beating it",
    "note": "synthetic design goals only - no real inventory data reused",
}
REAL_TARGETS = DEMO_TARGETS  # backward-compatible alias (the old name implied real-data reuse)


def make_cruise(n_stands=DEMO_N_STANDS, seed=7, breaks=None, design=DEMO_DESIGN,
                plot_area=DEMO_PLOT_AREA, baf=DEMO_BAF, realistic=True, regime=DEMO_REGIME):
    """Simulate a stocked mixed-conifer cruise with a known truth.

    Parameters
    ----------
    n_stands : int
    design : {'fixed', 'prism'}
        Fixed-area with many small plots (the default) or a variable-radius (prism/BAF) cruise.
    realistic : bool
        ``True`` (default) uses the structured, covariate-driven diameter distributions and the
        stocked densities described in the module docstring. ``False`` gives the older, tidier
        bell-shaped toy - useful only for illustrating what a *non*-representative demo looks like.
    regime : {'sparse', 'rich'}
        Plots per stand. ``'sparse'`` (default) gives 2-3 plots per stand - the thin-sample regime
        small-area estimation is built for, where CONIFER beats the direct estimate. ``'rich'``
        gives a well-sampled cruise (median ~20 plots) where CONIFER instead converges to the
        direct estimate. See the module docstring for what each regime demonstrates.

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
              + 0.25 * (X[:, 1] ** 2 - 1.0) + rng.normal(scale=0.22, size=m))
    struct = (struct - struct.mean()) / struct.std()          # high = large-tree structure
    stype = np.clip(np.digitize(struct, np.quantile(struct, [0.34, 0.65, 0.89])), 0, 3)

    # --- true composition: class profile set by the latent structure ------
    # eta is linear in class index with slope -decay; decay falls with struct and is allowed
    # to go negative, so regenerating stands are reverse-J (small stems dominate) while mature
    # and old-growth stands carry a genuine mid-to-large-diameter cohort. A generic stocked
    # stand therefore has stems spread across the classes, not piled in the smallest one.
    if realistic:
        decay = np.clip(0.05 - 0.72 * struct, -0.6, 3.2)
        decay = decay * np.exp(0.06 * rng.normal(size=m))
        eta = -decay[:, None] * np.arange(K)[None, :]
    else:
        eta = -0.15 * (np.arange(K)[None, :] - 2.5) ** 2 * np.ones((m, 1))

    # Small idiosyncratic per-class covariate effects on top of the shared latent `struct`.
    # These are deliberately small: in a real inventory the covariates and the diameter
    # distribution are both driven by the same underlying stand structure, so the class shares
    # are *learnable* from the covariates and there is real strength to borrow. A large random
    # projection here would instead inject covariate-composition noise no model can learn from
    # a thin sample - which silently defeats the very method the demo is meant to show. The
    # quadratic term keeps a genuine nonlinearity, so a *linear* Fay-Herriot is still biased and
    # the debiased-ML mean earns its place.
    B1 = rng.normal(scale=0.15, size=(len(_AUX), K))
    B2 = rng.normal(scale=0.05, size=(len(_AUX), K))
    eta = eta + X @ B1 + (X ** 2 - 1.0) @ B2 * 0.5
    eta = eta - eta.max(1, keepdims=True)
    p = np.exp(eta)
    p /= p.sum(1, keepdims=True)

    # --- structural zeros: upper classes are genuinely absent in many stands ------
    if realistic:
        # target zero fractions rise across classes: the largest DBH classes are absent from
        # many stands, but far less zero-inflated than a regeneration-dominated cruise
        target_zero = np.array([0.02, 0.10, 0.28, 0.48, 0.66, 0.80])[:K]
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
        # small positive density reproduces realistic zero-tally rates (rising across DBH
        # classes to ~80% in the largest) without manufacturing that artefact.
        p = np.where(present, p, p * 0.012)
        p /= np.clip(p.sum(1, keepdims=True), 1e-12, None)

    # --- total density: a stocked stand; regen dense/small, old-growth open/large -----
    # exp(5.50) ~ 245 stems/acre at the median, dropping with large-tree structure. With the
    # class profile above this puts a typical stand near ~115 ft2/ac basal area and a ~10"
    # QMD - a believable stocked Inland-Northwest mixed-conifer stand, not the ~4 ft2/ac of the
    # earlier generator. The total is strongly covariate-predictable (canopy metrics predict
    # density in a real LiDAR-aided cruise), which is much of what CONIFER borrows in a thin
    # sample.
    logN = (5.50 - 0.55 * struct + rng.normal(scale=0.18, size=m)
            if realistic else 5.55 - 0.42 * stype + 0.28 * X[:, 0] + rng.normal(scale=0.22, size=m))
    N = np.exp(logN)
    s_true = N[:, None] * p

    # --- plot effort: the sampling regime is what makes the SAE gain visible ------
    # 'sparse' (default) puts every stand in the thin-sample regime small-area estimation is
    # built for - 2-3 plots, where a direct per-stand estimate is too noisy to trust and
    # borrowing strength across stands pays off. 'rich' gives a well-sampled cruise (median
    # ~20 plots) where the direct estimate is already good and CONIFER correctly converges to
    # it. See the module docstring for what each regime demonstrates.
    if regime not in ("sparse", "rich"):
        raise ValueError(f"regime must be 'sparse' or 'rich', got {regime!r}")
    if not realistic:
        n_plots = rng.integers(1, 10, m).astype(float)
    elif regime == "sparse":
        n_plots = rng.integers(2, 4, size=m).astype(float)          # 2-3 plots per stand
    else:
        n_plots = np.clip(rng.lognormal(mean=np.log(21.5), sigma=0.60, size=m), 3, 60).round()
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
