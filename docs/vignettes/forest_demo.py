"""forest_demo.py — data + display helpers for the CONIFER getting-started vignette.

None of this is part of CONIFER. It only (a) synthesises a small, reproducible forest
inventory with a KNOWN truth so the vignette can check coverage honestly, and (b) formats
CONIFER's outputs as tidy tables so the notebook cells stay one-liners. The full source is
printed in Appendix B of the vignette, so nothing here is hidden — just kept out of the way.

Units follow US inventory convention: DBH (diameter at breast height) in INCHES; stem
density in stems per ACRE.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Six 4-inch DBH classes from 1" to 25".
DBH_EDGES = np.array([1, 5, 9, 13, 17, 21, 25], dtype=float)
COVARIATES = ["VCI", "UCI", "CanopyRelief", "FHD", "NDVI", "Slope", "TWI", "ClimateNA"]
STAND_TYPES = np.array(["even_aged", "legacy", "mixed_conifer"])


def _clabels(K):
    return [f"C{k+1}" for k in range(K)]


def dbh_class_table(edges=DBH_EDGES):
    """What C1..CK mean: each DBH class with its diameter range and midpoint (inches)."""
    K = len(edges) - 1
    return pd.DataFrame(
        {
            "DBH range (in)": [f"{int(edges[k])}–{int(edges[k+1])}" for k in range(K)],
            "midpoint (in)": [(edges[k] + edges[k + 1]) / 2 for k in range(K)],
        },
        index=_clabels(K),
    ).rename_axis("class")


def make_inventory(m=300, seed=11, edges=DBH_EDGES):
    """A reproducible synthetic inventory with a KNOWN truth.

    A fixed nonlinear map g(x) = linear + quadratic + interactions drives the true DBH-class
    composition, so a *linear* Fay-Herriot is biased precisely where stand structure is
    curved. Stand 'type' is the curvature tercile (used as a Mondrian group). Returns the four
    inputs CONIFER needs plus the hidden truth `s_true` (stems/acre by class), which a real
    inventory never has and which we use only to check coverage.

    Returns a dict with:
      counts   : (m, K) float   DBH-class tallies per stand   -> CONIFER input
      area_eff : (m,)   float   effective sampled area, acres -> CONIFER input
      X        : (m, p) float   auxiliary covariates          -> CONIFER input
      groups   : (m,)   int     stand-type label 0/1/2        -> optional CONIFER input
      s_true   : (m, K) float   TRUE stems/acre by class      -> only for the coverage check
    """
    rng = np.random.default_rng(seed)
    p = len(COVARIATES)
    K = len(edges) - 1
    X = rng.normal(size=(m, p))

    # fixed structural map on ALR(ref=last) coordinates -> (m, K-1)
    B_lin = rng.normal(scale=0.5, size=(p, K - 1))
    B_quad = rng.normal(scale=0.22, size=(p, K - 1))
    pairs = [(a, b) for a in range(p) for b in range(a + 1, p)]
    B_int = rng.normal(scale=0.18, size=(len(pairs), K - 1))

    def nonlinear(Z):
        quad = (Z ** 2 - 1.0) @ B_quad
        cross = np.stack([Z[:, a] * Z[:, b] for (a, b) in pairs], axis=1) @ B_int
        return quad + cross

    theta_true = X @ B_lin + nonlinear(X) + 0.30 * rng.normal(size=(m, K - 1))

    # inverse additive-log-ratio -> true class shares on the simplex
    ext = np.concatenate([theta_true, np.zeros((m, 1))], axis=1)
    P_true = np.exp(ext - ext.max(1, keepdims=True))
    P_true /= P_true.sum(1, keepdims=True)

    # stand type = curvature tercile -> Mondrian group 0/1/2
    curv = np.linalg.norm(nonlinear(X), axis=1)
    order = np.argsort(curv)
    groups = np.empty(m, int)
    for g, idx in enumerate(np.array_split(order, 3)):
        groups[idx] = g

    N_true = np.exp(rng.normal(np.log(250), 0.25, size=m))        # total stems/acre
    s_true = N_true[:, None] * P_true                            # true stems/acre by class
    area_eff = rng.uniform(0.6, 1.6, size=m)                     # effective acres sampled
    counts = rng.poisson(area_eff[:, None] * s_true).astype(float)  # observed class tallies

    return dict(counts=counts, area_eff=area_eff, X=X, groups=groups,
                s_true=s_true, edges=edges, names=COVARIATES,
                type_names=STAND_TYPES, K=K, m=m)


def preview_inputs(D, n=6):
    """The estimator's inputs for the first n stands: type, effective area, DBH-class counts."""
    df = pd.DataFrame(D["counts"][:n].astype(int), columns=_clabels(D["K"]))
    df.insert(0, "area_ac", np.round(D["area_eff"][:n], 2))
    df.insert(0, "type", D["type_names"][D["groups"][:n]])
    return df.rename_axis("stand")


def covariate_preview(D, n=6):
    """The auxiliary covariates X for the first n stands (standardized LiDAR/spectral metrics)."""
    return pd.DataFrame(np.round(D["X"][:n], 2), columns=D["names"]).rename_axis("stand")


def describe_outputs(est):
    """Everything a fitted DiameterDistribution hands back — the pieces you'll actually use."""
    rows = [
        ("s_hat_", "stems/acre by DBH class — the estimate"),
        ("N_hat_", "total stems/acre per stand"),
        ("p_hat_", "class shares (composition), rows sum to 1"),
        ("s_var_", "variance of s_hat_ (delta method)"),
        ("Su_", "estimated area random-effect covariance"),
        ("mse_theta_", "2nd-order MSE on the log-ratio scale"),
    ]
    data = []
    for name, desc in rows:
        val = getattr(est, name, None)
        shape = "—" if val is None else str(np.asarray(val).shape)
        data.append({"attribute": name, "shape": shape, "what it is": desc})
    return pd.DataFrame(data).set_index("attribute")


def quadratic_mean_diameter(est, edges=DBH_EDGES):
    """QMD (inches) per stand from the class midpoints — the diameter of the tree of mean basal area."""
    mids = (edges[:-1] + edges[1:]) / 2.0
    return np.sqrt((est.s_hat_ * mids ** 2).sum(1) / np.clip(est.s_hat_.sum(1), 1e-9, None))


def stand_estimate(est, D, stands=range(5)):
    """Per-stand estimate: stems/acre by class, total N, and QMD (inches)."""
    rows = list(stands)
    qmd = quadratic_mean_diameter(est, D["edges"])
    df = pd.DataFrame(np.round(est.s_hat_[rows], 1), columns=_clabels(D["K"]), index=rows)
    df.insert(0, "type", D["type_names"][D["groups"][rows]])
    df["N/ac"] = np.round(est.N_hat_[rows], 0)
    df["QMD_in"] = np.round(qmd[rows], 1)
    return df.rename_axis("stand")


def interval_table(est, lo, hi, D, stand, label="band"):
    """A single stand's estimate with a [lo, hi] band, per DBH class."""
    return pd.DataFrame(
        {
            "estimate": np.round(est.s_hat_[stand], 1),
            f"{label}_lo": np.round(np.clip(lo[stand], 0, None), 1),
            f"{label}_hi": np.round(hi[stand], 1),
        },
        index=_clabels(D["K"]),
    ).rename_axis("class")


def show(x, labels=None, name="value"):
    """Tidy view of a result: a short labelled vector -> Series; a long one -> a summary."""
    x = np.asarray(x, float)
    if x.ndim == 1 and x.size <= 12:
        return pd.Series(np.round(x, 2), index=labels, name=name)
    return pd.Series(x.ravel()).describe().round(2).to_frame(name)


def coverage_report(D, est, reps=50, alpha=0.10):
    """Honest repeated-split coverage for the three conformal set types: the fraction of
    held-out stands whose TRUE stem density falls inside the set, averaged over `reps` random
    calibration/test splits. Nominal coverage is 1 - alpha."""
    m = D["m"]
    out = {}

    def run(kind):
        c = []
        for r in range(reps):
            pp = np.random.default_rng(1000 + r).permutation(m)
            a, t = pp[: m // 2], pp[m // 2:]
            if kind == "per-class marginal":
                est.conformalize(D["s_true"], a, joint=False, alpha=alpha)
                lo, hi = est.predict_interval(joint=False)
                c.append(np.mean((D["s_true"][t] >= lo[t]) & (D["s_true"][t] <= hi[t])))
            elif kind == "joint L-inf band":
                est.conformalize(D["s_true"], a, joint=True, mode="maxscore", alpha=alpha)
                c.append(est.joint_covered(D["s_true"], t).mean())
            else:  # joint min-volume ellipsoid (Aitchison geometry)
                est.conformalize(D["s_true"], a, joint=True, mode="min_vol", geom="ilr", alpha=alpha)
                c.append(est.joint_covered(D["s_true"], t).mean())
        return round(float(np.mean(c)), 3), round(float(np.std(c)), 3)

    for kind in ["per-class marginal", "joint L-inf band", "joint min-volume ellipsoid"]:
        out[kind] = run(kind)
    df = pd.DataFrame(out, index=["coverage", "sd"]).T
    df["nominal"] = 1 - alpha
    return df


def benchmark_table(est, totals, D):
    """Population means before vs after benchmarking to design totals (the coherence check)."""
    return pd.DataFrame(
        {
            "target": np.round(totals, 1),
            "model_before": np.round(est.s_hat_.mean(0), 1),
            "benchmarked_after": np.round(est.s_bm_.mean(0), 1),
        },
        index=_clabels(D["K"]),
    ).rename_axis("class")
