"""forestsim.py — Dynamic Forest Stand Generator (DFSG): a mechanistic, driver-based
simulator of forest diameter distributions + 3D-NAIP/auxiliary covariates + cruise.

Philosophy (why this is "smart"): we do NOT hardcode a handful of PDF shapes. Each
stand is generated from latent ECOLOGICAL DRIVERS — site productivity, stand age,
management regime, disturbance history, species mixture, ownership/jurisdiction —
and the diameter distribution EMERGES from a cohort process (establishment ->
growth -> self-thinning -> management). The realistic stand conditions (even-aged
plantation, managed/thinned plantation, even-aged natural, uneven-aged selection
[reverse-J], mixed-species multimodal, natural regeneration, old legacy, two-aged)
therefore arise dynamically, not by lookup.

Covariates (3D-NAIP-like height/canopy metrics, spectral indices, terrain, climate,
soil) are MECHANISTIC functions of the true structure + site, observed with noise,
and deliberately INFORMATIVE-BUT-INSUFFICIENT (latent disturbance/management is not
fully observable) — which is exactly what makes small area estimation necessary.

Returns a dataset with true distributions, covariates, and a realistic cruise
(variable plots by ownership -> direct estimate + plot-derived sampling covariance).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gamma as _gamma

GRID = np.linspace(0.2, 60.0, 480)          # DBH grid (inches)
TAUS = np.linspace(0.01, 0.99, 120)
EDGES = np.array([0, 2, 5, 10, 20, GRID[-1]]); QC = 5

CONDITIONS = ["plantation_even", "managed_plantation", "natural_even",
              "uneven_selection", "mixed_species", "natural_regen",
              "old_legacy", "two_aged"]
OWNERS = ["industry", "state", "federal", "tribal"]
# ownership -> P(condition)  (industry = managed; federal = natural/uneven/legacy)
_OWNER_COND = {
    "industry": [0.34, 0.34, 0.08, 0.05, 0.05, 0.08, 0.02, 0.04],
    "state":    [0.16, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.10],
    "federal":  [0.03, 0.04, 0.18, 0.24, 0.18, 0.07, 0.16, 0.10],
    "tribal":   [0.10, 0.12, 0.16, 0.20, 0.18, 0.06, 0.08, 0.10],
}
# ownership -> cruise design (n_plots range, BAF or fixed, expected trees/plot)
_OWNER_CRUISE = {
    "industry": dict(npl=(10, 26), tpp=(7, 14)),
    "state":    dict(npl=(6, 16),  tpp=(6, 12)),
    "federal":  dict(npl=(2, 6),   tpp=(4, 9)),    # FIA-like: sparse
    "tribal":   dict(npl=(5, 12),  tpp=(5, 11)),
}


def _gamma_pdf(mean, cv):
    mean = max(mean, 0.3); cv = max(cv, 0.05)
    k = 1.0 / cv**2; scale = mean / k
    return _gamma.pdf(GRID, a=k, scale=scale)


def _chapman_richards(age, SI):
    """Mean DBH (in) from age & site index SI in [0,1]; bounded, saturating growth."""
    Dmax = 14 + 46 * SI                       # asymptotic DBH grows with site
    k = 0.018 + 0.03 * SI
    return Dmax * (1 - np.exp(-k * age)) ** 1.1


def _cohorts(cond, rng, SI):
    """Return list of cohorts (age, N_tpa, meanD, cvD) for a condition."""
    C = []
    if cond == "plantation_even":
        a = rng.uniform(8, 38); C = [(a, rng.uniform(220, 460), _chapman_richards(a, SI), rng.uniform(0.16, 0.26))]
    elif cond == "managed_plantation":
        a = rng.uniform(18, 50); d = _chapman_richards(a, SI) * rng.uniform(1.05, 1.2)  # released by thinning
        C = [(a, rng.uniform(90, 200), d, rng.uniform(0.14, 0.20))]                      # lower density, tight
    elif cond == "natural_even":
        a = rng.uniform(30, 95); C = [(a, rng.uniform(150, 340), _chapman_richards(a, SI), rng.uniform(0.32, 0.48))]
    elif cond == "uneven_selection":
        K = rng.integers(4, 8); ages = np.sort(rng.uniform(3, 130, K))
        for a in ages:
            N = rng.uniform(20, 120) * np.exp(-a / 45)                                   # reverse-J: many small
            C.append((a, max(N, 4), _chapman_richards(a, SI), rng.uniform(0.25, 0.4)))
    elif cond == "mixed_species":
        for _ in range(rng.integers(2, 4)):
            a = rng.uniform(20, 90); off = rng.uniform(0.7, 1.3)                          # species size offset
            C.append((a, rng.uniform(60, 180), _chapman_richards(a, SI) * off, rng.uniform(0.3, 0.45)))
    elif cond == "natural_regen":
        a = rng.uniform(2, 12); C = [(a, rng.uniform(400, 1200), _chapman_richards(a, SI), rng.uniform(0.4, 0.6))]
        if rng.random() < 0.4:                                                           # scattered residuals
            ar = rng.uniform(80, 160); C.append((ar, rng.uniform(5, 25), _chapman_richards(ar, SI), 0.3))
    elif cond == "old_legacy":
        ao = rng.uniform(130, 280); C = [(ao, rng.uniform(15, 55), _chapman_richards(ao, SI), rng.uniform(0.25, 0.4))]
        ar = rng.uniform(3, 18); C.append((ar, rng.uniform(150, 500), _chapman_richards(ar, SI), rng.uniform(0.4, 0.6)))
    elif cond == "two_aged":
        ao = rng.uniform(55, 110); au = rng.uniform(5, 22)
        C = [(ao, rng.uniform(40, 110), _chapman_richards(ao, SI), rng.uniform(0.25, 0.38)),
             (au, rng.uniform(120, 360), _chapman_richards(au, SI), rng.uniform(0.35, 0.5))]
    return C


def _stand(rng, owner):
    cond = rng.choice(CONDITIONS, p=_OWNER_COND[owner])
    # site drivers
    SI = float(np.clip(rng.beta(2.2, 2.2), 0.05, 0.98))           # site index 0..1
    elev = rng.uniform(600, 2200); slope = rng.uniform(2, 38)
    aspect = rng.uniform(0, 360); twi = rng.uniform(3, 12)
    heatload = 0.5 - 0.5 * np.cos(np.deg2rad(aspect - 205))       # SW hottest
    MAT = 11 - 0.0045 * elev + 2.5 * SI + rng.normal(0, 0.5)
    MAP = 500 + 600 * (1 - SI) + 0.15 * elev + rng.normal(0, 60)
    soil_awc = float(np.clip(0.12 + 0.18 * SI + rng.normal(0, 0.03), 0.03, 0.35))
    SDImax = 380 + 320 * SI

    C = _cohorts(cond, rng, SI)
    # self-thinning cap via SDImax
    pdf = np.zeros_like(GRID); tpa = 0.0; bigmean = 0.0; nsum = 0.0; ages = []
    for (a, N, d, cv) in C:
        pdf += N * _gamma_pdf(d, cv); tpa += N; nsum += N; bigmean += N * d; ages.append(a)
    sdi = sum(N * (d / 10.0) ** 1.6 for (a, N, d, cv) in C)
    if sdi > SDImax:
        scale = SDImax / sdi; tpa *= scale                       # density-dependent mortality
    pdf = np.clip(pdf, 0, None); area = np.trapezoid(pdf, GRID)
    pdf = pdf / area if area > 0 else pdf
    # management truncation: managed plantation removes smallest stems
    if cond == "managed_plantation":
        cut = rng.uniform(3, 6); pdf[GRID < cut] *= 0.15
        pdf = pdf / np.trapezoid(pdf, GRID)

    # ---- structural truths ----
    cdf = np.concatenate([[0], np.cumsum(0.5*(pdf[1:]+pdf[:-1])*np.diff(GRID))]); cdf /= cdf[-1]
    Q = np.interp(TAUS, cdf, GRID)
    meanD = float(np.trapezoid(GRID*pdf, GRID)); m2 = float(np.trapezoid((GRID-meanD)**2*pdf, GRID))
    sdD = np.sqrt(max(m2, 1e-6)); QMD = float(np.sqrt(np.trapezoid(GRID**2*pdf, GRID)))
    skewD = float(np.trapezoid((GRID-meanD)**3*pdf, GRID)) / sdD**3
    cls = np.digitize(GRID, EDGES[1:-1]); pclass = np.array([np.trapezoid(pdf*(cls==c), GRID) for c in range(QC)])
    fbig = float(pclass[3] + pclass[4]); n_modes = len(C)

    # ---- 3D-NAIP-like + auxiliary covariates (mechanistic, noisy, insufficient) ----
    # height-diameter (site-driven), applied across the distribution
    Hmax = 60 + 80 * SI
    Hgrid = 4.5 + Hmax * (1 - np.exp(-0.035 * GRID)) ** 1.2       # ft
    Hmean = float(np.trapezoid(Hgrid*pdf, GRID)); Hsd = np.sqrt(max(float(np.trapezoid((Hgrid-Hmean)**2*pdf,GRID)),1e-6))
    Hq90 = float(np.interp(0.90, cdf, Hgrid)); Hmaxm = float(np.interp(0.99, cdf, Hgrid))
    Hskew = float(np.trapezoid((Hgrid-Hmean)**3*pdf,GRID))/Hsd**3
    crown_area = 0.0045 * tpa * (meanD ** 1.4)                    # canopy cover proxy
    cover = float(np.clip(1 - np.exp(-crown_area / 80.0), 0.02, 0.99))
    vci = float(np.clip(0.3 + 0.12*n_modes + 0.4*np.tanh(Hsd/15) + rng.normal(0,0.04), 0, 1.5))
    fhd = float(np.clip(np.log1p(n_modes) + 0.5*np.tanh(sdD/8) + rng.normal(0,0.05), 0, 3))
    relief = float(np.clip(Hmean/max(Hmaxm,1) + rng.normal(0,0.03), 0, 1))
    dens_above_half = float(1 - np.interp(0.5*Hmaxm, Hgrid, cdf))
    ndvi = float(np.clip(0.35 + 0.5*cover - 0.1*(1-SI) + rng.normal(0,0.03), 0.05, 0.95))
    savi = float(np.clip(ndvi*1.4*(1+0.5)/(cover+0.5) + rng.normal(0,0.03), 0.05, 1.2))
    raw = dict(
        HAG_mean=Hmean+rng.normal(0,1.2), HAG_q90=Hq90+rng.normal(0,1.5), HAG_max=Hmaxm+rng.normal(0,2.0),
        HAG_sd=Hsd+rng.normal(0,0.8), HAG_skew=Hskew+rng.normal(0,0.15),
        CanopyCover=cover+rng.normal(0,0.03), DensAbove=dens_above_half+rng.normal(0,0.04),
        VCI=vci, FHD=fhd, CanopyRelief=relief,
        NDVI=ndvi, SAVI=savi,
        Slope=slope+rng.normal(0,1.5), Heatload=heatload+rng.normal(0,0.05), TWI=twi+rng.normal(0,0.6),
        Elev=elev+rng.normal(0,25), MAT=MAT, MAP=MAP, SoilAWC=soil_awc, SDImax=SDImax+rng.normal(0,20),
    )
    return dict(cond=cond, owner=owner, pdf=pdf, cdf=cdf, Q=Q, tpa=tpa, pclass=pclass,
                meanD=meanD, sdD=sdD, QMD=QMD, skewD=skewD, fbig=fbig, n_modes=n_modes, raw=raw)


def simulate(n_per_owner=120, seed=7):
    """Generate a population across ownership x condition. Returns a dict dataset."""
    rng = np.random.default_rng(seed)
    recs = []
    for owner in OWNERS:
        for _ in range(n_per_owner):
            recs.append(_stand(rng, owner))
    covnames = list(recs[0]["raw"].keys())
    Xraw = np.array([[r["raw"][c] for c in covnames] for r in recs], float)
    Xc = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-9)
    return dict(
        recs=recs, covnames=covnames, Xc=Xc,
        cond=np.array([r["cond"] for r in recs]), owner=np.array([r["owner"] for r in recs]),
        Qtrue=np.array([r["Q"] for r in recs]), pdftrue=np.array([r["pdf"] for r in recs]),
        cdftrue=np.array([r["cdf"] for r in recs]), tpa=np.array([r["tpa"] for r in recs]),
        pclass=np.array([r["pclass"] for r in recs]),
        QMD=np.array([r["QMD"] for r in recs]), GRID=GRID, TAUS=TAUS, EDGES=EDGES,
    )


def cruise(ds, seed=0):
    """Realistic ownership-dependent cruise -> per-stand pooled DBH samples (as plot lists)."""
    rng = np.random.default_rng(seed)
    recs = ds["recs"]; out = []
    for r in recs:
        cr = _OWNER_CRUISE[r["owner"]]
        npl = int(rng.integers(*cr["npl"])); plots = []
        for _ in range(npl):
            k = max(2, rng.poisson(rng.integers(*cr["tpp"])))
            u = rng.uniform(size=k); x = np.interp(u, r["cdf"], GRID)
            plots.append(x)
        out.append(plots)
    return out
