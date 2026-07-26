"""
jsb_recovery.py — recover a 4-parameter Johnson's S_B diameter distribution from
recovered DBH-class proportions (deck slides 2 & 4: JSB beats Weibull in complex
Idaho mixed-conifer stands).

Johnson's S_B (bounded on (xi, xi+lambda)):
    z = gamma + delta * ln( (x - xi) / (xi + lambda - x) ),   z ~ N(0,1)
    F(x) = Phi(z)

We recover (gamma, delta, xi, lambda) per stand by least squares between the JSB
grouped CDF and the recovered class proportions. xi/lambda are initialized from
the class support and (optionally) refined; gamma/delta are the shape parameters
that capture skew and peakedness / multimodal bounding.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def johnson_sb_cdf(x, gamma, delta, xi, lam):
    x = np.asarray(x, float)
    u = (x - xi) / (xi + lam - x)
    u = np.clip(u, 1e-9, None)
    z = gamma + delta * np.log(u)
    return norm.cdf(z)


def _grouped_probs(edges, gamma, delta, xi, lam):
    F = johnson_sb_cdf(edges, gamma, delta, xi, lam)
    p = np.diff(F)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def recover_johnson_sb(proportions, edges, refine_bounds: bool = True):
    """Fit JSB to one stand's class proportions.

    proportions : (K,) recovered proportions
    edges       : (K+1,) DBH class edges
    returns dict(gamma, delta, xi, lam, sse, success)
    """
    proportions = np.asarray(proportions, float)
    proportions = proportions / proportions.sum()
    edges = np.asarray(edges, float)
    rng_ = edges[-1] - edges[0]
    buf = 0.02 * rng_

    def make_bounds(theta):
        if refine_bounds:
            gamma, delta, xi, lam = theta
        else:
            gamma, delta = theta
            xi = edges[0] - buf
            lam = rng_ + 2 * buf
        return gamma, delta, xi, lam

    def obj(theta):
        gamma, delta, xi, lam = make_bounds(theta)
        if delta <= 1e-3 or lam <= rng_ * 0.5:
            return 1e6
        # support must contain all edges strictly
        if xi >= edges[0] or (xi + lam) <= edges[-1]:
            return 1e6
        phat = _grouped_probs(edges, gamma, delta, xi, lam)
        return np.sum((phat - proportions) ** 2)

    if refine_bounds:
        x0 = np.array([0.0, 1.0, edges[0] - buf, rng_ + 2 * buf])
        bnds = [(-10, 10), (1e-2, 20),
                (edges[0] - 0.5 * rng_, edges[0] - 1e-3),
                (rng_ + 1e-3, 3 * rng_)]
    else:
        x0 = np.array([0.0, 1.0])
        bnds = [(-10, 10), (1e-2, 20)]

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bnds)
    gamma, delta, xi, lam = make_bounds(res.x)
    return dict(gamma=float(gamma), delta=float(delta), xi=float(xi),
                lam=float(lam), sse=float(res.fun), success=bool(res.success))
