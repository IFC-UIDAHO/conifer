"""
composition.py — compositional algebra for DBH-class proportions.

The response in the PSAE deck (slide 5) is a vector of DBH-class proportions
modelled on the Additive Log-Ratio (ALR) scale to guarantee compositional
consistency (proportions in (0,1), summing to 1):

    y_ij = log(p_ij / p_iK),   j = 1..K-1   (K = reference class)

So a K-class composition maps to a q = K-1 dimensional unconstrained ALR vector
on which the multivariate Fay-Herriot model is fit; predictions are mapped back
with the inverse-ALR (softmax with reference).
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-6


def dbh_class_edges(d_min: float = 0.0, d_max: float = 80.0, n_classes: int = 6) -> np.ndarray:
    """Equal-width DBH class edges (cm). Returns length n_classes+1 array."""
    return np.linspace(d_min, d_max, n_classes + 1)


def proportions_from_counts(counts: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Normalize nonnegative counts to proportions with a small floor to avoid
    log(0) on the ALR scale. Works on 1D (single comp) or 2D (rows = areas)."""
    counts = np.asarray(counts, dtype=float)
    counts = np.clip(counts, eps, None)
    return counts / counts.sum(axis=-1, keepdims=True)


def alr(p: np.ndarray, ref: int = -1, eps: float = _EPS) -> np.ndarray:
    """Additive Log-Ratio transform. p: (..., K) proportions -> (..., K-1).

    ref selects the reference class (default last). Returns ALR coords with the
    reference column removed.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    K = p.shape[-1]
    ref = ref % K
    logp = np.log(p)
    y = logp - logp[..., [ref]]
    keep = [k for k in range(K) if k != ref]
    return y[..., keep]


def alr_inv(y: np.ndarray, ref: int = -1) -> np.ndarray:
    """Inverse-ALR (softmax with a fixed 0 for the reference class).

    y: (..., K-1) -> (..., K) proportions summing to 1.
    """
    y = np.asarray(y, dtype=float)
    q = y.shape[-1]
    K = q + 1
    ref = ref % K
    # rebuild full log-ratio vector with 0 in the reference slot
    shape = y.shape[:-1] + (K,)
    full = np.zeros(shape)
    keep = [k for k in range(K) if k != ref]
    full[..., keep] = y
    full[..., ref] = 0.0
    full = full - full.max(axis=-1, keepdims=True)  # stabilize
    ex = np.exp(full)
    return ex / ex.sum(axis=-1, keepdims=True)


def class_midpoints(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])
