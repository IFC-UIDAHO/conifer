"""data.py — synthetic stand-level data generator + real-data loader stub.

Design (faithful to deck slide 6): a FIXED nonlinear structural map g(x) drives
the true ALR means; stand "type" is defined by the LOCAL nonlinearity magnitude
||NL(x_i)|| (even_aged = low curvature, mixed_conifer = high), so a LINEAR
Fay-Herriot is biased exactly in complex stands -- the mechanism behind slide 7.

The ML synthetic mean is trained on a UNIT-LEVEL inventory (X_train, Y_train);
the m target stands carry few-plot direct estimates Y_i ~ N(theta_i, D_i) with a
KNOWN Monte-Carlo sampling covariance D_i (the FH assumption).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .composition import alr, alr_inv, dbh_class_edges
from .sampling_cov import _nearest_pd

COVARIATE_NAMES = ["VCI", "UCI", "CanopyRelief", "FHD", "NDVI", "Slope", "TWI", "ClimateNA"]
STAND_TYPES = ["even_aged", "legacy", "mixed_conifer"]   # ordered by curvature


@dataclass
class StandData:
    X: np.ndarray
    Y: np.ndarray
    D: np.ndarray
    theta_true: np.ndarray
    P_true: np.ndarray
    types: np.ndarray
    edges: np.ndarray
    X_train: np.ndarray = None      # unit-level covariates for ML mean
    Y_train: np.ndarray = None      # unit-level ALR responses for ML mean
    cov_names: list = field(default_factory=lambda: list(COVARIATE_NAMES))

    @property
    def m(self): return self.X.shape[0]
    @property
    def p(self): return self.X.shape[1]
    @property
    def q(self): return self.Y.shape[1]
    @property
    def K(self): return self.P_true.shape[1]


class _StructuralMap:
    """Fixed g(x) = linear + SMOOTH nonlinear (quadratic + interactions), shared by
    truth and (unknown to) the models. Smooth => RF can learn it; curved => a
    LINEAR Fay-Herriot cannot, so it is biased where curvature is large."""
    def __init__(self, p, q, rng):
        self.p = p
        self.B_lin = rng.normal(scale=0.5, size=(p, q))
        self.Qq = rng.normal(scale=0.22, size=(p, q))          # quadratic weights
        self._pairs = [(a, b) for a in range(p) for b in range(a + 1, p)]
        self.Qc = rng.normal(scale=0.18, size=(len(self._pairs), q))  # interactions

    def nonlinear(self, X):
        quad = (X ** 2 - 1.0) @ self.Qq
        cross = np.stack([X[:, a] * X[:, b] for (a, b) in self._pairs], axis=1)
        return quad + cross @ self.Qc

    def g(self, X):
        return X @ self.B_lin + self.nonlinear(X)


def _mc_sampling_cov(theta_i, n_plots, q, rng, R=250, plot_sd=0.25, trees=(10, 30)):
    reps = np.empty((R, q))
    for r in range(R):
        pa = []
        for _ in range(n_plots):
            tp = theta_i + rng.normal(scale=plot_sd, size=q)
            nt = int(rng.integers(trees[0], trees[1] + 1))
            counts = rng.multinomial(nt, alr_inv(tp))
            pa.append(alr(counts + 0.5))
        reps[r] = np.mean(pa, axis=0)
    return _nearest_pd(np.atleast_2d(np.cov(reps, rowvar=False, ddof=1)), jitter=1e-9)


def generate_stand_data(m=180, p=8, n_classes=6, plots_per_stand=(1, 2),
                        trees_per_plot=(5, 15), sigma_u=0.30, seed=7,
                        n_train=4000, train_plot_sd=0.45, mc_reps=200):
    rng = np.random.default_rng(seed)
    K = n_classes; q = K - 1
    edges = dbh_class_edges(0.0, 80.0, K)
    gmap = _StructuralMap(p, q, rng)

    # ---- unit-level training inventory (clean-ish, many points) ----
    X_train = rng.normal(size=(n_train, p))
    Y_train = gmap.g(X_train) + rng.normal(scale=train_plot_sd, size=(n_train, q))

    # ---- target stands ----
    X = rng.normal(size=(m, p))
    NL = gmap.nonlinear(X)
    A = rng.normal(scale=0.4, size=(q, q))
    Sigma_u = sigma_u**2 * (np.eye(q) + 0.5 * (A @ A.T) / q)
    U = (np.linalg.cholesky(Sigma_u) @ rng.normal(size=(q, m))).T
    theta_true = gmap.g(X) + U
    P_true = alr_inv(theta_true)

    # stand type by local curvature magnitude (terciles of ||NL(x_i)||)
    curv = np.linalg.norm(NL, axis=1)
    order = np.argsort(curv)
    types = np.empty(m, dtype=object)
    thirds = np.array_split(order, 3)
    for lab, idx in zip(STAND_TYPES, thirds):
        types[idx] = lab
    types = types.astype(str)

    Y = np.empty((m, q)); D = np.empty((m, q, q))
    n_plots_arr = rng.integers(plots_per_stand[0], plots_per_stand[1] + 1, size=m)
    for i in range(m):
        D[i] = _mc_sampling_cov(theta_true[i], int(n_plots_arr[i]), q, rng,
                                R=mc_reps, trees=trees_per_plot)
        Y[i] = theta_true[i] + np.linalg.cholesky(D[i]) @ rng.standard_normal(q)

    return StandData(X=X, Y=Y, D=D, theta_true=theta_true, P_true=P_true,
                     types=types, edges=edges, X_train=X_train, Y_train=Y_train)


def load_real_data_stub():
    raise NotImplementedError("Plug your PSAE stand table here.")
