"""path_b_bayes_rnnfh.py — joint Bayesian multivariate Random-Weight-NN Fay-Herriot."""
from __future__ import annotations
import numpy as np
from scipy.stats import invwishart

from .composition import alr_inv
from .sampling_cov import _nearest_pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BayesianMVRnnFH:
    def __init__(self, n_hidden=40, tau2=1.0, nu0=None, Psi0_scale=1.0,
                 n_iter=2000, burn=1000, thin=2, feature_scale=1.0, seed=11):
        self.H = n_hidden; self.tau2 = tau2; self.nu0 = nu0
        self.Psi0_scale = Psi0_scale; self.n_iter = n_iter; self.burn = burn
        self.thin = thin; self.feature_scale = feature_scale; self.seed = seed

    def _standardize(self, Y, D):
        self.mu_ = Y.mean(axis=0); self.s_ = Y.std(axis=0) + 1e-8
        Ys = (Y - self.mu_) / self.s_
        Sinv = np.diag(1.0 / self.s_)
        Ds = np.stack([Sinv @ D[i] @ Sinv for i in range(D.shape[0])])
        return Ys, Ds

    def _features(self, X):
        rnn = _sigmoid(self.feature_scale * (X @ self.W_.T + self.c_))
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X, rnn])

    def fit(self, X, Y, D):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float); Y = np.asarray(Y, float); D = np.asarray(D, float)
        m, p = X.shape; q = Y.shape[1]; H = self.H
        nu0 = self.nu0 if self.nu0 is not None else q + 2
        Psi0 = self.Psi0_scale * np.eye(q)
        Ys, Ds = self._standardize(Y, D)
        Dinv = np.stack([np.linalg.inv(_nearest_pd(Ds[i])) for i in range(m)])
        self.W_ = rng.normal(scale=1.0 / np.sqrt(p), size=(H, p))
        self.c_ = rng.normal(scale=1.0, size=H)
        Hmat = self._features(X); H = Hmat.shape[1]
        Hq = H * q
        P = (1.0 / self.tau2) * np.eye(Hq)
        for i in range(m):
            P += np.kron(Dinv[i], np.outer(Hmat[i], Hmat[i]))
        P = _nearest_pd(P, jitter=1e-8)
        Lp = np.linalg.cholesky(P)
        Pinv = np.linalg.inv(P)
        B = np.zeros((H, q)); U = np.zeros((m, q)); Sigma_u = np.eye(q)
        keep_theta = []; n_keep = 0
        for it in range(self.n_iter):
            Su_inv = np.linalg.inv(_nearest_pd(Sigma_u))
            F = Hmat @ B
            for i in range(m):
                Vi = np.linalg.inv(Su_inv + Dinv[i])
                mi = Vi @ (Dinv[i] @ (Ys[i] - F[i]))
                U[i] = rng.multivariate_normal(mi, _nearest_pd(Vi))
            g = np.zeros(Hq); R = Ys - U
            for i in range(m):
                g += np.kron(Dinv[i] @ R[i], Hmat[i])
            mean_b = Pinv @ g
            z = rng.standard_normal(Hq)
            b = mean_b + np.linalg.solve(Lp.T, z)
            B = b.reshape(q, H).T
            Psi_post = Psi0 + U.T @ U
            Sigma_u = np.atleast_2d(invwishart.rvs(df=nu0 + m,
                          scale=_nearest_pd(Psi_post), random_state=rng))
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                keep_theta.append((Hmat @ B + U).copy()); n_keep += 1
        theta_draws_s = np.array(keep_theta)
        self.theta_draws_ = theta_draws_s * self.s_ + self.mu_
        self.n_draws_ = n_keep; self.Hmat_ = Hmat
        return self

    def predict_alr(self):
        return self.theta_draws_.mean(axis=0)

    def predict_proportions(self):
        return alr_inv(self.theta_draws_).mean(axis=0)

    def posterior_variance(self):
        var = self.theta_draws_.var(axis=0)
        return var, var.sum(axis=1)

    def credible_intervals(self, level=0.90):
        a = (1 - level) / 2
        lo = np.quantile(self.theta_draws_, a, axis=0)
        hi = np.quantile(self.theta_draws_, 1 - a, axis=0)
        return lo, hi

    def proportion_draws(self):
        return alr_inv(self.theta_draws_)
