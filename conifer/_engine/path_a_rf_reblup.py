"""path_a_rf_reblup.py — Path A: the deck's Multi-output ML-SAE (slide 6).

RF multi-output synthetic mean (cross-fit via built-in OOB) -> cross-class
random-effect covariance Sigma_u by method of moments -> multivariate (robust)
EBLUP shrinkage on ALR residuals -> parametric bootstrap MSE.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .composition import alr_inv
from .sampling_cov import _nearest_pd


def _solve(A, b):
    return np.linalg.solve(A, b)


class RFReblupSAE:
    def __init__(self, mean_model="rf", n_estimators=500, robust=True,
                 huber_c=3.0, min_samples_leaf=8, random_state=0):
        assert mean_model in ("rf", "linear")
        self.mean_model = mean_model
        self.n_estimators = n_estimators
        self.robust = robust
        self.huber_c = huber_c
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _fit_mean(self, X, Y):
        if self.mean_model == "rf":
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators, oob_score=True, bootstrap=True,
                max_features=0.5, min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state, n_jobs=-1)
            rf.fit(X, Y)
            m_hat = np.array(rf.oob_prediction_)
            if m_hat.ndim == 1:
                m_hat = m_hat[:, None]
            # fill any all-NaN OOB rows with the in-bag prediction
            bad = ~np.isfinite(m_hat).all(axis=1)
            if bad.any():
                m_hat[bad] = rf.predict(X[bad])
            return rf, m_hat
        else:
            lin = Ridge(alpha=1.0)
            lin.fit(X, Y)
            return lin, lin.predict(X)

    @staticmethod
    def _estimate_Sigma_u(R, D):
        m = R.shape[0]
        rr = np.einsum("iq,ir->qr", R, R) / m
        Su = rr - D.mean(axis=0)
        return _nearest_pd(Su, jitter=1e-6)

    def _shrink(self, m_hat, Y, D, Sigma_u):
        m, q = Y.shape
        theta = np.empty((m, q))
        for i in range(m):
            resid = Y[i] - m_hat[i]
            V = Sigma_u + D[i]
            if self.robust:
                dist = np.sqrt(max(resid @ _solve(V, resid), 1e-12))
                resid = min(1.0, self.huber_c / dist) * resid
            theta[i] = m_hat[i] + Sigma_u @ _solve(V, resid)
        return theta

    def fit(self, X, Y, D, X_train=None, Y_train=None):
        X = np.asarray(X, float); Y = np.asarray(Y, float); D = np.asarray(D, float)
        # train the ML synthetic mean on UNIT-LEVEL data if provided (deck slide 6),
        # else fall back to area-level training (aggregated-only setting).
        if X_train is not None and Y_train is not None:
            self._model, _ = self._fit_mean(np.asarray(X_train, float),
                                            np.asarray(Y_train, float))
            self.m_hat_ = self._model.predict(X)
            self._train = (np.asarray(X_train, float), np.asarray(Y_train, float))
        else:
            self._model, self.m_hat_ = self._fit_mean(X, Y)
            self._train = None
        R = Y - self.m_hat_
        self.Sigma_u_ = self._estimate_Sigma_u(R, D)
        self.theta_hat_ = self._shrink(self.m_hat_, Y, D, self.Sigma_u_)
        self._X, self._Y, self._D = X, Y, D
        return self

    def predict_alr(self):
        return self.theta_hat_

    def predict_proportions(self):
        return alr_inv(self.theta_hat_)

    def bootstrap_mse(self, n_boot=60, seed=1, n_estimators_boot=150):
        rng = np.random.default_rng(seed)
        X, Y, D = self._X, self._Y, self._D
        m, q = Y.shape
        Lu = np.linalg.cholesky(_nearest_pd(self.Sigma_u_))
        LD = [np.linalg.cholesky(_nearest_pd(D[i])) for i in range(m)]
        sq = np.zeros((m, q))
        boot = RFReblupSAE(mean_model=self.mean_model, n_estimators=n_estimators_boot,
                           robust=self.robust, huber_c=self.huber_c,
                           min_samples_leaf=self.min_samples_leaf,
                           random_state=self.random_state)
        Xtr, Ytr = (self._train if self._train is not None else (None, None))
        for b in range(n_boot):
            U = (Lu @ rng.normal(size=(q, m))).T
            theta_star = self.m_hat_ + U
            E = np.stack([LD[i] @ rng.normal(size=q) for i in range(m)])
            boot.fit(X, theta_star + E, D, X_train=Xtr, Y_train=Ytr)
            sq += (boot.theta_hat_ - theta_star) ** 2
        mse_comp = sq / n_boot
        return mse_comp, mse_comp.sum(axis=1)
