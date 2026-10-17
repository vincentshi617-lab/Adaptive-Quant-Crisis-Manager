"""GARCH + HMM + Clayton Copula"""
import numpy as np
import pandas as pd
import logging
import warnings
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from config import (HMM_CANDIDATES, HMM_COV_TYPE, HMM_N_ITER, HMM_TOL, HMM_SEEDS)

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*not converging.*")
warnings.filterwarnings("ignore", message=".*transmat_.*")


# ==================== GARCH ====================
def _garch_neg_loglik(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    T = len(returns)
    h = np.zeros(T)
    h[0] = np.var(returns)
    for t in range(1, T):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
    h = np.clip(h, 1e-8, None)
    return 0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / h)


def _fit_garch_params(r_window):
    if len(r_window) < 20 or np.std(r_window) < 1e-6:
        return np.array([np.var(r_window) * 0.2, 0.1, 0.7])
    init = [np.var(r_window) * 0.2, 0.1, 0.7]
    bounds = ((1e-8, None), (1e-4, 0.5), (1e-4, 0.95))
    try:
        res = minimize(_garch_neg_loglik, init, args=(r_window,),
                       bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 100, "ftol": 1e-6})
        return res.x if res.success else np.array(init)
    except Exception:
        return np.array(init)


def fit_garch_predict(returns_series):
    r = returns_series.values * 100
    if len(r) < 20 or np.std(r) < 1e-6:
        return np.std(r) / 100, np.zeros_like(r)
    omega, alpha, beta = _fit_garch_params(r)
    T = len(r)
    h = np.zeros(T)
    h[0] = np.var(r)
    for t in range(1, T):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
    h = np.clip(h, 1e-8, None)
    next_vol = np.sqrt(omega + alpha * r[-1]**2 + beta * h[-1]) / 100
    residuals = np.clip(r / np.sqrt(h), -3, 3)
    return next_vol, residuals


def rolling_garch_vol_series(returns_series, min_obs=30, refit_freq=8):
    r = returns_series.values * 100
    T = len(r)
    vol_series = np.full(T, np.nan)
    if T < min_obs:
        return pd.Series(vol_series / 100, index=returns_series.index)
    params = _fit_garch_params(r[:min_obs])
    h = np.zeros(T)
    h[0] = np.var(r[:min_obs])
    for t in range(1, T):
        if t >= min_obs and t % refit_freq == 0:
            params = _fit_garch_params(r[max(0, t - 104):t])
        omega, alpha, beta = params
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        vol_series[t] = np.sqrt(h[t]) / 100
    return pd.Series(vol_series, index=returns_series.index)


# ==================== HMM ====================
class AdaptiveHMM:
    """收益+波动率双维特征，BIC选择状态数"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.crisis_state = None
        self.n_states = None

    def _build_features(self, returns_df, vol_df):
        common = returns_df.index.intersection(vol_df.index)
        ret = returns_df.loc[common].values
        vol_ann = vol_df.loc[common].values * np.sqrt(52)
        return np.hstack([ret, vol_ann])

    def fit(self, returns_df, vol_df):
        features = self._build_features(returns_df, vol_df)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)

        best_bic = np.inf
        best_model = None
        best_n = 2

        for n in HMM_CANDIDATES:
            for seed in HMM_SEEDS:
                try:
                    m = GaussianHMM(n_components=n, covariance_type=HMM_COV_TYPE,
                                    n_iter=HMM_N_ITER, tol=HMM_TOL, random_state=seed)
                    m.fit(scaled)
                    bic = m.bic(scaled)
                    if bic < best_bic:
                        best_bic, best_model, best_n = bic, m, n
                except Exception:
                    continue

        if best_model is None:
            best_model = GaussianHMM(n_components=2, covariance_type="diag",
                                      n_iter=200, random_state=42)
            best_model.fit(scaled)
            best_n = 2

        self.model = best_model
        self.n_states = best_n

        means = self.model.means_
        n_assets = returns_df.shape[1]
        ret_means = means[:, :n_assets].mean(axis=1)
        vol_means = means[:, n_assets:].mean(axis=1)
        score = ret_means - vol_means
        self.crisis_state = int(np.argmin(score))
        return self

    def predict_crisis_probability(self, returns_df, vol_df):
        features = self._build_features(returns_df, vol_df)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(scaled)
        raw = proba[:, self.crisis_state]
        # 4周滚动平均平滑
        smoothed = pd.Series(raw).rolling(window=4, min_periods=1).mean().values
        return smoothed


# ==================== Clayton Copula ====================
def fit_clayton_tail_dependence(returns_df, col1, col2):
    x, y = returns_df[col1].dropna(), returns_df[col2].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    if len(x) < 30:
        return 0.0
    u = np.clip(rankdata(x) / (len(x) + 1), 1e-5, 1 - 1e-5)
    v = np.clip(rankdata(y) / (len(y) + 1), 1e-5, 1 - 1e-5)

    def neg_loglik(theta):
        if theta <= 0:
            return 1e10
        t1 = np.log(1 + theta)
        t2 = (-2 - 1/theta) * np.log(u**(-theta) + v**(-theta) - 1)
        t3 = (-1 - theta) * (np.log(u) + np.log(v))
        ld = t1 + t2 + t3
        ld = ld[np.isfinite(ld)]
        return -np.sum(ld)

    try:
        res = minimize(neg_loglik, x0=[1.0], bounds=[(1e-3, 30.0)], method="L-BFGS-B")
        if res.success:
            return 2 ** (-1.0 / res.x[0])
    except Exception:
        pass
    return 0.3
