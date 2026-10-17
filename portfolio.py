"""风险平价 + 趋势过滤 + 危机仓位控制"""
import numpy as np
import pandas as pd
from config import (MAX_SINGLE_WEIGHT, DEFENSIVE_BASE, DEFENSIVE_SPLIT,
                    VOL_TARGET, ALPHA_DAMPER, MAX_TURNOVER, TRANSACTION_COST,
                    TREND_LOOKBACK, TREND_SCALE_LOW, TREND_SCALE_HIGH,
                    HMM_P_SAFE, HMM_P_RISK, HMM_MIN_EXPOSURE)


# ==================== 风险平价 ====================
def risk_parity_weights(cov_matrix, max_weight=MAX_SINGLE_WEIGHT):
    """按 1/vol 分配权重（Naive Risk Parity）"""
    vols = np.sqrt(np.diag(cov_matrix.values))
    inv_vol = 1.0 / (vols + 1e-8)
    w = inv_vol / inv_vol.sum()

    # 迭代收敛到上限约束
    for _ in range(50):
        if (w <= max_weight).all():
            break
        excess = np.maximum(w - max_weight, 0).sum()
        w = np.minimum(w, max_weight)
        room = (w < max_weight)
        if room.sum() == 0:
            break
        w[room] += excess * (w[room] / w[room].sum())
    w = w / w.sum()
    return pd.Series(w, index=cov_matrix.index)


def apply_trend_filter(w_base, price_df, current_date):
    """趋势过滤：破位资产权重减半"""
    trend_score = pd.Series(1.0, index=w_base.index)
    hist = price_df.loc[:current_date]
    if len(hist) < TREND_LOOKBACK:
        return w_base

    ma = hist.tail(TREND_LOOKBACK).mean()
    last = hist.iloc[-1]

    for asset in w_base.index:
        if asset in ma.index and last[asset] < ma[asset]:
            trend_score[asset] = TREND_SCALE_LOW

    w_filtered = w_base * trend_score
    if w_filtered.sum() > 0:
        w_filtered = w_filtered / w_filtered.sum()
    return w_filtered


def apply_defensive_floor(w, defensive_weights):
    """将防御底仓叠加进去"""
    all_assets = w.index.union(pd.Index(defensive_weights.keys()))
    w = w.reindex(all_assets, fill_value=0.0)
    remaining = 1.0 - sum(defensive_weights.values())

    if w.sum() > 1e-8:
        w = w / w.sum() * remaining

    for k, v in defensive_weights.items():
        w[k] = w.get(k, 0.0) + v
    return w


def defensive_floor_weights():
    return {
        "Gold_ETF":      DEFENSIVE_SPLIT[0] * DEFENSIVE_BASE,
        "High_Dividend": DEFENSIVE_SPLIT[1] * DEFENSIVE_BASE,
    }


# ==================== 危机管理器 ====================
class AdvancedCrisisManager:
    """EWMA平滑 + HMM概率 → 总仓位系数 + 目标波动控制"""

    def __init__(self, alpha=ALPHA_DAMPER, vol_target=VOL_TARGET):
        self.alpha = alpha
        self.vol_target = vol_target
        self.last_smoothed_p = None

    def smooth_crisis_probability(self, raw_p):
        if self.last_smoothed_p is None:
            self.last_smoothed_p = raw_p
        else:
            self.last_smoothed_p = (self.alpha * raw_p +
                                    (1 - self.alpha) * self.last_smoothed_p)
        return self.last_smoothed_p

    def probability_to_exposure(self, p_crisis):
        """HMM概率 → 总仓位系数 [HMM_MIN_EXPOSURE, 1.0]"""
        if p_crisis <= HMM_P_SAFE:
            return 1.0
        if p_crisis >= HMM_P_RISK:
            return HMM_MIN_EXPOSURE
        # 线性插值
        ratio = (p_crisis - HMM_P_SAFE) / (HMM_P_RISK - HMM_P_SAFE)
        return 1.0 - ratio * (1.0 - HMM_MIN_EXPOSURE)

    def apply_vol_targeting(self, w, cov_matrix):
        """目标波动率：超限则降杠杆，引入现金"""
        assets = w.index.intersection(cov_matrix.index)
        if len(assets) == 0:
            return w, 1.0
        wv = w.loc[assets].values
        cov = cov_matrix.loc[assets, assets].values
        weekly_vol = np.sqrt(wv @ cov @ wv)
        ann_vol = weekly_vol * np.sqrt(52)
        if ann_vol > self.vol_target:
            scale = self.vol_target / ann_vol
        else:
            scale = 1.0
        return w, scale


# ==================== 换手率/成本 ====================
def apply_turnover_limit(prev_w, new_w, max_to=MAX_TURNOVER):
    if prev_w is None:
        return new_w
    all_assets = prev_w.index.union(new_w.index)
    p = prev_w.reindex(all_assets, fill_value=0.0)
    n = new_w.reindex(all_assets, fill_value=0.0)
    turnover = np.sum(np.abs(n - p))
    if turnover <= max_to:
        return new_w
    scale = max_to / turnover
    adjusted = p + scale * (n - p)
    total = adjusted.sum()
    return adjusted / total if total > 1e-8 else new_w


def compute_turnover_cost(prev_w, new_w, cost_rate=TRANSACTION_COST):
    if prev_w is None:
        return 0.0
    all_assets = prev_w.index.union(new_w.index)
    p = prev_w.reindex(all_assets, fill_value=0.0)
    n = new_w.reindex(all_assets, fill_value=0.0)
    return cost_rate * np.sum(np.abs(n - p))
