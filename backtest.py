"""滚动向前回测 — 支持模块开关（用于消融实验）"""
import numpy as np
import pandas as pd
from config import (TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE, MIN_TRAIN_WEEKS,
                    GROWTH_ASSETS, FEATURE_FLAGS)
from data_utils import rmt_clean_correlation
from risk_models import (rolling_garch_vol_series, fit_garch_predict, AdaptiveHMM)
from portfolio import (risk_parity_weights, apply_trend_filter,
                        apply_defensive_floor, defensive_floor_weights,
                        AdvancedCrisisManager, apply_turnover_limit,
                        compute_turnover_cost)


def run_backtest(weekly_returns, price_df, flags=None, verbose=True):
    if flags is None:
        flags = FEATURE_FLAGS.copy()
    else:
        # 补齐默认值
        full = FEATURE_FLAGS.copy()
        full.update(flags)
        flags = full

    all_assets = weekly_returns.columns.tolist()
    n_total = len(weekly_returns)

    strat_returns, bench_returns, timeline = [], [], []
    crisis_probs, exposures = [], []
    weights_history = []
    current_weights = None

    crisis_mgr = AdvancedCrisisManager()

    start = TRAIN_WINDOW
    n_windows = (n_total - TRAIN_WINDOW - TEST_WINDOW) // STEP_SIZE

    if verbose:
        active = [k for k, v in flags.items() if v]
        print(f"[回测] 窗口={n_windows}, 启用模块: {active}\n")

    for w_idx in range(n_windows):
        train_end = start + w_idx * STEP_SIZE
        test_start = train_end
        test_end = min(test_start + TEST_WINDOW, n_total)
        if test_end <= test_start:
            break

        train_data = weekly_returns.iloc[train_end - TRAIN_WINDOW : train_end]
        test_data = weekly_returns.iloc[test_start : test_end]
        if len(train_data) < MIN_TRAIN_WEEKS:
            continue

        # ---------- GARCH ----------
        train_vol = pd.DataFrame(index=train_data.index)
        for c in all_assets:
            train_vol[c] = rolling_garch_vol_series(train_data[c], refit_freq=8)

        # ---------- HMM ----------
        hmm = None
        if flags["USE_HMM"]:
            try:
                hmm = AdaptiveHMM()
                hmm.fit(train_data[GROWTH_ASSETS], train_vol[GROWTH_ASSETS])
            except Exception:
                hmm = None

        # ---------- 测试期逐周 GARCH ----------
        combined = pd.concat([train_data, test_data], axis=0)
        test_vol_weekly = pd.DataFrame(index=test_data.index,
                                        columns=all_assets, dtype=float)
        for c in all_assets:
            for date in test_data.index:
                hist = combined[c].loc[:date]
                vol, _ = fit_garch_predict(hist)
                test_vol_weekly.loc[date, c] = vol

        # ---------- 危机概率 ----------
        if flags["USE_HMM"] and hmm is not None:
            raw_proba = hmm.predict_crisis_probability(
                test_data[GROWTH_ASSETS], test_vol_weekly[GROWTH_ASSETS])
        else:
            raw_proba = np.zeros(len(test_data))  # 无HMM = 无危机

        # ---------- 协方差 ----------
        try:
            if flags["USE_RMT"]:
                cov_raw = train_data.corr()
                cov_clean = rmt_clean_correlation(cov_raw, T=len(train_data),
                                                    N=len(all_assets))
                stds = train_data.std().values
                cov_matrix = pd.DataFrame(
                    cov_clean.values * np.outer(stds, stds),
                    index=all_assets, columns=all_assets)
                cov_matrix += np.eye(len(all_assets)) * 1e-6 * np.trace(cov_matrix.values) / len(all_assets)
            else:
                cov_matrix = train_data.cov()
        except Exception:
            cov_matrix = train_data.cov()

        # ---------- 基础权重 ----------
        try:
            if flags["USE_RISK_PARITY"]:
                w_base = risk_parity_weights(cov_matrix)
            else:
                w_base = pd.Series(1.0 / len(all_assets), index=all_assets)
        except Exception:
            w_base = pd.Series(1.0 / len(all_assets), index=all_assets)

        # ---------- 逐周执行 ----------
        defensive_floor = defensive_floor_weights()

        for i, date in enumerate(test_data.index):
            # HMM 平滑 + 仓位
            if flags["USE_HMM"]:
                p_smooth = crisis_mgr.smooth_crisis_probability(raw_proba[i])
                exposure = crisis_mgr.probability_to_exposure(p_smooth)
            else:
                p_smooth = 0.0
                exposure = 1.0

            # 趋势过滤
            if flags["USE_TREND"]:
                w_trend = apply_trend_filter(w_base, price_df, date)
            else:
                w_trend = w_base

            # 防御底仓
            if flags["USE_DEFENSIVE_FLOOR"]:
                w_with_def = apply_defensive_floor(w_trend, defensive_floor)
            else:
                w_with_def = w_trend

            # 目标波动
            if flags["USE_VOL_TARGET"]:
                w_with_def, vol_scale = crisis_mgr.apply_vol_targeting(
                    w_with_def, cov_matrix)
            else:
                vol_scale = 1.0

            final_exposure = exposure * vol_scale
            w_final = w_with_def * final_exposure
            w_final["Cash"] = 1.0 - final_exposure

            w_final = apply_turnover_limit(current_weights, w_final)

            actual_ret = test_data.loc[date]
            port_ret = float((w_final.reindex(all_assets, fill_value=0.0) *
                              actual_ret).sum())
            port_ret -= compute_turnover_cost(current_weights, w_final)

            bench_ret = float(actual_ret.mean())

            strat_returns.append(port_ret)
            bench_returns.append(bench_ret)
            timeline.append(date)
            crisis_probs.append(p_smooth)
            exposures.append(final_exposure)
            weights_history.append(w_final.copy())
            current_weights = w_final

        if verbose and (w_idx + 1) % 20 == 0:
            print(f"  窗口 {w_idx+1}/{n_windows} @ {test_data.index[-1].date()}")

    return {
        "timeline": pd.DatetimeIndex(timeline),
        "strat_returns": np.array(strat_returns),
        "bench_returns": np.array(bench_returns),
        "crisis_probs": np.array(crisis_probs),
        "exposures": np.array(exposures),
        "weights_history": weights_history,
    }


def compute_metrics(returns, freq=52):
    returns = np.asarray(returns)
    if len(returns) == 0:
        return {}
    cum = np.cumprod(1 + returns)
    total = cum[-1] - 1
    ann_ret = (1 + total) ** (freq / len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0
    dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    return {
        "Total Return": f"{total:.2%}",
        "Annual Return": f"{ann_ret:.2%}",
        "Annual Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar": f"{calmar:.2f}",
    }
