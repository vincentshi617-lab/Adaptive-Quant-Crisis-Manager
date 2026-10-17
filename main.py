"""主入口"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from config import USE_MOCK_DATA
from data_utils import load_price_data, to_weekly_returns
from backtest import run_backtest, compute_metrics


def plot_dashboard(results, save_path="dashboard.png"):
    timeline = results["timeline"]
    strat_ret = results["strat_returns"]
    bench_ret = results["bench_returns"]
    crisis_p = results["crisis_probs"]
    exposures = results["exposures"]

    cum_strat = np.cumprod(1 + strat_ret) - 1
    cum_bench = np.cumprod(1 + bench_ret) - 1

    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    ax = axes[0]
    ax.plot(timeline, cum_strat * 100, label="Strategy (Risk Parity + Trend + HMM)",
            color="darkgreen", linewidth=2)
    ax.plot(timeline, cum_bench * 100, label="Equal-Weight Benchmark",
            color="grey", linestyle="--", linewidth=1.5)

    in_crisis, crisis_start = False, None
    for i, (t, p) in enumerate(zip(timeline, crisis_p)):
        if p > 0.5 and not in_crisis:
            in_crisis, crisis_start = True, t
        elif p <= 0.5 and in_crisis:
            ax.axvspan(crisis_start, t, alpha=0.12, color="red")
            in_crisis = False
    if in_crisis:
        ax.axvspan(crisis_start, timeline[-1], alpha=0.12, color="red")

    ax.set_title("Out-of-Sample Performance (Red = Crisis Probability > 0.5)",
                 fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(timeline, 0, crisis_p, color="crimson", alpha=0.6)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Crisis Prob")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(timeline, exposures * 100, color="navy", linewidth=2)
    ax.fill_between(timeline, 0, exposures * 100, color="navy", alpha=0.2)
    ax.set_ylabel("Total Exposure (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    cum = np.cumprod(1 + strat_ret)
    dd_strat = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    cum_b = np.cumprod(1 + bench_ret)
    dd_bench = (cum_b - np.maximum.accumulate(cum_b)) / np.maximum.accumulate(cum_b)
    ax.fill_between(timeline, dd_strat * 100, 0, color="darkgreen", alpha=0.5, label="Strategy")
    ax.plot(timeline, dd_bench * 100, color="grey", linestyle="--", linewidth=1.5, label="Benchmark")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n📊 Dashboard已保存: {save_path}")


def main():
    print("=" * 70)
    print("  Adaptive Quant Crisis Manager — Offensive Framework")
    print("  (Risk Parity + Trend Following + HMM Position Sizing)")
    print("=" * 70)

    print(f"\n[Step 1] 数据加载 (模式: {'模拟' if USE_MOCK_DATA else '真实'})")
    price = load_price_data()
    weekly = to_weekly_returns(price)
    print(f"  资产数: {len(weekly.columns)}, 周数: {len(weekly)}")
    print(f"  时间范围: {weekly.index[0].date()} ~ {weekly.index[-1].date()}")

    print(f"\n[Step 2] 滚动向前回测")
    results = run_backtest(weekly, price, verbose=True)

    print(f"\n[Step 3] 绩效对比")
    strat_m = compute_metrics(results["strat_returns"])
    bench_m = compute_metrics(results["bench_returns"])

    print("\n" + "=" * 70)
    print(f"{'Metric':<18}{'Strategy':<18}{'Benchmark':<18}")
    print("-" * 70)
    for k in strat_m.keys():
        print(f"{k:<18}{strat_m[k]:<18}{bench_m.get(k, '-'):<18}")
    print("=" * 70)

    plot_dashboard(results, save_path="dashboard.png")
    print(f"\n✅ 全流程完成！")


if __name__ == "__main__":
    main()
