"""消融实验：验证每个模块的贡献"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from config import USE_MOCK_DATA, FEATURE_FLAGS
from data_utils import load_price_data, to_weekly_returns
from backtest import run_backtest, compute_metrics


# ==================== 实验配置列表 ====================
ABLATION_CONFIGS = [
    ("Full Model",              {}),  # 全部默认
    ("No RMT",                  {"USE_RMT": False}),
    ("No HMM",                  {"USE_HMM": False}),
    ("No Trend",                {"USE_TREND": False}),
    ("No Risk Parity",          {"USE_RISK_PARITY": False}),
    ("No Defensive Floor",      {"USE_DEFENSIVE_FLOOR": False}),
    ("No Vol Target",           {"USE_VOL_TARGET": False}),
    ("Only Risk Parity",
        {"USE_RMT": False, "USE_HMM": False, "USE_TREND": False,
         "USE_RISK_PARITY": True, "USE_DEFENSIVE_FLOOR": False,
         "USE_VOL_TARGET": False}),
    ("Equal Weight Only",
        {"USE_RMT": False, "USE_HMM": False, "USE_TREND": False,
         "USE_RISK_PARITY": False, "USE_DEFENSIVE_FLOOR": False,
         "USE_VOL_TARGET": False}),
]


def run_ablation():
    print("=" * 70)
    print("  Ablation Study: Module Contribution Analysis")
    print("=" * 70)

    print(f"\n[1] 加载数据...")
    price = load_price_data()
    weekly = to_weekly_returns(price)
    print(f"    {len(weekly.columns)} 资产 × {len(weekly)} 周")

    print(f"\n[2] 逐一运行 {len(ABLATION_CONFIGS)} 个配置...\n")
    all_results = {}
    all_metrics = {}

    for name, flag_override in ABLATION_CONFIGS:
        print(f"  → 运行: {name}")
        try:
            res = run_backtest(weekly, price, flags=flag_override, verbose=False)
            all_results[name] = res
            all_metrics[name] = compute_metrics(res["strat_returns"])
        except Exception as e:
            print(f"     ❌ 失败: {e}")
            all_metrics[name] = {}

    # ==================== 对比表 ====================
    print("\n" + "=" * 90)
    print(f"{'Configuration':<24}{'TotalRet':<12}{'AnnRet':<10}"
          f"{'AnnVol':<10}{'Sharpe':<9}{'MaxDD':<11}{'Calmar':<9}")
    print("-" * 90)
    for name in all_metrics:
        m = all_metrics[name]
        if not m:
            continue
        print(f"{name:<24}{m['Total Return']:<12}{m['Annual Return']:<10}"
              f"{m['Annual Vol']:<10}{m['Sharpe']:<9}{m['Max Drawdown']:<11}"
              f"{m['Calmar']:<9}")
    print("=" * 90)

    # 保存为 CSV
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics.to_csv("ablation_results.csv")
    print(f"\n💾 结果已保存: ablation_results.csv")

    # ==================== 可视化 ====================
    plot_ablation(all_results, all_metrics, save_path="ablation_dashboard.png")


def plot_ablation(all_results, all_metrics, save_path):
    """多配置净值/回撤/指标对比"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.2])

    # ---- 净值曲线 ----
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, res), color in zip(all_results.items(), colors):
        cum = np.cumprod(1 + res["strat_returns"]) - 1
        lw = 2.5 if name == "Full Model" else 1.2
        alpha = 1.0 if name == "Full Model" else 0.6
        ax1.plot(res["timeline"], cum * 100, label=name,
                 color=color, linewidth=lw, alpha=alpha)

    # 基准
    bench_cum = np.cumprod(1 + list(all_results.values())[0]["bench_returns"]) - 1
    ax1.plot(list(all_results.values())[0]["timeline"], bench_cum * 100,
             label="Equal-Weight Benchmark", color="black",
             linestyle="--", linewidth=2, alpha=0.7)

    ax1.set_title("Ablation Study: Cumulative Returns by Configuration",
                  fontweight="bold", fontsize=13)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(loc="upper left", ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- 回撤曲线 ----
    ax2 = fig.add_subplot(gs[1, :])
    for (name, res), color in zip(all_results.items(), colors):
        cum = np.cumprod(1 + res["strat_returns"])
        dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        lw = 2.5 if name == "Full Model" else 1.0
        alpha = 1.0 if name == "Full Model" else 0.5
        ax2.plot(res["timeline"], dd * 100, label=name,
                 color=color, linewidth=lw, alpha=alpha)
    ax2.set_title("Drawdowns by Configuration", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # ---- 夏普对比柱状图 ----
    ax3 = fig.add_subplot(gs[2, 0])
    names = list(all_metrics.keys())
    sharpes = [float(all_metrics[n].get("Sharpe", 0)) for n in names]
    bars = ax3.barh(names, sharpes, color=colors)
    ax3.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Target Sharpe=1.0")
    ax3.set_title("Sharpe Ratio by Configuration", fontweight="bold")
    ax3.set_xlabel("Sharpe")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ---- 最大回撤对比 ----
    ax4 = fig.add_subplot(gs[2, 1])
    mdd = [abs(float(all_metrics[n].get("Max Drawdown", "0%").strip("%"))) for n in names]
    bars = ax4.barh(names, mdd, color=colors)
    ax4.set_title("Max Drawdown by Configuration (%)", fontweight="bold")
    ax4.set_xlabel("Max Drawdown (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"📊 Dashboard已保存: {save_path}")


if __name__ == "__main__":
    run_ablation()
