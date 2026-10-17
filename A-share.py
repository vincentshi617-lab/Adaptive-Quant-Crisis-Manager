import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"] # 支持中文
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from config import FEATURE_FLAGS
from data_utils import load_real_a_share_data, to_weekly_returns
from backtest import run_backtest, compute_metrics

# ==================== 实验配置 ====================
EXPERIMENTS = {
    "Full Model":           {},  # 全部启用
    "No RMT (无去噪)":       {"USE_RMT": False},
    "No HMM (无危机识别)":    {"USE_HMM": False},
    "No Trend (无趋势过滤)":  {"USE_TREND": False},
    "Equal Weight (基准)":   {
        "USE_RMT": False, "USE_HMM": False, "USE_TREND": False,
        "USE_RISK_PARITY": False, "USE_DEFENSIVE_FLOOR": False, "USE_VOL_TARGET": False
    }
}

def run_experiments():
    print("="*70)
    print("  真实 A 股数据消融实验 (AkShare)")
    print("="*70)
    
    # 1. 获取数据
    price_df = load_real_a_share_data(start_date="20180101")
    weekly_returns = to_weekly_returns(price_df)
    print(f"-> 数据获取成功: {len(weekly_returns.columns)} 个资产, {len(weekly_returns)} 周")
    
    results_dict = {}
    metrics_dict = {}
    
    # 2. 逐一运行实验
    for name, flags in EXPERIMENTS.items():
        print(f"\n正在运行: {name} ...")
        # 注意：真实数据建议将步长调小，比如每 4 周测试未来 4 周
        res = run_backtest(weekly_returns, price_df, flags=flags, verbose=False)
        results_dict[name] = res
        metrics_dict[name] = compute_metrics(res["strat_returns"])
    
    # 3. 打印指标表
    print("\n" + "="*90)
    print(f"{'Configuration':<24}{'TotalRet':<12}{'AnnRet':<10}{'AnnVol':<10}{'Sharpe':<9}{'MaxDD':<11}{'Calmar':<9}")
    print("-" * 90)
    for name, m in metrics_dict.items():
        if not m: continue
        print(f"{name:<24}{m['Total Return']:<12}{m['Annual Return']:<10}"
              f"{m['Annual Vol']:<10}{m['Sharpe']:<9}{m['Max Drawdown']:<11}{m['Calmar']:<9}")
    print("="*90)

    # 4. 保存 CSV
    pd.DataFrame(metrics_dict).T.to_csv("real_data_ablation_metrics.csv")
    print("💾 指标已保存至 real_data_ablation_metrics.csv")

    # 5. 画图
    plot_comparison(results_dict, metrics_dict, "real_data_comparison.png")


def plot_comparison(results_dict, metrics_dict, save_path):
    """绘制消融实验对比图"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.5, 1.2])
    
    colors = {"Full Model": "#2ca02c", "No RMT (无去噪)": "#d62728", 
              "No HMM (无危机识别)": "#ff7f0e", "No Trend (无趋势过滤)": "#9467bd", 
              "Equal Weight (基准)": "#7f7f7f"}
    
    # ---- 图 1: 净值曲线 ----
    ax1 = fig.add_subplot(gs[0, :])
    for name, res in results_dict.items():
        cum = np.cumprod(1 + res["strat_returns"]) - 1
        lw = 2.5 if name == "Full Model" else 1.2
        alpha = 1.0 if name == "Full Model" else 0.7
        ls = "--" if "基准" in name else "-"
        ax1.plot(res["timeline"], cum * 100, label=name, color=colors[name], 
                 linewidth=lw, alpha=alpha, linestyle=ls)
        
    ax1.set_title("真实 A 股数据：各配置累计收益对比", fontweight="bold", fontsize=15)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ---- 图 2: 回撤曲线 ----
    ax2 = fig.add_subplot(gs[1, :])
    for name, res in results_dict.items():
        cum = np.cumprod(1 + res["strat_returns"])
        dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        lw = 2.0 if name == "Full Model" else 1.0
        alpha = 1.0 if name == "Full Model" else 0.5
        ax2.plot(res["timeline"], dd * 100, label=name, color=colors[name], 
                 linewidth=lw, alpha=alpha)
    ax2.set_title("回撤对比 (Drawdown)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    
    # ---- 图 3: 夏普 & Calmar 柱状图 ----
    names = list(metrics_dict.keys())
    
    ax3 = fig.add_subplot(gs[2, 0])
    sharpes = [float(metrics_dict[n].get("Sharpe", 0)) for n in names]
    bars = ax3.barh(names, sharpes, color=[colors[n] for n in names])
    ax3.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Target Sharpe = 1.0")
    ax3.set_title("夏普比率 (Sharpe Ratio)", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    calmars = [float(metrics_dict[n].get("Calmar", 0)) for n in names]
    bars = ax4.barh(names, calmars, color=[colors[n] for n in names])
    ax4.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Target Calmar = 0.5")
    ax4.set_title("Calmar 比率 (年化收益 / 最大回撤)", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"📊 对比图已保存至: {save_path}")


if __name__ == "__main__":
    run_experiments()
