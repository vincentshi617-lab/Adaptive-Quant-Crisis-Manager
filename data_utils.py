"""数据加载 + RMT去噪 + 特征工程"""
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from config import SECTORS, RMT_SIGMA_SQ, MOCK_SEED, MOCK_WEEKS, USE_MOCK_DATA


# def load_price_data():
#     if USE_MOCK_DATA:
#         return _generate_mock_data()
#     try:
#         import yfinance as yf
#         tickers = list(SECTORS.values())
#         end = datetime.now()
#         start = end - timedelta(days=8 * 365)
#         raw = yf.download(tickers, start=start, end=end, progress=False)
#         price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
#         rename_map = {v: k for k, v in SECTORS.items()}
#         data = price.rename(columns=rename_map).dropna(how="all").ffill().dropna()
#         return data
#     except Exception as e:
#         print(f"[WARN] 真实数据下载失败({e})，切换到模拟数据")
#         return _generate_mock_data()

import time
import pandas as pd
import akshare as ak
import yfinance as yf
from config import SECTORS

import time
import akshare as ak

def load_real_a_share_data(start_date="20180101"):
    """真实 A 股 ETF 周线数据（带重试）"""
    print("[Data] 正在从 AkShare 下载 A 股 ETF 数据...")
    price_dict = {}
    
    for name, symbol in SECTORS.items():
        code = symbol.split('.')[0]
        
        # 加入重试机制
        success = False
        for attempt in range(3):
            try:
                df = ak.fund_etf_hist_em(symbol=code, period="daily", 
                                         start_date=start_date, adjust="qfq")
                if df.empty:
                    print(f"  [WARN] {name}({code}) 无数据")
                    break
                
                # 将日期转为索引
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘']
                success = True
                break # 成功则跳出重试循环
                
            except Exception as e:
                if attempt < 2:
                    time.sleep(1) # 失败等1秒再试
                else:
                    print(f"  [ERROR] {name}({code}) 最终失败: {e}")
        
        # 防限流
        time.sleep(0.5)

    # 检查是否下载成功
    if not price_dict:
        raise ValueError("❌ 所有数据下载失败。请检查网络连接（是否开启了代理/VPN）或稍后再试。")

    price_df = pd.DataFrame(price_dict)
    price_df = price_df.ffill().dropna()
    
    # 确保索引是 DatetimeIndex
    price_df.index = pd.to_datetime(price_df.index)
    
    weekly_price = price_df.resample("W").last()
    return weekly_price

def _generate_mock_data():
    """真实化的模拟数据：牛市+危机交替，年化收益合理"""
    np.random.seed(MOCK_SEED)
    dates = pd.date_range(end=datetime.now(), periods=MOCK_WEEKS, freq="W")
    n_assets = len(SECTORS)
    assets = list(SECTORS.keys())

    # 相关性结构
    corr_base = np.eye(n_assets) * 0.55 + 0.45 * np.ones((n_assets, n_assets))
    np.fill_diagonal(corr_base, 1.0)
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if a == "Gold_ETF" and b not in ["Gold_ETF"]:
                corr_base[i, j] = corr_base[j, i] = -0.30
            if a == "High_Dividend" and b not in ["Gold_ETF", "High_Dividend"]:
                corr_base[i, j] = corr_base[j, i] = -0.20
    growth_idx = [i for i, a in enumerate(assets) if a not in ["Gold_ETF", "High_Dividend"]]
    for i in growth_idx:
        for j in growth_idx:
            if i != j:
                corr_base[i, j] = 0.55

    # 周化收益
    mean_ret = np.zeros(n_assets)
    for i, a in enumerate(assets):
        if a == "Gold_ETF":
            mean_ret[i] = 0.00096
        elif a == "High_Dividend":
            mean_ret[i] = 0.0015
        else:
            mean_ret[i] = 0.0024

    # 波动率基准（周）
    base_vol = np.ones(n_assets) * 0.022
    base_vol[assets.index("Gold_ETF")] = 0.013
    base_vol[assets.index("High_Dividend")] = 0.012

    # 危机期：3段，各6-8周
    vol_regime = np.ones(MOCK_WEEKS)
    crisis_intervals = [(80, 88), (180, 188), (280, 288)]
    for s, e in crisis_intervals:
        vol_regime[s:e] = 2.5

    gold_idx = assets.index("Gold_ETF")
    hd_idx = assets.index("High_Dividend")

    returns = np.zeros((MOCK_WEEKS, n_assets))
    for t in range(MOCK_WEEKS):
        L = np.linalg.cholesky(corr_base + np.eye(n_assets) * 1e-6)
        z = L @ np.random.randn(n_assets)
        returns[t] = mean_ret + z * base_vol * vol_regime[t]

        for s, e in crisis_intervals:
            if s <= t < e:
                for i in growth_idx:
                    returns[t, i] -= 0.018     # 成长股额外 -1.8%/周
                returns[t, gold_idx] += 0.012   # 黄金避险 +1.2%/周
                returns[t, hd_idx] += 0.005     # 高股息 +0.5%/周

    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=assets)


def to_weekly_returns(price_df):
    return price_df.pct_change(fill_method=None).dropna()


def rmt_clean_correlation(corr_matrix, T, N):
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    q = float(N) / T
    lambda_plus = RMT_SIGMA_SQ * (1 + np.sqrt(q)) ** 2

    noise_idx = np.where(eigenvalues <= lambda_plus)[0]
    if len(noise_idx) > 0:
        noise_mean = np.mean(eigenvalues[noise_idx])
        cleaned = eigenvalues.copy()
        cleaned[noise_idx] = noise_mean
    else:
        cleaned = eigenvalues.copy()

    reconstructed = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    diag_inv = 1.0 / np.sqrt(np.diag(reconstructed))
    reconstructed = reconstructed * diag_inv[:, None] * diag_inv

    return pd.DataFrame(reconstructed, index=corr_matrix.index, columns=corr_matrix.columns)
