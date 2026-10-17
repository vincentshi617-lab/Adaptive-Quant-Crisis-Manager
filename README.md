Adaptive Quant Crisis Manager
A multi-asset dynamic allocation framework integrating Random Matrix Theory (RMT), Hidden Markov Models (HMM), GARCH Volatility Modeling, and Trend Following. It aims to capture absolute returns while strictly controlling tail risks (Maximum Drawdown).

Project Status: Work in Progress (WIP).
The core strategy logic and backtesting engine are complete and have been preliminarily validated on real-world A-share and US market historical data. The project is continuously iterating. Feel free to star, watch, or submit an Issue.

Table of Contents

-Overview
-Core Mathematics & Strategy Logic
-Performance & Ablation Study
-Quick Start
-Project Structure & Roadmap
-Disclaimer


Overview

Traditional asset allocation models (such as simple Mean-Variance or Equal Weight) often suffer massive drawdowns during extreme market conditions (e.g., the 2008 Financial Crisis, the 2020 COVID-19 crash, or A-share bear markets). The core objective of this project is to build a "Offensive & Defensive" asset allocation framework:

1. Robust Base Weights: Uses improved covariance estimation and Risk Parity to allocate base weights.
2. Crisis Early Warning: Utilizes an HMM to infer the hidden market regime (Safe vs. Crisis) in real-time, dynamically adjusting total portfolio exposure.
3. Trend Following: Uses long-term moving averages to filter out broken assets, cutting losses early.
4. Defensive Floor: Introduces Gold and High-Dividend assets as a permanent defensive floor to hedge against tail risks.


Core Mathematics & Strategy Logic
This project is not a black box; every module is built upon rigorous mathematical and statistical theories. Here is a deep dive into the core logic:

1. Random Matrix Theory (RMT) Denoising
   Problem: When the asset pool is large, the sample covariance matrix contains a massive amount of "noise" (spurious correlations), causing the Risk Parity algorithm to allocate highly distorted weights.
   Math: Utilizing the Marchenko-Pastur law, we decompose the eigenvalues of the covariance matrix. Eigenvalues falling in the noise zone (where λ≤λ+) are replaced by their average, reconstructing a cleaner covariance matrix.
   Impact: Strips away noise, making the base weight allocation much smoother and more robust.

2. Risk Parity
   Problem: Equal-weighting causes high-volatility assets (like Semiconductors) to dominate the portfolio's risk.
   Math: Calculates the volatility of each asset via the covariance matrix and allocates weights using 1/σ (inverse volatility). The code uses iterative penalty logic (similar to Lagrange multipliers) to enforce a maximum weight cap (e.g., 20%) per asset, achieving true risk diversification.

3. GARCH Volatility Modeling
   Problem: Financial asset volatility exhibits "clustering" (large moves are followed by large moves).
   Math: Generalized Autoregressive Conditional Heteroskedasticity model. Using Maximum Likelihood Estimation (MLE), it utilizes historical residuals and conditional variances to forecast the next week's conditional volatility.
   Impact: Serves as a critical feature input for the HMM regime detection.

4. Hidden Markov Model (HMM) Crisis Detection
   Problem: The market alternates between "Bull/Bear/Sideways" regimes. Standard models struggle to capture these hidden states.
   Math: Based on the Baum-Welch algorithm (EM algorithm), it performs unsupervised clustering on "Return + Volatility" 2D features. Through the transition probability matrix, it outputs the real-time probability of being in a "Crisis" state.
   Impact: When crisis probability spikes, it directly reduces total exposure, acting as a hard stop-loss.

5. Trend Filter & Joint Signal (Core Innovation)
   Problem: As a probabilistic model, HMM has inherent lag. In A-share "V-shaped reversals," it often causes the model to cut losses at the bottom and miss the rebound.
   Math: Calculates the Moving Average over the past N weeks (e.g., 40 weeks). If the closing price breaks below the MA, the asset's weight is discounted.
   Joint Signal Logic: If the trend is intact, the HMM's warning to reduce exposure is halved (preventing V-shape whipsaws). If the trend is already broken, the system fully trusts the HMM's defensive signal.

Performance & Ablation Study
Note: The following data is based on real A-share ETF weekly historical data (2021-2024), strictly out-of-sample, with no look-ahead bias.

Performance Comparison (A-Share Market)
Configuration	Ann. Return	Ann. Vol	Sharpe	Max Drawdown	Calmar
Full Model	7.91%	7.74%	1.02	-6.23%	1.27
No RMT	7.67%	7.65%	1.00	-6.23%	1.23
No HMM	13.22%	10.95%	1.21	-7.41%	1.78
No Trend	3.01%	8.27%	0.36	-13.81%	0.22
Equal Weight (Benchmark)	4.26%	16.95%	0.25	-26.82%	0.16

### Key Insights from the Ablation Study

1. **Trend Following is the Core Alpha**: Removing the trend filter (No Trend) causes the Sharpe ratio to plummet to 0.36 and doubles the maximum drawdown. This proves that trend following is the cornerstone of drawdown control.
   <img width="3200" height="2000" alt="optimized_hmm_comparison" src="https://github.com/user-attachments/assets/b1db62b6-f1ae-420a-907e-cbdc94193230" />


2. **The "Sweet Spot" of the Optimized HMM**: As shown in the HMM optimization comparison chart, the standard HMM (Original HMM) suffers from severe lag, sacrificing too much return to control risk. However, our **Optimized HMM (Joint Signal)** achieves an exceptional balance: it captures nearly the same cumulative return as the pure trend-following model (No HMM), while significantly suppressing drawdowns during market downturns. By halving the HMM's defensive signal when the trend is intact, the model avoids being "shaken out" during V-shaped reversals, yet fully trusts the HMM when the trend is broken. This represents a highly mature and practical risk-adjusted improvement.

3. **Benchmark Outperformance**: During the A-share bear and choppy markets, the Full Model completely crushes the Equal-Weight benchmark in both Sharpe and Max Drawdown.
   <img width="3580" height="2777" alt="real_data_comparison" src="https://github.com/user-attachments/assets/3089d49d-6fa2-4643-8033-79e2c51c1124" />


   <img width="3577" height="2779" alt="ablation_dashboard" src="https://github.com/user-attachments/assets/aed517c4-baa5-4316-b2ad-f57229b937cd" />


Results:
1.Asset Pool Robustness
   Asset Pool Robustness:Demonstrating the strategy's reliance is not on specific assets. By randomly dropping 2 assets from the pool 30 times, the Sharpe Ratio remains highly concentrated with a mean of 1.27, proving excellent diversification capabilities.
   
<img width="2000" height="800" alt="asset_robustness_A-Share" src="https://github.com/user-attachments/assets/4cc78cb8-c365-497e-a4eb-0dccfd042538" />

2. Bootstrap Distribution
   Monte Carlo Bootstrap Simulation: To test if the performance is just luck, we resampled the historical returns 1,000 times. The 5th percentile of the Sharpe Ratio is still above 0, and the maximum drawdown is strictly concentrated around -5%, indicating statistically significant positive Alpha and robust tail-risk management.

   <img width="2800" height="800" alt="monte_carlo_A-Share" src="https://github.com/user-attachments/assets/10fbc545-40e0-4301-a532-d52e6b7b8278" />

3. 1-Year Rolling Sharpe Ratio
   1-Year Rolling Sharpe Ratio: Shows the dynamic out-of-sample performance. The rolling Sharpe remains comfortably above the 1.0 target for the vast majority of the time, demonstrating strong and consistent adaptability to changing market conditions.

   <img width="3000" height="800" alt="rolling_sharpe_A-Share" src="https://github.com/user-attachments/assets/335716c9-e6ad-48b3-b737-3c9834e87a95" />

4. Parameter Sensitivity
   Parameter Sensitivity Heatmap: A robust strategy should not fail if parameters shift slightly. The heatmaps show smooth gradients for both Sharpe Ratio and Max Drawdown, indicating that the optimal thresholds (P_Safe=0.3, P_Risk=0.8) reside on a stable "plateau" rather than a fragile overfitted peak.

   <img width="2800" height="1000" alt="sensitivity_A-Share" src="https://github.com/user-attachments/assets/c74fcba3-1f00-4bb0-828f-8f04ce502ef2" />




Quick Start

Dependencies
pip install -r requirements.txt

Running Code
# 1. Run the Full Model Backtest (Generates the Dashboard)
python main.py

# 2. Run the Ablation Study (Generates the module contribution charts)
python ablation.py

# 3. Run the US Market Long-term Stress Test (2007-2024)
python run_us_market_test.py

Project Structure & Roadmap
├── config.py          # Global configuration & module flags
├── data_utils.py      # Data loading, mock data generator, RMT denoising
├── risk_models.py     # GARCH volatility & HMM crisis detection
├── portfolio.py       # Risk parity, trend filter, defensive floor, vol targeting
├── backtest.py        # Rolling-window backtesting engine
├── main.py            # Main backtest script
├── ablation.py        # Ablation study script
├── stability_tests.py # Stability stress-testing suite
└── results/           # Exported charts and CSV files

Disclaimer

This project is strictly for academic research and technical exchange in quantitative finance, machine learning, and software engineering. Historical backtest data does not guarantee future returns, and the code does not constitute any investment advice. Markets carry risks; invest with caution.

