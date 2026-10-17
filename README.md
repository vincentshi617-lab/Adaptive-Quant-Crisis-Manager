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
3. 
