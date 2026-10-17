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
