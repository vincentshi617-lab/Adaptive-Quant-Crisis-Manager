SECTORS = {
    "Gold_ETF":         "518880.SS",
    "High_Dividend":    "512890.SS",
    "Financials_Broker":"512000.SS",
    "Real_Estate":      "512200.SS",
    "Consumer_Liquor":  "510150.SS",
    "Healthcare_Med":   "512010.SS",
    "Semiconductor":    "512760.SS",
    "AI_Software":      "515070.SS",
    "New_Energy_Veh":   "515030.SS",
    "Coal_Energy":      "515220.SS",
    "Nonferrous_Met":   "512400.SS",
    "Infrastructure":   "511010.SS",
}
GROWTH_ASSETS = [k for k in SECTORS if k not in ["Gold_ETF", "High_Dividend"]]
DEFENSIVE_ASSETS = ["Gold_ETF", "High_Dividend"]

# ==================== 回测 ====================
TRAIN_WINDOW = 104
TEST_WINDOW = 4
STEP_SIZE = 4
MIN_TRAIN_WEEKS = 52

# ==================== 风控（成功参数） ====================
VOL_TARGET = 0.12
ALPHA_DAMPER = 0.4
MAX_SINGLE_WEIGHT = 0.20
DEFENSIVE_BASE = 0.10
DEFENSIVE_SPLIT = (0.5, 0.5)

# ==================== 趋势 ====================
TREND_LOOKBACK = 40
TREND_SCALE_LOW = 0.4
TREND_SCALE_HIGH = 1.0

# ==================== HMM→仓位 ====================
HMM_P_SAFE = 0.30
HMM_P_RISK = 0.70
HMM_MIN_EXPOSURE = 0.35

# ==================== 交易成本 ====================
TRANSACTION_COST = 0.001
MAX_TURNOVER = 0.6

# ==================== HMM ====================
HMM_CANDIDATES = [2]
HMM_COV_TYPE = "diag"
HMM_N_ITER = 500
HMM_TOL = 1e-4
HMM_SEEDS = [42, 0]

# ==================== RMT ====================
RMT_SIGMA_SQ = 1.0

# ==================== 数据 ====================
USE_MOCK_DATA = True
MOCK_SEED = 42
MOCK_WEEKS = 400

# ==================== 模块开关（消融实验用） ====================
FEATURE_FLAGS = {
    "USE_RMT":             True,   # RMT 去噪
    "USE_HMM":             True,   # HMM 危机识别 → 仓位控制
    "USE_TREND":           True,   # 40周均线趋势过滤
    "USE_RISK_PARITY":     True,   # 风险平价基础权重
    "USE_DEFENSIVE_FLOOR": True,   # 永久防御底仓
    "USE_VOL_TARGET":      True,   # 目标波动控制
}
