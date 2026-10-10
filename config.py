"""NIFTY Lens config — single source of truth for universe, paths, costs, thresholds."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CACHE = DATA / "cache"
PRICES = CACHE / "prices"
INDICES = CACHE / "indices"
STATEMENTS = CACHE / "statements"
RESULTS = BASE / "results"
LOGS = BASE / "logs"

# Universe: current NIFTY-50 (.NS suffix for yfinance). Retrieval date recorded in data/meta.json.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
YF_SUFFIX = ".NS"
INDEX_SYMBOLS = {"NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX"}

# Costs (Section 9.2): 0.1% per side
COST_PER_SIDE = 0.001

# Wyckoff/VSA defaults (Section 6.4)
VOL_LOOKBACK = 20
RANGE_LOOKBACK = 20
SPRING_LOOKBACK = 20  # support window for Spring detection
UTAD_LOOKBACK = 20
CLIMAX_VOL_MULT = 2.0
NARROW_SPREAD_MAX = 0.5   # spread_ratio < 0.5 = narrow
WIDE_SPREAD_MIN = 1.5     # spread_ratio > 1.5 = wide
LOW_VOL_MAX = 0.7
HIGH_VOL_MIN = 1.5

# Pattern study horizons
FORWARD_HORIZONS = [1, 3, 5, 10]

# Backtest defaults
DMA_SHORT, DMA_LONG = 50, 200
RSI_PERIOD, RSI_LO, RSI_HI = 14, 30, 70
