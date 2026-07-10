DEFAULT_TICKERS = ["NVDA", "AMZN", "META", "GOOGL", "PLTR", "MU", "TSLA"]

TIMEFRAME_CONFIG = {
    "1 Min":  ("5d",  "1m"),
    "5 Min":  ("7d",  "5m"),
    "15 Min": ("60d", "15m"),
    "1 Hour": ("2y",  "1h"),
    "1 Day":  ("5y",  "1d"),
}

MACRO_TICKERS = {
    "VIX":   "^VIX",
    "US10Y": "^TNX",
    "DXY":   "DX-Y.NYB",
    "Gold":  "GC=F",
    "Oil":   "CL=F",
    "SPY":   "SPY",
    "QQQ":   "QQQ",
}

SECTOR_ETFS = {
    "Technology":        "XLK",
    "Healthcare":        "XLV",
    "Financials":        "XLF",
    "Energy":            "XLE",
    "Consumer Disc.":    "XLY",
    "Utilities":         "XLU",
    "Materials":         "XLB",
    "Industrials":       "XLI",
    "Real Estate":       "XLRE",
    "Comm. Services":    "XLC",
    "Consumer Staples":  "XLP",
}

BENCHMARK_TICKERS = {"SPY": "SPY", "QQQ": "QQQ", "SOXX": "SOXX", "VIX": "^VIX"}

# yfinance info['sector'] strings -> SECTOR_ETFS keys
SECTOR_NAME_MAP = {
    "Technology":            "Technology",
    "Information Technology": "Technology",
    "Healthcare":             "Healthcare",
    "Financial Services":     "Financials",
    "Financials":             "Financials",
    "Energy":                 "Energy",
    "Consumer Cyclical":      "Consumer Disc.",
    "Consumer Discretionary": "Consumer Disc.",
    "Utilities":              "Utilities",
    "Basic Materials":        "Materials",
    "Materials":              "Materials",
    "Industrials":            "Industrials",
    "Real Estate":            "Real Estate",
    "Communication Services": "Comm. Services",
    "Consumer Defensive":     "Consumer Staples",
    "Consumer Staples":       "Consumer Staples",
}

EMA_COLORS   = {9: "#FF6B35", 20: "#FFD700", 50: "#00BFFF", 200: "#FF69B4"}
CANDLE_UP    = "#00FF88"
CANDLE_DOWN  = "#FF4444"
VOL_UP       = "rgba(0,255,136,0.6)"
VOL_DOWN     = "rgba(255,68,68,0.6)"

RISK_FREE_RATE = 0.045
