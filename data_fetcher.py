import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from config import MACRO_TICKERS, SECTOR_ETFS, BENCHMARK_TICKERS


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=60)
def get_stock_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = _flatten(t.history(period="2d", interval="1d"))
        price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        change_pct = (price - prev) / prev * 100 if prev else 0.0

        return {
            "ticker":     ticker,
            "price":      round(price, 2),
            "change_pct": round(change_pct, 2),
            "volume":     info.get("volume", info.get("regularMarketVolume", 0)),
            "avg_volume": info.get("averageVolume", 0),
            "market_cap": info.get("marketCap", 0),
            "beta":       info.get("beta", 1.0),
            "week52_hi":  info.get("fiftyTwoWeekHigh", 0),
            "week52_lo":  info.get("fiftyTwoWeekLow", 0),
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", "N/A"),
        }
    except Exception:
        return {"ticker": ticker, "price": 0.0, "change_pct": 0.0,
                "volume": 0, "avg_volume": 0, "market_cap": 0,
                "beta": 1.0, "week52_hi": 0, "week52_lo": 0,
                "name": ticker, "sector": "N/A"}


@st.cache_data(ttl=120)
def get_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = _flatten(yf.download(ticker, period=period, interval=interval,
                                  auto_adjust=True, progress=False))
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_options_chain(ticker: str):
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None, []
        chains = {}
        for exp in expirations[:6]:
            chain = t.option_chain(exp)
            chains[exp] = {"calls": chain.calls, "puts": chain.puts}
        return chains, list(expirations[:6])
    except Exception:
        return None, []


@st.cache_data(ttl=120)
def get_macro_data() -> pd.DataFrame:
    rows = []
    for name, sym in MACRO_TICKERS.items():
        try:
            t = yf.Ticker(sym)
            hist = _flatten(t.history(period="2d", interval="1d"))
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
            prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
            rows.append({"Name": name, "Symbol": sym,
                         "Price": round(price, 2), "Change %": chg})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_sector_data() -> pd.DataFrame:
    rows = []
    for name, sym in SECTOR_ETFS.items():
        try:
            t = yf.Ticker(sym)
            hist = _flatten(t.history(period="5d", interval="1d"))
            if len(hist) < 2:
                continue
            price    = float(hist["Close"].iloc[-1])
            prev_1d  = float(hist["Close"].iloc[-2])
            prev_5d  = float(hist["Close"].iloc[0])
            chg_1d   = round((price - prev_1d) / prev_1d * 100, 2)
            chg_5d   = round((price - prev_5d) / prev_5d * 100, 2)
            rows.append({"Sector": name, "ETF": sym, "Price": round(price, 2),
                         "1D %": chg_1d, "5D %": chg_5d})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_benchmark_closes(period: str = "1y") -> pd.DataFrame:
    """Daily close series for SPY/QQQ/SOXX/VIX, aligned on a shared date index.
    Used for relative strength, correlation, and beta calculations."""
    series = {}
    for name, sym in BENCHMARK_TICKERS.items():
        try:
            hist = _flatten(yf.Ticker(sym).history(period=period, interval="1d"))
            if not hist.empty:
                series[name] = hist["Close"]
        except Exception:
            pass
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        hist = _flatten(t.history(period="1d", interval="1m"))
        return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    except Exception:
        return 0.0
