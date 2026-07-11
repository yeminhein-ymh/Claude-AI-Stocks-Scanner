import yfinance as yf
import streamlit as st


@st.cache_data(ttl=21600)  # fundamentals change slowly — cache 6h
def get_fundamentals(ticker: str) -> dict:
    """Best-effort fundamental snapshot. Any field can be None if yfinance doesn't have it."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "ticker":              ticker,
        "revenue_growth":      info.get("revenueGrowth"),
        "earnings_growth":     info.get("earningsGrowth"),
        "gross_margins":       info.get("grossMargins"),
        "operating_margins":   info.get("operatingMargins"),
        "profit_margins":      info.get("profitMargins"),
        "return_on_equity":    info.get("returnOnEquity"),
        "debt_to_equity":      info.get("debtToEquity"),
        "peg_ratio":           info.get("trailingPegRatio") or info.get("pegRatio"),
        "trailing_pe":         info.get("trailingPE"),
        "forward_pe":          info.get("forwardPE"),
        "free_cashflow":       info.get("freeCashflow"),
        "recommendation_mean": info.get("recommendationMean"),   # 1=Strong Buy .. 5=Strong Sell
        "recommendation_key":  info.get("recommendationKey"),
        "target_mean_price":   info.get("targetMeanPrice"),
        "current_price":       info.get("currentPrice") or info.get("regularMarketPrice"),
        "insider_ownership":   info.get("heldPercentInsiders"),
        "institution_ownership": info.get("heldPercentInstitutions"),
        "short_percent_float": info.get("shortPercentOfFloat"),
        "short_ratio":         info.get("shortRatio"),
        "sector":              info.get("sector"),
    }
