import numpy as np
import pandas as pd
from scipy.stats import norm
from config import RISK_FREE_RATE


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = "call") -> dict:
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        nd1 = norm.pdf(d1)

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = ((-S * nd1 * sigma / (2 * np.sqrt(T)))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = ((-S * nd1 * sigma / (2 * np.sqrt(T)))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega  = S * np.sqrt(T) * nd1 / 100

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega":  round(vega, 4),
        }
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def enrich_chain_greeks(chain_df: pd.DataFrame, spot: float,
                        expiry_str: str, option_type: str) -> pd.DataFrame:
    try:
        expiry = pd.Timestamp(expiry_str)
        T = max((expiry - pd.Timestamp.now()).days / 365, 1 / 365)
    except Exception:
        T = 30 / 365

    df = chain_df.copy()
    df["volume"]       = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0)
    df["impliedVolatility"] = pd.to_numeric(
        df.get("impliedVolatility", 0.3), errors="coerce").fillna(0.3)

    greeks = df.apply(
        lambda row: black_scholes_greeks(
            spot, row["strike"], T, RISK_FREE_RATE,
            max(row["impliedVolatility"], 0.01), option_type
        ), axis=1
    )
    df["delta"] = greeks.apply(lambda g: g["delta"])
    df["gamma"] = greeks.apply(lambda g: g["gamma"])
    df["theta"] = greeks.apply(lambda g: g["theta"])
    df["vega"]  = greeks.apply(lambda g: g["vega"])
    df["IV %"]  = (df["impliedVolatility"] * 100).round(1)
    df["OI/Vol"] = np.where(df["volume"] > 0,
                            (df["openInterest"] / df["volume"]).round(1), 0)
    return df


def calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    try:
        all_strikes = sorted(
            set(list(calls_df["strike"])) | set(list(puts_df["strike"]))
        )
        min_pain, max_pain_strike = float("inf"), all_strikes[0]
        for S in all_strikes:
            c_loss = sum(
                (S - k) * oi * 100
                for k, oi in zip(calls_df["strike"], calls_df["openInterest"])
                if k < S and oi > 0
            )
            p_loss = sum(
                (k - S) * oi * 100
                for k, oi in zip(puts_df["strike"], puts_df["openInterest"])
                if k > S and oi > 0
            )
            total = c_loss + p_loss
            if total < min_pain:
                min_pain, max_pain_strike = total, S
        return max_pain_strike
    except Exception:
        return 0.0


def calculate_pcr(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> dict:
    try:
        call_vol = calls_df["volume"].fillna(0).sum()
        put_vol  = puts_df["volume"].fillna(0).sum()
        call_oi  = calls_df["openInterest"].fillna(0).sum()
        put_oi   = puts_df["openInterest"].fillna(0).sum()
        return {
            "pcr_volume": round(put_vol / call_vol, 3) if call_vol > 0 else 0,
            "pcr_oi":     round(put_oi / call_oi, 3) if call_oi > 0 else 0,
            "call_volume": int(call_vol),
            "put_volume":  int(put_vol),
            "call_oi":     int(call_oi),
            "put_oi":      int(put_oi),
        }
    except Exception:
        return {"pcr_volume": 0, "pcr_oi": 0, "call_volume": 0,
                "put_volume": 0, "call_oi": 0, "put_oi": 0}


def calculate_gex(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                  spot: float, expiry_str: str) -> float:
    try:
        expiry = pd.Timestamp(expiry_str)
        T = max((expiry - pd.Timestamp.now()).days / 365, 1 / 365)
        gex = 0.0
        for _, row in calls_df.iterrows():
            g = black_scholes_greeks(spot, row["strike"], T, RISK_FREE_RATE,
                                     max(row.get("impliedVolatility", 0.3), 0.01), "call")
            gex += g["gamma"] * row.get("openInterest", 0) * 100
        for _, row in puts_df.iterrows():
            g = black_scholes_greeks(spot, row["strike"], T, RISK_FREE_RATE,
                                     max(row.get("impliedVolatility", 0.3), 0.01), "put")
            gex -= g["gamma"] * row.get("openInterest", 0) * 100
        return round(gex, 2)
    except Exception:
        return 0.0


def detect_unusual_activity(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                             ticker: str, spot: float) -> pd.DataFrame:
    rows = []
    for opt_type, df in [("CALL", calls_df), ("PUT", puts_df)]:
        df = df.copy()
        df["volume"]       = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        df["openInterest"] = pd.to_numeric(df.get("openInterest", 1), errors="coerce").fillna(1)
        df["lastPrice"]    = pd.to_numeric(df.get("lastPrice", 0), errors="coerce").fillna(0)

        for _, row in df.iterrows():
            vol = row["volume"]
            oi  = max(row["openInterest"], 1)
            prem = vol * row["lastPrice"] * 100
            if vol < 50:
                continue
            score = 0
            if vol / oi > 3.0:       score += 3
            elif vol / oi > 1.5:     score += 1
            if prem > 500_000:       score += 3
            elif prem > 100_000:     score += 1
            if vol > 1000:           score += 1
            if score >= 3:
                sentiment = "Bullish" if opt_type == "CALL" else "Bearish"
                itm = " [ITM]" if (opt_type == "CALL" and row["strike"] < spot) or \
                                   (opt_type == "PUT" and row["strike"] > spot) else ""
                rows.append({
                    "Ticker":   ticker,
                    "Type":     f"{opt_type}{itm}",
                    "Strike":   row["strike"],
                    "Vol":      int(vol),
                    "OI":       int(oi),
                    "Vol/OI":   round(vol / oi, 1),
                    "Premium":  f"${prem:,.0f}",
                    "Score":    score,
                    "Sentiment": sentiment,
                })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
    return df_out


def calculate_iv_rank(hist_iv: pd.Series, current_iv: float) -> float:
    if hist_iv.empty or hist_iv.max() == hist_iv.min():
        return 50.0
    return round((current_iv - hist_iv.min()) / (hist_iv.max() - hist_iv.min()) * 100, 1)
