"""
Composite AI scoring engine — Page 1 (AI Scanner).

Philosophy: never trade off a single indicator. Every indicator casts a weighted
vote inside its category; categories are combined into one composite AI Score.
Every score keeps the list of reasons that produced it, so the UI can always
answer "why" instead of just "buy/sell".
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import SECTOR_ETFS, SECTOR_NAME_MAP
from modules.data_fetcher import get_stock_info, get_ohlcv, get_options_chain, get_benchmark_closes
from modules.indicators import add_all_indicators, add_scoring_indicators
from modules.ai_engine import get_ai_prediction
from modules.options_analyzer import calculate_pcr, calculate_gex, detect_unusual_activity
from modules.risk_manager import calculate_rrr
from modules.fundamentals import get_fundamentals

# Category weights — must sum to 1.0. Documented here so they're transparent/tunable.
WEIGHTS = {
    "trend":              0.20,
    "momentum":           0.15,
    "volume":             0.10,
    "volatility":         0.10,  # inverted risk -> stability contributes
    "relative_strength":  0.15,
    "fundamental":        0.10,
    "options_flow":       0.10,
    "ml":                 0.10,
}

KELLY_FRACTION = 0.25   # quarter-Kelly, matches AlphaBot's risk convention
MAX_KELLY_PCT  = 15.0   # hard cap on suggested position size


# ─── Voting primitive ────────────────────────────────────────────────────────
def _score_from_votes(votes: list) -> tuple:
    """votes: list of (weight, vote, reason), vote in {-1, 0, 1}.
    Returns (score_0_100, [reasons for non-zero votes])."""
    if not votes:
        return 50.0, ["No data available"]
    total_w = sum(w for w, v, r in votes)
    if total_w == 0:
        return 50.0, ["No data available"]
    raw = sum(w * v for w, v, r in votes) / total_w  # in [-1, 1]
    score = (raw + 1) / 2 * 100
    reasons = [r for w, v, r in votes if v != 0]
    return round(score, 1), reasons


def _swing_structure(df: pd.DataFrame, window: int = 10):
    if len(df) < window * 2:
        return None
    recent = df.iloc[-window:]
    prior = df.iloc[-2 * window:-window]
    hh, hl = recent["High"].max() > prior["High"].max(), recent["Low"].min() > prior["Low"].min()
    lh, ll = recent["High"].max() < prior["High"].max(), recent["Low"].min() < prior["Low"].min()
    if hh and hl:
        return {"vote": 1, "reason": "Higher highs and higher lows (uptrend structure)"}
    if lh and ll:
        return {"vote": -1, "reason": "Lower highs and lower lows (downtrend structure)"}
    return {"vote": 0, "reason": "Mixed swing structure (no clean HH/HL or LH/LL pattern)"}


# ─── Category scorers ────────────────────────────────────────────────────────
def score_trend(df: pd.DataFrame) -> tuple:
    votes = []
    last = df.iloc[-1]
    close = last["Close"]
    emas = {p: last[f"EMA_{p}"] for p in [9, 20, 50, 200]
            if f"EMA_{p}" in df.columns and not pd.isna(last[f"EMA_{p}"])}

    weights = {9: 1.0, 20: 1.5, 50: 1.5, 200: 2.0}
    for p, ema in emas.items():
        v = 1 if close > ema else -1
        votes.append((weights[p], v, f"Price {'above' if v > 0 else 'below'} EMA{p}"))

    if 20 in emas and 50 in emas:
        v = 1 if emas[20] > emas[50] else -1
        votes.append((1.5, v, f"EMA20 {'above' if v > 0 else 'below'} EMA50"))
    if 50 in emas and 200 in emas:
        v = 1 if emas[50] > emas[200] else -1
        tag = "Golden Cross regime" if v > 0 else "Death Cross regime"
        votes.append((2.0, v, f"EMA50 {'above' if v > 0 else 'below'} EMA200 ({tag})"))

    if "ADX" in df.columns and not pd.isna(last["ADX"]):
        adx = last["ADX"]
        if adx > 25:
            direction = 1 if last.get("ADX_Pos", 0) >= last.get("ADX_Neg", 0) else -1
            di = "+DI" if direction > 0 else "-DI"
            votes.append((1.5, direction, f"ADX {adx:.1f} above 25, {di} dominant (strong trend)"))
        else:
            votes.append((0.5, 0, f"ADX {adx:.1f} below 25 (weak/no trend)"))

    swing = _swing_structure(df)
    if swing:
        votes.append((1.5, swing["vote"], swing["reason"]))

    return _score_from_votes(votes)


def score_momentum(df: pd.DataFrame) -> tuple:
    votes = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    if "RSI" in df.columns and not pd.isna(last["RSI"]):
        rsi = last["RSI"]
        if rsi > 60:
            votes.append((1.5, 1, f"RSI {rsi:.1f} in bullish zone (>60)"))
        elif rsi < 40:
            votes.append((1.5, -1, f"RSI {rsi:.1f} in bearish zone (<40)"))
        else:
            votes.append((0.5, 0, f"RSI {rsi:.1f} neutral (40-60)"))
        if not pd.isna(prev.get("RSI", np.nan)):
            v = 1 if rsi > prev["RSI"] else -1
            votes.append((0.5, v, f"RSI {'rising' if v > 0 else 'falling'}"))

    if "MACD_Hist" in df.columns and not pd.isna(last["MACD_Hist"]):
        h = last["MACD_Hist"]
        votes.append((1.5, 1 if h > 0 else -1, f"MACD histogram {'positive' if h > 0 else 'negative'} ({h:.3f})"))
        ph = prev.get("MACD_Hist", np.nan)
        if not pd.isna(ph):
            if ph < 0 <= h:
                votes.append((1.0, 1, "MACD bullish crossover"))
            elif ph > 0 >= h:
                votes.append((1.0, -1, "MACD bearish crossover"))

    if "Stoch_K" in df.columns and not pd.isna(last["Stoch_K"]):
        k, d = last["Stoch_K"], last.get("Stoch_D", np.nan)
        if not pd.isna(d):
            v = 1 if k > d else -1
            votes.append((1.0, v, f"Stochastic %K {'above' if v > 0 else 'below'} %D"))
        if k > 80:
            votes.append((0.5, -1, f"Stochastic {k:.0f} overbought"))
        elif k < 20:
            votes.append((0.5, 1, f"Stochastic {k:.0f} oversold"))

    if "ROC" in df.columns and not pd.isna(last["ROC"]):
        roc = last["ROC"]
        votes.append((1.0, 1 if roc > 0 else -1, f"ROC {roc:+.2f}% ({'positive' if roc > 0 else 'negative'} momentum)"))

    if "CCI" in df.columns and not pd.isna(last["CCI"]):
        cci = last["CCI"]
        if cci > 100:
            votes.append((0.5, 1, f"CCI {cci:.0f} above +100 (strong bullish momentum)"))
        elif cci < -100:
            votes.append((0.5, -1, f"CCI {cci:.0f} below -100 (strong bearish momentum)"))
        else:
            votes.append((0.25, 0, f"CCI {cci:.0f} neutral"))

    return _score_from_votes(votes)


def score_volume(df: pd.DataFrame) -> tuple:
    votes = []
    last = df.iloc[-1]

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_ma) and vol_ma > 0:
            rel_vol = last["Volume"] / vol_ma
            up_day = last["Close"] >= last["Open"]
            if rel_vol > 1.5:
                v = 1 if up_day else -1
                votes.append((1.5, v, f"Relative volume {rel_vol:.1f}x average on a {'up' if up_day else 'down'} day"))
            else:
                votes.append((0.5, 0, f"Relative volume {rel_vol:.1f}x average (unremarkable)"))

    if "OBV" in df.columns and len(df) > 20:
        obv_now, obv_prior = df["OBV"].iloc[-1], df["OBV"].iloc[-20]
        v = 1 if obv_now > obv_prior else -1
        tag = "accumulation" if v > 0 else "distribution"
        votes.append((1.5, v, f"OBV {'rising' if v > 0 else 'falling'} over 20 bars ({tag})"))

    if "MFI" in df.columns and not pd.isna(last.get("MFI", np.nan)):
        mfi = last["MFI"]
        if mfi > 60:
            votes.append((1.0, 1, f"MFI {mfi:.0f} shows buying pressure"))
        elif mfi < 40:
            votes.append((1.0, -1, f"MFI {mfi:.0f} shows selling pressure"))
        else:
            votes.append((0.5, 0, f"MFI {mfi:.0f} neutral"))

    return _score_from_votes(votes)


def score_volatility_risk(df: pd.DataFrame, info: dict) -> tuple:
    """Returns a RISK score 0-100 where higher = riskier (vote=+1 means 'adds risk')."""
    votes = []
    last = df.iloc[-1]
    close = last["Close"]

    if "ATR" in df.columns and not pd.isna(last["ATR"]) and close > 0:
        atr_pct = last["ATR"] / close * 100
        atr_series = (df["ATR"] / df["Close"] * 100).dropna()
        pct_rank = (atr_series < atr_pct).mean() * 100 if len(atr_series) > 30 else 50
        v = 1 if pct_rank > 60 else (-1 if pct_rank < 40 else 0)
        votes.append((1.5, v, f"ATR {atr_pct:.2f}% of price, {pct_rank:.0f}th percentile of its own 1y range"))

    if "BB_Width" in df.columns and not pd.isna(last.get("BB_Width", np.nan)):
        bw_series = df["BB_Width"].dropna()
        if len(bw_series) > 30:
            pct_rank = (bw_series < bw_series.iloc[-1]).mean() * 100
            v = 1 if pct_rank > 60 else (-1 if pct_rank < 40 else 0)
            votes.append((1.0, v, f"Bollinger Band width {pct_rank:.0f}th percentile (wider = more volatile)"))

    beta = info.get("beta")
    if beta:
        try:
            beta = float(beta)
            v = 1 if beta > 1.3 else (-1 if beta < 0.8 else 0)
            votes.append((1.0, v, f"Beta {beta:.2f}"))
        except (TypeError, ValueError):
            pass

    lo, hi = info.get("week52_lo"), info.get("week52_hi")
    if lo and hi and hi > lo:
        pos = (close - lo) / (hi - lo)
        if pos < 0.15:
            votes.append((1.0, 1, f"Price near 52-week low ({pos * 100:.0f}% of range) — elevated downside risk"))
        elif pos > 0.85:
            votes.append((0.5, 0, f"Price near 52-week high ({pos * 100:.0f}% of range)"))

    return _score_from_votes(votes)


def score_relative_strength(ticker_closes: pd.Series, benchmarks: pd.DataFrame,
                             sector_closes: pd.Series = None) -> tuple:
    votes = []

    def _ret(series, days):
        if series is None or len(series) < days + 1:
            return None
        return series.iloc[-1] / series.iloc[-days - 1] - 1

    tk_1m, tk_3m = _ret(ticker_closes, 21), _ret(ticker_closes, 63)

    if benchmarks is not None and not benchmarks.empty:
        if "SPY" in benchmarks.columns:
            spy_1m, spy_3m = _ret(benchmarks["SPY"], 21), _ret(benchmarks["SPY"], 63)
            if tk_1m is not None and spy_1m is not None:
                v = 1 if tk_1m > spy_1m else -1
                votes.append((2.0, v, f"1M return {tk_1m * 100:+.1f}% vs SPY {spy_1m * 100:+.1f}%"))
            if tk_3m is not None and spy_3m is not None:
                v = 1 if tk_3m > spy_3m else -1
                votes.append((1.5, v, f"3M return {tk_3m * 100:+.1f}% vs SPY {spy_3m * 100:+.1f}%"))
            if tk_1m is not None and tk_3m is not None and spy_1m is not None and spy_3m is not None:
                rs_1m, rs_3m = tk_1m - spy_1m, tk_3m - spy_3m
                v = 1 if rs_1m > rs_3m else -1
                tag = "accelerating" if v > 0 else "decelerating"
                votes.append((1.0, v, f"Relative strength {tag} (1M RS {rs_1m * 100:+.1f}pp vs 3M RS {rs_3m * 100:+.1f}pp)"))

        if "QQQ" in benchmarks.columns:
            qqq_1m = _ret(benchmarks["QQQ"], 21)
            if tk_1m is not None and qqq_1m is not None:
                v = 1 if tk_1m > qqq_1m else -1
                votes.append((1.0, v, f"1M return {tk_1m * 100:+.1f}% vs QQQ {qqq_1m * 100:+.1f}%"))

    if sector_closes is not None:
        sec_1m = _ret(sector_closes, 21)
        if tk_1m is not None and sec_1m is not None:
            v = 1 if tk_1m > sec_1m else -1
            votes.append((1.5, v, f"1M return {tk_1m * 100:+.1f}% vs sector {sec_1m * 100:+.1f}%"))

    return _score_from_votes(votes)


def score_fundamental(fund: dict) -> tuple:
    votes = []
    rg = fund.get("revenue_growth")
    if rg is not None:
        votes.append((1.5, 1 if rg > 0.05 else (-1 if rg < 0 else 0), f"Revenue growth {rg * 100:+.1f}%"))
    eg = fund.get("earnings_growth")
    if eg is not None:
        votes.append((1.5, 1 if eg > 0.05 else (-1 if eg < 0 else 0), f"Earnings growth {eg * 100:+.1f}%"))
    pm = fund.get("profit_margins")
    if pm is not None:
        votes.append((1.0, 1 if pm > 0.10 else (-1 if pm < 0 else 0), f"Profit margin {pm * 100:.1f}%"))
    roe = fund.get("return_on_equity")
    if roe is not None:
        votes.append((1.0, 1 if roe > 0.15 else (-1 if roe < 0 else 0), f"ROE {roe * 100:.1f}%"))
    dte = fund.get("debt_to_equity")
    if dte is not None:
        votes.append((0.75, -1 if dte > 150 else (1 if dte < 50 else 0), f"Debt/Equity {dte:.0f}%"))
    peg = fund.get("peg_ratio")
    if peg is not None and peg > 0:
        votes.append((0.75, 1 if peg < 1.5 else (-1 if peg > 3 else 0), f"PEG ratio {peg:.2f}"))
    rec = fund.get("recommendation_mean")
    if rec is not None:
        votes.append((1.5, 1 if rec < 2.3 else (-1 if rec > 3.2 else 0),
                      f"Analyst mean recommendation {rec:.1f} (1=Strong Buy, 5=Strong Sell)"))
    return _score_from_votes(votes)


def score_options_flow(chains: dict, expirations: list, spot: float) -> tuple:
    if not chains or not expirations:
        return 50.0, ["Options data unavailable"]

    votes = []
    near = expirations[:2]
    total_call_vol = total_put_vol = 0
    unusual_bull = unusual_bear = 0
    gex_total = 0.0

    for exp in near:
        cd = chains.get(exp)
        if not cd:
            continue
        pcr = calculate_pcr(cd["calls"], cd["puts"])
        total_call_vol += pcr["call_volume"]
        total_put_vol += pcr["put_volume"]
        uu = detect_unusual_activity(cd["calls"], cd["puts"], "", spot)
        if not uu.empty:
            unusual_bull += int((uu["Sentiment"] == "Bullish").sum())
            unusual_bear += int((uu["Sentiment"] == "Bearish").sum())
        try:
            gex_total += calculate_gex(cd["calls"], cd["puts"], spot, exp)
        except Exception:
            pass

    if total_call_vol + total_put_vol > 0:
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 2.0
        v = 1 if pcr_vol < 0.8 else (-1 if pcr_vol > 1.2 else 0)
        votes.append((1.5, v, f"Put/Call volume ratio {pcr_vol:.2f} on nearest expirations"))

    if unusual_bull + unusual_bear > 0:
        v = 1 if unusual_bull > unusual_bear else (-1 if unusual_bear > unusual_bull else 0)
        votes.append((1.5, v, f"Unusual options activity: {unusual_bull} bullish vs {unusual_bear} bearish flags"))

    v = 1 if gex_total > 0 else (-1 if gex_total < 0 else 0)
    tag = "dealers long gamma, dampens moves" if gex_total > 0 else "dealers short gamma, amplifies moves"
    votes.append((1.0, v, f"Net gamma exposure {gex_total:+,.0f} ({tag})"))

    return _score_from_votes(votes)


def score_ml(pred: dict) -> tuple:
    bull = pred.get("bullish_prob", 50)
    acc = pred.get("accuracy", 0)
    v = 1 if bull > 55 else (-1 if bull < 45 else 0)
    return _score_from_votes([(2.0, v, f"XGBoost ensemble: {bull:.1f}% bullish probability (backtest accuracy {acc:.1f}%)")])


# ─── Trailing stats / regime helpers ────────────────────────────────────────
def _annualized_sharpe_sortino(closes: pd.Series):
    rets = closes.pct_change().dropna()
    if len(rets) < 30:
        return None, None
    mean = rets.mean() * 252
    std = rets.std() * np.sqrt(252)
    sharpe = mean / std if std > 0 else 0
    downside = rets[rets < 0]
    dstd = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
    sortino = mean / dstd if dstd > 0 else 0
    return round(float(sharpe), 2), round(float(sortino), 2)


def _trend_stage(df: pd.DataFrame) -> str:
    if len(df) < 160 or "Close" not in df.columns:
        return "Unknown"
    sma150 = df["Close"].rolling(150).mean()
    if sma150.iloc[-30:].isna().any():
        return "Unknown"
    close = df["Close"].iloc[-1]
    slope = sma150.iloc[-1] - sma150.iloc[-30]
    above, rising = close > sma150.iloc[-1], slope > 0
    if above and rising:
        return "Stage 2 (Advancing)"
    if not above and not rising:
        return "Stage 4 (Declining)"
    if above and not rising:
        return "Stage 3 (Topping)"
    return "Stage 1 (Basing)"


def _darvas_box(df: pd.DataFrame, window: int = 20) -> dict:
    if len(df) < window:
        return {"high": None, "low": None}
    recent = df.iloc[-window:]
    return {"high": round(float(recent["High"].max()), 2), "low": round(float(recent["Low"].min()), 2)}


def _short_squeeze_probability(fund: dict, rs_score: float, volume_score: float):
    spf, sr = fund.get("short_percent_float"), fund.get("short_ratio")
    if spf is None and sr is None:
        return None
    score, n = 0.0, 0
    if spf is not None:
        score += min(spf * 100 / 20, 1.0) * 100
        n += 1
    if sr is not None:
        score += min(sr / 10, 1.0) * 100
        n += 1
    base = score / n if n else 0
    blended = base * 0.6 + rs_score * 0.2 + volume_score * 0.2
    return round(min(blended, 100), 1)


def _gamma_squeeze_probability(chains: dict, expirations: list, spot: float):
    if not chains or not expirations:
        return None
    exp = expirations[0]
    cd = chains.get(exp)
    if not cd:
        return None
    try:
        gex = calculate_gex(cd["calls"], cd["puts"], spot, exp)
        calls = cd["calls"]
        near = calls[(calls["strike"] >= spot * 0.97) & (calls["strike"] <= spot * 1.05)]
        oi_near = near["openInterest"].fillna(0).sum()
        total_oi = calls["openInterest"].fillna(0).sum() + cd["puts"]["openInterest"].fillna(0).sum()
        conc_ratio = oi_near / total_oi if total_oi > 0 else 0
        gex_component = 100 if gex < 0 else max(0, 50 - gex / 1000)
        score = gex_component * 0.6 + conc_ratio * 100 * 0.4
        return round(min(max(score, 0), 100), 1)
    except Exception:
        return None


def _classify(ai_score, risk_score, trend_stage, agreement_pct, bull_pct, bear_pct) -> str:
    if risk_score > 75 and ai_score < 55:
        return "High Risk"
    if ai_score >= 75 and agreement_pct >= 75:
        return "Strong Buy"
    if ai_score >= 62:
        return "Buy"
    if bear_pct >= 60 and ai_score < 40:
        return "Short Candidate"
    if ai_score <= 30:
        return "Avoid"
    if "Stage 2" in trend_stage and ai_score >= 55:
        return "Momentum Leader"
    if "Stage 1" in trend_stage and 45 <= ai_score < 62:
        return "Breakout Candidate"
    if bull_pct < 55 and bear_pct < 55 and ai_score < 62:
        return "Mean Reversion Candidate"
    if 45 <= ai_score < 62:
        return "Watchlist"
    return "Neutral"


# ─── Main entry points ───────────────────────────────────────────────────────
def analyze_ticker(ticker: str) -> dict:
    info = get_stock_info(ticker)
    df_daily = get_ohlcv(ticker, "2y", "1d")
    if df_daily.empty or len(df_daily) < 60:
        return {"ticker": ticker, "error": "Insufficient price history"}

    df_daily = add_all_indicators(df_daily)
    df_daily = add_scoring_indicators(df_daily)

    close = float(df_daily["Close"].iloc[-1])
    atr = float(df_daily["ATR"].iloc[-1]) if "ATR" in df_daily.columns and not pd.isna(df_daily["ATR"].iloc[-1]) else close * 0.02

    benchmarks = get_benchmark_closes("1y")
    sector_name = SECTOR_NAME_MAP.get(info.get("sector"))
    sector_etf = SECTOR_ETFS.get(sector_name) if sector_name else None
    sector_closes = None
    if sector_etf:
        df_sector = get_ohlcv(sector_etf, "1y", "1d")
        if not df_sector.empty:
            sector_closes = df_sector["Close"]

    fund = get_fundamentals(ticker)
    chains, expirations = get_options_chain(ticker)
    pred = get_ai_prediction(ticker, df_daily)

    trend_score, trend_reasons = score_trend(df_daily)
    momentum_score, momentum_reasons = score_momentum(df_daily)
    volume_score, volume_reasons = score_volume(df_daily)
    risk_score, risk_reasons = score_volatility_risk(df_daily, info)
    rs_score, rs_reasons = score_relative_strength(df_daily["Close"], benchmarks, sector_closes)
    fund_score, fund_reasons = score_fundamental(fund)
    flow_score, flow_reasons = score_options_flow(chains, expirations, close)
    ml_score, ml_reasons = score_ml(pred)

    categories = {
        "trend":             (trend_score, trend_reasons),
        "momentum":          (momentum_score, momentum_reasons),
        "volume":            (volume_score, volume_reasons),
        "volatility":        (100 - risk_score, risk_reasons),
        "relative_strength": (rs_score, rs_reasons),
        "fundamental":       (fund_score, fund_reasons),
        "options_flow":      (flow_score, flow_reasons),
        "ml":                (ml_score, ml_reasons),
    }

    ai_score = round(sum(categories[c][0] * w for c, w in WEIGHTS.items()), 1)

    bullish_dir = ai_score >= 50
    aligned = sum(1 for c in categories if (categories[c][0] >= 50) == bullish_dir)
    agreement_pct = aligned / len(categories) * 100

    bull_prob = pred.get("bullish_prob", 50) / 100
    trend_momentum_avg = (trend_score + momentum_score) / 200
    bull_adj = bull_prob * 0.6 + trend_momentum_avg * 0.4

    adx_last = float(df_daily["ADX"].iloc[-1]) if "ADX" in df_daily.columns and not pd.isna(df_daily["ADX"].iloc[-1]) else 20.0
    bb_width_series = df_daily["BB_Width"].dropna() if "BB_Width" in df_daily.columns else pd.Series(dtype=float)
    bb_width_pct = (bb_width_series < bb_width_series.iloc[-1]).mean() * 100 if len(bb_width_series) > 30 else 50.0

    sideways_raw = max(0.0, (25 - adx_last) / 25) * 0.6 + max(0.0, (40 - bb_width_pct) / 40) * 0.4
    sideways_prob = min(max(sideways_raw, 0.0), 0.6)

    remaining = 1 - sideways_prob
    bull_final = remaining * bull_adj
    bear_final = remaining * (1 - bull_adj)
    total = bull_final + bear_final + sideways_prob
    bull_pct = round(bull_final / total * 100, 1)
    bear_pct = round(bear_final / total * 100, 1)
    sideways_pct = round(100 - bull_pct - bear_pct, 1)

    confidence = round((abs(bull_pct - 50) / 50 * 0.5 + agreement_pct / 100 * 0.5) * 100, 1)

    sharpe, sortino = _annualized_sharpe_sortino(df_daily["Close"].tail(252))
    atr_series = (df_daily["ATR"] / df_daily["Close"] * 100).dropna()
    atr_rank = round(float((atr_series < atr_series.iloc[-1]).mean() * 100), 1) if len(atr_series) > 30 else None

    corr = {}
    if benchmarks is not None and not benchmarks.empty:
        tk_rets = df_daily["Close"].pct_change().dropna()
        for name in benchmarks.columns:
            b_rets = benchmarks[name].pct_change().dropna()
            joined = pd.concat([tk_rets, b_rets], axis=1, join="inner").dropna()
            if len(joined) > 30:
                corr[name] = round(float(joined.iloc[:, 0].corr(joined.iloc[:, 1])), 2)

    trend_stage = _trend_stage(df_daily)
    darvas = _darvas_box(df_daily)
    short_squeeze = _short_squeeze_probability(fund, rs_score, volume_score)
    gamma_squeeze = _gamma_squeeze_probability(chains, expirations, close)

    entry = close
    stop = round(entry - 1.5 * atr, 2)
    trailing_stop = round(close - 2 * atr, 2)
    tp1, tp2, tp3 = round(entry + 2 * atr, 2), round(entry + 3.5 * atr, 2), round(entry + 5 * atr, 2)
    rrr = calculate_rrr(entry, stop, tp2)

    win_prob = bull_pct / 100
    R = rrr["rrr"] if rrr["rrr"] > 0 else 0.01
    kelly_raw = win_prob - (1 - win_prob) / R
    kelly_pct = round(min(max(kelly_raw, 0.0), 1.0) * KELLY_FRACTION * 100, 2)
    kelly_pct = min(kelly_pct, MAX_KELLY_PCT)

    if adx_last > 25 and bb_width_pct < 50:
        holding_period = "Position Trade (weeks-months)"
    elif adx_last > 20:
        holding_period = "Swing Trade (days-weeks)"
    else:
        holding_period = "Mean-Reversion / Short Swing (days)"

    classification = _classify(ai_score, risk_score, trend_stage, agreement_pct, bull_pct, bear_pct)

    expected_return_pct = round((bull_pct / 100 * (atr / close * 2) - bear_pct / 100 * (atr / close * 1.5)) * 100, 2)
    expected_drawdown_pct = round(atr / close * 1.5 * 100, 2)

    return {
        "ticker": ticker,
        "name": info.get("name", ticker),
        "sector": info.get("sector", "N/A"),
        "price": round(close, 2),
        "ai_score": ai_score,
        "classification": classification,
        "bull_pct": bull_pct, "bear_pct": bear_pct, "sideways_pct": sideways_pct,
        "confidence": confidence,
        "signal_strength": round(agreement_pct, 1),
        "risk_score": risk_score,
        "category_scores": {c: categories[c][0] for c in categories},
        "category_reasons": {c: categories[c][1] for c in categories},
        "trend_stage": trend_stage,
        "darvas_box": darvas,
        "short_squeeze_prob": short_squeeze,
        "gamma_squeeze_prob": gamma_squeeze,
        "sharpe": sharpe, "sortino": sortino,
        "atr_rank": atr_rank, "vol_rank": atr_rank,
        "beta": info.get("beta"),
        "correlations": corr,
        "trade_plan": {
            "entry": round(entry, 2), "stop": stop, "trailing_stop": trailing_stop,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "reward_risk": rrr["rrr"], "kelly_pct": kelly_pct,
            "holding_period": holding_period,
        },
        "expected_return_pct": expected_return_pct,
        "expected_drawdown_pct": expected_drawdown_pct,
        "ml_prediction": pred,
    }


@st.cache_data(ttl=300, show_spinner=False)
def run_scan(tickers: tuple) -> list:
    results = []
    for tk in tickers:
        try:
            results.append(analyze_ticker(tk))
        except Exception as e:
            results.append({"ticker": tk, "error": str(e)})
    return results
