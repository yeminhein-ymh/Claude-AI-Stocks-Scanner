import pandas as pd
import numpy as np
import ta
from config import EMA_COLORS


def add_ema(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    for p in periods:
        df[f"EMA_{p}"] = ta.trend.EMAIndicator(close=df["Close"], window=p).ema_indicator()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=period).rsi()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, dev: float = 2.0) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(close=df["Close"], window=window, window_dev=dev)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=period
    ).average_true_range()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["TP"] = typical_price
    df["TV"] = typical_price * df["Volume"]
    if df.index.tz is not None:
        dates = df.index.normalize()
    else:
        dates = df.index.floor("D")
    df["_date"] = dates
    df["CumTV"] = df.groupby("_date")["TV"].cumsum()
    df["CumV"]  = df.groupby("_date")["Volume"].cumsum()
    df["VWAP"]  = df["CumTV"] / df["CumV"]
    df.drop(columns=["TP", "TV", "CumTV", "CumV", "_date"], inplace=True)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    df["ADX"]     = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    return df


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["MFI"] = ta.volume.MFIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=period
    ).money_flow_index()
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["CCI"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=period).cci()
    return df


def add_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=period, smooth_window=smooth
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    return df


def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    df["ROC"] = ta.momentum.ROCIndicator(close=df["Close"], window=period).roc()
    return df


def add_scoring_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the extra indicators the scoring engine needs on top of add_all_indicators."""
    if len(df) < 30:
        return df
    df = add_adx(df)
    df = add_obv(df)
    df = add_cci(df)
    df = add_stochastic(df)
    df = add_roc(df)
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df = add_mfi(df)
    return df


def get_fibonacci_levels(df: pd.DataFrame) -> dict:
    high = df["High"].max()
    low  = df["Low"].min()
    diff = high - low
    return {
        "0%":    low,
        "23.6%": low + 0.236 * diff,
        "38.2%": low + 0.382 * diff,
        "50%":   low + 0.500 * diff,
        "61.8%": low + 0.618 * diff,
        "78.6%": low + 0.786 * diff,
        "100%":  high,
    }


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 30:
        return df
    df = add_ema(df, [9, 20, 50, 200])
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df = add_vwap(df)
    return df


def get_scalping_signals(df: pd.DataFrame) -> list:
    signals = []
    if len(df) < 3:
        return signals

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    # VWAP bounce
    if "VWAP" in df.columns:
        if prev["Close"] < prev["VWAP"] and last["Close"] > last["VWAP"]:
            signals.append({"signal": "VWAP Reclaim", "type": "BULLISH",
                             "strength": "Medium", "price": last["Close"]})
        elif prev["Close"] > prev["VWAP"] and last["Close"] < last["VWAP"]:
            signals.append({"signal": "VWAP Break", "type": "BEARISH",
                             "strength": "Medium", "price": last["Close"]})

    # EMA crossover (9/20)
    if "EMA_9" in df.columns and "EMA_20" in df.columns:
        if prev["EMA_9"] < prev["EMA_20"] and last["EMA_9"] > last["EMA_20"]:
            signals.append({"signal": "EMA 9×20 Cross Up", "type": "BULLISH",
                             "strength": "Strong", "price": last["Close"]})
        elif prev["EMA_9"] > prev["EMA_20"] and last["EMA_9"] < last["EMA_20"]:
            signals.append({"signal": "EMA 9×20 Cross Down", "type": "BEARISH",
                             "strength": "Strong", "price": last["Close"]})

    # RSI extremes
    if "RSI" in df.columns and not pd.isna(last["RSI"]):
        if last["RSI"] < 30:
            signals.append({"signal": "RSI Oversold", "type": "BULLISH",
                             "strength": "Weak", "price": last["Close"]})
        elif last["RSI"] > 70:
            signals.append({"signal": "RSI Overbought", "type": "BEARISH",
                             "strength": "Weak", "price": last["Close"]})

    # Volume spike
    if "Volume" in df.columns:
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        if not pd.isna(avg_vol) and avg_vol > 0:
            if last["Volume"] > avg_vol * 2.5:
                direction = "BULLISH" if last["Close"] > last["Open"] else "BEARISH"
                signals.append({"signal": "Volume Spike", "type": direction,
                                 "strength": "Strong", "price": last["Close"]})

    # Bollinger squeeze breakout
    if "BB_Upper" in df.columns:
        if last["Close"] > last["BB_Upper"]:
            signals.append({"signal": "BB Upper Breakout", "type": "BULLISH",
                             "strength": "Medium", "price": last["Close"]})
        elif last["Close"] < last["BB_Lower"]:
            signals.append({"signal": "BB Lower Breakdown", "type": "BEARISH",
                             "strength": "Medium", "price": last["Close"]})

    return signals
