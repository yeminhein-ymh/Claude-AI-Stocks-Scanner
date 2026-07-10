import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # Price momentum
    feat["ret_1d"]  = df["Close"].pct_change(1)
    feat["ret_3d"]  = df["Close"].pct_change(3)
    feat["ret_5d"]  = df["Close"].pct_change(5)
    feat["ret_10d"] = df["Close"].pct_change(10)

    # EMA ratios
    for p in [9, 20, 50]:
        col = f"EMA_{p}"
        if col in df.columns:
            feat[f"price_ema{p}"] = df["Close"] / df[col] - 1

    # RSI
    if "RSI" in df.columns:
        feat["rsi"] = df["RSI"] / 100.0

    # MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        feat["macd_diff"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger
    if "BB_Width" in df.columns:
        feat["bb_width"] = df["BB_Width"]
    if "BB_Middle" in df.columns and "BB_Upper" in df.columns:
        feat["bb_pos"] = (df["Close"] - df["BB_Middle"]) / (df["BB_Upper"] - df["BB_Middle"] + 1e-9)

    # ATR normalised
    if "ATR" in df.columns:
        feat["atr_pct"] = df["ATR"] / df["Close"]

    # Volume
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean()
        feat["vol_ratio"] = df["Volume"] / (vol_ma + 1)

    # High-low range
    feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    return feat


@st.cache_resource(show_spinner=False)
def train_model(ticker: str, df: pd.DataFrame):
    if len(df) < 100:
        return None, None, 0.5

    features = _make_features(df)
    target   = (df["Close"].shift(-1) > df["Close"]).astype(int)

    combined = features.join(target.rename("target")).dropna()
    if len(combined) < 80:
        return None, None, 0.5

    X = combined.drop("target", axis=1)
    y = combined["target"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                               random_state=42, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    accuracy = model.score(X_te, y_te)

    return model, scaler, round(accuracy, 3)


def get_ai_prediction(ticker: str, df: pd.DataFrame) -> dict:
    result = {
        "bullish_prob": 0.5, "bearish_prob": 0.5,
        "confidence": 0.0, "accuracy": 0.0,
        "expected_move_pct": 0.0, "signal": "NEUTRAL",
        "trend_strength": 0.0, "feature_names": [],
    }
    try:
        model, scaler, accuracy = train_model(ticker, df)
        if model is None:
            return result

        features = _make_features(df)
        last_row = features.iloc[[-1]].copy()
        last_row_clean = last_row.fillna(0)

        X_pred = scaler.transform(last_row_clean)
        proba  = model.predict_proba(X_pred)[0]

        bull_prob = float(proba[1])
        bear_prob = float(proba[0])
        confidence = abs(bull_prob - 0.5) * 2

        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else df["Close"].iloc[-1] * 0.01
        expected_move = atr / df["Close"].iloc[-1] * 100 * (1 if bull_prob > 0.5 else -1)

        # Trend strength from EMAs
        ema_scores = []
        for p in [9, 20, 50, 200]:
            col = f"EMA_{p}"
            if col in df.columns:
                ema_scores.append(1 if df["Close"].iloc[-1] > df[col].iloc[-1] else 0)
        trend_strength = np.mean(ema_scores) if ema_scores else 0.5

        if bull_prob > 0.60:
            signal = "STRONG BUY"
        elif bull_prob > 0.55:
            signal = "BUY"
        elif bull_prob < 0.40:
            signal = "STRONG SELL"
        elif bull_prob < 0.45:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        # Feature importance
        fi = model.feature_importances_
        fn = list(last_row_clean.columns)
        top_idx = np.argsort(fi)[::-1][:5]
        top_features = [(fn[i], round(float(fi[i]) * 100, 1)) for i in top_idx]

        result.update({
            "bullish_prob": round(bull_prob * 100, 1),
            "bearish_prob": round(bear_prob * 100, 1),
            "confidence":   round(confidence * 100, 1),
            "accuracy":     round(accuracy * 100, 1),
            "expected_move_pct": round(expected_move, 2),
            "signal":       signal,
            "trend_strength": round(trend_strength * 100, 1),
            "top_features": top_features,
        })
    except Exception as e:
        pass
    return result


def get_scalp_levels(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {}
    last = df["Close"].iloc[-1]
    atr  = df["ATR"].iloc[-1] if "ATR" in df.columns else last * 0.005
    return {
        "price":    round(last, 2),
        "entry":    round(last, 2),
        "stop_long":  round(last - 1.5 * atr, 2),
        "stop_short": round(last + 1.5 * atr, 2),
        "tp1_long":   round(last + 2 * atr, 2),
        "tp2_long":   round(last + 3.5 * atr, 2),
        "tp1_short":  round(last - 2 * atr, 2),
        "tp2_short":  round(last - 3.5 * atr, 2),
        "atr":        round(atr, 2),
    }
