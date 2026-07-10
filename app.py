import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

import config
from modules.data_fetcher import (
    get_stock_info, get_ohlcv, get_options_chain, get_macro_data,
    get_sector_data, get_current_price,
)
from modules.indicators import add_all_indicators, get_fibonacci_levels, get_scalping_signals
from modules.options_analyzer import (
    enrich_chain_greeks, calculate_max_pain, calculate_pcr,
    calculate_gex, detect_unusual_activity,
)
from modules.ai_engine import get_ai_prediction, get_scalp_levels, train_model
from modules.risk_manager import render_risk_panel
from modules.scoring_engine import run_scan

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0E1117; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: #1A1F2E; border-radius: 8px; padding: 10px; border-left: 3px solid #00FF88; }
    .stMetric label { color: #AAB4C8 !important; font-size: 0.75rem; }
    .bull-badge { background: #003D1A; color: #00FF88; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .bear-badge { background: #3D0000; color: #FF4444; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .neutral-badge { background: #2A2A1A; color: #FFD700; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .signal-card { background: #1A1F2E; border-radius: 10px; padding: 12px; margin: 4px 0; }
    div[data-testid="stSidebarContent"] { background: #141921; }
    .stTabs [data-baseweb="tab-list"] { background: #141921; }
    .stTabs [data-baseweb="tab"] { color: #AAB4C8; }
    .stTabs [aria-selected="true"] { color: #00FF88 !important; border-bottom: 2px solid #00FF88; }
    h1, h2, h3 { color: #FAFAFA; }
    .stDataFrame { background: #1A1F2E; }
    hr { border-color: #2A3040; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = config.DEFAULT_TICKERS.copy()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = config.DEFAULT_TICKERS[0]
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Options Dashboard")
    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.toggle("🔄 Auto Refresh (60s)", value=False)
    if auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed > 60:
            st.session_state.last_refresh = time.time()
            st.rerun()
        st.caption(f"Next refresh in {max(0, 60 - int(elapsed))}s")

    st.markdown("### 📋 Watchlist")
    new_ticker = st.text_input("Add Ticker", placeholder="e.g. AAPL").upper().strip()
    if st.button("➕ Add", use_container_width=True) and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.success(f"Added {new_ticker}")
        else:
            st.warning(f"{new_ticker} already in watchlist")

    st.caption("Click ticker to select it as active:")
    to_remove = None
    for tk in st.session_state.watchlist:
        c1, c2 = st.columns([4, 1])
        if c1.button(
            f"{'▶ ' if tk == st.session_state.selected_ticker else '  '}{tk}",
            key=f"sel_{tk}", use_container_width=True
        ):
            st.session_state.selected_ticker = tk
        if c2.button("✕", key=f"del_{tk}"):
            to_remove = tk
    if to_remove and len(st.session_state.watchlist) > 1:
        st.session_state.watchlist.remove(to_remove)
        if st.session_state.selected_ticker == to_remove:
            st.session_state.selected_ticker = st.session_state.watchlist[0]
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Chart Settings")
    timeframe = st.selectbox("Timeframe", list(config.TIMEFRAME_CONFIG.keys()), index=4)
    period, interval = config.TIMEFRAME_CONFIG[timeframe]

    show_ema    = st.multiselect("EMAs", [9, 20, 50, 200], default=[9, 20, 50])
    show_bb     = st.checkbox("Bollinger Bands", value=True)
    show_vwap   = st.checkbox("VWAP", value=True)
    show_fib    = st.checkbox("Fibonacci", value=False)
    show_rsi    = st.checkbox("RSI Panel", value=True)
    show_macd   = st.checkbox("MACD Panel", value=True)
    show_vol    = st.checkbox("Volume Panel", value=True)

    st.markdown("---")
    st.caption(f"Active: **{st.session_state.selected_ticker}** | TF: **{timeframe}**")

TICKER = st.session_state.selected_ticker

# ─── Main Header ────────────────────────────────────────────────────────────
info = get_stock_info(TICKER)
chg_color = "#00FF88" if info["change_pct"] >= 0 else "#FF4444"
chg_sign  = "+" if info["change_pct"] >= 0 else ""

col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.markdown(
        f"## {info['name']} &nbsp; "
        f"<span style='color:#FAFAFA;font-size:1.6rem'>${info['price']}</span> &nbsp;"
        f"<span style='color:{chg_color};font-size:1.2rem'>{chg_sign}{info['change_pct']}%</span>",
        unsafe_allow_html=True,
    )
with col_h2:
    st.caption("Sector")
    st.write(info["sector"] or "N/A")
with col_h3:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🧠 AI Scanner", "📊 Watchlist", "📈 Charts", "🔗 Options Chain",
    "💥 Smart Money", "🤖 AI Signals", "⚡ Scalping",
    "🏦 Heatmap", "📉 Risk Mgmt",
])

# ═══════════════════════════════════════════════════════════
# TAB 0 — AI SCANNER
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("🧠 AI Scanner — Institutional Multi-Factor Ranking")
    st.caption(
        "Every score is a weighted vote across Trend, Momentum, Volume, Volatility, "
        "Relative Strength, Fundamentals, Options Flow, and an ML ensemble — never a single indicator."
    )

    run_clicked = st.button("🔄 Run AI Scan", use_container_width=False)
    if run_clicked or "scanner_results" not in st.session_state:
        with st.spinner(f"Scanning {len(st.session_state.watchlist)} tickers across 8 factor categories..."):
            st.session_state.scanner_results = run_scan(tuple(st.session_state.watchlist))
            st.session_state.scanner_tickers = tuple(st.session_state.watchlist)

    results = st.session_state.get("scanner_results", [])
    scanned_tickers = st.session_state.get("scanner_tickers", ())
    if scanned_tickers and scanned_tickers != tuple(st.session_state.watchlist):
        st.info("Watchlist has changed since the last scan — click 'Run AI Scan' to refresh.")

    if not results:
        st.info("Click 'Run AI Scan' to rank your watchlist.")
    else:
        valid   = [r for r in results if "error" not in r]
        errored = [r for r in results if "error" in r]
        if errored:
            st.warning("Skipped (insufficient data): " + ", ".join(f"{r['ticker']} ({r['error']})" for r in errored))

        if valid:
            table_rows = []
            for r in valid:
                cs = r["category_scores"]
                table_rows.append({
                    "Ticker":       r["ticker"],
                    "AI Score":     r["ai_score"],
                    "Class":        r["classification"],
                    "Bull %":       r["bull_pct"],
                    "Bear %":       r["bear_pct"],
                    "Sideways %":   r["sideways_pct"],
                    "Confidence":   r["confidence"],
                    "Risk":         r["risk_score"],
                    "Trend":        cs["trend"],
                    "Momentum":     cs["momentum"],
                    "Volume":       cs["volume"],
                    "RS":           cs["relative_strength"],
                    "Fund":         cs["fundamental"],
                    "Flow":         cs["options_flow"],
                    "ML":           cs["ml"],
                    "Exp Return %": r["expected_return_pct"],
                    "R:R":          r["trade_plan"]["reward_risk"],
                    "Trend Stage":  r["trend_stage"],
                })
            df_scan = pd.DataFrame(table_rows).sort_values("AI Score", ascending=False).reset_index(drop=True)

            def _color_class(val):
                colors = {
                    "Strong Buy": "#00FF88", "Buy": "#66FF99", "Momentum Leader": "#00CED1",
                    "Breakout Candidate": "#1E90FF", "Watchlist": "#FFD700", "Neutral": "#AAB4C8",
                    "Mean Reversion Candidate": "#FFA500", "High Risk": "#FF8C00",
                    "Short Candidate": "#FF6B6B", "Avoid": "#FF4444",
                }
                return f"color: {colors.get(val, '#FAFAFA')}; font-weight: bold"

            def _color_score(val):
                try:
                    v = float(val)
                    if v >= 65:
                        return "color: #00FF88"
                    if v <= 35:
                        return "color: #FF4444"
                    return "color: #FFD700"
                except Exception:
                    return ""

            score_cols = ["AI Score", "Trend", "Momentum", "Volume", "RS", "Fund", "Flow", "ML", "Confidence"]
            st.dataframe(
                df_scan.style.map(_color_class, subset=["Class"]).map(_color_score, subset=score_cols),
                use_container_width=True, height=min(80 + 35 * len(df_scan), 500),
            )

            st.divider()
            st.subheader("🔍 Deep Dive")
            drill_ticker = st.selectbox(
                "Select ticker for full breakdown",
                [r["ticker"] for r in valid], key="scanner_drill",
            )
            r = next(x for x in valid if x["ticker"] == drill_ticker)

            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("AI Score", f"{r['ai_score']}/100")
            h2.metric("Classification", r["classification"])
            h3.metric("Confidence", f"{r['confidence']}%")
            h4.metric("Risk Score", f"{r['risk_score']}/100")
            h5.metric("Trend Stage", r["trend_stage"])

            p1, p2, p3 = st.columns(3)
            p1.metric("Bullish Probability", f"{r['bull_pct']}%")
            p2.metric("Bearish Probability", f"{r['bear_pct']}%")
            p3.metric("Sideways Probability", f"{r['sideways_pct']}%")

            st.divider()
            st.markdown("#### Category Breakdown (WHY)")
            cat_labels = {
                "trend": "📈 Trend", "momentum": "⚡ Momentum", "volume": "📊 Volume",
                "volatility": "🌊 Volatility/Stability", "relative_strength": "💪 Relative Strength",
                "fundamental": "🏦 Fundamental", "options_flow": "🔗 Options Flow", "ml": "🤖 ML Ensemble",
            }
            cols = st.columns(4)
            for i, (key, label) in enumerate(cat_labels.items()):
                score   = r["category_scores"][key]
                reasons = r["category_reasons"][key]
                with cols[i % 4]:
                    with st.expander(f"{label}: {score}/100"):
                        for reason in reasons:
                            st.markdown(f"- {reason}")

            st.divider()
            st.markdown("#### Trade Plan")
            tp = r["trade_plan"]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Entry", f"${tp['entry']}")
            t2.metric("Stop Loss", f"${tp['stop']}")
            t3.metric("Trailing Stop", f"${tp['trailing_stop']}")
            t4.metric("Reward:Risk", f"1:{tp['reward_risk']}")
            t5, t6, t7, t8 = st.columns(4)
            t5.metric("TP1", f"${tp['tp1']}")
            t6.metric("TP2", f"${tp['tp2']}")
            t7.metric("TP3", f"${tp['tp3']}")
            t8.metric("Kelly Size (¼)", f"{tp['kelly_pct']}%")
            st.caption(f"Recommended holding period: **{tp['holding_period']}**")

            st.divider()
            st.markdown("#### Structure & Squeeze Signals")
            db = r["darvas_box"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Darvas Box", f"${db['low']} – ${db['high']}" if db["high"] else "N/A")
            s2.metric("Short Squeeze Prob",
                      f"{r['short_squeeze_prob']}%" if r["short_squeeze_prob"] is not None else "N/A (no short data)")
            s3.metric("Gamma Squeeze Prob",
                      f"{r['gamma_squeeze_prob']}%" if r["gamma_squeeze_prob"] is not None else "N/A")
            s4.metric("Beta", f"{r['beta']:.2f}" if r["beta"] else "N/A")

            st.divider()
            st.markdown("#### Trailing Statistics & Correlation")
            st1, st2, st3, st4 = st.columns(4)
            st1.metric("Sharpe (1Y)", r["sharpe"] if r["sharpe"] is not None else "N/A")
            st2.metric("Sortino (1Y)", r["sortino"] if r["sortino"] is not None else "N/A")
            st3.metric("ATR/Vol Rank", f"{r['atr_rank']}th pct" if r["atr_rank"] is not None else "N/A")
            st4.metric("Expected Move", f"{r['expected_return_pct']:+.2f}%")
            if r["correlations"]:
                st.caption("Correlation: " + " · ".join(f"{k} {v:+.2f}" for k, v in r["correlations"].items()))

            st.caption(
                "⚠️ Scores are model estimates from free market data, not guarantees. Dark pool activity, "
                "real institutional 13F flow, Wyckoff/Elliott Wave phases, and CAN SLIM/SEPA scores require "
                "paid data feeds and are not yet included."
            )

# ═══════════════════════════════════════════════════════════
# TAB 1 — WATCHLIST OVERVIEW
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Watchlist Overview")

    rows = []
    with st.spinner("Loading watchlist..."):
        for tk in st.session_state.watchlist:
            d = get_stock_info(tk)
            mktcap = d["market_cap"]
            cap_str = f"${mktcap/1e12:.2f}T" if mktcap > 1e12 else f"${mktcap/1e9:.1f}B" if mktcap > 1e9 else f"${mktcap/1e6:.0f}M"
            rows.append({
                "Ticker":     d["ticker"],
                "Price":      f"${d['price']:.2f}",
                "Change %":   d["change_pct"],
                "Volume":     f"{d['volume']:,}",
                "Mkt Cap":    cap_str,
                "Beta":       d.get("beta", "—"),
                "52W High":   f"${d['week52_hi']:.2f}",
                "52W Low":    f"${d['week52_lo']:.2f}",
                "Sector":     d.get("sector", "N/A"),
            })

    df_watch = pd.DataFrame(rows)

    def color_change(val):
        try:
            v = float(val)
            return "color: #00FF88" if v >= 0 else "color: #FF4444"
        except Exception:
            return ""

    st.dataframe(
        df_watch.style.map(color_change, subset=["Change %"]),
        use_container_width=True, height=300,
    )

    st.divider()
    st.subheader("🌎 Macro Dashboard")
    macro_df = get_macro_data()
    if not macro_df.empty:
        cols = st.columns(len(macro_df))
        for i, row in macro_df.iterrows():
            with cols[i]:
                delta_str = f"{'+' if row['Change %'] >= 0 else ''}{row['Change %']}%"
                st.metric(label=row["Name"], value=str(row["Price"]), delta=delta_str)

    st.divider()
    st.subheader("📰 Market Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Market Hours:** 9:30 AM – 4:00 PM ET
        **Pre-Market:** 4:00 AM – 9:30 AM ET
        **After-Hours:** 4:00 PM – 8:00 PM ET
        """)
    with c2:
        st.markdown("""
        **Key Levels to Watch:**
        - SPY VWAP & daily high/low
        - QQQ 200 EMA
        - VIX > 20 = elevated fear
        """)

# ═══════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader(f"📈 {TICKER} — {timeframe} Chart")

    with st.spinner("Loading chart data..."):
        df = get_ohlcv(TICKER, period, interval)

    if df.empty:
        st.error("No chart data available for this ticker / timeframe.")
    else:
        df = add_all_indicators(df)

        # Determine subplot layout
        n_sub = 1 + int(show_vol) + int(show_rsi) + int(show_macd)
        heights = [0.55]
        sub_titles = [f"{TICKER} {timeframe}"]
        if show_vol:
            heights.append(0.15); sub_titles.append("Volume")
        if show_rsi:
            heights.append(0.15); sub_titles.append("RSI (14)")
        if show_macd:
            heights.append(0.15); sub_titles.append("MACD")

        fig = make_subplots(
            rows=n_sub, cols=1, shared_xaxes=True,
            vertical_spacing=0.02, row_heights=heights,
            subplot_titles=sub_titles,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            name=TICKER,
            increasing_line_color=config.CANDLE_UP,
            decreasing_line_color=config.CANDLE_DOWN,
            increasing_fillcolor=config.CANDLE_UP,
            decreasing_fillcolor=config.CANDLE_DOWN,
        ), row=1, col=1)

        # EMAs
        for p in show_ema:
            col_name = f"EMA_{p}"
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name], name=f"EMA {p}",
                    line=dict(color=config.EMA_COLORS.get(p, "#FFFFFF"), width=1.2),
                    opacity=0.9,
                ), row=1, col=1)

        # VWAP
        if show_vwap and "VWAP" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VWAP"], name="VWAP",
                line=dict(color="#FF69B4", width=1.5, dash="dot"),
            ), row=1, col=1)

        # Bollinger Bands
        if show_bb and "BB_Upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Upper"], name="BB Upper",
                line=dict(color="rgba(100,180,255,0.6)", width=1),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Lower"], name="BB Lower",
                line=dict(color="rgba(100,180,255,0.6)", width=1),
                fill="tonexty", fillcolor="rgba(100,180,255,0.06)",
            ), row=1, col=1)

        # Fibonacci
        if show_fib:
            fib = get_fibonacci_levels(df)
            fib_colors = ["#FF4444", "#FF8C00", "#FFD700", "#00CED1", "#1E90FF", "#9370DB", "#00FF88"]
            for (label, level), color in zip(fib.items(), fib_colors):
                fig.add_hline(y=level, line=dict(color=color, width=0.8, dash="dash"),
                              annotation_text=f"Fib {label}", annotation_position="right",
                              row=1, col=1)

        cur_row = 2

        # Volume
        if show_vol:
            colors = [config.VOL_UP if c >= o else config.VOL_DOWN
                      for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name="Volume",
                marker_color=colors, showlegend=False,
            ), row=cur_row, col=1)
            cur_row += 1

        # RSI
        if show_rsi and "RSI" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="#FFD700", width=1.5),
            ), row=cur_row, col=1)
            fig.add_hline(y=70, line=dict(color="#FF4444", dash="dash", width=0.8),
                          row=cur_row, col=1)
            fig.add_hline(y=30, line=dict(color="#00FF88", dash="dash", width=0.8),
                          row=cur_row, col=1)
            fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
            cur_row += 1

        # MACD
        if show_macd and "MACD" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#00BFFF", width=1.2),
            ), row=cur_row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD_Signal"], name="Signal",
                line=dict(color="#FF6B35", width=1.2),
            ), row=cur_row, col=1)
            hist_colors = ["#00FF88" if v >= 0 else "#FF4444" for v in df["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=df.index, y=df["MACD_Hist"], name="Hist",
                marker_color=hist_colors, showlegend=False,
            ), row=cur_row, col=1)

        fig.update_layout(
            height=750, template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            showlegend=True, xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            font=dict(color="#FAFAFA"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1F2937", showspikes=True)
        fig.update_yaxes(showgrid=True, gridcolor="#1F2937")

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics below chart
        if not df.empty:
            last = df.iloc[-1]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Close",  f"${last['Close']:.2f}")
            m2.metric("High",   f"${last['High']:.2f}")
            m3.metric("Low",    f"${last['Low']:.2f}")
            m4.metric("ATR",    f"${df['ATR'].iloc[-1]:.2f}" if "ATR" in df.columns else "—")
            rsi_val = df["RSI"].iloc[-1] if "RSI" in df.columns else None
            rsi_str = f"{rsi_val:.1f}" if rsi_val and not np.isnan(rsi_val) else "—"
            m5.metric("RSI",    rsi_str)

# ═══════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader(f"🔗 {TICKER} Options Chain")

    spot = info["price"]
    chains, expirations = get_options_chain(TICKER)

    if not chains or not expirations:
        st.error("Options data not available for this ticker.")
    else:
        expiry_sel = st.selectbox("Expiration Date", expirations)
        chain_data = chains[expiry_sel]
        calls_raw  = chain_data["calls"]
        puts_raw   = chain_data["puts"]

        calls = enrich_chain_greeks(calls_raw, spot, expiry_sel, "call")
        puts  = enrich_chain_greeks(puts_raw,  spot, expiry_sel, "put")

        max_pain = calculate_max_pain(calls, puts)
        pcr      = calculate_pcr(calls, puts)
        gex      = calculate_gex(calls, puts, spot, expiry_sel)

        # Summary row
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Spot Price",   f"${spot:.2f}")
        s2.metric("Max Pain",     f"${max_pain:.2f}",
                  delta=f"{((max_pain/spot)-1)*100:.1f}% from spot")
        s3.metric("PCR (Volume)", str(pcr["pcr_volume"]),
                  delta="Bearish" if pcr["pcr_volume"] > 1 else "Bullish")
        s4.metric("PCR (OI)",     str(pcr["pcr_oi"]))
        s5.metric("GEX",          f"{gex:+,.0f}",
                  delta="Long Gamma" if gex > 0 else "Short Gamma")
        total_call_prem = (calls["lastPrice"] * calls["volume"] * 100).fillna(0).sum()
        total_put_prem  = (puts["lastPrice"]  * puts["volume"]  * 100).fillna(0).sum()
        s6.metric("Call Premium",  f"${total_call_prem/1e6:.2f}M")

        st.divider()
        disp_cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
                     "IV %", "delta", "gamma", "theta", "vega"]

        tab_c, tab_p = st.tabs(["📗 Calls", "📕 Puts"])
        with tab_c:
            calls_disp = calls[[c for c in disp_cols if c in calls.columns]].copy()
            calls_disp = calls_disp.rename(columns={
                "strike": "Strike", "lastPrice": "Last", "volume": "Volume",
                "openInterest": "OI",
            })

            def highlight_itm_call(row):
                return ["background-color: rgba(0,100,50,0.3)" if row.get("Strike", 0) < spot
                        else "" for _ in row]

            st.dataframe(
                calls_disp.style.apply(highlight_itm_call, axis=1),
                use_container_width=True, height=400,
            )

        with tab_p:
            puts_disp = puts[[c for c in disp_cols if c in puts.columns]].copy()
            puts_disp = puts_disp.rename(columns={
                "strike": "Strike", "lastPrice": "Last", "volume": "Volume",
                "openInterest": "OI",
            })

            def highlight_itm_put(row):
                return ["background-color: rgba(100,0,0,0.3)" if row.get("Strike", 999) > spot
                        else "" for _ in row]

            st.dataframe(
                puts_disp.style.apply(highlight_itm_put, axis=1),
                use_container_width=True, height=400,
            )

        # OI chart
        st.divider()
        st.subheader("Open Interest by Strike")
        try:
            strikes  = sorted(set(list(calls["strike"])) | set(list(puts["strike"])))
            call_oi  = calls.set_index("strike")["openInterest"].reindex(strikes).fillna(0)
            put_oi   = puts.set_index("strike")["openInterest"].reindex(strikes).fillna(0)

            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(x=strikes, y=call_oi, name="Call OI",
                                    marker_color="rgba(0,255,136,0.7)"))
            fig_oi.add_trace(go.Bar(x=strikes, y=-put_oi, name="Put OI",
                                    marker_color="rgba(255,68,68,0.7)"))
            fig_oi.add_vline(x=spot,     line=dict(color="white",  dash="dash"), annotation_text="Spot")
            fig_oi.add_vline(x=max_pain, line=dict(color="#FFD700", dash="dot"),  annotation_text="Max Pain")
            fig_oi.update_layout(
                height=350, template="plotly_dark",
                barmode="overlay", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                title="Call (green) vs Put (red) Open Interest",
            )
            st.plotly_chart(fig_oi, use_container_width=True)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════
# TAB 4 — SMART MONEY FLOW
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader(f"💥 Smart Money / Unusual Options Activity — {TICKER}")
    st.caption("Flags options with abnormally high volume-to-OI ratio and large premium.")

    chains2, expirations2 = get_options_chain(TICKER)
    if not chains2:
        st.error("No options data available.")
    else:
        all_unusual = []
        for exp, cd in chains2.items():
            uu = detect_unusual_activity(cd["calls"], cd["puts"], TICKER, spot)
            if not uu.empty:
                uu["Expiry"] = exp
                all_unusual.append(uu)

        if all_unusual:
            df_uu = pd.concat(all_unusual, ignore_index=True).sort_values("Score", ascending=False)

            # Summary badges
            bull_count = len(df_uu[df_uu["Sentiment"] == "Bullish"])
            bear_count = len(df_uu[df_uu["Sentiment"] == "Bearish"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Unusual Activity Detected", len(df_uu))
            c2.metric("Bullish Signals", bull_count)
            c3.metric("Bearish Signals", bear_count)

            if bull_count > bear_count:
                st.markdown('<span class="bull-badge">SMART MONEY BULLISH BIAS</span>', unsafe_allow_html=True)
            elif bear_count > bull_count:
                st.markdown('<span class="bear-badge">SMART MONEY BEARISH BIAS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="neutral-badge">MIXED SIGNALS</span>', unsafe_allow_html=True)

            st.divider()

            def color_sentiment(row):
                if row["Sentiment"] == "Bullish":
                    return ["color: #00FF88"] * len(row)
                elif row["Sentiment"] == "Bearish":
                    return ["color: #FF4444"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_uu.style.apply(color_sentiment, axis=1),
                use_container_width=True, height=450,
            )

            # Vol vs OI scatter
            fig_sc = px.scatter(
                df_uu, x="OI", y="Vol", color="Sentiment", size="Score",
                hover_data=["Strike", "Type", "Premium", "Expiry"],
                color_discrete_map={"Bullish": "#00FF88", "Bearish": "#FF4444"},
                template="plotly_dark", title="Volume vs Open Interest (Unusual Activity)",
            )
            fig_sc.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=350)
            st.plotly_chart(fig_sc, use_container_width=True)

        else:
            st.info("No unusual options activity detected for this ticker across all expirations.")

        st.divider()
        st.subheader("📊 Flow Summary by Expiration")
        flow_rows = []
        for exp, cd in chains2.items():
            pcr_e = calculate_pcr(cd["calls"], cd["puts"])
            flow_rows.append({
                "Expiry":      exp,
                "Call Volume": f"{pcr_e['call_volume']:,}",
                "Put Volume":  f"{pcr_e['put_volume']:,}",
                "Call OI":     f"{pcr_e['call_oi']:,}",
                "Put OI":      f"{pcr_e['put_oi']:,}",
                "PCR Vol":     pcr_e["pcr_volume"],
                "PCR OI":      pcr_e["pcr_oi"],
                "Bias":        "Bearish" if pcr_e["pcr_volume"] > 1 else "Bullish",
            })
        st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — AI SIGNALS
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader(f"🤖 AI Prediction Engine — {TICKER}")

    with st.spinner("Training XGBoost model on historical data..."):
        df_ai = get_ohlcv(TICKER, "2y", "1d")
        if not df_ai.empty:
            df_ai = add_all_indicators(df_ai)
            pred  = get_ai_prediction(TICKER, df_ai)
        else:
            pred = {"bullish_prob": 50, "bearish_prob": 50, "confidence": 0,
                    "accuracy": 0, "expected_move_pct": 0, "signal": "N/A",
                    "trend_strength": 50}

    sig_color = ("#00FF88" if "BUY" in pred["signal"] else
                 "#FF4444" if "SELL" in pred["signal"] else "#FFD700")

    st.markdown(
        f"<h2 style='color:{sig_color};text-align:center'>{pred['signal']}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='text-align:center;color:#AAB4C8'>Model Accuracy: {pred['accuracy']}%</p>",
                unsafe_allow_html=True)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bullish Probability", f"{pred['bullish_prob']}%")
    col2.metric("Bearish Probability", f"{pred['bearish_prob']}%")
    col3.metric("Confidence Score",    f"{pred['confidence']}%")
    col4.metric("Expected Move",       f"{pred['expected_move_pct']:+.2f}%")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred["bullish_prob"],
            title={"text": "Bullish Probability %", "font": {"color": "#FAFAFA"}},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                "bar":  {"color": "#00FF88" if pred["bullish_prob"] > 50 else "#FF4444"},
                "steps": [
                    {"range": [0, 40],   "color": "#3D0000"},
                    {"range": [40, 60],  "color": "#2A2A1A"},
                    {"range": [60, 100], "color": "#003D1A"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor="#0E1117",
            font={"color": "#FAFAFA"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        # Trend strength
        fig_trend = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred["trend_strength"],
            title={"text": "Trend Strength (EMA Alignment %)", "font": {"color": "#FAFAFA"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                "bar":  {"color": "#00BFFF"},
                "steps": [
                    {"range": [0, 33],  "color": "#1A1F2E"},
                    {"range": [33, 66], "color": "#1A2030"},
                    {"range": [66, 100],"color": "#1A2840"},
                ],
            },
        ))
        fig_trend.update_layout(
            height=280, paper_bgcolor="#0E1117",
            font={"color": "#FAFAFA"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Feature importance
    if "top_features" in pred and pred["top_features"]:
        st.divider()
        st.subheader("Top Predictive Features")
        feat_df = pd.DataFrame(pred["top_features"], columns=["Feature", "Importance %"])
        fig_fi = px.bar(feat_df, x="Importance %", y="Feature", orientation="h",
                        color="Importance %", color_continuous_scale="Viridis",
                        template="plotly_dark")
        fig_fi.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=250)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Historical probability chart
    st.divider()
    st.subheader("Price Action (2Y Daily)")
    if not df_ai.empty and "EMA_20" in df_ai.columns:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df_ai.index, y=df_ai["Close"],
                                      name="Close", line=dict(color="#00FF88", width=1.5)))
        fig_hist.add_trace(go.Scatter(x=df_ai.index, y=df_ai["EMA_20"],
                                      name="EMA 20", line=dict(color="#FFD700", width=1)))
        fig_hist.add_trace(go.Scatter(x=df_ai.index, y=df_ai["EMA_50"],
                                      name="EMA 50", line=dict(color="#00BFFF", width=1)))
        fig_hist.update_layout(height=300, template="plotly_dark",
                               paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 6 — SCALPING DASHBOARD
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader(f"⚡ Scalping Dashboard — {TICKER}")

    scal_tf = st.selectbox("Scalp Timeframe", ["1 Min", "5 Min", "15 Min"], index=1,
                            key="scal_tf")
    s_period, s_interval = config.TIMEFRAME_CONFIG[scal_tf]

    with st.spinner("Loading scalp data..."):
        df_scal = get_ohlcv(TICKER, s_period, s_interval)

    if not df_scal.empty:
        df_scal = add_all_indicators(df_scal)
        signals = get_scalping_signals(df_scal)
        levels  = get_scalp_levels(df_scal)

        # Signal cards
        st.subheader("Active Signals")
        if signals:
            sig_cols = st.columns(min(len(signals), 4))
            for i, sig in enumerate(signals):
                badge = "bull-badge" if sig["type"] == "BULLISH" else "bear-badge"
                with sig_cols[i % 4]:
                    st.markdown(f"""
                    <div class="signal-card">
                      <span class="{badge}">{sig['type']}</span><br>
                      <b>{sig['signal']}</b><br>
                      <span style="color:#AAB4C8">@ ${sig['price']:.2f} · {sig['strength']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No active scalp signals detected. Market may be in consolidation.")

        st.divider()
        # Entry box
        st.subheader("Trade Levels")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**LONG Setup**")
            ll1, ll2, ll3, ll4 = st.columns(4)
            ll1.metric("Entry",   f"${levels.get('entry', 0):.2f}")
            ll2.metric("Stop",    f"${levels.get('stop_long', 0):.2f}", delta=f"-${abs(levels.get('entry', 0) - levels.get('stop_long', 0)):.2f}")
            ll3.metric("TP1",     f"${levels.get('tp1_long', 0):.2f}")
            ll4.metric("TP2",     f"${levels.get('tp2_long', 0):.2f}")
        with c2:
            st.markdown("**SHORT Setup**")
            sl1, sl2, sl3, sl4 = st.columns(4)
            sl1.metric("Entry",   f"${levels.get('entry', 0):.2f}")
            sl2.metric("Stop",    f"${levels.get('stop_short', 0):.2f}")
            sl3.metric("TP1",     f"${levels.get('tp1_short', 0):.2f}")
            sl4.metric("TP2",     f"${levels.get('tp2_short', 0):.2f}")

        st.caption(f"ATR ({scal_tf}): ${levels.get('atr', 0):.2f} — Levels based on 1.5× / 2× / 3.5× ATR")

        st.divider()
        # Mini chart
        st.subheader(f"Live Chart — {scal_tf}")
        df_plot = df_scal.tail(150)
        fig_scal = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.7, 0.3], vertical_spacing=0.02)
        fig_scal.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"], close=df_plot["Close"], name=TICKER,
            increasing_line_color=config.CANDLE_UP,
            decreasing_line_color=config.CANDLE_DOWN,
        ), row=1, col=1)
        if "VWAP" in df_plot.columns:
            fig_scal.add_trace(go.Scatter(x=df_plot.index, y=df_plot["VWAP"],
                                           name="VWAP", line=dict(color="#FF69B4", dash="dot", width=1.5)),
                                row=1, col=1)
        for p in [9, 20]:
            col = f"EMA_{p}"
            if col in df_plot.columns:
                fig_scal.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=f"EMA {p}",
                                               line=dict(color=config.EMA_COLORS[p], width=1)),
                                    row=1, col=1)
        v_colors = [config.VOL_UP if c >= o else config.VOL_DOWN
                    for c, o in zip(df_plot["Close"], df_plot["Open"])]
        fig_scal.add_trace(go.Bar(x=df_plot.index, y=df_plot["Volume"],
                                   marker_color=v_colors, showlegend=False),
                            row=2, col=1)
        fig_scal.update_layout(height=500, template="plotly_dark",
                                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_scal, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 7 — HEATMAP
# ═══════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("🏦 Institutional Heatmap — Sector Rotation")

    with st.spinner("Loading sector data..."):
        sec_df = get_sector_data()

    if not sec_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_tree = px.treemap(
                sec_df, path=["Sector"], values=[1] * len(sec_df),
                color="1D %", color_continuous_scale=["#FF4444", "#333333", "#00FF88"],
                color_continuous_midpoint=0,
                hover_data={"ETF": True, "Price": True, "1D %": True, "5D %": True},
                title="Sector Performance (color = 1D %)",
            )
            fig_tree.update_traces(textinfo="label+text",
                                   text=[f"{r['ETF']}<br>{r['1D %']:+.2f}%" for _, r in sec_df.iterrows()])
            fig_tree.update_layout(height=450, paper_bgcolor="#0E1117",
                                   font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_tree, use_container_width=True)

        with c2:
            st.subheader("Sector Rankings")
            sec_sorted = sec_df.sort_values("1D %", ascending=False).reset_index(drop=True)
            for _, row in sec_sorted.iterrows():
                color = "#00FF88" if row["1D %"] >= 0 else "#FF4444"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A3040'>"
                    f"<span>{row['Sector']} ({row['ETF']})</span>"
                    f"<span style='color:{color};font-weight:bold'>{row['1D %']:+.2f}%</span></div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("5-Day Sector Performance")
        fig_bar = px.bar(
            sec_df.sort_values("5D %"), x="5D %", y="Sector", orientation="h",
            color="5D %", color_continuous_scale=["#FF4444", "#333333", "#00FF88"],
            color_continuous_midpoint=0,
            template="plotly_dark",
        )
        fig_bar.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("Watchlist Relative Strength")
    rs_rows = []
    for tk in st.session_state.watchlist:
        d = get_stock_info(tk)
        rs_rows.append({"Ticker": tk, "Change %": d["change_pct"], "Price": d["price"]})
    rs_df = pd.DataFrame(rs_rows).sort_values("Change %", ascending=False)
    fig_rs = px.bar(rs_df, x="Ticker", y="Change %", color="Change %",
                    color_continuous_scale=["#FF4444", "#333333", "#00FF88"],
                    color_continuous_midpoint=0, template="plotly_dark",
                    title="Watchlist Daily Performance")
    fig_rs.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=300)
    st.plotly_chart(fig_rs, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 8 — RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════
with tabs[8]:
    render_risk_panel()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4A5568;font-size:0.75rem'>"
    "Options Trading Dashboard · Data via Yahoo Finance · For educational use only · "
    "Not financial advice</p>",
    unsafe_allow_html=True,
)

