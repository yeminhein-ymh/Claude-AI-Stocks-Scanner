import streamlit as st
import numpy as np


def calculate_position_size(account_size: float, risk_pct: float,
                             entry: float, stop_loss: float) -> dict:
    risk_amount  = account_size * (risk_pct / 100)
    risk_per_sh  = abs(entry - stop_loss)
    if risk_per_sh == 0:
        return {"shares": 0, "contracts": 0, "risk_amount": 0, "position_value": 0}
    shares       = int(risk_amount / risk_per_sh)
    contracts    = max(1, int(shares / 100))
    return {
        "shares":         shares,
        "contracts":      contracts,
        "risk_amount":    round(risk_amount, 2),
        "position_value": round(shares * entry, 2),
    }


def calculate_rrr(entry: float, stop: float, target: float) -> dict:
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rrr    = reward / risk if risk > 0 else 0
    win_rate_needed = 1 / (1 + rrr) * 100 if rrr > 0 else 100
    return {
        "risk":              round(risk, 2),
        "reward":            round(reward, 2),
        "rrr":               round(rrr, 2),
        "win_rate_needed":   round(win_rate_needed, 1),
        "expected_value":    round(rrr - (1 - rrr / (1 + rrr)), 2) if rrr > 0 else -1,
    }


def probability_of_profit(entry: float, target: float, stop: float,
                           iv: float, days: int) -> float:
    T = days / 365
    if T <= 0 or iv <= 0:
        return 50.0
    from scipy.stats import norm
    move_pct = (target - entry) / entry
    z = move_pct / (iv * np.sqrt(T))
    return round((1 - norm.cdf(z)) * 100, 1)


def render_risk_panel():
    st.subheader("Position Sizing Calculator")
    c1, c2 = st.columns(2)
    with c1:
        account  = st.number_input("Account Size ($)", value=25_000, step=1000, min_value=1000)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.25)
    with c2:
        entry    = st.number_input("Entry Price", value=100.0, step=0.5, min_value=0.01)
        stop     = st.number_input("Stop Loss", value=97.0, step=0.5, min_value=0.01)
        target   = st.number_input("Take Profit", value=106.0, step=0.5, min_value=0.01)

    pos = calculate_position_size(account, risk_pct, entry, stop)
    rrr = calculate_rrr(entry, stop, target)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shares",         pos["shares"])
    col2.metric("Option Contracts", pos["contracts"])
    col3.metric("Max Risk $",     f"${pos['risk_amount']:,.0f}")
    col4.metric("Position Value",  f"${pos['position_value']:,.0f}")

    st.divider()
    st.subheader("Risk / Reward Analysis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk",       f"${rrr['risk']}")
    col2.metric("Reward",     f"${rrr['reward']}")
    col3.metric("R:R Ratio",  f"1 : {rrr['rrr']}")
    col4.metric("Min Win Rate", f"{rrr['win_rate_needed']}%")

    if rrr["rrr"] >= 2:
        st.success(f"Good setup — R:R of 1:{rrr['rrr']} exceeds 2:1 minimum.")
    elif rrr["rrr"] >= 1:
        st.warning(f"Marginal — R:R is 1:{rrr['rrr']}. Consider a tighter stop or wider target.")
    else:
        st.error(f"Poor setup — R:R of 1:{rrr['rrr']} is below 1:1. Avoid this trade.")

    st.divider()
    st.subheader("Greeks Exposure (Portfolio)")
    st.info("Enter your open options positions to calculate aggregate Greeks exposure.")
    num_pos = st.number_input("Number of positions", 1, 10, 1)
    total_delta = total_gamma = total_theta = total_vega = 0.0
    for i in range(num_pos):
        with st.expander(f"Position {i+1}"):
            pc, pc2, pc3, pc4, pc5 = st.columns(5)
            qty    = pc.number_input("Contracts", 1, 100, 1, key=f"qty_{i}")
            delta  = pc2.number_input("Delta", -1.0, 1.0, 0.5, 0.01, key=f"d_{i}")
            gamma  = pc3.number_input("Gamma", 0.0, 0.5, 0.05, 0.001, key=f"g_{i}")
            theta  = pc4.number_input("Theta", -10.0, 0.0, -0.1, 0.01, key=f"t_{i}")
            vega   = pc5.number_input("Vega", 0.0, 1.0, 0.1, 0.01, key=f"v_{i}")
            total_delta += delta * qty * 100
            total_gamma += gamma * qty * 100
            total_theta += theta * qty
            total_vega  += vega  * qty

    st.divider()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Net Delta",  f"{total_delta:,.0f}")
    g2.metric("Net Gamma",  f"{total_gamma:,.2f}")
    g3.metric("Net Theta",  f"${total_theta:,.0f}/day")
    g4.metric("Net Vega",   f"{total_vega:,.2f}")
