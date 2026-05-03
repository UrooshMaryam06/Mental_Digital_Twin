"""
Forecasting page: build a feature vector from sliders, call /predict/both,
display 12-step ahead forecast from the historical dataset.
"""
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
st.set_page_config(page_title="Forecasting", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.sidebar import render_sidebar
from services.api_client import predict_both
from services.data_loader import load_dataset
from utils.config import COLORS

render_sidebar()
st.markdown("## Demand & Price Forecasting")
st.divider()

df = load_dataset()

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("### Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Target Timestamp")
    date_val = st.date_input("Date", datetime.date(2018, 5, 1))
    time_val = st.time_input("Time", datetime.time(12, 0))
    timestamp = f"{date_val} {time_val}"
with col2:
    st.markdown("##### Renewable Gen (MW)")
    solar = st.number_input("Solar (MW)", 0.0, 20000.0, 5000.0, step=100.0)
    wind  = st.number_input("Wind Onshore (MW)", 0.0, 30000.0, 8000.0, step=100.0)
    hydro = st.number_input("Hydro Water Res. (MW)", 0.0, 20000.0, 2000.0, step=100.0)
with col3:
    st.markdown("##### Non-Renewable (MW)")
    gas     = st.number_input("Fossil Gas (MW)", 0.0, 20000.0, 4000.0, step=100.0)
    coal    = st.number_input("Fossil Hard Coal (MW)", 0.0, 20000.0, 1500.0, step=100.0)
    nuclear = st.number_input("Nuclear (MW)", 0.0, 20000.0, 7000.0, step=100.0)

st.markdown("##### External Forecasts")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    fc_wind = st.number_input("Forecast Wind Onshore (MW)", 0.0, 30000.0, 8000.0, step=100.0)
with fc2:
    fc_solar = st.number_input("Forecast Solar (MW)", 0.0, 20000.0, 5000.0, step=100.0)
with fc3:
    load_fc = st.number_input("Total Load Forecast (MW)", 0.0, 50000.0, 28000.0, step=100.0)
with fc4:
    fc_price = st.number_input("Price Day Ahead (EUR/MWh)", 0.0, 1000.0, 50.0, step=1.0)

features = {
    "timestamp": timestamp,
    "generation solar": solar,
    "generation wind onshore": wind,
    "generation nuclear": nuclear,
    "generation fossil gas": gas,
    "generation fossil hard coal": coal,
    "generation hydro water reservoir": hydro,
    "forecast wind onshore day ahead": fc_wind,
    "forecast solar day ahead": fc_solar,
    "total load forecast": load_fc,
    "price day ahead": fc_price
}

if st.button("Run Prediction"):
    result = predict_both(features)
    if result:
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Predicted Demand", f"{result.get('predicted_demand_12h_MW', 0):,.0f} MW")
        c2.metric("Predicted Price",  f"€{result.get('predicted_price_12h_EUR', 0):.2f} / MWh")

# ── Historical actual vs forecast chart ──────────────────────────────────────
if not df.empty:
    st.divider()
    st.markdown("### Historical Series")
    start, end = st.session_state.get("date_range", (df.index.min().date(), df.index.max().date()))
    mask = (df.index.date >= start) & (df.index.date <= end)
    plot_df = df[mask].copy()

    if len(plot_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['total load actual'],
            name="Demand (MW)",
            line=dict(color=COLORS["demand_color"], width=1),
        ))
        fig2_ax = fig  # share the figure, use secondary y
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['price actual'],
            name="Price (EUR/MWh)",
            line=dict(color=COLORS["price_color"], width=1),
            yaxis="y2",
        ))
        fig.update_layout(
            paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_card"],
            font=dict(color=COLORS["text_primary"]),
            yaxis=dict(title="Demand (MW)",    gridcolor=COLORS["border"]),
            yaxis2=dict(title="Price (EUR/MWh)", overlaying="y", side="right",
                        gridcolor=COLORS["border"]),
            legend=dict(bgcolor=COLORS["bg_card"]),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
