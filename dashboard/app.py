"""
Streamlit Dashboard — Informer Carbon Flux Forecasting
Interactive visualization of model predictions, metrics, and attention patterns
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import CarbonFluxDataset
from model.informer import build_model

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Carbon Flux Forecasting | Informer",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a2a1a, #0d1f0d);
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stMetric label { color: #2ecc71!important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cache: Load Model
# ─────────────────────────────────────────────

@st.cache_resource
def load_everything():
    ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
    cfg  = ckpt["config"]

    with open("data/processed/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    model = build_model(
        n_features=meta["n_features"],
        seq_len=cfg["seq_len"],
        label_len=cfg["label_len"],
        pred_len=cfg["pred_len"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = CarbonFluxDataset("test")
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_pred, all_true, all_enc_attns = [], [], []
    with torch.no_grad():
        for x_enc, x_dec, y in test_dl:
            pred, enc_attns, _ = model(x_enc, x_dec)
            all_pred.append(pred.numpy())
            all_true.append(y.numpy())
            all_enc_attns.append(enc_attns[0].numpy())

    pred_arr  = np.concatenate(all_pred, axis=0)
    true_arr  = np.concatenate(all_true, axis=0)
    attn_arr  = np.concatenate(all_enc_attns, axis=0)

    raw_df = pd.read_csv("data/raw/carbon_flux.csv", parse_dates=["date"])

    with open("checkpoints/history.json") as f:
        history = json.load(f)

    metrics = {}
    if os.path.exists("plots/test_metrics.json"):
        with open("plots/test_metrics.json") as f:
            metrics = json.load(f)

    return model, pred_arr, true_arr, attn_arr, raw_df, history, metrics, cfg, meta


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='text-align:center; color:#2ecc71;'>🌿 Carbon Flux Forecasting</h1>
<p style='text-align:center; color:#aaa; font-size:16px;'>
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting<br>
<b>Zhou et al., AAAI 2021 Best Paper</b> — Applied to Sundarbans & Chilika Carbon Sequestration
</p>
<hr style='border-color:#2ecc71;'>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading model and predictions..."):
    model, pred_arr, true_arr, attn_arr, raw_df, history, metrics, cfg, meta = load_everything()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Sundarban_Tiger.jpg/320px-Sundarban_Tiger.jpg", use_column_width=True)
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Seq Length | {cfg['seq_len']} days |
| Label Length | {cfg['label_len']} days |
| Forecast Horizon | {cfg['pred_len']} days |
| Features | {meta['n_features']} |
| Model Params | {model.count_parameters():,} |
| Data Source | {meta.get('data_source', 'synthetic').upper()} |
""")

if meta.get('data_source') == 'real_gee':
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛰️ GEE Data Sources")
    st.sidebar.markdown("""
    | Variable | Dataset |
    |----------|--------|
    | NDVI | MODIS MOD13A2 |
    | Temp | ERA5 Land |
    | Soil Moisture | SMAP SPL4SMGP |
    | Radiation | ERA5 Land |
    | VPD | ERA5 Land |
    | NEE | MODIS MOD17A2H |
    
    **Sites:** Sundarbans + Chilika Lake (averaged)  
    **Project:** `ee-deepeshy`  
    **Period:** 2018–2024
    """)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔮 Forecast Explorer", "📈 Training History", "🧠 Attention Analysis", "🌍 Raw Data Explorer"]
)

# ─────────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.markdown("## 📊 Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE",  f"{metrics.get('MAE', 'N/A')}")
    col2.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
    col3.metric("MAPE", f"{metrics.get('MAPE', 'N/A')}%")
    col4.metric("Parameters", f"{model.count_parameters():,}")

    st.markdown("---")
    st.markdown("### 🌿 About This Project")

    data_source = meta.get('data_source', 'synthetic')
    if data_source == 'real_gee':
        st.success("🛰️ **Data Source: Real GEE Satellite Data** — MODIS, SMAP, ERA5 via Google Earth Engine (project: ee-deepeshy)")
    else:
        st.info("🔬 **Data Source: Synthetic FLUXNET-style data** — run with `--data real` to use real GEE satellite data")

    st.markdown("""
    This dashboard presents the results of an **Informer Transformer** trained from scratch 
    to forecast **Net Ecosystem Exchange (NEE)** — the net carbon flux between ecosystems and the atmosphere.

    **Key innovations from the paper implemented from scratch:**
    - **ProbSparse Self-Attention** — O(L log L) vs O(L²) for standard Transformers
    - **ConvPool Distilling** — Halves sequence length between encoder layers
    - **Generative Decoder** — One-shot forecast instead of slow autoregressive decoding

    **Ecosystems:** Sundarbans Mangrove Forest + Chilika Lake (Odisha, India) — averaged (Option A)  
    **Forecast horizon:** {pred} days ahead from {seq} days of historical context
    """.format(pred=cfg['pred_len'], seq=cfg['seq_len']))

    # Actual vs Predicted scatter
    flat_pred = pred_arr[:, :, 0].flatten()
    flat_true = true_arr[:, :, 0].flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flat_true, y=flat_pred, mode="markers",
        marker=dict(color="#2ecc71", size=3, opacity=0.4),
        name="Predictions"
    ))
    lim = [min(flat_true.min(), flat_pred.min()), max(flat_true.max(), flat_pred.max())]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="white", dash="dash", width=1.5),
        name="Perfect Prediction"
    ))
    fig.update_layout(
        title="Actual vs Predicted NEE (All Test Samples)",
        xaxis_title="Actual NEE (scaled)",
        yaxis_title="Predicted NEE (scaled)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2: Forecast Explorer
# ─────────────────────────────────────────────

elif page == "🔮 Forecast Explorer":
    st.markdown("## 🔮 Forecast Explorer")
    st.markdown("Select a test sample to see the Informer's carbon flux forecast vs actual values.")

    n_test = len(pred_arr)
    idx = st.slider("Test Sample Index", 0, n_test - 1, 0)

    p = pred_arr[idx, :, 0]
    t = true_arr[idx, :, 0]
    days = list(range(len(p)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=t.tolist(), name="Actual NEE",
        line=dict(color="#2c3e50", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=days, y=p.tolist(), name="Predicted NEE",
        line=dict(color="#2ecc71", width=2.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=p.tolist() + t.tolist()[::-1],
        fill="toself", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Error Region"
    ))
    fig.update_layout(
        title=f"Carbon Flux Forecast — Test Sample {idx}",
        xaxis_title="Forecast Day",
        yaxis_title="NEE (Net Ecosystem Exchange, scaled)",
        template="plotly_dark",
        height=450,
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    sample_mae  = float(np.mean(np.abs(p - t)))
    sample_rmse = float(np.sqrt(np.mean((p - t)**2)))
    col1.metric("Sample MAE",  f"{sample_mae:.4f}")
    col2.metric("Sample RMSE", f"{sample_rmse:.4f}")
    col3.metric("Max Error",   f"{float(np.max(np.abs(p-t))):.4f}")

    # Error bar chart
    errors = p - t
    fig2 = go.Figure(go.Bar(
        x=days, y=errors.tolist(),
        marker_color=["#e74c3c" if e > 0 else "#2ecc71" for e in errors],
        name="Daily Error"
    ))
    fig2.update_layout(
        title="Daily Prediction Error (Positive = Overestimate)",
        xaxis_title="Forecast Day",
        yaxis_title="Error",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3: Training History
# ─────────────────────────────────────────────

elif page == "📈 Training History":
    st.markdown("## 📈 Training History")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss Curves", "Validation Metrics"])

    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                             line=dict(color="#e74c3c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                             line=dict(color="#3498db", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_mae"], name="MAE",
                             line=dict(color="#2ecc71", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_rmse"], name="RMSE",
                             line=dict(color="#9b59b6", width=2)), row=1, col=2)

    fig.update_layout(template="plotly_dark", height=420,
                      title="Informer Training History — Carbon Flux Model")
    st.plotly_chart(fig, use_container_width=True)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    st.info(f"Best model found at epoch **{best_epoch}** with val loss **{min(history['val_loss']):.4f}**")


# ─────────────────────────────────────────────
# PAGE 4: Attention Analysis
# ─────────────────────────────────────────────

elif page == "🧠 Attention Analysis":
    st.markdown("## 🧠 ProbSparse Attention Analysis")
    st.markdown("""
    The core innovation of Informer is **ProbSparse Self-Attention**.
    Instead of computing attention for all L² query-key pairs, it selects only the 
    top-u *dominant* queries (those with the most peaked, non-uniform attention distributions)
    and computes full attention only for those. This reduces complexity from O(L²) to O(L log L).
    
    The heatmap below shows which encoder timesteps (days in the lookback window) 
    receive the most attention from the selected dominant queries.
    """)

    n_samples_for_attn = min(100, len(attn_arr))
    avg_attn = attn_arr[:n_samples_for_attn].mean(axis=0)

    fig = go.Figure(go.Heatmap(
        z=avg_attn.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title="Attention Weight")
    ))
    fig.update_layout(
        title="ProbSparse Attention Heatmap (Averaged over Test Set)",
        xaxis_title="Encoder Timestep (Days in Lookback Window)",
        yaxis_title="Selected Dominant Query",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Insight")
    st.markdown("""
    - **Bright regions** = encoder timesteps that the model pays most attention to
    - The model learns to focus on **seasonally important days** in the lookback window
    - This interpretability is crucial for carbon monitoring — we can verify the model
      is attending to ecologically meaningful patterns (monsoon onset, post-monsoon peaks)
    """)


# ─────────────────────────────────────────────
# PAGE 5: Raw Data Explorer
# ─────────────────────────────────────────────

elif page == "🌍 Raw Data Explorer":
    st.markdown("## 🌍 Raw Carbon Flux Data Explorer")

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
    with col2:
        end = st.date_input("End Date", value=pd.Timestamp("2024-12-31"))

    mask = (raw_df["date"] >= pd.Timestamp(start)) & (raw_df["date"] <= pd.Timestamp(end))
    df_view = raw_df[mask]

    variable = st.selectbox(
        "Select Variable",
        ["nee", "temperature", "soil_moisture", "ndvi", "radiation", "vpd"]
    )

    labels = {
        "nee": "Net Ecosystem Exchange (tCO₂/ha/day)",
        "temperature": "Temperature (°C)",
        "soil_moisture": "Soil Moisture (fraction)",
        "ndvi": "NDVI",
        "radiation": "Solar Radiation (W/m²)",
        "vpd": "Vapour Pressure Deficit (kPa)"
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_view["date"].tolist(),
        y=df_view[variable].tolist(),
        mode="lines",
        line=dict(color="#2ecc71", width=1.5),
        name=variable.upper()
    ))

    if variable == "nee":
        fig.add_hline(y=0, line_dash="dash", line_color="white",
                      annotation_text="Carbon Neutral", annotation_position="top right")
        # Mark COVID period
        fig.add_vrect(x0="2021-01-01", x1="2021-12-31",
                      fillcolor="rgba(231,76,60,0.15)", line_width=0,
                      annotation_text="COVID-19 Impact", annotation_position="top left")

    fig.update_layout(
        title=f"{labels[variable]} — Sundarbans/Chilika Ecosystem (2018–2024)",
        xaxis_title="Date",
        yaxis_title=labels[variable],
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(df_view[variable].describe().round(4).to_frame().T, use_container_width=True)

    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df_view.tail(100), use_container_width=True)
