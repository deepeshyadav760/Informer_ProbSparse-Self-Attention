"""
Data Pipeline for Carbon Flux Forecasting
==========================================
Supports two data modes:
  - synthetic : FLUXNET-style simulated data (no GEE required)
  - real      : Real satellite data from Google Earth Engine
                (MODIS NDVI, MODIS LST, SMAP Soil Moisture,
                 ERA5 Radiation/VPD, MODIS GPP as NEE proxy)

Variables: NEE (Net Ecosystem Exchange), Temperature, Soil Moisture, NDVI, Radiation, VPD
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

# ─────────────────────────────────────────────
# 1. Generate Synthetic FLUXNET-style Dataset
# ─────────────────────────────────────────────

def generate_carbon_flux_data(n_years=7, seed=42):
    """
    Generate synthetic daily carbon flux data matching our
    Sundarbans/Chilika research findings (2018-2024).
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq="D")
    n = len(dates)
    t = np.arange(n)

    # --- Environmental Drivers ---

    # Temperature: seasonal cycle + warming trend + noise
    temp = (
        22 + 8 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
        + 0.002 * t
        + np.random.normal(0, 1.5, n)
    )

    # Soil Moisture: inverse seasonal + monsoon peak
    monsoon_peak = np.where(
        (dates.month >= 6) & (dates.month <= 9), 1.0, 0.0
    )
    soil_moisture = (
        0.35
        - 0.1 * np.sin(2 * np.pi * t / 365)
        + 0.2 * monsoon_peak
        + np.random.normal(0, 0.02, n)
    ).clip(0.05, 0.95)

    # NDVI: vegetation greenness — peak post-monsoon (Oct)
    ndvi = (
        0.55
        + 0.2 * np.sin(2 * np.pi * (t - 60) / 365)
        + np.random.normal(0, 0.03, n)
    ).clip(0.1, 0.95)

    # Solar Radiation: seasonal
    radiation = (
        200 + 100 * np.sin(2 * np.pi * t / 365)
        + np.random.normal(0, 15, n)
    ).clip(50, 400)

    # VPD (Vapour Pressure Deficit): high in summer
    vpd = (
        1.5 + 1.0 * np.sin(2 * np.pi * t / 365 - np.pi / 4)
        + np.random.normal(0, 0.2, n)
    ).clip(0.1, 4.0)

    # --- Target: NEE (Net Ecosystem Exchange) in tonnes CO2/ha/day ---
    # Negative = carbon uptake (sequestration), Positive = carbon release
    # Based on our research: Sundarbans ~545,293 tonnes/year → ~5.69 tCO2/ha/day for Chilika

    nee = (
        -3.5                                          # base uptake
        - 1.5 * ndvi                                  # more vegetation = more uptake
        - 0.5 * soil_moisture                         # moisture helps uptake
        + 0.08 * (temp - 20)                          # respiration increases with temp
        - 0.003 * radiation                           # photosynthesis
        + 0.3 * vpd                                   # stress reduces uptake
        + np.random.normal(0, 0.3, n)                 # measurement noise
    )

    # COVID-19 impact 2021: reduced atmospheric CO2 → reduced uptake
    covid_mask = (dates.year == 2021)
    nee[covid_mask] += 1.8   # reduced uptake (less negative NEE)

    df = pd.DataFrame({
        "date": dates,
        "temperature": temp,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "radiation": radiation,
        "vpd": vpd,
        "nee": nee
    })

    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df):
    """Add time-based and rolling features."""
    df = df.copy()
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Cyclical encoding of day of year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Rolling features
    for col in ["temperature", "soil_moisture", "ndvi"]:
        df[f"{col}_7d"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_30d"] = df[col].rolling(30, min_periods=1).mean()

    # Lagged NEE
    df["nee_lag1"] = df["nee"].shift(1).bfill()
    df["nee_lag7"] = df["nee"].shift(7).bfill()

    return df


# ─────────────────────────────────────────────
# 3. Sequence Creation for Informer
# ─────────────────────────────────────────────

def create_sequences(data, seq_len=96, label_len=48, pred_len=24):
    """
    Create encoder-decoder sequences for Informer.
    
    seq_len   : encoder input length (lookback window)
    label_len : known part of decoder input (overlap)
    pred_len  : forecast horizon
    """
    X_enc, X_dec, Y = [], [], []
    total = len(data)

    for i in range(total - seq_len - pred_len + 1):
        enc_start = i
        enc_end   = i + seq_len
        dec_start = enc_end - label_len
        dec_end   = enc_end + pred_len

        X_enc.append(data[enc_start:enc_end])
        X_dec.append(data[dec_start:dec_end])
        Y.append(data[enc_end:dec_end, -1])  # NEE is last column

    return np.array(X_enc), np.array(X_dec), np.array(Y)


# ─────────────────────────────────────────────
# 4. Train / Val / Test Split
# ─────────────────────────────────────────────

def split_data(X_enc, X_dec, Y, train_ratio=0.7, val_ratio=0.15):
    n = len(X_enc)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    splits = {}
    for name, s, e in [("train", 0, t1), ("val", t1, t2), ("test", t2, n)]:
        splits[name] = {
            "X_enc": X_enc[s:e],
            "X_dec": X_dec[s:e],
            "Y": Y[s:e]
        }
    return splits


# ─────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(seq_len=96, label_len=48, pred_len=24):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Generating synthetic FLUXNET-style carbon flux data...")
    df = generate_carbon_flux_data()
    df.to_csv("data/raw/carbon_flux.csv", index=False)
    print(f"Raw data saved: {len(df)} daily records (2018-2024)")

    print("Engineering features...")
    df = engineer_features(df)

    feature_cols = [
        "temperature", "soil_moisture", "ndvi", "radiation", "vpd",
        "doy_sin", "doy_cos",
        "temperature_7d", "soil_moisture_7d", "ndvi_7d",
        "temperature_30d", "soil_moisture_30d", "ndvi_30d",
        "nee_lag1", "nee_lag7",
        "nee"  # target — must be last column
    ]

    data = df[feature_cols].values

    print("Scaling features...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Creating sequences (seq={seq_len}, label={label_len}, pred={pred_len})...")
    X_enc, X_dec, Y = create_sequences(data_scaled, seq_len, label_len, pred_len)
    print(f"Sequences: X_enc={X_enc.shape}, X_dec={X_dec.shape}, Y={Y.shape}")

    splits = split_data(X_enc, X_dec, Y)
    for split, arrays in splits.items():
        for key, arr in arrays.items():
            np.save(f"data/processed/{split}_{key}.npy", arr)
        print(f"{split}: {arrays['Y'].shape[0]} samples")

    # Save feature info
    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len
    }
    with open("data/processed/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\nData pipeline complete.")
    return meta


# ─────────────────────────────────────────────
# 6. Real GEE Data Pipeline
# ─────────────────────────────────────────────

def run_pipeline_real(
    csv_path="data/raw/carbon_flux_real.csv",
    seq_len=96,
    label_len=48,
    pred_len=24
):
    """
    Run the full pipeline using real GEE satellite data.
    Expects carbon_flux_real.csv to already be downloaded by gee_fetcher.py.
    Falls back to synthetic data if real data file not found.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"WARNING: Real data not found at {csv_path}")
        print("Falling back to synthetic data. Run gee_fetcher.py first for real data.")
        return run_pipeline(seq_len=seq_len, label_len=label_len, pred_len=pred_len)

    print(f"Loading real GEE data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Keep only the core columns (drop per-site columns if present)
    core_cols = ["date", "temperature", "soil_moisture", "ndvi", "radiation", "vpd", "nee"]
    df = df[core_cols].copy()

    # Handle missing values via interpolation
    df = df.set_index("date")
    for col in ["temperature", "soil_moisture", "ndvi", "radiation", "vpd", "nee"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  Interpolating {null_count} missing values in '{col}'")
            df[col] = df[col].interpolate(method="time").bfill().ffill()
    df = df.reset_index()

    print(f"Real data loaded: {len(df)} daily records ({df['date'].min().date()} → {df['date'].max().date()})")

    # Save as canonical raw CSV for dashboard use
    df.to_csv("data/raw/carbon_flux.csv", index=False)
    print("Copied to data/raw/carbon_flux.csv (dashboard-compatible)")

    print("Engineering features...")
    df = engineer_features(df)

    feature_cols = [
        "temperature", "soil_moisture", "ndvi", "radiation", "vpd",
        "doy_sin", "doy_cos",
        "temperature_7d", "soil_moisture_7d", "ndvi_7d",
        "temperature_30d", "soil_moisture_30d", "ndvi_30d",
        "nee_lag1", "nee_lag7",
        "nee"  # target — must be last column
    ]

    data = df[feature_cols].values

    print("Scaling features...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Creating sequences (seq={seq_len}, label={label_len}, pred={pred_len})...")
    X_enc, X_dec, Y = create_sequences(data_scaled, seq_len, label_len, pred_len)
    print(f"Sequences: X_enc={X_enc.shape}, X_dec={X_dec.shape}, Y={Y.shape}")

    splits = split_data(X_enc, X_dec, Y)
    for split, arrays in splits.items():
        for key, arr in arrays.items():
            np.save(f"data/processed/{split}_{key}.npy", arr)
        print(f"{split}: {arrays['Y'].shape[0]} samples")

    # Save feature info
    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "data_source": "real_gee"
    }
    with open("data/processed/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\nReal data pipeline complete.")
    return meta


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "real":
        run_pipeline_real()
    else:
        run_pipeline()
