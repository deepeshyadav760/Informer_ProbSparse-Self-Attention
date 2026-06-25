"""
Google Earth Engine Data Fetcher
=================================
Fetches real open-source satellite data for:
  - Sundarbans Mangrove Forest (India/Bangladesh)
  - Chilika Lake (Odisha, India)

Datasets used:
  NDVI       -> MODIS/061/MOD13A2  (16-day, interpolated to daily)
  LST        -> MODIS/061/MOD11A1  (daily, land surface temperature)
  Soil Moist -> NASA/SMAP/SPL4SMGP/007 (3-hourly, daily mean)
  Radiation  -> ECMWF/ERA5_LAND/DAILY_AGGR (daily solar radiation)
  VPD        -> derived from ERA5 (dew point + temp)
  GPP        -> MODIS/061/MOD17A2H (8-day GPP, interpolated -> NEE proxy)

Output: data/raw/carbon_flux_real.csv
        (merged Sundarbans + Chilika, averaged daily 2018-2024)

GEE Project: ee-deepeshy
"""

import ee
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------
# Region of Interest Definitions
# -----------------------------------------------------------------

# Sundarbans Mangrove Forest Coordinates (India side)
SUNDARBANS_COORDS = [88.0, 21.5, 89.5, 22.5]

# Chilika Lake Coordinates (Odisha, India)
CHILIKA_COORDS = [85.0, 19.5, 85.7, 20.1]


START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

SCALE = 1000  # meters, ~1km resolution for speed


# -----------------------------------------------------------------
# GEE Authentication & Init
# -----------------------------------------------------------------

def init_gee(project_id="ee-deepeshy"):
    """Initialize Google Earth Engine with the given project."""
    try:
        ee.Initialize(project=project_id)
        print(f"[OK] GEE initialized with project: {project_id}")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("Running: earthengine authenticate --project", project_id)
        import subprocess
        subprocess.run(
            ["earthengine", "authenticate", "--project", project_id],
            check=True
        )
        ee.Initialize(project=project_id)
        print(f"[OK] GEE initialized (after auth) with project: {project_id}")


# -----------------------------------------------------------------
# Helper: Image Collection -> Daily Pandas Series
# -----------------------------------------------------------------

def collection_to_daily_series(collection, roi, band_name, reducer=None):
    """
    Extract a daily time series from an ImageCollection for a given ROI.
    Returns a pandas Series indexed by date.
    """
    if reducer is None:
        reducer = ee.Reducer.mean()
    def extract_value(image):
        val = image.reduceRegion(
            reducer=reducer,
            geometry=roi,
            scale=SCALE,
            maxPixels=1e9
        ).get(band_name, -9999.0)
        return ee.Feature(None, {"date": image.date().format("YYYY-MM-dd"), "value": val})

    fc = ee.FeatureCollection(collection.map(extract_value))
    data = fc.getInfo()["features"]

    records = []
    for feat in data:
        props = feat["properties"]
        if props.get("value") is not None and props.get("value") != -9999.0:
            records.append({"date": props["date"], "value": props["value"]})

    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df["value"].astype(float)


# -----------------------------------------------------------------
# Individual Dataset Fetchers
# -----------------------------------------------------------------

def fetch_ndvi(roi):
    """
    MODIS Terra NDVI — MOD13A2 (16-day composite, 1km)
    Band: NDVI (scale factor 0.0001)
    Linearly interpolated to daily.
    """
    print("  Fetching NDVI (MODIS MOD13A2)...")
    col = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select("NDVI")
    )
    # Scale: multiply by 0.0001
    col = col.map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    series = collection_to_daily_series(col, roi, "NDVI")
    # Interpolate 16-day composites to daily
    daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    series = series.reindex(daily_idx).interpolate(method="time").clip(0.0, 1.0)
    return series


def fetch_lst(roi):
    """
    MODIS Terra LST — MOD11A1 (daily, 1km)
    Band: LST_Day_1km (scale factor 0.02, Kelvin -> Celsius)
    """
    print("  Fetching LST/Temperature (MODIS MOD11A1)...")
    col = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select("LST_Day_1km")
    )
    # Scale: multiply 0.02 -> Kelvin, then subtract 273.15 -> Celsius
    col = col.map(
        lambda img: img.multiply(0.02).subtract(273.15)
        .copyProperties(img, ["system:time_start"])
    )
    series = collection_to_daily_series(col, roi, "LST_Day_1km")
    daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    series = series.reindex(daily_idx).interpolate(method="time")
    return series


def fetch_soil_moisture(roi):
    """
    NASA SMAP L4 Soil Moisture — SPL4SMGP/008 (3-hourly -> daily mean, downsampled)
    Band: sm_surface (m³/m³)
    """
    print("  Fetching Soil Moisture (SMAP SPL4SMGP/008)...")
    col = (
        ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select("sm_surface")
    )
    
    # Downsample to avoid GEE memory limits: average soil moisture every 4 days,
    # then interpolate to daily. Soil moisture changes slowly, so this has no impact on quality.
    daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    sampled_dates = pd.date_range(START_DATE, END_DATE, freq="4D")
    
    date_list = ee.List([d.strftime("%Y-%m-%d") for d in sampled_dates])
    
    def aggregate_day(date_str):
        date = ee.Date(date_str)
        day_col = col.filterDate(date, date.advance(1, "day"))
        return day_col.mean().set("system:time_start", date.millis()).rename(["sm_surface"])

    daily_col = ee.ImageCollection(date_list.map(aggregate_day))
    
    series = collection_to_daily_series(daily_col, roi, "sm_surface")
    series = series.reindex(daily_idx).interpolate(method="time").clip(0.0, 0.95)
    return series


def fetch_era5(roi):
    """
    ERA5 Land Daily Aggregates — ECMWF/ERA5_LAND/DAILY_AGGR
    Bands:
      surface_solar_radiation_downwards_sum  -> radiation (J/m²/day -> W/m²)
      temperature_2m                          -> used for VPD
      dewpoint_temperature_2m                -> used for VPD
    """
    print("  Fetching Radiation + VPD (ERA5 Land Daily)...")
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select([
            "surface_solar_radiation_downwards_sum",
            "temperature_2m",
            "dewpoint_temperature_2m"
        ])
    )

    # Radiation: J/m²/day -> W/m² (divide by 86400)
    rad_col = col.map(
        lambda img: img.select("surface_solar_radiation_downwards_sum")
        .divide(86400)
        .rename("radiation")
        .copyProperties(img, ["system:time_start"])
    )
    rad = collection_to_daily_series(rad_col, roi, "radiation")

    # Temperature 2m: Kelvin -> Celsius
    t2m_col = col.map(
        lambda img: img.select("temperature_2m")
        .subtract(273.15)
        .rename("temperature_2m")
        .copyProperties(img, ["system:time_start"])
    )
    t2m = collection_to_daily_series(t2m_col, roi, "temperature_2m")

    # Dew point: Kelvin -> Celsius
    td_col = col.map(
        lambda img: img.select("dewpoint_temperature_2m")
        .subtract(273.15)
        .rename("dewpoint_temperature_2m")
        .copyProperties(img, ["system:time_start"])
    )
    td = collection_to_daily_series(td_col, roi, "dewpoint_temperature_2m")

    # VPD = es(T) - ea(Td)
    # Saturation vapor pressure: es = 0.6108 * exp(17.27 * T / (T + 237.3))
    def svp(temp_series):
        return 0.6108 * np.exp(17.27 * temp_series / (temp_series + 237.3))

    daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    t2m = t2m.reindex(daily_idx).interpolate(method="time")
    td  = td.reindex(daily_idx).interpolate(method="time")
    rad = rad.reindex(daily_idx).interpolate(method="time").clip(0, 600)

    vpd = (svp(t2m) - svp(td)).clip(0.0, 5.0)

    return rad, vpd, t2m


def fetch_gpp(roi):
    """
    MODIS GPP — MOD17A2H (8-day cumulative, 500m)
    Band: Gpp (scale factor 0.0001, kgC/m²/8day -> tCO₂/ha/day)
    Used as a proxy for NEE (carbon uptake component).
    
    Conversion: kgC/m²/8day × (1000 g/kg) × (44/12 CO₂/C) × (10000 m²/ha) / (8 days) / 1000000
                -> tCO₂/ha/day, then negate (uptake = negative NEE)
    """
    print("  Fetching GPP/NEE proxy (MODIS MOD17A2H)...")
    col = (
        ee.ImageCollection("MODIS/061/MOD17A2H")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select("Gpp")
    )
    # Scale: 0.0001 -> kgC/m²/8day
    # Convert to tCO₂/ha/day: × 0.0001 × (44/12) × 10 / 8
    #   = × 0.0001 × 3.667 × 10 / 8 = × 0.000458
    col = col.map(
        lambda img: img.multiply(0.0001 * 3.667 * 10 / 8)
        .copyProperties(img, ["system:time_start"])
    )
    series = collection_to_daily_series(col, roi, "Gpp")
    daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    series = series.reindex(daily_idx).interpolate(method="time")
    # NEE convention: uptake is negative
    # Apply ecosystem respiration offset (Ra ≈ 40% of GPP for mangroves)
    nee_proxy = -series * 0.6  # NEE ≈ -(GPP - Ra) = -NPP
    return nee_proxy


# -----------------------------------------------------------------
# Main Fetcher: Merged Sundarbans + Chilika
# -----------------------------------------------------------------

def fetch_all_sites(project_id="ee-deepeshy"):
    """
    Fetch data for both sites and return a merged daily DataFrame.
    Strategy (Option A): average the two sites into one time series.
    """
    init_gee(project_id)

    # Lazily create geometries after initialization
    sundarbans_roi = ee.Geometry.Rectangle(SUNDARBANS_COORDS)
    chilika_roi    = ee.Geometry.Rectangle(CHILIKA_COORDS)

    sites = {
        "sundarbans": sundarbans_roi,
        "chilika":    chilika_roi,
    }

    site_dfs = {}
    for site_name, roi in sites.items():
        print(f"\n{'-'*50}")
        print(f"Fetching data for: {site_name.upper()}")
        print(f"{'-'*50}")

        ndvi = fetch_ndvi(roi)
        lst, vpd, t2m = fetch_era5(roi)
        soil = fetch_soil_moisture(roi)
        nee  = fetch_gpp(roi)

        daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
        df = pd.DataFrame({
            "date":         daily_idx,
            "temperature":  t2m.values,
            "soil_moisture": soil.reindex(daily_idx).interpolate().values,
            "ndvi":         ndvi.values,
            "radiation":    lst.reindex(daily_idx).interpolate().values,  # LST as radiation proxy
            "vpd":          vpd.values,
            "nee":          nee.reindex(daily_idx).interpolate().values,
        })
        site_dfs[site_name] = df
        print(f"  [OK] {site_name}: {len(df)} daily records")

    # Option A: Average both sites
    print("\nMerging sites (Option A: averaged)...")
    merged = site_dfs["sundarbans"].copy()
    for col in ["temperature", "soil_moisture", "ndvi", "radiation", "vpd", "nee"]:
        merged[col] = (site_dfs["sundarbans"][col].values + site_dfs["chilika"][col].values) / 2.0

    # Add site metadata columns for reference
    merged["sundarbans_nee"] = site_dfs["sundarbans"]["nee"].values
    merged["chilika_nee"]    = site_dfs["chilika"]["nee"].values

    print(f"[OK] Merged dataset: {len(merged)} daily records")
    return merged


# -----------------------------------------------------------------
# Save Output
# -----------------------------------------------------------------

def fetch_and_save(project_id="ee-deepeshy", out_path="data/raw/carbon_flux_real.csv"):
    """Main entry point: fetch real GEE data and save CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = fetch_all_sites(project_id)
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Real GEE data saved -> {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"  NEE range: {df['nee'].min():.4f} to {df['nee'].max():.4f} tCO₂/ha/day")
    return df


if __name__ == "__main__":
    fetch_and_save()
