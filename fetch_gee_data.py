"""
fetch_gee_data.py — Standalone script to fetch real GEE satellite data.

Run AFTER completing: earthengine authenticate

Usage:
  python fetch_gee_data.py

This will fetch MODIS, SMAP, and ERA5 data for Sundarbans + Chilika Lake
and save to data/raw/carbon_flux_real.csv
"""

import os, sys

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.gee_fetcher import fetch_and_save

if __name__ == "__main__":
    print("=" * 60)
    print("  GEE Satellite Data Fetcher")
    print("  Project: ee-deepeshy")
    print("  Sites: Sundarbans + Chilika Lake (merged - Option A)")
    print("  Period: 2018-2024")
    print("=" * 60)
    df = fetch_and_save(project_id="ee-deepeshy")
    print("\nDone! Now run the full pipeline:")
    print("  python main.py --data real")
