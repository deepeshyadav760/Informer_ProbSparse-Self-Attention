"""
main.py — Run the full Informer Carbon Flux Pipeline

Usage:
  python main.py                  # uses synthetic data (default)
  python main.py --data real      # uses real GEE satellite data
  python main.py --data real --fetch  # fetch GEE data first, then run
"""

import os
import sys
import argparse

# Fix Windows console encoding to handle UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Silence git warnings from MLflow (no git needed for local tracking)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def main():
    parser = argparse.ArgumentParser(description="Informer Carbon Flux Pipeline")
    parser.add_argument(
        "--data", choices=["synthetic", "real"], default="synthetic",
        help="Data source: 'synthetic' (default) or 'real' (GEE satellite data)"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch real GEE data before running pipeline (requires --data real)"
    )
    parser.add_argument(
        "--project", default="ee-deepeshy",
        help="Google Earth Engine project ID (default: ee-deepeshy)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Informer: Carbon Flux Forecasting Pipeline")
    print("  Zhou et al., AAAI 2021 Best Paper")
    print("  Applied to Sundarbans & Chilika Carbon Sequestration")
    print(f"  Data Mode: {args.data.upper()}")
    print("=" * 60)

    # ── Step 0: Fetch real GEE data (optional) ─────────────────
    if args.fetch and args.data == "real":
        print("\n[0/3] Fetching real satellite data from Google Earth Engine...")
        from data.gee_fetcher import fetch_and_save
        fetch_and_save(project_id=args.project)

    # ── Step 1: Data Pipeline ───────────────────────────────────
    print("\n[1/3] Running data pipeline...")
    if args.data == "real":
        from data.preprocess import run_pipeline_real
        run_pipeline_real()
    else:
        from data.preprocess import run_pipeline
        run_pipeline()

    # ── Step 2: Train ───────────────────────────────────────────
    print("\n[2/3] Training Informer model...")
    from train import train
    train()

    # ── Step 3: Evaluate ────────────────────────────────────────
    print("\n[3/3] Evaluating model & generating plots...")
    from evaluate import run_evaluation
    metrics, _, _ = run_evaluation()

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAPE: {metrics['MAPE']}%")
    print("=" * 60)
    print("\nTo launch the Streamlit dashboard:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
