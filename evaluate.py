"""
Evaluation & Interpretability Plots for Informer Carbon Flux Model
Generates:
  1. Forecast vs Actual plot
  2. Training history curves
  3. Error distribution
  4. Seasonal performance breakdown
  5. Attention heatmap (interpretability)
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.utils.data import DataLoader
from train import CarbonFluxDataset, mae, rmse, mape
from model.informer import build_model

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"pred": "#2ecc71", "true": "#2c3e50", "error": "#e74c3c", "attn": "#3498db"}


def load_model_and_data():
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
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    return model, test_dl, meta, cfg, ckpt


def run_inference(model, test_dl):
    all_pred, all_true, all_enc_attns = [], [], []
    with torch.no_grad():
        for x_enc, x_dec, y in test_dl:
            pred, enc_attns, _ = model(x_enc, x_dec)
            all_pred.append(pred.numpy())
            all_true.append(y.numpy())
            all_enc_attns.append(enc_attns[0].numpy())  # first encoder layer

    return (
        np.concatenate(all_pred, axis=0),    # [N, pred_len, 1]
        np.concatenate(all_true, axis=0),    # [N, pred_len, 1]
        np.concatenate(all_enc_attns, axis=0)  # [N, n_top, seq_len]
    )


def plot_forecast(pred, true, n_samples=3, save_dir="plots"):
    """Plot predicted vs actual NEE for n random test windows."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    idxs = np.random.choice(len(pred), n_samples, replace=False)
    for ax, idx in zip(axes, idxs):
        p = pred[idx, :, 0]
        t = true[idx, :, 0]
        days = np.arange(len(p))
        ax.plot(days, t, color=COLORS["true"], label="Actual NEE", linewidth=2)
        ax.plot(days, p, color=COLORS["pred"], label="Predicted NEE",
                linewidth=2, linestyle="--")
        ax.fill_between(days, p, t, alpha=0.15, color=COLORS["error"])
        ax.set_xlabel("Forecast Day", fontsize=11)
        ax.set_ylabel("NEE (scaled)", fontsize=11)
        ax.set_title(f"Carbon Flux Forecast — Test Sample {idx}", fontsize=13, fontweight="bold")
        ax.legend()

    plt.suptitle("Informer: Carbon Flux (NEE) Forecast vs Actual", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{save_dir}/forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_training_history(save_dir="plots"):
    with open("checkpoints/history.json") as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(h["train_loss"], label="Train Loss", color="#2c3e50", linewidth=2)
    axes[0].plot(h["val_loss"],   label="Val Loss",   color="#e74c3c", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    axes[1].plot(h["val_mae"],  label="MAE",  color="#3498db", linewidth=2)
    axes[1].plot(h["val_rmse"], label="RMSE", color="#9b59b6", linewidth=2)
    axes[1].set_title("Validation Metrics", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].legend()

    plt.suptitle("Informer Training History — Carbon Flux Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_error_distribution(pred, true, save_dir="plots"):
    errors = (pred[:, :, 0] - true[:, :, 0]).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(errors, bins=60, color=COLORS["error"], alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
    axes[0].set_title("Prediction Error Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Error (Predicted − Actual NEE)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].scatter(true[:, :, 0].flatten(), pred[:, :, 0].flatten(),
                    alpha=0.3, s=5, color=COLORS["attn"])
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    axes[1].plot(lim, lim, "k--", linewidth=1.5, label="Perfect Prediction")
    axes[1].set_title("Actual vs Predicted NEE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Actual NEE")
    axes[1].set_ylabel("Predicted NEE")
    axes[1].legend()

    plt.suptitle("Error Analysis — Informer Carbon Flux Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/error_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_attention_heatmap(enc_attns, save_dir="plots"):
    """Visualize ProbSparse attention — which timesteps the model focuses on."""
    # Average over all test samples
    avg_attn = enc_attns.mean(axis=0)   # [n_top, seq_len]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        avg_attn,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
        xticklabels=10
    )
    ax.set_title(
        "ProbSparse Self-Attention Heatmap\n(Rows = Selected Dominant Queries, Cols = Encoder Timesteps)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Encoder Timestep (Days Lookback)")
    ax.set_ylabel("Selected Query Position")

    plt.tight_layout()
    path = f"{save_dir}/attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def compute_metrics(pred, true):
    p = torch.tensor(pred)
    t = torch.tensor(true)
    return {
        "MAE":  round(mae(p, t), 4),
        "RMSE": round(rmse(p, t), 4),
        "MAPE": round(mape(p, t), 2)
    }


def run_evaluation():
    os.makedirs("plots", exist_ok=True)
    print("Loading model and test data...")
    model, test_dl, meta, cfg, ckpt = load_model_and_data()

    print("Running inference on test set...")
    pred, true, enc_attns = run_inference(model, test_dl)

    metrics = compute_metrics(pred, true)
    print("\n── Test Metrics ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nGenerating plots...")
    plot_forecast(pred, true, n_samples=3)
    plot_training_history()
    plot_error_distribution(pred, true)
    plot_attention_heatmap(enc_attns)

    # Save metrics
    import json
    with open("plots/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll evaluation plots saved to plots/")
    return metrics, pred, true


if __name__ == "__main__":
    run_evaluation()
