"""
Training Loop for Informer Carbon Flux Forecasting
With MLflow experiment tracking — every run is logged and traceable.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import time
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from model.informer import build_model

# ---------------------------------------------
# Dataset
# ---------------------------------------------

class CarbonFluxDataset(Dataset):
    def __init__(self, split="train"):
        self.X_enc = np.load(f"data/processed/{split}_X_enc.npy").astype(np.float32)
        self.X_dec = np.load(f"data/processed/{split}_X_dec.npy").astype(np.float32)
        self.Y     = np.load(f"data/processed/{split}_Y.npy").astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_enc[idx]),
            torch.tensor(self.X_dec[idx]),
            torch.tensor(self.Y[idx]).unsqueeze(-1)  # [pred_len, 1]
        )


# ---------------------------------------------
# Metrics
# ---------------------------------------------

def mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()

def mape(pred, true, eps=1e-8):
    return torch.mean(torch.abs((pred - true) / (true.abs() + eps))).item() * 100


# ---------------------------------------------
# Training
# ---------------------------------------------

def train(config=None):
    # Default config
    cfg = {
        "epochs":        30,
        "batch_size":    32,
        "lr":            1e-4,
        "seq_len":       96,
        "label_len":     48,
        "pred_len":      24,
        "patience":      5,
        "grad_clip":     1.0,
        "weight_decay":  1e-5,
    }
    if config:
        cfg.update(config)

    device = torch.device("cpu")
    print(f"Training on: {device}")

    # Load meta
    with open("data/processed/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n_features = meta["n_features"]

    # Datasets
    train_ds = CarbonFluxDataset("train")
    val_ds   = CarbonFluxDataset("val")
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = build_model(
        n_features=n_features,
        seq_len=cfg["seq_len"],
        label_len=cfg["label_len"],
        pred_len=cfg["pred_len"],
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)

    # MLflow tracking — use SQLite backend (MLflow 3.x removed file store support)
    # Use a path without spaces to avoid URL-encoding issues on Windows
    mlflow_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # fallback safety
    mlflow.set_experiment("informer_carbon_flux")

    with mlflow.start_run(run_name="informer_v1"):
        mlflow.log_params(cfg)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("model_params", model.count_parameters())

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            # -- Train -----------------------------------
            model.train()
            train_losses = []
            for x_enc, x_dec, y in train_dl:
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)

                optimizer.zero_grad()
                pred, _, _ = model(x_enc, x_dec)
                loss = criterion(pred, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # -- Validate ---------------------------------
            model.eval()
            val_losses, all_pred, all_true = [], [], []
            with torch.no_grad():
                for x_enc, x_dec, y in val_dl:
                    x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                    pred, _, _ = model(x_enc, x_dec)
                    val_losses.append(criterion(pred, y).item())
                    all_pred.append(pred.cpu())
                    all_true.append(y.cpu())

            val_loss = np.mean(val_losses)
            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)

            val_mae  = mae(all_pred, all_true)
            val_rmse = rmse(all_pred, all_true)
            val_mape = mape(all_pred, all_true)

            scheduler.step(val_loss)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_mae":    val_mae,
                "val_rmse":   val_rmse,
                "val_mape":   val_mape
            }, step=epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["val_rmse"].append(val_rmse)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{cfg['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE: {val_mae:.4f} | "
                f"RMSE: {val_rmse:.4f} | "
                f"MAPE: {val_mape:.2f}% | "
                f"{elapsed:.1f}s"
            )

            # -- Early Stopping & Checkpoint --------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_loss":   val_loss,
                    "config":     cfg,
                    "n_features": n_features
                }, "checkpoints/best_model.pt")
                print(f"  [OK] Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Save history
        import json
        with open("checkpoints/history.json", "w") as f:
            json.dump(history, f)

        # Log best model to MLflow
        mlflow.log_artifact("checkpoints/best_model.pt")
        mlflow.log_metric("best_val_loss", best_val_loss)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        return history


if __name__ == "__main__":
    # First run pipeline if data doesn't exist
    if not os.path.exists("data/processed/train_X_enc.npy"):
        from data.preprocess import run_pipeline
        run_pipeline()
    train()
