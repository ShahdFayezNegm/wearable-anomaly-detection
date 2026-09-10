from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_SEQUENCES_PATH = MODELS_DIR / "train_sequences.npy"
VAL_BASELINE_PATH = MODELS_DIR / "val_baseline_sequences.npy"

MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
HISTORY_PATH = MODELS_DIR / "training_history.csv"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

SEED = 42

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EPOCHS = 50
PATIENCE = 7
MIN_DELTA = 1e-5

HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Reconstruction Error
# ============================================================

def calculate_reconstruction_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    errors = []

    with torch.no_grad():

        for (batch,) in data_loader:

            batch = batch.to(device)

            reconstruction = model(batch)

            # Mean squared error per sequence
            batch_errors = torch.mean(
                (batch - reconstruction) ** 2,
                dim=(1, 2)
            )

            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors, dtype=np.float32)


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("LSTM Autoencoder Training")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    if not TRAIN_SEQUENCES_PATH.exists():
        raise FileNotFoundError(
            f"Training sequences not found:\n{TRAIN_SEQUENCES_PATH}"
        )

    if not VAL_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Validation baseline sequences not found:\n{VAL_BASELINE_PATH}"
        )

    print("\nLoading sequences...")

    train_sequences = np.load(TRAIN_SEQUENCES_PATH)
    val_sequences = np.load(VAL_BASELINE_PATH)

    print(f"Train sequences: {train_sequences.shape}")
    print(f"Val baseline sequences: {val_sequences.shape}")

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    train_sequences = train_sequences.astype(np.float32)
    val_sequences = val_sequences.astype(np.float32)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if train_sequences.ndim != 3:
        raise ValueError(
            f"Expected train sequences with 3 dimensions, "
            f"got {train_sequences.ndim}"
        )

    if val_sequences.ndim != 3:
        raise ValueError(
            f"Expected validation sequences with 3 dimensions, "
            f"got {val_sequences.ndim}"
        )

    if train_sequences.shape[1:] != val_sequences.shape[1:]:
        raise ValueError(
            "Train and validation sequence shapes do not match:\n"
            f"Train: {train_sequences.shape}\n"
            f"Validation: {val_sequences.shape}"
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_tensor = torch.from_numpy(train_sequences)
    val_tensor = torch.from_numpy(val_sequences)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_dim = train_sequences.shape[2]

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print("\nModel configuration:")
    print(f"Input dimension : {input_dim}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Num layers      : {NUM_LAYERS}")
    print(f"Dropout         : {DROPOUT}")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters      : {total_parameters:,}")

    # --------------------------------------------------------
    # Loss + Optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):

        # ====================================================
        # Train
        # ====================================================

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for (batch,) in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(
                reconstruction,
                batch
            )

            loss.backward()

            optimizer.step()

            batch_size = batch.size(0)

            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for (batch,) in val_loader:

                batch = batch.to(device)

                reconstruction = model(batch)

                loss = criterion(
                    reconstruction,
                    batch
                )

                batch_size = batch.size(0)

                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

        val_loss = val_loss_sum / val_samples

        # ====================================================
        # Save history
        # ====================================================

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if val_loss < best_val_loss - MIN_DELTA:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                MODEL_PATH,
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Reload best model
    # --------------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Validation reconstruction errors
    # --------------------------------------------------------

    print("\nCalculating validation reconstruction errors...")

    val_errors = calculate_reconstruction_errors(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    # --------------------------------------------------------
    # Anomaly threshold
    # --------------------------------------------------------
    #
    # We use a high percentile of NORMAL validation baseline
    # reconstruction errors.
    #
    # This is intentionally based only on baseline validation
    # sequences.
    # --------------------------------------------------------

    threshold = float(
        np.percentile(
            val_errors,
            99
        )
    )

    threshold_statistics = {
        "threshold": threshold,
        "method": "validation_baseline_99th_percentile",
        "num_validation_sequences": int(len(val_errors)),
        "mean_error": float(np.mean(val_errors)),
        "std_error": float(np.std(val_errors)),
        "min_error": float(np.min(val_errors)),
        "median_error": float(np.median(val_errors)),
        "max_error": float(np.max(val_errors)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            threshold_statistics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Training completed")
    print("=" * 70)

    print(f"Best epoch       : {best_epoch}")
    print(f"Best val loss    : {best_val_loss:.6f}")

    print("\nValidation baseline reconstruction errors:")
    print(f"Mean   : {np.mean(val_errors):.6f}")
    print(f"Std    : {np.std(val_errors):.6f}")
    print(f"Median : {np.median(val_errors):.6f}")
    print(f"99th % : {threshold:.6f}")
    print(f"Max    : {np.max(val_errors):.6f}")

    print("\nSaved files:")
    print(f"Model    : {MODEL_PATH}")
    print(f"History  : {HISTORY_PATH}")
    print(f"Threshold: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()