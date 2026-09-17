from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.prepare_sequences import (
    apply_imputation,
    apply_scaler,
    build_baseline_sequences,
    load_split,
)
from src.models.train import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MIN_DELTA,
    NUM_LAYERS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    calculate_reconstruction_errors,
    set_seed,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
RETRAINING_DIR = MODELS_DIR / "retraining"

PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.pkl"
)

TRAIN_SEQUENCES_PATH = (
    MODELS_DIR / "train_sequences.npy"
)

VAL_BASELINE_PATH = (
    MODELS_DIR / "val_baseline_sequences.npy"
)

TEST_BASELINE_PATH = (
    MODELS_DIR / "test_baseline_sequences.npy"
)

CURRENT_MODEL_PATH = (
    MODELS_DIR / "lstm_autoencoder.pt"
)

CURRENT_THRESHOLD_PATH = (
    MODELS_DIR / "threshold.json"
)

CURRENT_HISTORY_PATH = (
    MODELS_DIR / "training_history.csv"
)

DEFAULT_CURRENT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "val_features.csv"
)

DEFAULT_DRIFT_REPORT = (
    MODELS_DIR / "drift_report.json"
)

CANDIDATE_MODEL_PATH = (
    RETRAINING_DIR / "candidate_model.pt"
)

CANDIDATE_THRESHOLD_PATH = (
    RETRAINING_DIR / "candidate_threshold.json"
)

CANDIDATE_HISTORY_PATH = (
    RETRAINING_DIR / "candidate_history.csv"
)

RETRAINING_REPORT_PATH = (
    MODELS_DIR / "retraining_report.json"
)

ARCHIVE_DIR = (
    RETRAINING_DIR / "archive"
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_EPOCHS = 50

# Candidate must improve test-baseline loss
# by at least 1% before promotion.
MIN_REQUIRED_IMPROVEMENT = 0.01


# ============================================================
# Data utilities
# ============================================================

def load_sequences(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Sequence file not found:\n{path}"
        )

    data = np.load(
        path
    ).astype(np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D sequences in {path}, "
            f"got shape {data.shape}"
        )

    return data


def create_loader(
    sequences: np.ndarray,
    shuffle: bool,
) -> DataLoader:

    tensor = torch.from_numpy(
        sequences.astype(np.float32)
    )

    dataset = TensorDataset(
        tensor
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Drift report
# ============================================================

def load_drift_report(
    drift_report_path: Path,
) -> dict:

    if not drift_report_path.exists():
        raise FileNotFoundError(
            f"Drift report not found:\n"
            f"{drift_report_path}"
        )

    with drift_report_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    return report


# ============================================================
# Current production-like batch
# ============================================================

def build_current_sequences(
    current_data_path: Path,
) -> np.ndarray:

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n"
            f"{PREPROCESSOR_PATH}"
        )

    print(
        "\nPreparing current production-like batch..."
    )

    current_df = load_split(
        current_data_path
    )

    with PREPROCESSOR_PATH.open(
        "rb"
    ) as file:

        preprocessor = joblib.load(
            file
        )

    feature_columns = (
        preprocessor["feature_columns"]
    )

    train_medians = (
        preprocessor["train_medians"]
    )

    scaler = (
        preprocessor["scaler"]
    )

    sequence_length = int(
        preprocessor["sequence_length"]
    )

    current_df = apply_imputation(
        current_df,
        feature_columns,
        train_medians,
    )

    current_df = apply_scaler(
        current_df,
        feature_columns,
        scaler,
    )

    current_sequences = (
        build_baseline_sequences(
            current_df,
            feature_columns,
            sequence_length,
        )
    )

    print(
        f"Current baseline sequences: "
        f"{current_sequences.shape}"
    )

    return current_sequences


# ============================================================
# Model utilities
# ============================================================

def create_model(
    input_dim: int,
    device: torch.device,
) -> LSTMAutoencoder:

    return LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)


def save_checkpoint(
    model: nn.Module,
    input_dim: int,
    best_val_loss: float,
    best_epoch: int,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        output_path,
    )


def load_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[dict, LSTMAutoencoder]:

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = create_model(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        device=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return checkpoint, model


# ============================================================
# Candidate training
# ============================================================

def train_candidate(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    device: torch.device,
    epochs: int,
) -> dict:

    train_loader = create_loader(
        train_sequences,
        shuffle=True,
    )

    val_loader = create_loader(
        val_sequences,
        shuffle=False,
    )

    input_dim = train_sequences.shape[2]

    model = create_model(
        input_dim=input_dim,
        device=device,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for (batch,) in train_loader:

            batch = batch.to(
                device
            )

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = criterion(
                reconstruction,
                batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = batch.size(0)

            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_samples += batch_size

        train_loss = (
            train_loss_sum
            / train_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for (batch,) in val_loader:

                batch = batch.to(
                    device
                )

                reconstruction = model(
                    batch
                )

                loss = criterion(
                    reconstruction,
                    batch,
                )

                batch_size = batch.size(0)

                val_loss_sum += (
                    loss.item()
                    * batch_size
                )

                val_samples += batch_size

        val_loss = (
            val_loss_sum
            / val_samples
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < (
            best_val_loss
            - MIN_DELTA
        ):

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                input_dim=input_dim,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                output_path=CANDIDATE_MODEL_PATH,
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    CANDIDATE_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        history
    ).to_csv(
        CANDIDATE_HISTORY_PATH,
        index=False,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(
            best_val_loss
        ),
    }


# ============================================================
# Candidate threshold
# ============================================================

def calculate_candidate_threshold(
    candidate_model: nn.Module,
    val_sequences: np.ndarray,
    device: torch.device,
) -> dict:

    val_loader = create_loader(
        val_sequences,
        shuffle=False,
    )

    errors = (
        calculate_reconstruction_errors(
            model=candidate_model,
            data_loader=val_loader,
            device=device,
        )
    )

    threshold = float(
        np.percentile(
            errors,
            99,
        )
    )

    statistics = {
        "threshold": threshold,
        "method": (
            "validation_baseline_99th_percentile"
        ),
        "num_validation_sequences": int(
            len(errors)
        ),
        "mean_error": float(
            np.mean(errors)
        ),
        "std_error": float(
            np.std(errors)
        ),
        "min_error": float(
            np.min(errors)
        ),
        "median_error": float(
            np.median(errors)
        ),
        "max_error": float(
            np.max(errors)
        ),
    }

    with CANDIDATE_THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4,
        )

    return statistics


# ============================================================
# Evaluation
# ============================================================

def calculate_holdout_loss(
    model: nn.Module,
    test_sequences: np.ndarray,
    device: torch.device,
) -> float:

    loader = create_loader(
        test_sequences,
        shuffle=False,
    )

    errors = (
        calculate_reconstruction_errors(
            model=model,
            data_loader=loader,
            device=device,
        )
    )

    return float(
        np.mean(errors)
    )


# ============================================================
# Promotion
# ============================================================

def archive_current_artifacts(
    timestamp: str,
) -> Path:

    archive_path = (
        ARCHIVE_DIR
        / timestamp
    )

    archive_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in (
        CURRENT_MODEL_PATH,
        CURRENT_THRESHOLD_PATH,
        CURRENT_HISTORY_PATH,
    ):

        if source.exists():

            shutil.copy2(
                source,
                archive_path
                / source.name,
            )

    return archive_path


def promote_candidate(
    archive_path: Path,
) -> None:

    print(
        f"\nArchived current artifacts to:\n"
        f"{archive_path}"
    )

    shutil.copy2(
        CANDIDATE_MODEL_PATH,
        CURRENT_MODEL_PATH,
    )

    shutil.copy2(
        CANDIDATE_THRESHOLD_PATH,
        CURRENT_THRESHOLD_PATH,
    )

    shutil.copy2(
        CANDIDATE_HISTORY_PATH,
        CURRENT_HISTORY_PATH,
    )

    print(
        "\nCandidate model promoted successfully."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = ArgumentParser(
        description=(
            "Drift-triggered automatic retraining."
        )
    )

    parser.add_argument(
        "--current-data",
        type=Path,
        default=DEFAULT_CURRENT_DATA,
        help=(
            "Current production-like feature batch."
        ),
    )

    parser.add_argument(
        "--drift-report",
        type=Path,
        default=DEFAULT_DRIFT_REPORT,
        help=(
            "Drift detection JSON report."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Promote only if drift exists "
            "and the candidate improves "
            "the holdout metric."
        ),
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    set_seed(
        SEED
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RETRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("Drift-Triggered Automatic Retraining")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    # --------------------------------------------------------
    # 1. Read drift decision
    # --------------------------------------------------------

    print(
        "\n[1/6] Checking drift report..."
    )

    drift_report = load_drift_report(
        args.drift_report
    )

    drift_detected = bool(
        drift_report.get(
            "drift_detected",
            False,
        )
    )

    print(
        "Drift detected: "
        f"{drift_detected}"
    )

    print(
        "Overall drift rate: "
        f"{drift_report.get(
            'overall_drift_rate',
            0.0
        ):.2%}"
    )

    if not drift_detected:

        print(
            "\nNo drift detected."
        )

        print(
            "Retraining skipped."
        )

        return

    # --------------------------------------------------------
    # 2. Load original train/validation/test sequences
    # --------------------------------------------------------

    print(
        "\n[2/6] Loading model datasets..."
    )

    train_sequences = load_sequences(
        TRAIN_SEQUENCES_PATH
    )

    val_sequences = load_sequences(
        VAL_BASELINE_PATH
    )

    test_sequences = load_sequences(
        TEST_BASELINE_PATH
    )

    # --------------------------------------------------------
    # 3. Build current batch sequences
    # --------------------------------------------------------

    print(
        "\n[3/6] Building current batch..."
    )

    current_sequences = (
        build_current_sequences(
            args.current_data
        )
    )

    if (
        train_sequences.shape[1:]
        != current_sequences.shape[1:]
    ):

        raise ValueError(
            "Current batch sequence shape does "
            "not match training sequence shape.\n"
            f"Train: {train_sequences.shape}\n"
            f"Current: {current_sequences.shape}"
        )

    if (
        train_sequences.shape[1:]
        != val_sequences.shape[1:]
        or
        train_sequences.shape[1:]
        != test_sequences.shape[1:]
    ):

        raise ValueError(
            "Train, validation, and test sequence "
            "shapes must match."
        )

    # --------------------------------------------------------
    # 4. Combine old + current data
    # --------------------------------------------------------

    combined_train = np.concatenate(
        [
            train_sequences,
            current_sequences,
        ],
        axis=0,
    )

    print(
        f"Original training sequences: "
        f"{len(train_sequences):,}"
    )

    print(
        f"Current sequences: "
        f"{len(current_sequences):,}"
    )

    print(
        f"Combined training sequences: "
        f"{len(combined_train):,}"
    )

    print(
        "\n[4/6] Training candidate model..."
    )

    training_result = train_candidate(
        train_sequences=combined_train,
        val_sequences=val_sequences,
        device=device,
        epochs=args.epochs,
    )

    # --------------------------------------------------------
    # 5. Candidate evaluation
    # --------------------------------------------------------

    print(
        "\n[5/6] Evaluating candidate..."
    )

    _, candidate_model = load_checkpoint(
        CANDIDATE_MODEL_PATH,
        device,
    )

    candidate_threshold = (
        calculate_candidate_threshold(
            candidate_model,
            val_sequences,
            device,
        )
    )

    candidate_holdout_loss = (
        calculate_holdout_loss(
            candidate_model,
            test_sequences,
            device,
        )
    )

    _, current_model = load_checkpoint(
        CURRENT_MODEL_PATH,
        device,
    )

    current_holdout_loss = (
        calculate_holdout_loss(
            current_model,
            test_sequences,
            device,
        )
    )

    improvement = (
        current_holdout_loss
        - candidate_holdout_loss
    )

    improvement_rate = (
        improvement
        / current_holdout_loss
        if current_holdout_loss > 0
        else 0.0
    )

    evaluation_passed = (
        improvement_rate
        >= MIN_REQUIRED_IMPROVEMENT
    )

    print(
        f"\nCurrent holdout loss   : "
        f"{current_holdout_loss:.6f}"
    )

    print(
        f"Candidate holdout loss : "
        f"{candidate_holdout_loss:.6f}"
    )

    print(
        f"Improvement rate       : "
        f"{improvement_rate:.2%}"
    )

    print(
        f"Required improvement   : "
        f"{MIN_REQUIRED_IMPROVEMENT:.2%}"
    )

    print(
        f"Evaluation gate        : "
        f"{evaluation_passed}"
    )

    # --------------------------------------------------------
    # 6. Optional promotion
    # --------------------------------------------------------

    promoted = False
    archive_path = None

    if args.promote:

        if evaluation_passed:

            timestamp = (
                datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
            )

            archive_path = (
                archive_current_artifacts(
                    timestamp
                )
            )

            promote_candidate(
                archive_path
            )

            promoted = True

        else:

            print(
                "\nCandidate rejected."
            )

            print(
                "Current production model remains active."
            )

    else:

        print(
            "\nPromotion disabled."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "device": str(device),
        "epochs": int(args.epochs),
        "drift_detected": drift_detected,
        "overall_drift_rate": float(
            drift_report.get(
                "overall_drift_rate",
                0.0,
            )
        ),
        "current_data": str(
            args.current_data
        ),
        "original_train_sequences": int(
            len(train_sequences)
        ),
        "current_sequences": int(
            len(current_sequences)
        ),
        "combined_train_sequences": int(
            len(combined_train)
        ),
        "validation_sequences": int(
            len(val_sequences)
        ),
        "test_baseline_sequences": int(
            len(test_sequences)
        ),
        "candidate_best_epoch": int(
            training_result[
                "best_epoch"
            ]
        ),
        "candidate_best_val_loss": float(
            training_result[
                "best_val_loss"
            ]
        ),
        "candidate_threshold": float(
            candidate_threshold[
                "threshold"
            ]
        ),
        "current_holdout_loss": float(
            current_holdout_loss
        ),
        "candidate_holdout_loss": float(
            candidate_holdout_loss
        ),
        "improvement": float(
            improvement
        ),
        "improvement_rate": float(
            improvement_rate
        ),
        "minimum_required_improvement": (
            MIN_REQUIRED_IMPROVEMENT
        ),
        "evaluation_passed": bool(
            evaluation_passed
        ),
        "promotion_requested": bool(
            args.promote
        ),
        "promoted": bool(
            promoted
        ),
        "archive_path": (
            str(archive_path)
            if archive_path
            else None
        ),
    }

    with RETRAINING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"\nRetraining report saved to:\n"
        f"{RETRAINING_REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()