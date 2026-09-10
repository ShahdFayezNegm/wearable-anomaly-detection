from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

VAL_FEATURES_PATH = PROCESSED_DIR / "wearable_features_labeled.csv"

VAL_SPLIT_PATH = PROCESSED_DIR / "val_features.csv"
TEST_SPLIT_PATH = PROCESSED_DIR / "test_features.csv"

VAL_EVAL_SEQUENCES_PATH = MODELS_DIR / "val_eval_sequences.npy"
TEST_EVAL_SEQUENCES_PATH = MODELS_DIR / "test_eval_sequences.npy"

VAL_OUTPUT_PATH = MODELS_DIR / "val_anomaly_results.csv"
TEST_OUTPUT_PATH = MODELS_DIR / "test_anomaly_results.csv"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128
SEQUENCE_LENGTH = 12


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# Load threshold
# ============================================================

def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return float(data["threshold"])


# ============================================================
# Reconstruction errors
# ============================================================

def predict_reconstruction_errors(
    model,
    sequences,
):
    dataset = TensorDataset(
        torch.from_numpy(
            sequences.astype(np.float32)
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    errors = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(DEVICE)

            reconstruction = model(batch)

            batch_errors = torch.mean(
                (batch - reconstruction) ** 2,
                dim=(1, 2),
            )

            errors.extend(
                batch_errors.cpu().numpy()
            )

    return np.asarray(
        errors,
        dtype=np.float32,
    )


# ============================================================
# Build sequence metadata
# ============================================================

def build_sequence_metadata(
    split_path: Path,
    expected_count: int,
):
    """
    Reconstruct metadata in the same sliding-window order used
    for the evaluation sequences.

    Metadata is attached to the END of each sequence.
    """

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{split_path}"
        )

    df = pd.read_csv(split_path)

    required_columns = [
        "timestamp",
        "subject_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns in {split_path.name}: "
            f"{missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Participant ID may already exist.
    if "participant_id" not in df.columns:
        df["participant_id"] = (
            df["subject_id"]
            .astype(str)
            .str.replace(r"_(a|b)$", "", regex=True)
        )

    metadata_rows = []

    # Keep the same natural ordering used in the CSV.
    group_columns = [
        "participant_id",
        "session_type",
        "subject_id",
    ]

    for _, group in df.groupby(
        group_columns,
        sort=False,
    ):

        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        if len(group) < SEQUENCE_LENGTH:
            continue

        for end_idx in range(
            SEQUENCE_LENGTH - 1,
            len(group),
        ):

            row = group.iloc[end_idx]

            metadata_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "subject_id": row["subject_id"],
                    "participant_id": row["participant_id"],
                    "session_type": row["session_type"],
                    "protocol_stage": row["protocol_stage"],
                    "is_baseline": bool(row["is_baseline"]),
                }
            )

    metadata = pd.DataFrame(
        metadata_rows
    )

    if len(metadata) != expected_count:
        raise ValueError(
            "\nSequence/metadata count mismatch.\n"
            f"Expected sequences : {expected_count}\n"
            f"Metadata rows       : {len(metadata)}\n\n"
            "This means the metadata ordering does not exactly "
            "match the sequence construction."
        )

    return metadata


# ============================================================
# Detect anomalies
# ============================================================

def detect_split(
    split_name: str,
    sequence_path: Path,
    metadata_path: Path,
    output_path: Path,
    model,
    threshold: float,
):
    print("\n" + "=" * 70)
    print(f"{split_name} Anomaly Detection")
    print("=" * 70)

    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Sequence file not found:\n{sequence_path}"
        )

    sequences = np.load(
        sequence_path
    )

    print(
        f"Sequences shape: {sequences.shape}"
    )

    if sequences.ndim != 3:
        raise ValueError(
            f"Expected 3D sequences, got {sequences.ndim}D"
        )

    if sequences.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, "
            f"got {sequences.shape[1]}"
        )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    errors = predict_reconstruction_errors(
        model=model,
        sequences=sequences,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = build_sequence_metadata(
        split_path=metadata_path,
        expected_count=len(errors),
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    results = metadata.copy()

    results["reconstruction_error"] = errors

    results["anomaly_threshold"] = threshold

    results["is_anomaly"] = (
        results["reconstruction_error"] > threshold
    )

    results["anomaly_score"] = (
        results["reconstruction_error"] / threshold
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(results)
    anomalies = int(
        results["is_anomaly"].sum()
    )

    anomaly_rate = (
        anomalies / total
        if total > 0
        else 0.0
    )

    print(
        f"Total sequences : {total}"
    )

    print(
        f"Anomalies       : {anomalies}"
    )

    print(
        f"Normal          : {total - anomalies}"
    )

    print(
        f"Anomaly rate    : {anomaly_rate:.2%}"
    )

    print(
        f"Mean error      : "
        f"{results['reconstruction_error'].mean():.6f}"
    )

    print(
        f"Median error    : "
        f"{results['reconstruction_error'].median():.6f}"
    )

    print(
        f"Max error       : "
        f"{results['reconstruction_error'].max():.6f}"
    )

    print(
        f"Output          : {output_path}"
    )

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Wearable Anomaly Detection")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load model and threshold
    # --------------------------------------------------------

    model = load_model()

    threshold = load_threshold()

    print(
        f"Anomaly threshold: {threshold:.6f}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_results = detect_split(
        split_name="Validation",
        sequence_path=VAL_EVAL_SEQUENCES_PATH,
        metadata_path=VAL_SPLIT_PATH,
        output_path=VAL_OUTPUT_PATH,
        model=model,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = detect_split(
        split_name="Test",
        sequence_path=TEST_EVAL_SEQUENCES_PATH,
        metadata_path=TEST_SPLIT_PATH,
        output_path=TEST_OUTPUT_PATH,
        model=model,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Anomaly Detection Completed")
    print("=" * 70)

    print(
        f"Validation anomalies: "
        f"{int(val_results['is_anomaly'].sum())}/"
        f"{len(val_results)}"
    )

    print(
        f"Test anomalies      : "
        f"{int(test_results['is_anomaly'].sum())}/"
        f"{len(test_results)}"
    )

    print("\nSaved:")
    print(VAL_OUTPUT_PATH)
    print(TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()