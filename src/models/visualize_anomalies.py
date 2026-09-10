from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
TEST_RESULTS_PATH = MODELS_DIR / "test_anomaly_results.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.csv"
TEST_EVAL_PATH = MODELS_DIR / "test_eval_sequences.npy"

OUTPUT_DIR = MODELS_DIR / "anomaly_plots"


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQUENCE_LENGTH = 12
TOP_ANOMALIES = 3


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
# Build sequence metadata
# ============================================================

def build_metadata(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    metadata = []

    group_columns = [
        "participant_id",
        "session_type",
        "subject_id",
    ]

    for _, group in df.groupby(
        group_columns,
        sort=False,
    ):

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(group) < SEQUENCE_LENGTH:
            continue

        for end_idx in range(
            SEQUENCE_LENGTH - 1,
            len(group),
        ):

            row = group.iloc[end_idx]

            metadata.append(
                {
                    "timestamp": row["timestamp"],
                    "participant_id": row["participant_id"],
                    "subject_id": row["subject_id"],
                    "session_type": row["session_type"],
                    "protocol_stage": row["protocol_stage"],
                }
            )

    return pd.DataFrame(metadata)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Anomaly Visualization")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading data...")

    results = pd.read_csv(
        TEST_RESULTS_PATH
    )

    features = pd.read_csv(
        TEST_FEATURES_PATH
    )

    sequences = np.load(
        TEST_EVAL_PATH
    )

    results["timestamp"] = pd.to_datetime(
        results["timestamp"]
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"]
    )

    print(
        f"Test results : {len(results)} rows"
    )

    print(
        f"Test sequences: {sequences.shape}"
    )

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    metadata = build_metadata(
        features
    )

    if len(metadata) != len(sequences):
        raise ValueError(
            "\nSequence/metadata mismatch:\n"
            f"Sequences: {len(sequences)}\n"
            f"Metadata : {len(metadata)}"
        )

    # --------------------------------------------------------
    # Select anomalies
    # --------------------------------------------------------

    anomalies = (
        results[
            results["is_anomaly"] == True
        ]
        .sort_values(
            "reconstruction_error",
            ascending=False,
        )
        .head(TOP_ANOMALIES)
    )

    print(
        f"\nTop anomalies selected: "
        f"{len(anomalies)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    excluded = {
        "timestamp",
        "subject_id",
        "participant_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    print(
        f"Feature count: {len(feature_columns)}"
    )

    if len(feature_columns) != sequences.shape[2]:
        raise ValueError(
            "\nFeature count mismatch:\n"
            f"CSV features: {len(feature_columns)}\n"
            f"Sequence features: {sequences.shape[2]}"
        )

    selected_features = [
        "HR_mean",
        "EDA_mean",
        "TEMP_mean",
        "ACC_MAG_mean",
    ]

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    generated = 0

    print("\nGenerating plots...")

    for rank, (_, anomaly) in enumerate(
        anomalies.iterrows(),
        start=1,
    ):

        timestamp = anomaly["timestamp"]

        participant_id = anomaly[
            "participant_id"
        ]

        matches = metadata[
            (metadata["timestamp"] == timestamp)
            &
            (
                metadata["participant_id"]
                == participant_id
            )
        ]

        if matches.empty:
            print(
                f"Skipping {timestamp}: "
                "metadata not found."
            )
            continue

        sequence_index = matches.index[0]

        sequence = sequences[
            sequence_index
        ]

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        tensor = (
            torch.from_numpy(
                sequence.astype(np.float32)
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            reconstruction = model(
                tensor
            ).cpu().numpy()[0]

        # ----------------------------------------------------
        # Plot each selected feature
        # ----------------------------------------------------

        for feature in selected_features:

            if feature not in feature_columns:
                continue

            feature_index = feature_columns.index(
                feature
            )

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(
                range(SEQUENCE_LENGTH),
                sequence[:, feature_index],
                label="Actual",
            )

            plt.plot(
                range(SEQUENCE_LENGTH),
                reconstruction[:, feature_index],
                label="Reconstructed",
            )

            plt.title(
                f"{feature}\n"
                f"Participant: {participant_id} | "
                f"Stage: {anomaly['protocol_stage']}\n"
                f"Error: "
                f"{anomaly['reconstruction_error']:.2f}"
            )

            plt.xlabel(
                "Timestep"
            )

            plt.ylabel(
                "Standardized value"
            )

            plt.legend()

            plt.tight_layout()

            filename = (
                f"{rank:02d}_"
                f"{participant_id}_"
                f"{timestamp.strftime('%Y%m%d_%H%M%S')}_"
                f"{feature}.png"
            )

            output_path = (
                OUTPUT_DIR / filename
            )

            plt.savefig(
                output_path,
                dpi=150,
            )

            plt.close()

            generated += 1

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Visualization Completed")
    print("=" * 70)

    print(
        f"Anomalies visualized: {len(anomalies)}"
    )

    print(
        f"Plots generated: {generated}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()