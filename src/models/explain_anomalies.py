from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

TEST_RESULTS_PATH = MODELS_DIR / "test_anomaly_results.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.csv"

TEST_EVAL_SEQUENCES_PATH = MODELS_DIR / "test_eval_sequences.npy"
TEST_BASELINE_SEQUENCES_PATH = MODELS_DIR / "test_baseline_sequences.npy"

OUTPUT_PATH = MODELS_DIR / "test_explanations.csv"


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 12
TOP_ANOMALIES = 20
TOP_FEATURES = 8

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
# Feature columns
# ============================================================

def get_feature_columns(df):

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
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    return feature_columns


# ============================================================
# Build evaluation metadata
# ============================================================

def build_sequence_metadata(df):

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
                    "is_baseline": bool(
                        row["is_baseline"]
                    ),
                }
            )

    return pd.DataFrame(metadata)


# ============================================================
# Reconstruction
# ============================================================

def reconstruct(
    model,
    sequences,
):

    x = torch.from_numpy(
        sequences.astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():
        reconstruction = model(x)

    return reconstruction.cpu().numpy()


# ============================================================
# Baseline statistics
# ============================================================

def build_baseline_statistics(
    features_df,
    feature_columns,
):

    baseline = features_df[
        features_df["is_baseline"] == True
    ].copy()

    stats = {}

    for participant_id, group in baseline.groupby(
        "participant_id",
        sort=False,
    ):

        stats[participant_id] = {
            "mean": group[feature_columns].mean(),
            "std": group[feature_columns].std(),
        }

    return stats


# ============================================================
# Feature contribution to reconstruction error
# ============================================================

def feature_reconstruction_errors(
    sequence,
    reconstruction,
    feature_columns,
):

    mse = np.mean(
        (sequence - reconstruction) ** 2,
        axis=0,
    )

    feature_error = pd.Series(
        mse,
        index=feature_columns,
    )

    return feature_error.sort_values(
        ascending=False
    )


# ============================================================
# Raw baseline-relative changes
# ============================================================

def calculate_baseline_changes(
    feature_frame,
    participant_id,
    baseline_stats,
):

    if participant_id not in baseline_stats:
        return pd.Series(dtype=float)

    means = baseline_stats[
        participant_id
    ]["mean"]

    stds = baseline_stats[
        participant_id
    ]["std"]

    values = feature_frame.mean()

    safe_stds = stds.replace(
        0,
        np.nan,
    )

    standardized_change = (
        values - means
    ) / safe_stds

    return standardized_change.abs()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Anomaly Explainability - What Changed?")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    results = pd.read_csv(
        TEST_RESULTS_PATH
    )

    features_df = pd.read_csv(
        TEST_FEATURES_PATH
    )

    eval_sequences = np.load(
        TEST_EVAL_SEQUENCES_PATH
    )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    results["timestamp"] = pd.to_datetime(
        results["timestamp"]
    )

    features_df["timestamp"] = pd.to_datetime(
        features_df["timestamp"]
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        features_df
    )

    print(
        f"\nFeature count: {len(feature_columns)}"
    )

    if len(feature_columns) != eval_sequences.shape[2]:

        raise ValueError(
            "\nFeature count mismatch.\n"
            f"CSV features       : {len(feature_columns)}\n"
            f"Sequence features  : {eval_sequences.shape[2]}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = build_sequence_metadata(
        features_df
    )

    if len(metadata) != len(eval_sequences):

        raise ValueError(
            "\nSequence/metadata mismatch.\n"
            f"Sequences : {len(eval_sequences)}\n"
            f"Metadata  : {len(metadata)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    print("\nReconstructing test sequences...")

    reconstructions = reconstruct(
        model,
        eval_sequences,
    )

    # --------------------------------------------------------
    # Baseline statistics
    # --------------------------------------------------------

    baseline_stats = build_baseline_statistics(
        features_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Select anomalies
    # --------------------------------------------------------

    anomalies = results[
        results["is_anomaly"] == True
    ].copy()

    anomalies = anomalies.sort_values(
        "reconstruction_error",
        ascending=False,
    ).head(
        TOP_ANOMALIES
    )

    print(
        f"\nExplaining top {len(anomalies)} anomalies..."
    )

    explanation_rows = []

    # --------------------------------------------------------
    # Explain each anomaly
    # --------------------------------------------------------

    for _, anomaly in anomalies.iterrows():

        timestamp = anomaly["timestamp"]

        participant_id = anomaly[
            "participant_id"
        ]

        protocol_stage = anomaly[
            "protocol_stage"
        ]

        anomaly_score = anomaly[
            "anomaly_score"
        ]

        reconstruction_error = anomaly[
            "reconstruction_error"
        ]

        # -----------------------------------------------
        # Find exact sequence
        # -----------------------------------------------

        matches = metadata[
            (metadata["timestamp"] == timestamp)
            &
            (
                metadata["participant_id"]
                == participant_id
            )
        ]

        if matches.empty:
            continue

        sequence_index = matches.index[0]

        sequence = eval_sequences[
            sequence_index
        ]

        reconstruction = reconstructions[
            sequence_index
        ]

        # -----------------------------------------------
        # Reconstruction contribution
        # -----------------------------------------------

        reconstruction_errors = (
            feature_reconstruction_errors(
                sequence,
                reconstruction,
                feature_columns,
            )
        )

        top_reconstruction = (
            reconstruction_errors
            .head(TOP_FEATURES)
        )

        # -----------------------------------------------
        # Find corresponding raw feature rows
        # -----------------------------------------------

        participant_features = (
            features_df[
                features_df["participant_id"]
                == participant_id
            ]
            .sort_values("timestamp")
        )

        rows = participant_features[
            participant_features["timestamp"]
            <= timestamp
        ].tail(
            SEQUENCE_LENGTH
        )

        # -----------------------------------------------
        # Baseline-relative changes
        # -----------------------------------------------

        baseline_changes = (
            calculate_baseline_changes(
                rows[feature_columns],
                participant_id,
                baseline_stats,
            )
        )

        top_changes = (
            baseline_changes
            .sort_values(
                ascending=False
            )
            .head(TOP_FEATURES)
        )

        # -----------------------------------------------
        # Combine explanations
        # -----------------------------------------------

        reconstruction_text = ", ".join(
            [
                f"{feature} "
                f"(error={error:.4f})"
                for feature, error
                in top_reconstruction.items()
            ]
        )

        baseline_text = ", ".join(
            [
                f"{feature} "
                f"({change:.2f} SD)"
                for feature, change
                in top_changes.items()
                if not pd.isna(change)
            ]
        )

        explanation_rows.append(
            {
                "timestamp": timestamp,
                "participant_id": participant_id,
                "subject_id": anomaly["subject_id"],
                "session_type": anomaly["session_type"],
                "protocol_stage": protocol_stage,
                "reconstruction_error":
                    reconstruction_error,
                "anomaly_score":
                    anomaly_score,
                "top_reconstruction_features":
                    reconstruction_text,
                "top_baseline_changes":
                    baseline_text,
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    explanation_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Top Anomaly Explanations")
    print("=" * 70)

    if not explanation_df.empty:

        for _, row in explanation_df.iterrows():

            print(
                f"\nTime: {row['timestamp']}"
            )

            print(
                f"Participant: "
                f"{row['participant_id']}"
            )

            print(
                f"Stage: "
                f"{row['protocol_stage']}"
            )

            print(
                f"Reconstruction error: "
                f"{row['reconstruction_error']:.6f}"
            )

            print(
                f"Anomaly score: "
                f"{row['anomaly_score']:.2f}"
            )

            print(
                "\nTop reconstruction contributors:"
            )

            print(
                row[
                    "top_reconstruction_features"
                ]
            )

            print(
                "\nLargest baseline-relative changes:"
            )

            print(
                row[
                    "top_baseline_changes"
                ]
            )

    print("\n" + "=" * 70)
    print("Explainability Completed")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()