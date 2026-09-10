from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

VAL_PATH = MODELS_DIR / "val_anomaly_results.csv"
TEST_PATH = MODELS_DIR / "test_anomaly_results.csv"


# ============================================================
# Analysis
# ============================================================

def analyze_file(path: Path, name: str) -> None:

    print("\n" + "=" * 70)
    print(f"{name} Anomaly Analysis")
    print("=" * 70)

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    anomalies = df[df["is_anomaly"] == True].copy()

    print(f"\nTotal sequences      : {len(df)}")
    print(f"Anomalies            : {len(anomalies)}")
    print(
        f"Anomaly rate         : "
        f"{len(anomalies) / len(df):.2%}"
    )

    # --------------------------------------------------------
    # Protocol stage
    # --------------------------------------------------------

    print("\nAnomalies by protocol stage:")
    print("-" * 70)

    stage_counts = (
        anomalies["protocol_stage"]
        .value_counts()
        .to_frame("anomaly_count")
    )

    stage_counts["percentage"] = (
        stage_counts["anomaly_count"]
        / len(anomalies)
        * 100
    )

    print(stage_counts.to_string())

    # --------------------------------------------------------
    # Session type
    # --------------------------------------------------------

    print("\nAnomalies by session type:")
    print("-" * 70)

    session_counts = (
        anomalies["session_type"]
        .value_counts()
        .to_frame("anomaly_count")
    )

    session_counts["percentage"] = (
        session_counts["anomaly_count"]
        / len(anomalies)
        * 100
    )

    print(session_counts.to_string())

    # --------------------------------------------------------
    # Participant
    # --------------------------------------------------------

    print("\nAnomalies by participant:")
    print("-" * 70)

    participant_counts = (
        anomalies["participant_id"]
        .value_counts()
        .to_frame("anomaly_count")
    )

    print(participant_counts.to_string())

    # --------------------------------------------------------
    # Baseline vs non-baseline
    # --------------------------------------------------------

    print("\nBaseline vs non-baseline:")
    print("-" * 70)

    baseline_counts = (
        anomalies["is_baseline"]
        .value_counts()
        .rename(
            index={
                True: "Baseline",
                False: "Non-baseline",
            }
        )
        .to_frame("anomaly_count")
    )

    baseline_counts["percentage"] = (
        baseline_counts["anomaly_count"]
        / len(anomalies)
        * 100
    )

    print(baseline_counts.to_string())

    # --------------------------------------------------------
    # Reconstruction error statistics
    # --------------------------------------------------------

    print("\nAnomaly reconstruction errors:")
    print("-" * 70)

    print(
        anomalies["reconstruction_error"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Top anomalies
    # --------------------------------------------------------

    print("\nTop 20 highest-error anomalies:")
    print("-" * 70)

    columns = [
        "timestamp",
        "participant_id",
        "subject_id",
        "session_type",
        "protocol_stage",
        "reconstruction_error",
        "anomaly_score",
    ]

    top_anomalies = (
        anomalies
        .sort_values(
            "reconstruction_error",
            ascending=False,
        )
        [columns]
        .head(20)
    )

    print(
        top_anomalies.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    analyze_file(
        VAL_PATH,
        "Validation",
    )

    analyze_file(
        TEST_PATH,
        "Test",
    )

    print("\n" + "=" * 70)
    print("Anomaly Analysis Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
    