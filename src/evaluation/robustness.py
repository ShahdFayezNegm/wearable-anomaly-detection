from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = MODELS_DIR / "evaluation"

VAL_PATH = MODELS_DIR / "val_anomaly_results.csv"
TEST_PATH = MODELS_DIR / "test_anomaly_results.csv"


def analyze_thresholds(df: pd.DataFrame, name: str) -> None:

    print("\n" + "=" * 70)
    print(f"{name} Threshold Sensitivity")
    print("=" * 70)

    errors = df["reconstruction_error"].to_numpy()

    thresholds = {
        "95th_percentile": float(
            np.percentile(errors, 95)
        ),
        "97th_percentile": float(
            np.percentile(errors, 97)
        ),
        "99th_percentile": float(
            np.percentile(errors, 99)
        ),
        "99.5th_percentile": float(
            np.percentile(errors, 99.5)
        ),
    }

    rows = []

    for method, threshold in thresholds.items():

        anomaly_count = int(
            (errors > threshold).sum()
        )

        anomaly_rate = (
            anomaly_count / len(errors)
        )

        rows.append(
            {
                "threshold_method": method,
                "threshold": threshold,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
            }
        )

        print(
            f"{method:22s} | "
            f"threshold={threshold:10.4f} | "
            f"anomalies={anomaly_count:5d} | "
            f"rate={anomaly_rate:.2%}"
        )

    return pd.DataFrame(rows)


def analyze_participants(
    df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print(f"{name} Participant Robustness")
    print("=" * 70)

    participant_stats = (
        df.groupby("participant_id")
        .agg(
            total_sequences=(
                "reconstruction_error",
                "count",
            ),
            anomaly_count=(
                "is_anomaly",
                "sum",
            ),
            mean_error=(
                "reconstruction_error",
                "mean",
            ),
            median_error=(
                "reconstruction_error",
                "median",
            ),
            max_error=(
                "reconstruction_error",
                "max",
            ),
        )
        .reset_index()
    )

    participant_stats["anomaly_rate"] = (
        participant_stats["anomaly_count"]
        / participant_stats["total_sequences"]
    )

    participant_stats = participant_stats.sort_values(
        [
            "anomaly_rate",
            "mean_error",
        ],
        ascending=False,
    )

    print(
        participant_stats.to_string(
            index=False
        )
    )

    return participant_stats


def main():

    print("=" * 70)
    print("Anomaly Detection Robustness Analysis")
    print("=" * 70)

    val_df = pd.read_csv(
        VAL_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    val_thresholds = analyze_thresholds(
        val_df,
        "Validation",
    )

    test_thresholds = analyze_thresholds(
        test_df,
        "Test",
    )

    val_thresholds.to_csv(
        EVALUATION_DIR
        / "validation_threshold_sensitivity.csv",
        index=False,
    )

    test_thresholds.to_csv(
        EVALUATION_DIR
        / "test_threshold_sensitivity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Participant analysis
    # --------------------------------------------------------

    val_participants = analyze_participants(
        val_df,
        "Validation",
    )

    test_participants = analyze_participants(
        test_df,
        "Test",
    )

    val_participants.to_csv(
        EVALUATION_DIR
        / "validation_participant_robustness.csv",
        index=False,
    )

    test_participants.to_csv(
        EVALUATION_DIR
        / "test_participant_robustness.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("Robustness Analysis Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()