from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_RESULTS = MODELS_DIR / "test_anomaly_results.csv"
TEST_FEATURES = PROCESSED_DIR / "test_features.csv"


def main():

    print("=" * 70)
    print("Investigating Test Anomalies")
    print("=" * 70)

    results = pd.read_csv(TEST_RESULTS)
    features = pd.read_csv(TEST_FEATURES)

    results["timestamp"] = pd.to_datetime(results["timestamp"])
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    # --------------------------------------------------------
    # Focus on anomalous participant
    # --------------------------------------------------------

    anomaly_results = results[
        results["is_anomaly"] == True
    ].copy()

    participant_counts = (
        anomaly_results["participant_id"]
        .value_counts()
    )

    print("\nAnomalies by participant:")
    print(participant_counts.to_string())

    target_participant = participant_counts.index[0]

    print(
        f"\nInvestigating participant: "
        f"{target_participant}"
    )

    # --------------------------------------------------------
    # Baseline / non-baseline
    # --------------------------------------------------------

    participant_features = features[
        features["participant_id"] == target_participant
    ].copy()

    baseline = participant_features[
        participant_features["is_baseline"] == True
    ].copy()

    non_baseline = participant_features[
        participant_features["is_baseline"] == False
    ].copy()

    print("\nParticipant data:")
    print(f"Baseline rows     : {len(baseline)}")
    print(f"Non-baseline rows : {len(non_baseline)}")

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    excluded = {
        "timestamp",
        "subject_id",
        "participant_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
    }

    numeric_features = [
        column
        for column in participant_features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            participant_features[column]
        )
    ]

    print(
        f"\nNumeric features: {len(numeric_features)}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")
    print("-" * 70)

    missing = (
        participant_features[numeric_features]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing.to_string())

    # --------------------------------------------------------
    # Compare baseline vs non-baseline
    # --------------------------------------------------------

    comparison = []

    for feature in numeric_features:

        baseline_mean = baseline[feature].mean()
        non_baseline_mean = non_baseline[feature].mean()

        baseline_std = baseline[feature].std()

        if pd.isna(baseline_std) or baseline_std == 0:
            standardized_change = float("nan")
        else:
            standardized_change = (
                non_baseline_mean - baseline_mean
            ) / baseline_std

        comparison.append(
            {
                "feature": feature,
                "baseline_mean": baseline_mean,
                "non_baseline_mean": non_baseline_mean,
                "difference": (
                    non_baseline_mean - baseline_mean
                ),
                "baseline_std": baseline_std,
                "standardized_change": standardized_change,
            }
        )

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        "standardized_change",
        key=lambda x: x.abs(),
        ascending=False,
    )

    print("\nLargest baseline → non-baseline changes:")
    print("-" * 70)

    print(
        comparison_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top anomaly windows
    # --------------------------------------------------------

    top_anomalies = (
        anomaly_results
        .sort_values(
            "reconstruction_error",
            ascending=False,
        )
        .head(10)
    )

    print("\nTop anomaly windows:")
    print("-" * 70)

    columns = [
        "timestamp",
        "protocol_stage",
        "reconstruction_error",
        "anomaly_score",
    ]

    print(
        top_anomalies[columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save investigation
    # --------------------------------------------------------

    output_path = (
        MODELS_DIR
        / f"{target_participant}_feature_changes.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved feature comparison:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 70)
    print("Investigation Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()