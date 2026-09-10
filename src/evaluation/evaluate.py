from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

VAL_RESULTS_PATH = MODELS_DIR / "val_anomaly_results.csv"
TEST_RESULTS_PATH = MODELS_DIR / "test_anomaly_results.csv"

EVALUATION_DIR = MODELS_DIR / "evaluation"


# ============================================================
# Configuration
# ============================================================

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

def load_results(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "timestamp",
        "participant_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
        "reconstruction_error",
        "anomaly_threshold",
        "is_anomaly",
        "anomaly_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["is_anomaly"] = (
        df["is_anomaly"]
        .astype(bool)
    )

    df["is_baseline"] = (
        df["is_baseline"]
        .astype(bool)
    )

    return df


# ============================================================
# Stage statistics
# ============================================================

def build_stage_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df.groupby("protocol_stage")
        .agg(
            total_sequences=(
                "reconstruction_error",
                "count",
            ),
            anomaly_count=(
                "is_anomaly",
                "sum",
            ),
            mean_reconstruction_error=(
                "reconstruction_error",
                "mean",
            ),
            median_reconstruction_error=(
                "reconstruction_error",
                "median",
            ),
            max_reconstruction_error=(
                "reconstruction_error",
                "max",
            ),
            mean_anomaly_score=(
                "anomaly_score",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["anomaly_rate"] = (
        grouped["anomaly_count"]
        / grouped["total_sequences"]
    )

    return grouped.sort_values(
        "anomaly_rate",
        ascending=False,
    )


# ============================================================
# Participant statistics
# ============================================================

def build_participant_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
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
            mean_reconstruction_error=(
                "reconstruction_error",
                "mean",
            ),
            median_reconstruction_error=(
                "reconstruction_error",
                "median",
            ),
            max_reconstruction_error=(
                "reconstruction_error",
                "max",
            ),
        )
        .reset_index()
    )

    grouped["anomaly_rate"] = (
        grouped["anomaly_count"]
        / grouped["total_sequences"]
    )

    return grouped.sort_values(
        "anomaly_rate",
        ascending=False,
    )


# ============================================================
# Baseline statistics
# ============================================================

def build_baseline_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df.groupby("is_baseline")
        .agg(
            total_sequences=(
                "reconstruction_error",
                "count",
            ),
            anomaly_count=(
                "is_anomaly",
                "sum",
            ),
            mean_reconstruction_error=(
                "reconstruction_error",
                "mean",
            ),
            median_reconstruction_error=(
                "reconstruction_error",
                "median",
            ),
            max_reconstruction_error=(
                "reconstruction_error",
                "max",
            ),
        )
        .reset_index()
    )

    grouped["category"] = grouped[
        "is_baseline"
    ].map(
        {
            True: "Baseline",
            False: "Non-baseline",
        }
    )

    grouped["anomaly_rate"] = (
        grouped["anomaly_count"]
        / grouped["total_sequences"]
    )

    return grouped[
        [
            "category",
            "total_sequences",
            "anomaly_count",
            "anomaly_rate",
            "mean_reconstruction_error",
            "median_reconstruction_error",
            "max_reconstruction_error",
        ]
    ]


# ============================================================
# Overall summary
# ============================================================

def build_summary(
    df: pd.DataFrame,
    split_name: str,
) -> dict:

    total = len(df)

    anomalies = int(
        df["is_anomaly"].sum()
    )

    baseline = df[
        df["is_baseline"]
    ]

    non_baseline = df[
        ~df["is_baseline"]
    ]

    summary = {
        "split": split_name,
        "total_sequences": int(total),
        "anomaly_count": anomalies,
        "normal_count": int(total - anomalies),
        "anomaly_rate": (
            float(anomalies / total)
            if total > 0
            else 0.0
        ),
        "mean_reconstruction_error": float(
            df["reconstruction_error"].mean()
        ),
        "median_reconstruction_error": float(
            df["reconstruction_error"].median()
        ),
        "max_reconstruction_error": float(
            df["reconstruction_error"].max()
        ),
        "mean_anomaly_score": float(
            df["anomaly_score"].mean()
        ),
        "baseline": {
            "total_sequences": int(
                len(baseline)
            ),
            "anomaly_count": int(
                baseline["is_anomaly"].sum()
            ),
            "anomaly_rate": (
                float(
                    baseline["is_anomaly"].mean()
                )
                if len(baseline) > 0
                else 0.0
            ),
            "mean_reconstruction_error": (
                float(
                    baseline[
                        "reconstruction_error"
                    ].mean()
                )
                if len(baseline) > 0
                else None
            ),
        },
        "non_baseline": {
            "total_sequences": int(
                len(non_baseline)
            ),
            "anomaly_count": int(
                non_baseline["is_anomaly"].sum()
            ),
            "anomaly_rate": (
                float(
                    non_baseline[
                        "is_anomaly"
                    ].mean()
                )
                if len(non_baseline) > 0
                else 0.0
            ),
            "mean_reconstruction_error": (
                float(
                    non_baseline[
                        "reconstruction_error"
                    ].mean()
                )
                if len(non_baseline) > 0
                else None
            ),
        },
    }

    return summary


# ============================================================
# Plot 1: Reconstruction error distribution
# ============================================================

def plot_reconstruction_distribution(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.hist(
        val_df["reconstruction_error"],
        bins=50,
        alpha=0.6,
        label="Validation",
    )

    plt.hist(
        test_df["reconstruction_error"],
        bins=50,
        alpha=0.6,
        label="Test",
    )

    threshold = float(
        val_df["anomaly_threshold"].iloc[0]
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel(
        "Reconstruction Error"
    )

    plt.ylabel(
        "Number of Sequences"
    )

    plt.title(
        "Reconstruction Error Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        EVALUATION_DIR
        / "reconstruction_error_distribution.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()


# ============================================================
# Plot 2: Anomaly rate by stage
# ============================================================

def plot_anomaly_rate_by_stage(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    val_stage = build_stage_statistics(
        val_df
    )

    test_stage = build_stage_statistics(
        test_df
    )

    all_stages = sorted(
        set(
            val_stage["protocol_stage"]
        ).union(
            test_stage["protocol_stage"]
        )
    )

    val_rates = []

    test_rates = []

    for stage in all_stages:

        val_match = val_stage[
            val_stage["protocol_stage"]
            == stage
        ]

        test_match = test_stage[
            test_stage["protocol_stage"]
            == stage
        ]

        val_rates.append(
            (
                float(
                    val_match[
                        "anomaly_rate"
                    ].iloc[0]
                )
                if not val_match.empty
                else 0.0
            )
        )

        test_rates.append(
            (
                float(
                    test_match[
                        "anomaly_rate"
                    ].iloc[0]
                )
                if not test_match.empty
                else 0.0
            )
        )

    positions = range(
        len(all_stages)
    )

    width = 0.4

    plt.figure(
        figsize=(14, 7)
    )

    val_positions = [
        x - width / 2
        for x in positions
    ]

    test_positions = [
        x + width / 2
        for x in positions
    ]

    plt.bar(
        val_positions,
        val_rates,
        width=width,
        label="Validation",
    )

    plt.bar(
        test_positions,
        test_rates,
        width=width,
        label="Test",
    )

    plt.xticks(
        list(positions),
        all_stages,
        rotation=60,
        ha="right",
    )

    plt.ylabel(
        "Anomaly Rate"
    )

    plt.xlabel(
        "Protocol Stage"
    )

    plt.title(
        "Anomaly Rate by Protocol Stage"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        EVALUATION_DIR
        / "anomaly_rate_by_stage.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()


# ============================================================
# Plot 3: Validation vs Test
# ============================================================

def plot_validation_vs_test(
    val_summary: dict,
    test_summary: dict,
) -> None:

    labels = [
        "Mean Error",
        "Median Error",
        "Anomaly Rate",
    ]

    values_val = [
        val_summary[
            "mean_reconstruction_error"
        ],
        val_summary[
            "median_reconstruction_error"
        ],
        val_summary[
            "anomaly_rate"
        ],
    ]

    values_test = [
        test_summary[
            "mean_reconstruction_error"
        ],
        test_summary[
            "median_reconstruction_error"
        ],
        test_summary[
            "anomaly_rate"
        ],
    ]

    # Put anomaly rate on a percentage-like scale
    values_val[2] *= 100
    values_test[2] *= 100

    positions = range(
        len(labels)
    )

    width = 0.4

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        [
            x - width / 2
            for x in positions
        ],
        values_val,
        width=width,
        label="Validation",
    )

    plt.bar(
        [
            x + width / 2
            for x in positions
        ],
        values_test,
        width=width,
        label="Test",
    )

    plt.xticks(
        list(positions),
        [
            "Mean Error",
            "Median Error",
            "Anomaly Rate (%)",
        ],
    )

    plt.ylabel(
        "Value"
    )

    plt.title(
        "Validation vs Test"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        EVALUATION_DIR
        / "validation_vs_test.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Wearable Anomaly Detection Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    val_df = load_results(
        VAL_RESULTS_PATH
    )

    test_df = load_results(
        TEST_RESULTS_PATH
    )

    # --------------------------------------------------------
    # Build summaries
    # --------------------------------------------------------

    val_summary = build_summary(
        val_df,
        "validation",
    )

    test_summary = build_summary(
        test_df,
        "test",
    )

    # --------------------------------------------------------
    # Stage statistics
    # --------------------------------------------------------

    val_stage = build_stage_statistics(
        val_df
    )

    test_stage = build_stage_statistics(
        test_df
    )

    val_stage.to_csv(
        EVALUATION_DIR
        / "validation_stage_statistics.csv",
        index=False,
    )

    test_stage.to_csv(
        EVALUATION_DIR
        / "test_stage_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Participant statistics
    # --------------------------------------------------------

    val_participants = (
        build_participant_statistics(
            val_df
        )
    )

    test_participants = (
        build_participant_statistics(
            test_df
        )
    )

    val_participants.to_csv(
        EVALUATION_DIR
        / "validation_participant_statistics.csv",
        index=False,
    )

    test_participants.to_csv(
        EVALUATION_DIR
        / "test_participant_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Baseline statistics
    # --------------------------------------------------------

    val_baseline = (
        build_baseline_statistics(
            val_df
        )
    )

    test_baseline = (
        build_baseline_statistics(
            test_df
        )
    )

    val_baseline.to_csv(
        EVALUATION_DIR
        / "validation_baseline_statistics.csv",
        index=False,
    )

    test_baseline.to_csv(
        EVALUATION_DIR
        / "test_baseline_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "validation": val_summary,
        "test": test_summary,
    }

    with open(
        EVALUATION_DIR
        / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_reconstruction_distribution(
        val_df,
        test_df,
    )

    plot_anomaly_rate_by_stage(
        val_df,
        test_df,
    )

    plot_validation_vs_test(
        val_summary,
        test_summary,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    print(
        f"Sequences       : "
        f"{val_summary['total_sequences']}"
    )

    print(
        f"Anomalies       : "
        f"{val_summary['anomaly_count']}"
    )

    print(
        f"Anomaly rate    : "
        f"{val_summary['anomaly_rate']:.2%}"
    )

    print(
        f"Mean error      : "
        f"{val_summary['mean_reconstruction_error']:.6f}"
    )

    print(
        f"Median error    : "
        f"{val_summary['median_reconstruction_error']:.6f}"
    )

    print(
        f"Max error       : "
        f"{val_summary['max_reconstruction_error']:.6f}"
    )

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    print(
        f"Sequences       : "
        f"{test_summary['total_sequences']}"
    )

    print(
        f"Anomalies       : "
        f"{test_summary['anomaly_count']}"
    )

    print(
        f"Anomaly rate    : "
        f"{test_summary['anomaly_rate']:.2%}"
    )

    print(
        f"Mean error      : "
        f"{test_summary['mean_reconstruction_error']:.6f}"
    )

    print(
        f"Median error    : "
        f"{test_summary['median_reconstruction_error']:.6f}"
    )

    print(
        f"Max error       : "
        f"{test_summary['max_reconstruction_error']:.6f}"
    )

    print("\n" + "=" * 70)
    print("Top Validation Stages by Anomaly Rate")
    print("=" * 70)

    print(
        val_stage[
            [
                "protocol_stage",
                "total_sequences",
                "anomaly_count",
                "anomaly_rate",
                "mean_reconstruction_error",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Top Test Stages by Anomaly Rate")
    print("=" * 70)

    print(
        test_stage[
            [
                "protocol_stage",
                "total_sequences",
                "anomaly_count",
                "anomaly_rate",
                "mean_reconstruction_error",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Evaluation Completed")
    print("=" * 70)

    print(
        f"Output directory:\n{EVALUATION_DIR}"
    )


if __name__ == "__main__":
    main()