from pathlib import Path
import json

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = MODELS_DIR / "evaluation"

THRESHOLD_PATH = MODELS_DIR / "threshold.json"
HISTORY_PATH = MODELS_DIR / "training_history.csv"

VAL_RESULTS_PATH = MODELS_DIR / "val_anomaly_results.csv"
TEST_RESULTS_PATH = MODELS_DIR / "test_anomaly_results.csv"

VAL_STAGE_PATH = (
    EVALUATION_DIR / "validation_stage_statistics.csv"
)

TEST_STAGE_PATH = (
    EVALUATION_DIR / "test_stage_statistics.csv"
)

VAL_PARTICIPANT_PATH = (
    EVALUATION_DIR / "validation_participant_statistics.csv"
)

TEST_PARTICIPANT_PATH = (
    EVALUATION_DIR / "test_participant_statistics.csv"
)

OUTPUT_PATH = (
    EVALUATION_DIR / "model_report.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def summarize_results(
    path: Path,
) -> dict:

    df = pd.read_csv(path)

    return {
        "total_sequences": int(len(df)),
        "anomaly_count": int(
            df["is_anomaly"].sum()
        ),
        "anomaly_rate": float(
            df["is_anomaly"].mean()
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
        "baseline_anomalies": int(
            df.loc[
                df["is_baseline"],
                "is_anomaly",
            ].sum()
        ),
        "non_baseline_anomalies": int(
            df.loc[
                ~df["is_baseline"],
                "is_anomaly",
            ].sum()
        ),
    }


def top_stages(path: Path) -> list:

    df = pd.read_csv(path)

    df = df.sort_values(
        [
            "anomaly_rate",
            "mean_reconstruction_error",
        ],
        ascending=False,
    )

    columns = [
        "protocol_stage",
        "total_sequences",
        "anomaly_count",
        "anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "max_reconstruction_error",
    ]

    return (
        df[columns]
        .head(10)
        .to_dict(orient="records")
    )


def top_participants(path: Path) -> list:

    df = pd.read_csv(path)

    df = df.sort_values(
        [
            "anomaly_rate",
            "mean_reconstruction_error",
        ],
        ascending=False,
    )

    columns = [
        "participant_id",
        "total_sequences",
        "anomaly_count",
        "anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "max_reconstruction_error",
    ]

    return (
        df[columns]
        .head(10)
        .to_dict(orient="records")
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Building Final Model Report")
    print("=" * 70)

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_json(
        THRESHOLD_PATH
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = pd.read_csv(
        HISTORY_PATH
    )

    best_row = history.loc[
        history["val_loss"].idxmin()
    ]

    training_summary = {
        "epochs_trained": int(
            len(history)
        ),
        "best_epoch": int(
            best_row["epoch"]
        ),
        "best_train_loss": float(
            best_row["train_loss"]
        ),
        "best_validation_loss": float(
            best_row["val_loss"]
        ),
    }

    # --------------------------------------------------------
    # Validation / Test
    # --------------------------------------------------------

    validation_summary = summarize_results(
        VAL_RESULTS_PATH
    )

    test_summary = summarize_results(
        TEST_RESULTS_PATH
    )

    # --------------------------------------------------------
    # Stage statistics
    # --------------------------------------------------------

    validation_stages = top_stages(
        VAL_STAGE_PATH
    )

    test_stages = top_stages(
        TEST_STAGE_PATH
    )

    # --------------------------------------------------------
    # Participant statistics
    # --------------------------------------------------------

    validation_participants = top_participants(
        VAL_PARTICIPANT_PATH
    )

    test_participants = top_participants(
        TEST_PARTICIPANT_PATH
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "project": {
            "name": "Wearable Personalized Anomaly Detection",
            "task": "Unsupervised wearable time-series anomaly detection",
        },

        "model": {
            "architecture": "LSTM Autoencoder",
            "input_features": 41,
            "sequence_length": 12,
            "hidden_dimension": 64,
            "latent_dimension": 32,
            "num_layers": 2,
            "dropout": 0.2,
            "parameters": 134089,
        },

        "threshold": threshold,

        "training": training_summary,

        "validation": {
            "summary": validation_summary,
            "top_stages": validation_stages,
            "top_participants": validation_participants,
        },

        "test": {
            "summary": test_summary,
            "top_stages": test_stages,
            "top_participants": test_participants,
        },

        "interpretation": {
            "validation_anomaly_rate": (
                validation_summary["anomaly_rate"]
            ),
            "test_anomaly_rate": (
                test_summary["anomaly_rate"]
            ),
            "note": (
                "The model performs unsupervised anomaly detection. "
                "An anomaly indicates deviation from learned baseline "
                "behavior and is not a medical diagnosis."
            ),
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print("\nModel:")
    print(
        f"Architecture: "
        f"{report['model']['architecture']}"
    )

    print(
        f"Parameters: "
        f"{report['model']['parameters']:,}"
    )

    print(
        f"Sequence length: "
        f"{report['model']['sequence_length']}"
    )

    print(
        f"Features: "
        f"{report['model']['input_features']}"
    )

    print("\nTraining:")
    print(
        f"Epochs trained: "
        f"{training_summary['epochs_trained']}"
    )

    print(
        f"Best epoch: "
        f"{training_summary['best_epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{training_summary['best_validation_loss']:.6f}"
    )

    print("\nThreshold:")
    print(
        f"{threshold['threshold']:.6f}"
    )

    print("\nValidation:")
    print(
        f"Anomalies: "
        f"{validation_summary['anomaly_count']}/"
        f"{validation_summary['total_sequences']}"
    )

    print(
        f"Anomaly rate: "
        f"{validation_summary['anomaly_rate']:.2%}"
    )

    print("\nTest:")
    print(
        f"Anomalies: "
        f"{test_summary['anomaly_count']}/"
        f"{test_summary['total_sequences']}"
    )

    print(
        f"Anomaly rate: "
        f"{test_summary['anomaly_rate']:.2%}"
    )

    print("\nTop test participants:")

    for participant in test_participants:

        print(
            f"  {participant['participant_id']}: "
            f"{participant['anomaly_count']} anomalies "
            f"({participant['anomaly_rate']:.2%})"
        )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("Model Report Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()