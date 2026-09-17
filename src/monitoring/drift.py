from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


DEFAULT_REFERENCE_FILE = DATA_DIR / "train_features.csv"
DEFAULT_CURRENT_FILE = DATA_DIR / "val_features.csv"
DEFAULT_REPORT_FILE = MODELS_DIR / "drift_report.json"


# ============================================================
# Configuration
# ============================================================

# Feature-level drift threshold
FEATURE_DRIFT_THRESHOLD = 0.20

# Dataset-level drift threshold
# Example:
# 0.20 means at least 20% of valid features must drift
# before declaring overall dataset drift.
OVERALL_DRIFT_RATE_THRESHOLD = 0.20

MIN_VALID_VALUES = 20


EXCLUDED_COLUMNS = {
    "timestamp",
    "subject_id",
    "participant_id",
    "session_type",
    "protocol_stage",
    "is_baseline",
}


# ============================================================
# Load data
# ============================================================

def load_data(
    reference_file: Path,
    current_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file not found: {reference_file}"
        )

    if not current_file.exists():
        raise FileNotFoundError(
            f"Current file not found: {current_file}"
        )

    reference = pd.read_csv(reference_file)
    current = pd.read_csv(current_file)

    return reference, current


# ============================================================
# Select numeric model features
# ============================================================

def get_feature_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[str]:

    common_columns = sorted(
        set(reference.columns)
        & set(current.columns)
    )

    feature_columns = []

    for column in common_columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if (
            pd.api.types.is_numeric_dtype(reference[column])
            and pd.api.types.is_numeric_dtype(current[column])
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError(
            "No numeric feature columns were found."
        )

    return feature_columns


# ============================================================
# Calculate normalized Wasserstein distance
# ============================================================

def calculate_drift_score(
    reference_values: pd.Series,
    current_values: pd.Series,
) -> float:

    reference = (
        pd.to_numeric(
            reference_values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    current = (
        pd.to_numeric(
            current_values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < MIN_VALID_VALUES:
        return float("nan")

    if len(current) < MIN_VALID_VALUES:
        return float("nan")

    distance = wasserstein_distance(
        reference,
        current,
    )

    reference_scale = np.std(reference)

    if reference_scale <= 1e-12:
        reference_scale = (
            np.abs(np.mean(reference))
            + 1e-12
        )

    normalized_distance = (
        distance / reference_scale
    )

    return float(normalized_distance)


# ============================================================
# Build drift report
# ============================================================

def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_file: Path,
    current_file: Path,
) -> dict:

    feature_columns = get_feature_columns(
        reference,
        current,
    )

    results = []

    for feature in feature_columns:

        score = calculate_drift_score(
            reference[feature],
            current[feature],
        )

        if np.isnan(score):
            drifted = False
        else:
            drifted = (
                score >= FEATURE_DRIFT_THRESHOLD
            )

        results.append(
            {
                "feature": feature,
                "drift_score": (
                    None
                    if np.isnan(score)
                    else round(score, 6)
                ),
                "drift_detected": bool(drifted),
            }
        )

    valid_results = [
        item
        for item in results
        if item["drift_score"] is not None
    ]

    drifted_features = [
        item["feature"]
        for item in valid_results
        if item["drift_detected"]
    ]

    overall_drift_rate = (
        len(drifted_features)
        / len(valid_results)
        if valid_results
        else 0.0
    )

    # Dataset-level decision
    overall_drift_detected = (
        overall_drift_rate
        >= OVERALL_DRIFT_RATE_THRESHOLD
    )

    report = {
        "reference_file": str(
            reference_file
        ),
        "current_file": str(
            current_file
        ),
        "reference_rows": int(
            len(reference)
        ),
        "current_rows": int(
            len(current)
        ),
        "features_checked": int(
            len(feature_columns)
        ),
        "valid_features": int(
            len(valid_results)
        ),
        "feature_drift_threshold": (
            FEATURE_DRIFT_THRESHOLD
        ),
        "overall_drift_rate_threshold": (
            OVERALL_DRIFT_RATE_THRESHOLD
        ),
        "drifted_feature_count": int(
            len(drifted_features)
        ),
        "overall_drift_rate": round(
            float(overall_drift_rate),
            6,
        ),
        "drift_detected": bool(
            overall_drift_detected
        ),
        "drifted_features": drifted_features,
        "feature_results": results,
    }

    return report


# ============================================================
# Save report
# ============================================================

def save_report(
    report: dict,
    output_file: Path,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


# ============================================================
# CLI arguments
# ============================================================

def parse_args():

    parser = ArgumentParser(
        description="Detect feature distribution drift."
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help=(
            "Reference dataset used as baseline."
        ),
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT_FILE,
        help=(
            "Current dataset to compare against reference."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help=(
            "Output JSON report path."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    print("=" * 70)
    print("Wearable Drift Detection")
    print("=" * 70)

    print(
        f"\nReference : {args.reference}"
    )

    print(
        f"Current   : {args.current}"
    )

    reference, current = load_data(
        args.reference,
        args.current,
    )

    print(
        f"\nReference rows: {len(reference):,}"
    )

    print(
        f"Current rows  : {len(current):,}"
    )

    report = detect_drift(
        reference,
        current,
        args.reference,
        args.current,
    )

    save_report(
        report,
        args.output,
    )

    print(
        f"\nFeatures checked          : "
        f"{report['features_checked']}"
    )

    print(
        f"Drifted features         : "
        f"{report['drifted_feature_count']}"
    )

    print(
        f"Overall drift rate       : "
        f"{report['overall_drift_rate']:.2%}"
    )

    print(
        f"Feature threshold        : "
        f"{report['feature_drift_threshold']}"
    )

    print(
        f"Overall rate threshold   : "
        f"{report['overall_drift_rate_threshold']:.2%}"
    )

    print(
        f"Overall drift detected   : "
        f"{report['drift_detected']}"
    )

    print(
        f"\nReport saved to: "
        f"{args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()