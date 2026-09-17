import numpy as np
import pandas as pd

from src.monitoring.drift import detect_drift


def test_no_drift_when_reference_equals_current():
    reference = pd.read_csv(
        "data/processed/train_features.csv"
    )

    current = reference.copy()

    report = detect_drift(
        reference,
        current,
        "train_features.csv",
        "train_features.csv",
    )

    assert report["drifted_feature_count"] == 0
    assert report["overall_drift_rate"] == 0.0
    assert report["drift_detected"] is False


def test_synthetic_drift_is_detected():
    reference = pd.read_csv(
        "data/processed/train_features.csv"
    )

    current = reference.copy()

    drift_features = [
        column
        for column in reference.columns
        if column.startswith("ACC_")
    ]

    assert len(drift_features) >= 9

    for feature in drift_features:
        feature_std = reference[feature].std()

        if not np.isfinite(feature_std):
            continue

        current[feature] = (
            current[feature]
            + feature_std
        )

    report = detect_drift(
        reference,
        current,
        "train_features.csv",
        "synthetic_drift.csv",
    )

    assert report["drifted_feature_count"] >= 9
    assert (
        report["overall_drift_rate"]
        >= 0.20
    )
    assert report["drift_detected"] is True