import numpy as np
import pandas as pd

from src.monitoring.drift import detect_drift


DRIFT_FEATURES = [
    "ACC_X_mean",
    "ACC_X_std",
    "ACC_X_min",
    "ACC_X_max",
    "ACC_Y_mean",
    "ACC_Y_std",
    "ACC_Y_min",
    "ACC_Y_max",
    "ACC_Z_mean",
    "ACC_Z_std",
    "ACC_Z_min",
    "ACC_Z_max",
    "ACC_MAG_mean",
    "ACC_MAG_std",
    "ACC_MAG_min",
    "ACC_MAG_max",
]

METADATA_COLUMNS = [
    "timestamp",
    "subject_id",
    "participant_id",
    "session_type",
    "protocol_stage",
    "is_baseline",
]


def make_reference_data(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    data = {
        feature: rng.normal(loc=0.0, scale=1.0, size=rows)
        for feature in DRIFT_FEATURES
    }

    data.update(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="s",
            ),
            "subject_id": ["S10"] * rows,
            "participant_id": ["S10"] * rows,
            "session_type": ["AEROBIC"] * rows,
            "protocol_stage": ["PRE_PROTOCOL"] * rows,
            "is_baseline": [True] * rows,
        }
    )

    return pd.DataFrame(data)


def test_no_drift_when_reference_equals_current():
    reference = make_reference_data()
    current = reference.copy()

    report = detect_drift(
        reference,
        current,
        "reference.csv",
        "current.csv",
    )

    assert report["drifted_feature_count"] == 0
    assert report["overall_drift_rate"] == 0.0
    assert report["drift_detected"] is False


def test_synthetic_drift_is_detected():
    reference = make_reference_data()
    current = reference.copy()

    drift_features = [
        column
        for column in reference.columns
        if column.startswith("ACC_")
    ]

    assert len(drift_features) >= 9

    for feature in drift_features:
        feature_std = reference[feature].std()

        if not np.isfinite(feature_std) or feature_std == 0:
            continue

        current[feature] = current[feature] + feature_std * 3

    report = detect_drift(
        reference,
        current,
        "reference.csv",
        "synthetic_drift.csv",
    )

    assert report["drifted_feature_count"] >= 9
    assert report["overall_drift_rate"] >= 0.20
    assert report["drift_detected"] is True