from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wearable_timeseries_1hz.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "wearable_features_10s.csv"
)


# ============================================================
# Configuration
# ============================================================

WINDOW_SIZE = "10s"

SIGNAL_COLUMNS = [
    "HR",
    "BVP",
    "EDA",
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "ACC_MAG",
    "IBI",
]


# ============================================================
# Safe aggregation
# ============================================================

def safe_std(series: pd.Series) -> float:
    values = series.dropna()

    if len(values) < 2:
        return np.nan

    return float(values.std())


def safe_mean(series: pd.Series) -> float:
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_min(series: pd.Series) -> float:
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return float(values.min())


def safe_max(series: pd.Series) -> float:
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return float(values.max())


# ============================================================
# Load processed time series
# ============================================================

def load_processed_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    required_columns = {
        "timestamp",
        "subject_id",
        "session_type",
        *SIGNAL_COLUMNS,
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# Build window features
# ============================================================

def build_window_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create 10-second features independently for each
    subject/session.

    No personal baseline is calculated here.
    """

    feature_frames = []

    grouped = df.groupby(
        [
            "subject_id",
            "session_type",
        ],
        sort=False,
    )

    for (
        subject_id,
        session_type,
    ), group in grouped:

        group = (
            group
            .sort_values("timestamp")
            .set_index("timestamp")
        )

        window_features = pd.DataFrame()

        for signal in SIGNAL_COLUMNS:

            series = group[signal]

            window_features[
                f"{signal}_mean"
            ] = (
                series
                .resample(WINDOW_SIZE)
                .mean()
            )

            window_features[
                f"{signal}_std"
            ] = (
                series
                .resample(WINDOW_SIZE)
                .agg(safe_std)
            )

            window_features[
                f"{signal}_min"
            ] = (
                series
                .resample(WINDOW_SIZE)
                .agg(safe_min)
            )

            window_features[
                f"{signal}_max"
            ] = (
                series
                .resample(WINDOW_SIZE)
                .agg(safe_max)
            )

        window_features = (
            window_features
            .reset_index()
        )

        window_features["subject_id"] = (
            subject_id
        )

        window_features["session_type"] = (
            session_type
        )

        feature_frames.append(
            window_features
        )

    if not feature_frames:
        raise RuntimeError(
            "No feature windows were created."
        )

    features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    return features


# ============================================================
# Derived features
# ============================================================

def add_derived_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add features derived from the window statistics.

    These features do not use any information from other
    subjects or from validation/test data.
    """

    features = features.copy()

    # --------------------------------------------------------
    # Heart rate range
    # --------------------------------------------------------

    features["HR_range"] = (
        features["HR_max"]
        - features["HR_min"]
    )

    # --------------------------------------------------------
    # EDA range
    # --------------------------------------------------------

    features["EDA_range"] = (
        features["EDA_max"]
        - features["EDA_min"]
    )

    # --------------------------------------------------------
    # Temperature range
    # --------------------------------------------------------

    features["TEMP_range"] = (
        features["TEMP_max"]
        - features["TEMP_min"]
    )

    # --------------------------------------------------------
    # Accelerometer magnitude range
    # --------------------------------------------------------

    features["ACC_MAG_range"] = (
        features["ACC_MAG_max"]
        - features["ACC_MAG_min"]
    )

    # --------------------------------------------------------
    # Activity intensity proxy
    # --------------------------------------------------------

    features["activity_intensity"] = (
        features["ACC_MAG_std"].fillna(0)
        + features["ACC_MAG_range"].fillna(0)
    )

    return features


# ============================================================
# Clean feature dataset
# ============================================================

def clean_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic structural cleaning only.

    No train/validation/test information is used here.
    """

    features = features.copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = (
        features
        .sort_values(
            [
                "subject_id",
                "session_type",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return features


# ============================================================
# Main pipeline
# ============================================================

def build_features() -> None:

    print("=" * 70)
    print("Starting feature engineering")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    print(
        "\n[1/4] Loading processed data..."
    )

    df = load_processed_data()

    print(
        f"Loaded rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # 2. Windows
    # --------------------------------------------------------

    print(
        "\n[2/4] Building 10-second windows..."
    )

    features = build_window_features(
        df
    )

    print(
        f"Windows created: "
        f"{len(features):,}"
    )

    # --------------------------------------------------------
    # 3. Derived features
    # --------------------------------------------------------

    print(
        "\n[3/4] Adding derived features..."
    )

    features = add_derived_features(
        features
    )

    # --------------------------------------------------------
    # 4. Clean
    # --------------------------------------------------------

    print(
        "\n[4/4] Cleaning feature dataset..."
    )

    features = clean_features(
        features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print(
        "Feature engineering complete."
    )
    print(
        f"Rows: {len(features):,}"
    )
    print(
        f"Columns: {len(features.columns)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    build_features()