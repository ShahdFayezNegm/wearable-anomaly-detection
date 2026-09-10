from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "train_features.csv"
VAL_FILE = DATA_DIR / "val_features.csv"
TEST_FILE = DATA_DIR / "test_features.csv"

PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

TRAIN_SEQUENCES_FILE = MODEL_DIR / "train_sequences.npy"
VAL_BASELINE_SEQUENCES_FILE = (
    MODEL_DIR / "val_baseline_sequences.npy"
)
VAL_EVAL_SEQUENCES_FILE = (
    MODEL_DIR / "val_eval_sequences.npy"
)
TEST_BASELINE_SEQUENCES_FILE = (
    MODEL_DIR / "test_baseline_sequences.npy"
)
TEST_EVAL_SEQUENCES_FILE = (
    MODEL_DIR / "test_eval_sequences.npy"
)


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 12

EXCLUDE_COLUMNS = {
    "timestamp",
    "subject_id",
    "participant_id",
    "session_type",
    "protocol_stage",
    "is_baseline",
}


# ============================================================
# Load split
# ============================================================

def load_split(path: Path) -> pd.DataFrame:
    """
    Load one labeled feature split.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    required_columns = {
        "timestamp",
        "subject_id",
        "participant_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
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
# Feature selection
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Select numerical model features only.
    """

    feature_columns = []

    for column in df.columns:

        if column in EXCLUDE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No numerical feature columns found."
        )

    return feature_columns


# ============================================================
# Participant leakage check
# ============================================================

def validate_participant_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Ensure that no participant appears in more than one split.
    """

    train_ids = set(
        train_df["participant_id"]
    )

    val_ids = set(
        val_df["participant_id"]
    )

    test_ids = set(
        test_df["participant_id"]
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "Participant leakage between TRAIN and VALIDATION."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Participant leakage between TRAIN and TEST."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Participant leakage between VALIDATION and TEST."
        )


# ============================================================
# Missing value handling
# ============================================================

def fit_train_medians(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Learn imputation values from TRAIN only.
    """

    return (
        train_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .median()
    )


def apply_imputation(
    df: pd.DataFrame,
    feature_columns: list[str],
    medians: pd.Series,
) -> pd.DataFrame:
    """
    Apply train-derived median imputation.
    """

    df = df.copy()

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(medians)
    )

    remaining = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if remaining > 0:
        raise ValueError(
            "Missing values remain after imputation."
        )

    return df


# ============================================================
# Scaling
# ============================================================

def fit_scaler(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    """
    Fit StandardScaler using TRAIN BASELINE WINDOWS only.
    """

    baseline_df = train_df[
        train_df["is_baseline"] == True
    ].copy()

    if baseline_df.empty:
        raise ValueError(
            "No TRAIN baseline windows were found."
        )

    scaler = StandardScaler()

    scaler.fit(
        baseline_df[
            feature_columns
        ].to_numpy()
    )

    return scaler


def apply_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply scaler fitted on TRAIN baseline.
    """

    df = df.copy()

    df[feature_columns] = (
        scaler.transform(
            df[feature_columns].to_numpy()
        )
    )

    return df


# ============================================================
# Build sequences
# ============================================================

def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> np.ndarray:
    """
    Build temporal sequences.

    Sequences never cross:
        - participant
        - session
        - activity
        - subject/session boundaries
    """

    sequences = []

    grouped = df.groupby(
        [
            "participant_id",
            "subject_id",
            "session_type",
        ],
        sort=False,
    )

    for _, group in grouped:

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        values = (
            group[feature_columns]
            .to_numpy(
                dtype=np.float32
            )
        )

        if len(values) < sequence_length:
            continue

        for start in range(
            len(values)
            - sequence_length
            + 1
        ):

            end = (
                start
                + sequence_length
            )

            sequence = values[
                start:end
            ]

            sequences.append(
                sequence
            )

    if not sequences:
        raise RuntimeError(
            "No sequences were created."
        )

    return np.stack(
        sequences
    ).astype(np.float32)


# ============================================================
# Build baseline-only sequences
# ============================================================

def build_baseline_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> np.ndarray:
    """
    Build sequences using only windows marked as baseline.

    A sequence is accepted only when all its windows are
    baseline windows.
    """

    baseline_df = df[
        df["is_baseline"] == True
    ].copy()

    return build_sequences(
        baseline_df,
        feature_columns,
        sequence_length,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Preparing LSTM sequences")
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    print(
        "\n[1/8] Loading labeled datasets..."
    )

    train_df = load_split(
        TRAIN_FILE
    )

    val_df = load_split(
        VAL_FILE
    )

    test_df = load_split(
        TEST_FILE
    )

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(val_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # 2. Validate participant split
    # --------------------------------------------------------

    print(
        "\n[2/8] Checking participant leakage..."
    )

    validate_participant_split(
        train_df,
        val_df,
        test_df,
    )

    print(
        "No participant leakage detected."
    )

    # --------------------------------------------------------
    # 3. Feature selection
    # --------------------------------------------------------

    print(
        "\n[3/8] Selecting numerical features..."
    )

    feature_columns = get_feature_columns(
        train_df
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # 4. Fit imputation on TRAIN
    # --------------------------------------------------------

    print(
        "\n[4/8] Fitting imputation on TRAIN..."
    )

    train_medians = fit_train_medians(
        train_df,
        feature_columns,
    )

    train_df = apply_imputation(
        train_df,
        feature_columns,
        train_medians,
    )

    val_df = apply_imputation(
        val_df,
        feature_columns,
        train_medians,
    )

    test_df = apply_imputation(
        test_df,
        feature_columns,
        train_medians,
    )

    # --------------------------------------------------------
    # 5. Fit scaler on TRAIN BASELINE
    # --------------------------------------------------------

    print(
        "\n[5/8] Fitting scaler on TRAIN baseline..."
    )

    scaler = fit_scaler(
        train_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Apply scaler
    # --------------------------------------------------------

    train_df = apply_scaler(
        train_df,
        feature_columns,
        scaler,
    )

    val_df = apply_scaler(
        val_df,
        feature_columns,
        scaler,
    )

    test_df = apply_scaler(
        test_df,
        feature_columns,
        scaler,
    )

    # --------------------------------------------------------
    # 6. Build TRAIN baseline sequences
    # --------------------------------------------------------

    print(
        "\n[6/8] Building TRAIN baseline sequences..."
    )

    train_sequences = build_baseline_sequences(
        train_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # 7. Build validation/test sequences
    # --------------------------------------------------------

    print(
        "\n[7/8] Building validation/test sequences..."
    )

    val_baseline_sequences = build_baseline_sequences(
        val_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    test_baseline_sequences = build_baseline_sequences(
        test_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    # Evaluation sequences are grouped by complete session.
    #
    # They contain baseline + protocol windows.
    # The model will later calculate anomaly scores for
    # these sequences.

    val_eval_sequences = build_sequences(
        val_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    test_eval_sequences = build_sequences(
        test_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # 8. Save artifacts
    # --------------------------------------------------------

    print(
        "\n[8/8] Saving preprocessing artifacts..."
    )

    joblib.dump(
        {
            "scaler": scaler,
            "train_medians": train_medians,
            "feature_columns": feature_columns,
            "sequence_length": SEQUENCE_LENGTH,
            "train_participants": sorted(
                train_df["participant_id"]
                .unique()
            ),
            "validation_participants": sorted(
                val_df["participant_id"]
                .unique()
            ),
            "test_participants": sorted(
                test_df["participant_id"]
                .unique()
            ),
        },
        PREPROCESSOR_FILE,
    )

    np.save(
        TRAIN_SEQUENCES_FILE,
        train_sequences,
    )

    np.save(
        VAL_BASELINE_SEQUENCES_FILE,
        val_baseline_sequences,
    )

    np.save(
        VAL_EVAL_SEQUENCES_FILE,
        val_eval_sequences,
    )

    np.save(
        TEST_BASELINE_SEQUENCES_FILE,
        test_baseline_sequences,
    )

    np.save(
        TEST_EVAL_SEQUENCES_FILE,
        test_eval_sequences,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"TRAIN baseline sequences: "
        f"{train_sequences.shape}"
    )

    print(
        f"VALIDATION baseline sequences: "
        f"{val_baseline_sequences.shape}"
    )

    print(
        f"VALIDATION evaluation sequences: "
        f"{val_eval_sequences.shape}"
    )

    print(
        f"TEST baseline sequences: "
        f"{test_baseline_sequences.shape}"
    )

    print(
        f"TEST evaluation sequences: "
        f"{test_eval_sequences.shape}"
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        "Scaler fitted using TRAIN baseline windows only."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()