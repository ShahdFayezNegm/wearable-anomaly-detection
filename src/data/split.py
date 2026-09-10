from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wearable_features_labeled.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "train_features.csv"
VAL_FILE = OUTPUT_DIR / "val_features.csv"
TEST_FILE = OUTPUT_DIR / "test_features.csv"


# ============================================================
# Split configuration
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Participant ID handling
# ============================================================

def get_participant_id(session_id: str) -> str:
    """
    Convert session IDs into the underlying participant ID.

    Examples
    --------
    S11     -> S11
    S11_a   -> S11
    S11_b   -> S11

    S16     -> S16
    S16_a   -> S16
    S16_b   -> S16

    f14_a   -> f14
    f14_b   -> f14

    This prevents sessions from the same participant from
    being placed in different train/validation/test splits.
    """

    return re.sub(
        r"_[ab]$",
        "",
        str(session_id).strip(),
    )


# ============================================================
# Validate split sizes
# ============================================================

def validate_split_sizes() -> None:
    """
    Ensure train/validation/test proportions sum to 1.
    """

    total = (
        TRAIN_SIZE
        + VAL_SIZE
        + TEST_SIZE
    )

    if not abs(total - 1.0) < 1e-9:
        raise ValueError(
            "TRAIN_SIZE + VAL_SIZE + TEST_SIZE "
            "must equal 1.0"
        )


# ============================================================
# Load labeled features
# ============================================================

def load_features() -> pd.DataFrame:
    """
    Load the protocol-labeled feature dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Labeled feature file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    required_columns = {
        "timestamp",
        "subject_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# Add participant ID
# ============================================================

def add_participant_id(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a participant_id column derived from subject/session ID.
    """

    df = df.copy()

    df["participant_id"] = (
        df["subject_id"]
        .astype(str)
        .apply(get_participant_id)
    )

    return df


# ============================================================
# Split participants
# ============================================================

def split_participants(
    participants: list[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """
    Split participants into train/validation/test groups.
    """

    validate_split_sizes()

    if len(participants) < 3:
        raise ValueError(
            "At least 3 participants are required "
            "for train/validation/test splitting."
        )

    # --------------------------------------------------------
    # First split:
    # TRAIN vs temporary
    # --------------------------------------------------------

    train_participants, temp_participants = (
        train_test_split(
            participants,
            test_size=(
                VAL_SIZE
                + TEST_SIZE
            ),
            random_state=RANDOM_STATE,
        )
    )

    # --------------------------------------------------------
    # Second split:
    # VALIDATION vs TEST
    # --------------------------------------------------------

    relative_test_size = (
        TEST_SIZE
        / (VAL_SIZE + TEST_SIZE)
    )

    val_participants, test_participants = (
        train_test_split(
            temp_participants,
            test_size=relative_test_size,
            random_state=RANDOM_STATE,
        )
    )

    return (
        sorted(train_participants),
        sorted(val_participants),
        sorted(test_participants),
    )


# ============================================================
# Verify no participant leakage
# ============================================================

def verify_no_leakage(
    train_participants: list[str],
    val_participants: list[str],
    test_participants: list[str],
) -> None:
    """
    Verify that no participant appears in more than one split.
    """

    train_set = set(train_participants)
    val_set = set(val_participants)
    test_set = set(test_participants)

    train_val_overlap = (
        train_set
        & val_set
    )

    train_test_overlap = (
        train_set
        & test_set
    )

    val_test_overlap = (
        val_set
        & test_set
    )

    if train_val_overlap:
        raise RuntimeError(
            "Participant leakage between TRAIN and VALIDATION: "
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "Participant leakage between TRAIN and TEST: "
            f"{sorted(train_test_overlap)}"
        )

    if val_test_overlap:
        raise RuntimeError(
            "Participant leakage between VALIDATION and TEST: "
            f"{sorted(val_test_overlap)}"
        )


# ============================================================
# Verify session variants stay together
# ============================================================

def verify_session_variants(
    df: pd.DataFrame,
) -> None:
    """
    Verify that subject/session variants map to exactly one
    participant.

    Example:
        S11_a and S11_b must map to S11.
    """

    mapping = (
        df[
            [
                "participant_id",
                "subject_id",
            ]
        ]
        .drop_duplicates()
    )

    for participant_id, group in mapping.groupby(
        "participant_id"
    ):

        subject_ids = set(
            group["subject_id"]
        )

        expected_ids = {
            participant_id,
        }

        # Every variant must begin with the participant ID.
        for subject_id in subject_ids:

            normalized = get_participant_id(
                subject_id
            )

            if normalized != participant_id:
                raise RuntimeError(
                    f"Invalid participant mapping: "
                    f"{subject_id} -> {participant_id}"
                )


# ============================================================
# Save split
# ============================================================

def save_split(
    df: pd.DataFrame,
    participants: list[str],
    output_file: Path,
) -> None:
    """
    Save all rows belonging to the selected participants.
    """

    split_df = df[
        df["participant_id"].isin(
            participants
        )
    ].copy()

    split_df = (
        split_df
        .sort_values(
            [
                "participant_id",
                "subject_id",
                "session_type",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    split_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(split_df):,} rows → "
        f"{output_file.name}"
    )


# ============================================================
# Print participant/session mapping
# ============================================================

def print_participant_mapping(
    df: pd.DataFrame,
) -> None:
    """
    Print participants that contain multiple session variants.
    """

    mapping = (
        df[
            [
                "participant_id",
                "subject_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "participant_id",
                "subject_id",
            ]
        )
    )

    print("\nParticipant mapping:")
    print("-" * 70)

    found_variants = False

    for participant_id, group in mapping.groupby(
        "participant_id"
    ):

        sessions = (
            group["subject_id"]
            .tolist()
        )

        if len(sessions) > 1:

            found_variants = True

            print(
                f"{participant_id} → {sessions}"
            )

    if not found_variants:
        print(
            "No multi-session participant variants found."
        )


# ============================================================
# Print split summary
# ============================================================

def print_split_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    """
    Print useful statistics for one split.
    """

    participants = (
        df["participant_id"]
        .nunique()
    )

    sessions = (
        df["subject_id"]
        .nunique()
    )

    rows = len(df)

    baseline_windows = int(
        df["is_baseline"]
        .sum()
    )

    print(
        f"\n{name}"
    )

    print("-" * 50)

    print(
        f"Participants     : {participants}"
    )

    print(
        f"Sessions         : {sessions}"
    )

    print(
        f"Rows             : {rows:,}"
    )

    print(
        f"Baseline windows : {baseline_windows:,}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Participant-level dataset split")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_features()

    print(
        f"\nLoaded rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Add participant ID
    # --------------------------------------------------------

    df = add_participant_id(
        df
    )

    # --------------------------------------------------------
    # Verify participant/session mapping
    # --------------------------------------------------------

    print_participant_mapping(
        df
    )

    verify_session_variants(
        df
    )

    # --------------------------------------------------------
    # Get unique participants
    # --------------------------------------------------------

    participants = sorted(
        df["participant_id"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"\nTotal participants: "
        f"{len(participants)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_participants,
        val_participants,
        test_participants,
    ) = split_participants(
        participants
    )

    # --------------------------------------------------------
    # Verify leakage
    # --------------------------------------------------------

    verify_no_leakage(
        train_participants,
        val_participants,
        test_participants,
    )

    # --------------------------------------------------------
    # Print participant lists
    # --------------------------------------------------------

    print(
        f"Train participants: "
        f"{len(train_participants)}"
    )

    print(
        f"Validation participants: "
        f"{len(val_participants)}"
    )

    print(
        f"Test participants: "
        f"{len(test_participants)}"
    )

    print("\nTrain participants:")
    print(train_participants)

    print("\nValidation participants:")
    print(val_participants)

    print("\nTest participants:")
    print(test_participants)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_split(
        df,
        train_participants,
        TRAIN_FILE,
    )

    save_split(
        df,
        val_participants,
        VAL_FILE,
    )

    save_split(
        df,
        test_participants,
        TEST_FILE,
    )

    # --------------------------------------------------------
    # Load saved splits for verification
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_split_summary(
        "TRAIN",
        train_df,
    )

    print_split_summary(
        "VALIDATION",
        val_df,
    )

    print_split_summary(
        "TEST",
        test_df,
    )

    # --------------------------------------------------------
    # Final participant leakage verification
    # --------------------------------------------------------

    verify_no_leakage(
        sorted(
            train_df["participant_id"]
            .unique()
        ),
        sorted(
            val_df["participant_id"]
            .unique()
        ),
        sorted(
            test_df["participant_id"]
            .unique()
        ),
    )

    print("\n" + "=" * 70)
    print(
        "Participant-level split complete."
    )
    print(
        "No participant leakage detected."
    )
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()