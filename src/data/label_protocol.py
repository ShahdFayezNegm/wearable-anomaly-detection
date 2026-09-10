from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wearable_features_10s.csv"
)

RAW_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
    / "Wearable_Dataset"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wearable_features_labeled.csv"
)


# ============================================================
# Stress protocol definitions from the dataset notebook
# ============================================================

# For S01-S18:
# tag[3] -> tag[4] = Stroop
# tag[5] -> tag[6] = TMCT
# tag[7] -> tag[8] = Real Opinion
# tag[9] -> tag[10] = Opposite Opinion
# tag[11] -> tag[12] = Subtract
#
# For f01-f18:
# tag[2] -> tag[3] = TMCT
# tag[4] -> tag[5] = Real Opinion
# tag[6] -> tag[7] = Opposite Opinion
# tag[8] -> tag[9] = Subtract
#
# The notebook inserts the recording start time as index 0
# before processing the tags.
# ============================================================

STRESS_V1_INTERVALS = [
    (3, 4, "STROOP"),
    (5, 6, "TMCT"),
    (7, 8, "REAL_OPINION"),
    (9, 10, "OPPOSITE_OPINION"),
    (11, 12, "SUBTRACT"),
]

STRESS_V2_INTERVALS = [
    (2, 3, "TMCT"),
    (4, 5, "REAL_OPINION"),
    (6, 7, "OPPOSITE_OPINION"),
    (8, 9, "SUBTRACT"),
]


# ============================================================
# Helpers
# ============================================================

def get_tags_path(
    session_path: Path,
) -> Path:
    """
    Return the tags.csv path for a session.
    """
    return session_path / "tags.csv"


def read_session_start(
    session_path: Path,
) -> pd.Timestamp | None:
    """
    Read the recording start timestamp from EDA.csv.
    """

    eda_path = session_path / "EDA.csv"

    if not eda_path.exists():
        return None

    try:
        with eda_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            first_line = file.readline().strip()

        if not first_line:
            return None

        timestamp = pd.to_datetime(
            first_line.split(",")[0].strip(),
            errors="coerce",
        )

        if pd.isna(timestamp):
            return None

        return timestamp

    except Exception:
        return None


def read_tags(
    session_path: Path,
) -> list[pd.Timestamp]:
    """
    Read tags.csv as absolute timestamps.

    tags.csv contains one UTC timestamp per row.
    """

    tags_path = get_tags_path(session_path)

    if not tags_path.exists():
        return []

    try:
        tags_df = pd.read_csv(
            tags_path,
            header=None,
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return []

    if tags_df.empty:
        return []

    tags = []

    for value in tags_df.iloc[:, 0]:

        timestamp = pd.to_datetime(
            str(value).strip(),
            errors="coerce",
        )

        if pd.notna(timestamp):
            tags.append(timestamp)

    return tags


def create_extended_tags(
    session_path: Path,
) -> list[pd.Timestamp]:
    """
    Create the tag list used by the original dataset notebook.

    The notebook inserts the session start timestamp at index 0,
    then uses the tags as indices.
    """

    start_time = read_session_start(
        session_path
    )

    raw_tags = read_tags(
        session_path
    )

    if start_time is None:
        return raw_tags

    return [
        start_time,
        *raw_tags,
    ]


def get_stress_intervals(
    session_id: str,
    tags: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Build semantic STRESS intervals using the exact index
    mapping from the dataset notebook.
    """

    intervals = []

    is_first_version = session_id.startswith("S")

    mapping = (
        STRESS_V1_INTERVALS
        if is_first_version
        else STRESS_V2_INTERVALS
    )

    for (
        start_idx,
        end_idx,
        stage_name,
    ) in mapping:

        if (
            start_idx >= len(tags)
            or end_idx >= len(tags)
        ):
            continue

        start_time = tags[start_idx]
        end_time = tags[end_idx]

        if end_time <= start_time:
            continue

        intervals.append(
            (
                start_time,
                end_time,
                stage_name,
            )
        )

    return intervals


def classify_stress_timestamp(
    timestamp: pd.Timestamp,
    session_id: str,
    tags: list[pd.Timestamp],
    session_start: pd.Timestamp | None,
) -> tuple[str, bool]:
    """
    Classify a STRESS timestamp.

    Returns:
        protocol_stage, is_baseline
    """

    # --------------------------------------------------------
    # f14_a is explicitly the baseline recording.
    # --------------------------------------------------------

    if session_id == "f14_a":
        return "BASELINE", True

    # --------------------------------------------------------
    # No timestamps/tags available.
    # --------------------------------------------------------

    if not tags:
        return "UNKNOWN", False

    # --------------------------------------------------------
    # Special handling for f14_b.
    #
    # It contains the remaining protocol after the baseline
    # stored in f14_a.
    # --------------------------------------------------------

    intervals = get_stress_intervals(
        session_id,
        tags,
    )

    # --------------------------------------------------------
    # Explicit baseline before first stress interval.
    # --------------------------------------------------------

    if intervals:

        first_task_start = min(
            start
            for start, _, _ in intervals
        )

        if timestamp < first_task_start:
            return "BASELINE", True

    elif session_start is not None:

        return "UNKNOWN", False

    # --------------------------------------------------------
    # Match a stress task.
    # --------------------------------------------------------

    for (
        start_time,
        end_time,
        stage_name,
    ) in intervals:

        if (
            start_time
            <= timestamp
            < end_time
        ):
            return stage_name, False

    # --------------------------------------------------------
    # Between protocol stages / after tasks.
    # --------------------------------------------------------

    return "REST_OR_OTHER", False


def classify_activity_timestamp(
    timestamp: pd.Timestamp,
    session_id: str,
    tags: list[pd.Timestamp],
    activity_type: str,
) -> tuple[str, bool]:
    """
    Label AEROBIC and ANAEROBIC recordings.

    The notebook text available to us specifies the number
    of tags for these protocols, but the semantic labels of
    every interval are represented in the protocol figures.

    Therefore we use neutral interval names instead of
    inventing physiological stage names.
    """

    if not tags:

        return "UNKNOWN", False

    # --------------------------------------------------------
    # The first element is recording start.
    # --------------------------------------------------------

    boundaries = sorted(tags)

    # --------------------------------------------------------
    # Before the first event marker.
    # --------------------------------------------------------

    if len(boundaries) >= 2:

        first_boundary = boundaries[1]

        if timestamp < first_boundary:
            return "PRE_PROTOCOL", True

    # --------------------------------------------------------
    # Consecutive tag intervals.
    # --------------------------------------------------------

    for index in range(
        1,
        len(boundaries) - 1,
    ):

        start_time = boundaries[index]
        end_time = boundaries[index + 1]

        if (
            start_time
            <= timestamp
            < end_time
        ):

            stage_number = index

            return (
                f"{activity_type}_BLOCK_{stage_number:02d}",
                False,
            )

    # --------------------------------------------------------
    # After the final event.
    # --------------------------------------------------------

    return "POST_PROTOCOL", False


# ============================================================
# Label one session
# ============================================================

def label_session(
    session_df: pd.DataFrame,
    session_path: Path,
) -> pd.DataFrame:
    """
    Add protocol_stage and is_baseline to all windows
    belonging to one session.
    """

    session_df = session_df.copy()

    session_id = session_path.name

    activity_type = session_df[
        "session_type"
    ].iloc[0]

    tags = create_extended_tags(
        session_path
    )

    session_start = read_session_start(
        session_path
    )

    stages = []
    baseline_flags = []

    for timestamp in session_df[
        "timestamp"
    ]:

        timestamp = pd.Timestamp(
            timestamp
        )

        if activity_type == "STRESS":

            stage, is_baseline = (
                classify_stress_timestamp(
                    timestamp,
                    session_id,
                    tags,
                    session_start,
                )
            )

        else:

            stage, is_baseline = (
                classify_activity_timestamp(
                    timestamp,
                    session_id,
                    tags,
                    activity_type,
                )
            )

        stages.append(stage)
        baseline_flags.append(is_baseline)

    session_df[
        "protocol_stage"
    ] = stages

    session_df[
        "is_baseline"
    ] = baseline_flags

    return session_df


# ============================================================
# Main labeling pipeline
# ============================================================

def label_dataset() -> None:

    print("=" * 70)
    print("Adding protocol labels")
    print("=" * 70)

    # --------------------------------------------------------
    # Load engineered features
    # --------------------------------------------------------

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}"
        )

    features = pd.read_csv(
        FEATURES_FILE,
        parse_dates=["timestamp"],
    )

    required_columns = {
        "timestamp",
        "subject_id",
        "session_type",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    labeled_frames = []

    # --------------------------------------------------------
    # Process every session
    # --------------------------------------------------------

    grouped = features.groupby(
        [
            "subject_id",
            "session_type",
        ],
        sort=False,
    )

    total_sessions = 0

    for (
        subject_id,
        session_type,
    ), session_df in grouped:

        session_path = (
            RAW_DATASET
            / session_type
            / subject_id
        )

        if not session_path.exists():

            print(
                f"[WARNING] Raw session not found: "
                f"{session_path}"
            )

            session_df = session_df.copy()

            session_df[
                "protocol_stage"
            ] = "UNKNOWN"

            session_df[
                "is_baseline"
            ] = False

        else:

            session_df = label_session(
                session_df,
                session_path,
            )

        labeled_frames.append(
            session_df
        )

        total_sessions += 1

    if not labeled_frames:
        raise RuntimeError(
            "No sessions were labeled."
        )

    labeled_data = pd.concat(
        labeled_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    labeled_data = (
        labeled_data
        .sort_values(
            [
                "subject_id",
                "session_type",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"Sessions labeled: "
        f"{total_sessions}"
    )

    print(
        f"Rows: "
        f"{len(labeled_data):,}"
    )

    print(
        f"Baseline windows: "
        f"{int(labeled_data['is_baseline'].sum()):,}"
    )

    print("\nProtocol stages:")

    print(
        labeled_data[
            "protocol_stage"
        ]
        .value_counts()
        .head(30)
    )

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    label_dataset()