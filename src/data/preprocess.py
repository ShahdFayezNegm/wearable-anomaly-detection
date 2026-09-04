from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
    / "Wearable_Dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ACTIVITY_TYPES = {
    "STRESS",
    "AEROBIC",
    "ANAEROBIC",
}


# ============================================================
# Sensor configuration
# ============================================================

SINGLE_VALUE_SENSORS = {
    "BVP.csv",
    "EDA.csv",
    "HR.csv",
    "TEMP.csv",
}

ACC_FILE = "ACC.csv"
IBI_FILE = "IBI.csv"


# ============================================================
# Helper functions
# ============================================================

def parse_timestamp(value: str) -> pd.Timestamp:
    """
    Parse a timestamp from the wearable dataset.
    """

    timestamp = pd.to_datetime(
        value.strip(),
        errors="coerce"
    )

    if pd.isna(timestamp):
        raise ValueError(
            f"Invalid timestamp: {value}"
        )

    return timestamp


# ============================================================
# Read standard sensor files
# ============================================================

def read_standard_sensor(
    file_path: Path,
    sensor_name: str
) -> pd.DataFrame:
    """
    Read BVP, EDA, HR, or TEMP.

    File structure:

        row 1 -> start timestamp
        row 2 -> sampling frequency
        remaining rows -> sensor values
    """

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(lines) < 3:
        raise ValueError(
            f"{file_path.name} does not contain enough data."
        )

    # --------------------------------------------------------
    # Start timestamp
    # --------------------------------------------------------

    start_timestamp = parse_timestamp(
        lines[0].split(",")[0]
    )

    # --------------------------------------------------------
    # Sampling rate
    # --------------------------------------------------------

    sampling_rate = float(
        lines[1].split(",")[0].strip()
    )

    if sampling_rate <= 0:
        raise ValueError(
            f"{file_path.name}: invalid sampling rate "
            f"{sampling_rate}"
        )

    # --------------------------------------------------------
    # Sensor values
    # --------------------------------------------------------

    values = []

    for line in lines[2:]:

        try:
            value = float(
                line.split(",")[0].strip()
            )

            if np.isfinite(value):
                values.append(value)

        except ValueError:
            # Ignore malformed/header-like rows.
            continue

    if not values:
        raise ValueError(
            f"{file_path.name} contains no valid numeric values."
        )

    # --------------------------------------------------------
    # Create timestamps
    # --------------------------------------------------------

    sample_period = pd.to_timedelta(
        1 / sampling_rate,
        unit="s"
    )

    timestamps = (
        start_timestamp
        + np.arange(len(values)) * sample_period
    )

    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
            sensor_name: values,
        }
    )

    return dataframe


# ============================================================
# Read accelerometer
# ============================================================

def read_acc(file_path: Path) -> pd.DataFrame:
    """
    Read the 3-axis accelerometer.

    File structure:

        row 1 -> start timestamp
        row 2 -> sampling frequency
        remaining rows -> X,Y,Z
    """

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(lines) < 3:
        raise ValueError(
            "ACC.csv does not contain enough data."
        )

    # --------------------------------------------------------
    # Start timestamp
    # --------------------------------------------------------

    timestamp_values = lines[0].split(",")

    start_timestamp = parse_timestamp(
        timestamp_values[0]
    )

    # --------------------------------------------------------
    # Sampling frequency
    # --------------------------------------------------------

    sampling_rate_values = lines[1].split(",")

    sampling_rate = float(
        sampling_rate_values[0].strip()
    )

    if sampling_rate <= 0:
        raise ValueError(
            f"ACC.csv: invalid sampling rate "
            f"{sampling_rate}"
        )

    # --------------------------------------------------------
    # Data rows
    # --------------------------------------------------------

    rows = []

    for line in lines[2:]:

        parts = [
            part.strip()
            for part in line.split(",")
        ]

        if len(parts) != 3:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])

        except ValueError:
            continue

        if not all(
            np.isfinite([x, y, z])
        ):
            continue

        rows.append(
            (x, y, z)
        )

    if not rows:
        raise ValueError(
            "ACC.csv contains no valid numeric rows."
        )

    values = np.asarray(
        rows,
        dtype=float
    )

    # --------------------------------------------------------
    # Create timestamps
    # --------------------------------------------------------

    sample_period = pd.to_timedelta(
        1 / sampling_rate,
        unit="s"
    )

    timestamps = (
        start_timestamp
        + np.arange(len(values)) * sample_period
    )

    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ACC_X": values[:, 0],
            "ACC_Y": values[:, 1],
            "ACC_Z": values[:, 2],
        }
    )

    # --------------------------------------------------------
    # Accelerometer magnitude
    # --------------------------------------------------------

    dataframe["ACC_MAG"] = np.sqrt(
        dataframe["ACC_X"] ** 2
        + dataframe["ACC_Y"] ** 2
        + dataframe["ACC_Z"] ** 2
    )

    return dataframe


# ============================================================
# Read IBI
# ============================================================

def read_ibi(
    file_path: Path,
    session_start: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """
    Read IBI data.

    IMPORTANT:
    The first column of IBI.csv is a time offset in seconds,
    NOT an absolute datetime.

    Example:

        75.609375,0.781250
        76.390625,0.781250

    The first value is the number of seconds from the
    recording start, while the second value is the IBI.

    Malformed rows such as:

        1684532969.0, IBI

    are ignored safely.
    """

    if not file_path.exists():
        return None

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(lines) < 2:
        return None

    rows = []

    # --------------------------------------------------------
    # Skip first metadata row
    # --------------------------------------------------------

    for line in lines[1:]:

        parts = [
            part.strip()
            for part in line.split(",")
        ]

        if len(parts) != 2:
            continue

        # ----------------------------------------------------
        # Parse time offset
        # ----------------------------------------------------

        try:
            time_offset = float(parts[0])
        except ValueError:
            continue

        # ----------------------------------------------------
        # Parse IBI value
        # ----------------------------------------------------

        try:
            ibi_value = float(parts[1])
        except ValueError:
            continue

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not np.isfinite(time_offset):
            continue

        if not np.isfinite(ibi_value):
            continue

        if time_offset < 0:
            continue

        # ----------------------------------------------------
        # Convert relative time into absolute timestamp
        # ----------------------------------------------------

        timestamp = (
            session_start
            + pd.to_timedelta(
                time_offset,
                unit="s"
            )
        )

        rows.append(
            (
                timestamp,
                ibi_value
            )
        )

    if not rows:
        return None

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "IBI"
        ]
    )

    # --------------------------------------------------------
    # Remove duplicated timestamps
    # --------------------------------------------------------

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# Preprocess one session
# ============================================================

def preprocess_session(
    session_path: Path,
    activity_type: str
) -> pd.DataFrame:
    """
    Load and align one wearable session.

    Main wearable signals are converted to a common 1 Hz
    timeline.
    """

    subject_id = session_path.name

    sensor_frames = []

    # --------------------------------------------------------
    # BVP
    # --------------------------------------------------------

    bvp = read_standard_sensor(
        session_path / "BVP.csv",
        "BVP"
    )

    sensor_frames.append(bvp)

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    eda = read_standard_sensor(
        session_path / "EDA.csv",
        "EDA"
    )

    sensor_frames.append(eda)

    # --------------------------------------------------------
    # HR
    # --------------------------------------------------------

    hr = read_standard_sensor(
        session_path / "HR.csv",
        "HR"
    )

    sensor_frames.append(hr)

    # --------------------------------------------------------
    # TEMP
    # --------------------------------------------------------

    temp = read_standard_sensor(
        session_path / "TEMP.csv",
        "TEMP"
    )

    sensor_frames.append(temp)

    # --------------------------------------------------------
    # ACC
    # --------------------------------------------------------

    acc = read_acc(
        session_path / ACC_FILE
    )

    sensor_frames.append(acc)

    # --------------------------------------------------------
    # Resample main sensors to 1 second
    # --------------------------------------------------------

    resampled_frames = []

    for frame in sensor_frames:

        frame = (
            frame
            .set_index("timestamp")
            .sort_index()
            .resample("1s")
            .mean()
        )

        resampled_frames.append(frame)

    # --------------------------------------------------------
    # Merge main sensors
    # --------------------------------------------------------

    merged = resampled_frames[0]

    for frame in resampled_frames[1:]:

        merged = merged.join(
            frame,
            how="outer"
        )

    # --------------------------------------------------------
    # IBI
    # --------------------------------------------------------

    ibi = read_ibi(
        session_path / IBI_FILE,
        merged.index.min()
    )

    if ibi is not None:

        # ----------------------------------------------------
        # Keep IBI only inside the physiological recording
        # period.
        # ----------------------------------------------------

        start_time = merged.index.min()
        end_time = merged.index.max()

        ibi = ibi[
            (ibi["timestamp"] >= start_time)
            & (ibi["timestamp"] <= end_time)
        ]

        if not ibi.empty:

            ibi_resampled = (
                ibi
                .set_index("timestamp")
                .sort_index()
                .resample("1s")
                .mean()
            )

            merged = merged.join(
                ibi_resampled,
                how="left"
            )

    # --------------------------------------------------------
    # Ensure IBI column always exists
    # --------------------------------------------------------

    if "IBI" not in merged.columns:

        merged["IBI"] = np.nan

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    merged = merged.reset_index()

    merged["subject_id"] = subject_id

    merged["session_type"] = activity_type

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    column_order = [
        "timestamp",
        "subject_id",
        "session_type",
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

    merged = merged[
        column_order
    ]

    return merged


# ============================================================
# Process the complete dataset
# ============================================================

def preprocess_dataset() -> None:
    """
    Process STRESS, AEROBIC, and ANAEROBIC sessions.
    """

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_sessions = []

    # --------------------------------------------------------
    # Process every activity type
    # --------------------------------------------------------

    for activity_type in sorted(ACTIVITY_TYPES):

        activity_dir = (
            DATASET_ROOT
            / activity_type
        )

        if not activity_dir.exists():

            print(
                f"[WARNING] {activity_type} directory "
                f"not found. Skipping."
            )

            continue

        sessions = sorted(
            path
            for path in activity_dir.iterdir()
            if path.is_dir()
        )

        print(
            f"\nProcessing {activity_type}: "
            f"{len(sessions)} sessions"
        )

        # ----------------------------------------------------
        # Process sessions
        # ----------------------------------------------------

        for session_path in sessions:

            print(
                f"  → {session_path.name}"
            )

            try:

                dataframe = preprocess_session(
                    session_path,
                    activity_type
                )

                all_sessions.append(
                    dataframe
                )

            except Exception as exc:

                print(
                    f"  [WARNING] Failed to process "
                    f"{session_path.name}: {exc}"
                )

    # --------------------------------------------------------
    # Verify that something was processed
    # --------------------------------------------------------

    if not all_sessions:

        raise RuntimeError(
            "No sessions were successfully processed."
        )

    # --------------------------------------------------------
    # Combine all sessions
    # --------------------------------------------------------

    processed_data = pd.concat(
        all_sessions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort the data
    # --------------------------------------------------------

    processed_data = (
        processed_data
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
    # Save processed dataset
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "wearable_timeseries_1hz.csv"
    )

    processed_data.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "Preprocessing complete."
    )

    print(
        f"Sessions processed: "
        f"{len(all_sessions)}"
    )

    print(
        f"Rows generated: "
        f"{len(processed_data):,}"
    )

    print(
        f"Output: {output_file}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    preprocess_dataset()