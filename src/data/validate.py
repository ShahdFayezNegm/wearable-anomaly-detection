from pathlib import Path
from datetime import datetime
import csv
import math


# ============================================================
# Expected dataset structure
# ============================================================

EXPECTED_FILES = {
    "ACC.csv",
    "BVP.csv",
    "EDA.csv",
    "HR.csv",
    "IBI.csv",
    "TEMP.csv",
    "tags.csv",
}

EXPECTED_SAMPLING_RATES = {
    "ACC.csv": 32,
    "BVP.csv": 64,
    "EDA.csv": 4,
    "TEMP.csv": 4,
    "HR.csv": 1,
}

# Expected number of values in each data row
EXPECTED_COLUMNS = {
    "ACC.csv": 3,
    "BVP.csv": 1,
    "EDA.csv": 1,
    "HR.csv": 1,
    "IBI.csv": 2,
    "TEMP.csv": 1,
    "tags.csv": 1,
}


# ============================================================
# Helper functions
# ============================================================

def is_numeric(value: str) -> bool:
    """
    Return True if a value can be converted to a finite float.
    """
    try:
        numeric_value = float(value.strip())
        return math.isfinite(numeric_value)
    except (ValueError, TypeError):
        return False


def is_valid_datetime(value: str) -> bool:
    """
    Check whether a string can be parsed as a datetime.

    The wearable dataset uses timestamps such as:
    2013-06-12 16:18:58
    """
    value = value.strip()

    if not value:
        return False

    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]

    for fmt in datetime_formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue

    return False


def looks_like_header(row: list[str]) -> bool:
    """
    Detect a header-like row inside a numeric CSV file.

    Example:
        ['1684532969.0', 'IBI']
    """
    if not row:
        return False

    # A row is considered header-like when at least one
    # value is not numeric.
    return any(not is_numeric(value) for value in row)


# ============================================================
# Validate one session
# ============================================================

def validate_session(session_path: Path) -> dict:
    """
    Validate one wearable session.

    Returns:
        {
            "status": "PASS" | "WARNING" | "FAIL",
            "warnings": [...],
            "errors": [...]
        }
    """

    warnings = []
    errors = []

    # --------------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------------

    actual_files = {
        file.name
        for file in session_path.iterdir()
        if file.is_file()
    }

    missing_files = EXPECTED_FILES - actual_files

    if missing_files:
        errors.append(
            f"Missing files: {sorted(missing_files)}"
        )

    # --------------------------------------------------------
    # 2. Validate each file
    # --------------------------------------------------------

    file_start_times = {}

    for filename in EXPECTED_FILES.intersection(actual_files):

        file_path = session_path / filename

        try:
            with file_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                reader = csv.reader(file)

                rows = list(reader)

        except Exception as exc:

            errors.append(
                f"{filename}: could not be read ({exc})"
            )

            continue

        # ----------------------------------------------------
        # Empty file
        # ----------------------------------------------------

        if not rows:

            # tags.csv can legitimately be empty
            if filename == "tags.csv":

                warnings.append(
                    f"{filename}: file is empty"
                )

            else:

                errors.append(
                    f"{filename}: file is empty"
                )

            continue

        # ----------------------------------------------------
        # 3. Validate timestamp
        # ----------------------------------------------------

        first_row = rows[0]

        if not first_row:

            errors.append(
                f"{filename}: first row is empty"
            )

        else:

            first_timestamp = first_row[0].strip()

            if not is_valid_datetime(first_timestamp):

                errors.append(
                    f"{filename}: invalid start timestamp "
                    f"'{first_timestamp}'"
                )

            else:

                file_start_times[filename] = first_timestamp

        # ----------------------------------------------------
        # tags.csv
        # ----------------------------------------------------

        if filename == "tags.csv":

            for index, row in enumerate(rows, start=1):

                if not row:
                    continue

                timestamp = row[0].strip()

                if not is_valid_datetime(timestamp):

                    warnings.append(
                        f"tags.csv: invalid timestamp at "
                        f"row {index}: {row}"
                    )

            continue

        # ----------------------------------------------------
        # 4. Check metadata / sampling frequency row
        # ----------------------------------------------------

        if filename in EXPECTED_SAMPLING_RATES:

            if len(rows) < 2:

                errors.append(
                    f"{filename}: missing sampling-rate row"
                )

            else:

                metadata_row = rows[1]

                expected_rate = EXPECTED_SAMPLING_RATES[
                    filename
                ]

                for value in metadata_row:

                    if not is_numeric(value):

                        errors.append(
                            f"{filename}: invalid sampling "
                            f"rate value '{value}'"
                        )

                        continue

                    actual_rate = float(value)

                    if actual_rate != expected_rate:

                        errors.append(
                            f"{filename}: expected sampling "
                            f"rate {expected_rate} Hz but "
                            f"found {actual_rate} Hz"
                        )

        # ----------------------------------------------------
        # 5. Validate data rows
        # ----------------------------------------------------

        expected_columns = EXPECTED_COLUMNS[filename]

        # Normal sensor files have metadata in first two rows.
        # IBI also follows this structure.
        data_start_index = 2

        for index, row in enumerate(
            rows[data_start_index:],
            start=data_start_index + 1
        ):

            # Ignore completely empty rows
            if not row or all(
                value.strip() == ""
                for value in row
            ):
                warnings.append(
                    f"{filename}: empty row at row {index}"
                )
                continue

            # ------------------------------------------------
            # Detect header-like row
            # ------------------------------------------------

            if looks_like_header(row):

                warnings.append(
                    f"{filename}: header-like/non-numeric "
                    f"row at row {index}: {row}"
                )

                continue

            # ------------------------------------------------
            # Validate number of columns
            # ------------------------------------------------

            if len(row) != expected_columns:

                warnings.append(
                    f"{filename}: expected "
                    f"{expected_columns} columns but found "
                    f"{len(row)} at row {index}"
                )

                continue

            # ------------------------------------------------
            # Validate numeric values
            # ------------------------------------------------

            for value in row:

                if not is_numeric(value):

                    warnings.append(
                        f"{filename}: non-numeric value "
                        f"at row {index}: '{value}'"
                    )

                    break

        # ----------------------------------------------------
        # 6. ACC specific validation
        # ----------------------------------------------------

        if filename == "ACC.csv":

            for index, row in enumerate(
                rows[2:],
                start=3
            ):

                # Skip empty rows
                if not row:
                    continue

                # ACC must have X, Y, Z
                if len(row) != 3:

                    warnings.append(
                        f"ACC.csv: row {index} should contain "
                        f"3 axis values (X, Y, Z), found "
                        f"{len(row)}"
                    )

                    continue

                for value in row:

                    if not is_numeric(value):

                        warnings.append(
                            f"ACC.csv: non-numeric value "
                            f"at row {index}: '{value}'"
                        )

                        break

    # ========================================================
    # 7. Check timestamp consistency
    # ========================================================
    #
    # Only compare the start times of the main physiological
    # sensor files.
    #
    # tags.csv contains event markers, so its first timestamp
    # is NOT necessarily the recording start time.
    #
    # IBI.csv is an irregular, derived signal and should not
    # be used as the reference for sensor synchronization.
    # ========================================================

    sync_files = {
        "ACC.csv",
        "BVP.csv",
        "EDA.csv",
        "TEMP.csv",
        "HR.csv",
    }

    valid_start_times = []

    for filename in sync_files:

        if filename not in file_start_times:
            continue

        timestamp = file_start_times[filename]

        parsed_timestamp = None

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ):
            try:
                parsed_timestamp = datetime.strptime(
                    timestamp,
                    fmt
                )
                break
            except ValueError:
                continue

        if parsed_timestamp is not None:
            valid_start_times.append(
                (filename, parsed_timestamp)
            )

    if valid_start_times:

        earliest_time = min(
            timestamp
            for _, timestamp in valid_start_times
        )

        for filename, timestamp in valid_start_times:

            difference_seconds = abs(
                (
                    timestamp - earliest_time
                ).total_seconds()
            )

            # A small difference between sensor start times
            # is acceptable.
            if difference_seconds > 120:

                warnings.append(
                    f"{filename}: start timestamp differs "
                    f"by {difference_seconds:.1f} seconds "
                    f"from the earliest main sensor timestamp"
                )
    # ========================================================
    # 8. Determine overall status
    # ========================================================

    if errors:

        status = "FAIL"

    elif warnings:

        status = "WARNING"

    else:

        status = "PASS"

    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
    }


# ============================================================
# Main validation pipeline
# ============================================================

def main() -> None:
    """
    Validate all STRESS sessions.
    """

    project_root = Path(__file__).resolve().parents[2]

    stress_dir = (
        project_root
        / "data"
        / "raw"
        / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
        / "Wearable_Dataset"
        / "STRESS"
    )

    if not stress_dir.exists():

        raise FileNotFoundError(
            f"STRESS directory not found: {stress_dir}"
        )

    sessions = sorted(
        path
        for path in stress_dir.iterdir()
        if path.is_dir()
    )

    print(f"Found {len(sessions)} sessions.")
    print("=" * 70)

    passed = 0
    warnings_count = 0
    failed = 0

    # --------------------------------------------------------
    # Validate every session
    # --------------------------------------------------------

    for session in sessions:

        result = validate_session(session)

        status = result["status"]

        if status == "PASS":

            passed += 1

            print(
                f"[PASS] {session.name}"
            )

        elif status == "WARNING":

            warnings_count += 1

            print(
                f"[WARNING] {session.name}"
            )

            for warning in result["warnings"]:

                print(
                    f"  - WARNING: {warning}"
                )

        else:

            failed += 1

            print(
                f"[FAIL] {session.name}"
            )

            for error in result["errors"]:

                print(
                    f"  - ERROR: {error}"
                )

            for warning in result["warnings"]:

                print(
                    f"  - WARNING: {warning}"
                )

    # ========================================================
    # Final summary
    # ========================================================

    print("=" * 70)

    print("Validation Summary")
    print("-" * 70)

    print(
        f"Total sessions : {len(sessions)}"
    )

    print(
        f"PASS           : {passed}"
    )

    print(
        f"WARNING        : {warnings_count}"
    )

    print(
        f"FAIL           : {failed}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()