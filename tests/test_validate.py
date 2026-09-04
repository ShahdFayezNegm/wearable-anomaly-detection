from pathlib import Path

from src.data.validate import validate_session


DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
    / "Wearable_Dataset"
    / "STRESS"
)


def test_valid_session_passes():
    session_path = DATASET_ROOT / "f01"

    result = validate_session(session_path)

    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_empty_tags_produces_warning():
    session_path = DATASET_ROOT / "f14_a"

    result = validate_session(session_path)

    assert result["status"] == "WARNING"

    assert any(
        "tags.csv: file is empty" in warning
        for warning in result["warnings"]
    )


def test_malformed_ibi_produces_warning():
    session_path = DATASET_ROOT / "S02"

    result = validate_session(session_path)

    assert result["status"] == "WARNING"

    assert any(
        "IBI.csv" in warning
        and "header-like/non-numeric" in warning
        for warning in result["warnings"]
    )