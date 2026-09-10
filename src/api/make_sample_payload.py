from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "api_sample_request.json"
)

SEQUENCE_LENGTH = 12


def main():

    df = pd.read_csv(
        TEST_FEATURES_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    excluded = {
        "timestamp",
        "subject_id",
        "participant_id",
        "session_type",
        "protocol_stage",
        "is_baseline",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    groups = df.groupby(
        [
            "participant_id",
            "session_type",
            "subject_id",
        ],
        sort=False,
    )

    selected_group = None

    for _, group in groups:

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(group) >= SEQUENCE_LENGTH:

            selected_group = group.head(
                SEQUENCE_LENGTH
            )

            break

    if selected_group is None:
        raise ValueError(
            "Could not find a group with "
            "12 consecutive rows."
        )

    sequence = []

    for _, row in selected_group.iterrows():

        values = {}

        for feature in feature_columns:

            value = row[feature]

            if pd.isna(value):
                values[feature] = None
            else:
                values[feature] = float(
                    value
                )

        sequence.append(values)

    payload = {
        "sequence": sequence,
        "participant_id": str(
            selected_group[
                "participant_id"
            ].iloc[0]
        ),
        "session_type": str(
            selected_group[
                "session_type"
            ].iloc[0]
        ),
        "protocol_stage": str(
            selected_group[
                "protocol_stage"
            ].iloc[-1]
        ),
        "top_k": 5,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            allow_nan=False,
        )

    print("=" * 70)
    print("API Sample Payload Created")
    print("=" * 70)

    print(
        f"Participant : "
        f"{payload['participant_id']}"
    )

    print(
        f"Session     : "
        f"{payload['session_type']}"
    )

    print(
        f"Stage       : "
        f"{payload['protocol_stage']}"
    )

    print(
        f"Shape       : "
        f"({SEQUENCE_LENGTH}, {len(feature_columns)})"
    )

    print(
        f"Output:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()