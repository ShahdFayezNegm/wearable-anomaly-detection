from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

SOURCE_FILE = DATA_DIR / "train_features.csv"

OUTPUT_FILE = (
    DATA_DIR / "production_batch.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

NUMBER_OF_GROUPS = 4

ACC_SHIFT_STD_MULTIPLIER = 1.5


EXCLUDED_COLUMNS = {
    "timestamp",
    "subject_id",
    "participant_id",
    "session_type",
    "protocol_stage",
    "is_baseline",
}


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Generating Simulated Production Batch")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found:\n{SOURCE_FILE}"
        )

    df = pd.read_csv(
        SOURCE_FILE,
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

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Select baseline rows
    # --------------------------------------------------------

    baseline_df = df[
        df["is_baseline"] == True
    ].copy()

    if baseline_df.empty:
        raise RuntimeError(
            "No baseline rows found."
        )

    # --------------------------------------------------------
    # Select complete participant/session groups
    # --------------------------------------------------------

    group_columns = [
        "participant_id",
        "subject_id",
        "session_type",
    ]

    groups = (
        baseline_df[
            group_columns
        ]
        .drop_duplicates()
        .sample(
            n=min(
                NUMBER_OF_GROUPS,
                len(
                    baseline_df.groupby(
                        group_columns
                    )
                ),
            ),
            random_state=RANDOM_STATE,
        )
    )

    production = baseline_df.merge(
        groups,
        on=group_columns,
        how="inner",
    )

    if production.empty:
        raise RuntimeError(
            "No production rows were selected."
        )

    # --------------------------------------------------------
    # Apply synthetic distribution shift
    # --------------------------------------------------------

    acc_features = [
        column
        for column in production.columns
        if column.startswith("ACC_")
        and column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(
            production[column]
        )
    ]

    if not acc_features:
        raise RuntimeError(
            "No ACC features found."
        )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    print(
        f"\nSelected groups     : "
        f"{len(groups)}"
    )

    print(
        f"Production rows     : "
        f"{len(production):,}"
    )

    print(
        f"Shifted ACC features: "
        f"{len(acc_features)}"
    )

    for feature in acc_features:

        feature_std = production[
            feature
        ].std()

        if not np.isfinite(feature_std):
            continue

        shift = (
            ACC_SHIFT_STD_MULTIPLIER
            * feature_std
        )

        # Small deterministic random component
        noise = rng.normal(
            loc=shift,
            scale=0.05 * feature_std,
            size=len(production),
        )

        production[feature] = (
            production[feature]
            + noise
        )

    # --------------------------------------------------------
    # Sort exactly like normal feature data
    # --------------------------------------------------------

    production = (
        production
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    production.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved production batch to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(production):,}"
    )

    print(
        f"Columns: {len(production.columns)}"
    )

    print("\nShifted features:")

    for feature in acc_features:
        print(f"  - {feature}")

    print("=" * 70)


if __name__ == "__main__":
    main()