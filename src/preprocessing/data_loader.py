from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "solar_output"

REQUIRED_COLUMNS = [
    "timestamp",
    "hour",
    "day_of_year",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "irradiance",
    "solar_output",
]


# Original numerical/weather features
BASE_FEATURES = [
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "irradiance",
]


# Engineered temporal features
TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


# Final features used by the ML models
FEATURE_COLUMNS = (
    BASE_FEATURES
    + TIME_FEATURES
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load the client's solar dataset.

    Parameters
    ----------
    csv_path:
        Path to solar_dataset.csv

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(df):,} rows"
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate that the dataset contains all
    required columns and basic data integrity.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    # Check duplicate timestamps
    duplicate_timestamps = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    if duplicate_timestamps > 0:
        print(
            f"Warning: "
            f"{duplicate_timestamps} "
            "duplicate timestamps found."
        )

    # Check missing values
    missing_values = (
        df[REQUIRED_COLUMNS]
        .isnull()
        .sum()
    )

    missing_total = (
        missing_values.sum()
    )

    if missing_total > 0:

        print(
            "\nMissing values detected:"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )

    else:

        print(
            "No missing values detected."
        )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and chronologically order the dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = (
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps > 0:

        print(
            f"Removing "
            f"{invalid_timestamps} "
            "invalid timestamps."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        "hour",
        "day_of_year",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "irradiance",
        "solar_output",
    ]

    for column in numerical_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"Removing "
            f"{duplicate_count} "
            "duplicate timestamps."
        )

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Handle missing numerical values
    # --------------------------------------------------------

    missing_before = (
        df[numerical_columns]
        .isnull()
        .sum()
        .sum()
    )

    if missing_before > 0:

        print(
            f"Handling {missing_before} "
            "missing numerical values."
        )

        df[numerical_columns] = (
            df[numerical_columns]
            .interpolate(
                method="linear"
            )
            .ffill()
            .bfill()
        )

    # --------------------------------------------------------
    # Solar output cannot be negative
    # --------------------------------------------------------

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .clip(lower=0)
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create cyclic temporal features.

    hour:
        0 -> 23

    day_of_year:
        1 -> 365
    """

    df = df.copy()

    # --------------------------------------------------------
    # Hour encoding
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2
        * np.pi
        * df["hour"]
        / 24.0
    )

    df["hour_cos"] = np.cos(
        2
        * np.pi
        * df["hour"]
        / 24.0
    )

    # --------------------------------------------------------
    # Day-of-year encoding
    # --------------------------------------------------------

    df["day_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_year"]
        / 365.25
    )

    df["day_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_year"]
        / 365.25
    )

    return df


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def prepare_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Complete dataset preparation pipeline.
    """

    print()
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load
    df = load_dataset(
        csv_path
    )

    # Validate
    validate_dataset(
        df
    )

    # Clean
    df = clean_dataset(
        df
    )

    # Feature engineering
    df = create_time_features(
        df
    )

    print(
        f"\nFinal dataset shape: "
        f"{df.shape}"
    )

    print(
        "\nFeatures:"
    )

    for feature in FEATURE_COLUMNS:
        print(
            f"  - {feature}"
        )

    print(
        f"\nTarget: {TARGET_COLUMN}"
    )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split time-series data chronologically.

    70% -> Training
    15% -> Validation
    15% -> Testing

    IMPORTANT:
    No random shuffle is used.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if (
        train_ratio
        + validation_ratio
        >= 1
    ):
        raise ValueError(
            "train_ratio + validation_ratio "
            "must be less than 1."
        )

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n
        * (
            train_ratio
            + validation_ratio
        )
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    print()
    print(
        "Chronological split:"
    )

    print(
        f"Training:   {len(train):,} rows"
    )

    print(
        f"Validation: {len(validation):,} rows"
    )

    print(
        f"Testing:    {len(test):,} rows"
    )

    print()
    print(
        "Time ranges:"
    )

    print(
        f"Training:   "
        f"{train['timestamp'].min()} "
        f"→ "
        f"{train['timestamp'].max()}"
    )

    print(
        f"Validation: "
        f"{validation['timestamp'].min()} "
        f"→ "
        f"{validation['timestamp'].max()}"
    )

    print(
        f"Testing:    "
        f"{test['timestamp'].min()} "
        f"→ "
        f"{test['timestamp'].max()}"
    )

    return (
        train,
        validation,
        test,
    )