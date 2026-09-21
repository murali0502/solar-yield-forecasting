"""
XGBoost training pipeline for Solar Yield Forecasting.

Architecture:
    XGBoost + TCN + Hybrid Ensemble

XGBoost role:
    Learn nonlinear relationships between weather/time features
    and the next-hour solar output.

Input:
    Features at time t

Target:
    Solar output at time t+1

Prediction type:
    One-step-ahead / next-hour forecasting

Project structure:
    src/xgboost/train.py
    models/xgboost/
    results/xgboost/
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from xgboost import XGBRegressor


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

from src.preprocessing.data_loader import (
    prepare_dataset,
    chronological_split,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "solar_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "xgboost"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_ESTIMATORS = 800

LEARNING_RATE = 0.03

MAX_DEPTH = 7

MIN_CHILD_WEIGHT = 3

SUBSAMPLE = 0.85

COLSAMPLE_BYTREE = 0.90

REG_ALPHA = 0.01

REG_LAMBDA = 1.0


# ============================================================
# CREATE XGBOOST DATA
# ============================================================

def create_xgboost_data(
    df: pd.DataFrame,
):
    """
    Create one-step-ahead forecasting data.

    Features at time t
            ↓
        XGBoost
            ↓
    Solar output at time t+1

    The timestamp stored with each target is the timestamp
    of the value being predicted.

    Example:

        features at 10:00
                ↓
        prediction for 11:00

        target_timestamp = 11:00
    """

    data = df.copy()

    # --------------------------------------------------------
    # Validate timestamp
    # --------------------------------------------------------

    if "timestamp" not in data.columns:

        raise ValueError(
            "Dataset must contain a 'timestamp' column."
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found in dataset."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = (
        data
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Next-hour target timestamp
    # --------------------------------------------------------

    data["target_timestamp"] = (
        data["timestamp"].shift(-1)
    )

    # --------------------------------------------------------
    # Next-hour solar output
    # --------------------------------------------------------

    data["target_next_hour"] = (
        data[TARGET_COLUMN].shift(-1)
    )

    # --------------------------------------------------------
    # Remove final row because it has no next-hour target
    # --------------------------------------------------------

    data = (
        data
        .dropna(
            subset=[
                "target_timestamp",
                "target_next_hour",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = (
        data[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = (
        data[
            "target_next_hour"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Timestamp belonging to target
    # --------------------------------------------------------

    timestamps = (
        data[
            "target_timestamp"
        ]
        .copy()
    )

    return (
        X,
        y,
        timestamps,
    )


# ============================================================
# BUILD XGBOOST MODEL
# ============================================================

def build_model():

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=N_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        max_depth=MAX_DEPTH,

        min_child_weight=MIN_CHILD_WEIGHT,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        reg_alpha=REG_ALPHA,

        reg_lambda=REG_LAMBDA,

        random_state=RANDOM_SEED,

        n_jobs=-1,
    )

    return model


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    timestamps,
    actual,
    prediction,
    output_path,
):
    """
    Save timestamp-aligned predictions.

    Timestamp alignment is required by the hybrid ensemble
    so that XGBoost and TCN predictions can be matched
    to the same forecast hour.
    """

    result = pd.DataFrame(
        {
            "timestamp": (
                pd.to_datetime(
                    timestamps
                ).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),

            "actual": (
                np.asarray(
                    actual,
                    dtype=np.float32,
                )
            ),

            "prediction": (
                np.asarray(
                    prediction,
                    dtype=np.float32,
                )
            ),
        }
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print()

    print(
        "=" * 70
    )

    print(
        "XGBOOST TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Prediction type:"
    )

    print(
        "One-step-ahead / next-hour forecasting"
    )

    print()

    print(
        f"Random seed      : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Estimators       : "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Max depth        : "
        f"{MAX_DEPTH}"
    )

    print()


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "LOADING DATASET"
    )

    print(
        "=" * 70
    )

    df = prepare_dataset(
        DATA_PATH
    )

    print()

    print(
        f"Dataset shape: "
        f"{df.shape}"
    )

    print()

    print(
        "Features:"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"  - {feature}"
        )

    print()

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print()


    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    if "timestamp" not in df.columns:

        raise ValueError(
            "Prepared dataset must contain "
            "'timestamp'."
        )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required XGBoost features: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in dataset."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Dataset contains invalid timestamps."
        )

    # --------------------------------------------------------
    # Ensure chronological order
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Dataset validation: OK"
    )

    print()

    print(
        f"First timestamp: "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp : "
        f"{df['timestamp'].iloc[-1]}"
    )

    print()


    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "CHRONOLOGICAL DATA SPLIT"
    )

    print(
        "=" * 70
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(
        df
    )

    print()

    print(
        f"Training rows   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows : "
        f"{len(validation_df)}"
    )

    print(
        f"Test rows       : "
        f"{len(test_df)}"
    )

    print()


    # ========================================================
    # CREATE TRAINING DATA
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "CREATING XGBOOST FORECASTING DATA"
    )

    print(
        "=" * 70
    )

    (
        X_train,
        y_train,
        train_timestamps,
    ) = create_xgboost_data(
        train_df
    )

    (
        X_validation,
        y_validation,
        validation_timestamps,
    ) = create_xgboost_data(
        validation_df
    )

    (
        X_test,
        y_test,
        test_timestamps,
    ) = create_xgboost_data(
        test_df
    )

    print()

    print(
        f"X_train: "
        f"{X_train.shape}"
    )

    print(
        f"y_train: "
        f"{y_train.shape}"
    )

    print(
        f"X_validation: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation: "
        f"{y_validation.shape}"
    )

    print(
        f"X_test: "
        f"{X_test.shape}"
    )

    print(
        f"y_test: "
        f"{y_test.shape}"
    )

    print()


    # ========================================================
    # VERIFY TARGET TIMESTAMPS
    # ========================================================

    print(
        "Target timestamp ranges:"
    )

    print()

    print(
        f"Training:"
    )

    print(
        f"  {train_timestamps.iloc[0]}"
        f" -> "
        f"{train_timestamps.iloc[-1]}"
    )

    print()

    print(
        f"Validation:"
    )

    print(
        f"  {validation_timestamps.iloc[0]}"
        f" -> "
        f"{validation_timestamps.iloc[-1]}"
    )

    print()

    print(
        f"Testing:"
    )

    print(
        f"  {test_timestamps.iloc[0]}"
        f" -> "
        f"{test_timestamps.iloc[-1]}"
    )

    print()


    # ========================================================
    # VERIFY FEATURE COUNT
    # ========================================================

    if X_train.shape[1] != len(
        FEATURE_COLUMNS
    ):

        raise ValueError(
            "XGBoost feature count mismatch. "
            f"Expected {len(FEATURE_COLUMNS)}, "
            f"got {X_train.shape[1]}."
        )

    print(
        "XGBoost feature dimension: OK"
    )

    print()


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "BUILDING XGBOOST MODEL"
    )

    print(
        "=" * 70
    )

    model = build_model()

    print()

    print(
        "Model configuration:"
    )

    print(
        f"  n_estimators     = {N_ESTIMATORS}"
    )

    print(
        f"  learning_rate    = {LEARNING_RATE}"
    )

    print(
        f"  max_depth        = {MAX_DEPTH}"
    )

    print(
        f"  min_child_weight = {MIN_CHILD_WEIGHT}"
    )

    print(
        f"  subsample        = {SUBSAMPLE}"
    )

    print(
        f"  colsample        = {COLSAMPLE_BYTREE}"
    )

    print()


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "STARTING XGBOOST TRAINING"
    )

    print(
        "=" * 70
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],

        verbose=False,
    )

    print()

    print(
        "XGBoost training completed."
    )

    print()


    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "GENERATING PREDICTIONS"
    )

    print(
        "=" * 70
    )

    train_predictions = (
        model.predict(
            X_train
        )
    )

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )


    # ========================================================
    # CLEAN PREDICTIONS
    # ========================================================

    # Solar production cannot be negative.

    train_predictions = np.clip(
        train_predictions,
        0,
        None,
    )

    validation_predictions = np.clip(
        validation_predictions,
        0,
        None,
    )

    test_predictions = np.clip(
        test_predictions,
        0,
        None,
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING XGBOOST MODEL"
    )

    print(
        "=" * 70
    )

    model_path = (
        MODEL_DIR
        / "model.json"
    )

    model.save_model(
        model_path
    )

    print()

    print(
        "Model saved:"
    )

    print(
        model_path
    )

    print()


    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    validation_path = (
        RESULTS_DIR
        / "validation_predictions.csv"
    )

    save_predictions(
        timestamps=validation_timestamps,

        actual=y_validation.values,

        prediction=validation_predictions,

        output_path=validation_path,
    )

    print(
        "Validation predictions saved:"
    )

    print(
        validation_path
    )

    print()


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_path = (
        RESULTS_DIR
        / "test_predictions.csv"
    )

    save_predictions(
        timestamps=test_timestamps,

        actual=y_test.values,

        prediction=test_predictions,

        output_path=test_path,
    )

    print(
        "Test predictions saved:"
    )

    print(
        test_path
    )

    print()


    # ========================================================
    # SAVE FEATURE IMPORTANCE
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    feature_importance = pd.DataFrame(
        {
            "feature":
                FEATURE_COLUMNS,

            "importance":
                model.feature_importances_,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance_path = (
        RESULTS_DIR
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False,
    )

    print()

    print(
        "Feature importance saved:"
    )

    print(
        feature_importance_path
    )

    print()


    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING XGBOOST CONFIGURATION"
    )

    print(
        "=" * 70
    )

    config = {

        "model_type":
            "XGBoost",

        "prediction_type":
            "sequence_to_one",

        "target_definition":
            "next_hour",

        "input_description":
            "features_at_time_t",

        "output_description":
            "solar_output_at_time_t_plus_1",

        "features":
            list(FEATURE_COLUMNS),

        "number_of_features":
            len(FEATURE_COLUMNS),

        "target":
            TARGET_COLUMN,

        "n_estimators":
            N_ESTIMATORS,

        "learning_rate":
            LEARNING_RATE,

        "max_depth":
            MAX_DEPTH,

        "min_child_weight":
            MIN_CHILD_WEIGHT,

        "subsample":
            SUBSAMPLE,

        "colsample_bytree":
            COLSAMPLE_BYTREE,

        "reg_alpha":
            REG_ALPHA,

        "reg_lambda":
            REG_LAMBDA,

        "random_seed":
            RANDOM_SEED,

        "timestamp_alignment":
            "target_timestamp",

        "architecture":
            "XGBoost + TCN + Hybrid",

        "lstm_used":
            False,
    }

    config_path = (
        MODEL_DIR
        / "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    print()

    print(
        "Configuration saved:"
    )

    print(
        config_path
    )

    print()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "XGBOOST TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Model:"
    )

    print(
        model_path
    )

    print()

    print(
        "Input:"
    )

    print(
        "Features at time t"
    )

    print()

    print(
        "Output:"
    )

    print(
        "Solar output at time t+1"
    )

    print()

    print(
        "Validation predictions:"
    )

    print(
        validation_path
    )

    print()

    print(
        "Test predictions:"
    )

    print(
        test_path
    )

    print()

    print(
        "Feature importance:"
    )

    print(
        feature_importance_path
    )

    print()

    print(
        "Configuration:"
    )

    print(
        config_path
    )

    print()

    print(
        "XGBoost artifacts saved successfully."
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()