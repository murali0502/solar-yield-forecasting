"""
TCN training pipeline for Solar Yield Forecasting.

Architecture:
    24-hour lookback TCN
    Sequence-to-one prediction

Input:
    Previous 24 hours of solar/weather features

Output:
    Solar output for the next hour

Project structure:
    src/tcn/train.py
    src/tcn/model.py
    models/tcn/
    results/tcn/

The TCN is used together with XGBoost in the hybrid
ensemble stage.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

import tensorflow as tf


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
# IMPORT PROJECT MODULES
# ============================================================

from src.preprocessing.data_loader import (
    prepare_dataset,
    chronological_split,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from src.tcn.model import build_tcn


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
    / "tcn"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "tcn"
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

LOOKBACK = 24

BATCH_SIZE = 64

EPOCHS = 80

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(
    df,
    feature_scaler,
    target_scaler,
):
    """
    Convert a chronological dataframe into TCN sequences.

    Sequence structure:

        Previous 24 hours
                ↓
               TCN
                ↓
          Next-hour output

    Example:

        X[0:24] -> y[24]
        X[1:25] -> y[25]
        X[2:26] -> y[26]

    Therefore, the target timestamp is exactly the timestamp
    associated with y[i].

    Returns:
        X
        y
        timestamps
    """

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    feature_values = (
        feature_scaler.transform(
            df[FEATURE_COLUMNS]
        )
    )

    # --------------------------------------------------------
    # Scale target
    # --------------------------------------------------------

    target_values = (
        target_scaler.transform(
            df[[TARGET_COLUMN]]
        )
        .reshape(-1)
    )

    # --------------------------------------------------------
    # Check minimum data
    # --------------------------------------------------------

    if len(df) <= LOOKBACK:

        raise ValueError(
            "Not enough rows to create TCN sequences. "
            f"Need more than {LOOKBACK} rows, "
            f"but received {len(df)}."
        )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    X = []
    y = []
    timestamps = []

    for i in range(
        LOOKBACK,
        len(df),
    ):

        # ----------------------------------------------------
        # Previous 24 hours
        # ----------------------------------------------------

        X.append(
            feature_values[
                i - LOOKBACK:i
            ]
        )

        # ----------------------------------------------------
        # Next/current target timestamp
        #
        # The sequence ends at i-1.
        # The target is y[i].
        # ----------------------------------------------------

        y.append(
            target_values[i]
        )

        # ----------------------------------------------------
        # Preserve target timestamp
        # ----------------------------------------------------

        if "timestamp" in df.columns:

            timestamps.append(
                df["timestamp"].iloc[i]
            )

        else:

            timestamps.append(
                i
            )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
    )

    return (
        X,
        y,
        timestamps,
    )


# ============================================================
# INVERSE TARGET
# ============================================================

def inverse_target(
    values,
    target_scaler,
):
    """
    Convert scaled target values back to the original
    solar-output unit.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1, 1)

    return (
        target_scaler
        .inverse_transform(
            values
        )
        .reshape(-1)
    )


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

    The timestamp is essential because the hybrid stage
    combines XGBoost and TCN predictions by forecast time.
    """

    prediction_df = pd.DataFrame(
        {
            "timestamp": timestamps,

            "actual": actual,

            "prediction": prediction,
        }
    )

    prediction_df.to_csv(
        output_path,
        index=False,
    )

    return prediction_df


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print()

    print(
        "=" * 70
    )

    print(
        "TCN TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"Lookback window : "
        f"{LOOKBACK} hours"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs          : "
        f"{EPOCHS}"
    )

    print(
        f"Random seed     : "
        f"{RANDOM_SEED}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "24-hour sequence-to-one TCN"
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

    print(
        f"Dataset shape: "
        f"{df.shape}"
    )

    print(
        f"Features: "
        f"{FEATURE_COLUMNS}"
    )

    print(
        f"Target: "
        f"{TARGET_COLUMN}"
    )

    print()


    # ========================================================
    # VERIFY TIMESTAMP
    # ========================================================

    if "timestamp" not in df.columns:

        raise ValueError(
            "The prepared dataset must contain a "
            "'timestamp' column so that TCN predictions "
            "can be aligned with XGBoost predictions."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values were found "
            "in the prepared dataset."
        )

    # Ensure chronological ordering.

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
        "Timestamp validation: OK"
    )

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
    # FIT FEATURE SCALER
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "FITTING SCALERS"
    )

    print(
        "=" * 70
    )

    print(
        "Feature scaler: StandardScaler"
    )

    print(
        "Target scaler : MinMaxScaler"
    )

    print(
        "Scalers are fitted ONLY on training data."
    )

    print()

    feature_scaler = (
        StandardScaler()
    )

    feature_scaler.fit(
        train_df[
            FEATURE_COLUMNS
        ]
    )

    target_scaler = (
        MinMaxScaler()
    )

    target_scaler.fit(
        train_df[
            [TARGET_COLUMN]
        ]
    )

    print(
        "Scaler fitting: OK"
    )

    print()


    # ========================================================
    # CREATE TRAINING SEQUENCES
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "CREATING TCN SEQUENCES"
    )

    print(
        "=" * 70
    )

    (
        X_train,
        y_train,
        train_timestamps,
    ) = create_sequences(
        train_df,
        feature_scaler,
        target_scaler,
    )

    (
        X_validation,
        y_validation,
        validation_timestamps,
    ) = create_sequences(
        validation_df,
        feature_scaler,
        target_scaler,
    )

    (
        X_test,
        y_test,
        test_timestamps,
    ) = create_sequences(
        test_df,
        feature_scaler,
        target_scaler,
    )

    print()

    print(
        f"X_train       : "
        f"{X_train.shape}"
    )

    print(
        f"y_train       : "
        f"{y_train.shape}"
    )

    print(
        f"X_validation  : "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation  : "
        f"{y_validation.shape}"
    )

    print(
        f"X_test        : "
        f"{X_test.shape}"
    )

    print(
        f"y_test        : "
        f"{y_test.shape}"
    )

    print()

    # --------------------------------------------------------
    # Expected shape
    #
    # (samples, 24, 9)
    # --------------------------------------------------------

    expected_feature_count = (
        len(FEATURE_COLUMNS)
    )

    if X_train.shape[1] != LOOKBACK:

        raise ValueError(
            "Unexpected TCN lookback dimension. "
            f"Expected {LOOKBACK}, "
            f"got {X_train.shape[1]}."
        )

    if X_train.shape[2] != expected_feature_count:

        raise ValueError(
            "Unexpected TCN feature dimension. "
            f"Expected {expected_feature_count}, "
            f"got {X_train.shape[2]}."
        )

    print(
        "TCN sequence dimensions: OK"
    )

    print()


    # ========================================================
    # BUILD TCN MODEL
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "BUILDING TCN MODEL"
    )

    print(
        "=" * 70
    )

    model = build_tcn(
        lookback=LOOKBACK,
        number_of_features=(
            len(FEATURE_COLUMNS)
        ),
    )

    print()

    model.summary()

    print()


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        EarlyStopping(
            monitor="val_loss",

            patience=12,

            restore_best_weights=True,

            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=1,
        ),
    ]


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "STARTING TCN TRAINING"
    )

    print(
        "=" * 70
    )

    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_validation,
            y_validation,
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        # Keep temporal samples in their original order.
        shuffle=False,

        verbose=1,
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR
        / "model.keras"
    )

    model.save(
        model_path
    )

    print()

    print(
        "TCN model saved:"
    )

    print(
        model_path
    )

    print()


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "GENERATING VALIDATION PREDICTIONS"
    )

    print(
        "=" * 70
    )

    validation_predictions_scaled = (
        model.predict(
            X_validation,
            verbose=0,
        )
        .reshape(-1)
    )

    validation_predictions = (
        inverse_target(
            validation_predictions_scaled,
            target_scaler,
        )
    )

    validation_actual = (
        inverse_target(
            y_validation,
            target_scaler,
        )
    )

    # Solar production cannot be negative.

    validation_predictions = np.clip(
        validation_predictions,
        0,
        None,
    )

    validation_actual = np.clip(
        validation_actual,
        0,
        None,
    )


    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    validation_output_path = (
        RESULTS_DIR
        / "validation_predictions.csv"
    )

    save_predictions(
        timestamps=validation_timestamps,

        actual=validation_actual,

        prediction=validation_predictions,

        output_path=(
            validation_output_path
        ),
    )

    print(
        "Validation predictions saved:"
    )

    print(
        validation_output_path
    )

    print()


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "GENERATING TEST PREDICTIONS"
    )

    print(
        "=" * 70
    )

    test_predictions_scaled = (
        model.predict(
            X_test,
            verbose=0,
        )
        .reshape(-1)
    )

    test_predictions = (
        inverse_target(
            test_predictions_scaled,
            target_scaler,
        )
    )

    test_actual = (
        inverse_target(
            y_test,
            target_scaler,
        )
    )

    # Solar production cannot be negative.

    test_predictions = np.clip(
        test_predictions,
        0,
        None,
    )

    test_actual = np.clip(
        test_actual,
        0,
        None,
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_output_path = (
        RESULTS_DIR
        / "test_predictions.csv"
    )

    save_predictions(
        timestamps=test_timestamps,

        actual=test_actual,

        prediction=test_predictions,

        output_path=(
            test_output_path
        ),
    )

    print(
        "Test predictions saved:"
    )

    print(
        test_output_path
    )

    print()


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING TRAINING HISTORY"
    )

    print(
        "=" * 70
    )

    history_df = pd.DataFrame(
        history.history
    )

    history_path = (
        RESULTS_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print(
        "Training history saved:"
    )

    print(
        history_path
    )

    print()


    # ========================================================
    # SAVE SCALERS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING SCALERS"
    )

    print(
        "=" * 70
    )

    feature_scaler_path = (
        MODEL_DIR
        / "feature_scaler.joblib"
    )

    target_scaler_path = (
        MODEL_DIR
        / "target_scaler.joblib"
    )

    joblib.dump(
        feature_scaler,
        feature_scaler_path,
    )

    joblib.dump(
        target_scaler,
        target_scaler_path,
    )

    print(
        "Feature scaler:"
    )

    print(
        feature_scaler_path
    )

    print()

    print(
        "Target scaler:"
    )

    print(
        target_scaler_path
    )

    print()


    # ========================================================
    # SAVE MODEL CONFIGURATION
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SAVING TCN CONFIGURATION"
    )

    print(
        "=" * 70
    )

    config = {

        "model_type":
            "TCN",

        "prediction_type":
            "sequence_to_one",

        "lookback":
            LOOKBACK,

        "features":
            list(FEATURE_COLUMNS),

        "number_of_features":
            len(FEATURE_COLUMNS),

        "target":
            TARGET_COLUMN,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "random_seed":
            RANDOM_SEED,

        "target_definition":
            "next_hour",

        "input_description":
            "previous_24_hours",

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
        "TCN TRAINING COMPLETE"
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
        "Input shape:"
    )

    print(
        f"(samples, {LOOKBACK}, "
        f"{len(FEATURE_COLUMNS)})"
    )

    print()

    print(
        "Output shape:"
    )

    print(
        "(samples, 1)"
    )

    print()

    print(
        "Prediction type:"
    )

    print(
        "24-hour sequence-to-one"
    )

    print()

    print(
        "Target:"
    )

    print(
        "Next-hour solar output"
    )

    print()

    print(
        "Validation predictions:"
    )

    print(
        validation_output_path
    )

    print()

    print(
        "Test predictions:"
    )

    print(
        test_output_path
    )

    print()

    print(
        "Training history:"
    )

    print(
        history_path
    )

    print()

    print(
        "TCN artifacts saved successfully."
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