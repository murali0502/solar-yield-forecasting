"""
Hybrid ensemble for Solar Yield Forecasting.

Architecture:

    XGBoost
       +
      TCN
       ↓
Timestamp Alignment
       ↓
Validation-Optimized Weighted Ensemble
       ↓
Hybrid Prediction

Hybrid formula:

    Hybrid =
        XGBoost_weight × XGBoost_prediction
        +
        TCN_weight × TCN_prediction

The TCN weight is optimized using validation data only.

The test set is NOT used to determine the hybrid weight.
"""


from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

XGBOOST_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "xgboost"
)

TCN_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "tcn"
)

HYBRID_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
)

HYBRID_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid"
)


HYBRID_RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

HYBRID_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_WEIGHT_STEP = 0.01

MIN_WEIGHT = 0.0

MAX_WEIGHT = 1.0


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric_array(
    values,
    name,
):
    """
    Convert an input to a floating-point NumPy array and
    verify that it contains only finite values.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:

        array = array.reshape(-1)

    if len(array) == 0:

        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        array
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


# ============================================================
# LOAD AND ALIGN PREDICTIONS
# ============================================================

def load_predictions(
    split: str,
):
    """
    Load XGBoost and TCN predictions and align them
    using their actual target timestamps.

    Supported splits:

        validation
        test

    Required columns in both files:

        timestamp
        actual
        prediction

    The final dataframe contains:

        timestamp
        actual
        xgboost_prediction
        tcn_prediction

    Timestamp alignment prevents positional mismatches
    between XGBoost and TCN predictions.
    """

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    if split not in {
        "validation",
        "test",
    }:

        raise ValueError(
            "split must be "
            "'validation' or 'test'."
        )


    # --------------------------------------------------------
    # Prediction file paths
    # --------------------------------------------------------

    xgb_path = (
        XGBOOST_RESULTS
        / f"{split}_predictions.csv"
    )

    tcn_path = (
        TCN_RESULTS
        / f"{split}_predictions.csv"
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not xgb_path.exists():

        raise FileNotFoundError(
            "XGBoost predictions not found:\n"
            f"{xgb_path}"
        )

    if not tcn_path.exists():

        raise FileNotFoundError(
            "TCN predictions not found:\n"
            f"{tcn_path}"
        )


    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    xgb_df = pd.read_csv(
        xgb_path
    )

    tcn_df = pd.read_csv(
        tcn_path
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "timestamp",
        "actual",
        "prediction",
    ]


    for column in required_columns:

        if column not in xgb_df.columns:

            raise ValueError(
                "XGBoost prediction file is missing "
                f"'{column}'."
            )

        if column not in tcn_df.columns:

            raise ValueError(
                "TCN prediction file is missing "
                f"'{column}'."
            )


    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    xgb_df["timestamp"] = pd.to_datetime(
        xgb_df["timestamp"],
        errors="coerce",
    )

    tcn_df["timestamp"] = pd.to_datetime(
        tcn_df["timestamp"],
        errors="coerce",
    )


    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    if xgb_df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found in "
            "XGBoost predictions."
        )

    if tcn_df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found in "
            "TCN predictions."
        )


    # --------------------------------------------------------
    # Sort prediction files
    # --------------------------------------------------------

    xgb_df = (
        xgb_df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    tcn_df = (
        tcn_df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Check duplicate timestamps
    # --------------------------------------------------------

    if xgb_df["timestamp"].duplicated().any():

        duplicate_count = int(
            xgb_df["timestamp"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate timestamps found in "
            f"XGBoost predictions: "
            f"{duplicate_count}"
        )

    if tcn_df["timestamp"].duplicated().any():

        duplicate_count = int(
            tcn_df["timestamp"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate timestamps found in "
            f"TCN predictions: "
            f"{duplicate_count}"
        )


    # --------------------------------------------------------
    # Validate numeric values
    # --------------------------------------------------------

    xgb_actual = validate_numeric_array(
        xgb_df["actual"].values,
        "XGBoost actual values",
    )

    xgb_prediction = validate_numeric_array(
        xgb_df["prediction"].values,
        "XGBoost predictions",
    )

    tcn_actual = validate_numeric_array(
        tcn_df["actual"].values,
        "TCN actual values",
    )

    tcn_prediction = validate_numeric_array(
        tcn_df["prediction"].values,
        "TCN predictions",
    )


    # --------------------------------------------------------
    # Prevent negative predictions
    # --------------------------------------------------------

    xgb_df["prediction"] = np.clip(
        xgb_prediction,
        0,
        None,
    )

    tcn_df["prediction"] = np.clip(
        tcn_prediction,
        0,
        None,
    )


    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    xgb_df = xgb_df.rename(
        columns={
            "actual":
                "actual_xgb",

            "prediction":
                "xgboost_prediction",
        }
    )

    tcn_df = tcn_df.rename(
        columns={
            "actual":
                "actual_tcn",

            "prediction":
                "tcn_prediction",
        }
    )


    # --------------------------------------------------------
    # Align by timestamp
    # --------------------------------------------------------

    merged = pd.merge(
        xgb_df[
            [
                "timestamp",
                "actual_xgb",
                "xgboost_prediction",
            ]
        ],

        tcn_df[
            [
                "timestamp",
                "actual_tcn",
                "tcn_prediction",
            ]
        ],

        on="timestamp",

        how="inner",

        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Check alignment result
    # --------------------------------------------------------

    if merged.empty:

        raise ValueError(
            f"No matching timestamps found between "
            f"XGBoost and TCN {split} predictions."
        )


    # --------------------------------------------------------
    # Sort merged result
    # --------------------------------------------------------

    merged = (
        merged
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Verify actual target values
    # --------------------------------------------------------

    if not np.allclose(
        merged["actual_xgb"].values,
        merged["actual_tcn"].values,
        rtol=1e-5,
        atol=1e-5,
    ):

        differences = np.abs(
            merged["actual_xgb"].values
            -
            merged["actual_tcn"].values
        )

        max_difference = float(
            np.max(differences)
        )

        raise ValueError(
            "Actual target values do not match "
            "for matching timestamps. "
            f"Maximum difference: "
            f"{max_difference}"
        )


    # --------------------------------------------------------
    # Create common actual column
    # --------------------------------------------------------

    merged["actual"] = (
        merged["actual_xgb"]
    )


    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    merged = merged[
        [
            "timestamp",
            "actual",
            "xgboost_prediction",
            "tcn_prediction",
        ]
    ]


    # --------------------------------------------------------
    # Final numeric validation
    # --------------------------------------------------------

    validate_numeric_array(
        merged["actual"].values,
        f"{split} actual values",
    )

    validate_numeric_array(
        merged["xgboost_prediction"].values,
        f"{split} XGBoost predictions",
    )

    validate_numeric_array(
        merged["tcn_prediction"].values,
        f"{split} TCN predictions",
    )


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()

    print(
        f"{split.capitalize()} prediction alignment:"
    )

    print(
        f"  XGBoost predictions : "
        f"{len(xgb_df):,}"
    )

    print(
        f"  TCN predictions     : "
        f"{len(tcn_df):,}"
    )

    print(
        f"  Matching timestamps : "
        f"{len(merged):,}"
    )

    print(
        f"  First timestamp     : "
        f"{merged['timestamp'].iloc[0]}"
    )

    print(
        f"  Last timestamp      : "
        f"{merged['timestamp'].iloc[-1]}"
    )

    print(
        "  Actual values       : MATCH"
    )

    print(
        "  Target alignment    : OK"
    )

    print()


    return merged


# ============================================================
# CREATE HYBRID PREDICTION
# ============================================================

def create_hybrid_prediction(
    xgb_prediction,
    tcn_prediction,
    tcn_weight,
):
    """
    Create a weighted hybrid prediction.

    Formula:

        Hybrid =
            (1 - TCN_weight) × XGBoost
            +
            TCN_weight × TCN

    Therefore:

        XGBoost weight = 1 - TCN weight
    """

    xgb_prediction = validate_numeric_array(
        xgb_prediction,
        "XGBoost predictions",
    )

    tcn_prediction = validate_numeric_array(
        tcn_prediction,
        "TCN predictions",
    )


    # --------------------------------------------------------
    # Length validation
    # --------------------------------------------------------

    if len(xgb_prediction) != len(
        tcn_prediction
    ):

        raise ValueError(
            "Prediction lengths must match. "
            f"XGBoost={len(xgb_prediction)}, "
            f"TCN={len(tcn_prediction)}."
        )


    # --------------------------------------------------------
    # Weight validation
    # --------------------------------------------------------

    tcn_weight = float(
        tcn_weight
    )

    if not np.isfinite(
        tcn_weight
    ):

        raise ValueError(
            "tcn_weight must be finite."
        )

    if not (
        MIN_WEIGHT
        <= tcn_weight
        <= MAX_WEIGHT
    ):

        raise ValueError(
            "tcn_weight must be "
            "between 0 and 1."
        )


    # --------------------------------------------------------
    # Calculate XGBoost weight
    # --------------------------------------------------------

    xgb_weight = (
        1.0
        - tcn_weight
    )


    # --------------------------------------------------------
    # Weighted ensemble
    # --------------------------------------------------------

    prediction = (
        xgb_weight
        * xgb_prediction
        +
        tcn_weight
        * tcn_prediction
    )


    # --------------------------------------------------------
    # Solar output cannot be negative
    # --------------------------------------------------------

    prediction = np.clip(
        prediction,
        0,
        None,
    )


    return prediction


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    actual,
    prediction,
):
    """
    Calculate MAE, RMSE and R2.
    """

    actual = validate_numeric_array(
        actual,
        "Actual values",
    )

    prediction = validate_numeric_array(
        prediction,
        "Predictions",
    )


    if len(actual) != len(
        prediction
    ):

        raise ValueError(
            "Actual and prediction lengths "
            "must match."
        )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = (
        actual
        -
        prediction
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(error)
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (
            actual
            -
            np.mean(actual)
        ) ** 2
    )

    if ss_tot == 0:

        r2 = 0.0

    else:

        r2 = (
            1.0
            -
            (
                ss_res
                /
                ss_tot
            )
        )


    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# FIND BEST WEIGHT
# ============================================================

def find_best_weight(
    actual,
    xgb_prediction,
    tcn_prediction,
    step=DEFAULT_WEIGHT_STEP,
):
    """
    Optimize the TCN weight using validation data only.

    The search is:

        TCN weight = 0.00 → 1.00

    with the supplied step.

    The weight that produces the lowest validation MAE
    is selected.

    The test set must NOT be passed to this function
    for optimization.
    """

    actual = validate_numeric_array(
        actual,
        "Actual values",
    )

    xgb_prediction = validate_numeric_array(
        xgb_prediction,
        "XGBoost predictions",
    )

    tcn_prediction = validate_numeric_array(
        tcn_prediction,
        "TCN predictions",
    )


    # --------------------------------------------------------
    # Length validation
    # --------------------------------------------------------

    if not (
        len(actual)
        ==
        len(xgb_prediction)
        ==
        len(tcn_prediction)
    ):

        raise ValueError(
            "All prediction arrays must have "
            "the same length."
        )


    # --------------------------------------------------------
    # Validate step
    # --------------------------------------------------------

    step = float(
        step
    )

    if not np.isfinite(
        step
    ):

        raise ValueError(
            "step must be finite."
        )

    if step <= 0.0:

        raise ValueError(
            "step must be greater than zero."
        )

    if step > 1.0:

        raise ValueError(
            "step cannot be greater than 1.0."
        )


    # --------------------------------------------------------
    # Weight values
    # --------------------------------------------------------

    weights = np.arange(
        0.0,
        1.0 + step / 2.0,
        step,
        dtype=float,
    )


    # Ensure exactly 1.0 is available.

    weights = np.clip(
        weights,
        0.0,
        1.0,
    )


    if not np.isclose(
        weights[-1],
        1.0,
    ):

        weights = np.append(
            weights,
            1.0,
        )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = []


    for weight in weights:

        hybrid_prediction = (
            create_hybrid_prediction(
                xgb_prediction=xgb_prediction,

                tcn_prediction=tcn_prediction,

                tcn_weight=weight,
            )
        )


        metrics = calculate_metrics(
            actual,
            hybrid_prediction,
        )


        results.append(
            {
                "tcn_weight":
                    float(weight),

                "xgboost_weight":
                    float(
                        1.0 - weight
                    ),

                "MAE":
                    metrics["MAE"],

                "RMSE":
                    metrics["RMSE"],

                "R2":
                    metrics["R2"],
            }
        )


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Find best MAE
    # --------------------------------------------------------

    best_index = (
        results_df["MAE"]
        .idxmin()
    )


    best_weight = float(
        results_df.loc[
            best_index,
            "tcn_weight",
        ]
    )


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    best_row = (
        results_df
        .loc[
            best_index
        ]
    )


    print()

    print(
        "=" * 70
    )

    print(
        "HYBRID WEIGHT OPTIMIZATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Weight search step : "
        f"{step}"
    )

    print(
        f"Best TCN weight    : "
        f"{best_weight:.2f}"
    )

    print(
        f"Best XGBoost weight: "
        f"{1.0 - best_weight:.2f}"
    )

    print(
        f"Validation MAE     : "
        f"{best_row['MAE']:.6f}"
    )

    print(
        f"Validation RMSE    : "
        f"{best_row['RMSE']:.6f}"
    )

    print(
        f"Validation R2      : "
        f"{best_row['R2']:.6f}"
    )

    print(
        "=" * 70
    )

    print()


    return (
        best_weight,
        results_df,
    )


# ============================================================
# SAVE HYBRID PREDICTIONS
# ============================================================

def save_hybrid_predictions(
    aligned_df,
    tcn_weight,
    output_path=None,
):
    """
    Generate and save timestamp-aligned hybrid predictions.

    Input dataframe must contain:

        timestamp
        actual
        xgboost_prediction
        tcn_prediction
    """

    required_columns = [
        "timestamp",
        "actual",
        "xgboost_prediction",
        "tcn_prediction",
    ]


    for column in required_columns:

        if column not in aligned_df.columns:

            raise ValueError(
                "Aligned dataframe is missing "
                f"'{column}'."
            )


    # --------------------------------------------------------
    # Create hybrid
    # --------------------------------------------------------

    hybrid_prediction = (
        create_hybrid_prediction(
            xgb_prediction=
                aligned_df[
                    "xgboost_prediction"
                ].values,

            tcn_prediction=
                aligned_df[
                    "tcn_prediction"
                ].values,

            tcn_weight=tcn_weight,
        )
    )


    # --------------------------------------------------------
    # Create result
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "timestamp":
                pd.to_datetime(
                    aligned_df[
                        "timestamp"
                    ]
                ).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "actual":
                aligned_df[
                    "actual"
                ].values,

            "xgboost_prediction":
                aligned_df[
                    "xgboost_prediction"
                ].values,

            "tcn_prediction":
                aligned_df[
                    "tcn_prediction"
                ].values,

            "hybrid_prediction":
                hybrid_prediction,
        }
    )


    # --------------------------------------------------------
    # Default path
    # --------------------------------------------------------

    if output_path is None:

        output_path = (
            HYBRID_RESULTS
            / "hybrid_predictions.csv"
        )


    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    result.to_csv(
        output_path,
        index=False,
    )


    print(
        "Hybrid predictions saved:"
    )

    print(
        output_path
    )


    return result


# ============================================================
# SAVE HYBRID CONFIGURATION
# ============================================================

def save_hybrid_config(
    tcn_weight,
    validation_metrics=None,
    output_path=None,
):
    """
    Save the selected hybrid weights and configuration.

    The configuration is later used by the inference
    layer for production forecasting.
    """

    tcn_weight = float(
        tcn_weight
    )

    if not (
        0.0
        <= tcn_weight
        <= 1.0
    ):

        raise ValueError(
            "tcn_weight must be "
            "between 0 and 1."
        )


    xgb_weight = (
        1.0
        - tcn_weight
    )


    if output_path is None:

        output_path = (
            HYBRID_MODEL_DIR
            / "config.json"
        )


    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    config = {

        "model_type":
            "Weighted Hybrid Ensemble",

        "method":
            "validation_optimized_weighted_average",

        "architecture":
            "XGBoost + TCN + Hybrid",

        "lstm_used":
            False,

        "tcn_weight":
            tcn_weight,

        "xgboost_weight":
            xgb_weight,

        "weight_sum":
            tcn_weight + xgb_weight,

        "optimization_metric":
            "MAE",

        "optimization_dataset":
            "validation",

        "test_data_used_for_weight_optimization":
            False,
    }


    if validation_metrics is not None:

        config[
            "validation_metrics"
        ] = {
            key: float(value)
            for key, value
            in validation_metrics.items()
        }


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )


    print(
        "Hybrid configuration saved:"
    )

    print(
        output_path
    )


    return config


# ============================================================
# MAIN HYBRID PIPELINE
# ============================================================

def run_hybrid_training():
    """
    Complete hybrid training/evaluation pipeline.

    Steps:

        1. Load validation predictions
        2. Align XGBoost and TCN by timestamp
        3. Optimize TCN weight using validation data
        4. Save validation hybrid predictions
        5. Load test predictions
        6. Align test predictions by timestamp
        7. Apply the validation-selected weight to test data
        8. Save test hybrid predictions
        9. Save hybrid configuration

    IMPORTANT:

        The test set is NEVER used to optimize the weight.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "HYBRID ENSEMBLE TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "XGBoost + TCN + Validation-Optimized Hybrid"
    )

    print()


    # ========================================================
    # VALIDATION ALIGNMENT
    # ========================================================

    validation_df = load_predictions(
        "validation"
    )


    # ========================================================
    # FIND BEST WEIGHT
    # ========================================================

    (
        best_tcn_weight,
        weight_results,
    ) = find_best_weight(

        actual=
            validation_df[
                "actual"
            ].values,

        xgb_prediction=
            validation_df[
                "xgboost_prediction"
            ].values,

        tcn_prediction=
            validation_df[
                "tcn_prediction"
            ].values,

        step=DEFAULT_WEIGHT_STEP,
    )


    # ========================================================
    # VALIDATION HYBRID
    # ========================================================

    validation_hybrid_df = (
        save_hybrid_predictions(

            aligned_df=
                validation_df,

            tcn_weight=
                best_tcn_weight,

            output_path=(
                HYBRID_RESULTS
                / "validation_predictions.csv"
            ),
        )
    )


    # ========================================================
    # SAVE WEIGHT SEARCH
    # ========================================================

    weight_search_path = (
        HYBRID_RESULTS
        / "weight_search.csv"
    )

    weight_results.to_csv(
        weight_search_path,
        index=False,
    )

    print(
        "Weight search results saved:"
    )

    print(
        weight_search_path
    )

    print()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    validation_metrics = (
        calculate_metrics(

            validation_hybrid_df[
                "actual"
            ].values,

            validation_hybrid_df[
                "hybrid_prediction"
            ].values,
        )
    )


    # ========================================================
    # TEST ALIGNMENT
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "TESTING HYBRID ENSEMBLE"
    )

    print(
        "=" * 70
    )

    test_df = load_predictions(
        "test"
    )


    # ========================================================
    # APPLY VALIDATION WEIGHT TO TEST
    # ========================================================

    test_hybrid_df = (
        save_hybrid_predictions(

            aligned_df=
                test_df,

            tcn_weight=
                best_tcn_weight,

            output_path=(
                HYBRID_RESULTS
                / "test_predictions.csv"
            ),
        )
    )


    # ========================================================
    # TEST METRICS
    # ========================================================

    test_metrics = (
        calculate_metrics(

            test_hybrid_df[
                "actual"
            ].values,

            test_hybrid_df[
                "hybrid_prediction"
            ].values,
        )
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "validation": {
            key: float(value)
            for key, value
            in validation_metrics.items()
        },

        "test": {
            key: float(value)
            for key, value
            in test_metrics.items()
        },

        "tcn_weight":
            float(best_tcn_weight),

        "xgboost_weight":
            float(
                1.0
                -
                best_tcn_weight
            ),
    }


    metrics_path = (
        HYBRID_RESULTS
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    save_hybrid_config(

        tcn_weight=
            best_tcn_weight,

        validation_metrics=
            validation_metrics,

        output_path=(
            HYBRID_MODEL_DIR
            / "config.json"
        ),
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "HYBRID ENSEMBLE COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"TCN weight     : "
        f"{best_tcn_weight:.2f}"
    )

    print(
        f"XGBoost weight : "
        f"{1.0 - best_tcn_weight:.2f}"
    )

    print()

    print(
        "Validation metrics:"
    )

    print(
        f"  MAE  : "
        f"{validation_metrics['MAE']:.6f}"
    )

    print(
        f"  RMSE : "
        f"{validation_metrics['RMSE']:.6f}"
    )

    print(
        f"  R2   : "
        f"{validation_metrics['R2']:.6f}"
    )

    print()

    print(
        "Test metrics:"
    )

    print(
        f"  MAE  : "
        f"{test_metrics['MAE']:.6f}"
    )

    print(
        f"  RMSE : "
        f"{test_metrics['RMSE']:.6f}"
    )

    print(
        f"  R2   : "
        f"{test_metrics['R2']:.6f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Test data was NOT used to optimize "
        "the hybrid weight."
    )

    print()

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_hybrid_training()