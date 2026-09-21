from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid"
)


# ============================================================
# FILES
# ============================================================

TEST_FILE = (
    RESULTS_DIR
    / "test_predictions.csv"
)

CONFIG_FILE = (
    MODEL_DIR
    / "config.json"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "test_metrics.csv"
)


# ============================================================
# MAPE
# ============================================================

def calculate_mape(
    actual,
    prediction,
):
    """
    Calculate MAPE while ignoring zero actual values.

    Solar output contains many nighttime zero values.
    Including them would make MAPE unstable/infinite.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - prediction[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    actual,
    prediction,
):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction,
        )
    )

    r2 = r2_score(
        actual,
        prediction,
    )

    mape = calculate_mape(
        actual,
        prediction,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print()
    print("=" * 70)
    print("HYBRID TEST EVALUATION")
    print("=" * 70)

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Hybrid test predictions not found:\n"
            f"{TEST_FILE}\n\n"
            f"Run train_hybrid.py first."
        )

    # ========================================================
    # LOAD TEST PREDICTIONS
    # ========================================================

    predictions = pd.read_csv(
        TEST_FILE
    )

    print()
    print(
        f"Test samples: "
        f"{len(predictions):,}"
    )

    print()
    print(
        "Columns:"
    )

    print(
        predictions.columns.tolist()
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "timestamp",
        "actual",
        "xgboost_prediction",
        "tcn_prediction",
        "hybrid_prediction",
    ]

    for column in required_columns:

        if column not in predictions.columns:

            raise ValueError(
                f"Required column missing: "
                f"'{column}'"
            )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    if CONFIG_FILE.exists():

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(
                file
            )

    else:

        config = {}

    # ========================================================
    # EXTRACT ARRAYS
    # ========================================================

    actual = predictions[
        "actual"
    ].values

    xgb_prediction = predictions[
        "xgboost_prediction"
    ].values

    tcn_prediction = predictions[
        "tcn_prediction"
    ].values

    hybrid_prediction = predictions[
        "hybrid_prediction"
    ].values

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    xgb_metrics = calculate_metrics(
        actual,
        xgb_prediction,
    )

    tcn_metrics = calculate_metrics(
        actual,
        tcn_prediction,
    )

    hybrid_metrics = calculate_metrics(
        actual,
        hybrid_prediction,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print()

    print(
        "XGBoost"
    )

    print(
        f"  MAE:  "
        f"{xgb_metrics['MAE']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{xgb_metrics['RMSE']:.6f}"
    )

    print(
        f"  R2:   "
        f"{xgb_metrics['R2']:.6f}"
    )

    print(
        f"  MAPE: "
        f"{xgb_metrics['MAPE']:.6f}%"
    )

    print()

    print(
        "TCN"
    )

    print(
        f"  MAE:  "
        f"{tcn_metrics['MAE']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{tcn_metrics['RMSE']:.6f}"
    )

    print(
        f"  R2:   "
        f"{tcn_metrics['R2']:.6f}"
    )

    print(
        f"  MAPE: "
        f"{tcn_metrics['MAPE']:.6f}%"
    )

    print()

    print(
        "Hybrid"
    )

    print(
        f"  MAE:  "
        f"{hybrid_metrics['MAE']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{hybrid_metrics['RMSE']:.6f}"
    )

    print(
        f"  R2:   "
        f"{hybrid_metrics['R2']:.6f}"
    )

    print(
        f"  MAPE: "
        f"{hybrid_metrics['MAPE']:.6f}%"
    )

    # ========================================================
    # SAVE METRICS TABLE
    # ========================================================

    metrics_table = pd.DataFrame([

        {
            "model": "XGBoost",
            **xgb_metrics,
        },

        {
            "model": "TCN",
            **tcn_metrics,
        },

        {
            "model": "Hybrid",
            **hybrid_metrics,
        },

    ])

    metrics_table.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # WEIGHTS
    # ========================================================

    print()

    if config:

        print(
            "Hybrid configuration:"
        )

        print(
            f"  TCN weight: "
            f"{config.get('tcn_weight', 'N/A')}"
        )

        print(
            f"  XGBoost weight: "
            f"{config.get('xgboost_weight', 'N/A')}"
        )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_mae_row = (
        metrics_table.loc[
            metrics_table["MAE"].idxmin()
        ]
    )

    best_rmse_row = (
        metrics_table.loc[
            metrics_table["RMSE"].idxmin()
        ]
    )

    best_r2_row = (
        metrics_table.loc[
            metrics_table["R2"].idxmax()
        ]
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"Best MAE:  "
        f"{best_mae_row['model']}"
    )

    print(
        f"Best RMSE: "
        f"{best_rmse_row['model']}"
    )

    print(
        f"Best R2:   "
        f"{best_r2_row['model']}"
    )

    print()

    print(
        "Metrics saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("HYBRID EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()