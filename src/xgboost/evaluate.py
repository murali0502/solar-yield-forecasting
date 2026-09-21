from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost"
    / "predictions.csv"
)


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # Solar output contains many zeros.
    # Standard MAPE is unstable around zero.
    non_zero_mask = (
        np.abs(y_true) > 1e-6
    )

    if non_zero_mask.any():

        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero_mask]
                        - y_pred[non_zero_mask]
                    )
                    / y_true[non_zero_mask]
                )
            )
            * 100
        )

    else:

        mape = None

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": (
            float(mape)
            if mape is not None
            else None
        ),
    }


def evaluate():

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            "XGBoost predictions not found.\n"
            "Run train.py first."
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    y_true = predictions[
        "actual"
    ].values

    y_pred = predictions[
        "prediction"
    ].values

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print("=" * 70)
    print("XGBOOST TEST RESULTS")
    print("=" * 70)

    for name, value in metrics.items():

        if value is None:

            print(
                f"{name}: N/A"
            )

        else:

            print(
                f"{name}: {value:.6f}"
            )

    return metrics


if __name__ == "__main__":
    evaluate()