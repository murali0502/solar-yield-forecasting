from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tcn"
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

    # Exclude exact zero targets from MAPE.
    non_zero = (
        np.abs(y_true)
        > 1e-6
    )

    if non_zero.any():

        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero]
                        - y_pred[non_zero]
                    )
                    / y_true[non_zero]
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
            "TCN predictions not found.\n"
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
    print("TCN TEST RESULTS")
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


if __name__ == "__main__":
    evaluate()