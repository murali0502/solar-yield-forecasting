from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.inference.predictor import (
    SolarPredictor,
    TCN_LOOKBACK,
)


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "solar_dataset.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INFERENCE ACCURACY VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    predictor = SolarPredictor()

    # --------------------------------------------------------
    # Use a small sample from the final test period
    #
    # We need 24 historical rows for TCN.
    # --------------------------------------------------------

    start_index = len(df) - 48

    end_index = len(df) - 1

    results = []

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    for i in range(
        start_index + TCN_LOOKBACK,
        end_index + 1,
    ):

        history = df.iloc[
            i - TCN_LOOKBACK + 1:
            i + 1
        ].copy()

        result = predictor.predict(
            history
        )

        actual = float(
            df.iloc[i]["solar_output"]
        )

        results.append({
            "timestamp":
                df.iloc[i]["timestamp"],

            "actual":
                actual,

            "xgboost_prediction":
                result[
                    "xgboost_prediction"
                ],

            "tcn_prediction":
                result[
                    "tcn_prediction"
                ],

            "hybrid_prediction":
                result[
                    "hybrid_prediction"
                ],
        })

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    actual = result_df[
        "actual"
    ].values

    xgb = result_df[
        "xgboost_prediction"
    ].values

    tcn = result_df[
        "tcn_prediction"
    ].values

    hybrid = result_df[
        "hybrid_prediction"
    ].values

    print()
    print("=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)

    print()

    print(
        f"Samples tested: "
        f"{len(result_df)}"
    )

    print()

    print(
        "XGBoost"
    )

    print(
        f"  MAE:  "
        f"{mean_absolute_error(actual, xgb):.6f}"
    )

    print(
        f"  RMSE: "
        f"{np.sqrt(mean_squared_error(actual, xgb)):.6f}"
    )

    print(
        f"  R2:   "
        f"{r2_score(actual, xgb):.6f}"
    )

    print()

    print(
        "TCN"
    )

    print(
        f"  MAE:  "
        f"{mean_absolute_error(actual, tcn):.6f}"
    )

    print(
        f"  RMSE: "
        f"{np.sqrt(mean_squared_error(actual, tcn)):.6f}"
    )

    print(
        f"  R2:   "
        f"{r2_score(actual, tcn):.6f}"
    )

    print()

    print(
        "Hybrid"
    )

    print(
        f"  MAE:  "
        f"{mean_absolute_error(actual, hybrid):.6f}"
    )

    print(
        f"  RMSE: "
        f"{np.sqrt(mean_squared_error(actual, hybrid)):.6f}"
    )

    print(
        f"  R2:   "
        f"{r2_score(actual, hybrid):.6f}"
    )

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print()
    print(
        "Sample predictions:"
    )

    print()

    print(
        result_df.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("INFERENCE ACCURACY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()