from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT
# ============================================================

from src.hybrid.combine import (
    load_predictions,
    create_hybrid_prediction,
    find_best_weight,
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
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
# MAIN
# ============================================================

def train_hybrid():

    print()
    print("=" * 70)
    print("HYBRID XGBOOST + TCN TRAINING")
    print("=" * 70)

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print(
        "Loading validation predictions..."
    )

    validation = load_predictions(
        "validation"
    )

    actual = validation[
        "actual"
    ].values

    xgb_prediction = validation[
        "xgboost_prediction"
    ].values

    tcn_prediction = validation[
        "tcn_prediction"
    ].values

    print()
    print(
        f"Validation samples used: "
        f"{len(actual):,}"
    )

    # ========================================================
    # OPTIMIZE WEIGHT
    # ========================================================

    print()
    print(
        "Searching for optimal hybrid weight..."
    )

    (
        best_tcn_weight,
        weight_results,
    ) = find_best_weight(
        actual=actual,
        xgb_prediction=xgb_prediction,
        tcn_prediction=tcn_prediction,
        step=0.01,
    )

    best_xgb_weight = (
        1.0
        - best_tcn_weight
    )

    best_row = (
        weight_results.loc[
            weight_results["tcn_weight"]
            == best_tcn_weight
        ]
        .iloc[0]
    )

    print()
    print("=" * 70)
    print("OPTIMAL HYBRID WEIGHTS")
    print("=" * 70)

    print()

    print(
        f"TCN weight: "
        f"{best_tcn_weight:.2f}"
    )

    print(
        f"XGBoost weight: "
        f"{best_xgb_weight:.2f}"
    )

    print()

    print(
        f"Validation MAE: "
        f"{best_row['MAE']:.6f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_row['RMSE']:.6f}"
    )

    # ========================================================
    # SAVE WEIGHT SEARCH
    # ========================================================

    weight_results.to_csv(
        RESULTS_DIR
        / "weight_search.csv",
        index=False,
    )

    # ========================================================
    # VALIDATION HYBRID
    # ========================================================

    validation_hybrid = (
        create_hybrid_prediction(
            xgb_prediction,
            tcn_prediction,
            best_tcn_weight,
        )
    )

    validation_result = pd.DataFrame({

        "timestamp":
            validation[
                "timestamp"
            ],

        "actual":
            actual,

        "xgboost_prediction":
            xgb_prediction,

        "tcn_prediction":
            tcn_prediction,

        "hybrid_prediction":
            validation_hybrid,
    })

    validation_result.to_csv(
        RESULTS_DIR
        / "validation_predictions.csv",
        index=False,
    )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print(
        "Loading test predictions..."
    )

    test = load_predictions(
        "test"
    )

    test_actual = test[
        "actual"
    ].values

    test_xgb = test[
        "xgboost_prediction"
    ].values

    test_tcn = test[
        "tcn_prediction"
    ].values

    print()
    print(
        f"Test samples used: "
        f"{len(test_actual):,}"
    )

    # ========================================================
    # FINAL TEST HYBRID
    # ========================================================

    print()
    print(
        "Generating final hybrid "
        "test predictions..."
    )

    test_hybrid = (
        create_hybrid_prediction(
            test_xgb,
            test_tcn,
            best_tcn_weight,
        )
    )

    test_result = pd.DataFrame({

        "timestamp":
            test[
                "timestamp"
            ],

        "actual":
            test_actual,

        "xgboost_prediction":
            test_xgb,

        "tcn_prediction":
            test_tcn,

        "hybrid_prediction":
            test_hybrid,
    })

    test_result.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False,
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {

        "method":
            "validation_optimized_weighted_average",

        "tcn_weight":
            float(best_tcn_weight),

        "xgboost_weight":
            float(best_xgb_weight),

        "optimization_metric":
            "MAE",

        "weight_search_step":
            0.01,

        "tcn_lookback":
            24,

        "xgboost_model":
            "models/xgboost/model.json",

        "tcn_model":
            "models/tcn/model.keras",
    }

    with open(
        MODEL_DIR
        / "config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("HYBRID TRAINING COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Frozen TCN weight: "
        f"{best_tcn_weight:.2f}"
    )

    print(
        f"Frozen XGBoost weight: "
        f"{best_xgb_weight:.2f}"
    )

    print()

    print(
        RESULTS_DIR
        / "weight_search.csv"
    )

    print(
        RESULTS_DIR
        / "validation_predictions.csv"
    )

    print(
        RESULTS_DIR
        / "test_predictions.csv"
    )

    print(
        MODEL_DIR
        / "config.json"
    )


if __name__ == "__main__":
    train_hybrid()