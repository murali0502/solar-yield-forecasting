from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

HYBRID_DIR = RESULTS_DIR / "hybrid"
XGB_DIR = RESULTS_DIR / "xgboost"
TCN_DIR = RESULTS_DIR / "tcn"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FILES
# ============================================================
MODEL_COMPARISON = FIGURES_DIR / "model_comparison.csv"
HYBRID_TEST = HYBRID_DIR / "test_predictions.csv"
XGB_IMPORTANCE = XGB_DIR / "feature_importance.csv"
TCN_HISTORY = TCN_DIR / "training_history.csv"

# ============================================================
# HELPERS
# ============================================================
def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

def save_figure(filename):
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ============================================================
# 1. MODEL COMPARISON
# ============================================================
def model_comparison():
    require_file(MODEL_COMPARISON)

    metrics = pd.read_csv(MODEL_COMPARISON)

    required = ["model", "MAE", "RMSE", "R2", "MAPE"]
    missing = [c for c in required if c not in metrics.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {HYBRID_METRICS}: {missing}"
        )

    metrics = metrics.copy()
    metrics["R2_percent"] = metrics["R2"] * 100

    # Save a presentation-friendly copy
    metrics.to_csv(
        FIGURES_DIR / "model_comparison.csv",
        index=False
    )

    # MAE
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["MAE"])
    plt.ylabel("MAE")
    plt.title("Model Comparison — MAE (Lower is Better)")
    save_figure("model_comparison_mae.png")

    # RMSE
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Model Comparison — RMSE (Lower is Better)")
    save_figure("model_comparison_rmse.png")

    # R2
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["R2"])
    plt.ylabel("R²")
    plt.title("Model Comparison — R² (Higher is Better)")
    plt.ylim(
        max(0, metrics["R2"].min() - 0.02),
        min(1, metrics["R2"].max() + 0.01)
    )
    save_figure("model_comparison_r2.png")

    # MAPE
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["MAPE"])
    plt.ylabel("MAPE (%)")
    plt.title("Model Comparison — MAPE (Lower is Better)")
    save_figure("model_comparison_mape.png")

    # Combined normalized comparison figure as one chart
    normalized = metrics[["MAE", "RMSE", "R2", "MAPE"]].copy()

    # Lower-is-better metrics are inverted relative to the best value.
    for col in ["MAE", "RMSE", "MAPE"]:
        best = normalized[col].min()
        normalized[col] = best / normalized[col]

    best_r2 = normalized["R2"].max()
    normalized["R2"] = normalized["R2"] / best_r2

    x = np.arange(len(metrics))
    width = 0.18

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(["MAE", "RMSE", "R2", "MAPE"]):
        plt.bar(
            x + (i - 1.5) * width,
            normalized[col],
            width,
            label=col
        )

    plt.xticks(x, metrics["model"])
    plt.ylabel("Relative Score (Higher is Better)")
    plt.title("Normalized Model Performance Comparison")
    plt.legend()
    save_figure("model_comparison_normalized.png")

    print("\nModel comparison:")
    print(metrics[required].to_string(index=False))

# ============================================================
# 2. ACTUAL VS HYBRID PREDICTED
# ============================================================
def actual_vs_predicted():
    require_file(HYBRID_TEST)

    df = pd.read_csv(HYBRID_TEST)

    required = [
        "timestamp",
        "actual",
        "hybrid_prediction",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {HYBRID_TEST}: {missing}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Full test period
    plt.figure(figsize=(14, 6))
    plt.plot(
        df["timestamp"],
        df["actual"],
        label="Actual"
    )
    plt.plot(
        df["timestamp"],
        df["hybrid_prediction"],
        label="Hybrid Prediction"
    )
    plt.xlabel("Time")
    plt.ylabel("Solar Output")
    plt.title("Actual vs Hybrid Predicted Solar Output — Test Set")
    plt.legend()
    save_figure("actual_vs_hybrid_full_test.png")

    # First 7 days for clearer presentation
    start = df["timestamp"].min()
    end = start + pd.Timedelta(days=7)
    subset = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] < end)
    ]

    plt.figure(figsize=(14, 6))
    plt.plot(
        subset["timestamp"],
        subset["actual"],
        label="Actual"
    )
    plt.plot(
        subset["timestamp"],
        subset["hybrid_prediction"],
        label="Hybrid Prediction"
    )
    plt.xlabel("Time")
    plt.ylabel("Solar Output")
    plt.title("Actual vs Hybrid Prediction — First 7 Test Days")
    plt.legend()
    save_figure("actual_vs_hybrid_first_7_days.png")

# ============================================================
# 3. PREDICTION ERROR
# ============================================================
def prediction_error():
    require_file(HYBRID_TEST)

    df = pd.read_csv(HYBRID_TEST)

    required = [
        "timestamp",
        "actual",
        "hybrid_prediction",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {HYBRID_TEST}: {missing}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["error"] = (
        df["actual"] - df["hybrid_prediction"]
    )

    # Error over time
    plt.figure(figsize=(14, 6))
    plt.axhline(0, linewidth=1)
    plt.plot(
        df["timestamp"],
        df["error"]
    )
    plt.xlabel("Time")
    plt.ylabel("Prediction Error (Actual - Predicted)")
    plt.title("Hybrid Prediction Error — Test Set")
    save_figure("hybrid_prediction_error.png")

    # Error distribution
    plt.figure(figsize=(9, 6))
    plt.hist(df["error"], bins=40)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Hybrid Prediction Error Distribution")
    save_figure("hybrid_error_distribution.png")

# ============================================================
# 3B. ACTUAL VS PREDICTED SCATTER
# ============================================================
def actual_vs_predicted_scatter():
    require_file(HYBRID_TEST)

    df = pd.read_csv(HYBRID_TEST)

    required = [
        "actual",
        "hybrid_prediction",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {HYBRID_TEST}: {missing}"
        )

    actual = df["actual"].astype(float)
    predicted = df["hybrid_prediction"].astype(float)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.5,
        s=18
    )

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Actual Solar Output")
    plt.ylabel("Hybrid Predicted Solar Output")
    plt.title("Actual vs Hybrid Predicted Solar Output")

    save_figure("actual_vs_hybrid_scatter.png")

# ============================================================
# 4. XGBOOST FEATURE IMPORTANCE
# ============================================================
def feature_importance():
    require_file(XGB_IMPORTANCE)

    df = pd.read_csv(XGB_IMPORTANCE)

    required = ["feature", "importance"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {XGB_IMPORTANCE}: {missing}"
        )

    df = df.sort_values(
        "importance",
        ascending=True
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        df["feature"],
        df["importance"]
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance")
    save_figure("xgboost_feature_importance.png")

    print("\nXGBoost feature importance:")
    print(
        df.sort_values(
            "importance",
            ascending=False
        ).to_string(index=False)
    )

# ============================================================
# 5. TCN TRAINING HISTORY
# ============================================================
def tcn_training_history():
    require_file(TCN_HISTORY)

    df = pd.read_csv(TCN_HISTORY)

    if "loss" not in df.columns:
        raise ValueError(
            f"'loss' column not found in {TCN_HISTORY}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        df.index + 1,
        df["loss"],
        label="Training Loss"
    )

    if "val_loss" in df.columns:
        plt.plot(
            df.index + 1,
            df["val_loss"],
            label="Validation Loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TCN Training and Validation Loss")
    plt.legend()
    save_figure("tcn_training_history.png")

    print(
        f"\nTCN epochs recorded: {len(df)}"
    )

# ============================================================
# 6. HYBRID ENSEMBLE WEIGHTS
# ============================================================
def hybrid_weights():
    metrics_file = HYBRID_DIR / "metrics.json"
    require_file(metrics_file)

    import json

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    xgb_weight = float(data["xgboost_weight"])
    tcn_weight = float(data["tcn_weight"])

    labels = ["XGBoost", "TCN"]
    weights = [xgb_weight, tcn_weight]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, weights)

    plt.ylabel("Weight")
    plt.title("Hybrid Ensemble Model Weights")
    plt.ylim(0, 1)

    save_figure("hybrid_ensemble_weights.png")

    print("\nHybrid Ensemble weights:")
    print(f"XGBoost: {xgb_weight:.2f}")
    print(f"TCN:     {tcn_weight:.2f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print()
    print("=" * 70)
    print("SOLAR MODEL EVALUATION VISUALIZATIONS")
    print("=" * 70)

    print("\n[1/5] Model comparison...")
    model_comparison()

    print("\n[2/5] Actual vs Hybrid prediction...")
    actual_vs_predicted()

    print("\n[3/5] Prediction error...")
    prediction_error()

    print("\n[3B/6] Actual vs Hybrid scatter...")
    actual_vs_predicted_scatter()

    print("\n[4/5] XGBoost feature importance...")
    feature_importance()

    print("\n[5/5] TCN training history...")
    tcn_training_history()

    print()
    print("=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Figures directory:\n{FIGURES_DIR}")

if __name__ == "__main__":
    main()
