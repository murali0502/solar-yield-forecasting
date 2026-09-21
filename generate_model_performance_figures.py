import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SOLAR YIELD FORECASTING
# Generate individual model forecasting performance figures
# XGBoost vs Actual
# TCN vs Actual
# Hybrid Ensemble vs Actual
# ============================================================

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_FILE = os.path.join(
    BASE_DIR, "results", "xgboost", "test_predictions.csv"
)

TCN_FILE = os.path.join(
    BASE_DIR, "results", "tcn", "test_predictions.csv"
)

HYBRID_FILE = os.path.join(
    BASE_DIR, "results", "hybrid", "test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD EXISTING PREDICTIONS
# ============================================================

print("Loading existing prediction files...")

xgb = pd.read_csv(XGB_FILE)
tcn = pd.read_csv(TCN_FILE)
hybrid = pd.read_csv(HYBRID_FILE)

# Convert timestamp
xgb["timestamp"] = pd.to_datetime(xgb["timestamp"])
tcn["timestamp"] = pd.to_datetime(tcn["timestamp"])
hybrid["timestamp"] = pd.to_datetime(hybrid["timestamp"])


# ============================================================
# USE COMMON 7-DAY TEST PERIOD
# ============================================================

common_start = max(
    xgb["timestamp"].min(),
    tcn["timestamp"].min(),
    hybrid["timestamp"].min()
)

common_end = min(
    xgb["timestamp"].max(),
    tcn["timestamp"].max(),
    hybrid["timestamp"].max()
)

print(f"Common test period: {common_start} -> {common_end}")

# First 7 days of the common test period
plot_end = common_start + pd.Timedelta(days=7)

xgb_plot = xgb[
    (xgb["timestamp"] >= common_start) &
    (xgb["timestamp"] < plot_end)
].copy()

tcn_plot = tcn[
    (tcn["timestamp"] >= common_start) &
    (tcn["timestamp"] < plot_end)
].copy()

hybrid_plot = hybrid[
    (hybrid["timestamp"] >= common_start) &
    (hybrid["timestamp"] < plot_end)
].copy()


# ============================================================
# GENERAL PLOT FUNCTION
# ============================================================

def create_performance_plot(
    data,
    prediction_column,
    model_name,
    output_filename
):

    plt.figure(figsize=(13, 6))

    plt.plot(
        data["timestamp"],
        data["actual"],
        label="Actual Solar Output",
        linewidth=2
    )

    plt.plot(
        data["timestamp"],
        data[prediction_column],
        label=f"{model_name} Prediction",
        linewidth=1.7
    )

    plt.xlabel("Time")
    plt.ylabel("Solar Output (W)")

    plt.title(
        f"{model_name} Actual vs Predicted Solar Output"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.25
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Generated: {output_path}")


# ============================================================
# FIGURE 4.1 — XGBOOST
# ============================================================

create_performance_plot(
    xgb_plot,
    "prediction",
    "XGBoost",
    "xgboost_forecasting_performance.png"
)


# ============================================================
# FIGURE 4.2 — TCN
# ============================================================

create_performance_plot(
    tcn_plot,
    "prediction",
    "TCN",
    "tcn_forecasting_performance.png"
)


# ============================================================
# FIGURE 4.3 — HYBRID ENSEMBLE
# ============================================================

create_performance_plot(
    hybrid_plot,
    "hybrid_prediction",
    "Hybrid Ensemble",
    "hybrid_forecasting_performance.png"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL PERFORMANCE FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)

print()
print("Files created:")
print("1. results/figures/xgboost_forecasting_performance.png")
print("2. results/figures/tcn_forecasting_performance.png")
print("3. results/figures/hybrid_forecasting_performance.png")

print()
print("No model retraining was performed.")
print("Existing test predictions were used.")