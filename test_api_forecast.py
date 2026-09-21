import json

import pandas as pd
import requests


# ============================================================
# API
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/api/forecast"
)


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    "data/solar_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# SIMULATE FORECAST
# ============================================================
#
# Use:
#
# 24 hours historical data
# +
# next 6 hours as future weather inputs
#
# The solar_output column is deliberately NOT
# sent to the API.
#
# It is retained locally only so we can compare
# the model prediction against the known dataset.
# ============================================================

forecast_hours = 6

history_end = (
    len(df) - forecast_hours
)

historical = df.iloc[
    history_end - 24:
    history_end
].copy()

future = df.iloc[
    history_end:
    history_end + forecast_hours
].copy()


# ============================================================
# BUILD OBSERVATIONS
# ============================================================

def build_observation(row):

    return {
        "timestamp":
            row["timestamp"].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "temperature":
            float(row["temperature"]),

        "humidity":
            float(row["humidity"]),

        "cloud_cover":
            float(row["cloud_cover"]),

        "wind_speed":
            float(row["wind_speed"]),

        "irradiance":
            float(row["irradiance"]),
    }


historical_observations = [
    build_observation(row)
    for _, row in historical.iterrows()
]

future_observations = [
    build_observation(row)
    for _, row in future.iterrows()
]


# ============================================================
# REQUEST
# ============================================================

payload = {

    "historical_observations":
        historical_observations,

    "future_observations":
        future_observations,
}


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("API FORECAST TEST")
print("=" * 70)

print()

print(
    f"Historical observations: "
    f"{len(historical_observations)}"
)

print(
    f"Future observations: "
    f"{len(future_observations)}"
)

print()

print(
    "Forecast period:"
)

print(
    f"  {future['timestamp'].iloc[0]}"
    f" → "
    f"{future['timestamp'].iloc[-1]}"
)


# ============================================================
# SEND REQUEST
# ============================================================

response = requests.post(
    API_URL,
    json=payload,
    timeout=120,
)


# ============================================================
# RESPONSE
# ============================================================

print()

print(
    f"HTTP Status: "
    f"{response.status_code}"
)

print()

if not response.ok:

    print(
        "API ERROR:"
    )

    print(
        response.text
    )

else:

    result = response.json()

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    # --------------------------------------------------------
    # Compare against known dataset values
    # --------------------------------------------------------

    predictions = (
        result["predictions"]
    )

    print()
    print("=" * 70)
    print("FORECAST VS ACTUAL")
    print("=" * 70)

    print()

    for i, prediction in enumerate(
        predictions
    ):

        actual = float(
            future[
                "solar_output"
            ].iloc[i]
        )

        print(
            f"{prediction['timestamp']} | "
            f"Actual: {actual:8.3f} | "
            f"Hybrid: "
            f"{prediction['hybrid_prediction']:8.3f}"
        )


print()

print("=" * 70)
print("API FORECAST TEST COMPLETE")
print("=" * 70)