import json

import pandas as pd
import requests


# ============================================================
# API
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/api/predict"
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

# Use 24 consecutive observations
# from the end of the dataset.

history = df.tail(
    24
).copy()


# ============================================================
# BUILD REQUEST
# ============================================================

observations = []

for _, row in history.iterrows():

    observations.append({

        "timestamp":
            str(row["timestamp"]),

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
    })


payload = {
    "observations": observations
}


# ============================================================
# SEND REQUEST
# ============================================================

print()
print("=" * 70)
print("API PREDICTION TEST")
print("=" * 70)

print()

print(
    f"Sending {len(observations)} "
    "hourly observations..."
)

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

if response.ok:

    result = response.json()

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

else:

    print(
        "API ERROR:"
    )

    print(
        response.text
    )


print()

print("=" * 70)
print("API PREDICTION TEST COMPLETE")
print("=" * 70)