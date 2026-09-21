# ================================================================
# SOLAR YIELD FORECAST API - ROUTES
# ================================================================
# Architecture:
#     XGBoost + TCN + Hybrid Ensemble
#
# TCN:
#     24-hour lookback
#     9 input features
#     Sequence-to-one next-hour prediction
#
# LSTM:
#     NOT USED
# ================================================================

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ForecastRequest,
    ForecastResponse,
    DayForecastRequest,
    DayForecastResponse,
)

from src.inference.predictor import SolarPredictor


# ================================================================
# CONSTANTS
# ================================================================

ARCHITECTURE = "XGBoost + TCN + Hybrid"
DEEP_LEARNING_MODEL = "TCN"
LSTM_USED = False
TCN_LOOKBACK_HOURS = 24

MODEL_FEATURES = [
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "irradiance",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


# ================================================================
# ROUTER
# ================================================================

# api/main.py already applies /api.
router = APIRouter(
    tags=["Solar Forecast"]
)


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "solar_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


# ================================================================
# LOAD PREDICTOR ONCE
# ================================================================

print()
print("=" * 70)
print("INITIALIZING SOLAR FORECAST API")
print("=" * 70)

model = SolarPredictor()

print("Solar predictor initialized.")
print("=" * 70)
print()


# ================================================================
# GET PREDICTOR
# ================================================================

def get_predictor():
    """
    Return the globally loaded SolarPredictor.
    """
    return model


# ================================================================
# VALIDATE NUMBER
# ================================================================

def validate_number(
    value,
    name,
):
    """
    Convert input to float and verify that it is finite.
    """

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            f"{name} must be numeric."
        )

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


# ================================================================
# CLEAN PREDICTION
# ================================================================

def clean_prediction(
    value,
):
    """
    Clean a model prediction.

    Solar output cannot be negative.
    """

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return max(
        0.0,
        value,
    )


# ================================================================
# NORMALIZE CLOUD COVER
# ================================================================

def normalize_cloud_cover(
    cloud_cover,
):
    """
    Convert API cloud cover percentage to
    model fraction.

    Example:
        0   -> 0.0
        30  -> 0.3
        100 -> 1.0
    """

    cloud_cover = validate_number(
        cloud_cover,
        "cloud_cover",
    )

    if not 0 <= cloud_cover <= 100:
        raise ValueError(
            "cloud_cover must be between 0 and 100."
        )

    return cloud_cover / 100.0


# ================================================================
# NORMALIZE DAY OF YEAR
# ================================================================

def normalize_day_of_year(
    day_of_year,
):
    """
    Validate day of year.
    """

    try:
        day_of_year = int(day_of_year)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "day_of_year must be an integer."
        )

    if not 1 <= day_of_year <= 366:
        raise ValueError(
            "day_of_year must be between 1 and 366."
        )

    return day_of_year


# ================================================================
# PREVIOUS DAY
# ================================================================

def previous_day(
    day_of_year,
):
    """
    Return previous day of year.
    """

    if day_of_year <= 1:
        return 365

    return day_of_year - 1


# ================================================================
# CLEAR SKY IRRADIANCE
# ================================================================

def clear_sky_irradiance(
    hour,
    day_of_year,
):
    """
    Estimate clear-sky irradiance.

    Used for daily forecasting when future
    irradiance is not supplied directly.
    """

    hour_angle = (
        (float(hour) - 12.5)
        * (math.pi / 12.0)
    )

    daylight = max(
        0.0,
        math.cos(hour_angle),
    )

    seasonal_factor = (
        0.75
        +
        0.25
        *
        math.cos(
            (
                (
                    float(day_of_year)
                    - 172.0
                )
                / 365.0
            )
            *
            2.0
            *
            math.pi
        )
    )

    irradiance = (
        1000.0
        *
        (daylight ** 1.3)
        *
        seasonal_factor
    )

    return max(
        0.0,
        float(irradiance),
    )


# ================================================================
# ESTIMATE IRRADIANCE
# ================================================================

def estimate_irradiance(
    hour,
    day_of_year,
    cloud_cover_fraction,
):
    """
    Estimate irradiance from clear-sky irradiance
    and cloud-cover attenuation.
    """

    clear_sky = clear_sky_irradiance(
        hour,
        day_of_year,
    )

    attenuation = (
        1.0
        -
        0.75
        *
        cloud_cover_fraction
    )

    attenuation = max(
        0.0,
        min(
            1.0,
            attenuation,
        ),
    )

    irradiance = (
        clear_sky
        *
        attenuation
    )

    return max(
        0.0,
        float(irradiance),
    )


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def engineer_features(
    temperature,
    humidity,
    cloud_cover,
    wind_speed,
    irradiance,
    hour,
    day_of_year,
):
    """
    Build exactly the 9 features used during training.
    """

    hour_sin = math.sin(
        2.0
        * math.pi
        * float(hour)
        / 24.0
    )

    hour_cos = math.cos(
        2.0
        * math.pi
        * float(hour)
        / 24.0
    )

    day_sin = math.sin(
        2.0
        * math.pi
        * float(day_of_year)
        / 365.25
    )

    day_cos = math.cos(
        2.0
        * math.pi
        * float(day_of_year)
        / 365.25
    )

    return {
        "temperature": float(
            temperature
        ),

        "humidity": float(
            humidity
        ),

        "cloud_cover": float(
            cloud_cover
        ),

        "wind_speed": float(
            wind_speed
        ),

        "irradiance": float(
            irradiance
        ),

        "hour_sin": float(
            hour_sin
        ),

        "hour_cos": float(
            hour_cos
        ),

        "day_sin": float(
            day_sin
        ),

        "day_cos": float(
            day_cos
        ),
    }


# ================================================================
# MODEL FEATURES
# ================================================================

def get_model_features():
    """
    Return feature lists.

    Both models were trained with 9 features.
    """

    xgb_features = list(
        getattr(
            model,
            "xgb_features",
            MODEL_FEATURES,
        )
    )

    tcn_features = list(
        getattr(
            model,
            "tcn_features",
            MODEL_FEATURES,
        )
    )

    if len(xgb_features) != 9:
        raise RuntimeError(
            "XGBoost feature count mismatch. "
            f"Expected 9, got {len(xgb_features)}."
        )

    if len(tcn_features) != 9:
        raise RuntimeError(
            "TCN feature count mismatch. "
            f"Expected 9, got {len(tcn_features)}."
        )

    return (
        xgb_features,
        tcn_features,
    )


# ================================================================
# LOOKBACK
# ================================================================

def get_lookback():
    """
    Return the trained TCN lookback.
    """

    return TCN_LOOKBACK_HOURS


# ================================================================
# HEALTH
# ================================================================

@router.get(
    "/health"
)
def health():
    """
    API health endpoint.
    """

    xgb_features, tcn_features = (
        get_model_features()
    )

    return {
        "status": "healthy",

        "service": (
            "Solar Yield Forecast API"
        ),

        "models_loaded": True,

        "architecture": ARCHITECTURE,

        "deep_learning_model": (
            DEEP_LEARNING_MODEL
        ),

        "lstm_used": LSTM_USED,

        "lookback_hours": (
            get_lookback()
        ),

        "xgboost_features": len(
            xgb_features
        ),

        "tcn_features": len(
            tcn_features
        ),
    }


# ================================================================
# MODEL INFO
# ================================================================

@router.get(
    "/model-info"
)
def model_info():
    """
    Return model architecture information.
    """

    return {
        "architecture": ARCHITECTURE,

        "models": [
            "XGBoost",
            "Temporal Convolutional Network",
            "Hybrid Ensemble",
        ],

        "deep_learning_model": (
            DEEP_LEARNING_MODEL
        ),

        "lstm_used": LSTM_USED,

        "lookback_hours": (
            get_lookback()
        ),

        "input_features": 9,

        "feature_names": list(
            MODEL_FEATURES
        ),

        "prediction_type": (
            "next-hour sequence-to-one"
        ),

        "target_definition": (
            "previous 24 hours -> next hour"
        ),

        "hybrid_method": (
            "weighted ensemble"
        ),

        "xgboost_weight": round(
            float(model.xgb_weight),
            6,
        ),

        "tcn_weight": round(
            float(model.tcn_weight),
            6,
        ),
    }

# ================================================================
# METRICS
# ================================================================

@router.get(
    "/metrics"
)
def metrics():
    """
    Return verified test-set evaluation metrics for
    XGBoost, TCN, and Hybrid models.

    Metrics are loaded directly from the project's
    evaluation artifact:

        results/figures/model_comparison.csv

    Validation metrics and hybrid weights are loaded from:

        results/hybrid/metrics.json

    The response format is aligned with the React frontend.
    """

    try:
        # --------------------------------------------------------
        # RESULTS PATHS
        # --------------------------------------------------------

        comparison_path = (
            PROJECT_ROOT
            / "results"
            / "figures"
            / "model_comparison.csv"
        )

        hybrid_metrics_path = (
            PROJECT_ROOT
            / "results"
            / "hybrid"
            / "metrics.json"
        )

        # --------------------------------------------------------
        # VERIFY FILES
        # --------------------------------------------------------

        if not comparison_path.exists():
            raise FileNotFoundError(
                "Model comparison file not found: "
                f"{comparison_path}"
            )

        # --------------------------------------------------------
        # LOAD MODEL COMPARISON
        # --------------------------------------------------------

        comparison_df = pd.read_csv(
            comparison_path
        )

        required_columns = {
            "model",
            "MAE",
            "RMSE",
            "R2",
        }

        missing_columns = (
            required_columns
            -
            set(comparison_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Model comparison file is missing "
                f"required columns: {sorted(missing_columns)}"
            )

        # --------------------------------------------------------
        # NORMALIZE MODEL NAMES
        # --------------------------------------------------------

        comparison_df["model_normalized"] = (
            comparison_df["model"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # --------------------------------------------------------
        # FIND MODEL ROW
        # --------------------------------------------------------

        def get_model_metrics(
            model_name,
        ):
            model_key = (
                model_name
                .strip()
                .lower()
            )

            rows = comparison_df[
                comparison_df[
                    "model_normalized"
                ]
                == model_key
            ]

            if rows.empty:
                raise ValueError(
                    f"Metrics for {model_name} "
                    "were not found in model_comparison.csv."
                )

            row = rows.iloc[0]

            mae = float(
                row["MAE"]
            )

            rmse = float(
                row["RMSE"]
            )

            r2 = float(
                row["R2"]
            )

            # Optional metric if present.
            mape = None

            if "MAPE" in comparison_df.columns:
                mape = float(
                    row["MAPE"]
                )

            result = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }

            if mape is not None:
                result["mape"] = mape

            return result

        # --------------------------------------------------------
        # LOAD VERIFIED TEST METRICS
        # --------------------------------------------------------

        xgboost_metrics = (
            get_model_metrics(
                "XGBoost"
            )
        )

        tcn_metrics = (
            get_model_metrics(
                "TCN"
            )
        )

        hybrid_metrics = (
            get_model_metrics(
                "Hybrid"
            )
        )

        # --------------------------------------------------------
        # LOAD VALIDATION + HYBRID WEIGHTS
        # --------------------------------------------------------

        validation_metrics = None
        tcn_weight = float(
            model.tcn_weight
        )
        xgboost_weight = float(
            model.xgb_weight
        )

        if hybrid_metrics_path.exists():

            with open(
                hybrid_metrics_path,
                "r",
                encoding="utf-8",
            ) as file:
                hybrid_data = json.load(
                    file
                )

            validation_data = (
                hybrid_data.get(
                    "validation"
                )
            )

            if validation_data:
                validation_metrics = {
                    "mae": float(
                        validation_data[
                            "MAE"
                        ]
                    ),
                    "rmse": float(
                        validation_data[
                            "RMSE"
                        ]
                    ),
                    "r2": float(
                        validation_data[
                            "R2"
                        ]
                    ),
                }

            if (
                "tcn_weight"
                in hybrid_data
            ):
                tcn_weight = float(
                    hybrid_data[
                        "tcn_weight"
                    ]
                )

            if (
                "xgboost_weight"
                in hybrid_data
            ):
                xgboost_weight = float(
                    hybrid_data[
                        "xgboost_weight"
                    ]
                )

        # --------------------------------------------------------
        # RESPONSE
        #
        # IMPORTANT:
        # This structure matches App.tsx:
        #
        # data.test.xgboost.mae
        # data.test.tcn.mae
        # data.test.hybrid.mae
        # --------------------------------------------------------

        response = {
            "architecture": ARCHITECTURE,

            "deep_learning_model": (
                DEEP_LEARNING_MODEL
            ),

            "lstm_used": LSTM_USED,

            "lookback_hours": (
                get_lookback()
            ),

            "evaluation": "test",

            "test": {
                "xgboost": xgboost_metrics,
                "tcn": tcn_metrics,
                "hybrid": hybrid_metrics,
            },

            "validation": (
                validation_metrics
            ),

            "weights": {
                "xgboost": round(
                    xgboost_weight,
                    6,
                ),
                "tcn": round(
                    tcn_weight,
                    6,
                ),
            },

            "source": (
                "results/figures/"
                "model_comparison.csv"
            ),
        }

        return response

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Metrics loading failed: {exc}"
            ),
        )
    
# ================================================================
# SINGLE PREDICTION
# ================================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    """
    Generate a single next-hour prediction.

    IMPORTANT:
    Because TCN requires 24 observations, this
    endpoint uses the real dataset to construct
    the required historical context.
    """

    try:

        temperature = validate_number(
            request.temperature,
            "temperature",
        )

        humidity = validate_number(
            request.humidity,
            "humidity",
        )

        cloud_cover_fraction = (
            normalize_cloud_cover(
                request.cloud_cover
            )
        )

        wind_speed = validate_number(
            request.wind_speed,
            "wind_speed",
        )

        irradiance = validate_number(
            request.irradiance,
            "irradiance",
        )

        hour = int(
            request.hour
        )

        day_of_year = (
            normalize_day_of_year(
                request.day_of_year
            )
        )

        if not 0 <= hour <= 23:
            raise ValueError(
                "hour must be between 0 and 23."
            )

        if not 0 <= humidity <= 100:
            raise ValueError(
                "humidity must be between 0 and 100."
            )

        if wind_speed < 0:
            raise ValueError(
                "wind_speed cannot be negative."
            )

        if irradiance < 0:
            raise ValueError(
                "irradiance cannot be negative."
            )

        # --------------------------------------------------------
        # Build a reference timestamp.
        # --------------------------------------------------------

        reference_date = (
            pd.Timestamp("2023-01-01")
            +
            pd.Timedelta(
                days=day_of_year - 1
            )
        )

        timestamp = (
            reference_date
            +
            pd.Timedelta(
                hours=hour
            )
        )

        # --------------------------------------------------------
        # Load historical dataset.
        # --------------------------------------------------------

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        dataset = pd.read_csv(
            DATA_PATH
        )

        if len(dataset) < 24:
            raise ValueError(
                "Dataset does not contain enough "
                "historical observations."
            )

        # --------------------------------------------------------
        # Prepare historical data.
        #
        # The dataset already contains the original
        # environmental observations.
        # --------------------------------------------------------

        historical = dataset.copy()

        historical["timestamp"] = pd.to_datetime(
            historical["timestamp"]
        )

        historical = (
            historical
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Take latest 24 historical rows.
        #
        # Replace the final row with the user's
        # current conditions.
        # --------------------------------------------------------

        context = (
            historical
            .tail(24)
            .copy()
            .reset_index(drop=True)
        )

        context_timestamp = timestamp

        context.loc[
            23,
            "timestamp"
        ] = context_timestamp

        context.loc[
            23,
            "temperature"
        ] = temperature

        context.loc[
            23,
            "humidity"
        ] = humidity

        context.loc[
            23,
            "cloud_cover"
        ] = cloud_cover_fraction

        context.loc[
            23,
            "wind_speed"
        ] = wind_speed

        context.loc[
            23,
            "irradiance"
        ] = irradiance

        # --------------------------------------------------------
        # Direct inference.
        # --------------------------------------------------------

        prepared = model.prepare_features(
            context
        )

        xgb_prediction = (
            model.predict_xgboost(
                prepared
            )
        )

        tcn_prediction = (
            model.predict_tcn(
                prepared
            )
        )

        hybrid_prediction = (
            model.combine_predictions(
                xgb_prediction,
                tcn_prediction,
            )
        )

        forecast_timestamp = (
            timestamp
            +
            pd.Timedelta(
                hours=1
            )
        )

        return {
            "input_timestamp": (
                timestamp.isoformat()
            ),

            "forecast_timestamp": (
                forecast_timestamp.isoformat()
            ),

            "xgboost_prediction": round(
                clean_prediction(
                    xgb_prediction
                ),
                6,
            ),

            "tcn_prediction": round(
                clean_prediction(
                    tcn_prediction
                ),
                6,
            ),

            "hybrid_prediction": round(
                clean_prediction(
                    hybrid_prediction
                ),
                6,
            ),

            "architecture": ARCHITECTURE,

            "tcn_lookback_hours": (
                get_lookback()
            ),

            "lstm_used": LSTM_USED,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {exc}"
            ),
        )


# ================================================================
# MULTI-HOUR FORECAST
# ================================================================

@router.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
):
    """
    Generate rolling multi-hour forecasts.

    The supplied history must contain at least
    24 observations before the first prediction.
    """

    try:

        if not request.history:
            raise ValueError(
                "At least one historical observation is required."
            )

        if not request.future:
            raise ValueError(
                "At least one future observation is required."
            )

        # --------------------------------------------------------
        # Build historical rows.
        # --------------------------------------------------------

        history_rows = []

        for item in request.history:

            timestamp = pd.Timestamp(
                item.timestamp
            )

            temperature = validate_number(
                item.temperature,
                "temperature",
            )

            humidity = validate_number(
                item.humidity,
                "humidity",
            )

            cloud_cover = (
                normalize_cloud_cover(
                    item.cloud_cover
                )
            )

            wind_speed = validate_number(
                item.wind_speed,
                "wind_speed",
            )

            irradiance = validate_number(
                item.irradiance,
                "irradiance",
            )

            if not 0 <= humidity <= 100:
                raise ValueError(
                    "humidity must be between 0 and 100."
                )

            if wind_speed < 0:
                raise ValueError(
                    "wind_speed cannot be negative."
                )

            if irradiance < 0:
                raise ValueError(
                    "irradiance cannot be negative."
                )

            feature = engineer_features(
                temperature=temperature,
                humidity=humidity,
                cloud_cover=cloud_cover,
                wind_speed=wind_speed,
                irradiance=irradiance,
                hour=timestamp.hour,
                day_of_year=timestamp.dayofyear,
            )

            feature["timestamp"] = timestamp

            history_rows.append(
                feature
            )

        working_df = pd.DataFrame(
            history_rows
        )

        working_df = (
            working_df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # TCN requires 24 observations.
        # --------------------------------------------------------

        if len(working_df) < get_lookback():
            raise ValueError(
                "At least 24 historical observations "
                "are required before the first forecast."
            )

        # --------------------------------------------------------
        # Future predictions.
        # --------------------------------------------------------

        predictions = []

        for item in request.future:

            timestamp = pd.Timestamp(
                item.timestamp
            )

            temperature = validate_number(
                item.temperature,
                "temperature",
            )

            humidity = validate_number(
                item.humidity,
                "humidity",
            )

            cloud_cover = (
                normalize_cloud_cover(
                    item.cloud_cover
                )
            )

            wind_speed = validate_number(
                item.wind_speed,
                "wind_speed",
            )

            irradiance = validate_number(
                item.irradiance,
                "irradiance",
            )

            if not 0 <= humidity <= 100:
                raise ValueError(
                    "humidity must be between 0 and 100."
                )

            if wind_speed < 0:
                raise ValueError(
                    "wind_speed cannot be negative."
                )

            if irradiance < 0:
                raise ValueError(
                    "irradiance cannot be negative."
                )

            feature = engineer_features(
                temperature=temperature,
                humidity=humidity,
                cloud_cover=cloud_cover,
                wind_speed=wind_speed,
                irradiance=irradiance,
                hour=timestamp.hour,
                day_of_year=timestamp.dayofyear,
            )

            feature["timestamp"] = timestamp

            future_row = pd.DataFrame(
                [feature]
            )

            # ----------------------------------------------------
            # Add future row to rolling context.
            # ----------------------------------------------------

            working_df = pd.concat(
                [
                    working_df,
                    future_row,
                ],
                ignore_index=True,
            )

            context = (
                working_df
                .tail(
                    get_lookback()
                )
                .copy()
            )

            # ----------------------------------------------------
            # Direct predictor.
            # ----------------------------------------------------

            prepared = model.prepare_features(
                context
            )

            xgb_value = (
                model.predict_xgboost(
                    prepared
                )
            )

            tcn_value = (
                model.predict_tcn(
                    prepared
                )
            )

            hybrid_value = (
                model.combine_predictions(
                    xgb_value,
                    tcn_value,
                )
            )

            predictions.append(
                {
                    "timestamp": (
                        (
                            timestamp
                            +
                            pd.Timedelta(
                                hours=1
                            )
                        )
                        .isoformat()
                    ),

                    "xgboost": round(
                        clean_prediction(
                            xgb_value
                        ),
                        6,
                    ),

                    "tcn": round(
                        clean_prediction(
                            tcn_value
                        ),
                        6,
                    ),

                    "hybrid": round(
                        clean_prediction(
                            hybrid_value
                        ),
                        6,
                    ),
                }
            )

        return {
            "architecture": ARCHITECTURE,

            "lstm_used": LSTM_USED,

            "lookback_hours": (
                get_lookback()
            ),

            "predictions": predictions,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Forecast failed: {exc}"
            ),
        )


# ================================================================
# BUILD TARGET DAY
# ================================================================

def build_day_dataframe(
    day_of_year,
    temperature,
    humidity,
    cloud_cover_fraction,
    wind_speed,
):
    """
    Construct 24 hourly target-day rows.
    """

    rows = []

    irradiance_values = []

    reference_year = 2023

    base_date = (
        pd.Timestamp(
            f"{reference_year}-01-01"
        )
        +
        pd.Timedelta(
            days=day_of_year - 1
        )
    )

    for hour in range(24):

        timestamp = (
            base_date
            +
            pd.Timedelta(
                hours=hour
            )
        )

        irradiance = (
            estimate_irradiance(
                hour=hour,
                day_of_year=day_of_year,
                cloud_cover_fraction=(
                    cloud_cover_fraction
                ),
            )
        )

        feature = engineer_features(
            temperature=temperature,
            humidity=humidity,
            cloud_cover=cloud_cover_fraction,
            wind_speed=wind_speed,
            irradiance=irradiance,
            hour=hour,
            day_of_year=day_of_year,
        )

        feature["timestamp"] = timestamp

        rows.append(
            feature
        )

        irradiance_values.append(
            irradiance
        )

    return (
        pd.DataFrame(rows),
        irradiance_values,
    )


# ================================================================
# BUILD 48-HOUR TCN CONTEXT
# ================================================================

def build_day_context(
    day_of_year,
    temperature,
    humidity,
    cloud_cover_fraction,
    wind_speed,
):
    """
    Build 48 chronological rows:

        previous day: 00-23
        target day:   00-23

    This gives every target hour a full
    24-hour TCN lookback.
    """

    previous = previous_day(
        day_of_year
    )

    rows = []

    reference_year = 2023

    previous_base_date = (
        pd.Timestamp(
            f"{reference_year}-01-01"
        )
        +
        pd.Timedelta(
            days=previous - 1
        )
    )

    target_base_date = (
        pd.Timestamp(
            f"{reference_year}-01-01"
        )
        +
        pd.Timedelta(
            days=day_of_year - 1
        )
    )

    # ------------------------------------------------------------
    # Previous day
    # ------------------------------------------------------------

    for hour in range(24):

        timestamp = (
            previous_base_date
            +
            pd.Timedelta(
                hours=hour
            )
        )

        irradiance = (
            estimate_irradiance(
                hour=hour,
                day_of_year=previous,
                cloud_cover_fraction=(
                    cloud_cover_fraction
                ),
            )
        )

        feature = engineer_features(
            temperature=temperature,
            humidity=humidity,
            cloud_cover=cloud_cover_fraction,
            wind_speed=wind_speed,
            irradiance=irradiance,
            hour=hour,
            day_of_year=previous,
        )

        feature["timestamp"] = timestamp

        rows.append(
            feature
        )

    # ------------------------------------------------------------
    # Target day
    # ------------------------------------------------------------

    for hour in range(24):

        timestamp = (
            target_base_date
            +
            pd.Timedelta(
                hours=hour
            )
        )

        irradiance = (
            estimate_irradiance(
                hour=hour,
                day_of_year=day_of_year,
                cloud_cover_fraction=(
                    cloud_cover_fraction
                ),
            )
        )

        feature = engineer_features(
            temperature=temperature,
            humidity=humidity,
            cloud_cover=cloud_cover_fraction,
            wind_speed=wind_speed,
            irradiance=irradiance,
            hour=hour,
            day_of_year=day_of_year,
        )

        feature["timestamp"] = timestamp

        rows.append(
            feature
        )

    context_df = pd.DataFrame(
        rows
    )

    return (
        context_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ================================================================
# DAILY TCN PREDICTION
# ================================================================

def predict_tcn_day(
    context_df,
):
    """
    Generate 24 TCN predictions.

    Each prediction uses:

        24 previous observations
                  ↓
                TCN
                  ↓
             next-hour output
    """

    _, tcn_features = (
        get_model_features()
    )

    lookback = get_lookback()

    if lookback != 24:
        raise RuntimeError(
            "TCN daily prediction requires "
            "a 24-hour lookback."
        )

    required_rows = (
        lookback + 23
    )

    if len(context_df) < required_rows:
        raise ValueError(
            f"TCN daily context requires at least "
            f"{required_rows} rows. "
            f"Received: {len(context_df)}."
        )

    # ------------------------------------------------------------
    # Extract exact model features.
    # ------------------------------------------------------------

    feature_matrix = (
        context_df[
            tcn_features
        ]
        .astype(np.float32)
    )

    # ------------------------------------------------------------
    # Scale using training scaler.
    # ------------------------------------------------------------

    scaled = (
        model.feature_scaler
        .transform(
            feature_matrix
        )
        .astype(np.float32)
    )

    predictions = []

    # ------------------------------------------------------------
    # Generate 24 predictions.
    # ------------------------------------------------------------

    for hour in range(24):

        start = hour

        end = (
            start
            +
            lookback
        )

        sequence = scaled[
            start:end
        ]

        expected_shape = (
            lookback,
            len(tcn_features),
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Invalid TCN sequence shape at hour "
                f"{hour}: {sequence.shape}; "
                f"expected {expected_shape}."
            )

        model_input = (
            sequence
            .reshape(
                1,
                lookback,
                len(tcn_features),
            )
            .astype(np.float32)
        )

        # --------------------------------------------------------
        # TCN prediction.
        # --------------------------------------------------------

        scaled_prediction = (
            model.tcn_model
            .predict(
                model_input,
                verbose=0,
            )
        )

        scaled_prediction = float(
            np.asarray(
                scaled_prediction
            )
            .reshape(-1)[0]
        )

        # --------------------------------------------------------
        # Inverse target scaling.
        # --------------------------------------------------------

        prediction = (
            model.target_scaler
            .inverse_transform(
                np.array(
                    [
                        [
                            scaled_prediction
                        ]
                    ],
                    dtype=np.float32,
                )
            )
            .reshape(-1)[0]
        )

        predictions.append(
            clean_prediction(
                prediction
            )
        )

    return np.asarray(
        predictions,
        dtype=np.float64,
    )


# ================================================================
# 24-HOUR DAY FORECAST
# ================================================================

@router.post(
    "/predict_day",
    response_model=DayForecastResponse,
)
def predict_day(
    request: DayForecastRequest,
):
    """
    Generate a complete 24-hour solar yield forecast.

    Architecture:

        XGBoost → 24 hourly power predictions (W)
        TCN     → 24 hourly power predictions (W)
        Hybrid  → weighted combination (W)

    Daily energy totals are calculated as:

        sum(hourly W) / 1000 = daily kWh
    """

    try:

        # ========================================================
        # VALIDATE INPUT
        # ========================================================

        day_of_year = (
            normalize_day_of_year(
                request.day_of_year
            )
        )

        temperature = validate_number(
            request.temperature,
            "temperature",
        )

        humidity = validate_number(
            request.humidity,
            "humidity",
        )

        cloud_cover_pct = validate_number(
            request.cloud_cover,
            "cloud_cover",
        )

        wind_speed = validate_number(
            request.wind_speed,
            "wind_speed",
        )

        if not 0 <= humidity <= 100:
            raise ValueError(
                "humidity must be between 0 and 100."
            )

        if not 0 <= cloud_cover_pct <= 100:
            raise ValueError(
                "cloud_cover must be between 0 and 100."
            )

        if wind_speed < 0:
            raise ValueError(
                "wind_speed cannot be negative."
            )

        # ========================================================
        # CLOUD COVER
        # ========================================================

        cloud_cover_fraction = (
            cloud_cover_pct / 100.0
        )

        # ========================================================
        # TARGET DAY
        # ========================================================

        (
            day_df,
            irradiance_values,
        ) = build_day_dataframe(
            day_of_year=day_of_year,
            temperature=temperature,
            humidity=humidity,
            cloud_cover_fraction=(
                cloud_cover_fraction
            ),
            wind_speed=wind_speed,
        )

        # ========================================================
        # XGBOOST - DIRECT 24 ROW INFERENCE
        # ========================================================

        xgb_features, _ = (
            get_model_features()
        )

        xgb_feature_matrix = (
            day_df[
                xgb_features
            ]
            .astype(np.float32)
        )

        xgb_predictions = (
            model.xgb_model
            .predict(
                xgb_feature_matrix
            )
        )

        xgb_predictions = np.maximum(
            np.asarray(
                xgb_predictions,
                dtype=np.float64,
            ),
            0.0,
        )

        if len(
            xgb_predictions
        ) != 24:
            raise ValueError(
                "XGBoost did not return "
                "24 daily predictions."
            )

        # ========================================================
        # BUILD 48-HOUR TCN CONTEXT
        # ========================================================

        context_df = build_day_context(
            day_of_year=day_of_year,
            temperature=temperature,
            humidity=humidity,
            cloud_cover_fraction=(
                cloud_cover_fraction
            ),
            wind_speed=wind_speed,
        )

        # ========================================================
        # TCN - 24 SEQUENCE PREDICTIONS
        # ========================================================

        tcn_predictions = (
            predict_tcn_day(
                context_df
            )
        )

        if len(
            tcn_predictions
        ) != 24:
            raise ValueError(
                "TCN did not return "
                "24 daily predictions."
            )

        # ========================================================
        # HYBRID
        # ========================================================

        xgb_weight = float(
            model.xgb_weight
        )

        tcn_weight = float(
            model.tcn_weight
        )

        hybrid_predictions = (
            (
                xgb_weight
                *
                xgb_predictions
            )
            +
            (
                tcn_weight
                *
                tcn_predictions
            )
        )

        hybrid_predictions = np.maximum(
            np.nan_to_num(
                hybrid_predictions,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        )

        # ========================================================
        # DAILY TOTALS
        # ========================================================

        # Hourly model predictions represent power in watts (W).
        # Each prediction covers one hour, therefore:
        #     daily_Wh  = sum(hourly_W)
        #     daily_kWh = daily_Wh / 1000
        #
        # Keep the hourly arrays in W for the chart, while exposing
        # the daily totals in genuine kWh.
        xgb_total = float(
            np.sum(
                xgb_predictions
            ) / 1000.0
        )

        tcn_total = float(
            np.sum(
                tcn_predictions
            ) / 1000.0
        )

        hybrid_total = float(
            np.sum(
                hybrid_predictions
            ) / 1000.0
        )

        # ========================================================
        # DIAGNOSTICS
        # ========================================================

        diagnostics = {
            "cloud_cover_input_percent": round(
                cloud_cover_pct,
                4,
            ),

            "cloud_cover_model_fraction": round(
                cloud_cover_fraction,
                4,
            ),

            "tcn_weight": round(
                tcn_weight,
                4,
            ),

            "xgboost_weight": round(
                xgb_weight,
                4,
            ),

            "tcn_min": round(
                float(
                    np.min(
                        tcn_predictions
                    )
                ),
                4,
            ),

            "tcn_max": round(
                float(
                    np.max(
                        tcn_predictions
                    )
                ),
                4,
            ),

            "tcn_mean": round(
                float(
                    np.mean(
                        tcn_predictions
                    )
                ),
                4,
            ),

            "xgb_min": round(
                float(
                    np.min(
                        xgb_predictions
                    )
                ),
                4,
            ),

            "xgb_max": round(
                float(
                    np.max(
                        xgb_predictions
                    )
                ),
                4,
            ),

            "hybrid_min": round(
                float(
                    np.min(
                        hybrid_predictions
                    )
                ),
                4,
            ),

            "hybrid_max": round(
                float(
                    np.max(
                        hybrid_predictions
                    )
                ),
                4,
            ),
        }

        # ========================================================
        # RESPONSE
        # ========================================================

        return {
            "architecture": ARCHITECTURE,

            "lstm_used": LSTM_USED,

            "lookback_hours": (
                get_lookback()
            ),

            "hours": list(
                range(24)
            ),

            "irradiance": [
                round(
                    float(value),
                    2,
                )
                for value
                in irradiance_values
            ],

            "xgboost": [
                round(
                    float(value),
                    4,
                )
                for value
                in xgb_predictions
            ],

            "tcn": [
                round(
                    float(value),
                    4,
                )
                for value
                in tcn_predictions
            ],

            "hybrid": [
                round(
                    float(value),
                    4,
                )
                for value
                in hybrid_predictions
            ],

            # Hourly predictions are power values in watts.
            "output_unit": "W",

            # Daily totals are energy values in kilowatt-hours.
            "energy_unit": "kWh",

            "totals_kwh": {
                "xgboost": round(
                    xgb_total,
                    4,
                ),

                "tcn": round(
                    tcn_total,
                    4,
                ),

                "hybrid": round(
                    hybrid_total,
                    4,
                ),
            },

            "inputs": {
                "day_of_year": (
                    day_of_year
                ),

                "temperature": round(
                    temperature,
                    4,
                ),

                "humidity": round(
                    humidity,
                    4,
                ),

                "cloud_cover_pct": round(
                    cloud_cover_pct,
                    4,
                ),

                "wind_speed": round(
                    wind_speed,
                    4,
                ),
            },

            "tcn_diagnostics": diagnostics,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Daily forecast failed: {exc}"
            ),
        )


# ================================================================
# END OF ROUTES
# ================================================================