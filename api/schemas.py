# ================================================================
# SOLAR YIELD FORECAST API - SCHEMAS
# ================================================================
# Architecture:
#     XGBoost + TCN + Hybrid Ensemble
#
# Deep Learning Model:
#     Temporal Convolutional Network (TCN)
#
# TCN Lookback:
#     24 hours
#
# Prediction:
#     Next-hour sequence-to-one forecasting
#
# LSTM:
#     NOT USED
# ================================================================

from typing import List, Optional

from pydantic import BaseModel, Field


# ================================================================
# COMMON MODEL INFORMATION
# ================================================================

ARCHITECTURE = "XGBoost + TCN + Hybrid"
TCN_LOOKBACK_HOURS = 24
LSTM_USED = False


# ================================================================
# SINGLE PREDICTION REQUEST
# ================================================================

class PredictionRequest(BaseModel):
    """
    Request body for a single next-hour prediction.

    The API accepts cloud_cover as a percentage:

        0   = 0%
        30  = 30%
        100 = 100%

    The API internally converts this to the model representation.
    """

    timestamp: Optional[str] = Field(
        default=None,
        description=(
            "Timestamp of the input observation. "
            "Optional."
        ),
    )

    temperature: float = Field(
        ...,
        description=(
            "Temperature in degrees Celsius."
        ),
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Relative humidity percentage "
            "(0-100)."
        ),
    )

    cloud_cover: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Cloud cover percentage "
            "(0-100)."
        ),
    )

    wind_speed: float = Field(
        ...,
        ge=0,
        description=(
            "Wind speed."
        ),
    )

    irradiance: float = Field(
        ...,
        ge=0,
        description=(
            "Solar irradiance."
        ),
    )

    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description=(
            "Hour of day (0-23)."
        ),
    )

    day_of_year: int = Field(
        ...,
        ge=1,
        le=366,
        description=(
            "Day of year (1-366)."
        ),
    )


# ================================================================
# SINGLE PREDICTION RESPONSE
# ================================================================

class PredictionResponse(BaseModel):
    """
    Response for a single next-hour prediction.
    """

    input_timestamp: Optional[str] = Field(
        default=None,
        description=(
            "Timestamp of the input observation."
        ),
    )

    forecast_timestamp: Optional[str] = Field(
        default=None,
        description=(
            "Timestamp being predicted "
            "(input timestamp + 1 hour)."
        ),
    )

    xgboost_prediction: float = Field(
        ...,
        description=(
            "XGBoost solar yield prediction."
        ),
    )

    tcn_prediction: float = Field(
        ...,
        description=(
            "TCN solar yield prediction."
        ),
    )

    hybrid_prediction: float = Field(
        ...,
        description=(
            "Hybrid XGBoost + TCN prediction."
        ),
    )

    architecture: str = Field(
        default=ARCHITECTURE,
        description=(
            "Forecasting architecture."
        ),
    )

    tcn_lookback_hours: int = Field(
        default=TCN_LOOKBACK_HOURS,
        description=(
            "Number of historical hours "
            "used by the TCN."
        ),
    )

    lstm_used: bool = Field(
        default=LSTM_USED,
        description=(
            "Indicates whether an LSTM model "
            "is used."
        ),
    )


# ================================================================
# FORECAST OBSERVATION
# ================================================================

class ForecastObservation(BaseModel):
    """
    One timestamped weather/irradiance observation.

    Used for historical context and future observations.
    """

    timestamp: str = Field(
        ...,
        description=(
            "ISO-format timestamp."
        ),
    )

    temperature: float = Field(
        ...,
        description=(
            "Temperature in degrees Celsius."
        ),
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Relative humidity percentage."
        ),
    )

    cloud_cover: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Cloud cover percentage."
        ),
    )

    wind_speed: float = Field(
        ...,
        ge=0,
        description=(
            "Wind speed."
        ),
    )

    irradiance: float = Field(
        ...,
        ge=0,
        description=(
            "Solar irradiance."
        ),
    )


# ================================================================
# MULTI-HOUR FORECAST REQUEST
# ================================================================

class ForecastRequest(BaseModel):
    """
    Request for rolling multi-hour forecasting.

    history:
        Previously observed data.

    future:
        Future weather/irradiance observations.
        Each observation at time t is used to predict t+1.
    """

    history: List[
        ForecastObservation
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Historical observations used as "
            "temporal context."
        ),
    )

    future: List[
        ForecastObservation
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Future observations used for "
            "rolling next-hour forecasts."
        ),
    )


# ================================================================
# MULTI-HOUR FORECAST RESULT
# ================================================================

class ForecastPrediction(BaseModel):
    """
    Prediction for one forecast timestamp.
    """

    timestamp: str = Field(
        ...,
        description=(
            "Forecast target timestamp."
        ),
    )

    xgboost: float = Field(
        ...,
        description=(
            "XGBoost prediction."
        ),
    )

    tcn: float = Field(
        ...,
        description=(
            "TCN prediction."
        ),
    )

    hybrid: float = Field(
        ...,
        description=(
            "Hybrid prediction."
        ),
    )


class ForecastResponse(BaseModel):
    """
    Response for multi-hour forecasting.
    """

    architecture: str = Field(
        default=ARCHITECTURE,
    )

    lstm_used: bool = Field(
        default=LSTM_USED,
    )

    lookback_hours: int = Field(
        default=TCN_LOOKBACK_HOURS,
    )

    predictions: List[
        ForecastPrediction
    ] = Field(
        ...,
        description=(
            "Chronological forecast predictions."
        ),
    )


# ================================================================
# 24-HOUR DAY FORECAST REQUEST
# ================================================================

class DayForecastRequest(BaseModel):
    """
    Request for a complete 24-hour forecast.

    Weather values represent the requested forecast day.

    cloud_cover is supplied as percentage:

        0   = clear
        30  = 30% cloud cover
        100 = completely cloudy
    """

    day_of_year: int = Field(
        ...,
        ge=1,
        le=366,
        description=(
            "Day of year (1-366)."
        ),
    )

    temperature: float = Field(
        ...,
        description=(
            "Representative temperature "
            "for the forecast day."
        ),
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Representative relative humidity "
            "percentage."
        ),
    )

    cloud_cover: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Representative cloud cover "
            "percentage."
        ),
    )

    wind_speed: float = Field(
        ...,
        ge=0,
        description=(
            "Representative wind speed."
        ),
    )


# ================================================================
# 24-HOUR DAY FORECAST TOTALS
# ================================================================

class DayForecastTotals(BaseModel):
    """
    Aggregated daily predictions.

    Values are returned using the same unit convention
    as the model's solar_output target.
    """

    xgboost: float = Field(
        ...,
        description=(
            "Total daily XGBoost forecast."
        ),
    )

    tcn: float = Field(
        ...,
        description=(
            "Total daily TCN forecast."
        ),
    )

    hybrid: float = Field(
        ...,
        description=(
            "Total daily Hybrid forecast."
        ),
    )


# ================================================================
# 24-HOUR DAY FORECAST INPUT SUMMARY
# ================================================================

class DayForecastInputs(BaseModel):
    """
    Echo of the inputs used to generate the
    daily forecast.
    """

    day_of_year: int

    temperature: float

    humidity: float

    cloud_cover_pct: float

    wind_speed: float


# ================================================================
# TCN DIAGNOSTICS
# ================================================================

class TCNDiagnostics(BaseModel):
    """
    Diagnostic information useful for validating
    the daily TCN forecast.
    """

    cloud_cover_input_percent: float = Field(
        ...,
        description=(
            "Cloud cover supplied by the API user "
            "as percentage."
        ),
    )

    cloud_cover_model_fraction: float = Field(
        ...,
        description=(
            "Cloud cover after conversion to "
            "the model representation."
        ),
    )

    tcn_min: float = Field(
        ...,
        description=(
            "Minimum TCN prediction for the day."
        ),
    )

    tcn_max: float = Field(
        ...,
        description=(
            "Maximum TCN prediction for the day."
        ),
    )

    tcn_mean: float = Field(
        ...,
        description=(
            "Mean TCN prediction for the day."
        ),
    )

    xgb_min: float = Field(
        ...,
        description=(
            "Minimum XGBoost prediction for the day."
        ),
    )

    xgb_max: float = Field(
        ...,
        description=(
            "Maximum XGBoost prediction for the day."
        ),
    )

    hybrid_min: float = Field(
        ...,
        description=(
            "Minimum Hybrid prediction for the day."
        ),
    )

    hybrid_max: float = Field(
        ...,
        description=(
            "Maximum Hybrid prediction for the day."
        ),
    )


# ================================================================
# 24-HOUR DAY FORECAST RESPONSE
# ================================================================

class DayForecastResponse(BaseModel):
    """
    Complete 24-hour solar forecast response.
    """

    architecture: str = Field(
        default=ARCHITECTURE,
        description=(
            "Forecasting architecture."
        ),
    )

    lstm_used: bool = Field(
        default=LSTM_USED,
        description=(
            "LSTM is not used."
        ),
    )

    lookback_hours: int = Field(
        default=TCN_LOOKBACK_HOURS,
        description=(
            "TCN temporal lookback."
        ),
    )

    hours: List[int] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "Forecast hours 0 through 23."
        ),
    )

    irradiance: List[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "Estimated irradiance for each hour."
        ),
    )

    xgboost: List[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "24 hourly XGBoost predictions."
        ),
    )

    tcn: List[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "24 hourly TCN predictions."
        ),
    )

    hybrid: List[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "24 hourly Hybrid predictions."
        ),
    )

    totals_kwh: DayForecastTotals = Field(
        ...,
        description=(
            "Aggregated daily forecast values."
        ),
    )

    inputs: DayForecastInputs = Field(
        ...,
        description=(
            "Input summary."
        ),
    )

    tcn_diagnostics: TCNDiagnostics = Field(
        ...,
        description=(
            "TCN prediction diagnostics."
        ),
    )


# ================================================================
# END OF SCHEMAS
# ================================================================