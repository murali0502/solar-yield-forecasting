"""
Production inference layer for Solar Yield Forecasting.

Architecture:

    XGBoost
       +
      TCN
       ↓
    Weighted Hybrid
       ↓
  Solar Forecast

XGBoost:
    Uses features at time t to predict solar output at t+1.

TCN:
    Uses the previous 24 hourly observations to predict
    solar output at t+1.

Hybrid:
    Combines XGBoost and TCN using weights optimized
    on validation data.

IMPORTANT:
    This file performs inference only.
    Model training is handled separately.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# MODEL PATHS
# ============================================================

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "model.json"
)

TCN_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tcn"
    / "model.keras"
)

TCN_FEATURE_SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tcn"
    / "feature_scaler.joblib"
)

TCN_TARGET_SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tcn"
    / "target_scaler.joblib"
)

TCN_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "tcn"
    / "config.json"
)

XGB_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "config.json"
)

HYBRID_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "hybrid"
    / "config.json"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# TCN CONFIGURATION
# ============================================================

TCN_LOOKBACK = 24

EXPECTED_FEATURE_COUNT = (
    len(FEATURE_COLUMNS)
)


# ============================================================
# SOLAR PREDICTOR
# ============================================================

class SolarPredictor:
    """
    Production inference wrapper for:

        XGBoost
            +
        TCN
            ↓
        Hybrid prediction

    XGBoost:
        Current feature row → next-hour prediction

    TCN:
        Previous 24 hours → next-hour prediction

    Hybrid:

        Hybrid =
            XGBoost × xgboost_weight
            +
            TCN × tcn_weight
    """

    def __init__(self):

        print()

        print(
            "=" * 70
        )

        print(
            "LOADING SOLAR FORECAST MODELS"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # CHECK REQUIRED FILES
        # ====================================================

        self._check_required_files()

        # ====================================================
        # LOAD XGBOOST
        # ====================================================

        print()

        print(
            "Loading XGBoost model..."
        )

        self.xgb_model = (
            xgb.XGBRegressor()
        )

        self.xgb_model.load_model(
            XGB_MODEL_PATH
        )

        print(
            "XGBoost model loaded."
        )

        # ====================================================
        # LOAD TCN
        # ====================================================

        print()

        print(
            "Loading TCN model..."
        )

        self.tcn_model = load_model(
            TCN_MODEL_PATH
        )

        print(
            "TCN model loaded."
        )

        # ====================================================
        # LOAD TCN SCALERS
        # ====================================================

        print()

        print(
            "Loading TCN feature scaler..."
        )

        self.feature_scaler = (
            joblib.load(
                TCN_FEATURE_SCALER_PATH
            )
        )

        print(
            "Loading TCN target scaler..."
        )

        self.target_scaler = (
            joblib.load(
                TCN_TARGET_SCALER_PATH
            )
        )

        print(
            "TCN scalers loaded."
        )

        # ====================================================
        # LOAD OPTIONAL CONFIGURATION
        # ====================================================

        self.tcn_config = (
            self._load_json_config(
                TCN_CONFIG_PATH
            )
        )

        self.xgb_config = (
            self._load_json_config(
                XGB_CONFIG_PATH
            )
        )

        # ====================================================
        # LOAD HYBRID CONFIG
        # ====================================================

        print()

        print(
            "Loading hybrid configuration..."
        )

        with open(
            HYBRID_CONFIG_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            self.hybrid_config = (
                json.load(file)
            )

        self.tcn_weight = float(
            self.hybrid_config[
                "tcn_weight"
            ]
        )

        self.xgb_weight = float(
            self.hybrid_config[
                "xgboost_weight"
            ]
        )

        print(
            f"TCN weight     : "
            f"{self.tcn_weight:.4f}"
        )

        print(
            f"XGBoost weight : "
            f"{self.xgb_weight:.4f}"
        )

        # ====================================================
        # VALIDATE HYBRID WEIGHTS
        # ====================================================

        weight_sum = (
            self.tcn_weight
            +
            self.xgb_weight
        )

        if not np.isfinite(
            weight_sum
        ):

            raise ValueError(
                "Hybrid weights contain "
                "invalid values."
            )

        if not np.isclose(
            weight_sum,
            1.0,
            atol=1e-6,
        ):

            raise ValueError(
                "Hybrid weights must sum "
                "to 1.0. "
                f"Current sum: {weight_sum}"
            )

        # ====================================================
        # VALIDATE MODEL DIMENSIONS
        # ====================================================

        self._validate_model_dimensions()

        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        print()

        print(
            "Model loading completed."
        )

        print()

        print(
            "Architecture:"
        )

        print(
            "XGBoost + TCN + Hybrid"
        )

        print()

        print(
            "LSTM used:"
        )

        print(
            "False"
        )

        print()

    # ========================================================
    # LOAD JSON CONFIG
    # ========================================================

    @staticmethod
    def _load_json_config(
        path,
    ):

        if not path.exists():

            return {}

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except (
            json.JSONDecodeError,
            OSError,
        ):

            return {}

    # ========================================================
    # CHECK FILE
    # ========================================================

    @staticmethod
    def _check_file(
        path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                "Required model file not found:\n"
                f"{path}"
            )

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    def _check_required_files(
        self,
    ):

        required_files = [

            XGB_MODEL_PATH,

            TCN_MODEL_PATH,

            TCN_FEATURE_SCALER_PATH,

            TCN_TARGET_SCALER_PATH,

            HYBRID_CONFIG_PATH,
        ]

        for path in required_files:

            self._check_file(
                path
            )

    # ========================================================
    # VALIDATE MODEL DIMENSIONS
    # ========================================================

    def _validate_model_dimensions(
        self,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            "MODEL DIMENSION VALIDATION"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # XGBoost feature count
        # ----------------------------------------------------

        xgb_feature_count = (
            self.xgb_model
            .n_features_in_
        )

        print(
            "XGBoost feature count:"
        )

        print(
            f"  {xgb_feature_count}"
        )

        if (
            xgb_feature_count
            != EXPECTED_FEATURE_COUNT
        ):

            raise ValueError(
                "XGBoost feature count mismatch. "
                f"Expected "
                f"{EXPECTED_FEATURE_COUNT}, "
                f"got "
                f"{xgb_feature_count}."
            )

        # ----------------------------------------------------
        # TCN input
        # ----------------------------------------------------

        tcn_input_shape = (
            self.tcn_model
            .input_shape
        )

        tcn_output_shape = (
            self.tcn_model
            .output_shape
        )

        print()

        print(
            "TCN input shape:"
        )

        print(
            f"  {tcn_input_shape}"
        )

        print()

        print(
            "TCN output shape:"
        )

        print(
            f"  {tcn_output_shape}"
        )

        # ----------------------------------------------------
        # TCN input validation
        # ----------------------------------------------------

        if len(
            tcn_input_shape
        ) != 3:

            raise ValueError(
                "Invalid TCN input shape. "
                f"Expected 3 dimensions, "
                f"got {tcn_input_shape}."
            )

        actual_lookback = (
            tcn_input_shape[1]
        )

        actual_features = (
            tcn_input_shape[2]
        )

        if (
            actual_lookback
            != TCN_LOOKBACK
        ):

            raise ValueError(
                "TCN lookback mismatch. "
                f"Expected {TCN_LOOKBACK}, "
                f"got {actual_lookback}."
            )

        if (
            actual_features
            != EXPECTED_FEATURE_COUNT
        ):

            raise ValueError(
                "TCN feature count mismatch. "
                f"Expected "
                f"{EXPECTED_FEATURE_COUNT}, "
                f"got {actual_features}."
            )

        # ----------------------------------------------------
        # TCN output validation
        # ----------------------------------------------------

        if len(
            tcn_output_shape
        ) != 2:

            raise ValueError(
                "Invalid TCN output shape. "
                "This project uses a "
                "sequence-to-one TCN. "
                f"Expected (None, 1), "
                f"got {tcn_output_shape}."
            )

        if (
            tcn_output_shape[1]
            != 1
        ):

            raise ValueError(
                "Invalid TCN output size. "
                "Expected one prediction, "
                f"got {tcn_output_shape}."
            )

        print()

        print(
            "TCN configuration:"
        )

        print(
            f"  Lookback : "
            f"{TCN_LOOKBACK}"
        )

        print(
            f"  Features : "
            f"{EXPECTED_FEATURE_COUNT}"
        )

        print(
            "  Output   : "
            "one next-hour prediction"
        )

        print()

        print(
            "Model dimensions: OK"
        )

        print(
            "=" * 70
        )

    # ========================================================
    # TIME FEATURES
    # ========================================================

    @staticmethod
    def add_time_features(
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = data.copy()

        # ----------------------------------------------------
        # Timestamp validation
        # ----------------------------------------------------

        if (
            "timestamp"
            not in df.columns
        ):

            raise ValueError(
                "Input data must contain "
                "'timestamp'."
            )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                errors="raise",
            )
        )

        # ----------------------------------------------------
        # Hour
        # ----------------------------------------------------

        hour = (
            df["timestamp"]
            .dt.hour
            .astype(float)
        )

        # ----------------------------------------------------
        # Day of year
        # ----------------------------------------------------

        day_of_year = (
            df["timestamp"]
            .dt.dayofyear
            .astype(float)
        )

        # ----------------------------------------------------
        # Hour cyclic encoding
        # ----------------------------------------------------

        df["hour_sin"] = np.sin(
            2.0
            * np.pi
            * hour
            / 24.0
        )

        df["hour_cos"] = np.cos(
            2.0
            * np.pi
            * hour
            / 24.0
        )

        # ----------------------------------------------------
        # Day cyclic encoding
        #
        # IMPORTANT:
        # Must match preprocessing/training.
        # ----------------------------------------------------

        df["day_sin"] = np.sin(
            2.0
            * np.pi
            * day_of_year
            / 365.25
        )

        df["day_cos"] = np.cos(
            2.0
            * np.pi
            * day_of_year
            / 365.25
        )

        return df

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    @staticmethod
    def validate_input(
        data: pd.DataFrame,
    ):

        required = [

            "timestamp",

            "temperature",

            "humidity",

            "cloud_cover",

            "wind_speed",

            "irradiance",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Missing required input "
                "columns: "
                + ", ".join(missing)
            )

        if data.empty:

            raise ValueError(
                "Input data is empty."
            )

        numeric_columns = [

            "temperature",

            "humidity",

            "cloud_cover",

            "wind_speed",

            "irradiance",
        ]

        for column in numeric_columns:

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            if values.isna().any():

                raise ValueError(
                    f"Invalid or missing value "
                    f"found in '{column}'."
                )

            if not np.isfinite(
                values.values
            ).all():

                raise ValueError(
                    f"Non-finite value found "
                    f"in '{column}'."
                )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def prepare_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate_input(
            data
        )

        df = (
            self.add_time_features(
                data
            )
        )

        missing_features = [

            column

            for column in FEATURE_COLUMNS

            if column not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Unable to create required "
                "features: "
                +
                ", ".join(
                    missing_features
                )
            )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for column in FEATURE_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        # ----------------------------------------------------
        # Final finite-value validation
        # ----------------------------------------------------

        feature_matrix = (
            df[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            feature_matrix
        ).all():

            raise ValueError(
                "Prepared feature matrix "
                "contains invalid values."
            )

        return df

    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

    def predict_xgboost(
        self,
        data: pd.DataFrame,
    ) -> float:

        features = (
            data[
                FEATURE_COLUMNS
            ]
            .iloc[
                [-1]
            ]
        )

        prediction = (
            self.xgb_model
            .predict(
                features
            )
        )

        prediction = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        if not np.isfinite(
            prediction
        ):

            raise ValueError(
                "XGBoost produced an "
                "invalid prediction."
            )

        # Solar output cannot be negative.

        return max(
            0.0,
            prediction,
        )

    # ========================================================
    # TCN PREDICTION
    # ========================================================

    def predict_tcn(
        self,
        data: pd.DataFrame,
    ) -> float:

        if len(data) < TCN_LOOKBACK:

            raise ValueError(
                "TCN requires at least "
                f"{TCN_LOOKBACK} hourly "
                "observations. "
                f"Received: {len(data)}."
            )

        # ----------------------------------------------------
        # Extract latest 24 hours
        # ----------------------------------------------------

        sequence = (
            data[
                FEATURE_COLUMNS
            ]
            .iloc[
                -TCN_LOOKBACK:
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Scale using training scaler
        # ----------------------------------------------------

        scaled_sequence = (
            self.feature_scaler
            .transform(
                sequence
            )
        )

        scaled_sequence = np.asarray(
            scaled_sequence,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Validate scaled shape
        # ----------------------------------------------------

        expected_shape = (
            TCN_LOOKBACK,
            EXPECTED_FEATURE_COUNT,
        )

        if (
            scaled_sequence.shape
            != expected_shape
        ):

            raise ValueError(
                "TCN sequence shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {scaled_sequence.shape}."
            )

        # ----------------------------------------------------
        # Batch dimension
        #
        # (24, 9)
        #     ↓
        # (1, 24, 9)
        # ----------------------------------------------------

        model_input = (
            np.expand_dims(
                scaled_sequence,
                axis=0,
            )
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        scaled_prediction = (
            self.tcn_model
            .predict(
                model_input,
                verbose=0,
            )
        )

        # ----------------------------------------------------
        # Ensure sequence-to-one output
        # ----------------------------------------------------

        scaled_prediction = (
            np.asarray(
                scaled_prediction
            )
            .reshape(-1)
        )

        if len(
            scaled_prediction
        ) != 1:

            raise ValueError(
                "TCN must produce exactly "
                "one prediction. "
                f"Received "
                f"{len(scaled_prediction)}."
            )

        # ----------------------------------------------------
        # Inverse target scaling
        # ----------------------------------------------------

        prediction = (
            self.target_scaler
            .inverse_transform(
                scaled_prediction
                .reshape(
                    -1,
                    1,
                )
            )
        )

        prediction = float(
            prediction
            .reshape(-1)[0]
        )

        if not np.isfinite(
            prediction
        ):

            raise ValueError(
                "TCN produced an "
                "invalid prediction."
            )

        # Solar output cannot be negative.

        return max(
            0.0,
            prediction,
        )

    # ========================================================
    # HYBRID PREDICTION
    # ========================================================

    def combine_predictions(
        self,
        xgb_prediction,
        tcn_prediction,
    ) -> float:

        hybrid_prediction = (
            (
                self.xgb_weight
                * xgb_prediction
            )
            +
            (
                self.tcn_weight
                * tcn_prediction
            )
        )

        hybrid_prediction = float(
            hybrid_prediction
        )

        if not np.isfinite(
            hybrid_prediction
        ):

            raise ValueError(
                "Hybrid produced an "
                "invalid prediction."
            )

        return max(
            0.0,
            hybrid_prediction,
        )

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        data: pd.DataFrame,
    ) -> dict:
        """
        Generate a one-hour-ahead prediction.

        Input:
            Historical observations ending at time t.

        Output:
            Forecast for time t+1 hour.
        """

        # ====================================================
        # PREPARE
        # ====================================================

        prepared = (
            self.prepare_features(
                data
            )
        )

        # ====================================================
        # SORT CHRONOLOGICALLY
        # ====================================================

        prepared = (
            prepared
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # INPUT TIMESTAMP
        # ====================================================

        input_timestamp = (
            prepared[
                "timestamp"
            ].iloc[-1]
        )

        # ====================================================
        # FORECAST TIMESTAMP
        # ====================================================

        forecast_timestamp = (
            input_timestamp
            +
            pd.Timedelta(
                hours=1
            )
        )

        # ====================================================
        # XGBOOST
        # ====================================================

        xgb_prediction = (
            self.predict_xgboost(
                prepared
            )
        )

        # ====================================================
        # TCN
        # ====================================================

        tcn_prediction = (
            self.predict_tcn(
                prepared
            )
        )

        # ====================================================
        # HYBRID
        # ====================================================

        hybrid_prediction = (
            self.combine_predictions(
                xgb_prediction=
                    xgb_prediction,

                tcn_prediction=
                    tcn_prediction,
            )
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "input_timestamp":
                input_timestamp.isoformat(),

            "forecast_timestamp":
                forecast_timestamp.isoformat(),

            "xgboost_prediction":
                round(
                    xgb_prediction,
                    6,
                ),

            "tcn_prediction":
                round(
                    tcn_prediction,
                    6,
                ),

            "hybrid_prediction":
                round(
                    hybrid_prediction,
                    6,
                ),

            "xgboost_weight":
                round(
                    self.xgb_weight,
                    6,
                ),

            "tcn_weight":
                round(
                    self.tcn_weight,
                    6,
                ),

            "architecture":
                "XGBoost + TCN + Hybrid",

            "tcn_lookback_hours":
                TCN_LOOKBACK,

            "lstm_used":
                False,
        }


# ============================================================
# TERMINAL TEST
# ============================================================

def terminal_test():

    print()

    print(
        "=" * 70
    )

    print(
        "SOLAR INFERENCE TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "solar_dataset.csv"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"{dataset_path}"
        )

    print()

    print(
        "Loading dataset..."
    )

    df = pd.read_csv(
        dataset_path
    )

    # --------------------------------------------------------
    # Check enough rows
    # --------------------------------------------------------

    if len(df) < TCN_LOOKBACK:

        raise ValueError(
            "Dataset contains fewer than "
            f"{TCN_LOOKBACK} rows."
        )

    # --------------------------------------------------------
    # Last 24 observations
    # --------------------------------------------------------

    test_data = (
        df.tail(
            TCN_LOOKBACK
        )
        .copy()
    )

    print()

    print(
        f"Testing with "
        f"{len(test_data)} observations."
    )

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    predictor = (
        SolarPredictor()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = (
        predictor.predict(
            test_data
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "INFERENCE RESULT"
    )

    print(
        "=" * 70
    )

    print()

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=" * 70
    )

    print(
        "INFERENCE TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    terminal_test()