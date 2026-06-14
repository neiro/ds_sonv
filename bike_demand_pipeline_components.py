from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


BASE_NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed_ms",
    "visibility_10m",
    "dew_point_temperature",
    "solar_radiation_mjm2",
    "rainfallmm",
    "snowfall_cm",
]

CATEGORICAL_FEATURES = ["seasons", "holiday", "functioning_day"]

TIME_FEATURES = [
    "time_period_evening",
    "time_period_late_evening",
    "time_period_morning",
    "time_period_night",
]

BASE_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES + TIME_FEATURES

ENGINEERED_FEATURES = [
    "dew_point_gap",
    "has_rain",
    "has_snow",
    "has_precipitation",
    "comfortable_temperature",
    "freezing_weather",
    "hot_weather",
    "hot_and_humid",
    "temp_x_solar",
    "temp_x_humidity",
    "rain_x_wind",
    "snow_x_freezing",
    "time_period_daytime",
]

NUMERIC_WITH_ENGINEERED = BASE_NUMERIC_FEATURES + TIME_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES_AFTER_ENGINEERING = NUMERIC_WITH_ENGINEERED + CATEGORICAL_FEATURES


class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable weather and time-period features for bike demand models."""

    input_features = BASE_FEATURES
    output_features = MODEL_FEATURES_AFTER_ENGINEERING

    def __init__(self, unknown_feature_policy: str = "ignore") -> None:
        self.unknown_feature_policy = unknown_feature_policy

    def _validate_unknown_feature_policy(self) -> None:
        allowed_policies = {"ignore", "raise"}
        if self.unknown_feature_policy not in allowed_policies:
            raise ValueError(
                "unknown_feature_policy must be one of "
                f"{sorted(allowed_policies)}, got {self.unknown_feature_policy!r}"
            )

    def _check_columns(self, frame: pd.DataFrame) -> list[str]:
        missing_columns = [
            column for column in self.input_features if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        unexpected_columns = [
            column for column in frame.columns if column not in self.input_features
        ]
        if unexpected_columns and self.unknown_feature_policy == "raise":
            raise ValueError(f"Unexpected columns: {unexpected_columns}")
        return unexpected_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BikeFeatureEngineer":
        self._validate_unknown_feature_policy()
        frame = pd.DataFrame(X)
        unexpected_columns = self._check_columns(frame)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.ignored_features_in_ = np.asarray(unexpected_columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")

        frame = pd.DataFrame(X)
        self._check_columns(frame)

        frame = frame.loc[:, self.input_features].copy()
        numeric_columns = BASE_NUMERIC_FEATURES + TIME_FEATURES
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype("object")

        temperature = frame["temperature"]
        humidity = frame["humidity"]
        wind = frame["wind_speed_ms"]
        dew_point = frame["dew_point_temperature"]
        solar = frame["solar_radiation_mjm2"]
        rainfall = frame["rainfallmm"]
        snowfall = frame["snowfall_cm"]
        rainfall_filled = rainfall.fillna(0)
        snowfall_filled = snowfall.fillna(0)

        frame["dew_point_gap"] = temperature - dew_point
        frame["has_rain"] = (rainfall_filled > 0).astype(float)
        frame["has_snow"] = (snowfall_filled > 0).astype(float)
        frame["has_precipitation"] = (
            (rainfall_filled > 0) | (snowfall_filled > 0)
        ).astype(float)
        frame["comfortable_temperature"] = temperature.between(
            12, 26, inclusive="both"
        ).astype(float)
        frame["freezing_weather"] = (temperature <= 0).astype(float)
        frame["hot_weather"] = (temperature >= 28).astype(float)
        frame["hot_and_humid"] = ((temperature >= 28) & (humidity >= 70)).astype(float)
        frame["temp_x_solar"] = temperature * solar
        frame["temp_x_humidity"] = temperature * humidity
        frame["rain_x_wind"] = rainfall * wind
        frame["snow_x_freezing"] = snowfall * frame["freezing_weather"]
        frame["time_period_daytime"] = (
            frame[TIME_FEATURES].fillna(0).sum(axis=1) == 0
        ).astype(float)

        return frame[self.output_features]
