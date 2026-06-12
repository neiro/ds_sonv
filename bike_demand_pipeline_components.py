from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

    def __init__(self) -> None:
        self.input_features_ = BASE_FEATURES
        self.output_features_ = MODEL_FEATURES_AFTER_ENGINEERING

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BikeFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in self.input_features_:
            if column not in frame.columns:
                frame[column] = np.nan

        for column in BASE_NUMERIC_FEATURES + TIME_FEATURES:
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

        frame["dew_point_gap"] = temperature - dew_point
        frame["has_rain"] = (rainfall.fillna(0) > 0).astype(float)
        frame["has_snow"] = (snowfall.fillna(0) > 0).astype(float)
        frame["has_precipitation"] = (
            (rainfall.fillna(0) > 0) | (snowfall.fillna(0) > 0)
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

        return frame[self.output_features_]
