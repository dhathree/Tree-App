from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "rainfall_mm",
    "vegetation_index",
    "soil_moisture_pct",
    "slope_deg",
    "tree_density_per_ha",
    "soil_type",
]


@dataclass
class TrainResult:
    mae: float
    r2: float
    model_path: Path


def _generate_synthetic_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    temperature = rng.normal(31, 6, n_samples).clip(15, 48)
    humidity = rng.normal(45, 18, n_samples).clip(10, 95)
    wind = rng.normal(22, 10, n_samples).clip(0, 75)
    rainfall = rng.gamma(shape=2.2, scale=12, size=n_samples).clip(0, 260)
    ndvi = rng.uniform(0.08, 0.95, n_samples)
    soil_moisture = rng.normal(34, 12, n_samples).clip(5, 82)
    slope = rng.normal(14, 8, n_samples).clip(0, 45)
    tree_density = rng.normal(470, 190, n_samples).clip(50, 1250)
    soil_types = np.array(["sandy", "clay", "loamy", "peaty", "silty"])
    soil_type = rng.choice(soil_types, size=n_samples, p=[0.22, 0.18, 0.34, 0.12, 0.14])

    dryness = (temperature * 0.95 + wind * 0.85 + slope * 0.32) - (humidity * 0.75 + rainfall * 0.15 + soil_moisture * 0.55)
    fuel = (tree_density / 12) * (1.2 - ndvi)
    soil_penalty = np.select(
        [soil_type == "peaty", soil_type == "sandy", soil_type == "clay"],
        [10.0, 6.0, -2.0],
        default=1.5,
    )

    fire_risk = dryness + fuel + soil_penalty + rng.normal(0, 6.5, n_samples)
    fire_risk = np.interp(fire_risk, (fire_risk.min(), fire_risk.max()), (0, 100))

    return pd.DataFrame(
        {
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "wind_speed_kmh": wind.round(2),
            "rainfall_mm": rainfall.round(2),
            "vegetation_index": ndvi.round(3),
            "soil_moisture_pct": soil_moisture.round(2),
            "slope_deg": slope.round(2),
            "tree_density_per_ha": tree_density.round(0).astype(int),
            "soil_type": soil_type,
            "fire_risk_score": fire_risk.round(2),
        }
    )


def build_pipeline() -> Pipeline:
    numerical = [c for c in FEATURE_COLUMNS if c != "soil_type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["soil_type"]),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_save(model_path: Path, data_path: Path) -> TrainResult:
    df = _generate_synthetic_dataset()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    X = df[FEATURE_COLUMNS]
    y = df["fire_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return TrainResult(mae=float(mae), r2=float(r2), model_path=model_path)


def load_or_train(model_path: Path, data_path: Path):
    if model_path.exists():
        return joblib.load(model_path)

    train_and_save(model_path=model_path, data_path=data_path)
    return joblib.load(model_path)


def predict_risk(model, sample: pd.DataFrame) -> Tuple[float, str]:
    score = float(model.predict(sample[FEATURE_COLUMNS])[0])
    score = max(0.0, min(100.0, score))

    if score < 35:
        label = "Low"
    elif score < 65:
        label = "Moderate"
    elif score < 85:
        label = "High"
    else:
        label = "Critical"

    return score, label
