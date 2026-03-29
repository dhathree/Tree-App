from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forest_ai.risk_model import FEATURE_COLUMNS, load_or_train, predict_risk


if __name__ == "__main__":
    model = load_or_train(ROOT / "models/fire_risk_model.joblib", ROOT / "data/sample_env_data.csv")

    sample = pd.DataFrame(
        [
            {
                "temperature_c": 36,
                "humidity_pct": 24,
                "wind_speed_kmh": 30,
                "rainfall_mm": 3,
                "vegetation_index": 0.35,
                "soil_moisture_pct": 20,
                "slope_deg": 18,
                "tree_density_per_ha": 680,
                "soil_type": "sandy",
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    score, label = predict_risk(model, sample)
    print(f"Predicted Fire Risk: {score:.2f} ({label})")
