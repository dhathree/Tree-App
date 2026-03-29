from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forest_ai.risk_model import train_and_save


if __name__ == "__main__":
    result = train_and_save(
        model_path=ROOT / "models/fire_risk_model.joblib",
        data_path=ROOT / "data/sample_env_data.csv",
    )
    print(f"Model saved to: {result.model_path}")
    print(f"MAE: {result.mae:.3f}")
    print(f"R2:  {result.r2:.3f}")
