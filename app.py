from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.forest_ai.chatbot import RiskContext, answer
from src.forest_ai.detection import run_tree_detection
from src.forest_ai.risk_model import FEATURE_COLUMNS, load_or_train, predict_risk

st.set_page_config(page_title="Aerial Forest AI", page_icon="🌲", layout="wide")

MODEL_PATH = Path("models/fire_risk_model.joblib")
DATA_PATH = Path("data/sample_env_data.csv")


@st.cache_resource
def get_model():
    return load_or_train(MODEL_PATH, DATA_PATH)


model = get_model()

st.title("🌲 AI-Based Aerial Tree Mapping & Forest Fire Risk Prediction")
st.caption("YOLOv8 + Computer Vision + Environmental Risk Modeling + Chatbot")

left, right = st.columns([1.3, 1])

with left:
    st.subheader("1) Aerial Tree Detection")
    uploaded = st.file_uploader("Upload drone image", type=["jpg", "jpeg", "png"])

    detections_count = 0
    inference_engine = "Not run"

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        image_np = np.array(img)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        annotated_bgr, detections, inference_engine = run_tree_detection(image_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        detections_count = len(detections)

        st.image([image_np, annotated_rgb], caption=["Input", f"Detections via {inference_engine}"], use_container_width=True)

        if detections:
            table = pd.DataFrame(
                [
                    {"class": d.cls, "confidence": round(d.confidence, 3), "bbox": d.bbox_xyxy}
                    for d in detections[:100]
                ]
            )
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No trees detected in this frame. Try a greener aerial image.")

with right:
    st.subheader("2) Forest Fire Risk Predictor")
    defaults = {
        "temperature_c": 34.0,
        "humidity_pct": 28.0,
        "wind_speed_kmh": 21.0,
        "rainfall_mm": 8.0,
        "vegetation_index": 0.41,
        "soil_moisture_pct": 24.0,
        "slope_deg": 17.0,
        "tree_density_per_ha": 620,
        "soil_type": "sandy",
    }

    values = {}
    values["temperature_c"] = st.slider("Temperature (°C)", 10.0, 50.0, defaults["temperature_c"])
    values["humidity_pct"] = st.slider("Humidity (%)", 5.0, 100.0, defaults["humidity_pct"])
    values["wind_speed_kmh"] = st.slider("Wind Speed (km/h)", 0.0, 90.0, defaults["wind_speed_kmh"])
    values["rainfall_mm"] = st.slider("Recent Rainfall (mm)", 0.0, 300.0, defaults["rainfall_mm"])
    values["vegetation_index"] = st.slider("Vegetation Index (NDVI)", 0.0, 1.0, defaults["vegetation_index"])
    values["soil_moisture_pct"] = st.slider("Soil Moisture (%)", 0.0, 100.0, defaults["soil_moisture_pct"])
    values["slope_deg"] = st.slider("Terrain Slope (°)", 0.0, 50.0, defaults["slope_deg"])
    values["tree_density_per_ha"] = st.slider("Tree Density (per ha)", 50, 1500, defaults["tree_density_per_ha"])
    values["soil_type"] = st.selectbox("Soil Type", ["sandy", "clay", "loamy", "peaty", "silty"], index=0)

    sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    risk_score, risk_label = predict_risk(model, sample)

    color = {
        "Low": "#2e7d32",
        "Moderate": "#f9a825",
        "High": "#ef6c00",
        "Critical": "#c62828",
    }[risk_label]

    st.markdown(
        f"""
        <div style='padding: 12px; border-radius: 10px; background: #0e1117; border: 1px solid {color};'>
          <h3 style='margin: 0; color: {color};'>Risk Score: {risk_score:.1f}/100 ({risk_label})</h3>
          <p style='margin-top: 8px;'>Higher values indicate more favorable conditions for wildfire ignition/spread.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("3) Chatbot for Real-Time Risk Insights")
question = st.text_input("Ask about current risk, tree detections, or key risk drivers:")

context = RiskContext(
    risk_score=risk_score,
    risk_label=risk_label,
    tree_detections=detections_count,
    inference_engine=inference_engine,
)

if question.strip():
    st.success(answer(question, context))
else:
    st.caption("Example: 'Why is the risk high today?'")
