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


def generate_demo_aerial() -> np.ndarray:
    """Create a synthetic aerial-like image for UI preview without upload."""
    canvas = np.full((520, 900, 3), (58, 88, 48), dtype=np.uint8)

    # Terrain patches
    cv2.rectangle(canvas, (0, 0), (900, 220), (67, 105, 56), -1)
    cv2.rectangle(canvas, (0, 220), (900, 380), (73, 95, 63), -1)
    cv2.rectangle(canvas, (0, 380), (900, 520), (78, 86, 70), -1)

    # Roads/fire-lines
    cv2.line(canvas, (30, 460), (860, 90), (176, 170, 140), 8)
    cv2.line(canvas, (150, 500), (840, 260), (160, 155, 125), 6)

    # Tree blobs
    rng = np.random.default_rng(7)
    for _ in range(230):
        x = int(rng.integers(10, 890))
        y = int(rng.integers(10, 510))
        r = int(rng.integers(5, 14))
        color = (int(rng.integers(35, 70)), int(rng.integers(95, 170)), int(rng.integers(35, 70)))
        cv2.circle(canvas, (x, y), r, color, -1)

    return canvas


def render_header() -> None:
    st.markdown(
        """
        <style>
          .main {background: linear-gradient(180deg, #071013 0%, #0f1720 65%, #101826 100%);} 
          .hero {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid #2d3a40;
            background: linear-gradient(135deg, rgba(27,42,34,0.9), rgba(20,30,42,0.9));
            margin-bottom: 1rem;
          }
          .chip {
            display: inline-block;
            margin: 0.2rem 0.35rem 0.2rem 0;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid #2d5a4b;
            background: #11281f;
            font-size: 0.8rem;
            color: #9fe8c2;
          }
          .metric-card {
            border: 1px solid #243648;
            border-radius: 12px;
            padding: 0.8rem;
            background: rgba(15, 21, 29, 0.75);
          }
        </style>
        <div class="hero">
            <h2 style="margin:0 0 0.3rem 0;">🌲 AI-Based Aerial Tree Mapping & Fire Risk Intelligence</h2>
            <p style="margin:0;color:#b7c9d8;">Internship-ready full-stack ML demo combining drone vision, environmental analytics, and interactive risk insights.</p>
            <div style="margin-top:0.45rem;">
              <span class="chip">YOLOv8</span>
              <span class="chip">Computer Vision</span>
              <span class="chip">Risk Modeling</span>
              <span class="chip">Interactive Chatbot</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(score: float, label: str) -> str:
    color = {
        "Low": "#2e7d32",
        "Moderate": "#f9a825",
        "High": "#ef6c00",
        "Critical": "#c62828",
    }[label]
    return f"""
    <div class='metric-card' style='border-color:{color};'>
      <h3 style='margin: 0; color: {color};'>Risk Score: {score:.1f}/100 ({label})</h3>
      <p style='margin:0.4rem 0 0 0;color:#b8c5d0;'>Higher score means stronger ignition/spread conditions.</p>
    </div>
    """


model = get_model()
render_header()

with st.sidebar:
    st.header("⚙️ Environmental Inputs")
    st.caption("Tune these values to simulate local field conditions.")

    values = {
        "temperature_c": st.slider("Temperature (°C)", 10.0, 50.0, 34.0),
        "humidity_pct": st.slider("Humidity (%)", 5.0, 100.0, 28.0),
        "wind_speed_kmh": st.slider("Wind Speed (km/h)", 0.0, 90.0, 21.0),
        "rainfall_mm": st.slider("Recent Rainfall (mm)", 0.0, 300.0, 8.0),
        "vegetation_index": st.slider("Vegetation Index (NDVI)", 0.0, 1.0, 0.41),
        "soil_moisture_pct": st.slider("Soil Moisture (%)", 0.0, 100.0, 24.0),
        "slope_deg": st.slider("Terrain Slope (°)", 0.0, 50.0, 17.0),
        "tree_density_per_ha": st.slider("Tree Density (per ha)", 50, 1500, 620),
        "soil_type": st.selectbox("Soil Type", ["sandy", "clay", "loamy", "peaty", "silty"], index=0),
    }

sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
risk_score, risk_label = predict_risk(model, sample)

main_left, main_right = st.columns([1.45, 1])

with main_left:
    st.subheader("1) Drone Imagery Tree Detection")
    uploaded = st.file_uploader("Upload aerial image (jpg/png)", type=["jpg", "jpeg", "png"])

    detections_count = 0
    inference_engine = "Not run"

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        image_np = np.array(img)
    else:
        image_np = cv2.cvtColor(generate_demo_aerial(), cv2.COLOR_BGR2RGB)
        st.caption("Showing synthetic demo aerial image. Upload your own drone frame for real inference.")

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    annotated_bgr, detections, inference_engine = run_tree_detection(image_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    detections_count = len(detections)

    st.image([image_np, annotated_rgb], caption=["Input Aerial Frame", f"Detection Output ({inference_engine})"], use_container_width=True)

    if detections:
        table = pd.DataFrame(
            [{"class": d.cls, "confidence": round(d.confidence, 3), "bbox": d.bbox_xyxy} for d in detections[:60]]
        )
        st.dataframe(table, use_container_width=True)
    else:
        st.info("No vegetation clusters detected. Try a greener image.")

with main_right:
    st.subheader("2) Wildfire Risk Panel")
    st.markdown(risk_badge(risk_score, risk_label), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Temp", f"{values['temperature_c']:.1f}°C")
    c2.metric("Humidity", f"{values['humidity_pct']:.1f}%")
    c3.metric("Wind", f"{values['wind_speed_kmh']:.1f} km/h")

    c4, c5, c6 = st.columns(3)
    c4.metric("Soil Moisture", f"{values['soil_moisture_pct']:.1f}%")
    c5.metric("NDVI", f"{values['vegetation_index']:.2f}")
    c6.metric("Tree Density", f"{values['tree_density_per_ha']}")

    st.markdown("---")
    st.subheader("3) Insight Chatbot")
    question = st.text_input("Ask: risk now, why high, detected trees, etc.")

    context = RiskContext(
        risk_score=risk_score,
        risk_label=risk_label,
        tree_detections=detections_count,
        inference_engine=inference_engine,
    )

    if question.strip():
        st.success(answer(question, context))
    else:
        st.caption("Try: 'Why is the fire risk high?' or 'How many trees were detected?'")
