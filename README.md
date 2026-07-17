# AI-Based Aerial Tree Mapping and Forest Fire Risk Prediction System

A complete, runnable portfolio project built for internship shortlisting (IIT Hyderabad style profile).

## 🚀 What this project demonstrates
- **Aerial tree detection** using **YOLOv8** (with a fallback CV detector so demo always runs).
- **Multi-source environmental fusion** style pipeline (imagery + climate/soil/terrain variables).
- **Forest fire risk prediction** model with explainable input factors.
- **Chatbot interface** for querying current risk insights.

## 🧰 Tech Stack
- Python
- YOLOv8 (`ultralytics`)
- OpenCV + Computer Vision
- Scikit-learn
- Streamlit

## 📁 Project Structure
```text
Tree-App/
├── app.py
├── requirements.txt
├── data/
├── models/
├── scripts/
│   ├── train_fire_risk_model.py
│   └── predict_risk.py
└── src/forest_ai/
    ├── __init__.py
    ├── detection.py
    ├── risk_model.py
    └── chatbot.py
```

## ⚙️ Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```


## 🎨 Frontend Design Highlights
- Glassmorphism-style dark hero section with technology chips.
- Sidebar-first control panel for environmental variables.
- Two-panel main layout: drone detection workspace + wildfire risk intelligence cards.
- Built-in synthetic aerial preview so UI can be showcased even before uploading data.

## ▶️ Run
### 1) Train fire risk model + generate sample dataset
```bash
python scripts/train_fire_risk_model.py
```

### 2) Quick CLI prediction
```bash
python scripts/predict_risk.py
```

### 3) Launch full web app
```bash
streamlit run app.py
```

Open the URL shown in terminal (typically `http://localhost:8501`).

## 🧪 Demo workflow
1. Upload an aerial drone image.
2. Inspect tree detections from YOLOv8/fallback detector.
3. Adjust environmental variables (temperature, humidity, wind, rainfall, soil, vegetation, density).
4. Get risk score (0-100) and severity class.
5. Ask chatbot questions like:
   - "What is the current fire risk?"
   - "Why is the risk high?"
   - "How many trees were detected?"

## 📌 Notes for real deployment
- Replace synthetic data generation with real GIS + satellite + weather APIs.
- Fine-tune YOLOv8 on local tree species datasets for species-level labels.
- Add temporal sequence models for spread prediction.
- Connect chatbot to live telemetry + alerting systems.

## 👨‍💻 Resume-ready project statement
Developed an AI-driven aerial forest monitoring system that combines YOLOv8-based tree detection, environmental feature fusion, and machine learning-based wildfire risk prediction, exposed via an interactive chatbot dashboard for near real-time decision support.
