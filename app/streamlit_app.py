"""
Streamlit Web App — Traffic Sign Detection
Run: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import time

from class_names import CLASS_NAMES
from data_loader import preprocess_single_image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #0f3460;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #e94560;
    }
    .confidence-high { color: #00d4aa; font-weight: bold; font-size: 1.4rem; }
    .confidence-low  { color: #ff6b6b; font-weight: bold; font-size: 1.4rem; }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "traffic_sign_model.keras")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, img_bgr: np.ndarray):
    tensor = preprocess_single_image(img_bgr)
    preds = model.predict(tensor, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return preds, top3_idx


def confidence_bar_chart(preds: np.ndarray, top3_idx):
    labels = [CLASS_NAMES[i] for i in top3_idx]
    values = [float(preds[i]) * 100 for i in top3_idx]
    colors = ["#e94560", "#0f3460", "#16213e"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 3 Predictions",
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 105]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-light.png", width=80)
    st.title("🚦 Traffic Sign\nDetection")
    st.markdown("---")
    mode = st.radio("Select Mode", ["📷 Image Upload", "🎥 Webcam (Live)"])
    st.markdown("---")
    st.markdown("**Model:** Custom CNN (GTSRB)")
    st.markdown("**Classes:** 43 traffic signs")
    st.markdown("**Dataset:** [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)")
    st.markdown("---")
    st.markdown("**Built by** [Sujal Sarnobat](https://github.com/sujalsarnobat)")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:white; margin:0">🚦 Traffic Sign Detection System</h1>
    <p style="color:#aaa; margin:0">Deep Learning · CNN · GTSRB Dataset · 43 Classes</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

if model is None:
    st.error(
        "⚠️ Model not found. Train it first:\n\n"
        "```bash\npython src/train.py --data_dir data/\n```"
    )
    st.stop()

# ── Image Upload Mode ─────────────────────────────────────────────────────────
if mode == "📷 Image Upload":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Choose a traffic sign image", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with col2:
        if uploaded:
            st.subheader("Prediction")
            with st.spinner("Analysing..."):
                preds, top3 = predict(model, img_bgr)
                time.sleep(0.3)  # slight delay for UX

            top_class = top3[0]
            top_conf = float(preds[top_class]) * 100
            top_label = CLASS_NAMES[top_class]

            conf_class = "confidence-high" if top_conf >= 60 else "confidence-low"
            emoji = "✅" if top_conf >= 60 else "⚠️"

            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color:white; margin:0">{emoji} {top_label}</h3>
                <p class="{conf_class}" style="margin:0.5rem 0">{top_conf:.1f}% confident</p>
                <p style="color:#aaa; margin:0">Class ID: {top_class}</p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(confidence_bar_chart(preds, top3), use_container_width=True)
        else:
            st.info("👈 Upload an image to get a prediction.")

# ── Webcam Mode ───────────────────────────────────────────────────────────────
elif mode == "🎥 Webcam (Live)":
    st.subheader("Real-Time Webcam Detection")
    st.info("📌 Hold a traffic sign image up to your camera for detection.")

    run = st.toggle("Start Webcam")
    FRAME_WINDOW = st.image([])
    result_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Make sure it is connected and not in use.")
        else:
            frame_count = 0
            last_label, last_conf = "Detecting...", 0.0

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost webcam feed.")
                    break

                if frame_count % 3 == 0:
                    preds, top3 = predict(model, frame)
                    last_conf = float(preds[top3[0]]) * 100
                    last_label = CLASS_NAMES[top3[0]]

                # Overlay
                color = (0, 255, 100) if last_conf >= 60 else (0, 100, 255)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), color, -1)
                cv2.putText(frame, f"{last_label}  {last_conf:.1f}%",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 2, cv2.LINE_AA)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, use_container_width=True)

                result_placeholder.markdown(
                    f"**Detected:** {last_label} &nbsp;|&nbsp; "
                    f"**Confidence:** {last_conf:.1f}%"
                )

                frame_count += 1

                if not run:
                    break

            cap.release()
    else:
        st.markdown("Toggle **Start Webcam** above to begin live detection.")
