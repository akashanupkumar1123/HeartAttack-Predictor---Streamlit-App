import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

from streamlit_lottie import st_lottie
from streamlit_extras.metric_cards import style_metric_cards

from utils import set_particle_background
set_particle_background()  # Apply animated particle background globally


# -------------------- Custom CSS --------------------
# Define custom styles for background, buttons, sliders
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;   /* Light gray background */
    }
    .block-container {
        padding-top: 2rem;           /* Extra spacing at top */
    }
    .stButton > button {
        background-color: #ff4b4b;   /* Red button background */
        color: white;                /* White text */
        border-radius: 8px;          /* Rounded corners */
        font-weight: bold;           /* Bold text */
    }
    .stSlider > div {
        color: #ff4b4b;              /* Slider text in red */
    }
</style>
""", unsafe_allow_html=True)


# -------------------- Page Function --------------------
def predictor_page():
    """Renders the heart risk prediction page with form inputs, ML model prediction, and results."""
    
    st.title("💖 Smart Risk Predictor")

    # ----------------- Lottie Loader -----------------
    # Utility function to load JSON-based Lottie animations
    def load_lottie_file(filename):
        try:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
            file_path = os.path.join(base_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # Load and display animated heart
    lottie_heart = load_lottie_file("heart_lottie.json")
    if lottie_heart:
        st_lottie(lottie_heart, height=250, key="heart")
    else:
        st.warning("⚠️ Could not load heart animation.")

    # ----------------- Load Model -----------------
    # Load trained LightGBM model (7-feature version)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "final_7_feature_lgbm.pkl"))
    model = joblib.load(model_path)

    # ----------------- Info Expander -----------------
    # Provide explanation of how predictions are made
    with st.expander("ℹ️ How this works"):
        st.info("""
            This tool uses a machine learning model trained on medical features 
            to estimate the risk of a heart attack.
            Enter your health information below to receive a risk prediction score and status.
        """)

    # ----------------- Input Form -----------------
    # Collect user health data through sliders and dropdowns
    with st.form("predict_form"):
        st.markdown("### 📝 Enter Your Health Info Below")

        # Split inputs into 2 columns for better UI
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Age", 18, 100, 45)
            sys_bp = st.slider("🩺 Systolic BP", 80, 200, 120)

            cholesterol_options = {
                "1️⃣ Normal": 1,
                "2️⃣ Above Normal": 2,
                "3️⃣ Well Above Normal": 3
            }
            chol_label = st.selectbox("🧈 Cholesterol Level", list(cholesterol_options.keys()))
            chol = cholesterol_options[chol_label]

            bmi = st.slider("⚖️ BMI", 15.0, 45.0, 25.0)

        with col2:
            glucose_options = {
                "1️⃣ Normal": 1,
                "2️⃣ Above Normal": 2,
                "3️⃣ Well Above Normal": 3
            }
            glucose_label = st.selectbox("🍭 Glucose Level", list(glucose_options.keys()))
            glucose = glucose_options[glucose_label]

            gender = st.selectbox("🛋 Gender", ["Male", "Female"])
            smokes = st.selectbox("🚬 Smokes", ["No", "Yes"])

        # Submit button
        submit = st.form_submit_button("💡 Predict Risk", type="primary")

    # ----------------- Prediction -----------------
    if submit:
        # Prepare input data in DataFrame format for model
        input_df = pd.DataFrame({
            "age_years": [age],
            "systolic_bp": [sys_bp],
            "cholesterol_level": [chol],
            "bmi": [bmi],
            "glucose_level": [glucose],
            "gender": [1 if gender == "Male" else 0],
            "smokes": [1 if smokes == "Yes" else 0]
        })

        try:
            # Get probability of high risk from model
            pred_prob = model.predict_proba(input_df)[0][1]
            prediction = int(pred_prob > 0.5)  # Threshold = 0.5

            # ----------------- Display Results -----------------
            st.subheader("🎯 Prediction Result")
            st.metric(label="🧠 Risk Probability", value=f"{round(pred_prob * 100, 2)} %")

            if prediction == 1:
                # High risk
                st.markdown("## ❗💔 **High Risk Detected!**")
                st.error("Please consult a cardiologist at the earliest.")
                st.markdown("### 🔴 Stay cautious. Your heart matters!")
            else:
                # Low risk
                st.markdown("## ✅💖 **Low Risk**")
                st.success("You seem healthy! Keep maintaining your lifestyle.")
                st.markdown("### 🟢 Keep it up and stay active!")

            # Style the metric card
            style_metric_cards(border_left_color="#D61355", background_color="#FAF0F3", border_radius_px=5)

        except Exception as e:
            # Catch errors if model prediction fails
            st.error(f"🚫 Prediction failed: {e}")
