# utils.py

import joblib
import pandas as pd
import numpy as np
import os
import streamlit as st  # Needed for set_particle_background()


# -------------------- 🌌 BACKGROUND PARTICLE ANIMATION -------------------- #

def set_particle_background():
    """
    Injects animated particle background into Streamlit UI using HTML + JS.
    Call this function at the top of any Streamlit page.
    """
    st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                background: transparent !important;
            }
            canvas#bgCanvas {
                position: fixed;
                top: 0;
                left: 0;
                z-index: -1;
                width: 100vw;
                height: 100vh;
            }
        </style>
        <canvas id="bgCanvas"></canvas>
        <script>
        const canvas = document.getElementById("bgCanvas");
        const ctx = canvas.getContext("2d");
        let w = window.innerWidth;
        let h = window.innerHeight;
        canvas.width = w;
        canvas.height = h;

        // Particle system setup
        const particles = [];
        const numParticles = 120;

        for (let i = 0; i < numParticles; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * 1.2,
                vy: (Math.random() - 0.5) * 1.2,
                radius: Math.random() * 2.2 + 1
            });
        }

        // Animation loop
        function draw() {
            ctx.clearRect(0, 0, w, h);
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255, 75, 75, 0.5)";  // red tint
                ctx.fill();

                // Move particle
                p.x += p.vx;
                p.y += p.vy;

                // Bounce off walls
                if (p.x < 0 || p.x > w) p.vx *= -1;
                if (p.y < 0 || p.y > h) p.vy *= -1;
            }
            requestAnimationFrame(draw);
        }

        draw();

        // Responsive canvas
        window.onresize = () => {
            w = window.innerWidth;
            h = window.innerHeight;
            canvas.width = w;
            canvas.height = h;
        };
        </script>
    """, unsafe_allow_html=True)


# -------------------- 💾 MODEL LOADING -------------------- #

@st.cache_resource
def load_model(model_path="models/final_7_feature_lgbm.pkl"):
    """
    Loads a trained ML model (LightGBM in this case) from disk.

    Args:
        model_path (str): Relative path to the model inside project.
    
    Returns:
        Loaded model object (e.g., LightGBM).
    """
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model_path))
    try:
        model = joblib.load(abs_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Model not found at {abs_path}")


# -------------------- 📊 DATA PREPARATION -------------------- #

def preprocess_input(user_input: pd.DataFrame, feature_list: list):
    """
    Ensures the user-provided input has the exact features required by the model.

    Args:
        user_input (pd.DataFrame): Raw input data from user.
        feature_list (list): Ordered list of features required for model training.

    Returns:
        pd.DataFrame: Cleaned and reordered input data ready for prediction.
    """
    missing = [col for col in feature_list if col not in user_input.columns]
    if missing:
        raise ValueError(f"🚨 Missing required input features: {missing}")
    
    return user_input[feature_list]


# -------------------- 🔮 PREDICTION -------------------- #

def predict_risk(model, input_data: pd.DataFrame):
    """
    Makes prediction using the trained model.

    Args:
        model: Trained model with `predict` and `predict_proba`.
        input_data (pd.DataFrame): Preprocessed input features.

    Returns:
        tuple: (probability of positive class, predicted class label)
    """
    proba = model.predict_proba(input_data)[0][1]  # Probability of class 1
    pred_class = model.predict(input_data)[0]
    return proba, pred_class


# -------------------- 📁 MODEL COMPARISON DATA -------------------- #

def load_model_comparison_df():
    """
    Loads the saved MLflow model comparison DataFrame.

    Returns:
        pd.DataFrame: Experiment results containing Accuracy, F1 Score, etc.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(current_dir, "..", "mlruns", "final_model_comparisons.pkl"))

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found at: {data_path}")
    
    return joblib.load(data_path)


# -------------------- 🎯 METRICS FORMATTER -------------------- #

def format_metrics(acc, f1=None):
    """
    Formats metrics into a human-readable dictionary for display.

    Args:
        acc (float): Accuracy score (0–1).
        f1 (float): F1 score (0–1).

    Returns:
        dict: Formatted accuracy & F1 score with emojis.
    """
    return {
        "🧠 Accuracy": f"{acc*100:.2f}%" if acc else "N/A",
        "📈 F1-Score": f"{f1*100:.2f}%" if f1 else "N/A"
    }
