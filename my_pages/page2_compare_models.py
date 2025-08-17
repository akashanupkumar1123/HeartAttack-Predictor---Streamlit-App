import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from utils import set_particle_background
set_particle_background()  # Apply animated particle background globally


def compare_models_page():
    """Renders the model comparison page with metrics table and accuracy visualization."""

    # -------------------- Custom CSS --------------------
    # Define custom page styles (background, spacing, buttons)
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;   /* Light gray background */
        }
        .block-container {
            padding-top: 2rem;           /* Add padding at the top */
        }
        .stButton > button {
            background-color: #ff4b4b;   /* Red buttons */
            color: white;
            border-radius: 8px;          /* Rounded corners */
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- Sidebar Info Box --------------------
    # Display the baseline model accuracy before improvements
    with st.sidebar:
        st.markdown("""
        <div style='
            background-color:#fffbe5;
            border-left: 8px solid #FFB300;
            padding: 1em;
            border-radius: 12px;
            margin-bottom: 1em;'>
            <h3 style='color:#E07A5F; margin:0;'>🌱 Base Model Accuracy</h3>
            <span style='font-size:2em; font-weight:bold; color:#43AA8B;'>72.8%</span>
            <br>
            <span style='color:#888;'>This was the starting point before tuning and improvements.</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- Page Title & Intro --------------------
    st.title("📊 Compare Model Performances")
    st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Explore and compare the performance of different trained models.")
    st.markdown("---")

    # -------------------- Load Model Comparison Data --------------------
    # Load the pre-saved DataFrame containing results of different models
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns", "final_model_comparisons.pkl"))
    df = joblib.load(data_path)

    # -------------------- Emoji Mapping for Models --------------------
    # Add emojis to model names for more visual appeal
    emoji_map = {
        "XGBoost_Optuna_Tuned": "🚀",
        "MODELSTACKED+META_LEARNER": "🧩",
        "MODELSTACKED+OPTUNA": "🔧",
        "MODELSTACKED_META_LEARNER+NEURAL_NETS": "🧠",
        "XGB_FEATURES": "📦",
        "LightGBM_XGB_FEATURES": "🌿",
        "PCA+XGB_FEATURES": "🔍",
        "LightGBM+PCA+XGB_FEATURES": "⚡",
        "LightGBM+PCA+XGB_FEATURES+OPTUNA": "🌱"
    }
    df["Model Name"] = df["Model Name"].apply(lambda x: f"{emoji_map.get(x, '')} {x}")

    # -------------------- Display Top 9 Models --------------------
    st.markdown("### ✅ Top 9 Models")

    # Select only key columns, sort by accuracy, reset index
    styled_df = (
        df[["Model Name", "Accuracy", "F1 Score"]]
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    styled_df.index += 1  # Index starts from 1 instead of 0
    styled_df.index.name = "Model No"

    # Display table with gradient color formatting
    st.dataframe(
        styled_df.style
            .background_gradient(subset=["Accuracy", "F1 Score"], cmap="Greens")
            .format({
                "Accuracy": lambda x: f"{x * 100:.2f}%",
                "F1 Score": lambda x: f"{x * 100:.2f}%"
            })
    )

    # -------------------- Plot Accuracy Comparison --------------------
    st.markdown("### 📈 Accuracy Comparison")

    # Add percentage column for clearer plot labels
    df["Accuracy %"] = df["Accuracy"] * 100

    # Create bar chart using Plotly Express
    fig = px.bar(
        df.sort_values(by="Accuracy", ascending=False),
        x="Model Name",
        y="Accuracy %",
        color="Accuracy %",
        color_continuous_scale="Teal",
        text_auto=".1f",  # Show labels with 1 decimal
        title="🔬 Accuracy Across Models",
        height=500
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        showlegend=False
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- Footer --------------------
    st.markdown("---")
    st.success("✅ Comparison complete! Pick the best model and proceed to insights 👉")
