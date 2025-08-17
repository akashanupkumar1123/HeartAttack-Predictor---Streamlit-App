import streamlit as st
import pandas as pd
import os
import joblib

from utils import set_particle_background
set_particle_background()  # Apply animated particle background globally


def mlflow_stats_page():
    """Renders the MLflow experiment statistics page with results, rankings, and reflections."""

    # -------------------- CSS Styling --------------------
    # Custom styling for main content, buttons, and sliders
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;   /* Light background */
        }
        .block-container {
            padding-top: 2rem;           /* Add spacing at top */
        }
        .stButton > button {
            background-color: #ff4b4b;   /* Red buttons */
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .stSlider > div {
            color: #ff4b4b;              /* Red slider text */
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- Title & Intro --------------------
    st.title("📁 MLflow Experiment Stats")
    st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)
    st.markdown("Explore the results from your top experiments tracked via **MLflow** 🧪")

    # -------------------- Load Model Comparison Data --------------------
    # Load pre-computed experiment results (stored with joblib)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns", "final_model_comparisons.pkl"))
    df = joblib.load(data_path)

    # -------------------- Rename & Rank --------------------
    # Standardize column names and add ranking index
    df.columns = ["Run ID", "Accuracy", "F1 Score", "Model Name"]
    df = df.copy()
    df.index = range(1, len(df) + 1)   # Rank models starting at 1
    df.index.name = "Model Rank"

    # -------------------- Display Styled Table --------------------
    # Format Accuracy & F1 with 4 decimals + gradient coloring
    styled_df = (
        df.style
        .format({
            "Accuracy": "{:.4f}",
            "F1 Score": "{:.4f}"
        })
        .background_gradient(cmap='Blues', subset=["Accuracy", "F1 Score"])
    )
    st.dataframe(styled_df, use_container_width=True, height=500)

    # -------------------- Show Best Model --------------------
    # Identify top-performing model (based on Accuracy)
    top_model = df.loc[df["Accuracy"].idxmax()]
    st.success(
        f"🏆 **Best Model:** `{top_model['Model Name']}` "
        f"with **Accuracy: {top_model['Accuracy']:.4f}** "
        f"and **F1 Score: {top_model['F1 Score']:.4f}**"
    )

    st.markdown("---")

    # -------------------- Project Summary --------------------
    # Expandable section to reflect on project outcomes
    with st.expander("📘 Project Summary & Reflection"):
        st.markdown("""
        ### 🧠 Summary:
        - Multiple models were trained using LightGBM, CatBoost, XGBoost, PCA + Feature Engineering, and stacking techniques.  
        - The best performance was achieved by combining PCA + important features with Optuna-tuned LightGBM.  
        - **SHAP** was used to explain model behavior, while **MLflow** ensured tracking of each experiment.  

        ### ✅ Final Words:
        You now have a full pipeline:  
        **Data ➡️ ML ➡️ Explainability ➡️ Dashboard** 🎯  

        Made with ❤️ using Streamlit, Optuna, SHAP, and MLflow.
        """)

    # -------------------- Fun Ending --------------------
    # Celebrate completion with Streamlit balloons 🎈
    st.balloons()
