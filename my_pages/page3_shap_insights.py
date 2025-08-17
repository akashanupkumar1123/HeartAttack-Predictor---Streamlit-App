import streamlit as st
from PIL import Image
import os

from utils import set_particle_background
set_particle_background()  # Apply animated particle background globally


def shap_insights_page():
    """Renders the SHAP insights page with static plots for explainability."""

    # -------------------- Custom CSS --------------------
    # Styling for main content area, buttons, and sliders
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;   /* Light gray background */
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

    # -------------------- Page Title & Intro --------------------
    st.title("🔍 SHAP Insights")
    st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)

    st.markdown("## 🔍 Model Explainability with SHAP")
    st.markdown("Get a peek into **how your model thinks** using SHAP values! 🎯")
    st.markdown("---")

    # -------------------- Sidebar Explanation --------------------
    # Collapsible sidebar section explaining SHAP for non-technical users
    with st.sidebar.expander("🧠 What is SHAP?"):
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values help explain **how each feature contributes** 
        to predictions made by the model.  
        - Positive SHAP → pushes prediction higher  
        - Negative SHAP → pulls prediction lower  
        - Bigger magnitude → more influence  
        """)

    # -------------------- Define SHAP Plot Paths --------------------
    # Paths to static SHAP summary images
    bar_path = os.path.join(os.path.dirname(__file__), "..", "shap_plots", "shap_summary_bar.png")
    dot_path = os.path.join(os.path.dirname(__file__), "..", "shap_plots", "shap_summary_dot.png")

    # -------------------- SHAP Plot Toggle --------------------
    # Allow user to switch between bar and dot plot
    plot_choice = st.radio(
        "🎨 Select SHAP Plot Type:",
        ["📊 Summary Bar Plot", "🌈 Summary Dot Plot"],
        horizontal=True
    )

    # -------------------- Display Selected Plot --------------------
    if plot_choice == "📊 Summary Bar Plot":
        st.image(
            Image.open(bar_path),
            caption="Top Features - Mean SHAP Value (Bar)",
            use_container_width=True
        )
    else:
        st.image(
            Image.open(dot_path),
            caption="SHAP Summary - Feature Impact (Dot)",
            use_container_width=True
        )

    # -------------------- Footer --------------------
    st.markdown("---")
    st.success("📈 SHAP plots are ready to be explored. Try hovering or zooming when using interactive versions!")

    # Friendly closing note
    st.markdown("##### 🧬 SHAP gives life to numbers — now you're looking at your model's *thought process*! 💭")
