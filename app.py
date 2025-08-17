# streamlit_app/app.py

import streamlit as st
from streamlit_option_menu import option_menu
from utils import load_model, format_metrics, set_particle_background 
import os

# -------------------- Page Config --------------------
# Configure the main Streamlit app settings
st.set_page_config(
    page_title="Heart Risk Predictor",   # Browser tab title
    page_icon="💓",                      # Tab icon (emoji)
    layout="wide",                       # Use wide screen layout
    initial_sidebar_state="expanded"     # Sidebar starts expanded
)

# -------------------- Custom Sidebar Styling --------------------
# Apply custom CSS for sidebar, buttons, sliders, and fonts
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #ffe5e5;  /* Soft red background */
    color: #222222;            /* Dark text */
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #b30000 !important;  /* Deep red headings */
}

[data-testid="stSidebar"] .css-1d391kg,  
[data-testid="stSidebar"] .css-hxt7ib {
    color: #222222 !important;
}

.stButton > button {
    background-color: #ff4b4b;   /* Button color */
    color: white;
    border-radius: 12px;         /* Rounded corners */
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
    transition: background-color 0.3s ease;
    font-family: 'Segoe UI', sans-serif;
}

.stButton > button:hover {
    background-color: #e04343;   /* Darker hover effect */
}

.stSlider > div {
    color: #b30000;              /* Slider text color */
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
# Add branding, app title, and description to sidebar
with st.sidebar:
    st.image("assets/heart.gif", width=120)  # Heart animation/logo
    st.markdown("## ❤️ **Self Heart Risk Predictor**")
    st.markdown("### 🧠 *AI-Powered Health Companion*")
    st.markdown("---")
    st.markdown("Built with Streamlit, MLflow, and SHAP")

# -------------------- Main Navigation --------------------
# Sticky navigation menu at the top (horizontal)
st.markdown('<div class="option-menu-sticky">', unsafe_allow_html=True)

# Navigation bar with options
selected = option_menu(
    menu_title=None,
    options=["🏡 Home", "🩺 Predictor", "📊 Compare", "🔍 SHAP Insights", "📁 MLflow Stats"],
    icons=["house", "activity", "bar-chart", "search", "folder2"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "#f9f9f9", "padding": "5px"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Page Imports --------------------
# Import page-specific modules
from my_pages.page1_predictor import predictor_page
from my_pages.page2_compare_models import compare_models_page
from my_pages.page3_shap_insights import shap_insights_page
from my_pages.page4_mlflow_stats import mlflow_stats_page

# -------------------- Routing --------------------
# Display content based on selected page
if selected == "🏡 Home":
    # Inject background style for Home page only
    st.markdown("""
        <style>
        body {
            background-color: #fff5f5 !important;  /* Soft rose background */
        }

        [data-testid="stAppViewContainer"] {
            background-color: #fff5f5 !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

    set_particle_background()  # Optional animated background particles

    # Home page UI
    st.markdown("<h1 style='text-align: center;'>💖 Welcome to Heart Risk Predictor</h1>", unsafe_allow_html=True)
    st.image("assets/welcome_banner.gif", use_container_width=True)
    st.markdown("""
    <div style='text-align: center; font-size: 20px; margin-top: 20px;'>
        Empowering lives with AI.<br><br>Choose a page above to begin exploring your health insights!
    </div>
    """, unsafe_allow_html=True)

elif selected == "🩺 Predictor":
    predictor_page()  # Load predictor page

elif selected == "📊 Compare":
    compare_models_page()  # Load model comparison page

elif selected == "🔍 SHAP Insights":
    shap_insights_page()  # Load SHAP insights page

elif selected == "📁 MLflow Stats":
    mlflow_stats_page()  # Load MLflow stats page
