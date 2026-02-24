import streamlit as st
import pandas as pd
import time
from preprocessing import detect_language, preprocess_pipeline
from model import ToxicityClassifier
from utils import create_summary_stats

# --- Page Configuration ---
st.set_page_config(
    page_title="Toxicity Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}
.result-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
}
.safe {
    color: #00c853;
    font-weight: bold;
}
.toxic {
    color: #ff4b4b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🛡️ Hate Speech & Toxicity Detector")
st.markdown("Detect **Toxic** comments in **English** and **Hindi** using AI.")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")

model_type = st.sidebar.radio(
    "Select Model",
    ["Advanced (Transformer)", "Baseline (TF-IDF)"]
)

use_transformer = True if "Transformer" in model_type else False

st.sidebar.success(f"Active Model: {model_type}")
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This tool uses NLP to identify toxic content for moderation systems.")

# --- Load Model Once ---
if "model" not in st.session_state:
    st.session_state["model"] = ToxicityClassifier()

model = st.session_state["model"]

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Analyze Text")
    text_input = st.text_area(
        "Enter comment:",
        height=150,
        placeholder="Type a comment to analyze..."
    )

    analyze_btn = st.button("Analyze Text", type="primary")

# --- Analysis ---
if analyze_btn and text_input.strip() != "":

    with st.spinner("Analyzing..."):
        time.sleep(0.5)

        # Detect Language
        lang = detect_language(text_input)

        # Predict
        result = model.predict(text_input, use_advanced=use_transformer)

    st.markdown("---")

    with col2:
        st.subheader("📊 Analysis Result")

        st.markdown(f"""
        <div class="result-card">
        <p><strong>Detected Language:</strong> {lang.upper()}</p>
        <p><strong>Prediction:</strong> 
        <span class="{ 'toxic' if result['is_toxic'] else 'safe' }">
        {result['label']}
        </span>
        </p>
        <p><strong>Confidence:</strong> {result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(result["confidence"] / 100)

        if result["is_toxic"]:
            st.error("⚠️ Toxic content detected. Consider moderation.")
        else:
            st.success("✅ Content appears safe.")

elif analyze_btn:
    st.warning("Please enter some text.")
