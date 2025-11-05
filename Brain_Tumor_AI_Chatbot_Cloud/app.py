import os
import streamlit as st

# ... keep your existing imports above

st.set_page_config(page_title="Brain Tumor AI Chatbot", page_icon="🧠", layout="wide")

# === Load CSS robustly no matter where the app folder lives ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # graceful fallback so the app still runs
    st.markdown(
        "<style>body{font-family:Inter,system-ui,Arial,sans-serif;background:#0b1021;color:#e8ecf2}</style>",
        unsafe_allow_html=True,
    )
