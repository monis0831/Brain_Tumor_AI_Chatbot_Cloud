\
import os, requests
import numpy as np
import streamlit as st
from PIL import Image
from utils import simple_saliency

st.set_page_config(page_title="Brain Tumor AI Chatbot", page_icon="🧠", layout="wide")

# Load CSS
with open("assets/styles.css","r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Brain Tumor AI Chatbot — Cloud Demo")
st.caption("Upload an MRI image and type 'analyze'. Works in demo mode if TensorFlow model not available.")

MODEL_URL = os.environ.get("MODEL_URL", "").strip()
MODEL_PATH = "models/brain_tumor_cnn.h5"

@st.cache_data(show_spinner=False)
def download_model(url, dest):
    if not url: return False, "No MODEL_URL provided"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f: f.write(r.content)
        return True, "Model downloaded"
    except Exception as e:
        return False, str(e)

if MODEL_URL and not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        ok, msg = download_model(MODEL_URL, MODEL_PATH)
        st.info(msg)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def load_model():
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try: return tf.keras.models.load_model(MODEL_PATH)
        except Exception: return None
    return None

model = load_model()

st.sidebar.header("Upload MRI")
file = st.sidebar.file_uploader("Choose JPG/PNG", type=["jpg","jpeg","png"])
img_size = st.sidebar.slider("Image Size", 96, 256, 128, 16)
st.sidebar.write("Model:", "Loaded ✅" if model else "Demo ⚠️")

if "history" not in st.session_state:
    st.session_state["history"] = [{"role":"assistant", "content":"Hi! Upload an MRI and type **analyze** to predict."}]

for m in st.session_state["history"]:
    who = "You" if m["role"]=="user" else "Bot"
    st.markdown(f"<div class='card'><b>{who}:</b> {m['content']}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Type a message…")
if prompt:
    st.session_state["history"].append({"role":"user","content":prompt})
    q = prompt.lower().strip()

    if file and ("analyze" in q or "predict" in q):
        img = Image.open(file).convert("RGB").resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32)/255.0
        prob = 0.5; engine = "demo"
        if model is not None and TF_AVAILABLE:
            p = float(model.predict(np.expand_dims(arr,0), verbose=0)[0][0])
            prob = p; engine = "tensorflow"
        label = "Tumor Detected ✅" if prob >= 0.5 else "No Tumor ❌"
        st.image(img, caption="Input MRI", use_container_width=True)
        sal = simple_saliency(arr)
        import cv2
        sal255 = (sal*255).astype('uint8')
        sal255 = cv2.applyColorMap(sal255, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted((arr*255).astype('uint8'), 0.6, sal255, 0.4, 0)
        st.image(overlay, caption="Saliency Overlay", use_container_width=True)
        reply = f"**Prediction:** {label} (confidence {prob:.2%}) — engine: {engine}."
    else:
        if "accuracy" in q or "improve" in q:
            reply = "Use more MRI data, train longer, apply augmentation, or transfer learning. This demo runs on CPU."
        elif "what" in q:
            reply = "This is a cloud Streamlit chatbot for Brain Tumor MRI detection. Not for medical use."
        else:
            reply = "Upload an MRI on the left and type 'analyze'."
    st.session_state["history"].append({"role":"assistant","content":reply})
    st.rerun()
