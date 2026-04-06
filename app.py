import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.big-title {font-size:30px; font-weight:bold;}
.result {font-size:24px; font-weight:bold;}
.card {padding:20px; border-radius:10px; background:#111;}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
def train_model():
    data = []
    for _ in range(300):
        sample = {
            'brightness': np.random.rand()*255,
            'contrast': np.random.rand(),
            'red_mean': np.random.rand()*255,
            'green_mean': np.random.rand()*255,
            'blue_mean': np.random.rand()*255,
            'label': np.random.randint(0,2)
        }
        data.append(sample)

    df = pd.DataFrame(data)
    X = df.drop('label', axis=1)
    y = df['label']

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/skin_model.pkl")

def load_model():
    if not os.path.exists("models/skin_model.pkl"):
        train_model()
    return joblib.load("models/skin_model.pkl")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(image):
    img = cv2.resize(image,(100,100))

    brightness = np.mean(img)
    contrast = np.std(img)

    mean = img.mean(axis=(0,1))

    return {
        'brightness': brightness,
        'contrast': contrast,
        'blue_mean': mean[0],
        'green_mean': mean[1],
        'red_mean': mean[2]
    }

# =========================
# TITLE
# =========================
st.markdown('<div class="big-title">AI Skin Health Dashboard</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Analyzer","📊 Report"])

# =========================
# TAB 1
# =========================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Skin Image", type=["jpg","png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
        image = cv2.imdecode(file_bytes,1)

        st.image(image, use_column_width=True)

        if st.button("Analyze"):
            model = load_model()

            features = extract_features(image)
            X = np.array(list(features.values())).reshape(1,-1)

            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]

            st.subheader("Result")
            st.write(f"Probability: {prob:.2f}")

            if pred==1:
                st.error("⚠️ Acne / Skin Issue Detected")
            else:
                st.success("✅ Healthy Skin")

            st.session_state['result'] = pred
            st.session_state['prob'] = prob

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if 'result' in st.session_state:
        result = st.session_state['result']
        prob = st.session_state['prob']

        if result==1:
            state="Skin Issue"
            color="red"
        else:
            state="Healthy"
            color="green"

        st.markdown(f'<div class="result" style="color:{color}">{state}</div>', unsafe_allow_html=True)

        df = pd.DataFrame({
            "Condition":[state],
            "Probability":[prob]
        })

        st.table(df)

    else:
        st.write("Run analysis first...")

    st.markdown('</div>', unsafe_allow_html=True)