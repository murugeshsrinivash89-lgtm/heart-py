import streamlit as st
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="VYNTARA", layout="wide")

# CSS (premium UI)
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.big-title {
    font-size: 45px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.5);
}
.result {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">VYNTARA AI Health Dashboard</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💓 Monitor", "🧠 Stress Quiz"])

# ---------------- TAB 1 ----------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        hr = st.slider("Heart Rate", 50, 150, 80)
    with col2:
        rmssd = st.slider("RMSSD", 0.01, 0.1, 0.05)
    with col3:
        sdnn = st.slider("SDNN", 0.01, 0.1, 0.05)

    # Logic
    if hr > 110:
        state = "HIGH HR"
        color = "red"
    elif rmssd < 0.04 and sdnn < 0.05:
        state = "STRESS"
        color = "orange"
    else:
        state = "NORMAL"
        color = "green"

    st.markdown(f'<div class="result" style="color:{color}">{state}</div>', unsafe_allow_html=True)

    # Graph
    t = np.linspace(0, 5, 200)
    signal = 0.6 * np.sin(2 * np.pi * (hr/60) * t)
    signal += 0.05 * np.random.randn(len(t))

    st.line_chart(signal)

    # Table (DATA LOG)
    st.subheader("📊 Live Data Table")

    data = pd.DataFrame({
        "Heart Rate": [hr],
        "RMSSD": [rmssd],
        "SDNN": [sdnn],
        "State": [state]
    })

    st.dataframe(data, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧠 Stress Level Quiz")

    q1 = st.radio("Do you feel tired?", ["Yes", "No"])
    q2 = st.radio("Sleep quality?", ["Good", "Bad"])
    q3 = st.radio("Mood today?", ["Happy", "Stressed"])

    score = 0
    if q1 == "Yes":
        score += 1
    if q2 == "Bad":
        score += 1
    if q3 == "Stressed":
        score += 1

    if score >= 2:
        result = "High Stress"
        color = "red"
    else:
        result = "Low Stress"
        color = "green"

    st.markdown(f'<div class="result" style="color:{color}">{result}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
