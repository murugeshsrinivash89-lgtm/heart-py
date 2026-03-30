import streamlit as st
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="VYNTARA", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
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
}
.result {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">VYNTARA AI Health Dashboard</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["💓 Monitor", "🧠 Stress Quiz"])

# ================= MONITOR =================
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

    # GRAPH
    t = np.linspace(0, 5, 200)
    signal = 0.6 * np.sin(2 * np.pi * (hr/60) * t)
    signal += 0.05 * np.random.randn(len(t))

    st.line_chart(signal)

    # TABLE
    st.subheader("📊 Live Data Table")

    data = pd.DataFrame({
        "Heart Rate": [hr],
        "RMSSD": [rmssd],
        "SDNN": [sdnn],
        "State": [state]
    })

    st.dataframe(data, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= QUIZ =================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧠 Advanced Stress Quiz")

    q1 = st.radio("1. Do you feel tired often?", ["Yes", "No"])
    q2 = st.radio("2. Sleep quality?", ["Good", "Average", "Bad"])
    q3 = st.radio("3. Mood today?", ["Happy", "Neutral", "Stressed"])
    q4 = st.radio("4. Do you feel anxious?", ["Yes", "Sometimes", "No"])
    q5 = st.radio("5. Workload level?", ["Low", "Medium", "High"])
    q6 = st.radio("6. Do you get headaches?", ["Yes", "No"])
    q7 = st.radio("7. Energy level?", ["High", "Medium", "Low"])
    q8 = st.radio("8. Focus level?", ["Good", "Average", "Poor"])

    score = 0

    if q1 == "Yes": score += 1
    if q2 == "Bad": score += 2
    if q3 == "Stressed": score += 2
    if q4 == "Yes": score += 2
    if q4 == "Sometimes": score += 1
    if q5 == "High": score += 2
    if q6 == "Yes": score += 1
    if q7 == "Low": score += 2
    if q8 == "Poor": score += 2

    # RESULT
    if score >= 8:
        result = "HIGH STRESS"
        color = "red"
    elif score >= 4:
        result = "MODERATE STRESS"
        color = "orange"
    else:
        result = "LOW STRESS"
        color = "green"

    st.markdown(f'<div class="result" style="color:{color}">{result}</div>', unsafe_allow_html=True)

    # TABLE RESULT
    st.subheader("📋 Quiz Summary")

    quiz_data = pd.DataFrame({
        "Score": [score],
        "Stress Level": [result]
    })

    st.table(quiz_data)

    st.markdown('</div>', unsafe_allow_html=True)
