import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="VYNTARA")

# UI CSS
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}
.result {
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">Heart Monitor</div>', unsafe_allow_html=True)

# Input
hr = st.slider("Heart Rate (BPM)", 50, 150, 80)
rmssd = st.slider("RMSSD", 0.01, 0.1, 0.05)
sdnn = st.slider("SDNN", 0.01, 0.1, 0.05)

# Result logic
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

# GRAPH (Streamlit built-in)
t = np.linspace(0, 5, 200)
signal = 0.6 * np.sin(2 * np.pi * (hr/60) * t)
signal += 0.05 * np.random.randn(len(t))

st.line_chart(signal)
