import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="VYNTARA", page_icon="❤️", layout="centered")

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
st.markdown('<div class="big-title">❤️ VYNTARA Heart Analyzer</div>', unsafe_allow_html=True)
st.write("")

# Input card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    hr = st.number_input("💓 Heart Rate (BPM)", min_value=0.0)
    rmssd = st.number_input("📊 RMSSD", min_value=0.0)
    sdnn = st.number_input("📉 SDNN", min_value=0.0)

    st.write("")

    if st.button("🚀 Analyze"):

        # RESULT LOGIC
        if hr > 110:
            state = "HIGH HR"
            color = "red"
        elif rmssd < 0.04 and sdnn < 0.05:
            state = "STRESS"
            color = "orange"
        else:
            state = "NORMAL"
            color = "green"

        st.markdown(f'<div class="result" style="color:{color};">⚡ {state}</div>', unsafe_allow_html=True)

        # GRAPH (heartbeat simulation)
        t = np.linspace(0, 5, 500)
        signal = 0.6 * np.sin(2 * np.pi * (hr/60) * t)

        # small noise
        signal += 0.05 * np.random.randn(len(t))

        fig, ax = plt.subplots()
        ax.plot(t, signal)
        ax.set_title("Simulated Heart Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
