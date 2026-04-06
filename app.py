import streamlit as st
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Health Assistant", layout="centered")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.big-title {
    font-size:32px;
    font-weight:bold;
    text-align:center;
    color:#00FFD1;
}
.card {
    padding:20px;
    border-radius:12px;
    background:#111;
    margin-bottom:15px;
}
.result {
    font-size:24px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="big-title">🧠 AI Health Assistant</div>', unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Aim",
    "⚙️ Procedure",
    "📸 Analyzer",
    "🧠 Quiz"
])

# =========================
# AIM
# =========================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Aim")
    st.write("This app simulates an AI system to analyze health condition using image + quiz logic.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROCEDURE
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Procedure")
    st.write("""
1. Upload an image  
2. System generates features  
3. User answers quiz  
4. Final result is predicted  
""")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYZER
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    file = st.file_uploader("Upload Image", type=["jpg","png"])

    if file:
        st.image(file, use_column_width=True)

        if st.button("🔍 Analyze Image"):
            brightness = np.random.uniform(50, 200)
            contrast = np.random.uniform(0, 100)

            image_score = brightness + contrast
            st.session_state["image_score"] = image_score

            st.success("Image analyzed successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# QUIZ + RESULT
# =========================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Health Quiz")

    q1 = st.radio("Do you feel tired?", ["Yes", "No"])
    q2 = st.radio("Sleep quality?", ["Good", "Average", "Bad"])
    q3 = st.radio("Stress level?", ["Low", "Medium", "High"])

    quiz_score = 0
    if q1 == "Yes": quiz_score += 2
    if q2 == "Bad": quiz_score += 2
    if q3 == "High": quiz_score += 2

    if st.button("⚡ Get Result"):

        image_score = st.session_state.get("image_score", 100)

        total = (image_score / 100) + quiz_score

        if total > 5:
            result = "⚠️ Risk Detected"
            color = "red"
        elif total > 3:
            result = "⚡ Moderate Condition"
            color = "orange"
        else:
            result = "✅ Healthy"
            color = "green"

        st.markdown(f'<div class="result" style="color:{color}">{result}</div>', unsafe_allow_html=True)
        st.write(f"Score: {total:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)