import streamlit as st
import numpy as np

# =========================
# STYLE (MASS UI)
# =========================
st.markdown("""
<style>
body {background-color:#0e1117;}
.big-title {
    font-size:36px;
    font-weight:bold;
    text-align:center;
    color:#00FFD1;
}
.card {
    padding:20px;
    border-radius:15px;
    background:#161b22;
    margin-bottom:20px;
}
.result {
    font-size:26px;
    font-weight:bold;
    text-align:center;
}
.sub {
    font-size:18px;
    color:#9aa4b2;
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

    st.markdown("### 🎯 Aim")
    st.markdown("""
<div class="sub">
This AI system analyzes user input (image + responses) to estimate basic health condition.
It simulates intelligent decision-making without heavy ML models.
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROCEDURE
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### ⚙️ Procedure")
    st.markdown("""
<div class="sub">
1. User uploads an image  
2. System extracts simulated features  
3. Score is calculated using logic  
4. Quiz answers are added  
5. Final health condition is predicted  
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYZER
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg","png"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        if st.button("🔍 Analyze Image"):
            brightness = np.random.uniform(50,200)
            contrast = np.random.uniform(0,100)

            image_score = brightness + contrast
            st.session_state['image_score'] = image_score

            st.success("Image processed successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# QUIZ + FINAL RESULT
# =========================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🧠 Health Quiz")

    q1 = st.radio("Do you feel tired?", ["Yes","No"])
    q2 = st.radio("Sleep quality?", ["Good","Average","Bad"])
    q3 = st.radio("Stress level?", ["Low","Medium","High"])

    quiz_score = 0

    if q1 == "Yes": quiz_score += 2
    if q2 == "Bad": quiz_score += 2
    if q3 == "High": quiz_score += 2

    if st.button("⚡ Get Final Result"):

        image_score = st.session_state.get('image_score', 100)

        total = image_score/100 + quiz_score

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