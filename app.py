import streamlit as st
import numpy as np

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.big-title {
    font-size:34px;
    font-weight:bold;
    text-align:center;
    color:#FF4B4B;
}
.card {
    padding:20px;
    border-radius:12px;
    background:#111;
    margin-bottom:15px;
}
.result {
    font-size:26px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="big-title">🧬 Clinical Cancer Risk AI</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Aim & Theory","⚙️ Procedure","🧠 Assessment"])

# =========================
# AIM + THEORY
# =========================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🎯 Aim")
    st.write("To assess potential cancer risk based on clinical symptoms and user input.")

    st.subheader("🧠 Theory")
    st.write("""
This system uses symptom-based scoring inspired by clinical screening methods.
Higher symptom intensity → higher risk score.
This is NOT a diagnostic tool, only a risk indicator.
""")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROCEDURE
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚙️ Procedure")
    st.write("""
1. User answers 10 clinical questions  
2. System calculates risk score  
3. Optional image uploaded  
4. Combined analysis performed  
5. Final risk result displayed  
""")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ASSESSMENT
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧠 Clinical Questions")

    q1 = st.radio("1. Unexplained weight loss?", ["No","Yes"])
    q2 = st.radio("2. Persistent fatigue?", ["No","Yes"])
    q3 = st.radio("3. Chronic pain?", ["No","Yes"])
    q4 = st.radio("4. Lump or swelling?", ["No","Yes"])
    q5 = st.radio("5. Skin changes?", ["No","Yes"])
    q6 = st.radio("6. Persistent cough?", ["No","Yes"])
    q7 = st.radio("7. Difficulty swallowing?", ["No","Yes"])
    q8 = st.radio("8. Unusual bleeding?", ["No","Yes"])
    q9 = st.radio("9. Appetite loss?", ["No","Yes"])
    q10 = st.radio("10. Family history of cancer?", ["No","Yes"])

    # ================= IMAGE UPLOAD =================
    st.subheader("📸 Optional Image Upload")
    file = st.file_uploader("Upload Image", type=["jpg","png"])

    if file:
        st.image(file, use_column_width=True)

    # ================= SCORING =================
    score = 0

    answers = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]

    for ans in answers:
        if ans == "Yes":
            score += 1

    # image simulation effect
    if file:
        score += np.random.randint(0,3)

    # ================= RESULT =================
    if st.button("🔍 Analyze Risk"):

        if score >= 8:
            result = "⚠️ HIGH RISK"
            color = "red"
            advice = "Immediate medical consultation required"
        elif score >= 4:
            result = "⚡ MODERATE RISK"
            color = "orange"
            advice = "Regular check-up recommended"
        else:
            result = "✅ LOW RISK"
            color = "green"
            advice = "Maintain healthy lifestyle"

        st.markdown(f'<div class="result" style="color:{color}">{result}</div>', unsafe_allow_html=True)
        st.write(f"Risk Score: {score}/10")
        st.info(advice)

    st.markdown('</div>', unsafe_allow_html=True)