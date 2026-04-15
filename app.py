import streamlit as st
import numpy as np
import re
import random
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AION - ADA", layout="wide")

# =========================
# STYLING (JARVIS UI)
# =========================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at center, #020617, #000000);
    color: #00f0ff;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    color: #00f0ff;
    text-shadow: 0 0 20px #00f0ff;
}

/* Orb */
.orb {
    width: 200px;
    height: 200px;
    margin: auto;
    border-radius: 50%;
    background: radial-gradient(circle, #00f0ff, #001f3f);
    box-shadow: 0 0 60px #00f0ff;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 30px #00f0ff;}
    50% { transform: scale(1.1); box-shadow: 0 0 80px #00f0ff;}
    100% { transform: scale(1); box-shadow: 0 0 30px #00f0ff;}
}

/* Chat bubbles */
.user {
    text-align: right;
    color: #00f0ff;
    margin: 10px;
}

.bot {
    text-align: left;
    color: white;
    margin: 10px;
}

/* Input */
.stTextInput input {
    background-color: #020617;
    color: #00f0ff;
    border: 1px solid #00f0ff;
}

/* Button */
.stButton button {
    background-color: #00f0ff;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UI HEADER
# =========================
st.markdown("<div class='title'>⚡ AION — ADA SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# =========================
# SIMPLE ADA ENGINE
# =========================

def reply(text):
    text = text.lower()

    if "time" in text:
        return datetime.datetime.now().strftime("%H:%M:%S")

    if "date" in text:
        return datetime.datetime.now().strftime("%A, %d %B")

    if "joke" in text:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"

    if "name" in text:
        return "I am ADA, your custom AI assistant."

    if "hello" in text or "hi" in text:
        return "Hello Nivash 👋"

    try:
        expr = re.sub(r"[^0-9+\-*/().]","",text)
        if expr:
            return str(eval(expr))
    except:
        pass

    return "I am evolving… try again."

# =========================
# SESSION
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# INPUT
# =========================
user = st.text_input("Interface with AION...")

if st.button("Send"):
    if user:
        r = reply(user)
        st.session_state.chat.append(("user", user))
        st.session_state.chat.append(("bot", r))

# =========================
# CHAT DISPLAY
# =========================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>AION: {msg}</div>", unsafe_allow_html=True)