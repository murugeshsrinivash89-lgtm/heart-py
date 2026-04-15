import streamlit as st
import numpy as np
import random, re, math, datetime

st.set_page_config(layout="wide")

# ================= UI =================
st.markdown("""
<style>

/* BACKGROUND GRID */
body {
  background-color: #020617;
  background-image:
    linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
  background-size: 40px 40px;
}

/* HEADER */
.header {
  display:flex;
  justify-content:space-between;
  color:#00f0ff;
  font-size:20px;
}

/* ORB */
.orb {
  width:220px;
  height:220px;
  margin:20px auto;
  border-radius:50%;
  background: radial-gradient(circle,#00f0ff,#001f3f);
  box-shadow:0 0 120px #00f0ff;
  animation:pulse 2s infinite;
}

@keyframes pulse {
  0%{transform:scale(1)}
  50%{transform:scale(1.1)}
  100%{transform:scale(1)}
}

/* CHAT */
.user {
  text-align:right;
  color:#00f0ff;
  margin:10px;
}
.bot {
  text-align:left;
  color:white;
  margin:10px;
}

/* INPUT */
.stTextInput input {
  background:#020617;
  color:#00f0ff;
  border:1px solid #00f0ff;
}

/* BUTTON */
.stButton button {
  background:#00f0ff;
  color:black;
  box-shadow:0 0 10px #00f0ff;
}

/* ALERT */
.alert {
  border:1px solid red;
  color:red;
  padding:10px;
  margin:10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
now = datetime.datetime.now()
st.markdown(f"""
<div class="header">
  <div>AION</div>
  <div>{now.strftime('%H:%M:%S')} • STANDBY</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= INTENTS =================
INTENTS = {
    "hi":["Hello 👋","Hey!"],
    "name":["I am ADA."],
    "stress":["I'm here for you.","Take a breath."]
}

# ================= SIMPLE MODEL =================
def respond(text):
    t=text.lower()

    if "hi" in t:
        return random.choice(INTENTS["hi"])

    if "name" in t:
        return random.choice(INTENTS["name"])

    if "stress" in t or "sad" in t:
        return random.choice(INTENTS["stress"])

    if "time" in t:
        return datetime.datetime.now().strftime("%H:%M:%S")

    if any(x in t for x in ["+","-","*","/"]):
        try:
            expr = re.sub("[^0-9\+\-\*\/\.]","",t)
            if expr == "":
                return "Math error"
            return str(eval(expr))
        except:
            return "Math error"

    return "Not found"

# ================= MEMORY =================
if "chat" not in st.session_state:
    st.session_state.chat=[]

# ================= ALERT =================
st.markdown("<div class='alert'>⚠ SYSTEM ALERT: Online</div>", unsafe_allow_html=True)

# ================= INPUT =================
msg = st.text_input("Interface with AION...")

if st.button("SEND"):
    if msg:
        r = respond(msg)
        st.session_state.chat.append(("user",msg))
        st.session_state.chat.append(("bot",r))

# ================= CHAT DISPLAY =================
for role, m in st.session_state.chat:
    if role=="user":
        st.markdown(f"<div class='user'>YOU: {m}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>AION: {m}</div>", unsafe_allow_html=True)