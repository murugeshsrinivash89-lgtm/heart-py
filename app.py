import streamlit as st
import requests
import datetime
import time

API = "http://localhost:8000/chat"

st.set_page_config(layout="wide")

# ================== CSS ==================
st.markdown("""
<style>

/* BACKGROUND GRID */
body {
    background-color:#020617;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size:40px 40px;
}

/* HEADER */
.header {
    display:flex;
    justify-content:space-between;
    color:#00f0ff;
    padding:10px;
    font-size:18px;
}

/* STATUS BADGE */
.status {
    border:1px solid #00f0ff;
    padding:5px 15px;
    border-radius:20px;
    box-shadow:0 0 10px #00f0ff;
}

/* ORB */
.orb {
    width:250px;
    height:250px;
    margin:auto;
    border-radius:50%;
    background: radial-gradient(circle,#00f0ff,#001f3f);
    box-shadow:0 0 120px #00f0ff;
    animation:pulse 2s infinite;
}

@keyframes pulse {
0%{transform:scale(1);}
50%{transform:scale(1.12);}
100%{transform:scale(1);}
}

/* CHAT */
.user {
    text-align:right;
    color:#00f0ff;
    margin:10px;
    font-size:18px;
}

.bot {
    text-align:left;
    color:white;
    margin:10px;
    font-size:18px;
}

/* INPUT */
.stTextInput input {
    background:#020617;
    color:#00f0ff;
    border:1px solid #00f0ff;
}

.stButton button {
    background:#00f0ff;
    color:black;
    font-weight:bold;
    box-shadow:0 0 10px #00f0ff;
}

/* ALERT BOX */
.alert {
    border:1px solid red;
    padding:10px;
    color:#ff4d4d;
    margin:10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
now = datetime.datetime.now()

st.markdown(f"""
<div class="header">
<div>ADA SYSTEM</div>
<div class="status">{now.strftime('%H:%M:%S')} • STANDBY</div>
</div>
""", unsafe_allow_html=True)

# ================= ORB =================
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= INPUT =================
user = st.text_input("Interface with ADA...")

if st.button("SEND"):
    if user:
        status = st.empty()
        status.markdown("<div class='status'>PROCESSING...</div>", unsafe_allow_html=True)

        try:
            res = requests.post(API, json={"message": user})
            data = res.json()

            reply = data["reply"]

            # OPEN URL
            if data.get("open_url"):
                st.markdown(f"[OPEN LINK]({data['open_url']})")

        except:
            reply = "⚠️ SYSTEM ERROR: Backend not connected"
            st.markdown("<div class='alert'>Backend Offline</div>", unsafe_allow_html=True)

        time.sleep(0.4)

        st.session_state.chat.append(("user", user))
        st.session_state.chat.append(("bot", reply))

        status.markdown("<div class='status'>STANDBY</div>", unsafe_allow_html=True)

# ================= CHAT DISPLAY =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {msg}</div>", unsafe_allow_html=True)