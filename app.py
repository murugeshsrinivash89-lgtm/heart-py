import streamlit as st
import numpy as np
import re
import random
import pickle
import os
import datetime

# ================= CONFIG =================
st.set_page_config(layout="wide")

# ================= UI =================
st.markdown("""
<style>
body {
    background-color:#020617;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size:40px 40px;
}
.title {
    text-align:center;
    font-size:42px;
    color:#00f0ff;
    margin-top:10px;
}
.orb {
    width:220px;height:220px;margin:20px auto;border-radius:50%;
    background: radial-gradient(circle,#00f0ff,#001f3f);
    box-shadow:0 0 80px #00f0ff;
    animation:pulse 2s infinite;
}
@keyframes pulse {
0%{transform:scale(1);}
50%{transform:scale(1.1);}
100%{transform:scale(1);}
}
.user {text-align:right;color:#00f0ff;font-size:18px;margin:8px;}
.bot {text-align:left;color:white;font-size:18px;margin:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>ADA SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= INTENTS =================
INTENTS = [
    {
        "tag": "greeting",
        "patterns": ["hello","hi","hey","hii","hiii","helo","hey bro","yo","sup"],
        "responses": ["Hello Nivash 👋","Hi there!","Hey! What can I do?"]
    },
    {
        "tag": "name",
        "patterns": ["what is your name","who are you","your name"],
        "responses": ["I am ADA — your AI assistant."]
    },
    {
        "tag": "time",
        "patterns": ["time","current time","tell time"],
        "responses": ["__TIME__"]
    },
    {
        "tag": "math",
        "patterns": ["calculate","add","subtract","multiply","divide","+"],
        "responses": ["__MATH__"]
    }
]

# ================= TOKENIZER =================
class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.vocab_size = 0

    def tokenize(self, text):
        return re.sub(r"[^a-z0-9 ]","",text.lower()).split()

    def build(self, data):
        self.word2idx = {"[UNK]":0}
        idx = 1
        for s in data:
            for w in self.tokenize(s):
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    idx += 1
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        vec = np.zeros(self.vocab_size)
        for w in self.tokenize(text):
            vec[self.word2idx.get(w,0)] = 1
        return vec

# ================= MODEL =================
class Model:
    def __init__(self, inp, out):
        self.W = np.random.randn(inp, out) * 0.1

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def predict(self, x):
        p = self.softmax(x @ self.W)
        return np.argmax(p), np.max(p)

# ================= LOAD =================
@st.cache_resource
def load():
    tok = Tokenizer()
    patterns = [p for i in INTENTS for p in i["patterns"]]
    tok.build(patterns)

    if os.path.exists("ada.pkl"):
        model = pickle.load(open("ada.pkl","rb"))
    else:
        model = Model(tok.vocab_size, len(INTENTS))
        pickle.dump(model, open("ada.pkl","wb"))

    return tok, model

tok, model = load()

# ================= RESPONSE =================
def respond(text):
    vec = tok.encode(text)
    idx, conf = model.predict(vec)

    # fallback
    if conf < 0.55:
        return "I'm not sure I understand."

    intent = INTENTS[idx]
    res = random.choice(intent["responses"])

    if res == "__TIME__":
        return datetime.datetime.now().strftime("%H:%M:%S")

    if res == "__MATH__" and any(x in text for x in ["+","-","*","/"]):
        try:
            expr = re.sub(r"[^0-9+\-*/().]", "", text)
            return str(eval(expr))
        except:
            return "Math error"

    return res

# ================= CHAT =================
if "chat" not in st.session_state:
    st.session_state.chat = []

user = st.text_input("Interface with ADA...")

if st.button("SEND"):
    if user:
        reply = respond(user)
        st.session_state.chat.append(("user", user))
        st.session_state.chat.append(("bot", reply))

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {msg}</div>", unsafe_allow_html=True)