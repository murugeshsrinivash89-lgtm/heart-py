import streamlit as st
import numpy as np
import re
import random
import math
import datetime
import pickle
import os

# =========================
# INTENTS
# =========================

INTENTS = [
    {"tag": "greeting", "patterns": ["hello","hi","hey"], "responses": ["Hello!","Hi there!","Hey!"]},
    {"tag": "goodbye", "patterns": ["bye","exit"], "responses": ["Goodbye!","See you!"]},
    {"tag": "time", "patterns": ["time"], "responses": ["__TIME__"]},
    {"tag": "date", "patterns": ["date"], "responses": ["__DATE__"]},
    {"tag": "joke", "patterns": ["joke"], "responses": ["Why do programmers prefer dark mode? Because light attracts bugs!"]},
    {"tag": "math", "patterns": ["calculate","add","multiply"], "responses": ["__MATH__"]}
]

# =========================
# TOKENIZER
# =========================

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.vocab_size = 0

    def tokenise(self, text):
        return re.sub(r"[^a-z0-9 ]","",text.lower()).split()

    def build_vocab(self, sentences):
        self.word2idx = {"[UNK]":0}
        idx = 1
        for s in sentences:
            for w in self.tokenise(s):
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    idx += 1
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        vec = np.zeros(self.vocab_size)
        for w in self.tokenise(text):
            vec[self.word2idx.get(w,0)] = 1
        return vec

# =========================
# NN
# =========================

def relu(x): return np.maximum(0,x)
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

class NN:
    def __init__(self, inp, out):
        self.W = np.random.randn(inp, out) * 0.1

    def forward(self, x):
        return softmax(x @ self.W)

    def predict(self, x):
        p = self.forward(x)
        return np.argmax(p), np.max(p)

# =========================
# ENGINE
# =========================

class ADA:
    def __init__(self):
        self.tokenizer = Tokenizer()

        patterns = [p for d in INTENTS for p in d["patterns"]]
        self.tokenizer.build_vocab(patterns)

        self.tags = [d["tag"] for d in INTENTS]

        self.model = NN(self.tokenizer.vocab_size, len(self.tags))

    def chat(self, text):
        vec = self.tokenizer.encode(text)
        idx, conf = self.model.predict(vec)

        tag = self.tags[idx]
        intent = next(i for i in INTENTS if i["tag"] == tag)

        response = random.choice(intent["responses"])

        if response == "__TIME__":
            return datetime.datetime.now().strftime("%H:%M:%S")

        if response == "__DATE__":
            return datetime.datetime.now().strftime("%A %d %B")

        if response == "__MATH__":
            try:
                return str(eval(text))
            except:
                return "Math error"

        return response

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="ADA JARVIS", layout="wide")

st.markdown("""
<style>
body { background-color: #0a0f1c; color: #00ffff; }
.chat { padding:10px; border-radius:10px; margin:5px; }
.user { text-align:right; color:#00ffff; }
.bot { text-align:left; color:white; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADA — JARVIS")

@st.cache_resource
def load():
    return ADA()

ada = load()

if "chat" not in st.session_state:
    st.session_state.chat = []

user = st.text_input("Ask ADA")

if st.button("Send"):
    if user:
        reply = ada.chat(user)

        st.session_state.chat.append(("user", user))
        st.session_state.chat.append(("bot", reply))

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat user'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat bot'>{msg}</div>", unsafe_allow_html=True)
