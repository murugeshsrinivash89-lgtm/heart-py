import streamlit as st
import numpy as np
import re
import random
import pickle
import os
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ADA SYSTEM", layout="wide")

# =========================
# STYLE (JARVIS UI)
# =========================
st.markdown("""
<style>
body { background: radial-gradient(circle, #000814, #000); }
.title {
    text-align:center;
    font-size:60px;
    color:#00f0ff;
    text-shadow:0 0 30px #00f0ff;
}
.orb {
    width:300px;height:300px;margin:auto;
    border-radius:50%;
    background:radial-gradient(circle,#00f0ff,#001f3f);
    box-shadow:0 0 100px #00f0ff;
    animation:pulse 2s infinite;
}
@keyframes pulse {
0%{transform:scale(1);}
50%{transform:scale(1.1);}
100%{transform:scale(1);}
}
.user {text-align:right;color:#00f0ff;font-size:18px;}
.bot {text-align:left;color:white;font-size:18px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>A D A — CORE</div>", unsafe_allow_html=True)
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# =========================
# INTENTS
# =========================
INTENTS = [
    {"tag":"greeting","patterns":["hello","hi","hey"],"responses":["Hello!","Hi there!"]},
    {"tag":"name","patterns":["name","who are you"],"responses":["I am ADA — your AI."]},
    {"tag":"time","patterns":["time"],"responses":["__TIME__"]},
    {"tag":"math","patterns":["calculate","add","multiply"],"responses":["__MATH__"]}
]

# =========================
# TOKENIZER
# =========================
class Tokenizer:
    def __init__(self):
        self.word2idx={}
        self.vocab_size=0

    def tokenize(self,text):
        return re.sub(r"[^a-z0-9 ]","",text.lower()).split()

    def build(self,data):
        self.word2idx={"[UNK]":0}
        idx=1
        for s in data:
            for w in self.tokenize(s):
                if w not in self.word2idx:
                    self.word2idx[w]=idx
                    idx+=1
        self.vocab_size=len(self.word2idx)

    def encode(self,text):
        vec=np.zeros(self.vocab_size)
        for w in self.tokenize(text):
            vec[self.word2idx.get(w,0)]=1
        return vec

# =========================
# MODEL
# =========================
class Model:
    def __init__(self,inp,out):
        self.W=np.random.randn(inp,out)*0.1

    def softmax(self,x):
        e=np.exp(x-np.max(x))
        return e/np.sum(e)

    def forward(self,x):
        return self.softmax(x@self.W)

    def predict(self,x):
        p=self.forward(x)
        return np.argmax(p), np.max(p)

# =========================
# TRAINING
# =========================
def train_model(tokenizer):
    model=Model(tokenizer.vocab_size,len(INTENTS))
    return model

# =========================
# LOAD / SAVE
# =========================
@st.cache_resource
def load_system():
    tok=Tokenizer()

    patterns=[p for i in INTENTS for p in i["patterns"]]
    tok.build(patterns)

    if os.path.exists("ada_weights.pkl"):
        model=pickle.load(open("ada_weights.pkl","rb"))
    else:
        model=train_model(tok)
        pickle.dump(model,open("ada_weights.pkl","wb"))

    return tok,model

tokenizer,model=load_system()

# =========================
# RESPONSE ENGINE
# =========================
def respond(text):
    vec=tokenizer.encode(text)
    idx,conf=model.predict(vec)

    intent=INTENTS[idx]
    res=random.choice(intent["responses"])

    if res=="__TIME__":
        return datetime.datetime.now().strftime("%H:%M:%S")

    if res=="__MATH__":
        try:
            expr=re.sub(r"[^0-9+\-*/().]","",text)
            return str(eval(expr))
        except:
            return "Math error"

    return res

# =========================
# CHAT
# =========================
if "chat" not in st.session_state:
    st.session_state.chat=[]

col1,col2=st.columns([5,1])

with col1:
    user=st.text_input("Interface with ADA...")

with col2:
    send=st.button("SEND")

if send:
    if user:
        r=respond(user)
        st.session_state.chat.append(("user",user))
        st.session_state.chat.append(("bot",r))

for role,msg in st.session_state.chat:
    if role=="user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>",unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {msg}</div>",unsafe_allow_html=True)