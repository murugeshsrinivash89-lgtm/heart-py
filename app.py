import streamlit as st
import numpy as np
import re, os, pickle, random, math, datetime

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
.header {display:flex;justify-content:space-between;color:#00f0ff;padding:10px;}
.status {border:1px solid #00f0ff;padding:6px 14px;border-radius:20px;box-shadow:0 0 10px #00f0ff;}
.orb {
  width:220px;height:220px;margin:10px auto;border-radius:50%;
  background: radial-gradient(circle,#00f0ff,#001f3f);
  box-shadow:0 0 120px #00f0ff; animation:pulse 2s infinite;
}
@keyframes pulse {0%{transform:scale(1)}50%{transform:scale(1.1)}100%{transform:scale(1)}}
.user {text-align:right;color:#00f0ff;margin:8px;font-size:18px;}
.bot  {text-align:left;color:white;margin:8px;font-size:18px;}
</style>
""", unsafe_allow_html=True)

now = datetime.datetime.now()
st.markdown(f"""
<div class="header">
  <div>ADA SYSTEM</div>
  <div class="status">{now.strftime('%H:%M:%S')} • ACTIVE</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= INTENTS =================
INTENTS = [
    {"tag":"greeting",
     "patterns":["hello","hi","hey","hii","yo","sup","hey ada"],
     "responses":["Hello Nivash 👋","Hi there!","Hey!","ADA online."]},

    {"tag":"name",
     "patterns":["what is your name","who are you"],
     "responses":["I am ADA — your Artificial Digital Assistant."]},

    {"tag":"emotion",
     "patterns":[
         "i feel stressed","i am stressed","feeling stressed",
         "i feel depressed","i am depressed","feeling low",
         "i feel sad","i am sad","i feel anxious","i feel lonely"
     ],
     "responses":[
         "I understand… I’m here with you.",
         "You're not alone. Tell me what’s going on.",
         "Take a deep breath… we’ll go step by step.",
         "It’s okay to feel this way."
     ]},

    {"tag":"time",
     "patterns":["time","what time is it"],
     "responses":["__TIME__"]},

    {"tag":"math",
     "patterns":["calculate","solve","math","add","subtract","multiply","divide"],
     "responses":["__MATH__"]}
]

# ================= TOKENIZER =================
class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.vocab_size = 0

    def norm(self, t):
        return re.sub(r"[^a-z0-9\s]", "", t.lower())

    def tok(self, t):
        return self.norm(t).split()

    def build(self, sentences):
        self.word2idx = {"[UNK]":0}
        i=1
        for s in sentences:
            for w in self.tok(s):
                if w not in self.word2idx:
                    self.word2idx[w]=i; i+=1
        self.vocab_size=len(self.word2idx)

    def encode(self,text):
        v=np.zeros(self.vocab_size)
        for w in self.tok(text):
            v[self.word2idx.get(w,0)]=1
        return v

# ================= NN =================
def relu(x): return np.maximum(0,x)
def relu_g(x): return (x>0).astype(float)
def softmax(x):
    x=x-np.max(x)
    e=np.exp(x)
    return e/np.sum(e)

class NN:
    def __init__(self,i,h,o):
        self.W1=np.random.randn(i,h)*0.1
        self.W2=np.random.randn(h,o)*0.1

    def fwd(self,x):
        self.x=x
        self.z1=x@self.W1
        self.a1=relu(self.z1)
        self.z2=self.a1@self.W2
        self.p=softmax(self.z2)
        return self.p

    def bwd(self,y):
        dz2=self.p.copy()
        dz2[y]-=1
        dW2=np.outer(self.a1,dz2)
        da1=dz2@self.W2.T
        dz1=da1*relu_g(self.z1)
        dW1=np.outer(self.x,dz1)
        return dW1,dW2

    def step(self,g,lr):
        self.W1-=lr*g[0]
        self.W2-=lr*g[1]

    def predict(self,x):
        p=self.fwd(x)
        return np.argmax(p),np.max(p)

# ================= TRAIN =================
WEIGHTS="ada.pkl"

@st.cache_resource
def load_model():
    tok=Tokenizer()
    patterns=[p for d in INTENTS for p in d["patterns"]]
    tok.build(patterns)

    X=[];Y=[]
    for i,d in enumerate(INTENTS):
        for p in d["patterns"]:
            X.append(tok.encode(p));Y.append(i)

    X=np.array(X);Y=np.array(Y)
    model=NN(tok.vocab_size,64,len(INTENTS))

    if os.path.exists(WEIGHTS):
        with open(WEIGHTS,"rb") as f:
            model.W1,model.W2=pickle.load(f)
        return tok,model

    # TRAIN 1000 epochs
    for ep in range(1000):
        for i in range(len(X)):
            p=model.fwd(X[i])
            g=model.bwd(Y[i])
            model.step(g,0.01)

    with open(WEIGHTS,"wb") as f:
        pickle.dump((model.W1,model.W2),f)

    return tok,model

tok,model=load_model()

# ================= MEMORY =================
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion=None

# ================= RESPONSE =================
def math_eval(t):
    try:
        t=t.replace("plus","+").replace("minus","-").replace("times","*").replace("divide","/")
        return str(eval(t))
    except:
        return None

def respond(text):
    x=tok.encode(text)
    idx,conf=model.predict(x)

    if conf<0.4:
        if st.session_state.last_emotion:
            return "I remember you felt "+st.session_state.last_emotion+"… want to talk more?"
        return "I’m not sure I understand."

    tag=INTENTS[idx]["tag"]

    if tag=="emotion":
        st.session_state.last_emotion="stressed"
        return random.choice(INTENTS[idx]["responses"])

    if tag=="time":
        return datetime.datetime.now().strftime("%H:%M:%S")

    if tag=="math":
        r=math_eval(text)
        return r if r else "Give proper math."

    return random.choice(INTENTS[idx]["responses"])

# ================= CHAT =================
if "chat" not in st.session_state:
    st.session_state.chat=[]

msg=st.text_input("Interface with ADA...")

if st.button("SEND"):
    if msg:
        r=respond(msg)
        st.session_state.chat.append(("user",msg))
        st.session_state.chat.append(("bot",r))

for role,msg in st.session_state.chat:
    if role=="user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>",unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {msg}</div>",unsafe_allow_html=True)