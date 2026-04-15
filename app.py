import streamlit as st
import numpy as np
import re, os, pickle, random, math, datetime, ast, operator

st.set_page_config(layout="wide")

# ================= UI =================
st.markdown("""
<style>
body {background:#020617;}
.user {text-align:right;color:#00f0ff;font-size:18px;}
.bot {text-align:left;color:white;font-size:18px;}
</style>
""", unsafe_allow_html=True)

st.title("ADA SYSTEM")

# ================= INTENTS =================
INTENTS = [
    {"tag":"greeting",
     "patterns":["hi","hello","hey","hii"],
     "responses":["Hello 👋","Hi there!","Hey!"]},

    {"tag":"emotion",
     "patterns":["i feel stressed","i am stressed","i feel sad","i am depressed"],
     "responses":["I'm here with you.","You’re not alone.","Take a deep breath."]},

    {"tag":"name",
     "patterns":["what is your name","who are you"],
     "responses":["I am ADA — your assistant."]},

    {"tag":"math",
     "patterns":["calculate","solve","add","subtract","multiply","divide"],
     "responses":["__MATH__"]},

    {"tag":"time",
     "patterns":["time","what time is it"],
     "responses":["__TIME__"]}
]

# ================= TOKENIZER =================
class Tok:
    def __init__(self):
        self.w2i={}

    def build(self,data):
        self.w2i={"[UNK]":0}
        i=1
        for s in data:
            for w in s.split():
                if w not in self.w2i:
                    self.w2i[w]=i;i+=1

    def enc(self,t):
        v=np.zeros(len(self.w2i))
        for w in t.split():
            v[self.w2i.get(w,0)]=1
        return v

# ================= NN =================
def relu(x): return np.maximum(0,x)
def relu_g(x): return (x>0).astype(float)
def softmax(x):
    e=np.exp(x-np.max(x))
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

# ================= SAFE MATH =================
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow
}

def eval_expr(expr):
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        else:
            raise ValueError
    return _eval(ast.parse(expr, mode='eval').body)

# ================= TRAIN =================
WEIGHTS="ada.pkl"

@st.cache_resource
def load_model():
    tok=Tok()
    patterns=[p for d in INTENTS for p in d["patterns"]]
    tok.build(patterns)

    X=[];Y=[]
    for i,d in enumerate(INTENTS):
        for p in d["patterns"]:
            X.append(tok.enc(p));Y.append(i)

    X=np.array(X);Y=np.array(Y)
    model=NN(len(tok.w2i),64,len(INTENTS))

    if os.path.exists(WEIGHTS):
        try:
            with open(WEIGHTS,"rb") as f:
                model.W1,model.W2=pickle.load(f)
            return tok,model
        except:
            os.remove(WEIGHTS)

    # TRAIN 1000 epochs
    for ep in range(1000):
        for i in range(len(X)):
            model.fwd(X[i])
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
def respond(text):
    x=tok.enc(text.lower())
    idx,conf=model.predict(x)

    if conf<0.4:
        if st.session_state.last_emotion:
            return "You mentioned you're feeling "+st.session_state.last_emotion+" earlier. I'm here."
        return "I’m not sure I understand."

    tag=INTENTS[idx]["tag"]

    if tag=="emotion":
        st.session_state.last_emotion="stressed"
        return random.choice(INTENTS[idx]["responses"])

    if tag=="time":
        return datetime.datetime.now().strftime("%H:%M:%S")

    if tag=="math":
        try:
            expr=re.sub("[^0-9\+\-\*\/\.]","",text)
            return str(eval_expr(expr))
        except:
            return "Invalid math."

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