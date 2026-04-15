import streamlit as st
import numpy as np
import re, os, pickle, random, math, datetime

# ================= UI =================
st.set_page_config(layout="wide")

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
  width:240px;height:240px;margin:10px auto;border-radius:50%;
  background: radial-gradient(circle,#00f0ff,#001f3f);
  box-shadow:0 0 120px #00f0ff; animation:pulse 2s infinite;
}
@keyframes pulse {0%{transform:scale(1)}50%{transform:scale(1.12)}100%{transform:scale(1)}}
.user {text-align:right;color:#00f0ff;margin:8px;font-size:18px;}
.bot  {text-align:left;color:white;margin:8px;font-size:18px;}
.stTextInput input {background:#020617;color:#00f0ff;border:1px solid #00f0ff;}
.stButton button {background:#00f0ff;color:black;font-weight:bold;box-shadow:0 0 10px #00f0ff;}
.alert {border:1px solid #ff4d4d;color:#ff4d4d;padding:10px;margin:10px;}
</style>
""", unsafe_allow_html=True)

now = datetime.datetime.now()
st.markdown(f"""
<div class="header">
  <div>ADA SYSTEM</div>
  <div class="status">{now.strftime('%H:%M:%S')} • STANDBY</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= INTENTS (EXPANDED) =================
INTENTS = [
    {"tag":"greeting",
     "patterns":[
         "hello","hi","hey","hii","hiii","helo","hey bro","hi ada","hello ada","yo","sup","hiya",
         "good morning","good afternoon","good evening","whats up","what is up","hey there"
     ],
     "responses":[
         "Hello Nivash 👋","Hi there!","Hey! What can I do?","Greetings. ADA online."
     ]},
    {"tag":"goodbye",
     "patterns":["bye","goodbye","see you","later","exit","quit","bye ada","good night"],
     "responses":["Goodbye.","See you later.","Signing off.","Take care."]},
    {"tag":"name",
     "patterns":["what is your name","who are you","your name","introduce yourself"],
     "responses":["I am ADA — your Artificial Digital Assistant."]},
    {"tag":"time",
     "patterns":["time","current time","tell me the time","what time is it","time now"],
     "responses":["__TIME__"]},
    {"tag":"date",
     "patterns":["date","today date","what day is it","current date"],
     "responses":["__DATE__"]},
    {"tag":"math",
     "patterns":[
         "calculate","solve","compute","add","subtract","multiply","divide","math",
         "what is 2 plus 2","what is 5 times 3","10 minus 4","square root of 144"
     ],
     "responses":["__MATH__"]},
    {"tag":"joke",
     "patterns":["tell me a joke","joke","make me laugh","funny"],
     "responses":[
         "Why do programmers prefer dark mode? Because light attracts bugs.",
         "A SQL query walks into a bar and asks: Can I join you?"
     ]},
    {"tag":"thanks",
     "patterns":["thanks","thank you","ty","thanks a lot"],
     "responses":["You're welcome.","Anytime.","Glad to help."]},
]

# ================= TOKENIZER =================
class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.vocab_size = 0

    def norm(self, t):
        return re.sub(r"[^a-z0-9\s]", "", t.lower().strip())

    def tok(self, t):
        return self.norm(t).split()

    def build(self, sentences):
        self.word2idx = {"[UNK]":0}
        i = 1
        for s in sentences:
            for w in self.tok(s):
                if w not in self.word2idx:
                    self.word2idx[w] = i; i += 1
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        v = np.zeros(self.vocab_size, dtype=np.float32)
        for w in self.tok(text):
            v[self.word2idx.get(w,0)] = 1.0
        return v

# ================= NN (2-layer MLP) =================
def relu(x): return np.maximum(0.0, x)
def relu_g(x): return (x>0).astype(np.float32)
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

class MLP:
    def __init__(self, inp, h, out):
        s1 = math.sqrt(2.0/inp)
        s2 = math.sqrt(2.0/h)
        self.W1 = (np.random.randn(inp, h)*s1).astype(np.float32)
        self.b1 = np.zeros((h,), np.float32)
        self.W2 = (np.random.randn(h, out)*s2).astype(np.float32)
        self.b2 = np.zeros((out,), np.float32)

    def fwd(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        p  = softmax(z2)
        self.cache = (x,z1,a1,p)
        return p

    def bwd(self, y_idx):
        x,z1,a1,p = self.cache
        dz2 = p.copy(); dz2[y_idx] -= 1.0
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_g(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        return dW1,db1,dW2,db2

    def step(self, grads, lr):
        dW1,db1,dW2,db2 = grads
        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2

    def predict(self, x):
        p = self.fwd(x)
        return int(np.argmax(p)), float(np.max(p))

# ================= DATA + TRAIN =================
WEIGHTS = "ada_streamlit.pkl"

@st.cache_resource
def load_and_train():
    tok = Tokenizer()
    patterns = [p for d in INTENTS for p in d["patterns"]]
    tok.build(patterns)

    tag2idx = {d["tag"]:i for i,d in enumerate(INTENTS)}
    X, Y = [], []
    for d in INTENTS:
        for p in d["patterns"]:
            X.append(tok.encode(p))
            Y.append(tag2idx[d["tag"]])

    X = np.array(X); Y = np.array(Y)
    model = MLP(tok.vocab_size, 128, len(INTENTS))

    # load cached weights if exists
    if os.path.exists(WEIGHTS):
        with open(WEIGHTS,"rb") as f:
            state = pickle.load(f)
        model.W1, model.b1, model.W2, model.b2 = state
        return tok, model

    # train (fast but real)
    epochs = 1200
    lr0 = 0.01
    idxs = list(range(len(X)))
    for ep in range(1, epochs+1):
        random.shuffle(idxs)
        lr = lr0 * 0.5*(1+math.cos(math.pi*ep/epochs))  # cosine decay
        loss = 0.0
        for i in idxs:
            p = model.fwd(X[i])
            loss += -math.log(max(p[Y[i]], 1e-12))
            grads = model.bwd(Y[i])
            model.step(grads, lr)
        if loss/len(X) < 0.02:
            break

    # save
    with open(WEIGHTS,"wb") as f:
        pickle.dump((model.W1, model.b1, model.W2, model.b2), f)

    return tok, model

tok, model = load_and_train()

# ================= UTIL =================
def safe_math(text):
    t = text.lower()
    # words → symbols
    t = re.sub(r"\bplus\b"," + ",t)
    t = re.sub(r"\bminus\b"," - ",t)
    t = re.sub(r"\btimes|multiply|multiplied\b"," * ",t)
    t = re.sub(r"\bdivide|divided|over\b"," / ",t)
    t = re.sub(r"square\s*root\s*of\s*(\d+\.?\d*)", r"(\1**0.5)", t)

    m = re.search(r"[0-9\.\+\-\*/\(\)\s]+", t)
    if not m: return None
    expr = re.sub(r"\s+","", m.group())
    if not re.match(r"^[0-9\.\+\-\*/\(\)]+$", expr): return None
    try:
        r = eval(expr)
        if isinstance(r,float) and r.is_integer(): r=int(r)
        return f"The answer is {r}."
    except:
        return None

# ================= RESPONSE =================
CONF = 0.5

def respond(text):
    x = tok.encode(text)
    idx, conf = model.predict(x)

    if conf < CONF:
        return "I'm not sure I understand. Try rephrasing."

    intent = INTENTS[idx]
    res = random.choice(intent["responses"])

    if res == "__TIME__":
        return datetime.datetime.now().strftime("%I:%M %p")
    if res == "__DATE__":
        return datetime.datetime.now().strftime("%A, %B %d, %Y")
    if res == "__MATH__":
        # only run if math-like
        if any(s in text.lower() for s in ["+","-","*","/","plus","minus","times","divide","calculate","solve"]):
            out = safe_math(text)
            if out: return out
        return "Give a math expression like '5 + 3' or 'square root of 144'."

    return res

# ================= CHAT =================
if "chat" not in st.session_state:
    st.session_state.chat = []

user = st.text_input("Interface with ADA...")

if st.button("SEND"):
    if user.strip():
        reply = respond(user)
        st.session_state.chat.append(("user", user))
        st.session_state.chat.append(("bot", reply))

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>YOU: {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {msg}</div>", unsafe_allow_html=True)