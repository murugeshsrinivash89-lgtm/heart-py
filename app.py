import streamlit as st
import random, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# ================= UI =================
st.markdown("""
<style>
body {
  background-color: #020617;
  background-image:
    linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
  background-size: 40px 40px;
}
.header {display:flex;justify-content:space-between;color:#00f0ff;}
.orb {
  width:200px;height:200px;margin:auto;border-radius:50%;
  background: radial-gradient(circle,#00f0ff,#001f3f);
  box-shadow:0 0 100px #00f0ff;
  animation:pulse 2s infinite;
}
@keyframes pulse {
  0%{transform:scale(1)}50%{transform:scale(1.1)}100%{transform:scale(1)}
}
.user{text-align:right;color:#00f0ff;margin:10px;}
.bot{text-align:left;color:white;margin:10px;}
</style>
""", unsafe_allow_html=True)

now = datetime.datetime.now()
st.markdown(f"<div class='header'><div>ADA</div><div>{now.strftime('%H:%M:%S')}</div></div>", unsafe_allow_html=True)
st.markdown("<div class='orb'></div>", unsafe_allow_html=True)

# ================= PATTERN GENERATOR =================
def expand(base):
    prefixes = ["i am","i feel","im","feeling","right now i feel","lately i feel","sometimes i feel"]
    suffixes = ["","so much","a lot","today","these days","again","inside","very strongly"]
    data = []
    for b in base:
        for p in prefixes:
            for s in suffixes:
                data.append(f"{p} {b} {s}".strip())
    return data

# ================= 12 EMOTIONS =================
DATA = {
"happy": expand(["happy","joyful","excited","great","amazing"]),
"sad": expand(["sad","depressed","down","hopeless","empty"]),
"stress": expand(["stressed","overwhelmed","pressure","burnt out","tense"]),
"love": expand(["in love","love someone","crush","romantic","attached"]),
"angry": expand(["angry","mad","furious","frustrated","irritated"]),
"lonely": expand(["lonely","alone","isolated","ignored","no friends"]),
"fear": expand(["scared","afraid","panic","nervous","worried"]),
"motivation": expand(["lazy","no motivation","tired","cant focus","low energy"]),
"confidence": expand(["confident","strong","capable","ready","bold"]),
"confusion": expand(["confused","lost","dont understand","unclear","doubt"]),
"jealous": expand(["jealous","insecure","comparing","envy","feeling less"]),
"guilt": expand(["guilty","regret","ashamed","mistake","bad feeling"])
}

# ================= GENERAL =================
DATA["greeting"] = ["hi","hello","hey","yo"]
DATA["name"] = ["what is your name","who are you"]
DATA["creator"] = ["who created you","who made you"]

# ================= RESPONSES =================
RESPONSES = {
"happy":["Nice 😄","Love that energy 🔥"],
"sad":["I'm here 💙","You're not alone"],
"stress":["Breathe slowly","One step at a time"],
"love":["That’s nice 🙂","Tell me more"],
"angry":["Calm down first","What happened?"],
"lonely":["I'm here with you","Talk to me"],
"fear":["You're safe","Take it slow"],
"motivation":["Start now 🔥","You got this"],
"confidence":["That’s powerful 💪","Keep going"],
"confusion":["Let’s figure it out","Explain more"],
"jealous":["Focus on yourself","You’re enough"],
"guilt":["It’s okay to learn","Forgive yourself"],

"greeting":["Hey 👋","Hello!"],
"name":["I am ADA."],
"creator":["I was created by Srinivash 🔥"]
}

# ================= NLP TRAIN =================
sentences, tags = [], []
for tag, patterns in DATA.items():
    for p in patterns:
        sentences.append(p)
        tags.append(tag)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# ================= AI =================
def predict(text):
    t = text.lower()

    # 🚨 SAFETY LAYER
    if any(x in t for x in ["dead","die","suicide","kill myself"]):
        return "Hey… I'm really concerned. You're not alone 💙 please talk to someone you trust."

    if any(x in t for x in ["depressed","hopeless"]):
        return "I'm here for you 💙 tell me more."

    # 🤖 NLP
    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, X)
    idx = sim.argmax()
    tag = tags[idx]
    confidence = sim[0][idx]

    if confidence < 0.25:
        return "Tell me more..."

    return random.choice(RESPONSES[tag])

# ================= MEMORY =================
if "chat" not in st.session_state:
    st.session_state.chat=[]

msg = st.text_input("Talk to ADA...")

if st.button("SEND"):
    if msg:
        reply = predict(msg)
        st.session_state.chat.append(("user",msg))
        st.session_state.chat.append(("bot",reply))

# ================= DISPLAY =================
for role, m in st.session_state.chat:
    if role=="user":
        st.markdown(f"<div class='user'>YOU: {m}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {m}</div>", unsafe_allow_html=True)