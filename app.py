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

# ================= DATA GENERATOR =================
def expand(base):
    prefixes = ["i am","i feel","im","feeling","right now i feel","lately i feel","sometimes i feel"]
    suffixes = ["","so much","a lot","today","these days","again","inside"]
    data = []
    for b in base:
        for p in prefixes:
            for s in suffixes:
                data.append(f"{p} {b} {s}".strip())
    return data

# ================= EMOTION DATA =================
DATA = {
"happy": expand(["happy","good","great","amazing","awesome","joyful","excited","positive"]),
"sad": expand(["sad","down","depressed","empty","broken","hurt","lost","hopeless"]),
"stress": expand(["stressed","overwhelmed","pressured","tired","burnt out","anxious","tense"]),
"love": expand(["in love","love her","love someone","like a girl","crush","miss her"]),
"angry": expand(["angry","mad","frustrated","irritated","furious","rage"]),
"lonely": expand(["alone","lonely","isolated","ignored","no one cares","no friends"]),
"fear": expand(["scared","afraid","nervous","worried","panic","fear"]),
"motivation": expand(["lazy","no motivation","no energy","cant focus","need motivation"])
}

# ================= GENERAL INTENTS =================
DATA["greeting"] = [
"hi","hello","hey","hii","yo","sup","good morning","good evening"
]

DATA["name"] = [
"what is your name","who are you","your name","tell me your name"
]

DATA["creator"] = [
"who created you","who made you","your creator","who built you"
]

DATA["help"] = [
"what can you do","help me","features","what do you know"
]

# ================= RESPONSES =================
RESPONSES = {
"happy":["Nice 😄 keep going!","Love that energy 🔥"],
"sad":["I’m here 💙 tell me more","You’re not alone"],
"stress":["Breathe slowly","One step at a time"],
"love":["Ohh nice 🙂 tell me more","That’s sweet 😏"],
"angry":["Calm down first","What happened?"],
"lonely":["I’m here with you","Talk to me"],
"fear":["You’ll be okay","Face it slowly"],
"motivation":["Start now 🔥","Discipline wins"],

"greeting":["Hey 👋","Hello!","Hi there!"],
"name":["I am ADA — your AI assistant."],
"creator":["I was created by Srinivash 🔥"],
"help":["I can chat, understand emotions, and support you."]
}

# ================= NLP TRAIN =================
sentences = []
tags = []

for tag, patterns in DATA.items():
    for p in patterns:
        sentences.append(p)
        tags.append(tag)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# ================= AI =================
def predict(text):
    vec = vectorizer.transform([text])
    sim = cosine_similarity(vec, X)
    idx = sim.argmax()
    tag = tags[idx]
    confidence = sim[0][idx]

    if confidence < 0.25:
        return random.choice([
            "Tell me more...",
            "I’m listening...",
            "Explain more..."
        ])

    return random.choice(RESPONSES[tag])

# ================= MEMORY =================
if "chat" not in st.session_state:
    st.session_state.chat=[]

msg = st.text_input("Talk to ADA...")

if st.button("SEND"):
    if msg:
        reply = predict(msg.lower())
        st.session_state.chat.append(("user",msg))
        st.session_state.chat.append(("bot",reply))

# ================= DISPLAY =================
for role, m in st.session_state.chat:
    if role=="user":
        st.markdown(f"<div class='user'>YOU: {m}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>ADA: {m}</div>", unsafe_allow_html=True)