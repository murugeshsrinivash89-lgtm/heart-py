import random

# ------------------------------
# TRAINING DATA
# ------------------------------
data = [
    "i feel low and tired",
    "i am happy today",
    "i feel confused about life",
    "sometimes i feel alone",
    "i am feeling better now",
    "i don't know what to do",
    "everything feels heavy",
    "i want to improve myself",
    "i feel stressed and anxious",
    "today was a good day"
]

# ------------------------------
# TOKENIZATION
# ------------------------------
def tokenize(sentence):
    return sentence.lower().split()

tokens = []
for sentence in data:
    tokens.extend(tokenize(sentence))

vocab = list(set(tokens))

# ------------------------------
# BIGRAM MODEL
# ------------------------------
bigram_counts = {}

for sentence in data:
    words = ["<start>"] + tokenize(sentence) + ["<end>"]
    for i in range(len(words)-1):
        pair = (words[i], words[i+1])
        bigram_counts[pair] = bigram_counts.get(pair, 0) + 1

bigram_prob = {}

for (w1, w2), count in bigram_counts.items():
    total = sum(c for (x, y), c in bigram_counts.items() if x == w1)
    bigram_prob.setdefault(w1, {})
    bigram_prob[w1][w2] = count / total

# ------------------------------
# SAMPLING
# ------------------------------
def sample_next(word, temperature=0.85):
    if word not in bigram_prob:
        return "<end>"

    choices = list(bigram_prob[word].keys())
    probs = list(bigram_prob[word].values())

    probs = [p ** (1/temperature) for p in probs]
    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(choices, probs)[0]

# ------------------------------
# GENERATE SENTENCE
# ------------------------------
def generate_sentence(max_len=12):
    word = "<start>"
    result = []

    for _ in range(max_len):
        word = sample_next(word)

        if word == "<end>":
            break

        result.append(word)

    return " ".join(result)

# ------------------------------
# NLP (EMOTION DETECTION)
# ------------------------------
emotion_keywords = {
    "low": ["sad", "tired", "alone", "low", "depressed"],
    "happy": ["happy", "good", "great", "better"],
    "confused": ["confused", "don't know", "lost"],
    "stress": ["stress", "anxious", "pressure"]
}

def detect_emotion(user_input):
    user_input = user_input.lower()

    for emotion, words in emotion_keywords.items():
        for w in words:
            if w in user_input:
                return emotion

    return "neutral"

# ------------------------------
# MEMORY
# ------------------------------
memory = []
MAX_MEMORY = 5

def update_memory(user_input, emotion):
    memory.append({"text": user_input, "emotion": emotion})
    if len(memory) > MAX_MEMORY:
        memory.pop(0)

def get_memory_context():
    if not memory:
        return "neutral"

    emotions = [m["emotion"] for m in memory]
    return max(set(emotions), key=emotions.count)

# ------------------------------
# HUMAN RESPONSE ENGINE
# ------------------------------
def humanize(base_text, emotion):
    starters = [
        "hmm...",
        "i see...",
        "okay...",
        "right...",
        "yeah..."
    ]

    questions = [
        "what do you think is causing it?",
        "has this been happening often?",
        "when did this start?",
        "does it stay or come and go?"
    ]

    prefix = random.choice(starters)

    if emotion == "low":
        extra = random.choice([
            "that sounds really heavy...",
            "that must feel draining..."
        ])
    elif emotion == "stress":
        extra = random.choice([
            "that seems like a lot to handle...",
            "that's quite intense..."
        ])
    elif emotion == "happy":
        extra = random.choice([
            "that’s actually nice to hear...",
            "sounds like a good moment..."
        ])
    else:
        extra = ""

    question = random.choice(questions)

    return f"{prefix} {extra} {base_text}. {question}"

# ------------------------------
# RESPONSE GENERATION
# ------------------------------
def generate_response(user_input):
    emotion = detect_emotion(user_input)
    update_memory(user_input, emotion)

    memory_emotion = get_memory_context()

    base = generate_sentence()

    response = humanize(base, memory_emotion)

    return response

# ------------------------------
# SIMPLE UI (TERMINAL STYLE)
# ------------------------------
def start_chat():
    print("\n==============================")
    print("   MINI AI COMPANION STARTED")
    print("==============================\n")

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            print("AI: take care... see you soon.")
            break

        response = generate_response(user)
        print("AI:", response)

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    start_chat()