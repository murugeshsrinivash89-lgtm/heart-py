"""
ADA — Artificial Digital Assistant
Built from scratch with Python + NumPy only.
No TensorFlow. No PyTorch. No pre-trained models.
"""

import streamlit as st
import numpy as np
import pickle
import os
import re
import math
import ast
import operator
import random
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  NLP PIPELINE
# ══════════════════════════════════════════════════════════════

STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
              "have", "has", "had", "do", "does", "did", "will", "would", "could",
              "should", "may", "might", "shall", "can", "need", "dare", "ought",
              "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
              "against", "between", "into", "through", "during", "before", "after",
              "above", "below", "from", "up", "down", "out", "off", "over", "under",
              "again", "further", "then", "once", "and", "but", "or", "nor", "so",
              "yet", "both", "either", "neither", "not", "only", "own", "same",
              "than", "too", "very", "just", "it", "its", "this", "that", "these",
              "those", "my", "your", "his", "her", "our", "their"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return [w for w in normalize_text(text).split() if w not in STOP_WORDS and len(w) > 1]


def build_vocabulary(training_data: list) -> list:
    vocab = set()
    for intent in training_data:
        for pattern in intent["patterns"]:
            for word in tokenize(pattern):
                vocab.add(word)
    return sorted(vocab)


def text_to_bow(text: str, vocab: list) -> np.ndarray:
    tokens = set(tokenize(text))
    vocab_index = {w: i for i, w in enumerate(vocab)}
    bow = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        if token in vocab_index:
            bow[vocab_index[token]] = 1.0
    return bow


# ══════════════════════════════════════════════════════════════
#  NEURAL NETWORK — PURE NUMPY
# ══════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Architecture: Input → Dense(128) → ReLU → Dense(64) → ReLU → Dense(N) → Softmax
    Trained via manual backpropagation.
    """

    def __init__(self, input_size: int, output_size: int,
                 hidden1: int = 128, hidden2: int = 64):
        # He initialization for ReLU networks
        self.W1 = np.random.randn(input_size, hidden1).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1), dtype=np.float32)
        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2), dtype=np.float32)
        self.W3 = np.random.randn(hidden2, output_size).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_size), dtype=np.float32)
        self._cache = {}
        # Adam optimizer state
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self.get_weights().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.get_weights().items()}

    # ── Activations ──────────────────────────────────────────

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

    # ── Forward pass ─────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache["X"]  = X
        self._cache["Z1"] = X @ self.W1 + self.b1
        self._cache["A1"] = self.relu(self._cache["Z1"])
        self._cache["Z2"] = self._cache["A1"] @ self.W2 + self.b2
        self._cache["A2"] = self.relu(self._cache["Z2"])
        self._cache["Z3"] = self._cache["A2"] @ self.W3 + self.b3
        self._cache["A3"] = self.softmax(self._cache["Z3"])
        return self._cache["A3"]

    # ── Loss ─────────────────────────────────────────────────

    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[0]
        idx = np.argmax(y_true, axis=1)
        log_likelihood = -np.log(y_pred[np.arange(m), idx] + 1e-12)
        return float(np.mean(log_likelihood))

    # ── Backward pass (Adam optimizer) ──────────────────────

    def backward(self, y_true: np.ndarray, lr: float = 0.001) -> None:
        """Adam optimizer: beta1=0.9, beta2=0.999 — fast, stable convergence."""
        m = y_true.shape[0]
        c = self._cache
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        dZ3 = (c["A3"] - y_true) / m
        dW3 = c["A2"].T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_grad(c["Z2"])
        dW2 = c["A1"].T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_grad(c["Z1"])
        dW1 = c["X"].T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads  = {"W1": dW1, "b1": db1, "W2": dW2,
                  "b2": db2, "W3": dW3, "b3": db3}
        params = {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                  "b2": self.b2, "W3": self.W3, "b3": self.b3}

        for key, grad in grads.items():
            np.clip(grad, -5.0, 5.0, out=grad)
            self._m[key] = b1 * self._m[key] + (1 - b1) * grad
            self._v[key] = b2 * self._v[key] + (1 - b2) * (grad ** 2)
            m_hat = self._m[key] / (1 - b1 ** self._t)
            v_hat = self._v[key] / (1 - b2 ** self._t)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Predict ──────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    # ── Weight I/O ───────────────────────────────────────────

    def get_weights(self) -> dict:
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3}

    def set_weights(self, w: dict) -> None:
        self.W1 = w["W1"].astype(np.float32)
        self.b1 = w["b1"].astype(np.float32)
        self.W2 = w["W2"].astype(np.float32)
        self.b2 = w["b2"].astype(np.float32)
        self.W3 = w["W3"].astype(np.float32)
        self.b3 = w["b3"].astype(np.float32)
        # Reset Adam state after loading weights
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self.get_weights().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.get_weights().items()}


# ══════════════════════════════════════════════════════════════
#  TRAINING INTENTS
# ══════════════════════════════════════════════════════════════

INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
            "good day", "what is up", "whats up", "howdy", "greetings", "hi there",
            "hello ada", "hey ada", "sup", "yo", "hiya", "salutations", "hola",
            "how are you", "how do you do", "nice to meet you", "pleased to meet you",
        ],
        "responses": [
            "Hello! I'm ADA, your Artificial Digital Assistant. How can I help you today? 👋",
            "Hey there! ADA is online and ready. What can I do for you?",
            "Hi! Great to have you here. What's on your mind?",
            "Hello! I'm fully operational and ready to assist. What do you need?",
            "Hey! Good to see you. Ask me anything — I'm here for it. 🤖",
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "see ya", "later", "take care",
            "farewell", "good night", "goodnight", "cya", "i am leaving", "im leaving",
            "have to go", "gotta go", "ttyl", "talk later", "bye bye", "adios",
            "signing off", "until next time", "catch you later", "peace out",
        ],
        "responses": [
            "Goodbye! Take care of yourself. I'll be right here when you return. 👋",
            "Bye! It was great chatting. Come back anytime — I never sleep! 😄",
            "See you soon! Stay safe and well. 🌟",
            "Farewell! Remember, ADA is always here whenever you need me. 💙",
        ]
    },
    {
        "tag": "name",
        "patterns": [
            "what is your name", "what is your name ada", "who are you", "what are you",
            "tell me your name", "your name please", "what should i call you",
            "introduce yourself", "who is ada", "what is ada", "are you ada",
            "what do you call yourself", "do you have a name", "what can i call you",
            "tell me about yourself", "describe yourself", "what kind of ai are you",
        ],
        "responses": [
            "I'm **ADA** — Artificial Digital Assistant! I'm a neural network built entirely from scratch using Python and NumPy, with no external AI frameworks. Nice to meet you! 🤖",
            "My name is ADA (Artificial Digital Assistant). I was trained using a custom backpropagation neural network. What can I help you with?",
            "I'm ADA — your personal AI companion. Custom-built, open-source, and here to serve. What would you like to explore?",
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "thanks", "thank you", "thank you so much", "thanks a lot",
            "appreciate it", "cheers", "many thanks", "much appreciated",
            "thx", "ty", "thank u", "that was helpful", "you are helpful",
            "you are great", "good job", "well done", "great job", "awesome",
            "you are amazing", "brilliant", "perfect", "excellent work",
        ],
        "responses": [
            "You're welcome! Always happy to help. 😊",
            "Of course! That's exactly what I'm here for.",
            "Glad I could help! Feel free to ask anything else.",
            "Anytime! I'm here whenever you need me. 🌟",
        ]
    },
    {
        "tag": "joke",
        "patterns": [
            "tell me a joke", "say a joke", "make me laugh", "joke please",
            "something funny", "humor me", "tell me something funny",
            "got any jokes", "make me smile", "cheer me up with a joke",
            "give me a joke", "i want a joke", "do you know any jokes",
        ],
        "responses": [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads! 😂",
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why did the math book look so sad? It had too many problems. 📚",
            "I asked my neural network for a joke. It returned: ERROR 404 — Humor not found! 🤖",
            "Why did the developer go broke? Because they used up all their cache! 💸",
            "What do you call 8 hobbits? A hobbyte! 🧙",
            "Why is Python the best programming language? Because it has no class... wait, it does! 🐍",
        ]
    },
    {
        "tag": "math",
        "patterns": [
            "calculate", "compute", "what is", "solve", "evaluate", "figure out",
            "plus", "minus", "times", "divided by", "multiply", "add", "subtract",
            "square root", "power", "percentage", "math", "arithmetic",
            "how much is", "whats the answer", "what does equal",
        ],
        "responses": ["__MATH__"]
    },
    {
        "tag": "time",
        "patterns": [
            "what time is it", "current time", "what is the time", "time now",
            "tell me the time", "what time", "check the clock", "time please",
            "clock", "what hour is it", "give me the time",
        ],
        "responses": ["__TIME__"]
    },
    {
        "tag": "date",
        "patterns": [
            "what is the date", "what is today", "today date", "current date",
            "what day is it today", "what month is it", "what year is it",
            "day today", "date today", "tell me the date", "todays date",
            "what is the day today", "which day is today",
        ],
        "responses": ["__DATE__"]
    },
    {
        "tag": "emotion_sad",
        "patterns": [
            "i am sad", "i feel sad", "feeling sad", "i feel down",
            "i am feeling down", "i feel low", "i feel depressed",
            "i am depressed", "feeling depressed", "i feel hopeless",
            "life is hard", "i am unhappy", "i feel unhappy",
            "i feel empty", "feeling empty", "i feel lost", "i am lost",
            "i feel broken", "i am broken", "nothing feels good",
            "everything feels wrong", "i feel terrible", "i am terrible",
        ],
        "responses": [
            "I'm really sorry you're feeling this way. Your feelings are completely valid, and it's okay to feel sad sometimes. Would you like to talk about what's going on? 💙",
            "I hear you, and I'm here for you. Sadness can feel very heavy. Remember that difficult times do pass. Is there something specific on your mind?",
            "It takes real courage to share how you feel. I genuinely care, and I'm listening. What's been happening for you lately?",
            "You're not alone in this. I'm right here with you. Would you like some uplifting ideas, or would you rather just have someone listen? 💙",
        ]
    },
    {
        "tag": "emotion_stressed",
        "patterns": [
            "i feel stressed", "i am stressed", "feeling stressed",
            "i feel anxious", "i am anxious", "i feel overwhelmed",
            "i am overwhelmed", "too much pressure", "i feel pressure",
            "i cannot handle it", "i can not handle it", "too stressed",
            "work stress", "so much stress", "under a lot of pressure",
            "i feel panicked", "i am panicking", "having a panic attack",
            "everything is too much", "i feel burnt out", "i am burnt out",
        ],
        "responses": [
            "Take a slow, deep breath right now — in for 4, hold for 4, out for 4. You've handled hard things before, and you can handle this too. Want to talk about what's overwhelming you? 🧘",
            "I understand — stress can feel absolutely suffocating. Remember: you don't have to solve everything at once. One small step at a time. What's the biggest thing weighing on you?",
            "It sounds like you have a lot on your plate. That's genuinely hard. You're doing your best, and that truly is enough. Can I suggest some quick calming techniques?",
            "Hey, I hear you. Being overwhelmed is real and valid. Let's slow down together. 💙 What's stressing you most right now?",
        ]
    },
    {
        "tag": "emotion_lonely",
        "patterns": [
            "i feel lonely", "i am lonely", "feeling lonely",
            "i have no friends", "no one cares about me", "nobody cares",
            "i feel alone", "i am alone", "feeling alone", "i have nobody",
            "no one talks to me", "i feel isolated", "i am isolated",
            "nobody listens to me", "i feel disconnected", "no one understands me",
            "i have no one", "everyone left me", "people ignore me",
        ],
        "responses": [
            "I'm right here with you, and I genuinely care. Loneliness is one of the hardest feelings to carry. You matter more than you know. Want to chat for a while? 💙",
            "You're not alone — I'm here! I know I'm an AI, but I really am listening. Tell me more about how you're feeling.",
            "Reaching out takes bravery, and I'm so glad you did. You deserve to be heard. What's been going on?",
            "I hear you. That feeling of disconnection is real and painful. Let's talk — what's been on your mind lately? 💙",
        ]
    },
    {
        "tag": "emotion_happy",
        "patterns": [
            "i feel happy", "i am happy", "feeling happy",
            "i feel great", "feeling great", "i feel good",
            "i am feeling good", "i feel wonderful", "i feel amazing",
            "great day today", "best day ever", "i feel excited",
            "i am excited", "feeling excited", "i am thrilled",
            "life is good", "everything is great", "i am on top of the world",
        ],
        "responses": [
            "That's absolutely wonderful to hear! 😊 Your happiness is contagious. What's making today so great?",
            "I love hearing this! 🌟 You deserve every bit of joy. What's the highlight of your day?",
            "That's fantastic! Keep riding that wave of positivity. 🎉 What's bringing you so much joy right now?",
            "Yay! Happy people make the world brighter. 🌈 Tell me what's going on — I want to celebrate with you!",
        ]
    },
    {
        "tag": "emotion_angry",
        "patterns": [
            "i feel angry", "i am angry", "feeling angry",
            "i feel frustrated", "i am frustrated", "feeling frustrated",
            "i am mad", "i feel mad", "i am furious", "i feel furious",
            "i feel irritated", "i am irritated", "so angry right now",
            "i feel rage", "i am raging", "i want to scream",
            "this makes me angry", "i hate everything", "i am so fed up",
        ],
        "responses": [
            "It's okay to feel angry — that's a completely natural emotion. Taking a few deep breaths can help cool things down. Want to talk about what happened? 🔥",
            "I hear you. Anger often means something important to you was crossed. Let's talk it through. What's going on?",
            "That frustration sounds real and valid. You're safe to express it here, without judgment. What's been getting to you?",
            "Sometimes we all feel that way. I'm here to listen — no judgment at all. What made you feel this way? 💙",
        ]
    },
    {
        "tag": "capabilities",
        "patterns": [
            "what can you do", "what are your capabilities", "help me",
            "how can you help", "what do you know", "show me what you can do",
            "what features do you have", "what are your features",
            "what are you capable of", "tell me what you can do",
        ],
        "responses": [
            "Here's what I can help with:\n\n🔢 **Math** — calculations, expressions, formulas\n🕐 **Time & Date** — current time and date\n😄 **Jokes** — a good laugh whenever you need one\n💙 **Emotional Support** — I'm always here to listen\n💬 **Conversation** — friendly chat about anything!\n\nJust ask me anything!",
        ]
    },
    {
        "tag": "fallback",
        "patterns": ["undefined placeholder unknown xyzzy"],
        "responses": [
            "Hmm, I'm not quite sure about that one. Try rephrasing? I'm still learning! 🤔",
            "That one's a bit tricky for me. Could you say it differently?",
            "I didn't fully catch that. Try asking me about math, time, date, or just have a chat!",
            "I'm not certain about that yet — but I'm always learning! Try something else? 🤖",
        ]
    },
]


# ══════════════════════════════════════════════════════════════
#  SAFE MATH PARSER
# ══════════════════════════════════════════════════════════════

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "sqrt": m