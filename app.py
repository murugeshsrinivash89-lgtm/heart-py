import numpy as np
import pandas as pd
import cv2
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# =========================
# SIGNAL PROCESSING
# =========================
def extract_ppg_features(ppg_signal):
    features = {}

    features['mean'] = np.mean(ppg_signal)
    features['std'] = np.std(ppg_signal)
    features['max'] = np.max(ppg_signal)
    features['min'] = np.min(ppg_signal)

    # Simple peak detection
    peaks = np.where(ppg_signal > np.mean(ppg_signal))[0]
    features['peak_count'] = len(peaks)

    return features


def extract_temp_features(temp_signal):
    return {
        'temp_mean': np.mean(temp_signal),
        'temp_std': np.std(temp_signal)
    }


# =========================
# IMAGE PROCESSING
# =========================
def extract_image_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image not found. Check path!")

    img = cv2.resize(img, (100, 100))

    mean_colors = img.mean(axis=(0, 1))

    return {
        'blue_mean': mean_colors[0],
        'green_mean': mean_colors[1],
        'red_mean': mean_colors[2]
    }


# =========================
# MODEL TRAINING
# =========================
def train_model():
    print("⚙️ Training model...")

    np.random.seed(42)
    data = []

    for _ in range(300):
        sample = {
            'mean': np.random.rand(),
            'std': np.random.rand(),
            'max': np.random.rand(),
            'min': np.random.rand(),
            'peak_count': np.random.randint(50, 150),
            'temp_mean': np.random.uniform(30, 37),
            'temp_std': np.random.rand(),
            'red_mean': np.random.rand() * 255,
            'green_mean': np.random.rand() * 255,
            'blue_mean': np.random.rand() * 255,
            'label': np.random.randint(0, 2)
        }
        data.append(sample)

    df = pd.DataFrame(data)

    X = df.drop('label', axis=1)
    y = df['label']

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model trained & saved!")


# =========================
# PREDICTION
# =========================
def predict(image_path):
    if not os.path.exists("models/model.pkl"):
        train_model()

    model = joblib.load("models/model.pkl")

    # Simulated signals (replace with real sensors later)
    ppg_signal = np.random.rand(100)
    temp_signal = np.random.uniform(30, 37, 50)

    ppg_features = extract_ppg_features(ppg_signal)
    temp_features = extract_temp_features(temp_signal)
    image_features = extract_image_features(image_path)

    # Combine all features
    features = {**ppg_features, **temp_features, **image_features}

    X = np.array(list(features.values())).reshape(1, -1)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    print("\n===== RESULT =====")
    print(f"Risk Probability: {probability:.2f}")

    if prediction == 1:
        print("⚠️ Risk of circulatory issue detected")
    else:
        print("✅ Normal circulation")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🔍 Fingertip Circulation Predictor")

    image_path = input("Enter fingertip image path: ")

    try:
        predict(image_path)
    except Exception as e:
        print(e)