import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(
        "Model files not found.\n"
        f"  Expected: {MODEL_PATH}\n"
        f"  Expected: {VECTORIZER_PATH}\n"
        
    ) from e


def predict(text: str):
    
    if not text or not text.strip():
        raise ValueError("Cannot predict on empty text.")

    vector = tfidf.transform([text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = round(max(probability) * 100, 2)

    label = "FAKE" if prediction == 1 else "REAL"
    return label, confidence