import re
import string
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorization.pkl"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def interpret_score(score):
    if score is None:
        return "Score not available"
    if score > 2.5:
        return "Very strong Fake prediction"
    if score > 1.0:
        return "Strong Fake prediction"
    if score > 0:
        return "Moderate Fake prediction"
    if score > -1.0:
        return "Moderate Real prediction"
    if score > -2.5:
        return "Strong Real prediction"
    return "Very strong Real prediction"


def score_to_confidence(score):
    if score is None:
        return None
    score = max(min(score, 5), -5)
    return round(((score + 5) / 10) * 100, 2)


def detect_suspicious_terms(text: str):
    suspicious_words = [
        "shocking", "secret", "miracle", "exposed", "unbelievable",
        "hidden", "conspiracy", "must read", "viral", "breaking",
        "banned", "truth revealed", "what they don't want you to know"
    ]
    lower_text = text.lower()
    return [word for word in suspicious_words if word in lower_text]


def assess_input_quality(text: str):
    word_count = len(text.split())
    char_count = len(text)

    if word_count < 5:
        quality = "Very Low"
    elif word_count < 20:
        quality = "Low"
    elif word_count < 60:
        quality = "Medium"
    else:
        quality = "High"

    return {
        "word_count": word_count,
        "char_count": char_count,
        "quality": quality
    }


def predict_news(news_text: str) -> dict:
    cleaned = clean_text(news_text)
    vectorized = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(vectorized)[0]

    score = None
    if hasattr(loaded_model, "decision_function"):
        score = float(loaded_model.decision_function(vectorized)[0])
    elif hasattr(loaded_model, "predict_proba"):
        probs = loaded_model.predict_proba(vectorized)[0]
        if len(probs) > 1:
            score = float(probs[1] * 10 - 5)
        else:
            score = float(max(probs))

    label = label_map[str(int(prediction))]
    confidence = score_to_confidence(score)
    suspicious_terms = detect_suspicious_terms(news_text)
    input_stats = assess_input_quality(cleaned)

    return {
        "prediction": label,
        "raw_prediction": int(prediction),
        "score": score,
        "confidence": confidence,
        "interpretation": interpret_score(score),
        "cleaned_text": cleaned,
        "suspicious_terms": suspicious_terms,
        "word_count": input_stats["word_count"],
        "char_count": input_stats["char_count"],
        "input_quality": input_stats["quality"],
        "model_used": "Loaded ML Model"
    }