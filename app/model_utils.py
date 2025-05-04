import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")

def save_models(vectorizer, knn_model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(knn_model, MODEL_PATH)

def load_models():
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        knn_model = joblib.load(MODEL_PATH)
        return vectorizer, knn_model
    return None, None

