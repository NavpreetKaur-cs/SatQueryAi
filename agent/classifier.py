import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR_CANDIDATES = [
    BASE_DIR / "models" / "task_classifier",
    BASE_DIR / "models" / "ask_classifier",
]


def _resolve_model_dir() -> Path:
    for candidate in MODEL_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    default_dir = MODEL_DIR_CANDIDATES[0]
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def load_classifier() -> Dict[str, Any]:
    model_dir = _resolve_model_dir()
    classifier_path = model_dir / "classifier.joblib"
    config_path = model_dir / "config.json"

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Classifier config not found: {config_path}")

    classifier = joblib.load(str(classifier_path))
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(embedding_model_name)

    return {
        "classifier": classifier,
        "config": config,
        "embedding_model": embedding_model,
        "model_dir": model_dir,
    }

def classify_query(query: str) -> Dict[str, Any]:
    if query is None or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    model_bundle = load_classifier()
    classifier = model_bundle["classifier"]
    embedding_model = model_bundle["embedding_model"]

    embedding = embedding_model.encode([str(query).strip()])
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]
    confidence = float(max(probabilities))

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, probabilities)
    }
    return {
        "task": str(prediction),
        "confidence": confidence,
        "probabilities": probability_map,
        "method": "semantic_classifier",
    }