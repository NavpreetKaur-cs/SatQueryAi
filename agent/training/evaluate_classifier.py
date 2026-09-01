import json
import os

import joblib

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

TRAINING_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AGENT_DIR = os.path.dirname(
    TRAINING_DIR
)

DATA_PATH = os.path.join(
    TRAINING_DIR,
    "data",
    "task_queries.json"
)

MODEL_DIR = os.path.join(
    AGENT_DIR,
    "models",
    "task_classifier"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "classifier.joblib"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(
    CLASSIFIER_PATH
):

    raise FileNotFoundError(
        f"Classifier not found:\n"
        f"{CLASSIFIER_PATH}\n\n"
        "Run train_classifier.py first."
    )


if not os.path.exists(
    CONFIG_PATH
):

    raise FileNotFoundError(
        f"Config not found:\n"
        f"{CONFIG_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


queries = [
    item["query"]
    for item in data
]

labels = [
    item["task"]
    for item in data
]


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


embedding_model_name = config[
    "embedding_model"
]


# ============================================================
# LOAD MODELS
# ============================================================

print(
    "Loading classifier..."
)

classifier = joblib.load(
    CLASSIFIER_PATH
)


print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    embedding_model_name
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print(
    "Creating embeddings..."
)

embeddings = embedding_model.encode(

    queries,

    show_progress_bar=True
)


# ============================================================
# PREDICTION
# ============================================================

predictions = classifier.predict(
    embeddings
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        labels,
        predictions
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    labels,
    predictions,
    labels=classifier.classes_
)


print(
    "\nConfusion Matrix:"
)

print(
    "Classes:"
)

print(
    list(
        classifier.classes_
    )
)

print(
    matrix
)


# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================

print(
    "\nExample Predictions:"
)

for query, actual, predicted in zip(
    queries[:10],
    labels[:10],
    predictions[:10]
):

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Actual: {actual}"
    )

    print(
        f"Predicted: {predicted}"
    )