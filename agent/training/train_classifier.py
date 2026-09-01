import json
import os

import joblib

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print(
    f"Loading dataset from:\n{DATA_PATH}"
)

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


if not data:

    raise ValueError(
        "Training dataset is empty."
    )


queries = [
    item["query"]
    for item in data
]

labels = [
    item["task"]
    for item in data
]


print(
    f"Loaded {len(queries)} queries."
)


print(
    "\nTask distribution:"
)

for label in sorted(
    set(labels)
):

    print(
        f"  {label}: "
        f"{labels.count(label)}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    queries,

    labels,

    test_size=0.25,

    random_state=42,

    stratify=labels,
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

print(
    "\nLoading SentenceTransformer..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print(
    "\nCreating training embeddings..."
)

X_train_embeddings = embedding_model.encode(

    X_train,

    show_progress_bar=True
)


print(
    "\nCreating testing embeddings..."
)

X_test_embeddings = embedding_model.encode(

    X_test,

    show_progress_bar=True
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print(
    "\nTraining Logistic Regression..."
)

classifier = LogisticRegression(

    max_iter=2000,

    random_state=42
)

classifier.fit(

    X_train_embeddings,

    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = classifier.predict(
    X_test_embeddings
)

accuracy = accuracy_score(
    y_test,
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
        y_test,
        predictions
    )
)


# ============================================================
# SAVE CLASSIFIER
# ============================================================

classifier_path = os.path.join(
    MODEL_DIR,
    "classifier.joblib"
)

joblib.dump(
    classifier,
    classifier_path
)


# ============================================================
# SAVE CONFIG
# ============================================================

config_path = os.path.join(
    MODEL_DIR,
    "config.json"
)

config = {

    "embedding_model":
        EMBEDDING_MODEL_NAME,

    "classes":
        list(
            classifier.classes_
        ),

    "classifier":
        "LogisticRegression",

    "max_iter":
        2000,

    "accuracy":
        accuracy,
}


with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


print(
    "\nTraining complete."
)

print(
    f"Classifier saved to:\n{classifier_path}"
)

print(
    f"Config saved to:\n{config_path}"
)