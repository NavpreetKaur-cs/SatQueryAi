"""CPU-compatible CDVQA feature extraction and trained classifiers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import joblib
import numpy as np


MODEL_VERSION = "cdvqa-histgradient-v3"
LEGACY_MODEL_VERSION = "cdvqa-histgradient-v2"


def normalize_label(label: str) -> str:
    """Normalize common CDVQA answer variants to a compact canonical form."""

    normalized = str(label).strip().lower()
    aliases = {
        "trees": "tree",
        "tree": "tree",
        "low vegetation": "low_vegetation",
        "low_vegetation": "low_vegetation",
        "nvg_surface": "nvg_surface",
        "non-vegetated ground surface": "nvg_surface",
        "non vegetated ground surface": "nvg_surface",
        "buildings": "buildings",
        "building": "buildings",
        "playgrounds": "playgrounds",
        "water": "water",
        "yes": "yes",
        "no": "no",
    }
    return aliases.get(normalized, normalized)


def normalize_question_template(query: str) -> str:
    """Normalize CDVQA wording so repeated question templates share a prior."""

    normalized = re.sub(r"\d+", "<n>", str(query).strip().lower())
    return re.sub(r"\s+", " ", normalized)


def is_binary_query(query: str) -> bool:
    """Return whether a question expects a yes/no change answer."""

    normalized = str(query).strip().lower()
    return bool(
        re.search(
            r"^(is|are|was|were|has|have|did|does|do)\b.*\b(change|changed|different|increase|increased|decrease|decreased|grow|grown|shrink|shrunk|loss|gain)",
            normalized,
        )
    )


_QUERY_FLAGS = (
    "vegetation",
    "tree",
    "low_vegetation",
    "nvg_surface",
    "water",
    "building",
    "buildings",
    "built_up",
    "playground",
    "change",
    "increase",
    "decrease",
    "amount",
    "percentage",
)
QUERY_FEATURE_SIZE = len(_QUERY_FLAGS)
INTERACTION_FEATURE_SIZE = 18
PAIR_QUERY_FEATURE_SIZE = QUERY_FEATURE_SIZE + INTERACTION_FEATURE_SIZE


def _query_features(query: str) -> np.ndarray:
    normalized = query.lower()
    flags = [
        float(
            bool(
                re.search(
                    {
                        "built_up": r"built[- ]?up|building|urban|construction|house|road",
                        "building": r"building|urban|construction|house|road",
                        "buildings": r"building|urban|construction|house|road",
                        "vegetation": r"vegetat|green|grass|forest|plant",
                        "tree": r"tree",
                        "low_vegetation": r"low vegetation|grass|shrub",
                        "nvg_surface": r"non[- ]vegetated|ground surface|bare ground|nvg",
                        "water": r"water|lake|river|pond|flood",
                        "playground": r"playground",
                        "change": r"change|different|differ|changed",
                        "increase": r"increase|increased|grow|gain",
                        "decrease": r"decrease|decreased|shrink|loss",
                        "amount": r"how much|amount|what percentage|percent",
                        "percentage": r"percentage|percent|%",
                    }[flag],
                    normalized,
                )
            )
        )
        for flag in _QUERY_FLAGS
    ]
    return np.asarray(flags, dtype=np.float32)


def query_features(query: str) -> np.ndarray:
    """Return the compact question-intent feature vector."""

    return _query_features(query)


def _image_features(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image_float = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (
        hsv[:, :, index]
        for index in range(3)
    )

    class_features = np.asarray(
        [
            np.mean((hue >= 35) & (hue <= 90) & (saturation >= 35) & (value >= 25)),
            np.mean((hue >= 85) & (hue <= 130) & (saturation >= 35) & (value <= 190)),
            np.mean((saturation <= 55) & (value >= 90)),
            np.mean((hue >= 35) & (hue <= 90) & (saturation >= 70) & (value < 150)),
            np.mean((saturation < 35) & (value < 170)),
            np.mean((saturation < 55) & (value >= 170)),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            image_float.mean(axis=(0, 1)),
            image_float.std(axis=(0, 1)),
            np.percentile(image_float, [10, 50, 90], axis=(0, 1)).ravel(),
            class_features,
        ]
    ).astype(np.float32)


def extract_pair_features(
    before_path: Path,
    after_path: Path,
    query: str,
) -> np.ndarray:
    before = _image_features(before_path)
    after = _image_features(after_path)
    base_features = np.concatenate(
        [before, after, np.abs(after - before), after - before]
    ).astype(np.float32)
    return compose_pair_features(base_features, query)


def compose_pair_features(base_features: np.ndarray, query: str) -> np.ndarray:
    """Combine cached image-pair features with query intent features."""

    query_vector = _query_features(query)
    signed_delta = base_features[63:84]
    before = base_features[:21]
    after = base_features[21:42]
    query_interactions = np.concatenate(
        [
            before[-6:] * query_vector[:6].max(initial=0.0),
            after[-6:] * query_vector[:6].max(initial=0.0),
            signed_delta[-6:] * query_vector[:6].max(initial=0.0),
        ]
    )
    return np.concatenate(
        [base_features, query_vector, query_interactions]
    ).astype(np.float32)


def train_classifiers(
    features: np.ndarray,
    labels: Iterable[str],
    checkpoint_path: Path,
    queries: Iterable[str] | None = None,
) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    label_array = np.asarray([normalize_label(x) for x in labels], dtype=object)
    binary_mask = np.isin(label_array, ["yes", "no"])
    class_mask = np.isin(
        label_array,
        ["tree", "low_vegetation", "nvg_surface", "water", "buildings", "playgrounds"],
    )
    range_mask = np.array(
        [bool(re.fullmatch(r"\d+_to_\d+|\d+", str(x))) for x in label_array],
        dtype=bool,
    )
    template_priors: Dict[str, str] = {}
    if queries is not None:
        template_counts: Dict[str, Dict[str, int]] = {}
        for query, label in zip(queries, label_array):
            template = normalize_question_template(query)
            counts = template_counts.setdefault(template, {})
            counts[str(label)] = counts.get(str(label), 0) + 1
        template_priors = {
            template: max(counts, key=counts.get)
            for template, counts in template_counts.items()
        }

    def make_model() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            random_state=42,
        )

    binary_model = make_model()
    if binary_mask.any():
        binary_model.fit(features[binary_mask], label_array[binary_mask])

    class_model = make_model()
    if class_mask.any():
        class_model.fit(features[class_mask], label_array[class_mask])

    range_model = make_model()
    if range_mask.any():
        range_model.fit(features[range_mask], label_array[range_mask])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MODEL_VERSION,
        "binary_model": binary_model,
        "class_model": class_model,
        "range_model": range_model,
        "feature_size": int(features.shape[1]),
        "training_examples": int(features.shape[0]),
        "binary_examples": int(binary_mask.sum()),
        "class_examples": int(class_mask.sum()),
        "range_examples": int(range_mask.sum()),
        "template_priors": template_priors,
    }
    joblib.dump(payload, checkpoint_path, compress=3)
    return payload


def load_classifier(checkpoint_path: str | Path) -> Dict[str, Any]:
    payload = joblib.load(checkpoint_path)
    version = payload.get("version")
    if version not in (MODEL_VERSION, LEGACY_MODEL_VERSION):
        raise ValueError("Unsupported CDVQA classifier checkpoint version")
    if "all_model" in payload and "binary_model" not in payload:
        payload["binary_model"] = payload["all_model"]
    if "class_model" not in payload and "all_model" in payload:
        payload["class_model"] = payload["all_model"]
    if "range_model" not in payload and "all_model" in payload:
        payload["range_model"] = payload["all_model"]
    return payload


def predict_classifier(
    payload: Dict[str, Any],
    before_path: Path,
    after_path: Path,
    query: str,
) -> Tuple[str, float]:
    normalized = query.lower()
    template_prediction = payload.get("template_priors", {}).get(
        normalize_question_template(query)
    )
    if template_prediction is not None and not is_binary_query(normalized):
        return str(template_prediction), 0.5

    features = extract_pair_features(before_path, after_path, query).reshape(1, -1)

    if is_binary_query(normalized):
        model = payload.get("binary_model") or payload.get("all_model")
    elif re.search(r"percent|percentage|0_to_|10_to_|20_to_|30_to_|40_to_|50_to_|60_to_|70_to_|80_to_|90_to_|100", normalized):
        model = payload.get("range_model") or payload.get("all_model") or payload.get("binary_model")
    elif re.search(r"vegetat|tree|grass|green|water|building|urban|ground surface|playground|nvg", normalized):
        model = payload.get("class_model") or payload.get("all_model") or payload.get("binary_model")
    else:
        model = payload.get("binary_model") or payload.get("all_model") or payload.get("class_model")

    if model is None:
        raise ValueError("No classifier model available in checkpoint")

    probabilities = model.predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    return str(model.classes_[index]), float(probabilities[index])
