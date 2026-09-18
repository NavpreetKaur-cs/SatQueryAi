#!/usr/bin/env python
"""Train CPU-compatible CDVQA classifiers on the complete training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from models.change_analysis.trained_classifier import (
    compose_pair_features,
    extract_pair_features,
    train_classifiers,
)


def load_split(
    split: str,
    cdvqa_dir: Path,
    second_dir: Path,
) -> Tuple[np.ndarray, List[str], List[str], int, int]:
    with open(cdvqa_dir / f"{split}_images.json", encoding="utf-8") as handle:
        image_entries = json.load(handle)["images"]
    with open(cdvqa_dir / f"{split}_questions.json", encoding="utf-8") as handle:
        question_entries = json.load(handle)["questions"]
    with open(cdvqa_dir / f"{split}_answers.json", encoding="utf-8") as handle:
        answer_entries = json.load(handle)["answers"]

    images_by_id = {entry["id"]: entry for entry in image_entries}
    answers_by_id = {entry["question_id"]: str(entry["answer"]).lower() for entry in answer_entries}
    feature_cache: Dict[str, np.ndarray] = {}
    features: List[np.ndarray] = []
    labels: List[str] = []
    queries: List[str] = []
    skipped = 0

    for index, question in enumerate(question_entries, start=1):
        image_entry = images_by_id.get(question.get("img_id"))
        answer = answers_by_id.get(question.get("id"))
        if not image_entry or answer is None:
            skipped += 1
            continue

        file_name = image_entry["file_name"]
        before_path = second_dir / "im1" / file_name
        after_path = second_dir / "im2" / file_name
        if not before_path.exists() or not after_path.exists():
            skipped += 1
            continue

        try:
            cache_key = str(file_name)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = extract_pair_features(
                    before_path,
                    after_path,
                    "",
                )[:84]
            features.append(
                compose_pair_features(feature_cache[cache_key], question["question"])
            )
            labels.append(answer)
            queries.append(question["question"])
        except (OSError, ValueError):
            skipped += 1

        if index % 1000 == 0:
            print(f"{split}: processed {index}/{len(question_entries)} questions")

    if not features:
        raise RuntimeError(f"No usable {split} records found")
    return np.asarray(features, dtype=np.float32), labels, queries, len(feature_cache), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdvqa-dir", type=Path, default=Path("data/cdvqa"))
    parser.add_argument(
        "--second-dir",
        type=Path,
        default=Path(r"C:\second dataset\SECOND_train_set"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/change_analysis/checkpoints/cdvqa_classifier.joblib"),
    )
    args = parser.parse_args()

    print("Loading the complete Train split...")
    features, labels, queries, image_count, skipped = load_split(
        "Train", args.cdvqa_dir, args.second_dir
    )
    print(
        f"Training examples: {len(labels)}; unique image pairs: {image_count}; "
        f"skipped records: {skipped}; feature size: {features.shape[1]}"
    )
    payload = train_classifiers(features, labels, args.checkpoint, queries)
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Binary examples: {payload['binary_examples']}")
    print(f"Class examples: {payload['class_examples']}")
    print(f"Range examples: {payload['range_examples']}")


if __name__ == "__main__":
    main()
