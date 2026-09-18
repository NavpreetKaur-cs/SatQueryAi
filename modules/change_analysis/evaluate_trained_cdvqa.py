#!/usr/bin/env python
"""Evaluate the trained CDVQA classifiers on a complete split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.change_analysis.trained_classifier import (
    compose_pair_features,
    load_classifier,
    extract_pair_features,
    normalize_label,
    normalize_question_template,
    is_binary_query,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("Val", "Test"), default="Val")
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

    payload = load_classifier(args.checkpoint)
    with open(args.cdvqa_dir / f"{args.split}_images.json", encoding="utf-8") as handle:
        images = {entry["id"]: entry for entry in json.load(handle)["images"]}
    with open(args.cdvqa_dir / f"{args.split}_questions.json", encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    with open(args.cdvqa_dir / f"{args.split}_answers.json", encoding="utf-8") as handle:
        answers = {
            entry["question_id"]: normalize_label(entry["answer"])
            for entry in json.load(handle)["answers"]
        }

    total = correct = binary_total = binary_correct = skipped = 0
    feature_cache = {}
    legacy_all_model = payload.get("all_model")
    for question in questions:
        image = images.get(question.get("img_id"))
        reference = answers.get(question.get("id"))
        if not image or reference is None:
            skipped += 1
            continue
        before = args.second_dir / "im1" / image["file_name"]
        after = args.second_dir / "im2" / image["file_name"]
        if not before.exists() or not after.exists():
            skipped += 1
            continue
        try:
            normalized_query = question["question"].lower()
            import re
            binary_query = is_binary_query(normalized_query)
            template_prediction = payload.get("template_priors", {}).get(
                normalize_question_template(question["question"])
            )
            if template_prediction is not None and not binary_query:
                prediction = str(template_prediction)
                total += 1
                correct += prediction == reference
                continue

            cache_key = str(image["file_name"])
            if cache_key not in feature_cache:
                feature_cache[cache_key] = extract_pair_features(
                    before, after, ""
                )[:84]
            features = compose_pair_features(
                feature_cache[cache_key], question["question"]
            ).reshape(1, -1)
            if binary_query:
                model = payload.get("binary_model") or legacy_all_model
            elif re.search(
                r"percent|percentage|0_to_|10_to_|20_to_|30_to_|40_to_|50_to_|60_to_|70_to_|80_to_|90_to_|100",
                normalized_query,
            ):
                model = payload.get("range_model") or legacy_all_model or payload.get("binary_model")
            elif re.search(
                r"vegetat|tree|grass|green|water|building|urban|ground surface|playground|nvg",
                normalized_query,
            ):
                model = payload.get("class_model") or legacy_all_model or payload.get("binary_model")
            else:
                model = payload.get("binary_model") or legacy_all_model or payload.get("class_model")
            if model is None:
                raise ValueError("No compatible classifier head found for the query")
            if model is not None:
                probabilities = model.predict_proba(features)[0]
                prediction = str(model.classes_[int(probabilities.argmax())])
        except (OSError, ValueError, RuntimeError):
            skipped += 1
            continue
        total += 1
        correct += prediction == reference
        if reference in ("yes", "no"):
            binary_total += 1
            binary_correct += prediction == reference

    print(f"{args.split} total accuracy: {correct / total * 100:.2f}% ({correct}/{total})")
    print(
        f"{args.split} binary accuracy: "
        f"{binary_correct / binary_total * 100:.2f}% ({binary_correct}/{binary_total})"
    )
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
