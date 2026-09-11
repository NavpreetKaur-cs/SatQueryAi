"""Small CDVQA-compatible data loading and exact-answer evaluation helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .interface import change_analysis_tool


def load_cdvqa(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSON, JSONL, or CSV records with image-pair/question/answer fields.

    Field aliases commonly used by CDVQA exports are normalized to
    ``before_image``, ``after_image``, ``question`` and ``answer``.
    """
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        text = source.read_text(encoding="utf-8")
        data = json.loads(text) if source.suffix.lower() == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
        rows = data.get("data", data) if isinstance(data, dict) else data
    aliases = {"before_image": ("before_image", "image1", "img1", "t1"), "after_image": ("after_image", "image2", "img2", "t2"), "question": ("question", "query"), "answer": ("answer", "label", "gt_answer")}
    normalized = []
    for row in rows:
        record = {target: next((row.get(key) for key in keys if row.get(key) is not None), None) for target, keys in aliases.items()}
        if not all(record.values()):
            raise ValueError(f"CDVQA record misses a required field: {row}")
        normalized.append(record)
    return normalized


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def evaluate_cdvqa(records: Iterable[Dict[str, Any]], image_root: str | Path = ".") -> Dict[str, Any]:
    records = list(records)
    correct = 0
    predictions = []
    root = Path(image_root)
    for row in records:
        result = change_analysis_tool(root / row["before_image"], root / row["after_image"], row["question"], {"save_visuals": False})
        prediction = result["answer"]
        match = _canonical(row["answer"]) in _canonical(prediction)
        correct += match
        predictions.append({"prediction": prediction, "reference": row["answer"], "correct": match, "success": result["success"]})
    return {"count": len(records), "exact_substring_accuracy": correct / len(records) if records else 0.0, "predictions": predictions}
