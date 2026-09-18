from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agent.classifier import classify_query


SUPPORTED_TASKS = {
    "single_image",
    "change_analysis",
    "multimodal",
}


@dataclass
class RouteDecision:
    task: str
    confidence: float = 0.0
    method: str = "semantic_classifier"
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.task


def _normalize_task(task: Optional[str]) -> str:
    if task is None:
        raise ValueError("Classifier returned no task.")

    normalized = str(task).strip()
    aliases = {
        "single-image": "single_image",
        "single_image_analysis": "single_image",
        "singleImage": "single_image",
        "change-analysis": "change_analysis",
        "change_analysis_detection": "change_analysis",
        "change-detecion": "change_analysis",
        "multimodal_analysis": "multimodal",
        "optical_sar": "multimodal",
        "optical-sar": "multimodal",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown classifier task: {task}")

    return normalized


def _modality(image: Any) -> str:
    value = image.get("modality") if isinstance(image, dict) else getattr(image, "modality", None)
    return str(value or "").strip().lower()


def route_query(query: str, images=None, metadata=None) -> RouteDecision:
    if not query or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    images = list(images or [])
    modalities = {_modality(image) for image in images}
    has_optical = bool(modalities & {"optical", "multispectral"})
    has_sar = bool(modalities & {"sar", "radar"})
    query_text = str(query).strip().lower()

    if len(images) == 1:
        return RouteDecision(
            task="single_image",
            confidence=1.0,
            method="image_count",
            details={"image_count": 1, "modalities": sorted(modalities)},
        )

    if len(images) >= 2 and (
        (has_optical and has_sar)
        or ("optical" in query_text and "sar" in query_text)
    ):
        return RouteDecision(
            task="multimodal",
            confidence=1.0 if has_optical and has_sar else 0.9,
            method="image_modalities",
            details={"image_count": len(images), "modalities": sorted(modalities)},
        )

    if len(images) >= 2:
        return RouteDecision(
            task="change_analysis",
            confidence=1.0,
            method="image_count",
            details={"image_count": len(images), "modalities": sorted(modalities)},
        )

    prediction = classify_query(str(query).strip())
    task = _normalize_task(prediction.get("task"))

    return RouteDecision(
        task=task,
        confidence=float(prediction.get("confidence", 0.0)),
        method=prediction.get("method", "semantic_classifier"),
        details={
            "probabilities": prediction.get("probabilities", {}),
            "query_length": len(str(query).strip()),
        },
    )