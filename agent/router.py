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


def route_query(query: str, images=None, metadata=None) -> RouteDecision:
    if not query or not str(query).strip():
        raise ValueError("Query cannot be empty.")

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