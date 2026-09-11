"""Stable agent-facing interface for bi-temporal change analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import ChangeAnalysisError, analyze_change


def _image_path(image: Any) -> Optional[str]:
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)

    if isinstance(image, dict):
        return image.get("path")

    return getattr(image, "path", None)


def _label(image: Any) -> str:
    return str(_image_path(image) or "")


def change_analysis_tool(
    image1: Any,
    image2: Any,
    query: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Answer a change question for before and after images.

    Metadata may contain:
        output_dir
        threshold
        min_region_pixels
        save_visuals
        label1_path
        label2_path

    When label1_path and label2_path are provided, the tool also generates
    a semantic change map and semantic change legend.
    """

    before_path = _image_path(image1)
    after_path = _image_path(image2)

    base_metadata: Dict[str, Any] = {
        "module": "change_analysis",
        "image1": _label(image1),
        "image2": _label(image2),
    }

    if metadata:
        base_metadata["input_metadata"] = metadata

    if not before_path or not after_path:
        return _failure(
            "Both temporal images must provide a path.",
            base_metadata,
        )

    if not query or not query.strip():
        return _failure(
            "Query is empty.",
            base_metadata,
        )

    try:
        result = analyze_change(
            before_path,
            after_path,
            query,
            metadata or {},
        )

    except (ChangeAnalysisError, OSError, ValueError) as exc:
        return _failure(
            str(exc),
            base_metadata,
        )

    result["metadata"] = {
        **base_metadata,
        **result.get("metadata", {}),
    }

    # Expose semantic legend if semantic map was generated.
    visuals = result["metadata"].get("visuals", {})

    semantic_map = visuals.get("semantic_change_map")

    if semantic_map:
        legend_path = (
            Path(semantic_map).parent
            / "semantic_change_legend.png"
        )

        if legend_path.exists():
            visuals["semantic_change_legend"] = str(
                legend_path
            )

    result["metadata"]["visuals"] = visuals

    return result


def _failure(
    message: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "success": False,
        "answer": "",
        "confidence": None,
        "model": "opencv-change-baseline-v1",
        "visual_output": None,
        "metadata": {
            **metadata,
            "error": message,
        },
    }