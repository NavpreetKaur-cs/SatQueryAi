"""
modules/multimodal/interface.py

Entry point that the agent (agent/tools.py -> multimodal_tool) calls for
optical + SAR joint analysis.

This file is the CONTRACT boundary between the agent and this module.
Its function signature and return shape must exactly match the mock
`multimodal_tool` currently registered in agent/tools.py:

    def multimodal_tool(optical_image, sar_image, query, metadata=None) -> dict

Everything below this file (preprocessing, feature_extraction, fusion,
task_head) can change freely as long as this function's signature and
output schema stay stable, so the agent integration never breaks while
the real pipeline is being built out.
"""

import os
from typing import Any, Dict, Optional

def _image_label(image: Any) -> str:

    if image is None:
        return ""
    if isinstance(image, dict):
        return str(image.get("path") or image.get("name") or "")
    if hasattr(image, "path"):
        return str(image.path)
    return str(image)


def _get_path(image: Any) -> Optional[str]:
    if image is None:
        return None
    if isinstance(image, dict):
        return image.get("path")
    if hasattr(image, "path"):
        return image.path
    return None


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _validate_image_input(image: Any, label: str) -> Optional[str]:

    if image is None:
        return f"Missing {label} image."

    path = _get_path(image)
    if not path:
        return f"{label} image has no 'path' set."

    if not os.path.exists(path):
        return f"{label} image path does not exist: {path}"

    _, ext = os.path.splitext(path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        return (
            f"{label} image has unsupported extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return None

def multimodal_tool(
    optical_image: Any,
    sar_image: Any,
    query: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_metadata = {
        "module": "multimodal",
        "optical_image": _image_label(optical_image),
        "sar_image": _image_label(sar_image),
    }
    if metadata:
        base_metadata["input_metadata"] = metadata

    optical_error = _validate_image_input(optical_image, "optical")
    if optical_error:
        return _failure(optical_error, base_metadata)

    sar_error = _validate_image_input(sar_image, "sar")
    if sar_error:
        return _failure(sar_error, base_metadata)

    if not query or not query.strip():
        return _failure("Query is empty.", base_metadata)

    answer = "Multimodal pipeline not yet implemented — inputs validated OK."
    confidence = None
    visual_output = None

    return {
        "success": True,
        "answer": answer,
        "confidence": confidence,
        "model": "multimodal-stub-v0",
        "visual_output": visual_output,
        "metadata": base_metadata,
    }


def _failure(message: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": False,
        "answer": "",
        "confidence": None,
        "model": "multimodal-stub-v0",
        "visual_output": None,
        "metadata": {**base_metadata, "error": message},
    }