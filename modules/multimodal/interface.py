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

from .preprocessing import load_and_validate_pair, CoRegistrationError
from .feature_extraction import MultimodalFeatureExtractor
from .fusion import MultimodalFusion
from .task_head import TaskHead


# ============================================================
# Helpers
# ============================================================

def _image_label(image: Any) -> str:
    """
    Mirrors agent/tools.py::_image_label so metadata output is consistent
    whether the caller passes an ImageInput-like object or a plain dict.
    """
    if image is None:
        return ""
    if isinstance(image, dict):
        return str(image.get("path") or image.get("name") or "")
    if hasattr(image, "path"):
        return str(image.path)
    return str(image)


def _get_path(image: Any) -> Optional[str]:
    """Extract a filesystem path from an ImageInput-like object or dict."""
    if image is None:
        return None
    if isinstance(image, dict):
        return image.get("path")
    if hasattr(image, "path"):
        return image.path
    return None


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _validate_image_input(image: Any, label: str) -> Optional[str]:
    """
    Basic sanity checks on a single image input.
    Returns an error message string if invalid, otherwise None.
    """
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

# Built once and reused across calls, since MultimodalFeatureExtractor
# loads CNN backbones — recreating it per-request would be slow and
# wasteful. If this module is ever loaded in a multi-process server
# without a shared cache, each process will build its own copy, which
# is fine (they're independent, stateless-per-call, in-memory models).

_feature_extractor: Optional[MultimodalFeatureExtractor] = None
_fusion: Optional[MultimodalFusion] = None
_task_head: Optional[TaskHead] = None


def _get_pipeline_components():
    global _feature_extractor, _fusion, _task_head
    if _feature_extractor is None:
        _feature_extractor = MultimodalFeatureExtractor(
            optical_channels=3, sar_channels=1, pretrained=False
        )
    if _fusion is None:
        _fusion = MultimodalFusion(strategy="concat", feature_dim=512)
    if _task_head is None:
        _task_head = TaskHead(feature_dim=512)
    return _feature_extractor, _fusion, _task_head


def multimodal_tool(
    optical_image: Any,
    sar_image: Any,
    query: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Real entry point for optical + SAR joint analysis.

    Args:
        optical_image: ImageInput-like object or dict, with a 'path' to the
            optical/multispectral image.
        sar_image: ImageInput-like object or dict, with a 'path' to the
            SAR image.
        query: natural-language question about the image pair.
        metadata: optional extra context passed through from the agent.

    Returns:
        dict matching the AgentResult schema (minus 'task'/'error', which
        the executor fills in):
            {
                "success": bool,
                "answer": str,
                "confidence": float | None,
                "model": str,
                "visual_output": Any | None,
                "metadata": dict,
            }
    """
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

    optical_path = _get_path(optical_image)
    sar_path = _get_path(sar_image)

    try:
        optical_arr, sar_arr = load_and_validate_pair(optical_path, sar_path)
    except CoRegistrationError as e:
        return _failure(f"Co-registration check failed: {e}", base_metadata)
    except Exception as e:
        # Covers rasterio I/O errors (corrupt file, unreadable format, etc.)
        return _failure(f"Failed to load images: {e}", base_metadata)

    try:
        extractor, fusion, task_head = _get_pipeline_components()

        if (
            extractor.optical_backbone[0].in_channels != optical_arr.shape[0]
            or extractor.sar_backbone[0].in_channels != sar_arr.shape[0]
        ):
            extractor = MultimodalFeatureExtractor(
                optical_channels=optical_arr.shape[0],
                sar_channels=sar_arr.shape[0],
                pretrained=False,
            )

        optical_feat, sar_feat = extractor.extract(optical_arr, sar_arr)
        fused_vector = fusion.fuse(optical_feat.vector, sar_feat.vector)
        task_result = task_head.answer_query(fused_vector, query)
    except Exception as e:
        return _failure(f"Pipeline error: {e}", base_metadata)

    base_metadata["category"] = task_result.category

    return {
        "success": True,
        "answer": task_result.answer,
        "confidence": task_result.confidence,
        "model": "multimodal-v0" + ("-untrained" if not task_head.is_trained else ""),
        "visual_output": None,  # TODO: wire up visualize.py once it exists
        "metadata": base_metadata,
    }


def _failure(message: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Standard failure response, kept in the same schema as success."""
    return {
        "success": False,
        "answer": "",
        "confidence": None,
        "model": "multimodal-stub-v0",
        "visual_output": None,
        "metadata": {**base_metadata, "error": message},
    }