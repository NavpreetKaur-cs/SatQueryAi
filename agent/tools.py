from typing import Any, Dict


def _image_label(image: Any) -> str:
    if image is None:
        return ""
    if isinstance(image, dict):
        return str(image.get("path") or image.get("name") or "")
    if hasattr(image, "path"):
        return str(image.path)
    return str(image)


# ============================================================
# REAL SPECIALIST TOOLS
# ============================================================


def single_image_tool(image, query, metadata=None):
    """Run the adapted remote-sensing VLM through the agent contract."""
    from modules.single_image.service import analyze_single_image

    image_input = image.path if hasattr(image, "path") else image
    return analyze_single_image(image_input, query)


def change_analysis_tool(image1, image2, query, metadata=None):
    """Run Person 5's bi-temporal change-analysis implementation."""
    from modules.change_analysis.interface import (
        change_analysis_tool as real_change_analysis_tool,
    )

    return real_change_analysis_tool(
        image1=image1,
        image2=image2,
        query=query,
        metadata=metadata,
    )


def multimodal_tool(optical_image, sar_image, query, metadata=None):
    try:
        from modules.multimodal.interface import multimodal_tool as real_multimodal_tool

        return real_multimodal_tool(
            optical_image=optical_image,
            sar_image=sar_image,
            query=query,
            metadata=metadata,
        )
    except Exception as exc:
        return {
            "success": False,
            "answer": "",
            "confidence": None,
            "model": "multimodal-stub-v0",
            "visual_output": None,
            "metadata": {
                "module": "multimodal",
                "optical_image": _image_label(optical_image),
                "sar_image": _image_label(sar_image),
                "error": str(exc),
            },
        }


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "single_image": {
        "name": "single_image_analysis",
        "description": "Analyzes a single remote-sensing image for VQA, captioning, or grounding.",
        "handler": single_image_tool,
        "input_type": "single_image",
    },
    "change_analysis": {
        "name": "change_analysis",
        "description": "Analyzes two temporal remote-sensing images for changes.",
        "handler": change_analysis_tool,
        "input_type": "bi_temporal",
    },
    "multimodal": {
        "name": "optical_sar_analysis",
        "description": "Combines optical and SAR imagery for cross-modal analysis.",
        "handler": multimodal_tool,
        "input_type": "optical_sar",
    },
}


def get_tool(task: str) -> Dict[str, Any]:
    if task not in TOOL_REGISTRY:
        raise ValueError(f"No tool registered for task: {task}")
    return TOOL_REGISTRY[task]