from pathlib import Path
from typing import Any, Dict


_REMOTE_SENSING_VLM = None
_ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "adaptation"
    / "checkpoints"
    / "best"
)


def _image_label(image: Any) -> str:
    if image is None:
        return ""
    if isinstance(image, dict):
        return str(image.get("path") or image.get("name") or "")
    if hasattr(image, "path"):
        return str(image.path)
    return str(image)


# ============================================================
# MOCK SPECIALIST TOOLS
# ============================================================


def single_image_tool(image, query, metadata=None):
    """Run Person 1's adapted VLM through the agent tool contract."""
    global _REMOTE_SENSING_VLM

    from modules.model_adaptation.infer import RemoteSensingVLM

    if _REMOTE_SENSING_VLM is None:
        _REMOTE_SENSING_VLM = RemoteSensingVLM(
            adapter_path=_ADAPTER_PATH
        )

    image_input = image.path if hasattr(image, "path") else image
    return _REMOTE_SENSING_VLM.infer(
        image=image_input,
        query=query,
    )


def change_analysis_tool(image1, image2, query, metadata=None):
    return {
        "success": True,
        "answer": "Mock change-analysis result.",
        "confidence": 0.85,
        "model": "change-analysis-baseline",
        "visual_output": None,
        "metadata": {
            "module": "change_analysis",
            "image1": _image_label(image1),
            "image2": _image_label(image2),
        },
    }


def multimodal_tool(optical_image, sar_image, query, metadata=None):
    return {
        "success": True,
        "answer": "Mock optical + SAR analysis.",
        "confidence": 0.82,
        "model": "multimodal-baseline",
        "visual_output": None,
        "metadata": {
            "module": "multimodal",
            "optical_image": _image_label(optical_image),
            "sar_image": _image_label(sar_image),
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