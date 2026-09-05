from pathlib import Path
from functools import lru_cache

CAPTION_WORDS = ("describe", "caption", "what is visible", "what do you see")


def get_requested_task(query):
    text = (query or "").lower()

    if any(word in text for word in CAPTION_WORDS):
        return "caption"

    return "vqa"


@lru_cache(maxsize=1)
def _get_vlm():
    """Create the adapted VLM once and reuse it across image requests."""
    from modules.model_adaptation.infer import RemoteSensingVLM

    return RemoteSensingVLM()


def _failure(error, image_path, query):
    return {
        "success": False,
        "answer": "",
        "confidence": None,
        "model": None,
        "visual_output": None,
        "metadata": {
            "module": "single_image",
            "image_path": str(image_path),
            "query": query,
            "requested_task": get_requested_task(query),
        },
        "error": error,
    }


def analyze_single_image(image_path, query):
    if not image_path:
        return _failure("Please provide an image path.", image_path, query)
    if not Path(image_path).is_file():
        return _failure(f"Image file not found: {image_path}", image_path, query)

    supported_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    if Path(image_path).suffix.lower() not in supported_extensions:
        return _failure("Unsupported image format.", image_path, query)

    if not query:
        return _failure("Please provide a question about the image.", image_path, query)

    try:
        result = _get_vlm().infer(image=image_path, query=query)
    except Exception as exc:
        return _failure(f"Remote-sensing inference failed: {exc}", image_path, query)

    if not isinstance(result, dict):
        return _failure("Remote-sensing model returned an invalid response.", image_path, query)

    # The adaptation module owns the shared output contract.  Add service context
    # without changing the model's answer, confidence, or visual output.
    result.setdefault("success", False)
    result.setdefault("answer", "")
    result.setdefault("confidence", None)
    result.setdefault("visual_output", None)
    result.setdefault("model", None)
    result.setdefault("error", None)
    result["metadata"] = {
        **result.get("metadata", {}),
        "module": "single_image",
        "image_path": str(image_path),
        "query": query,
        "requested_task": get_requested_task(query),
    }
    return result
