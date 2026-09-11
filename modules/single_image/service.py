from pathlib import Path


_REMOTE_SENSING_VLM = None

CAPTION_WORDS = ("describe", "caption", "what is visible", "what do you see")


def get_requested_task(query):
    text = (query or "").lower()

    if any(word in text for word in CAPTION_WORDS):
        return "caption"

    return "vqa"


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

    if isinstance(image_path, (str, Path)):
        if not Path(image_path).is_file():
            return _failure(f"Image file not found: {image_path}", image_path, query)

        supported_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

        if Path(image_path).suffix.lower() not in supported_extensions:
            return _failure("Unsupported image format.", image_path, query)

    else:
        try:
            from PIL import Image
        except ImportError:
            return _failure("Pillow is required for PIL image inputs.", image_path, query)

        if not isinstance(image_path, Image.Image):
            return _failure("Unsupported image input type.", image_path, query)

    if not query:
        return _failure("Please provide a question about the image.", image_path, query)

    global _REMOTE_SENSING_VLM

    try:
        if _REMOTE_SENSING_VLM is None:
            from modules.model_adaptation.infer import RemoteSensingVLM

            _REMOTE_SENSING_VLM = RemoteSensingVLM()

        result = _REMOTE_SENSING_VLM.infer(image=image_path, query=query)
        result.setdefault("metadata", {})
        result["metadata"].update({
            "module": "single_image",
            "image_path": str(image_path),
            "query": query,
            "requested_task": get_requested_task(query),
        })
        return result
    except Exception as exc:
        return _failure(str(exc), image_path, query)
