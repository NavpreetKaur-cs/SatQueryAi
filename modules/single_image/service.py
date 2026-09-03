from pathlib import Path

CAPTION_WORDS = ("describe", "caption", "what is visible", "what do you see")


def get_requested_task(query):
    text = query.lower()

    if any(word in text for word in CAPTION_WORDS):
        return "caption"

    return "vqa"


def analyze_single_image(image_path, query):
    if not image_path:
        return {
            "success": False,
            "answer": "",
            "error": "Please provide an image path.",
        }
    if not Path(image_path).is_file():
        return {
            "success": False,
            "answer": "",
            "error": f"Image file not found: {image_path}",
        }

    supported_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    if Path(image_path).suffix.lower() not in supported_extensions:
        return {
            "success": False,
            "answer": "",
            "error": "Unsupported image format.",
        }
    
    if not query:
        return {
            "success": False,
            "answer": "",
            "error": "Please provide a question about the image.",
            "requested_task": get_requested_task(query),
        }
    return {
        "success": False,
        "answer": "",
        "confidence": None,
        "model": None,
        "visual_output": None,
        "metadata": {
            "module": "single_image",
            "image_path": image_path,
            "query": query,
        },
        "error": "Remote-sensing model is not connected yet.",
    }
