from unittest.mock import Mock, patch

from modules.single_image.service import analyze_single_image
from modules.single_image.service import get_requested_task

def test_identifies_caption_requests():
    assert get_requested_task("Describe this satellite image") == "caption"


def test_identifies_vqa_requests():
    assert get_requested_task("Is there a water body?") == "vqa"


def test_requires_an_image_path():
    result = analyze_single_image("", "Describe this image")

    assert result["success"] is False
    assert result["error"] == "Please provide an image path."

def test_rejects_a_missing_image_file():
    result = analyze_single_image("missing_image.tif", "Describe this image")

    assert result["success"] is False
    assert result["error"] == "Image file not found: missing_image.tif"


def test_calls_adapted_model_and_preserves_its_output(tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(b"image")
    model = Mock()
    model.infer.return_value = {
        "success": True,
        "answer": "Yes, there is a water body.",
        "confidence": 0.91,
        "visual_output": None,
        "model": "InternVL3-1B-hf-lora",
        "error": None,
    }

    with patch("modules.single_image.service._get_vlm", return_value=model):
        result = analyze_single_image(str(image), "Is there a water body?")

    model.infer.assert_called_once_with(
        image=str(image), query="Is there a water body?"
    )
    assert result["success"] is True
    assert result["answer"] == "Yes, there is a water body."
    assert result["metadata"]["requested_task"] == "vqa"

