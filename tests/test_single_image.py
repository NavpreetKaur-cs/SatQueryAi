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

