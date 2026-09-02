from modules.single_image.service import analyze_single_image


def test_requires_an_image_path():
    result = analyze_single_image("", "Describe this image")

    assert result["success"] is False
    assert result["error"] == "Please provide an image path."

def test_rejects_a_missing_image_file():
    result = analyze_single_image("missing_image.tif", "Describe this image")

    assert result["success"] is False
    assert result["error"] == "Image file not found: missing_image.tif"

