import cv2
import numpy as np

from modules.change_analysis.interface import change_analysis_tool


def test_detects_and_visualizes_synthetic_change(tmp_path):
    before = np.full((80, 80, 3), (40, 140, 40), dtype=np.uint8)
    after = before.copy()
    # A bright, low-saturation region is classified as the built-up proxy.
    after[25:55, 25:55] = (180, 180, 180)
    before_path, after_path = tmp_path / "before.png", tmp_path / "after.png"
    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)

    result = change_analysis_tool(
        str(before_path), str(after_path), "Has built-up area changed?",
        {"output_dir": str(tmp_path / "outputs"), "min_region_pixels": 10},
    )

    assert result["success"]
    assert "increased" in result["answer"]
    assert result["metadata"]["change_percentage"] > 10
    assert (tmp_path / "outputs" / "change_map.png").exists()
    assert (tmp_path / "outputs" / "change_overlay.png").exists()


def test_rejects_missing_input():
    result = change_analysis_tool("missing-before.png", "missing-after.png", "What changed?")
    assert not result["success"]
    assert "Cannot read image" in result["metadata"]["error"]
