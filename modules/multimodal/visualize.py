"""
modules/multimodal/visualize.py

Generates a visual artifact (an overlay image) for the agent's
`visual_output` field, so the user gets something to look at alongside
the text answer — not just a number.

Since the classifier currently predicts presence/absence for a whole
image pair (not a per-pixel mask — that would require a segmentation
model, which is a bigger undertaking than this module's task_head),
this produces a simple, honest visualization: the optical image with a
colored border/banner indicating which category was asked about and
whether it was predicted present, plus a confidence readout. This is
useful today and easy to upgrade later if task_head.py grows into a
per-pixel segmentation model instead of whole-image classification.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CATEGORY_COLORS = {
    "built_up": (200, 60, 60),     # red
    "water": (60, 120, 200),        # blue
    "vegetation": (60, 160, 80),    # green
    "unknown": (140, 140, 140),      # gray
}


def _optical_array_to_uint8_rgb(optical_array: np.ndarray) -> np.ndarray:
    """
    Convert a (bands, height, width) normalized float array (as produced
    by preprocessing.normalize_optical) into a (height, width, 3) uint8
    RGB array suitable for saving as a viewable image.
    """
    arr = optical_array
    if arr.shape[0] >= 3:
        rgb = arr[:3]  # assume first 3 bands are usable as RGB (e.g. B04/B03/B02 order)
    else:
        rgb = np.repeat(arr[:1], 3, axis=0)  # single-band -> grayscale-as-RGB

    rgb = np.transpose(rgb, (1, 2, 0))  # (height, width, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def generate_overlay(
    optical_array: np.ndarray,
    category: str,
    present: Optional[bool],
    confidence: Optional[float],
    output_path: str,
    border_size: int = 12,
) -> str:
    """
    Save a visualization: the optical image with a colored border
    indicating the queried category, and a text banner with the
    prediction and confidence.

    Args:
        optical_array: normalized (bands, height, width) array from
            preprocessing.normalize_optical.
        category: one of "built_up", "water", "vegetation", "unknown".
        present: True/False if a prediction was made, None if untrained
            or category unknown.
        confidence: presence probability, or None.
        output_path: where to save the resulting PNG.
        border_size: thickness in pixels of the colored border.

    Returns:
        The output_path (for convenience / chaining).
    """
    rgb = _optical_array_to_uint8_rgb(optical_array)
    image = Image.fromarray(rgb, mode="RGB")

    color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])

    # Add a colored border around the image to flag the queried category.
    bordered_width = image.width + 2 * border_size
    bordered_height = image.height + 2 * border_size
    bordered = Image.new("RGB", (bordered_width, bordered_height), color)
    bordered.paste(image, (border_size, border_size))

    # Add a text banner below with the answer summary.
    banner_height = 30
    canvas = Image.new(
        "RGB", (bordered_width, bordered_height + banner_height), (255, 255, 255)
    )
    canvas.paste(bordered, (0, 0))

    draw = ImageDraw.Draw(canvas)
    if present is None:
        label = f"{category}: model not trained"
    else:
        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        label = f"{category}: {'present' if present else 'not present'} (conf={conf_str})"

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((8, bordered_height + 6), label, fill=(0, 0, 0), font=font)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return output_path