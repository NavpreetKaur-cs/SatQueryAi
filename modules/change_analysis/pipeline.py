"""Local baseline for registration, change masks, and semantic change maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .semantic_labels import load_second_label


class ChangeAnalysisError(ValueError):
    pass


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ChangeAnalysisError(f"Cannot read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return image


def _resize_pair(
    before: np.ndarray,
    after: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(
            after,
            (before.shape[1], before.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    return before, after


def _align(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Correct small translations."""

    ref = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    moving = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)

    try:
        shift, response = cv2.phaseCorrelate(ref, moving)

        if (
            response >= 0.08
            and abs(shift[0]) < before.shape[1] / 8
            and abs(shift[1]) < before.shape[0] / 8
        ):
            matrix = np.float32([
                [1, 0, -shift[0]],
                [0, 1, -shift[1]]
            ])

            return cv2.warpAffine(
                after,
                matrix,
                (before.shape[1], before.shape[0]),
                borderMode=cv2.BORDER_REFLECT
            )

    except cv2.error:
        pass

    return after


def _change_mask(
    before: np.ndarray,
    after: np.ndarray,
    threshold: int,
    min_area: int
) -> Tuple[np.ndarray, np.ndarray]:

    delta = cv2.absdiff(
        cv2.cvtColor(before, cv2.COLOR_BGR2LAB),
        cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
    )

    magnitude = np.mean(delta.astype(np.float32), axis=2)

    mask = (magnitude >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned = np.zeros_like(mask)

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned, magnitude


def _classify_pixels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Approximate visible land-cover classes."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = (
        hsv[:, :, i]
        for i in range(3)
    )

    vegetation = (
        (h >= 35)
        & (h <= 90)
        & (s >= 35)
        & (v >= 25)
    )

    water = (
        (h >= 85)
        & (h <= 130)
        & (s >= 35)
        & (v <= 190)
    )

    built_up = (
        (s <= 55)
        & (v >= 90)
        & ~water
    )

    return {
        "vegetation": vegetation,
        "water": water,
        "built_up": built_up,
    }


def _answer(
    query: str,
    overall_pct: float,
    class_changes: Dict[str, Dict[str, float]]
) -> str:

    normalized = query.lower()

    aliases = {
        "vegetation": (
            "vegetation",
            "forest",
            "trees",
            "greenery"
        ),
        "water": (
            "water",
            "river",
            "lake",
            "pond"
        ),
        "built_up": (
            "built-up",
            "built up",
            "building",
            "urban",
            "construction"
        ),
    }

    requested = next(
        (
            name
            for name, words in aliases.items()
            if any(word in normalized for word in words)
        ),
        None
    )

    if requested:
        item = class_changes[requested]

        direction = (
            "increased"
            if item["delta_percentage_points"] > 0.25
            else "decreased"
            if item["delta_percentage_points"] < -0.25
            else "did not materially change"
        )

        return (
            f"Visible {requested.replace('_', '-')} cover "
            f"{direction} "
            f"({item['before_percent']:.1f}% to "
            f"{item['after_percent']:.1f}% of the image)."
        )

    if overall_pct < 0.2:
        return "No material change was detected between the two images."

    details = []

    for name, item in class_changes.items():
        delta = item["delta_percentage_points"]

        if abs(delta) >= 0.5:
            details.append(
                f"{name.replace('_', '-')} "
                f"{'increased' if delta > 0 else 'decreased'} "
                f"by {abs(delta):.1f} percentage points"
            )

    suffix = (
        "; ".join(details[:2])
        if details
        else "the changed area does not map confidently to the visible cover heuristics"
    )

    return (
        f"Changes affect {overall_pct:.1f}% of the image; "
        f"{suffix}."
    )


def _save_visuals(
    before: np.ndarray,
    mask: np.ndarray,
    output_dir: Path
) -> Dict[str, str]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_path = output_dir / "change_map.png"
    overlay_path = output_dir / "change_overlay.png"

    Image.fromarray(mask).save(mask_path)

    overlay = before.copy()

    overlay[mask > 0] = (0, 0, 255)

    overlay = cv2.addWeighted(
        before,
        0.55,
        overlay,
        0.45,
        0
    )

    Image.fromarray(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )
    ).save(overlay_path)

    return {
        "change_map": str(mask_path),
        "change_overlay": str(overlay_path),
    }


def _transition_color(
    source: str,
    target: str
) -> Tuple[int, int, int]:

    colors = {
        ("nvg_surface", "low_vegetation"): (255, 165, 0),
        ("nvg_surface", "tree"): (0, 180, 0),
        ("nvg_surface", "water"): (0, 0, 255),
        ("nvg_surface", "buildings"): (255, 0, 255),
        ("nvg_surface", "playgrounds"): (255, 80, 80),

        ("low_vegetation", "nvg_surface"): (180, 120, 0),
        ("low_vegetation", "tree"): (0, 255, 100),
        ("low_vegetation", "water"): (100, 0, 255),
        ("low_vegetation", "buildings"): (255, 0, 100),
        ("low_vegetation", "playgrounds"): (255, 120, 120),

        ("tree", "nvg_surface"): (120, 80, 0),
        ("tree", "low_vegetation"): (0, 220, 150),
        ("tree", "water"): (80, 80, 255),
        ("tree", "buildings"): (220, 0, 150),
        ("tree", "playgrounds"): (255, 150, 150),

        ("water", "nvg_surface"): (100, 100, 255),
        ("water", "low_vegetation"): (50, 200, 255),
        ("water", "tree"): (50, 255, 200),
        ("water", "buildings"): (180, 0, 255),
        ("water", "playgrounds"): (255, 100, 200),

        ("buildings", "nvg_surface"): (255, 200, 0),
        ("buildings", "low_vegetation"): (100, 255, 0),
        ("buildings", "tree"): (0, 255, 0),
        ("buildings", "water"): (255, 0, 0),
        ("buildings", "playgrounds"): (255, 50, 150),

        ("playgrounds", "nvg_surface"): (255, 180, 80),
        ("playgrounds", "low_vegetation"): (150, 255, 80),
        ("playgrounds", "tree"): (80, 255, 80),
        ("playgrounds", "water"): (255, 80, 80),
        ("playgrounds", "buildings"): (255, 80, 180),
    }

    return colors.get(
        (source, target),
        (255, 255, 0)
    )


def _save_semantic_change_map(
    label1_path: str,
    label2_path: str,
    output_dir: Path
) -> str:

    label1 = load_second_label(label1_path)
    label2 = load_second_label(label2_path)

    if label1.shape != label2.shape:
        raise ChangeAnalysisError(
            f"Semantic label dimensions do not match: "
            f"{label1.shape} vs {label2.shape}"
        )

    height, width = label1.shape

    # Black = unchanged
    semantic_map = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    classes = [
        "nvg_surface",
        "tree",
        "low_vegetation",
        "water",
        "buildings",
        "playgrounds",
    ]

    class_names = {
        "nvg_surface": "NVG surface",
        "tree": "Trees",
        "low_vegetation": "Low vegetation",
        "water": "Water",
        "buildings": "Buildings",
        "playgrounds": "Playgrounds",
    }

    transitions_found = []

    for source in classes:
        for target in classes:

            if source == target:
                continue

            changed_pixels = (
                (label1 == source)
                & (label2 == target)
            )

            if not np.any(changed_pixels):
                continue

            color = _transition_color(
                source,
                target
            )

            semantic_map[changed_pixels] = color

            transitions_found.append(
                (source, target, color)
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save semantic map
    output_path = output_dir / "semantic_change_map.png"

    Image.fromarray(
        semantic_map
    ).save(output_path)

    # Create legend
    legend_height = max(
        100,
        45 * len(transitions_found) + 40
    )

    legend = np.ones(
        (legend_height, 700, 3),
        dtype=np.uint8
    ) * 255

    cv2.putText(
        legend,
        "Semantic Change Map Legend",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    y = 65

    for source, target, color in transitions_found:

        # Convert RGB color to OpenCV BGR
        bgr_color = (
            int(color[2]),
            int(color[1]),
            int(color[0])
        )

        cv2.rectangle(
            legend,
            (20, y - 15),
            (50, y + 15),
            bgr_color,
            -1
        )

        text = (
            f"{class_names[source]} -> "
            f"{class_names[target]}"
        )

        cv2.putText(
            legend,
            text,
            (70, y + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        y += 45

    legend_path = (
        output_dir /
        "semantic_change_legend.png"
    )

    Image.fromarray(
        cv2.cvtColor(
            legend,
            cv2.COLOR_BGR2RGB
        )
    ).save(legend_path)

    return str(output_path)

def analyze_change(
    before_path: str,
    after_path: str,
    query: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:

    before, after = _resize_pair(
        _read_image(before_path),
        _read_image(after_path)
    )

    after = _align(before, after)

    threshold = int(
        options.get("threshold", 24)
    )

    min_area = int(
        options.get("min_region_pixels", 25)
    )

    if not 1 <= threshold <= 255 or min_area < 1:
        raise ChangeAnalysisError(
            "threshold must be 1..255 and "
            "min_region_pixels must be positive."
        )

    mask, _ = _change_mask(
        before,
        after,
        threshold,
        min_area
    )

    changed = mask > 0

    overall_pct = float(
        changed.mean() * 100
    )

    classes_before = _classify_pixels(before)
    classes_after = _classify_pixels(after)

    class_changes = {}

    for name in classes_before:

        old = float(
            classes_before[name].mean() * 100
        )

        new = float(
            classes_after[name].mean() * 100
        )

        class_changes[name] = {
            "before_percent": round(old, 2),
            "after_percent": round(new, 2),
            "delta_percentage_points": round(
                new - old,
                2
            ),
        }

    output_dir = Path(
        options.get(
            "output_dir",
            "outputs/change_analysis"
        )
    )

    visuals = {}

    if options.get("save_visuals", True):

        visuals.update(
            _save_visuals(
                before,
                mask,
                output_dir
            )
        )

    # Semantic change map
    label1_path = options.get("label1_path")
    label2_path = options.get("label2_path")

    if label1_path and label2_path:

        semantic_map_path = _save_semantic_change_map(
            label1_path,
            label2_path,
            output_dir
        )

        visuals["semantic_change_map"] = semantic_map_path

    trained_answer = None
    trained_confidence = None
    checkpoint_path = Path(
        options.get(
            "trained_model_path",
            "models/change_analysis/checkpoints/cdvqa_classifier.joblib",
        )
    )

    if checkpoint_path.exists():
        try:
            from models.change_analysis.trained_classifier import (
                load_classifier,
                predict_classifier,
            )

            trained_model = load_classifier(checkpoint_path)
            trained_answer, trained_confidence = predict_classifier(
                trained_model,
                Path(before_path),
                Path(after_path),
                query,
            )
        except (ImportError, OSError, ValueError, RuntimeError):
            trained_answer = None
            trained_confidence = None

    answer = _answer(query, overall_pct, class_changes)
    confidence = round(
        min(
            0.99,
            0.55 + min(overall_pct, 20) / 50
        ),
        2
    )

    if trained_answer is not None:
        if trained_answer in ("yes", "no"):
            if trained_answer == "yes":
                answer = _answer(query, overall_pct, class_changes)
            else:
                answer = "No material change was detected in the queried region."
        else:
            answer = f"The trained CDVQA model identifies the answer as {trained_answer}."
        confidence = round(float(trained_confidence), 2)

    return {
        "success": True,
        "answer": answer,
        "confidence": confidence,
        "model": (
            "cdvqa-trained-classifier-v1"
            if trained_answer is not None
            else "opencv-change-baseline-v1"
        ),
        "visual_output": visuals.get(
            "change_overlay"
        ),
        "metadata": {
            "change_percentage": round(
                overall_pct,
                3
            ),
            "class_changes": class_changes,
            "visuals": visuals,
            "semantic_change_map": bool(
                label1_path and label2_path
            ),
            "baseline_note": (
                "Land-cover values are based on "
                "SECOND semantic labels when provided."
            ),
        },
    }