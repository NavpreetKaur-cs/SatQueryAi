from pathlib import Path
from collections import Counter

from PIL import Image
import numpy as np


# ============================================================
# OFFICIAL SECOND DATASET COLOR MAPPING
# ============================================================

SECOND_PALETTE = {
    (255, 255, 255): "no_change",
    (0, 128, 0): "low_vegetation",
    (0, 255, 0): "tree",
    (128, 128, 128): "nvg_surface",
    (0, 0, 255): "water",
    (128, 0, 0): "buildings",
    (255, 0, 0): "playgrounds",
}


# Human-readable names used by CDVQA
CLASS_NAMES = {
    "nvg_surface": "non-vegetated ground surface",
    "tree": "trees",
    "low_vegetation": "low vegetation",
    "water": "water",
    "buildings": "buildings",
    "playgrounds": "playgrounds",
}


# ============================================================
# LOAD SECOND LABEL
# ============================================================

def load_second_label(path):
    """
    Load a SECOND semantic label image and convert
    RGB colors into semantic class names.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Label not found: {path}"
        )

    image = Image.open(path).convert("RGB")
    array = np.array(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected RGB label image, got shape {array.shape}"
        )

    class_map = np.empty(
        array.shape[:2],
        dtype=object
    )

    unknown_colors = set()

    unique_colors = np.unique(
        array.reshape(-1, 3),
        axis=0
    )

    for color in unique_colors:

        rgb = tuple(
            int(value)
            for value in color
        )

        if rgb not in SECOND_PALETTE:
            unknown_colors.add(rgb)
            continue

        mask = np.all(
            array == np.array(rgb),
            axis=2
        )

        class_map[mask] = SECOND_PALETTE[rgb]

    if unknown_colors:
        raise ValueError(
            "Unknown SECOND label colors found: "
            f"{sorted(unknown_colors)}"
        )

    return class_map


# ============================================================
# CLASS COUNTS
# ============================================================

def get_class_counts(label_map):
    """
    Count pixels belonging to each semantic class.
    """

    counter = Counter(
        label_map.reshape(-1)
    )

    return dict(counter)


# ============================================================
# CLASS PERCENTAGES
# ============================================================

def get_class_percentages(label_map):
    """
    Calculate the percentage of the image occupied
    by each semantic class.
    """

    total_pixels = label_map.size

    if total_pixels == 0:
        return {}

    counts = get_class_counts(label_map)

    percentages = {}

    for class_name in SECOND_PALETTE.values():

        percentages[class_name] = (
            counts.get(class_name, 0)
            / total_pixels
            * 100
        )

    return percentages


# ============================================================
# SECOND LABEL ANALYSIS
# ============================================================

def analyze_second_labels(label1_path, label2_path):
    """
    Analyze a pair of SECOND semantic change labels.

    label1 = first / pre-event image
    label2 = second / post-event image
    """

    label1 = load_second_label(label1_path)
    label2 = load_second_label(label2_path)

    if label1.shape != label2.shape:
        raise ValueError(
            "Label dimensions do not match: "
            f"{label1.shape} vs {label2.shape}"
        )

    total_pixels = label1.size

    if total_pixels == 0:
        raise ValueError(
            "Label images contain no pixels."
        )

    # ========================================================
    # BEFORE / AFTER CLASS PERCENTAGES
    # ========================================================

    before_percent = get_class_percentages(label1)
    after_percent = get_class_percentages(label2)

    # ========================================================
    # CHANGED / UNCHANGED PIXELS
    # ========================================================

    # White represents "no change".
    # A pixel is part of a changed region if either
    # label marks it as a semantic change class.

    changed_mask = (
        (label1 != "no_change")
        |
        (label2 != "no_change")
    )

    changed_pixels = int(
        np.sum(changed_mask)
    )

    unchanged_pixels = (
        total_pixels
        - changed_pixels
    )

    changed_percentage = (
        changed_pixels
        / total_pixels
        * 100
    )

    unchanged_percentage = (
        unchanged_pixels
        / total_pixels
        * 100
    )

    # ========================================================
    # CLASS-WISE CHANGES
    # ========================================================

    class_changes = {}

    for class_name in CLASS_NAMES:

        before_count = int(
            np.sum(label1 == class_name)
        )

        after_count = int(
            np.sum(label2 == class_name)
        )

        delta = (
            after_count
            - before_count
        )

        class_changes[class_name] = {

            "before_count": before_count,

            "after_count": after_count,

            "before_percent": (
                before_count
                / total_pixels
                * 100
            ),

            "after_percent": (
                after_count
                / total_pixels
                * 100
            ),

            "delta_pixels": delta,

            "delta_percentage_points": (
                delta
                / total_pixels
                * 100
            ),

            # This will be corrected below after
            # transition information is available.
            "changed": False,
        }

    # ========================================================
    # TRANSITION MATRIX
    # ========================================================

    transitions = {}

    for source_class in CLASS_NAMES:

        transitions[source_class] = {}

        source_mask = (
            label1 == source_class
        )

        for target_class in CLASS_NAMES:

            count = int(
                np.sum(
                    source_mask
                    &
                    (label2 == target_class)
                )
            )

            if count > 0:

                transitions[
                    source_class
                ][target_class] = count

    # ========================================================
    # DETERMINE WHICH CLASSES ACTUALLY CHANGED
    # ========================================================

    # A class is considered changed if pixels belonging
    # to that class transition to a different semantic class.
    #
    # Merely having pixels of a class does NOT mean the class
    # itself changed.

    for class_name in CLASS_NAMES:

        outgoing_changes = 0

        incoming_changes = 0

        # ----------------------------------------------------
        # Pixels that were this class before and became
        # another class after
        # ----------------------------------------------------

        source_transitions = transitions.get(
            class_name,
            {}
        )

        for target_class, count in source_transitions.items():

            if target_class != class_name:
                outgoing_changes += count

        # ----------------------------------------------------
        # Pixels that became this class after coming from
        # another class before
        # ----------------------------------------------------

        for source_class, target_data in transitions.items():

            if source_class == class_name:
                continue

            incoming_changes += target_data.get(
                class_name,
                0
            )

        class_changes[class_name]["changed"] = (
            outgoing_changes > 0
            or incoming_changes > 0
        )

        class_changes[class_name][
            "outgoing_change_pixels"
        ] = outgoing_changes

        class_changes[class_name][
            "incoming_change_pixels"
        ] = incoming_changes

        class_changes[class_name][
            "total_transition_pixels"
        ] = (
            outgoing_changes
            + incoming_changes
        )

    # ========================================================
    # RETURN ANALYSIS
    # ========================================================

    return {

        "image_shape": label1.shape,

        "total_pixels": total_pixels,

        "changed_pixels": changed_pixels,

        "unchanged_pixels": unchanged_pixels,

        "changed_percentage": changed_percentage,

        "unchanged_percentage": unchanged_percentage,

        "before_percent": before_percent,

        "after_percent": after_percent,

        "class_changes": class_changes,

        "transitions": transitions,
    }


# ============================================================
# ANSWER NORMALIZATION
# ============================================================

def normalize_semantic_answer(answer):
    """
    Normalize semantic answers so that equivalent singular
    and plural forms are treated consistently.
    """

    if answer is None:
        return "unknown"

    answer = str(answer).lower().strip()

    aliases = {

        "tree": "trees",

        "trees": "trees",

        "building": "buildings",

        "buildings": "buildings",

        "playground": "playgrounds",

        "playgrounds": "playgrounds",

        "low_vegetation": "low vegetation",

        "low vegetation": "low vegetation",

        "water": "water",

        "nvg_surface": "NVG_surface",

        "nvg surface": "NVG_surface",

        "non-vegetated ground surface":
            "NVG_surface",

        "non vegetated ground surface":
            "NVG_surface",
    }

    return aliases.get(
        answer,
        answer
    )


# ============================================================
# CDVQA SEMANTIC QUESTION ANSWERER
# ============================================================

def answer_semantic_question(question, analysis):
    """
    Answer a CDVQA-style question using SECOND
    semantic label analysis.
    """

    q = question.lower().strip()

    # ========================================================
    # IDENTIFY SEMANTIC CLASS
    # ========================================================

    class_aliases = {

        "non-vegetated ground surface":
            "nvg_surface",

        "non vegetated ground surface":
            "nvg_surface",

        "nvg_surface":
            "nvg_surface",

        "nvg surface":
            "nvg_surface",

        "trees":
            "tree",

        "tree":
            "tree",

        "low vegetation":
            "low_vegetation",

        "low_vegetation":
            "low_vegetation",

        "water":
            "water",

        "buildings":
            "buildings",

        "building":
            "buildings",

        "playgrounds":
            "playgrounds",

        "playground":
            "playgrounds",
    }

    detected_class = None

    # Check longer phrases first.
    for name in sorted(
        class_aliases,
        key=len,
        reverse=True
    ):

        if name in q:

            detected_class = (
                class_aliases[name]
            )

            break

    # ========================================================
    # CHANGE TO WHAT?
    # ========================================================

    if (
        "changed to" in q
        or "change to" in q
    ):

        if detected_class is None:
            return "unknown"

        transitions = (
            analysis["transitions"]
            .get(detected_class, {})
        )

        # CDVQA asks for the class that the regions of the
        # queried class mainly correspond to in the other image.
        # Same-class transitions are valid answers too.
        if not transitions:
            return normalize_semantic_answer(detected_class)

        target = max(
            transitions,
            key=transitions.get
        )

        return normalize_semantic_answer(target)

    # ========================================================
    # LARGEST / SMALLEST CHANGE
    # ========================================================

    if (
        "largest change" in q
        or "smallest change" in q
        or "largest" in q
        or "smallest" in q
    ):

        class_changes = (
            analysis["class_changes"]
        )

        # ----------------------------------------------------
        # FIRST / PRE-EVENT IMAGE
        # ----------------------------------------------------

        if (
            "pre-change" in q
            or "pre-event" in q
            or "first image" in q
        ):

            values = {
                cls: data["before_count"]
                for cls, data
                in class_changes.items()
            }

        # ----------------------------------------------------
        # SECOND / POST-EVENT IMAGE
        # ----------------------------------------------------

        elif (
            "second image" in q
            or "post-event" in q
            or "post-change" in q
        ):

            values = {
                cls: data["after_count"]
                for cls, data
                in class_changes.items()
            }

        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        else:

            values = {
                cls: data["total_transition_pixels"]
                for cls, data
                in class_changes.items()
            }

        # ----------------------------------------------------
        # LARGEST
        # ----------------------------------------------------

        if "largest" in q:

            result = max(
                values,
                key=values.get
            )

        # ----------------------------------------------------
        # SMALLEST
        # ----------------------------------------------------

        else:

            nonzero = {
                cls: value
                for cls, value
                in values.items()
                if value > 0
            }

            if not nonzero:
                return "unknown"

            result = min(
                nonzero,
                key=nonzero.get
            )

        if result == "nvg_surface":
            return "NVG_surface"

        if result == "tree":
            return "trees"

        if result == "building":
            return "buildings"

        if result == "playground":
            return "playgrounds"

        return result

    # ========================================================
    # CLASS-SPECIFIC CHANGE PERCENTAGE
    # ========================================================
    #
    # IMPORTANT:
    # Questions such as "change ratio of trees in the
    # post-event image" are class-specific. They must be
    # answered from the corresponding semantic change label,
    # NOT from the overall image change percentage.
    #
    # The class percentage is the number of pixels assigned
    # to that class in the requested label divided by the
    # total image area.

    if (
        detected_class is not None
        and (
            "change percentage" in q
            or "change proportion" in q
            or "change ratio" in q
            or "how much area" in q
        )
    ):

        class_data = (
            analysis[
                "class_changes"
            ][detected_class]
        )

        # ----------------------------------------------------
        # FIRST / PRE-EVENT
        # ----------------------------------------------------

        if (
            "pre-change" in q
            or "pre-event" in q
            or "first image" in q
        ):

            percentage = (
                class_data[
                    "before_percent"
                ]
            )

        # ----------------------------------------------------
        # SECOND / POST-EVENT
        # ----------------------------------------------------

        elif (
            "second image" in q
            or "post-event" in q
            or "post-change" in q
        ):

            percentage = (
                class_data[
                    "after_percent"
                ]
            )

        else:

            # If no time is specified, use the first image
            # as the default, matching the existing behavior.
            percentage = (
                class_data[
                    "before_percent"
                ]
            )

        return _percentage_bucket(
            percentage
        )

    # ========================================================
    # OVERALL CHANGE PERCENTAGE
    # ========================================================

    if (
        detected_class is None
        and (
            "percentage of changed regions" in q
            or "percentage of changed areas" in q
            or "percentage of changed area" in q
            or "how much of the area has changed" in q
            or "how much area has changed" in q
            or "change ratio" in q
            or "change proportion" in q
        )
    ):

        percentage = (
            analysis[
                "changed_percentage"
            ]
        )

        return _percentage_bucket(
            percentage
        )

    # ========================================================
    # OVERALL UNCHANGED PERCENTAGE
    # ========================================================

    if (
        "percentage of non-change regions" in q
        or "percentage of unchanged areas" in q
        or "percentage of unchanged area" in q
        or "how much of the area has not changed" in q
        or "how much area has not changed" in q
        or "percentage of non changed" in q
    ):

        percentage = (
            analysis[
                "unchanged_percentage"
            ]
        )

        return _percentage_bucket(
            percentage
        )

    # ========================================================
    # INCREASE
    # ========================================================

    if (
        "increase" in q
        or "increased" in q
    ):

        if detected_class is None:
            return "unknown"

        delta = (
            analysis[
                "class_changes"
            ][detected_class][
                "delta_pixels"
            ]
        )

        if delta > 0:
            return "yes"

        return "no"

    # ========================================================
    # DECREASE
    # ========================================================

    if (
        "decrease" in q
        or "decreased" in q
    ):

        if detected_class is None:
            return "unknown"

        delta = (
            analysis[
                "class_changes"
            ][detected_class][
                "delta_pixels"
            ]
        )

        if delta < 0:
            return "yes"

        return "no"

    # ========================================================
    # GENERAL CHANGE
    # ========================================================

    if (
        "change" in q
        or "changed" in q
    ):

        if detected_class is None:
            return "unknown"

        class_data = (
            analysis[
                "class_changes"
            ][detected_class]
        )

        # For CDVQA "did X change in the first/second image?"
        # the queried label directly represents the changed
        # regions for that time point.
        if (
            "pre-change" in q
            or "pre-event" in q
            or "first image" in q
        ):

            count = class_data["before_count"]

        elif (
            "second image" in q
            or "post-event" in q
            or "post-change" in q
        ):

            count = class_data["after_count"]

        else:
            count = class_data["total_transition_pixels"]

        return "yes" if count > 0 else "no"

    # ========================================================
    # UNKNOWN QUESTION
    # ========================================================

    return "unknown"


# ============================================================
# PERCENTAGE BUCKET
# ============================================================

def _percentage_bucket(percentage):
    """
    Convert a numerical percentage into
    the bucket format used by CDVQA.
    """

    if percentage == 0:
        return "0"

    if percentage < 10:
        return "0_to_10"

    if percentage < 20:
        return "10_to_20"

    if percentage < 30:
        return "20_to_30"

    if percentage < 40:
        return "30_to_40"

    if percentage < 50:
        return "40_to_50"

    if percentage < 60:
        return "50_to_60"

    if percentage < 70:
        return "60_to_70"

    if percentage < 80:
        return "70_to_80"

    if percentage < 90:
        return "80_to_90"

    return "90_to_100"