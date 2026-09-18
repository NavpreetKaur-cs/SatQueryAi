from agent.request import ImageInput
from agent.router import route_query


def test_single_image_is_selected_from_image_count():
    decision = route_query(
        "What is visible in this scene?",
        [ImageInput("scene.tif", modality="optical")],
    )

    assert decision.task == "single_image"
    assert decision.method == "image_count"


def test_two_unlabelled_images_are_change_analysis():
    decision = route_query(
        "What changed between these images?",
        [ImageInput("before.tif"), ImageInput("after.tif")],
    )

    assert decision.task == "change_analysis"
    assert decision.method == "image_count"


def test_optical_and_sar_images_are_multimodal():
    decision = route_query(
        "Identify built-up areas.",
        [
            ImageInput("optical.tif", modality="optical"),
            ImageInput("sar.tif", modality="sar"),
        ],
    )

    assert decision.task == "multimodal"
    assert decision.method == "image_modalities"


def test_explicit_optical_sar_query_is_multimodal_without_labels():
    decision = route_query(
        "Use the optical and SAR images together.",
        [ImageInput("first.tif"), ImageInput("second.tif")],
    )

    assert decision.task == "multimodal"
