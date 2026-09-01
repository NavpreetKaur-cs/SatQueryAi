from agent import process
from agent.request import (
    AgentRequest,
    ImageInput,
)


def test_single_image():

    request = AgentRequest(

        query=(
            "Describe the land cover "
            "in this image."
        ),

        images=[
            ImageInput(
                path="test_image.tif",
                modality="optical"
            )
        ]
    )

    result = process(
        request
    )

    print(
        result.to_dict()
    )

    assert result.success
    assert result.task == "single_image"


def test_change_analysis():

    request = AgentRequest(

        query=(
            "What changed between "
            "these two images?"
        ),

        images=[
            ImageInput(
                path="before.tif"
            ),

            ImageInput(
                path="after.tif"
            )
        ]
    )

    result = process(
        request
    )

    print(
        result.to_dict()
    )

    assert result.success
    assert result.task == "change_analysis"


def test_multimodal():

    request = AgentRequest(

        query=(
            "Use the optical and SAR "
            "images to identify built-up areas."
        ),

        images=[

            ImageInput(
                path="optical.tif",
                modality="optical"
            ),

            ImageInput(
                path="sar.tif",
                modality="sar"
            )
        ]
    )

    result = process(
        request
    )

    print(
        result.to_dict()
    )

    assert result.success
    assert result.task == "multimodal"