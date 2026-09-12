# from agent import process
# from agent.request import (
#     AgentRequest,
#     ImageInput,
# )
# from agent.tools import get_tool
# from modules.multimodal.interface import multimodal_tool as real_multimodal_tool


# def test_multimodal_uses_real_module_handler():
#     tool = get_tool("multimodal")
#     assert tool["handler"] is real_multimodal_tool


# def test_single_image():

#     request = AgentRequest(

#         query=(
#             "Describe the land cover "
#             "in this image."
#         ),

#         images=[
#             ImageInput(
#                 path="test_image.tif",
#                 modality="optical"
#             )
#         ]
#     )

#     result = process(
#         request
#     )

#     print(
#         result.to_dict()
#     )

#     assert result.success
#     assert result.task == "single_image"


# def test_change_analysis():

#     request = AgentRequest(

#         query=(
#             "What changed between "
#             "these two images?"
#         ),

#         images=[
#             ImageInput(
#                 path="before.tif"
#             ),

#             ImageInput(
#                 path="after.tif"
#             )
#         ]
#     )

#     result = process(
#         request
#     )

#     print(
#         result.to_dict()
#     )

#     assert result.success
#     assert result.task == "change_analysis"


# def test_multimodal():

#     request = AgentRequest(

#         query=(
#             "Use the optical and SAR "
#             "images to identify built-up areas."
#         ),

#         images=[

#             ImageInput(
#                 path="optical.tif",
#                 modality="optical"
#             ),

#             ImageInput(
#                 path="sar.tif",
#                 modality="sar"
#             )
#         ]
#     )

#     result = process(
#         request
#     )

#     print(
#         result.to_dict()
#     )

#     assert result.success
#     assert result.task == "multimodal"
from pathlib import Path

from agent import process
from agent.request import AgentRequest, ImageInput


def test_trained_multimodal_model():
    optical_path = Path("data/optical_sar/optical.png")
    sar_path = Path("data/optical_sar/sar.png")

    assert optical_path.exists()
    assert sar_path.exists()

    request = AgentRequest(
        query="Use the optical and SAR images to determine whether built-up areas are present.",
        images=[
            ImageInput(str(optical_path), modality="optical"),
            ImageInput(str(sar_path), modality="sar"),
        ],
    )

    result = process(request)
    output = result.to_dict()

    print(output)

    assert result.task == "multimodal"
    assert result.success is True

    # Confirms the trained checkpoint was loaded.
    assert result.model == "multimodal-v0"

    # A trained model should provide an answer and confidence.
    assert result.answer
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0

    summary = result.metadata["execution_summary"]
    assert summary["selected_task"] == "multimodal"
    assert "optical_sar_analysis" in summary["tools_used"]

    print("Answer:", result.answer)
    print("Confidence:", result.confidence)
    print("Model:", result.model)
    print("Visual output:", result.visual_output)