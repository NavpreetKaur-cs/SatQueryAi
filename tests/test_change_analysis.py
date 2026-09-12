from pathlib import Path

from agent import process
from agent.request import AgentRequest, ImageInput

before = Path("uploads/before.jpg")
after = Path("uploads/after.jpg")

request = AgentRequest(
    query="What changed between these two satellite images?",
    images=[
        ImageInput(path=str(before)),
        ImageInput(path=str(after)),
    ],
    metadata={
        "output_dir": "outputs/change_analysis/manual_test",
        "threshold": 24,
        "min_region_pixels": 10,
        "save_visuals": True,
    },
)

result = process(request)

print("Success:", result.success)
print("Task:", result.task)
print("Answer:", result.answer)
print("Confidence:", result.confidence)
print("Model:", result.model)
print("Visual output:", result.visual_output)
print("Error:", result.error)
print("Metadata:", result.metadata)