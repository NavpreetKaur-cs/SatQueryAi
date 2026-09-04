from agent import process
from agent.request import AgentRequest, ImageInput


def run_test(query, images, expected_task):
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print(f"Expected: {expected_task}")

    request = AgentRequest(
        query=query,
        images=images
    )

    result = process(request)

    print(f"Predicted: {result.task}")
    print(f"Success: {result.success}")
    print(f"Result: {result.to_dict()}")

    assert result.success, f"Agent failed for: {query}"
    assert (
        result.task == expected_task
    ), f"Expected {expected_task}, got {result.task}"


def test_unseen_single_image():

    queries = [
        "What kind of landscape is shown here?",
        "What can you see in this satellite scene?",
        "Are there signs of human settlement?",
        "Which areas appear to contain crops?",
        "What are the dominant features of this region?",
    ]

    for query in queries:

        run_test(
            query=query,
            images=[
                ImageInput(
                    path="test_image.tif",
                    modality="optical"
                )
            ],
            expected_task="single_image"
        )


def test_unseen_change_analysis():

    queries = [
        "Did the city grow during this period?",
        "Were any new structures added later?",
        "Has the amount of green space changed?",
        "Did natural areas become developed?",
        "Which parts of the region changed the most?",
    ]

    for query in queries:

        run_test(
            query=query,
            images=[
                ImageInput(
                    path="before.tif"
                ),
                ImageInput(
                    path="after.tif"
                )
            ],
            expected_task="change_analysis"
        )


def test_unseen_multimodal():

    queries = [
        "Analyze the area using both sensor observations.",
        "Combine the available imagery to understand this region.",
        "What additional information can the two images provide?",
        "Interpret this location using multiple sensor perspectives.",
        "Use the different satellite observations together.",
    ]

    for query in queries:

        run_test(
            query=query,
            images=[
                ImageInput(
                    path="optical.tif",
                    modality="optical"
                ),
                ImageInput(
                    path="sar.tif",
                    modality="sar"
                )
            ],
            expected_task="multimodal"
        )