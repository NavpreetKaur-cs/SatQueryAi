from typing import List


def create_plan(task: str, query: str) -> List[str]:
    if not query or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    if task == "single_image":
        return [
            "input_validation",
            "single_image_analysis",
            "result_integration",
        ]

    if task == "change_analysis":
        return [
            "input_validation",
            "change_detection",
            "change_description",
            "result_integration",
        ]

    if task == "multimodal":
        return [
            "input_validation",
            "optical_analysis",
            "sar_analysis",
            "cross_modal_fusion",
            "result_integration",
        ]

    raise ValueError(f"Unknown task: {task}")