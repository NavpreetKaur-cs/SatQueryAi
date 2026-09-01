from typing import Any, Dict

from agent.tools import get_tool


def execute(
    task: str,
    inputs: Dict[str, Any]
) -> Dict[str, Any]:

    tool = get_tool(task)

    handler = tool["handler"]

    result = handler(
        **inputs
    )

    if not isinstance(
        result,
        dict
    ):

        raise TypeError(
            "Tool must return a dictionary."
        )

    result.setdefault(
        "success",
        False
    )

    result.setdefault(
        "answer",
        ""
    )

    result.setdefault(
        "confidence",
        None
    )

    result.setdefault(
        "model",
        None
    )

    result.setdefault(
        "visual_output",
        None
    )

    result.setdefault(
        "metadata",
        {}
    )

    result["tool"] = tool["name"]

    return result