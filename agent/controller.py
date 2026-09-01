from typing import Dict, Any

from agent.request import AgentRequest
from agent.response import AgentResult

from agent.router import route_query
from agent.planner import create_plan
from agent.executor import execute


def _prepare_inputs(
    task: str,
    request: AgentRequest
) -> Dict[str, Any]:

    images = request.images

    metadata = (
        request.metadata
        or {}
    )

    if task == "single_image":

        return {
            "image": images[0],
            "query": request.query,
            "metadata": metadata,
        }

    if task == "change_analysis":

        return {
            "image1": images[0],
            "image2": images[1],
            "query": request.query,
            "metadata": metadata,
        }

    if task == "multimodal":

        optical_image = None
        sar_image = None

        # ------------------------------------------------
        # Identify images using modality metadata
        # ------------------------------------------------

        for image in images:

            modality = (
                image.modality or ""
            ).lower()

            if modality in [
                "optical",
                "multispectral"
            ]:

                optical_image = image

            elif modality in [
                "sar",
                "radar"
            ]:

                sar_image = image

        # ------------------------------------------------
        # If modality metadata is unavailable,
        # preserve the conventional ordering.
        # ------------------------------------------------

        if (
            optical_image is None
            or sar_image is None
        ):

            optical_image = images[0]
            sar_image = images[1]

        return {
            "optical_image": optical_image,
            "sar_image": sar_image,
            "query": request.query,
            "metadata": metadata,
        }

    raise ValueError(
        f"Unsupported task: {task}"
    )


def process(
    request: AgentRequest
) -> AgentResult:

    trace = []

    try:

        # ==================================================
        # 1. ROUTING
        # ==================================================

        decision = route_query(
            query=request.query,
            images=request.images,
            metadata=request.metadata
        )

        trace.append({
            "step": "routing",

            "task": decision.task,

            "confidence": (
                decision.confidence
            ),

            "method": decision.method,

            "status": "success",

            "details": decision.details,
        })

        # ==================================================
        # 2. PLANNING
        # ==================================================

        plan = create_plan(
            decision.task,
            request.query
        )

        trace.append({
            "step": "planning",

            "plan": plan,

            "status": "success",
        })

        # ==================================================
        # 3. INPUT PREPARATION
        # ==================================================

        inputs = _prepare_inputs(
            decision.task,
            request
        )

        trace.append({
            "step": "input_preparation",

            "status": "success",

            "image_count": len(
                request.images
            ),
        })

        # ==================================================
        # 4. TOOL SELECTION
        # ==================================================

        from agent.tools import get_tool

        tool = get_tool(
            decision.task
        )

        trace.append({
            "step": "tool_selection",

            "tool": tool["name"],

            "description": tool[
                "description"
            ],

            "status": "success",
        })

        # ==================================================
        # 5. EXECUTION
        # ==================================================

        result = execute(
            decision.task,
            inputs
        )

        trace.append({
            "step": "execution",

            "task": decision.task,

            "tool": result.get(
                "tool"
            ),

            "model": result.get(
                "model"
            ),

            "status": (
                "success"
                if result.get(
                    "success",
                    False
                )
                else "failed"
            ),
        })

        # ==================================================
        # 6. RESULT INTEGRATION
        # ==================================================

        trace.append({
            "step": "result_integration",

            "status": "success",
        })

        return AgentResult(

            success=result.get(
                "success",
                False
            ),

            answer=result.get(
                "answer",
                result.get(
                    "result",
                    ""
                )
            ),

            task=decision.task,

            confidence=result.get(
                "confidence"
            ),

            model=result.get(
                "model"
            ),

            visual_output=result.get(
                "visual_output"
            ),

            trace=trace,

            error=None,

            metadata=result.get(
                "metadata",
                {}
            ),
        )

    except Exception as e:

        trace.append({
            "step": "error",

            "status": "failed",

            "message": str(e),
        })

        return AgentResult(

            success=False,

            answer="",

            task="unknown",

            confidence=None,

            model=None,

            visual_output=None,

            trace=trace,

            error=str(e),

            metadata={},
        )