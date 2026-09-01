from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    """Standard response returned by SatQuery AI agent."""

    success: bool
    answer: str
    task: str
    confidence: Optional[float] = None
    model: Optional[str] = None
    visual_output: Optional[Any] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "answer": self.answer,
            "task": self.task,
            "confidence": self.confidence,
            "model": self.model,
            "visual_output": self.visual_output,
            "trace": self.trace,
            "error": self.error,
            "metadata": self.metadata,
        }