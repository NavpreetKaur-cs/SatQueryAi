from importlib import import_module
__all__ = ["process", "AgentRequest", "AgentResult"]
def __getattr__(name):
    if name == "process":
        return import_module("agent.controller").process
    if name == "AgentRequest":
        return import_module("agent.request").AgentRequest
    if name == "AgentResult":
        return import_module("agent.response").AgentResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")