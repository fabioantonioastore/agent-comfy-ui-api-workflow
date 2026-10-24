from collections.abc import Iterable

from langchain_core.tools import BaseTool

from ..pipelines import AgentPipeline
from ..session import WorkflowSession
from .editing import editing_tools
from .nodes import node_tools
from .pipelines import pipeline_tool
from .questions import ask_user_tool

__all__ = ["build_tools"]


def build_tools(
    session: WorkflowSession, pipelines: Iterable[type[AgentPipeline]]
) -> list[BaseTool]:
    """The agent's tools, pipelines first so the model sees them before single nodes."""
    pipelines = tuple(pipelines)
    return [
        *(pipeline_tool(session, pipeline) for pipeline in pipelines),
        *node_tools(session, pipelines),
        *editing_tools(session),
        ask_user_tool(session),
    ]
