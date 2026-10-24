from collections.abc import Iterable
from typing import Any

from comfy_ui_api_workflow.nodes.base import BaseNode
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..errors import ToolInputError
from ..pipelines import AgentPipeline
from ..session import WorkflowSession, quote_names
from .common import LINK_SYNTAX, make_tool


class AddNodeArguments(BaseModel):
    node_type: str = Field(description='Class name from the essential node list, like "ImageScaleBy".')
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Input name to value: a number, text, true/false, one of the choices, the "
            f"name of a file the user attached, or {LINK_SYNTAX}."
        ),
    )


class DescribeNodeArguments(BaseModel):
    node_type: str = Field(description='Class name of the node, like "KSampler".')


def essential(session: WorkflowSession, name: str) -> type[BaseNode]:
    node_type = session.catalog.essential_type(name)
    if node_type is not None:
        return node_type
    known = session.catalog.node_type(name)
    if known is not None:
        raise ToolInputError(
            f"{known.__name__} is not one of the essential nodes you can add. Use a "
            "pipeline, or a node from the essential node list."
        )
    raise ToolInputError(f"There is no node {name!r}. Use a class name from the essential node list.")


def node_tools(
    session: WorkflowSession, pipelines: Iterable[type[AgentPipeline]]
) -> list[StructuredTool]:
    covering: dict[str, list[str]] = {}
    for pipeline in pipelines:
        for class_name in pipeline.COVERS:
            covering.setdefault(class_name, []).append(pipeline.NAME)

    async def add_node(node_type: str, inputs: dict[str, Any] | None = None) -> str:
        chosen = essential(session, node_type)
        async with session.lock:
            arguments = session.node_arguments(chosen, inputs or {})
            session.require(chosen, arguments)
            node = session.workflow.add_node(chosen, **arguments)
            session.mark(node.node_id)
            outputs = session.outputs_text(node.node_id)
        text = f"Added {node.node_id}. " + (f"Outputs: {outputs}." if outputs else "It has no outputs.")
        pipelines_for_it = covering.get(chosen.__name__)
        if pipelines_for_it:
            text += (
                f" Note: the pipeline {quote_names(pipelines_for_it)} already builds this step; "
                "prefer it unless you need a variant it does not offer."
            )
        return text

    async def describe_node(node_type: str) -> str:
        found = session.catalog.node_type(node_type)
        in_workflow = found is not None and any(
            type(node) is found for node in session.workflow.nodes.values()
        )
        if found is None or not (session.catalog.is_essential(found) or in_workflow):
            essential(session, node_type)
        return session.catalog.describe(found)

    return [
        make_tool(
            add_node,
            name="add_node",
            description=(
                "Adds one node from the essential node list to the workflow and returns "
                "its id and outputs. Use it only for a step no pipeline does. Give every "
                "required input; call describe_node first when you have not seen the "
                "node's inputs in this conversation."
            ),
            args_schema=AddNodeArguments,
        ),
        make_tool(
            describe_node,
            name="describe_node",
            description=(
                "Shows a node's inputs (type, range, choices, default), its outputs and "
                "its documentation. Call it before add_node or update_node on a node "
                "whose inputs you have not seen in this conversation."
            ),
            args_schema=DescribeNodeArguments,
        ),
    ]
