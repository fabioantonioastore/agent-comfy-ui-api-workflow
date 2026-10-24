from typing import Any, Literal

from comfy_ui_api_workflow import Workflow
from comfy_ui_api_workflow.nodes.audio import SaveAudioAdvanced
from comfy_ui_api_workflow.nodes.image import SaveImage
from comfy_ui_api_workflow.nodes.video import SaveVideo
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..errors import ToolInputError
from ..pipelines import Link, is_link
from ..session import WorkflowSession, quote_names
from .common import LINK_SYNTAX, make_tool


class UpdateNodeArguments(BaseModel):
    node_id: str = Field(description='Id of the node to change, like "KSampler_4".')
    inputs: dict[str, Any] = Field(
        description=(
            f"Input name to new value, or to {LINK_SYNTAX} to rewire it. "
            "null removes an optional input."
        )
    )


class RemoveNodeArguments(BaseModel):
    node_id: str = Field(description="Id of the node to remove.")
    reconnect: bool = Field(
        True,
        description=(
            "Reconnect the nodes that used it to its own input of the same type, so a "
            "step in the middle of a chain can go without breaking the chain."
        ),
    )


class SaveOutputArguments(BaseModel):
    source: str = Field(description=f"The output to save: {LINK_SYNTAX}.")
    filename_prefix: str = Field("output", description="Prefix of the saved file name.")
    format: Literal["default", "mp4", "webm", "mp3", "flac", "opus"] = Field(
        "default",
        description=(
            "For a VIDEO: mp4 (default) or webm. For an AUDIO: mp3 (default), flac or "
            "opus. Images are saved as PNG."
        ),
    )


class NoArguments(BaseModel):
    pass


def _same_type(output_type: str | None, wanted: str | None) -> bool:
    if output_type is None or wanted is None:
        return False
    outputs = {part.strip() for part in output_type.split(",")}
    inputs = {part.strip() for part in wanted.split(",")}
    return "*" in outputs | inputs or bool(outputs & inputs)


def _rewires(session: WorkflowSession, node_id: str, dependents: list[str]) -> list[tuple[str, str, Link]]:
    sources = [
        (value[0], value[1])
        for value in session.workflow.nodes[node_id].inputs().values()
        if is_link(value)
    ]
    rewires: list[tuple[str, str, Link]] = []
    for dependent in dependents:
        for name, value in session.workflow.nodes[dependent].inputs().items():
            if not is_link(value) or value[0] != node_id:
                continue
            wanted = session.link_type((value[0], value[1]))
            match = next(
                (source for source in sources if _same_type(session.link_type(source), wanted)),
                None,
            )
            if match is None:
                raise ToolInputError(
                    f"Cannot reconnect {dependent}.{name}: {node_id} has no {wanted} input to "
                    f"pass through. Remove or change {dependent} first."
                )
            rewires.append((dependent, name, match))
    return rewires


def editing_tools(session: WorkflowSession) -> list[StructuredTool]:
    async def update_node(node_id: str, inputs: dict[str, Any]) -> str:
        if not inputs:
            raise ToolInputError("Give at least one input to change.")
        async with session.lock:
            node = session.node(node_id)
            node_type = type(node)
            values: dict[str, Any] = {}
            removed: list[str] = []
            for key, value in inputs.items():
                spec = session.catalog.input_spec(node_type, key)
                if spec is None:
                    names = ", ".join(spec.name for spec in session.catalog.inputs(node_type))
                    raise ToolInputError(f"{node_type.__name__} has no input {key!r}. Its inputs are: {names}.")
                if value is None:
                    removed.append(spec.name)
                else:
                    values[spec.name] = session.input_value(node_type, spec.name, value)
            if values:
                session.workflow.update_node(node_id, **values)
            for name in removed:
                if hasattr(node, name):
                    delattr(node, name)
            session.mark(node_id)
            parts = [f"{name}={session.render(value)}" for name, value in values.items()]
        parts += [f"{name} removed" for name in removed]
        return f"Updated {node_id}: {', '.join(parts)}."

    async def remove_node(node_id: str, reconnect: bool = True) -> str:
        async with session.lock:
            session.node(node_id)
            order = list(session.workflow.nodes)
            dependents = sorted(session.workflow.dependents(node_id), key=order.index)
            if dependents and not reconnect:
                raise ToolInputError(
                    f"Cannot remove {node_id}: {quote_names(dependents)} use its outputs. "
                    "Remove or rewire them first, or set reconnect to true."
                )
            rewires = _rewires(session, node_id, dependents)
            for dependent, name, link in rewires:
                session.workflow.update_node(dependent, **{name: link})
            session.workflow.remove_node(node_id)
            session.forget(node_id)
            session.mark(*{dependent for dependent, _, _ in rewires})
            text = f"Removed {node_id}."
            if rewires:
                text += " Reconnected " + ", ".join(
                    f"{dependent}.{name} to {session.link_text(link)}" for dependent, name, link in rewires
                )
                text += "."
        return text

    async def save_output(source: str, filename_prefix: str = "output", format: str = "default") -> str:
        async with session.lock:
            link = session.parse_link(source)
            if link is None:
                raise ToolInputError(
                    f"{source!r} is not an output of the workflow. Write it as "
                    '"<node_id>.<output name>"; view_workflow lists the outputs.'
                )
            kind = session.link_type(link)
            if kind == "IMAGE":
                node = session.workflow.add_node(SaveImage, images=link, filename_prefix=filename_prefix)
            elif kind == "VIDEO":
                container = "webm" if format == "webm" else "mp4"
                node = session.workflow.add_node(
                    SaveVideo,
                    video=link,
                    filename_prefix=filename_prefix,
                    format=container,
                    format_codec="av1" if container == "webm" else "h264",
                )
            elif kind == "AUDIO":
                node = session.workflow.add_node(
                    SaveAudioAdvanced,
                    audio=link,
                    filename_prefix=filename_prefix,
                    format=format if format in ("mp3", "flac", "opus") else "mp3",
                )
            else:
                raise ToolInputError(
                    f"{source} is {kind}, which is not a file. Save an IMAGE, VIDEO or AUDIO "
                    "output; decode a LATENT with VAEDecode first."
                )
            session.mark(node.node_id)
        return f"Added {node.node_id}, which saves {source}."

    async def view_workflow() -> str:
        return session.view()

    async def validate_workflow() -> str:
        return session.validate().text()

    async def new_workflow() -> str:
        async with session.lock:
            count = len(session.workflow.nodes)
            session.previous.append(session.snapshot())
            session.workflow = Workflow()
            session.agent_nodes.clear()
            session.changed = True
        return f"Started an empty workflow. The previous one ({count} nodes) was set aside."

    return [
        make_tool(
            update_node,
            name="update_node",
            description=(
                "Changes inputs of a node already in the workflow: new values (prompt, "
                "size, steps, seed...) or links to rewire it. The new values go in "
                "together: if one is rejected, none is applied. Keeps the node's id."
            ),
            args_schema=UpdateNodeArguments,
        ),
        make_tool(
            remove_node,
            name="remove_node",
            description=(
                "Removes a node. By default the nodes that used it are reconnected to its "
                "input of the same type, so removing a step in the middle of a chain keeps "
                "the chain working."
            ),
            args_schema=RemoveNodeArguments,
        ),
        make_tool(
            save_output,
            name="save_output",
            description=(
                "Adds the node that saves an IMAGE, VIDEO or AUDIO output as a file "
                "(SaveImage, SaveVideo or SaveAudioAdvanced). Pipelines already save "
                "their result unless you set save to false."
            ),
            args_schema=SaveOutputArguments,
        ),
        make_tool(
            view_workflow,
            name="view_workflow",
            description=(
                "Shows every node with its inputs and where its outputs go. Call it before "
                "editing a workflow you did not build in this conversation."
            ),
            args_schema=NoArguments,
        ),
        make_tool(
            validate_workflow,
            name="validate_workflow",
            description=(
                "Checks that the workflow can run on the server: an output node, every node "
                "connected, model files the server has, files the user attached. The "
                "application runs the same check when you finish."
            ),
            args_schema=NoArguments,
        ),
        make_tool(
            new_workflow,
            name="new_workflow",
            description=(
                "Sets the current workflow aside and starts an empty one. Use it when the "
                "user asks for a separate workflow, or when the new request is a different "
                "task that shares nothing with the current workflow; say so in your answer. "
                "The application keeps the workflows set aside."
            ),
            args_schema=NoArguments,
        ),
    ]
