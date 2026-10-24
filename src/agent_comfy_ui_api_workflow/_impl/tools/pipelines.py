import inspect
from collections.abc import Iterable
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from comfy_ui_api_workflow.errors.pipeline import PipelineError
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from ..errors import ToolInputError
from ..families import ModelChoice
from ..pipelines import AgentPipeline
from ..session import WorkflowSession
from .common import make_tool


def _field_type(annotation: Any) -> Any:
    # The model writes links as text, so every link in an annotation becomes a str.
    if get_origin(annotation) in (Union, UnionType):
        options = get_args(annotation)
    else:
        options = (annotation,)
    mapped: list[Any] = []
    for option in options:
        if get_origin(option) is tuple:
            option = str
        elif get_origin(option) is list:
            option = list[_field_type(get_args(option)[0])]
        if option not in mapped:
            mapped.append(option)
    return mapped[0] if len(mapped) == 1 else Union[tuple(mapped)]


def pipeline_schema(
    pipeline: type[AgentPipeline], models: Iterable[ModelChoice] = ()
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for parameter in pipeline.parameters():
        description = pipeline.PARAMETERS.get(parameter.name, "")
        if parameter.annotation is ModelChoice:
            # The model writes a model id; only the models that can do the job are offered.
            usable = [model for model in models if pipeline.CAPABILITY in model.capabilities]
            if not usable:
                raise ValueError(f"no model on the server can do {pipeline.NAME}")
            ids = tuple(model.id for model in usable)
            options = " ".join(f"{model.id}: {model.description}" for model in usable)
            fields[parameter.name] = (
                Literal[ids],
                Field(ids[0], description=f"{description} Options: {options}"),
            )
            continue
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (_field_type(parameter.annotation), Field(default, description=description))
    return create_model(f"{pipeline.__name__}Arguments", **fields)


def pipeline_tool(session: WorkflowSession, pipeline: type[AgentPipeline]) -> StructuredTool:
    schema = pipeline_schema(pipeline, session.models.values())
    model_name = pipeline.model_parameter()
    default_model = None if model_name is None else schema.model_fields[model_name].default

    async def use(**values: Any) -> str:
        if model_name is not None:
            # LangChain passes only the arguments the model gave, not the defaults.
            values.setdefault(model_name, default_model)
        async with session.lock:
            arguments = session.pipeline_arguments(pipeline, values)
            before = set(session.workflow.nodes)
            try:
                added = session.workflow.add_pipeline(pipeline, **arguments)
            except PipelineError as error:
                # The lib names the class; the model knows the pipeline by its tool name.
                message = str(error).replace(repr(pipeline.__name__), repr(pipeline.NAME), 1)
                raise ToolInputError(message) from error
            new_ids = [node_id for node_id in session.workflow.nodes if node_id not in before]
            session.mark(*new_ids)
            text = f"Added {pipeline.NAME}: {', '.join(new_ids)}."
            outputs = added.output.outputs
            if outputs:
                text += " Outputs: " + ", ".join(
                    f"{name} = {session.link_text(link)} ({session.link_type(link)})"
                    for name, link in outputs.items()
                )
                text += "."
            return text

    return make_tool(
        use,
        name=pipeline.NAME,
        description=f"{pipeline.DESCRIPTION} Prefer this pipeline over adding its nodes one by one.",
        args_schema=schema,
    )
