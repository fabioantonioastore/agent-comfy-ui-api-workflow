import asyncio
import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from comfy_ui_api_workflow import Workflow
from comfy_ui_api_workflow.nodes.base import BaseNode

from .catalog import NodeCatalog
from .errors import ToolInputError
from .families import ModelChoice
from .files import FILE_INPUTS, InputFiles
from .inventory import ServerInventory, model_folder
from .pipelines import AgentPipeline, Link, is_link
from .validation import ValidationReport, validate_workflow

_MISSING = object()
# Prompts must stay whole for the agent to edit them; longer text is cut in views.
_TEXT_LIMIT = 1000


def quote_names(names: list[str]) -> str:
    quoted = [repr(name) for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _article(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


@dataclass(frozen=True)
class Question:
    text: str
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowDiff:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def text(self) -> str:
        if not self:
            return "No changes."
        parts: list[str] = []
        if self.added:
            parts.append(f"Added: {', '.join(self.added)}.")
        if self.removed:
            parts.append(f"Removed: {', '.join(self.removed)}.")
        if self.changed:
            changed = ", ".join(f"{node_id} ({', '.join(names)})" for node_id, names in self.changed.items())
            parts.append(f"Changed: {changed}.")
        return " ".join(parts)


def diff_workflows(before: Mapping[str, Any], after: Mapping[str, Any]) -> WorkflowDiff:
    changed: dict[str, tuple[str, ...]] = {}
    for node_id, node in after.items():
        old = before.get(node_id)
        if old is None:
            continue
        if old["class_type"] != node["class_type"]:
            changed[node_id] = ("class_type",)
            continue
        names = sorted({*old["inputs"], *node["inputs"]})
        differ = tuple(
            name
            for name in names
            if old["inputs"].get(name, _MISSING) != node["inputs"].get(name, _MISSING)
        )
        if differ:
            changed[node_id] = differ
    return WorkflowDiff(
        added=tuple(node_id for node_id in after if node_id not in before),
        removed=tuple(node_id for node_id in before if node_id not in after),
        changed=changed,
    )


def api_json(workflow: Workflow) -> dict[str, Any]:
    # ComfyUI receives JSON, where a link is a list; the copy shares nothing with the nodes.
    return json.loads(json.dumps(workflow.get_comfy_ui_workflow_format()))


class WorkflowSession:
    """The workflow the agent builds, shared by its tools, and what the turn did to it."""

    def __init__(
        self,
        catalog: NodeCatalog,
        inventory: ServerInventory,
        files: InputFiles | None = None,
        workflow: Workflow | None = None,
        models: Iterable[ModelChoice] = (),
    ) -> None:
        self.catalog = catalog
        self.inventory = inventory
        self.models: dict[str, ModelChoice] = {model.id: model for model in models}
        self.files = InputFiles() if files is None else files
        self.workflow = Workflow() if workflow is None else workflow
        self.lock = asyncio.Lock()
        # Nodes the agent added or changed: validation is strict only about them.
        self.agent_nodes: set[str] = set()
        self.previous: list[dict[str, Any]] = []
        self.question: Question | None = None
        self.changed = False
        self.nudges = 0

    @classmethod
    def from_api(
        cls,
        data: Mapping[str, Any],
        catalog: NodeCatalog,
        inventory: ServerInventory,
        files: InputFiles | None = None,
        models: Iterable[ModelChoice] = (),
    ) -> "WorkflowSession":
        workflow = Workflow()
        workflow.load_workflow(copy.deepcopy(dict(data)))
        return cls(catalog, inventory, files, workflow, models)

    def start_turn(self) -> None:
        self.question = None
        self.changed = False
        self.nudges = 0

    def snapshot(self) -> dict[str, Any]:
        return api_json(self.workflow)

    def mark(self, *node_ids: str) -> None:
        self.agent_nodes.update(node_ids)
        self.changed = True

    def forget(self, node_id: str) -> None:
        self.agent_nodes.discard(node_id)
        self.changed = True

    def validate(self) -> ValidationReport:
        return validate_workflow(
            self.snapshot(),
            catalog=self.catalog,
            inventory=self.inventory,
            files=self.files,
            strict_nodes=self.agent_nodes,
        )

    def node(self, node_id: str) -> BaseNode:
        node = self.workflow.nodes.get(node_id)
        if node is None:
            ids = ", ".join(self.workflow.nodes) or "none, the workflow is empty"
            raise ToolInputError(f"There is no node {node_id!r}. The nodes are: {ids}.")
        return node

    # Links are written "<node_id>.<output name>" (or "<node_id>.<output index>").
    def parse_link(self, value: Any) -> Link | None:
        if not isinstance(value, str) or "." not in value:
            return None
        node_id, _, output = value.rpartition(".")
        node = self.workflow.nodes.get(node_id)
        if node is None:
            return None
        outputs = type(node).OUTPUTS
        if output in outputs:
            return (node_id, outputs[output])
        by_lower = {name.lower(): index for name, index in outputs.items()}
        if output.lower() in by_lower:
            return (node_id, by_lower[output.lower()])
        if output.isdigit():
            # "4.5" can be a number written as text: a link only if the output exists.
            index = int(output)
            return (node_id, index) if index in outputs.values() else None
        names = ", ".join(sorted(outputs, key=outputs.__getitem__)) or "none"
        raise ToolInputError(f"Node {node_id!r} has no output {output!r}. Its outputs are: {names}.")

    def resolve(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        link = self.parse_link(value)
        return value if link is None else link

    def link_type(self, link: Link) -> str | None:
        node = self.workflow.nodes.get(link[0])
        if node is None:
            return None
        types = type(node).RETURN_TYPES
        return types[link[1]] if 0 <= link[1] < len(types) else None

    def link_text(self, link: Link) -> str:
        node = self.workflow.nodes.get(link[0])
        if node is not None:
            for name, index in type(node).OUTPUTS.items():
                if index == link[1]:
                    return f"{link[0]}.{name}"
        return f"{link[0]}.{link[1]}"

    def outputs_text(self, node_id: str) -> str:
        outputs = sorted(type(self.workflow.nodes[node_id]).OUTPUTS.items(), key=lambda item: item[1])
        return ", ".join(
            f"{node_id}.{name} ({self.link_type((node_id, index)) or '?'})" for name, index in outputs
        )

    # Values the agent sends in tool calls, checked against the server and the files.
    def attached(self, value: Any, kind: str) -> str:
        file = self.files.get(value) if isinstance(value, str) else None
        if file is None:
            ask = f"If the request needs {_article(kind)} the user has not sent, ask for it with ask_user."
            if len(self.files):
                listing = ", ".join(attached.describe() for attached in self.files)
                raise ToolInputError(
                    f"{value!r} is neither a file the user attached ({listing}) nor an output "
                    f'of the workflow like "Node_3.IMAGE". {ask}'
                )
            raise ToolInputError(
                f"{value!r} is not an output of the workflow, and no files are attached. {ask}"
            )
        if file.kind != kind:
            raise ToolInputError(
                f"{value!r} is {_article(file.kind)} file, but this input takes {_article(kind)}."
            )
        return file.name

    def media(self, value: Any, kind: str) -> Any:
        if value is None:
            return None
        link = self.parse_link(value)
        return link if link is not None else self.attached(value, kind)

    def check_model(self, folder: str, value: Any) -> None:
        if not isinstance(value, str) or not self.inventory.knows(folder):
            return
        if not self.inventory.has(folder, value):
            available = ", ".join(self.inventory.folder(folder)) or "none"
            raise ToolInputError(
                f"{value!r} is not in the server's {folder} folder. The files there are: {available}."
            )

    def input_value(self, node_type: type[BaseNode], name: str, value: Any) -> Any:
        value = self.resolve(value)
        if value is None or is_link(value):
            return value
        kind = FILE_INPUTS.get((node_type.CLASS_TYPE, name))
        if kind is not None:
            return self.attached(value, kind)
        folder = model_folder(node_type.CLASS_TYPE, name)
        if folder is not None:
            self.check_model(folder, value)
        return value

    def node_arguments(self, node_type: type[BaseNode], inputs: Mapping[str, Any]) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for key, value in inputs.items():
            spec = self.catalog.input_spec(node_type, key)
            if spec is None:
                names = ", ".join(spec.name for spec in self.catalog.inputs(node_type))
                raise ToolInputError(
                    f"{node_type.__name__} has no input {key!r}. Its inputs are: {names}. "
                    "Call describe_node for their types."
                )
            arguments[spec.name] = self.input_value(node_type, spec.name, value)
        return arguments

    def require(self, node_type: type[BaseNode], arguments: Mapping[str, Any]) -> None:
        missing = [
            spec.name
            for spec in self.catalog.inputs(node_type)
            if spec.required and spec.name not in arguments
        ]
        if missing:
            noun = "input" if len(missing) == 1 else "inputs"
            raise ToolInputError(
                f"{node_type.__name__} needs the {noun} {quote_names(missing)}. "
                "Call describe_node for their types."
            )

    def model(self, model_id: Any) -> ModelChoice:
        model = self.models.get(model_id) if isinstance(model_id, str) else None
        if model is None:
            known = ", ".join(self.models) or "none"
            raise ToolInputError(f"There is no model {model_id!r}. The models are: {known}.")
        return model

    def pipeline_arguments(
        self, pipeline: type[AgentPipeline], values: Mapping[str, Any]
    ) -> dict[str, Any]:
        model_name = pipeline.model_parameter()
        arguments: dict[str, Any] = {}
        for name, value in values.items():
            kind = pipeline.FILES.get(name)
            if name == model_name:
                arguments[name] = self.model(value)
            elif kind is None:
                arguments[name] = self.resolve(value)
            elif isinstance(value, list):
                arguments[name] = [self.media(item, kind) for item in value]
            else:
                arguments[name] = self.media(value, kind)
        return arguments

    # The workflow as the agent reads it: one line per node, then where its outputs go.
    def render(self, value: Any) -> str:
        if is_link(value):
            return self.link_text((value[0], value[1]))
        if isinstance(value, str) and len(value) > _TEXT_LIMIT:
            shown = json.dumps(value[:_TEXT_LIMIT], ensure_ascii=False)
            return f"{shown[:-1]}…\" (+{len(value) - _TEXT_LIMIT} chars)"
        return json.dumps(value, ensure_ascii=False)

    def output_node_ids(self) -> list[str]:
        return [
            node_id
            for node_id, node in self.workflow.nodes.items()
            if self.catalog.is_output_node(type(node).CLASS_TYPE)
        ]

    def view(self) -> str:
        nodes = self.workflow.nodes
        if not nodes:
            return "The workflow is empty."
        outputs = self.output_node_ids()
        reached: set[str] = set()
        pending = list(outputs)
        while pending:
            node_id = pending.pop()
            if node_id not in reached:
                reached.add(node_id)
                pending.extend(self.workflow.dependencies(node_id))
        used: dict[Link, list[str]] = {}
        for node_id, node in nodes.items():
            for name, value in node.inputs().items():
                if is_link(value):
                    used.setdefault((value[0], value[1]), []).append(f"{node_id}.{name}")

        lines = [f"Workflow with {len(nodes)} nodes. Output nodes: {', '.join(outputs) or 'none'}."]
        for node_id, node in nodes.items():
            node_type = type(node)
            arguments = ", ".join(
                f"{name}={self.render(value)}" for name, value in node.inputs().items()
            )
            line = f"{node_id} = {node_type.__name__}({arguments})"
            if node_id in outputs:
                line += "  [output node]"
            elif node_id not in reached:
                line += "  [not connected to any output node]"
            lines.append(line)
            for name, index in sorted(node_type.OUTPUTS.items(), key=lambda item: item[1]):
                targets = used.get((node_id, index))
                if targets:
                    lines.append(f"    {name} → {', '.join(targets)}")
        return "\n".join(lines)
