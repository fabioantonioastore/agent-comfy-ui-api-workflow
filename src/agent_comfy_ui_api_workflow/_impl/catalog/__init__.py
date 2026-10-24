import functools
import importlib
import inspect
import json
import pkgutil
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from comfy_ui_api_workflow import nodes
from comfy_ui_api_workflow.nodes.base import BaseNode

from .essential import ESSENTIAL_GROUPS

__all__ = ["ESSENTIAL_GROUPS", "InputSpec", "NodeCatalog", "get_catalog", "node_types"]

_NUMBERED = re.compile(r"^(.*?\D)(\d+)$")
_MAX_CHOICES = 60
_SUMMARY_LENGTH = 160


@functools.cache
def node_types() -> dict[str, type[BaseNode]]:
    """Every node class of comfy_ui_api_workflow, by ComfyUI class type."""
    prefix = f"{nodes.__name__}."
    for module in pkgutil.walk_packages(nodes.__path__, prefix):
        importlib.import_module(module.name)
    found: dict[str, type[BaseNode]] = {}
    pending = list(BaseNode.__subclasses__())
    while pending:
        node_type = pending.pop()
        pending.extend(node_type.__subclasses__())
        if node_type.__module__.startswith(prefix):
            found[node_type.CLASS_TYPE] = node_type
    return found


@functools.cache
def _lookup() -> dict[str, type[BaseNode]]:
    names: dict[str, type[BaseNode]] = {}
    for class_type, node_type in node_types().items():
        for name in (class_type, node_type.__name__):
            names[name] = node_type
            names.setdefault(name.lower(), node_type)
    return names


@functools.cache
def object_info() -> dict[str, Any]:
    data = resources.files(__package__).joinpath("new_object_info.json")
    return json.loads(data.read_text(encoding="utf-8"))


def _options(annotation: Any) -> tuple[Any, ...]:
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def _spec_meta(spec: Any) -> dict[str, Any]:
    return spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}


def _section_inputs(sections: Mapping[str, Any]) -> dict[str, Any]:
    return {**(sections.get("required") or {}), **(sections.get("optional") or {})}


def _input_meta(info: Mapping[str, Any], comfy_name: str) -> dict[str, Any]:
    # Dynamic inputs are sent as "<combo>.<input>"; their metadata sits inside the
    # option or the template that opens them.
    parts = comfy_name.split(".")
    spec = _section_inputs(info.get("input") or {}).get(parts[0])
    for part in parts[1:]:
        if spec is None:
            return {}
        kind, meta = spec[0], _spec_meta(spec)
        if kind == "COMFY_DYNAMICCOMBO_V3":
            spec = next(
                (
                    inputs[part]
                    for option in meta.get("options", [])
                    if part in (inputs := _section_inputs(option.get("inputs") or {}))
                ),
                None,
            )
        elif kind == "COMFY_AUTOGROW_V3":
            template = meta.get("template") or {}
            inputs = _section_inputs(template.get("input") or {})
            spec = inputs.get(template.get("prefix", "")) or next(iter(inputs.values()), None)
        else:
            return {}
    meta = _spec_meta(spec) if spec is not None else {}
    return {key: meta[key] for key in ("tooltip", "min", "max", "default") if key in meta}


@dataclass(frozen=True)
class InputSpec:
    name: str
    comfy_name: str
    socket: str | None
    annotation: Any
    required: bool
    default: Any
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def link_only(self) -> bool:
        options = [option for option in _options(self.annotation) if option is not NoneType]
        return bool(options) and all(get_origin(option) is tuple for option in options)

    @property
    def accepts_link(self) -> bool:
        return self.annotation is Any or any(
            get_origin(option) is tuple for option in _options(self.annotation)
        )

    @property
    def choices(self) -> tuple[Any, ...]:
        return tuple(
            choice
            for option in _options(self.annotation)
            if get_origin(option) is Literal
            for choice in get_args(option)
        )

    @property
    def has_default(self) -> bool:
        return self.default is not inspect.Parameter.empty and self.default is not None

    def value_text(self) -> str:
        if self.link_only:
            return f"link to {self.socket or 'an output'}"
        if self.choices:
            shown = ", ".join(map(str, self.choices[:_MAX_CHOICES]))
            if len(self.choices) > _MAX_CHOICES:
                shown += f", … ({len(self.choices)} options)"
            return f"one of: {shown}"
        kinds = [option for option in _options(self.annotation) if option in (bool, int, float, str)]
        if bool in kinds:
            return "true or false"
        if int in kinds or float in kinds:
            word = "number" if float in kinds else "int"
            low, high = self.meta.get("min"), self.meta.get("max")
            if low is not None and high is not None:
                return f"{word} from {low} to {high}"
            if low is not None:
                return f"{word}, at least {low}"
            if high is not None:
                return f"{word}, at most {high}"
            return word
        if str in kinds:
            return "text"
        if self.socket:
            return f"link to {self.socket} or a value"
        return "any value"

    def line(self) -> str:
        parts = [self.value_text()]
        if self.required:
            parts.append("required")
        elif self.has_default:
            parts.append(f"default {self.default!r}")
        else:
            parts.append("optional")
        tooltip = str(self.meta.get("tooltip") or "").strip()
        return f"- {self.name}: {', '.join(parts)}." + (f" {tooltip}" if tooltip else "")


@functools.cache
def _input_specs(node_type: type[BaseNode]) -> tuple[InputSpec, ...]:
    info = object_info().get(node_type.CLASS_TYPE, {})
    specs: list[InputSpec] = []
    for parameter in inspect.signature(node_type).parameters.values():
        if parameter.name == "node_id" or parameter.kind is parameter.VAR_KEYWORD:
            continue
        comfy_name = node_type.INPUT_NAMES.get(parameter.name, parameter.name)
        specs.append(
            InputSpec(
                name=parameter.name,
                comfy_name=comfy_name,
                socket=node_type.INPUT_SOCKET_TYPES.get(parameter.name),
                annotation=parameter.annotation,
                required=parameter.default is inspect.Parameter.empty,
                default=parameter.default,
                meta=_input_meta(info, comfy_name),
            )
        )
    return tuple(specs)


def _numbered_runs(specs: Iterable[InputSpec]) -> list[list[InputSpec]]:
    # ConcatenateVideo has videos_video0 … videos_video99: one line says it better.
    runs: list[list[InputSpec]] = []
    for spec in specs:
        match = _NUMBERED.match(spec.name)
        if runs and match:
            previous = _NUMBERED.match(runs[-1][-1].name)
            if (
                previous
                and previous.group(1) == match.group(1)
                and int(previous.group(2)) + 1 == int(match.group(2))
                and runs[-1][-1].socket == spec.socket
            ):
                runs[-1].append(spec)
                continue
        runs.append([spec])
    return runs


def _run_line(run: list[InputSpec]) -> str:
    first, last = run[0], run[-1]
    required = sum(spec.required for spec in run)
    needed = f"the first {required} required, the rest optional" if required else "all optional"
    tooltip = str(first.meta.get("tooltip") or "").strip()
    line = f"- {first.name} … {last.name}: {first.value_text()}; {needed}, filled in order."
    return line + (f" {tooltip}" if tooltip else "")


def _summary(text: str) -> str:
    first = text.strip().split("\n")[0].strip()
    # A description that opens a list ("Recipes:") says nothing on its own line.
    if first.endswith(":"):
        return ""
    end = first.find(". ")
    if end != -1:
        first = first[: end + 1]
    if len(first) > _SUMMARY_LENGTH:
        first = first[: _SUMMARY_LENGTH - 1].rstrip() + "…"
    return first


class NodeCatalog:
    """The essential nodes the agent may add, with their documentation."""

    def __init__(
        self,
        *,
        extra: Mapping[str, Iterable[str]] | None = None,
        hidden: Iterable[str] = (),
    ) -> None:
        hidden = set(hidden)
        groups: dict[str, tuple[type[BaseNode], ...]] = {}
        for group, names in {**ESSENTIAL_GROUPS, **(extra or {})}.items():
            kept: list[type[BaseNode]] = []
            for name in names:
                node_type = self.node_type(name)
                if node_type is None:
                    raise ValueError(f"{name!r} is not a node of comfy_ui_api_workflow")
                if node_type.__name__ not in hidden and node_type.CLASS_TYPE not in hidden:
                    kept.append(node_type)
            if kept:
                groups[group] = tuple(kept)
        self.groups = groups
        self.__essential = frozenset(node_type for kept in groups.values() for node_type in kept)

    @staticmethod
    def node_type(name: str) -> type[BaseNode] | None:
        names = _lookup()
        return names.get(name) or names.get(name.lower())

    def essential_type(self, name: str) -> type[BaseNode] | None:
        node_type = self.node_type(name)
        return node_type if node_type in self.__essential else None

    def is_essential(self, node_type: type[BaseNode]) -> bool:
        return node_type in self.__essential

    @staticmethod
    def info(node_type: type[BaseNode]) -> dict[str, Any]:
        return object_info().get(node_type.CLASS_TYPE, {})

    @staticmethod
    def inputs(node_type: type[BaseNode]) -> tuple[InputSpec, ...]:
        return _input_specs(node_type)

    def input_spec(self, node_type: type[BaseNode], name: str) -> InputSpec | None:
        return next(
            (spec for spec in self.inputs(node_type) if name in (spec.name, spec.comfy_name)),
            None,
        )

    @staticmethod
    def is_output_node(class_type: str) -> bool:
        return bool(object_info().get(class_type, {}).get("output_node"))

    def output_type(self, name: str, index: int) -> str | None:
        node_type = self.node_type(name)
        if node_type is None or not 0 <= index < len(node_type.RETURN_TYPES):
            return None
        return node_type.RETURN_TYPES[index]

    def describe(self, node_type: type[BaseNode]) -> str:
        info = self.info(node_type)
        title = node_type.__name__
        if node_type.CLASS_TYPE != node_type.__name__:
            title += f" (ComfyUI class_type {node_type.CLASS_TYPE!r})"
        about = str(info.get("description") or info.get("display_name") or "").strip()
        header = f"{title}: {about}" if about else title
        if info.get("category"):
            header += f" (category {info['category']})"
        lines = [header]
        specs = self.inputs(node_type)
        if specs:
            lines.append(
                "Inputs (numbers, text and choices also accept a link to an output of "
                "the same type):"
            )
            for run in _numbered_runs(specs):
                if len(run) >= 3:
                    lines.append(_run_line(run))
                else:
                    lines.extend(spec.line() for spec in run)
        else:
            lines.append("Inputs: none.")
        tooltips = info.get("output_tooltips") or []
        if node_type.OUTPUTS:
            lines.append('Outputs (link to one as "<node_id>.<output name>"):')
            for name, index in sorted(node_type.OUTPUTS.items(), key=lambda item: item[1]):
                kind = node_type.RETURN_TYPES[index] if index < len(node_type.RETURN_TYPES) else "?"
                tip = tooltips[index] if index < len(tooltips) and tooltips[index] else ""
                lines.append(f"- {name} ({kind})" + (f": {tip}" if tip else ""))
        else:
            lines.append("Outputs: none.")
        return "\n".join(lines)

    def summary_line(self, node_type: type[BaseNode]) -> str:
        info = self.info(node_type)
        required = ", ".join(spec.name for spec in self.inputs(node_type) if spec.required)
        outputs = ", ".join(sorted(node_type.OUTPUTS, key=node_type.OUTPUTS.__getitem__))
        summary = _summary(info.get("description") or "") or info.get("display_name") or ""
        line = f"- {node_type.__name__}({required}) → {outputs or 'no outputs'}"
        return f"{line}: {summary}" if summary else line

    def index(self) -> str:
        lines: list[str] = []
        for group, kept in self.groups.items():
            lines.append(f"{group}:")
            lines.extend(self.summary_line(node_type) for node_type in kept)
            lines.append("")
        return "\n".join(lines).strip()


@functools.cache
def get_catalog() -> NodeCatalog:
    return NodeCatalog()
