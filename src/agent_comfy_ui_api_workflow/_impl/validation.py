import copy
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from comfy_ui_api_workflow import Workflow
from comfy_ui_api_workflow.exceptions import ComfyUIAPIWorkflowException

from .catalog import NodeCatalog
from .files import FILE_INPUTS, InputFiles
from .inventory import ServerInventory, model_folder
from .pipelines import is_link


@dataclass(frozen=True)
class Problem:
    message: str
    node_id: str | None = None
    blocking: bool = True


@dataclass(frozen=True)
class ValidationReport:
    problems: tuple[Problem, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(problem.blocking for problem in self.problems)

    def text(self) -> str:
        if not self.problems:
            return "Valid: ready to run."
        lines = ["Valid, with warnings:" if self.ok else "Not valid yet:"]
        lines += [
            f"- {'error' if problem.blocking else 'warning'}: {problem.message}"
            for problem in self.problems
        ]
        return "\n".join(lines)


def _sources(inputs: Mapping[str, Any]) -> Iterator[str]:
    for value in inputs.values():
        if is_link(value):
            yield value[0]


def _reaching(data: Mapping[str, Any], outputs: list[str]) -> set[str]:
    reached: set[str] = set()
    pending = list(outputs)
    while pending:
        node_id = pending.pop()
        if node_id in reached or node_id not in data:
            continue
        reached.add(node_id)
        pending.extend(_sources(data[node_id]["inputs"]))
    return reached


def validate_workflow(
    data: Mapping[str, Any],
    *,
    catalog: NodeCatalog,
    inventory: ServerInventory,
    files: InputFiles,
    strict_nodes: Collection[str] | None = None,
) -> ValidationReport:
    """Checks an API-format workflow against the server and the user's files.

    Problems in strict_nodes (every node when it is None) block; problems in other
    nodes, like the untouched nodes of a workflow the user loaded, are warnings.
    """

    def strict(node_id: str) -> bool:
        return strict_nodes is None or node_id in strict_nodes

    if not data:
        return ValidationReport((Problem("The workflow is empty."),))
    try:
        Workflow().load_workflow(copy.deepcopy(dict(data)))
    except ComfyUIAPIWorkflowException as error:
        return ValidationReport((Problem(f"The workflow does not load back: {error}"),))

    problems: list[Problem] = []
    outputs = [
        node_id for node_id, node in data.items() if catalog.is_output_node(node["class_type"])
    ]
    if not outputs:
        problems.append(
            Problem(
                "The workflow has no output node, so ComfyUI would run nothing. Save the "
                "result with save_output or with a pipeline that saves."
            )
        )
    elif not any(data[node_id]["class_type"].startswith("Save") for node_id in outputs):
        problems.append(
            Problem(
                "The workflow saves no file: its output nodes only preview. Add a save "
                "node if the user wants the result as a file.",
                blocking=False,
            )
        )

    reached = _reaching(data, outputs)
    for node_id, node in data.items():
        class_type, inputs = node["class_type"], node["inputs"]
        if outputs and node_id not in reached:
            problems.append(
                Problem(
                    f"{node_id} ({class_type}) is not connected to any output node, so it "
                    "does nothing. Connect it or remove it.",
                    node_id,
                    strict(node_id),
                )
            )
        for name, value in inputs.items():
            if not isinstance(value, str):
                continue
            folder = model_folder(class_type, name)
            if folder is not None and inventory.knows(folder) and not inventory.has(folder, value):
                available = ", ".join(inventory.folder(folder)) or "none"
                problems.append(
                    Problem(
                        f"{node_id}.{name} names {value!r}, which is not in the server's "
                        f"{folder} folder. The files there are: {available}.",
                        node_id,
                        strict(node_id),
                    )
                )
            kind = FILE_INPUTS.get((class_type, name))
            if kind is not None and value not in files:
                if strict(node_id):
                    message = (
                        f"{node_id}.{name} loads {value!r}, which the user did not attach. "
                        f"Ask for the {kind} with ask_user, or use a file the user attached."
                    )
                else:
                    message = (
                        f"{node_id}.{name} loads {value!r}; the application must upload "
                        f"{value!r} when it runs the workflow."
                    )
                problems.append(Problem(message, node_id, strict(node_id)))
    return ValidationReport(tuple(problems))
