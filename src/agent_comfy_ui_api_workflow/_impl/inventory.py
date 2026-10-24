import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_ui_api_workflow import Workflow
from comfy_ui_api_workflow.nodes.model.loaders import (
    CheckpointLoaderSimple,
    DualCLIPLoader,
    LoraLoader,
    UNETLoader,
    UpscaleModelLoader,
    VAELoader,
)
from comfy_ui_api_workflow.nodes.utilities import PreviewAny

from .errors import InventoryError

# The folders the probe lists: the worker's handler checks a loader of each.
FOLDERS = (
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "upscale_models",
)
# A file name no server has: the worker answers it with the files of the folder.
PLACEHOLDER = "__list__"
# The loader inputs that name a model file, and the server folder of that file.
MODEL_INPUTS: dict[tuple[str, str], str] = {
    ("CheckpointLoaderSimple", "ckpt_name"): "checkpoints",
    ("UNETLoader", "unet_name"): "diffusion_models",
    ("CLIPLoader", "clip_name"): "text_encoders",
    ("DualCLIPLoader", "clip_name1"): "text_encoders",
    ("DualCLIPLoader", "clip_name2"): "text_encoders",
    ("VAELoader", "vae_name"): "vae",
    ("LoraLoader", "lora_name"): "loras",
    ("LoraLoaderModelOnly", "lora_name"): "loras",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
    ("AudioEncoderLoader", "audio_encoder_name"): "audio_encoders",
}
_LISTING = re.compile(r"available (\w+) files(?::\s*(.*))?$")


def model_folder(class_type: str, input_name: str) -> str | None:
    return MODEL_INPUTS.get((class_type, input_name))


@dataclass(frozen=True)
class ServerInventory:
    """The model files on the ComfyUI server, by folder. Read only."""

    files: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {
            folder: tuple(sorted(set(names))) for folder, names in self.files.items()
        }
        object.__setattr__(self, "files", normalized)

    def folder(self, name: str) -> tuple[str, ...]:
        return tuple(self.files.get(name, ()))

    def knows(self, folder: str) -> bool:
        return folder in self.files

    def has(self, folder: str, filename: str) -> bool:
        return filename in self.files.get(folder, ())

    def describe(self) -> str:
        return "\n".join(
            f"- {folder}: {', '.join(names) if names else '(none)'}"
            for folder, names in sorted(self.files.items())
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {folder: list(names) for folder, names in sorted(self.files.items())}

    @classmethod
    def from_dict(cls, data: Any) -> "ServerInventory":
        if not isinstance(data, dict) or not all(
            isinstance(names, list) and all(isinstance(name, str) for name in names)
            for names in data.values()
        ):
            raise InventoryError("it must map each folder to a list of file names")
        return cls(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ServerInventory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_listing(text: str) -> ServerInventory:
    """Reads the worker's reply to a workflow that names missing model files."""
    files: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = _LISTING.search(line)
        if match is None:
            continue
        folder, names = match.group(1), match.group(2) or ""
        files.setdefault(folder, []).extend(re.findall(r"'([^']*)'", names))
    if not files:
        raise InventoryError(f"the worker's reply lists no model folder: {text[:300]!r}")
    return ServerInventory(files)


def _combo_options(spec: Any) -> list[str] | None:
    # A combo input is [[option, ...], {...}], or ["COMBO", {"options": [...]}] in V3 nodes.
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        options = spec[0]
    elif spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options", [])
    else:
        return None
    return [option for option in options if isinstance(option, str)]


def parse_object_info(info: Mapping[str, Any]) -> ServerInventory:
    """Reads the files of each folder from the options of its loaders in /object_info."""
    files: dict[str, list[str]] = {}
    for (class_type, input_name), folder in MODEL_INPUTS.items():
        node = info.get(class_type)
        if not isinstance(node, dict):
            continue
        inputs = node.get("input") or {}
        spec = (inputs.get("required") or {}).get(input_name) or (inputs.get("optional") or {}).get(input_name)
        options = _combo_options(spec)
        if options is not None:
            files.setdefault(folder, []).extend(options)
    if not files:
        raise InventoryError("the server's object_info has none of the model loaders")
    return ServerInventory(files)


def probe_workflow() -> dict[str, Any]:
    """A workflow whose loaders name the placeholder, one loader for each folder."""
    workflow = Workflow()
    checkpoint = workflow.add_node(CheckpointLoaderSimple, ckpt_name=PLACEHOLDER)
    lora = workflow.add_node(
        LoraLoader,
        model=checkpoint.output["MODEL"],
        clip=checkpoint.output["CLIP"],
        lora_name=PLACEHOLDER,
    )
    vae = workflow.add_node(VAELoader, vae_name=PLACEHOLDER)
    upscaler = workflow.add_node(UpscaleModelLoader, model_name=PLACEHOLDER)
    unet = workflow.add_node(UNETLoader, unet_name=PLACEHOLDER, weight_dtype="default")
    # The handler checks DualCLIPLoader, not CLIPLoader.
    text_encoder = workflow.add_node(
        DualCLIPLoader, clip_name1=PLACEHOLDER, clip_name2=PLACEHOLDER, type="sdxl", device="default"
    )
    for output in (
        lora.output["MODEL"],
        vae.output["VAE"],
        upscaler.output["UPSCALE_MODEL"],
        unet.output["MODEL"],
        text_encoder.output["CLIP"],
    ):
        workflow.add_node(PreviewAny, source=output)
    return workflow.get_comfy_ui_workflow_format()
