"""The RunPod worker agrees with the library: run with --runpod (paid, almost no GPU time)."""

import asyncio
from typing import Any

import pytest
from comfy_ui_api_workflow import Workflow

from agent_comfy_ui_api_workflow import RunPodClient
from agent_comfy_ui_api_workflow._impl.catalog import ESSENTIAL_GROUPS, get_catalog
from agent_comfy_ui_api_workflow._impl.catalog import object_info as library_object_info
from agent_comfy_ui_api_workflow._impl.families import ModelChoice, family_nodes, find_models, unclassified_files
from agent_comfy_ui_api_workflow._impl.files import FILE_INPUTS
from agent_comfy_ui_api_workflow._impl.inventory import MODEL_INPUTS, parse_object_info
from agent_comfy_ui_api_workflow._impl.pipelines import GENERATION_PIPELINES, MEDIA_PIPELINES
from agent_comfy_ui_api_workflow._impl.pipelines.media import FramesToVideo, LoadVideoParts
from agent_comfy_ui_api_workflow._impl.session import api_json
from builders import text_to_image

pytestmark = pytest.mark.runpod

FOLDERS = {
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "upscale_models",
    "audio_encoders",
}
# Arguments that build each pipeline; generation pipelines get each model that can run them.
ARGUMENTS: dict[str, dict[str, Any]] = {
    "text_to_image": {"prompt": "x", "upscale": 1.5},
    "image_to_image": {"image": "a.png", "prompt": "x"},
    "inpaint": {"image": "a.png", "prompt": "x", "region": [0, 0, 8, 8]},
    "outpaint": {"image": "a.png", "prompt": "x", "right": 8},
    "text_to_video": {"prompt": "x"},
    "image_to_video": {"image": "a.png", "prompt": "x"},
    "sound_to_video": {"image": "a.png", "audio": "a.wav", "prompt": "x", "duration_seconds": 10.0},
    "text_to_audio": {"style": "x"},
    "upscale_image_with_model": {"image": "a.png", "scale": 2.0},
    "upscale_image": {"image": "a.png"},
    "load_video": {"video": "a.mp4"},
    "trim_video": {"video": "a.mp4", "start_seconds": 1.0},
    "add_audio_to_video": {"video": "a.mp4", "audio": "a.wav"},
    "concatenate_videos": {"videos": ["a.mp4", "b.mp4"]},
    "video_to_animation": {"video": "a.mp4", "format": "png"},
    "edit_audio": {"audio": "a.wav", "start_seconds": 1.0, "volume_db": 3},
    "combine_audio": {"audios": ["a.wav", "b.wav"], "mode": "mix"},
}


@pytest.fixture(scope="module")
def server_info() -> dict[str, Any]:
    async def fetch():
        async with RunPodClient.from_env() as client:
            return await client.object_info()

    return asyncio.run(fetch())


def classes(workflow: Workflow) -> set[str]:
    return {node["class_type"] for node in api_json(workflow).values()}


def pipeline_classes(models: tuple[ModelChoice, ...]) -> set[str]:
    """Every node class the pipelines add, built with each model of the server."""
    found: set[str] = set()
    for pipeline in (*GENERATION_PIPELINES, *MEDIA_PIPELINES):
        if pipeline.NAME not in ARGUMENTS:
            continue
        usable = [model for model in models if pipeline.CAPABILITY in model.capabilities]
        for model in usable or [None]:
            arguments = dict(ARGUMENTS[pipeline.NAME])
            if model is not None:
                arguments["model"] = model
                if pipeline.NAME == "image_to_video" and model.family != "wan22_5b":
                    arguments["end_image"] = "b.png"
            workflow = Workflow()
            workflow.add_pipeline(pipeline, **arguments)
            found |= classes(workflow)
    # frames_to_video takes its frames from another pipeline.
    workflow = Workflow()
    parts = workflow.add_pipeline(LoadVideoParts, video="a.mp4").output
    workflow.add_pipeline(FramesToVideo, frames=parts["frames"], fps=parts["fps"], audio=parts["audio"])
    return found | classes(workflow)


def inputs(spec: dict[str, Any]) -> dict[str, list]:
    return {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}


def kind(value: list) -> str:
    return "COMBO" if isinstance(value[0], list) else value[0]


def options(value: list) -> list:
    if isinstance(value[0], list):
        return value[0]
    extra = value[1] if len(value) > 1 and isinstance(value[1], dict) else {}
    return extra.get("options", [])


def differences(name: str, library: dict[str, Any], server: dict[str, Any]) -> list[str]:
    """What a workflow built by the library would get wrong about this node on the server."""
    if name not in server:
        return [f"{name} is not on the server"]
    ours, theirs = inputs(library[name]), inputs(server[name])
    problems = [
        f"{name}.{input_name} is required on the server and unknown to the library"
        for input_name in server[name]["input"].get("required", {})
        if input_name not in ours
    ]
    for input_name, spec in ours.items():
        if input_name not in theirs:
            problems.append(f"{name}.{input_name} is not on the server")
        elif kind(spec) != kind(theirs[input_name]):
            problems.append(f"{name}.{input_name} is {kind(spec)} here and {kind(theirs[input_name])} there")
        elif kind(spec) == "COMBO" and (name, input_name) not in MODEL_INPUTS | FILE_INPUTS.keys():
            missing = set(options(spec)) - set(options(theirs[input_name]))
            if missing:
                problems.append(f"{name}.{input_name} options the server lacks: {sorted(missing)}")
    if library[name].get("output") != server[name].get("output"):
        problems.append(f"{name} outputs {library[name].get('output')} here, {server[name].get('output')} there")
    return problems


async def test_every_model_file_belongs_to_a_family():
    async with RunPodClient.from_env() as client:
        inventory = await client.discover_inventory()

    assert FOLDERS <= set(inventory.files)
    models = find_models(inventory)
    assert models
    assert unclassified_files(inventory, models) == {}


def test_the_server_runs_every_node_the_library_adds(server_info):
    models = find_models(parse_object_info(server_info))
    # The groups name some nodes by their Python class, like VideoSlice for "Video Slice".
    catalog = get_catalog()
    groups = [*ESSENTIAL_GROUPS.values(), *family_nodes(models).values()]
    names = {catalog.node_type(name).CLASS_TYPE for group in groups for name in group}
    names |= pipeline_classes(models)
    library = library_object_info()

    problems = [problem for name in sorted(names) for problem in differences(name, library, server_info)]

    assert problems == []


async def test_a_missing_model_fails_with_its_name():
    workflow = Workflow()
    text_to_image(workflow, checkpoint="not_on_this_server.safetensors")

    async with RunPodClient.from_env() as client:
        result = await client.run(api_json(workflow))

    assert not result.ok
    assert "not_on_this_server.safetensors" in (result.error or "")


async def test_health_reports_the_jobs_and_workers():
    async with RunPodClient.from_env() as client:
        health = await client.health()

    assert {"jobs", "workers"} <= set(health)
