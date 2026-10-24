import pytest
from comfy_ui_api_workflow.nodes.audio import EmptyAudio
from comfy_ui_api_workflow.nodes.image import EmptyImage
from comfy_ui_api_workflow.nodes.image.upscaling import ImageScaleBy
from comfy_ui_api_workflow.nodes.model.patch.unet import FreeU
from comfy_ui_api_workflow.nodes.video import CreateVideo
from langchain_core.messages import ToolMessage

from agent_comfy_ui_api_workflow._impl.families import find_models
from agent_comfy_ui_api_workflow._impl.pipelines import MEDIA_PIPELINES, available_pipelines
from agent_comfy_ui_api_workflow._impl.tools import build_tools
from builders import text_to_image
from conftest import SDXL
from media import MP4, png

FIXED_TOOLS = {
    "add_node",
    "describe_node",
    "update_node",
    "remove_node",
    "view_workflow",
    "validate_workflow",
    "save_output",
    "new_workflow",
    "ask_user",
}


@pytest.fixture
def tools(session):
    return {tool.name: tool for tool in build_tools(session, MEDIA_PIPELINES)}


@pytest.fixture
def generation_tools(session, inventory):
    models = find_models(inventory)
    session.models = {model.id: model for model in models}
    return {tool.name: tool for tool in build_tools(session, available_pipelines(models))}


async def call(tools, name: str, **args) -> ToolMessage:
    return await tools[name].ainvoke(
        {"type": "tool_call", "id": "call-1", "name": name, "args": args}
    )


def ok(message: ToolMessage) -> str:
    assert message.status == "success", message.content
    return message.content


def failed(message: ToolMessage) -> str:
    assert message.status == "error", message.content
    return message.content


def test_one_tool_per_pipeline_plus_the_fixed_ones(tools):
    assert set(tools) == {pipeline.NAME for pipeline in MEDIA_PIPELINES} | FIXED_TOOLS


def test_pipeline_schemas_take_file_names_and_choices(tools):
    schema = tools["trim_video"].tool_call_schema.model_json_schema()

    assert schema["properties"]["video"]["type"] == "string"
    assert schema["required"] == ["video"]
    assert "0 keeps everything" in schema["properties"]["duration_seconds"]["description"]
    audio_format = tools["edit_audio"].tool_call_schema.model_json_schema()["properties"]["format"]
    assert audio_format["enum"] == ["mp3", "flac", "opus"]
    assert tools["trim_video"].description.startswith("Cuts a video to a time range")


async def test_a_pipeline_uses_an_attached_file(session, tools):
    session.files.add("clip.mp4", MP4)

    content = ok(
        await call(tools, "trim_video", video="clip.mp4", start_seconds=1.0, duration_seconds=2.0)
    )

    assert content == (
        "Added trim_video: LoadVideo_0, VideoSlice_1, SaveVideo_2. "
        "Outputs: video = VideoSlice_1.VIDEO (VIDEO)."
    )
    assert session.agent_nodes == {"LoadVideo_0", "VideoSlice_1", "SaveVideo_2"}
    assert session.changed


async def test_a_pipeline_refuses_a_file_the_user_did_not_attach(tools):
    content = failed(await call(tools, "trim_video", video="clip.mp4"))

    assert "'clip.mp4'" in content
    assert "no files are attached" in content
    assert "ask_user" in content


async def test_a_pipeline_refuses_a_file_of_the_wrong_kind(session, tools):
    session.files.add("me.png", png())

    content = failed(await call(tools, "trim_video", video="me.png"))

    assert "'me.png' is an image file" in content


async def test_a_pipeline_takes_links_to_outputs(session, tools):
    session.files.add("clip.mp4", MP4)
    ok(await call(tools, "load_video", video="clip.mp4"))

    content = ok(
        await call(
            tools,
            "frames_to_video",
            frames="GetVideoComponents_1.images",
            fps="GetVideoComponents_1.fps",
            audio="GetVideoComponents_1.audio",
        )
    )

    assert content.startswith("Added frames_to_video: CreateVideo_2, SaveVideo_3.")


async def test_pipeline_errors_name_the_tool_and_add_nothing(session, tools):
    session.files.add("a.mp4", MP4)

    content = failed(await call(tools, "concatenate_videos", videos=["a.mp4"]))

    assert "Cannot add pipeline 'concatenate_videos'" in content
    assert session.workflow.nodes == {}


async def test_add_node_links_outputs_and_points_at_the_pipeline(session, tools):
    text_to_image(session.workflow)

    content = ok(
        await call(
            tools,
            "add_node",
            node_type="ImageScaleBy",
            inputs={"image": "VAEDecode_5.IMAGE", "upscale_method": "lanczos", "scale_by": 2.0},
        )
    )

    assert content.startswith("Added ImageScaleBy_7. Outputs: ImageScaleBy_7.IMAGE (IMAGE).")
    assert "upscale_image" in content
    assert "ImageScaleBy_7" in session.agent_nodes


async def test_add_node_accepts_comfy_input_names(session, tools):
    session.files.add("clip.mp4", MP4)
    ok(await call(tools, "add_node", node_type="LoadVideo", inputs={"file": "clip.mp4"}))

    ok(
        await call(
            tools,
            "add_node",
            node_type="SaveVideo",
            inputs={"video": "LoadVideo_0.VIDEO", "format": "webm", "format.codec": "av1"},
        )
    )

    assert session.snapshot()["SaveVideo_1"]["inputs"]["format.codec"] == "av1"


async def test_add_node_refuses_nodes_outside_the_essential_list(tools):
    assert "not one of the essential nodes" in failed(
        await call(tools, "add_node", node_type="ModelMergeSimple", inputs={})
    )
    assert "no node 'MakeMagic'" in failed(
        await call(tools, "add_node", node_type="MakeMagic", inputs={})
    )


async def test_add_node_names_unknown_and_missing_inputs(tools):
    content = failed(
        await call(tools, "add_node", node_type="ImageScaleBy", inputs={"img": "x"})
    )
    assert "ImageScaleBy has no input 'img'" in content
    assert "image, upscale_method, scale_by" in content

    content = failed(
        await call(tools, "add_node", node_type="ImageScaleBy", inputs={"scale_by": 2.0})
    )
    assert "ImageScaleBy needs the inputs 'image' and 'upscale_method'" in content


async def test_add_node_refuses_a_model_that_is_not_on_the_server(tools):
    content = failed(
        await call(
            tools,
            "add_node",
            node_type="CheckpointLoaderSimple",
            inputs={"ckpt_name": "flux.safetensors"},
        )
    )

    assert "'flux.safetensors'" in content
    assert SDXL in content


async def test_add_node_refuses_a_file_the_user_did_not_attach(tools):
    content = failed(
        await call(tools, "add_node", node_type="LoadImage", inputs={"image": "me.png"})
    )

    assert "ask_user" in content


async def test_lib_errors_come_back_as_tool_errors(session, tools):
    text_to_image(session.workflow)

    content = failed(
        await call(
            tools,
            "add_node",
            node_type="ImageScaleBy",
            inputs={"image": "KSampler_4.LATENT", "upscale_method": "lanczos"},
        )
    )

    assert "LATENT" in content
    assert "IMAGE" in content


async def test_describe_node(tools):
    assert ok(await call(tools, "describe_node", node_type="imagescaleby")).startswith(
        "ImageScaleBy"
    )
    assert "not one of the essential nodes" in failed(
        await call(tools, "describe_node", node_type="ModelMergeSimple")
    )


async def test_describe_node_works_for_a_node_already_in_the_workflow(session, tools):
    text_to_image(session.workflow)
    session.workflow.add_node(FreeU, model=("CheckpointLoaderSimple_0", 0))

    assert ok(await call(tools, "describe_node", node_type="FreeU")).startswith("FreeU")


async def test_update_node_changes_values(session, tools):
    text_to_image(session.workflow)

    content = ok(await call(tools, "update_node", node_id="KSampler_4", inputs={"steps": 30, "cfg": 5.5}))

    assert content == "Updated KSampler_4: steps=30, cfg=5.5."
    assert session.workflow.nodes["KSampler_4"].steps == 30
    assert "KSampler_4" in session.agent_nodes


async def test_update_node_is_all_or_nothing(session, tools):
    text_to_image(session.workflow)

    failed(await call(tools, "update_node", node_id="KSampler_4", inputs={"steps": 30, "cfg": 500.0}))

    assert session.workflow.nodes["KSampler_4"].steps == 25


async def test_update_node_null_removes_an_optional_input(session, tools):
    frames = session.workflow.add_node(EmptyImage, width=64, height=64, batch_size=4)
    sound = session.workflow.add_node(EmptyAudio)
    session.workflow.add_node(CreateVideo, images=frames.output["IMAGE"], audio=sound.output["AUDIO"])

    content = ok(await call(tools, "update_node", node_id="CreateVideo_2", inputs={"audio": None}))

    assert content == "Updated CreateVideo_2: audio removed."
    assert "audio" not in session.snapshot()["CreateVideo_2"]["inputs"]


async def test_update_node_of_a_missing_node(session, tools):
    text_to_image(session.workflow)

    content = failed(await call(tools, "update_node", node_id="Nope_1", inputs={"steps": 1}))

    assert "There is no node 'Nope_1'" in content
    assert "KSampler_4" in content


async def test_remove_node_reconnects_the_chain(session, tools):
    text_to_image(session.workflow)
    session.workflow.add_node(
        ImageScaleBy, image=("VAEDecode_5", 0), upscale_method="lanczos", scale_by=2.0
    )
    session.workflow.update_node("SaveImage_6", images=("ImageScaleBy_7", 0))

    content = ok(await call(tools, "remove_node", node_id="ImageScaleBy_7"))

    assert content == "Removed ImageScaleBy_7. Reconnected SaveImage_6.images to VAEDecode_5.IMAGE."
    assert session.snapshot()["SaveImage_6"]["inputs"]["images"] == ["VAEDecode_5", 0]


async def test_remove_node_without_reconnect_names_the_dependents(session, tools):
    text_to_image(session.workflow)

    content = failed(await call(tools, "remove_node", node_id="VAEDecode_5", reconnect=False))

    assert "SaveImage_6" in content


async def test_remove_node_that_cannot_be_reconnected_changes_nothing(session, tools):
    text_to_image(session.workflow)
    before = session.snapshot()

    content = failed(await call(tools, "remove_node", node_id="CheckpointLoaderSimple_0"))

    assert "Cannot reconnect" in content
    assert session.snapshot() == before


async def test_save_output_picks_the_save_node_by_type(session, tools):
    text_to_image(session.workflow)

    ok(await call(tools, "save_output", source="VAEDecode_5.IMAGE", filename_prefix="copy"))
    assert session.snapshot()["SaveImage_7"]["inputs"] == {
        "images": ["VAEDecode_5", 0],
        "filename_prefix": "copy",
    }

    session.files.add("clip.mp4", MP4)
    ok(await call(tools, "add_node", node_type="LoadVideo", inputs={"file": "clip.mp4"}))
    ok(await call(tools, "save_output", source="LoadVideo_8.VIDEO", format="webm"))
    saved = session.snapshot()["SaveVideo_9"]["inputs"]
    assert (saved["format"], saved["format.codec"]) == ("webm", "av1")

    assert "VAEDecode" in failed(await call(tools, "save_output", source="KSampler_4.LATENT"))


async def test_view_and_validate(session, tools):
    text_to_image(session.workflow)

    assert ok(await call(tools, "view_workflow")).startswith("Workflow with 7 nodes")
    assert ok(await call(tools, "validate_workflow")) == "Valid: ready to run."


async def test_new_workflow_sets_the_current_one_aside(session, tools):
    text_to_image(session.workflow)

    content = ok(await call(tools, "new_workflow"))

    assert content == "Started an empty workflow. The previous one (7 nodes) was set aside."
    assert session.workflow.nodes == {}
    assert len(session.previous) == 1


def test_the_model_parameter_lists_the_models_that_can_do_the_job(generation_tools):
    schema = generation_tools["text_to_image"].tool_call_schema.model_json_schema()

    # With a single option pydantic writes "const" instead of "enum".
    model = schema["properties"]["model"]
    assert model.get("enum", [model.get("const")]) == ["sd_xl_base_1.0"]
    assert model["default"] == "sd_xl_base_1.0"
    assert "SDXL" in model["description"]
    assert "model" not in schema["required"]
    video = generation_tools["text_to_video"].tool_call_schema.model_json_schema()["properties"]["model"]
    assert video.get("enum", [video.get("const")]) == ["wan2.2_ti2v_5B_fp16"]


async def test_a_generation_pipeline_uses_the_default_model(session, generation_tools):
    content = ok(await call(generation_tools, "text_to_image", prompt="a koi pond", seed=1))

    assert content.startswith("Added text_to_image: CheckpointLoaderSimple_0, ")
    assert content.endswith("Outputs: images = VAEDecode_5.IMAGE (IMAGE).")
    assert session.snapshot()["CheckpointLoaderSimple_0"]["inputs"]["ckpt_name"] == SDXL


async def test_invalid_arguments_come_back_with_the_reason(generation_tools):
    content = failed(await call(generation_tools, "text_to_image", prompt="x", model="flux"))

    assert "model" in content
    assert "sd_xl_base_1.0" in content


async def test_ask_user_records_the_question_and_ends_the_turn(session, tools):
    ok(await call(tools, "ask_user", question="Envie a foto do seu rosto.", needs=["image"]))

    assert session.question.text == "Envie a foto do seu rosto."
    assert session.question.needs == ("image",)
    assert tools["ask_user"].return_direct
