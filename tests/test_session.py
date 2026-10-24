import json

import pytest
from comfy_ui_api_workflow.nodes.image import EmptyImage
from comfy_ui_api_workflow.nodes.image.upscaling import ImageScaleBy

from agent_comfy_ui_api_workflow._impl.catalog import get_catalog
from agent_comfy_ui_api_workflow._impl.errors import ToolInputError
from agent_comfy_ui_api_workflow._impl.session import WorkflowSession, diff_workflows
from builders import text_to_image
from conftest import SDXL


def test_parses_links_by_output_name_or_index(session):
    text_to_image(session.workflow)

    assert session.parse_link("CheckpointLoaderSimple_0.CLIP") == ("CheckpointLoaderSimple_0", 1)
    assert session.parse_link("CheckpointLoaderSimple_0.vae") == ("CheckpointLoaderSimple_0", 2)
    assert session.parse_link("CheckpointLoaderSimple_0.0") == ("CheckpointLoaderSimple_0", 0)


def test_text_that_only_looks_like_a_link_stays_text(session):
    text_to_image(session.workflow)

    assert session.parse_link("a cat. sitting") is None
    assert session.parse_link(SDXL) is None
    assert session.parse_link("CheckpointLoaderSimple_0.7") is None
    assert session.parse_link(42) is None


def test_a_wrong_output_name_of_a_real_node_is_an_error(session):
    text_to_image(session.workflow)

    with pytest.raises(ToolInputError) as error:
        session.parse_link("CheckpointLoaderSimple_0.LATENT")

    assert "MODEL, CLIP, VAE" in str(error.value)


def test_resolve_converts_links_also_inside_lists(session):
    text_to_image(session.workflow)

    assert session.resolve(["VAEDecode_5.IMAGE", "plain"]) == [("VAEDecode_5", 0), "plain"]
    assert session.resolve(3.5) == 3.5


def test_link_type_and_text(session):
    text_to_image(session.workflow)

    assert session.link_type(("CheckpointLoaderSimple_0", 1)) == "CLIP"
    assert session.link_text(("CheckpointLoaderSimple_0", 1)) == "CheckpointLoaderSimple_0.CLIP"


def test_view_lists_inputs_and_where_each_output_goes(session):
    text_to_image(session.workflow)

    text = session.view()

    assert text.startswith("Workflow with 7 nodes. Output nodes: SaveImage_6.")
    assert (
        'CheckpointLoaderSimple_0 = CheckpointLoaderSimple(ckpt_name="sd_xl_base_1.0.safetensors")'
        in text
    )
    assert "    CLIP → CLIPTextEncode_1.clip, CLIPTextEncode_2.clip" in text
    assert (
        'CLIPTextEncode_1 = CLIPTextEncode(text="a lighthouse", clip=CheckpointLoaderSimple_0.CLIP)'
        in text
    )
    assert (
        'SaveImage_6 = SaveImage(images=VAEDecode_5.IMAGE, filename_prefix="out")  [output node]'
        in text
    )


def test_view_flags_nodes_that_reach_no_output(session):
    text_to_image(session.workflow)
    session.workflow.add_node(EmptyImage, width=64, height=64)

    (line,) = [line for line in session.view().splitlines() if line.startswith("EmptyImage_7 =")]

    assert line.endswith("[not connected to any output node]")


def test_view_of_an_empty_workflow(session):
    assert session.view() == "The workflow is empty."


def test_snapshot_is_plain_json(session):
    text_to_image(session.workflow)

    snapshot = session.snapshot()

    assert snapshot["VAEDecode_5"]["inputs"]["samples"] == ["KSampler_4", 0]
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_diff_names_added_removed_and_changed_nodes(session):
    text_to_image(session.workflow)
    before = session.snapshot()
    session.workflow.update_node("KSampler_4", steps=30, cfg=5.0)
    session.workflow.add_node(
        ImageScaleBy, image=("VAEDecode_5", 0), upscale_method="lanczos", scale_by=2.0
    )
    after = session.snapshot()
    del after["SaveImage_6"]

    diff = diff_workflows(before, after)

    assert diff.added == ("ImageScaleBy_7",)
    assert diff.removed == ("SaveImage_6",)
    assert diff.changed == {"KSampler_4": ("cfg", "steps")}
    assert diff.text() == "Added: ImageScaleBy_7. Removed: SaveImage_6. Changed: KSampler_4 (cfg, steps)."
    assert not diff_workflows(before, before)
    assert diff_workflows(before, before).text() == "No changes."


def test_from_api_keeps_the_ids_of_a_loaded_workflow(inventory):
    data = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    }

    session = WorkflowSession.from_api(data, get_catalog(), inventory)

    assert list(session.workflow.nodes) == ["4", "6"]
    assert session.snapshot() == data
    assert session.agent_nodes == set()


def test_start_turn_clears_the_turn_state(session):
    session.mark("A_1")
    session.nudges = 2

    session.start_turn()

    assert (session.changed, session.nudges, session.question) == (False, 0, None)
    assert session.agent_nodes == {"A_1"}
