import pytest
from comfy_ui_api_workflow.nodes.model.sampling import KSampler
from comfy_ui_api_workflow.nodes.video import SaveVideo, VideoSlice

from agent_comfy_ui_api_workflow._impl.catalog import (
    ESSENTIAL_GROUPS,
    NodeCatalog,
    get_catalog,
)


@pytest.fixture(scope="module")
def catalog():
    return get_catalog()


def test_every_essential_node_is_in_the_lib_and_in_object_info(catalog):
    for names in ESSENTIAL_GROUPS.values():
        for name in names:
            node_type = catalog.node_type(name)
            assert node_type is not None, name
            assert catalog.info(node_type), f"{name} has no object_info entry"


def test_finds_nodes_by_class_name_or_comfy_name_ignoring_case(catalog):
    assert catalog.node_type("ksampler") is KSampler
    assert catalog.node_type("Video Slice") is VideoSlice
    assert catalog.node_type("VideoSlice") is VideoSlice
    assert catalog.node_type("NoSuchNode") is None


def test_only_essential_nodes_can_be_added(catalog):
    assert catalog.essential_type("KSampler") is KSampler
    assert catalog.essential_type("ModelMergeSimple") is None


def test_describe_has_types_ranges_defaults_tooltips_and_outputs(catalog):
    text = catalog.describe(KSampler)

    assert text.startswith("KSampler")
    assert "- model: link to MODEL, required. The model used for denoising" in text
    assert "- steps: int from 1 to 10000, default 20. The number of steps" in text
    assert "- sampler_name: one of: " in text and "dpmpp_2m" in text
    assert 'Outputs (link to one as "<node_id>.<output name>"):' in text
    assert "- LATENT (LATENT): The denoised latent." in text


def test_describe_shows_the_comfy_class_type_when_it_differs(catalog):
    assert "class_type 'Video Slice'" in catalog.describe(VideoSlice)


def test_describe_reads_tooltips_of_nested_dynamic_inputs(catalog):
    text = catalog.describe(SaveVideo)

    assert "- format_codec: one of: auto, h264, av1" in text
    assert "The output video codec." in text


def test_describe_groups_numbered_inputs(catalog):
    text = catalog.describe(catalog.node_type("ConcatenateVideo"))

    assert "- videos_video0 … videos_video99: link to VIDEO" in text
    assert "videos_video57" not in text


def test_describe_keeps_short_numbered_pairs(catalog):
    text = catalog.describe(catalog.node_type("AudioMerge"))

    assert "- audio1: link to AUDIO, required." in text
    assert "- audio2: link to AUDIO, required." in text


def test_input_spec_accepts_the_comfy_name(catalog):
    assert catalog.input_spec(SaveVideo, "format.codec").name == "format_codec"
    assert catalog.input_spec(SaveVideo, "format_codec").comfy_name == "format.codec"
    assert catalog.input_spec(SaveVideo, "nope") is None


def test_output_nodes_come_from_object_info(catalog):
    assert catalog.is_output_node("SaveImage")
    assert catalog.is_output_node("SaveVideo")
    assert not catalog.is_output_node("KSampler")
    assert not catalog.is_output_node("Unknown")


def test_output_type(catalog):
    assert catalog.output_type("CheckpointLoaderSimple", 1) == "CLIP"
    assert catalog.output_type("KSampler", 5) is None


def test_index_lists_every_essential_node_by_group(catalog):
    text = catalog.index()

    for group, names in ESSENTIAL_GROUPS.items():
        assert f"{group}:" in text
        for name in names:
            assert f"- {catalog.node_type(name).__name__}(" in text
    assert "- KSampler(model, sampler_name, scheduler, positive, negative, latent_image) → LATENT" in text


def test_a_description_that_opens_a_list_falls_back_to_the_display_name(catalog):
    line = catalog.summary_line(catalog.node_type("CLIPLoader"))

    assert line == "- CLIPLoader(clip_name, type) → CLIP: Load CLIP"


def test_extra_groups_extend_the_essential_nodes():
    catalog = NodeCatalog(extra={"Wan 2.2 video": ("Wan22ImageToVideoLatent",)})

    assert catalog.essential_type("Wan22ImageToVideoLatent") is not None
    assert "Wan 2.2 video:" in catalog.index()


def test_hidden_nodes_are_left_out():
    catalog = NodeCatalog(hidden={"LoraLoader"})

    assert catalog.essential_type("LoraLoader") is None
    assert "- LoraLoader(" not in catalog.index()


def test_an_unknown_essential_name_is_an_error():
    with pytest.raises(ValueError):
        NodeCatalog(extra={"Broken": ("NoSuchNode",)})
