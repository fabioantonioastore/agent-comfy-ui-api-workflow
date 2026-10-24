import pytest
from comfy_ui_api_workflow import Workflow

from agent_comfy_ui_api_workflow._impl.errors import InventoryError
from agent_comfy_ui_api_workflow._impl.inventory import (
    FOLDERS,
    PLACEHOLDER,
    ServerInventory,
    model_folder,
    parse_listing,
    parse_object_info,
    probe_workflow,
)

# What the worker answered on 2026-10-16 when every loader named the placeholder.
LISTING = (
    "Workflow validation failed — the following referenced files are not available on "
    "this worker:\n"
    "• Node CheckpointLoaderSimple_0 (CheckpointLoaderSimple.ckpt_name): got the placeholder "
    "'__list__' instead of a real model filename — the client sent a UI default that was "
    "never replaced with an actual file. Set it to one of the available checkpoints files: "
    "'sd_xl_base_1.0.safetensors'\n"
    "• Node DualCLIPLoader_5 (DualCLIPLoader.clip_name1): got the placeholder '__list__' "
    "instead of a real model filename — the client sent a UI default that was never "
    "replaced with an actual file. Set it to one of the available text_encoders files: "
    "'umt5_xxl_fp8_e4m3fn_scaled.safetensors'\n"
    "• Node DualCLIPLoader_5 (DualCLIPLoader.clip_name2): got the placeholder '__list__' "
    "instead of a real model filename — the client sent a UI default that was never "
    "replaced with an actual file. Set it to one of the available text_encoders files: "
    "'umt5_xxl_fp8_e4m3fn_scaled.safetensors'\n"
    "• Node LoraLoader_1 (LoraLoader.lora_name): got the placeholder '__list__' instead of "
    "a real model filename — the client sent a UI default that was never replaced with an "
    "actual file. Set it to one of the available loras files\n"
    "• Node UNETLoader_4 (UNETLoader.unet_name): got the placeholder '__list__' instead of "
    "a real model filename — the client sent a UI default that was never replaced with an "
    "actual file. Set it to one of the available diffusion_models files: "
    "'wan2.2_ti2v_5B_fp16.safetensors'\n"
    "• Node UpscaleModelLoader_3 (UpscaleModelLoader.model_name): got the placeholder "
    "'__list__' instead of a real model filename — the client sent a UI default that was "
    "never replaced with an actual file. Set it to one of the available upscale_models "
    "files\n"
    "• Node VAELoader_2 (VAELoader.vae_name): got the placeholder '__list__' instead of a "
    "real model filename — the client sent a UI default that was never replaced with an "
    "actual file. Set it to one of the available vae files: 'pixel_space', "
    "'wan2.2_vae.safetensors'\n\n"
    "Upload the missing file(s) to the matching directory on your network volume, or "
    "update the workflow to use one of the available files."
)

# Loaders in /object_info as ComfyUI 0.42.1 serves it: V3 nodes give a combo's
# options in a "COMBO" spec.
OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"], {}]}}},
    "DualCLIPLoader": {
        "input": {
            "required": {
                "clip_name1": [["umt5_xxl_fp8_e4m3fn_scaled.safetensors"], {}],
                "clip_name2": [["umt5_xxl_fp8_e4m3fn_scaled.safetensors"], {}],
                "type": [["sdxl", "ace"], {}],
            }
        }
    },
    "UpscaleModelLoader": {
        "input": {"required": {"model_name": ["COMBO", {"options": ["RealESRGAN_x4plus.safetensors"]}]}}
    },
    "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[], {}]}}},
    "KSampler": {"input": {"required": {"seed": ["INT", {"default": 0}]}}},
}


def test_reads_the_loader_options_of_object_info():
    inventory = parse_object_info(OBJECT_INFO)

    assert inventory.folder("checkpoints") == ("sd_xl_base_1.0.safetensors",)
    assert inventory.folder("text_encoders") == ("umt5_xxl_fp8_e4m3fn_scaled.safetensors",)
    assert inventory.folder("upscale_models") == ("RealESRGAN_x4plus.safetensors",)
    assert inventory.knows("loras")
    assert inventory.folder("loras") == ()
    assert not inventory.knows("vae")


def test_object_info_without_loaders_is_an_error():
    with pytest.raises(InventoryError):
        parse_object_info({"KSampler": OBJECT_INFO["KSampler"]})


def test_parses_every_folder_of_the_worker_listing():
    inventory = parse_listing(LISTING)

    assert inventory.folder("checkpoints") == ("sd_xl_base_1.0.safetensors",)
    assert inventory.folder("diffusion_models") == ("wan2.2_ti2v_5B_fp16.safetensors",)
    assert inventory.folder("text_encoders") == ("umt5_xxl_fp8_e4m3fn_scaled.safetensors",)
    assert inventory.folder("vae") == ("pixel_space", "wan2.2_vae.safetensors")
    assert inventory.folder("loras") == ()
    assert inventory.knows("loras")
    assert inventory.has("checkpoints", "sd_xl_base_1.0.safetensors")


def test_a_reply_without_a_listing_is_an_error():
    with pytest.raises(InventoryError) as error:
        parse_listing("CUDA out of memory")

    assert "CUDA out of memory" in str(error.value)


def test_a_folder_that_was_not_listed_is_unknown():
    inventory = ServerInventory({"checkpoints": ["a.safetensors"]})

    assert not inventory.knows("loras")
    assert inventory.folder("loras") == ()


def test_names_are_sorted_without_duplicates():
    inventory = ServerInventory({"vae": ["b", "a", "b"]})

    assert inventory.folder("vae") == ("a", "b")


def test_round_trips_through_a_json_file(tmp_path):
    inventory = parse_listing(LISTING)
    path = tmp_path / "inventory.json"

    inventory.save(path)

    assert ServerInventory.load(path) == inventory


def test_a_malformed_file_is_an_error(tmp_path):
    path = tmp_path / "inventory.json"
    path.write_text('{"checkpoints": "not a list"}', encoding="utf-8")

    with pytest.raises(InventoryError):
        ServerInventory.load(path)


def test_describe_lists_each_folder():
    text = parse_listing(LISTING).describe()

    assert "- checkpoints: sd_xl_base_1.0.safetensors" in text
    assert "- loras: (none)" in text


def test_the_probe_names_the_placeholder_in_a_loader_of_every_folder():
    workflow = probe_workflow()

    probed = {
        model_folder(node["class_type"], name)
        for node in workflow.values()
        for name, value in node["inputs"].items()
        if value == PLACEHOLDER
    }

    assert probed == set(FOLDERS)
    Workflow().load_workflow(workflow)


def test_model_folder_of_loader_inputs():
    assert model_folder("CheckpointLoaderSimple", "ckpt_name") == "checkpoints"
    assert model_folder("UNETLoader", "unet_name") == "diffusion_models"
    assert model_folder("KSampler", "seed") is None
