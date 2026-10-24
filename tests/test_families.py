from agent_comfy_ui_api_workflow._impl.families import (
    family_nodes,
    find_models,
    unclassified_files,
    upscale_factor,
)
from agent_comfy_ui_api_workflow._impl.inventory import ServerInventory
from conftest import (
    ACE_LM,
    SDXL,
    WAN21_VAE,
    WAN_14B_T2V,
    WAN_MODEL,
    WAN_TEXT_ENCODER,
    WAN_VAE,
    WAV2VEC2,
)


def test_finds_sdxl_and_wan_on_the_current_server(inventory):
    models = {model.id: model for model in find_models(inventory)}

    assert set(models) == {"sd_xl_base_1.0", "wan2.2_ti2v_5B_fp16"}
    sdxl = models["sd_xl_base_1.0"]
    assert sdxl.family == "sdxl"
    assert sdxl.files == {"checkpoint": SDXL}
    assert {"text_to_image", "image_to_image", "inpaint", "outpaint"} <= sdxl.capabilities
    wan = models["wan2.2_ti2v_5B_fp16"]
    assert wan.family == "wan22_5b"
    assert wan.files == {"unet": WAN_MODEL, "text_encoder": WAN_TEXT_ENCODER, "vae": WAN_VAE}
    assert wan.capabilities == {"text_to_video", "image_to_video"}


def test_wan_needs_its_text_encoder_and_vae():
    inventory = ServerInventory({"diffusion_models": [WAN_MODEL], "vae": [WAN_VAE]})

    assert find_models(inventory) == ()


def test_several_sdxl_checkpoints_are_several_choices():
    inventory = ServerInventory(
        {"checkpoints": [SDXL, "juggernautXL_v9.safetensors", "mystery_model.ckpt"]}
    )

    assert [model.id for model in find_models(inventory)] == ["juggernautXL_v9", "sd_xl_base_1.0"]


def test_files_no_family_uses_are_reported():
    inventory = ServerInventory(
        {"checkpoints": [SDXL, "mystery_model.ckpt"], "diffusion_models": ["flux1-dev.safetensors"]}
    )

    assert unclassified_files(inventory, find_models(inventory)) == {
        "checkpoints": ("mystery_model.ckpt",),
        "diffusion_models": ("flux1-dev.safetensors",),
    }


def test_families_bring_their_own_nodes(inventory):
    groups = family_nodes(find_models(inventory))

    assert groups == {"Wan 2.2 video": ("ModelSamplingSD3", "Wan22ImageToVideoLatent")}


def test_finds_every_family_of_the_server_of_2026_10_23(server_inventory):
    models = {model.family: model for model in find_models(server_inventory)}

    assert set(models) == {
        "sdxl",
        "wan22_5b",
        "wan22_t2v_14b",
        "wan22_i2v_14b",
        "minimax_h3",
        "wan22_s2v",
        "ace_step_15",
        "upscaler",
    }
    assert unclassified_files(server_inventory, models.values()) == {}
    t2v = models["wan22_t2v_14b"]
    assert t2v.id == "wan2.2_t2v_14B_fp8_scaled"
    assert t2v.capabilities == {"text_to_video"}
    assert t2v.files["vae"] == WAN21_VAE
    assert t2v.files["fast_low_noise"] == "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    assert models["wan22_i2v_14b"].capabilities == {"image_to_video"}
    assert models["minimax_h3"].capabilities == {"text_to_video", "image_to_video"}
    assert models["wan22_s2v"].files["audio_encoder"] == WAV2VEC2
    assert models["ace_step_15"].files["lm"] == ACE_LM
    assert models["upscaler"].capabilities == {"upscale_with_model"}


def test_wan_14b_needs_both_experts_and_works_without_its_loras():
    parts = {"text_encoders": [WAN_TEXT_ENCODER], "vae": [WAN21_VAE]}
    low = WAN_14B_T2V.replace("high_noise", "low_noise")

    assert find_models(ServerInventory({**parts, "diffusion_models": [WAN_14B_T2V]})) == ()
    (model,) = find_models(ServerInventory({**parts, "diffusion_models": [WAN_14B_T2V, low]}))
    assert model.files["low_noise"] == low
    assert "fast_high_noise" not in model.files


def test_sound_to_video_needs_the_audio_encoder(server_inventory):
    files = {folder: list(server_inventory.folder(folder)) for folder in server_inventory.files}
    files["audio_encoders"] = []

    families = {model.family for model in find_models(ServerInventory(files))}

    assert "wan22_s2v" not in families
    assert "wan22_t2v_14b" in families


def test_the_upscale_factor_comes_from_the_file_name():
    assert upscale_factor("RealESRGAN_x4plus.safetensors") == 4
    assert upscale_factor("RealESRGAN_x2.pth") == 2
    assert upscale_factor("4x-UltraSharp.pth") == 4
    assert upscale_factor("2xNomosUni_compact.safetensors") == 2
    assert upscale_factor("UltraSharp.pth") == 4
