import pytest

from agent_comfy_ui_api_workflow._impl.catalog import get_catalog
from agent_comfy_ui_api_workflow._impl.inventory import ServerInventory
from agent_comfy_ui_api_workflow._impl.session import WorkflowSession

# The server as it was on 2026-10-16.
SDXL = "sd_xl_base_1.0.safetensors"
WAN_MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan2.2_vae.safetensors"

# The server since 2026-10-23 (image fabioninja/comfyui-api:v3).
WAN_14B_T2V = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
WAN_14B_T2V_FAST = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
WAN_14B_I2V = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_S2V = "wan2.2_s2v_14B_fp8_scaled.safetensors"
WAN21_VAE = "wan_2.1_vae.safetensors"
WAV2VEC2 = "wav2vec2_large_english_fp16.safetensors"
MINIMAX_H3 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
ACE_STEP = "acestep_v1.5_xl_turbo_bf16.safetensors"
ACE_ENCODER = "qwen_0.6b_ace15.safetensors"
ACE_LM = "qwen_4b_ace15.safetensors"
REALESRGAN = "RealESRGAN_x4plus.safetensors"
SERVER_FILES = {
    "audio_encoders": [WAV2VEC2],
    "checkpoints": [SDXL],
    "diffusion_models": [
        ACE_STEP,
        MINIMAX_H3,
        WAN_14B_I2V,
        WAN_14B_I2V.replace("high_noise", "low_noise"),
        WAN_S2V,
        WAN_14B_T2V,
        WAN_14B_T2V.replace("high_noise", "low_noise"),
        WAN_MODEL,
    ],
    "loras": [
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        WAN_14B_T2V_FAST,
        WAN_14B_T2V_FAST.replace("high_noise", "low_noise"),
    ],
    "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", ACE_ENCODER, ACE_LM, WAN_TEXT_ENCODER],
    "upscale_models": [REALESRGAN],
    "vae": [
        "ace_1.5_vae.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
        "minimax_h3_video_vae_int8_convrot.safetensors",
        "pixel_space",
        WAN_VAE,
        WAN21_VAE,
    ],
}


@pytest.fixture
def server_inventory():
    return ServerInventory(SERVER_FILES)


@pytest.fixture
def inventory():
    return ServerInventory(
        {
            "checkpoints": [SDXL],
            "diffusion_models": [WAN_MODEL],
            "text_encoders": [WAN_TEXT_ENCODER],
            "vae": ["pixel_space", WAN_VAE],
            "loras": [],
            "upscale_models": [],
        }
    )


@pytest.fixture
def session(inventory):
    return WorkflowSession(get_catalog(), inventory)


def pytest_addoption(parser):
    parser.addoption(
        "--runpod",
        action="store_true",
        help="send real jobs to the RunPod ComfyUI endpoint in .env (paid GPU time)",
    )
    parser.addoption(
        "--anthropic",
        action="store_true",
        help="call the Claude API with the ANTHROPIC_API_KEY in .env (paid tokens)",
    )


def pytest_collection_modifyitems(config, items):
    for marker, option in (("runpod", "--runpod"), ("anthropic", "--anthropic")):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=f"paid; run with {option}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
