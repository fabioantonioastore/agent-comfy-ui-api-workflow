"""The model families the generation pipelines know, found in the server's inventory."""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from .inventory import ServerInventory

IMAGE_CAPABILITIES = frozenset({"text_to_image", "image_to_image", "inpaint", "outpaint"})
VIDEO_CAPABILITIES = frozenset({"text_to_video", "image_to_video"})


@dataclass(frozen=True)
class ModelChoice:
    """A model the generation pipelines can use: its family and the server files it needs."""

    id: str
    family: str
    files: Mapping[str, str]
    capabilities: frozenset[str]
    description: str

    def file(self, role: str) -> str:
        return self.files[role]


def _stem(filename: str) -> str:
    name = PurePath(filename).name
    return name.rsplit(".", 1)[0] if "." in name else name


def _first(inventory: ServerInventory, folder: str, pattern: re.Pattern[str]) -> str | None:
    return next((name for name in inventory.folder(folder) if pattern.search(name)), None)


def _low_noise(names: Sequence[str], high: str) -> str | None:
    # Wan 2.2 A14B comes as two experts whose files differ only in high/low noise.
    low = re.sub("high_noise", "low_noise", high, flags=re.IGNORECASE)
    return low if low != high and low in names else None


_SDXL_CHECKPOINT = re.compile(
    r"sd_?xl|sdxl|xl_base|juggernaut.*xl|realvis.*xl|dreamshaper.*xl|pony|illustrious|noobai",
    re.IGNORECASE,
)
_WAN_5B = re.compile(r"wan2\.?2.*ti2v.*5b", re.IGNORECASE)
_WAN_14B = re.compile(r"wan2\.?2_(t2v|i2v)_high_noise_14b", re.IGNORECASE)
_WAN_S2V = re.compile(r"wan2\.?2_s2v_14b", re.IGNORECASE)
_UMT5 = re.compile(r"umt5", re.IGNORECASE)
_WAN22_VAE = re.compile(r"wan2\.?2_vae", re.IGNORECASE)
_WAN21_VAE = re.compile(r"wan_?2\.?1_vae", re.IGNORECASE)
_WAV2VEC2 = re.compile(r"wav2vec2", re.IGNORECASE)
_MINIMAX_H3 = re.compile(r"minimax_h3_fl2va", re.IGNORECASE)
_MINIMAX_H3_ENCODER = re.compile(r"qwen3vl.*minimax_h3", re.IGNORECASE)
_MINIMAX_H3_VIDEO_VAE = re.compile(r"minimax_h3_video_vae", re.IGNORECASE)
_MINIMAX_H3_AUDIO_VAE = re.compile(r"minimax_h3_audio_vae", re.IGNORECASE)
_MINIMAX_H3_TURBO = re.compile(r"minimax_h3_fl2v_turbo", re.IGNORECASE)
_ACE_STEP_15 = re.compile(r"acestep_v1\.5.*turbo", re.IGNORECASE)
_ACE_STEP_15_ENCODER = re.compile(r"qwen_0\.6b_ace15", re.IGNORECASE)
_ACE_STEP_15_LM = re.compile(r"qwen_(1\.7b|4b)_ace15", re.IGNORECASE)
_ACE_STEP_15_VAE = re.compile(r"ace_1\.5_vae", re.IGNORECASE)
_UPSCALE_FACTOR = re.compile(r"(?<![a-z0-9])(\d)x|x(\d)(?!\d)", re.IGNORECASE)


def _lightx2v(task: str) -> re.Pattern[str]:
    # The 4-step LoRA of each high-noise expert; the low-noise one is named like it.
    return re.compile(rf"wan2\.?2_{task}_lightx2v.*high_noise", re.IGNORECASE)


def upscale_factor(filename: str) -> int:
    """How many times larger an upscale model makes an image, from its file name."""
    match = _UPSCALE_FACTOR.search(PurePath(filename).name)
    # Upscale models without a factor in their name are almost always 4x.
    return int(match.group(1) or match.group(2)) if match else 4


def _sdxl(inventory: ServerInventory) -> list[ModelChoice]:
    return [
        ModelChoice(
            id=_stem(name),
            family="sdxl",
            files={"checkpoint": name},
            capabilities=IMAGE_CAPABILITIES,
            description=(
                "SDXL: photos, illustrations and art at about one megapixel "
                "(1024x1024 native)."
            ),
        )
        for name in inventory.folder("checkpoints")
        if _SDXL_CHECKPOINT.search(name)
    ]


def _wan22_5b(inventory: ServerInventory) -> list[ModelChoice]:
    encoder = _first(inventory, "text_encoders", _UMT5)
    vae = _first(inventory, "vae", _WAN22_VAE)
    if encoder is None or vae is None:
        return []
    return [
        ModelChoice(
            id=_stem(unet),
            family="wan22_5b",
            files={"unet": unet, "text_encoder": encoder, "vae": vae},
            capabilities=VIDEO_CAPABILITIES,
            description=(
                "Wan 2.2 TI2V 5B: 720p video clips of up to 5 seconds at 24 fps, from a "
                "prompt or from a start image. No sound."
            ),
        )
        for unet in inventory.folder("diffusion_models")
        if _WAN_5B.search(unet)
    ]


def _wan22_14b(inventory: ServerInventory) -> list[ModelChoice]:
    encoder = _first(inventory, "text_encoders", _UMT5)
    vae = _first(inventory, "vae", _WAN21_VAE)
    if encoder is None or vae is None:
        return []
    unets, loras = inventory.folder("diffusion_models"), inventory.folder("loras")
    models: list[ModelChoice] = []
    for high in unets:
        match = _WAN_14B.search(high)
        low = _low_noise(unets, high) if match else None
        if match is None or low is None:
            continue
        task = match.group(1).lower()
        files = {"high_noise": high, "low_noise": low, "text_encoder": encoder, "vae": vae}
        fast_high = next((name for name in loras if _lightx2v(task).search(name)), None)
        fast_low = _low_noise(loras, fast_high) if fast_high else None
        if fast_high is not None and fast_low is not None:
            files |= {"fast_high_noise": fast_high, "fast_low_noise": fast_low}
        if task == "t2v":
            capability, source = "text_to_video", "from a prompt"
        else:
            capability, source = "image_to_video", "from a start image, or between a start and an end image"
        models.append(
            ModelChoice(
                id=_stem(high).replace("_high_noise", ""),
                family=f"wan22_{task}_14b",
                files=files,
                capabilities=frozenset({capability}),
                description=(
                    "Wan 2.2 A14B: more detail and better motion than the 5B, 720p clips "
                    f"of up to 5 seconds at 16 fps, {source}. No sound."
                ),
            )
        )
    return models


def _wan22_s2v(inventory: ServerInventory) -> list[ModelChoice]:
    encoder = _first(inventory, "text_encoders", _UMT5)
    vae = _first(inventory, "vae", _WAN21_VAE)
    audio_encoder = _first(inventory, "audio_encoders", _WAV2VEC2)
    if encoder is None or vae is None or audio_encoder is None:
        return []
    files = {"text_encoder": encoder, "vae": vae, "audio_encoder": audio_encoder}
    # S2V runs fast with the 4-step LoRA of the text-to-video high-noise expert.
    fast = _first(inventory, "loras", _lightx2v("t2v"))
    if fast is not None:
        files["fast"] = fast
    return [
        ModelChoice(
            id=_stem(unet),
            family="wan22_s2v",
            files={"unet": unet, **files},
            capabilities=frozenset({"sound_to_video"}),
            description=(
                "Wan 2.2 S2V 14B: animates a photo in sync with an audio, speech or song, "
                "at 16 fps; it makes no sound of its own, the video keeps the audio."
            ),
        )
        for unet in inventory.folder("diffusion_models")
        if _WAN_S2V.search(unet)
    ]


def _minimax_h3(inventory: ServerInventory) -> list[ModelChoice]:
    encoder = _first(inventory, "text_encoders", _MINIMAX_H3_ENCODER)
    video_vae = _first(inventory, "vae", _MINIMAX_H3_VIDEO_VAE)
    audio_vae = _first(inventory, "vae", _MINIMAX_H3_AUDIO_VAE)
    if encoder is None or video_vae is None or audio_vae is None:
        return []
    files = {"text_encoder": encoder, "video_vae": video_vae, "audio_vae": audio_vae}
    fast = _first(inventory, "loras", _MINIMAX_H3_TURBO)
    if fast is not None:
        files["fast"] = fast
    return [
        ModelChoice(
            id=_stem(unet),
            family="minimax_h3",
            files={"unet": unet, **files},
            capabilities=VIDEO_CAPABILITIES,
            description=(
                "MiniMax H3: video with its own stereo sound (speech, sound effects and "
                "music, as the prompt describes), 768p clips of 5 to 15 seconds at 24 fps, "
                "from a prompt, a start image, or a start and an end image."
            ),
        )
        for unet in inventory.folder("diffusion_models")
        if _MINIMAX_H3.search(unet)
    ]


def _ace_step_15(inventory: ServerInventory) -> list[ModelChoice]:
    encoder = _first(inventory, "text_encoders", _ACE_STEP_15_ENCODER)
    # The language model that writes the audio codes before the music is sampled.
    lm = _first(inventory, "text_encoders", _ACE_STEP_15_LM)
    vae = _first(inventory, "vae", _ACE_STEP_15_VAE)
    if encoder is None or lm is None or vae is None:
        return []
    return [
        ModelChoice(
            id=_stem(unet),
            family="ace_step_15",
            files={"unet": unet, "text_encoder": encoder, "lm": lm, "vae": vae},
            capabilities=frozenset({"text_to_audio"}),
            description=(
                "ACE-Step 1.5 XL turbo: music, songs with sung lyrics in many languages "
                "or instrumentals, from a style description; up to 10 minutes."
            ),
        )
        for unet in inventory.folder("diffusion_models")
        if _ACE_STEP_15.search(unet)
    ]


def _upscalers(inventory: ServerInventory) -> list[ModelChoice]:
    return [
        ModelChoice(
            id=_stem(name),
            family="upscaler",
            files={"upscaler": name},
            capabilities=frozenset({"upscale_with_model"}),
            description=(
                f"Upscale model: enlarges images {upscale_factor(name)}x and restores "
                "sharp detail, without changing what they show."
            ),
        )
        for name in inventory.folder("upscale_models")
    ]


_FAMILIES: tuple[Callable[[ServerInventory], list[ModelChoice]], ...] = (
    _sdxl,
    _wan22_5b,
    _wan22_14b,
    _minimax_h3,
    _wan22_s2v,
    _ace_step_15,
    _upscalers,
)
# Nodes a family needs beyond the essential ones, offered only when it is on the server.
_FAMILY_NODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "wan22_5b": ("Wan 2.2 video", ("ModelSamplingSD3", "Wan22ImageToVideoLatent")),
    "wan22_t2v_14b": ("Wan 2.2 14B text to video", ("ModelSamplingSD3", "EmptyHunyuanLatentVideo")),
    "wan22_i2v_14b": (
        "Wan 2.2 14B image to video",
        ("ModelSamplingSD3", "WanImageToVideo", "WanFirstLastFrameToVideo"),
    ),
    "wan22_s2v": (
        "Wan 2.2 sound to video",
        ("AudioEncoderLoader", "AudioEncoderEncode", "WanSoundImageToVideo", "WanSoundImageToVideoExtend"),
    ),
    "minimax_h3": (
        "MiniMax H3 video with sound",
        (
            "MiniMaxH3ImageToVideo",
            "RandomNoise",
            "KSamplerSelect",
            "BasicScheduler",
            "BasicGuider",
            "SamplerCustomAdvanced",
            "VAEDecodeAudio",
        ),
    ),
    "ace_step_15": (
        "ACE-Step 1.5 music",
        ("TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio", "ModelSamplingAuraFlow", "VAEDecodeAudio"),
    ),
}
# Folders whose files are whole models; the other folders hold their parts.
_MODEL_FOLDERS = ("checkpoints", "diffusion_models")


def find_models(inventory: ServerInventory) -> tuple[ModelChoice, ...]:
    models: list[ModelChoice] = []
    for detect in _FAMILIES:
        models.extend(detect(inventory))
    return tuple(models)


def unclassified_files(
    inventory: ServerInventory, models: Iterable[ModelChoice]
) -> dict[str, tuple[str, ...]]:
    """Model files no family uses: the server has them, the pipelines cannot use them."""
    used = {name for model in models for name in model.files.values()}
    found: dict[str, tuple[str, ...]] = {}
    for folder in _MODEL_FOLDERS:
        left = tuple(name for name in inventory.folder(folder) if name not in used)
        if left:
            found[folder] = left
    return found


def family_nodes(models: Iterable[ModelChoice]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    for model in models:
        if model.family in _FAMILY_NODES:
            group, names = _FAMILY_NODES[model.family]
            groups[group] = names
    return groups
