"""Pipelines that generate images, videos and audio with a model of the server."""

import math
import random
from dataclasses import dataclass
from typing import Literal

from comfy_ui_api_workflow.nodes.audio import TrimAudioDuration
from comfy_ui_api_workflow.nodes.image import GetImageSize, LoadImage, LoadImageMask
from comfy_ui_api_workflow.nodes.image.batch import ImageFromBatch
from comfy_ui_api_workflow.nodes.image.compositing import ImageCompositeMasked
from comfy_ui_api_workflow.nodes.image.mask import GrowMask, MaskComposite, SolidMask
from comfy_ui_api_workflow.nodes.image.transform import ImagePadForOutpaint
from comfy_ui_api_workflow.nodes.image.upscaling import (
    ImageScale,
    ImageScaleBy,
    ImageScaleToTotalPixels,
    ImageUpscaleWithModel,
)
from comfy_ui_api_workflow.nodes.model.conditioning import AudioEncoderEncode, CLIPTextEncode
from comfy_ui_api_workflow.nodes.model.conditioning.ace import TextEncodeAceStepAudio1_5
from comfy_ui_api_workflow.nodes.model.conditioning.minimax import MiniMaxH3ImageToVideo
from comfy_ui_api_workflow.nodes.model.conditioning.transform import ConditioningZeroOut
from comfy_ui_api_workflow.nodes.model.conditioning.wan import (
    Wan22ImageToVideoLatent,
    WanFirstLastFrameToVideo,
    WanImageToVideo,
)
from comfy_ui_api_workflow.nodes.model.conditioning.wan.sound import (
    WanSoundImageToVideo,
    WanSoundImageToVideoExtend,
)
from comfy_ui_api_workflow.nodes.model.latent import (
    EmptyLatentImage,
    LatentUpscaleBy,
    SetLatentNoiseMask,
    VAEDecode,
    VAEDecodeAudio,
    VAEEncode,
    VAEEncodeForInpaint,
)
from comfy_ui_api_workflow.nodes.model.latent.ace import EmptyAceStep1_5LatentAudio
from comfy_ui_api_workflow.nodes.model.latent.advanced import LatentConcat, LatentCut
from comfy_ui_api_workflow.nodes.model.latent.hunyuan_video import EmptyHunyuanLatentVideo
from comfy_ui_api_workflow.nodes.model.loaders import (
    AudioEncoderLoader,
    CheckpointLoaderSimple,
    CLIPLoader,
    DualCLIPLoader,
    LoraLoaderModelOnly,
    UNETLoader,
    UpscaleModelLoader,
    VAELoader,
)
from comfy_ui_api_workflow.nodes.model.patch import ModelSamplingAuraFlow
from comfy_ui_api_workflow.nodes.model.patch.stable_diffusion import ModelSamplingSD3
from comfy_ui_api_workflow.nodes.model.sampling import KSampler, KSamplerAdvanced
from comfy_ui_api_workflow.nodes.model.sampling.custom import SamplerCustomAdvanced
from comfy_ui_api_workflow.nodes.model.sampling.guiders import BasicGuider
from comfy_ui_api_workflow.nodes.model.sampling.noise import RandomNoise
from comfy_ui_api_workflow.nodes.model.sampling.samplers import KSamplerSelect
from comfy_ui_api_workflow.nodes.model.sampling.schedulers import BasicScheduler
from comfy_ui_api_workflow.nodes.video import CreateVideo

from ..families import ModelChoice, upscale_factor
from .base import AgentPipeline, AudioFormat, Link, VideoFormat, is_link

AspectRatio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
Orientation = Literal["landscape", "portrait", "square"]
TimeSignature = Literal["2", "3", "4", "6"]

# SDXL was trained near one megapixel; these are its buckets for each aspect ratio.
_SDXL_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "21:9": (1536, 640),
}
SDXL_NEGATIVE = "blurry, low quality, deformed, watermark, text"
_SDXL_STEPS = 25
_SDXL_CFG = 7.0

_WAN_SIZES: dict[str, tuple[int, int]] = {
    "landscape": (1280, 704),
    "portrait": (704, 1280),
    "square": (960, 960),
}
# Wan's own negative prompt, which the model was tuned with.
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
_WAN_STEPS = 20
_WAN_CFG = 5.0
_WAN_SHIFT = 8.0

# Wan 2.2 A14B, as in ComfyUI's templates: the high-noise expert samples the first
# half of the steps and the low-noise one the rest. With the LightX2V LoRAs, 4 steps.
_WAN_14B_SIZES: dict[str, tuple[int, int]] = {
    "landscape": (1280, 720),
    "portrait": (720, 1280),
    "square": (960, 960),
}
_WAN_14B_STEPS, _WAN_14B_FAST_STEPS = 20, 4
_WAN_14B_CFG = 3.5
_WAN_14B_SHIFT = 5.0

# MiniMax H3 renders a 768 pixel short edge at 24 fps; its turbo LoRA takes 8 steps.
_MINIMAX_SIZES: dict[str, tuple[int, int]] = {
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "square": (768, 768),
}
_MINIMAX_STEPS, _MINIMAX_FAST_STEPS = 20, 8

# Wan 2.2 S2V makes the video in parts of 77 frames at 16 fps, each following its
# stretch of the audio.
_S2V_SIZES: dict[str, tuple[int, int]] = {
    "landscape": (832, 480),
    "portrait": (480, 832),
    "square": (640, 640),
}
_S2V_FPS = 16
_S2V_PART = 77
_S2V_MAX_SECONDS = 30.0
_S2V_STEPS, _S2V_FAST_STEPS = 20, 4
_S2V_CFG = 6.0
_S2V_SHIFT = 8.0

# ACE-Step 1.5 XL turbo, as in ComfyUI's template.
_ACE_STEPS = 8
_ACE_SHIFT = 3.0
_ACE_MAX_SECONDS = 600.0
_INSTRUMENTAL = "[instrumental]"

_MODEL = "The model to use; the default is the first one."
_SEED = "Random seed; empty picks a new one. Use the same seed to repeat a result."
_STEPS = "Sampling steps; empty uses the model's default."
_CFG = "How strictly the result follows the prompt; empty uses the model's default."
_SAVE = "Add the save node. Set false when another step uses the output."
_FAST = (
    "Use the model's speed LoRA when the server has it (Wan 2.2 14B and S2V: 4 steps, "
    "MiniMax H3: 8): several times faster, a little less detail. Set false only when "
    "the user asks for the best quality."
)


@dataclass(frozen=True)
class _VideoSpec:
    fps: int
    max_frames: int
    sizes: dict[str, tuple[int, int]]
    # Sizes are multiples of this, and clips have block * n + extra frames.
    multiple: int
    block: int = 4
    extra: int = 1


_VIDEO_SPECS: dict[str, _VideoSpec] = {
    "wan22_5b": _VideoSpec(24, 121, _WAN_SIZES, 32),
    "wan22_t2v_14b": _VideoSpec(16, 81, _WAN_14B_SIZES, 16),
    "wan22_i2v_14b": _VideoSpec(16, 81, _WAN_14B_SIZES, 16),
    "minimax_h3": _VideoSpec(24, 362, _MINIMAX_SIZES, 32, block=17, extra=5),
}
_TEXT_TO_VIDEO = ("wan22_5b", "wan22_t2v_14b", "minimax_h3")
_IMAGE_TO_VIDEO = ("wan22_5b", "wan22_i2v_14b", "minimax_h3")


def _seed(seed: int | None) -> int:
    return random.randrange(2**32) if seed is None else seed


def _multiple(value: int, step: int, *, up: bool = False) -> int:
    count = math.ceil(value / step) if up else round(value / step)
    return max(step, count * step)


def _require(model: ModelChoice, families: tuple[str, ...], pipeline: str) -> None:
    if model.family not in families:
        raise ValueError(f"{pipeline} cannot use the {model.family} model {model.id!r}")


def _text(pipeline: AgentPipeline, clip: Link, text: str) -> Link:
    return pipeline.add_node(CLIPTextEncode, text=text, clip=clip).output["CONDITIONING"]


def _sample(
    pipeline: AgentPipeline,
    model: Link,
    positive: Link,
    negative: Link,
    latent: Link,
    *,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float = 1.0,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
) -> Link:
    return pipeline.add_node(
        KSampler,
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        denoise=denoise,
    ).output["LATENT"]


def _sdxl(pipeline: AgentPipeline, model: ModelChoice) -> tuple[Link, Link, Link]:
    loader = pipeline.add_node(CheckpointLoaderSimple, ckpt_name=model.file("checkpoint"))
    return loader.output["MODEL"], loader.output["CLIP"], loader.output["VAE"]


def _with_lora(pipeline: AgentPipeline, model: Link, lora: str) -> Link:
    return pipeline.add_node(
        LoraLoaderModelOnly, model=model, lora_name=lora, strength_model=1.0
    ).output["MODEL"]


def _fit(pipeline: AgentPipeline, image: str | Link, width: int, height: int) -> Link:
    return pipeline.add_node(
        ImageScale,
        image=pipeline.image(image),
        upscale_method="lanczos",
        width=width,
        height=height,
        crop="center",
    ).output["IMAGE"]


class TextToImage(AgentPipeline):
    NAME = "text_to_image"
    CAPABILITY = "text_to_image"
    DESCRIPTION = (
        "Generates images from a prompt and saves them. Use it for any request to create "
        "an image that does not start from an image of the user. Output: images (IMAGE)."
    )
    PARAMETERS = {
        "prompt": "What the image shows, in English, as a detailed visual description.",
        "model": _MODEL,
        "negative_prompt": "What the image must not show.",
        "aspect_ratio": "Shape of the image; the size is the model's native size for it.",
        "width": "Exact width in pixels instead of the aspect ratio; rounded to a multiple of 8.",
        "height": "Exact height in pixels instead of the aspect ratio; rounded to a multiple of 8.",
        "batch_size": "How many images to generate at once.",
        "steps": _STEPS,
        "cfg": _CFG,
        "seed": _SEED,
        "upscale": "Above 1 (up to 2), a second pass makes the image that much larger, with more detail.",
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file names.",
    }
    COVERS = ("EmptyLatentImage", "KSampler", "LatentUpscaleBy")

    def build(
        self,
        prompt: str,
        model: ModelChoice,
        negative_prompt: str = SDXL_NEGATIVE,
        aspect_ratio: AspectRatio = "1:1",
        width: int | None = None,
        height: int | None = None,
        batch_size: int = 1,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        upscale: float = 1.0,
        save: bool = True,
        filename_prefix: str = "image",
    ):
        _require(model, ("sdxl",), self.NAME)
        size_width, size_height = _SDXL_SIZES[aspect_ratio]
        if width is not None:
            size_width = _multiple(width, 8)
        if height is not None:
            size_height = _multiple(height, 8)
        unet, clip, vae = _sdxl(self, model)
        positive, negative = _text(self, clip, prompt), _text(self, clip, negative_prompt)
        latent = self.add_node(
            EmptyLatentImage, width=size_width, height=size_height, batch_size=batch_size
        ).output["LATENT"]
        settings = {"seed": _seed(seed), "cfg": _SDXL_CFG if cfg is None else cfg}
        steps = steps or _SDXL_STEPS
        sampled = _sample(self, unet, positive, negative, latent, steps=steps, **settings)
        if upscale > 1.0:
            bigger = self.add_node(
                LatentUpscaleBy, samples=sampled, upscale_method="bislerp", scale_by=upscale
            ).output["LATENT"]
            sampled = _sample(
                self, unet, positive, negative, bigger, steps=max(8, steps // 2), denoise=0.45, **settings
            )
        images = self.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]
        if save:
            self.save_image(images, filename_prefix)
        return {"images": images}


class ImageToImage(AgentPipeline):
    NAME = "image_to_image"
    CAPABILITY = "image_to_image"
    DESCRIPTION = (
        "Redraws an image after a prompt, keeping its composition: a new style, a new "
        "look, the same scene. The image is scaled to about one megapixel first. Output: "
        "images (IMAGE)."
    )
    PARAMETERS = {
        "image": "An image file the user attached, or an IMAGE output.",
        "prompt": "What the new image shows, in English: describe the whole scene, not only the change.",
        "model": _MODEL,
        "strength": "How much changes, from 0.1 (almost nothing) to 1 (only the layout stays); 0.5 to 0.7 restyles.",
        "negative_prompt": "What the image must not show.",
        "steps": _STEPS,
        "cfg": _CFG,
        "seed": _SEED,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file names.",
    }
    FILES = {"image": "image"}
    COVERS = ("VAEEncode",)

    def build(
        self,
        image: str | Link,
        prompt: str,
        model: ModelChoice,
        strength: float = 0.6,
        negative_prompt: str = SDXL_NEGATIVE,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        save: bool = True,
        filename_prefix: str = "image",
    ):
        _require(model, ("sdxl",), self.NAME)
        unet, clip, vae = _sdxl(self, model)
        scaled = self.add_node(
            ImageScaleToTotalPixels, image=self.image(image), upscale_method="lanczos", megapixels=1.0
        ).output["IMAGE"]
        latent = self.add_node(VAEEncode, pixels=scaled, vae=vae).output["LATENT"]
        sampled = _sample(
            self,
            unet,
            _text(self, clip, prompt),
            _text(self, clip, negative_prompt),
            latent,
            seed=_seed(seed),
            steps=steps or _SDXL_STEPS,
            cfg=_SDXL_CFG if cfg is None else cfg,
            denoise=strength,
        )
        images = self.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]
        if save:
            self.save_image(images, filename_prefix)
        return {"images": images}


class Inpaint(AgentPipeline):
    NAME = "inpaint"
    CAPABILITY = "inpaint"
    DESCRIPTION = (
        "Repaints one area of an image after a prompt and keeps the rest exactly as it "
        "was: replace an object, change a face, fill a hole. The area is a mask image, a "
        "rectangle, or else the transparent pixels of the image. Works at the image's "
        "own size, best up to about 1.5 megapixels. Output: images (IMAGE)."
    )
    PARAMETERS = {
        "image": "An image file the user attached, or an IMAGE output.",
        "prompt": "What the repainted area shows, in English, in the context of the whole image.",
        "model": _MODEL,
        "mask": "A black and white image the user attached, white where to repaint; or a MASK output.",
        "region": "Instead of a mask: the rectangle to repaint as [x, y, width, height] in pixels of the image.",
        "strength": "1 replaces the area entirely; below 1 (like 0.5) only changes what is there.",
        "grow_mask": "Pixels added around the area so the repaint blends in.",
        "negative_prompt": "What the area must not show.",
        "steps": _STEPS,
        "cfg": _CFG,
        "seed": _SEED,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file names.",
    }
    FILES = {"image": "image", "mask": "image"}
    COVERS = ("VAEEncodeForInpaint", "SetLatentNoiseMask", "InpaintModelConditioning")

    def build(
        self,
        image: str | Link,
        prompt: str,
        model: ModelChoice,
        mask: str | Link | None = None,
        region: list[int] | None = None,
        strength: float = 1.0,
        grow_mask: int = 8,
        negative_prompt: str = SDXL_NEGATIVE,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        save: bool = True,
        filename_prefix: str = "image",
    ):
        _require(model, ("sdxl",), self.NAME)
        if mask is not None and region is not None:
            raise ValueError("give either a mask or a region, not both")
        if is_link(image):
            pixels, alpha = (image[0], image[1]), None
        else:
            loaded = self.add_node(LoadImage, image=image)
            pixels, alpha = loaded.output["IMAGE"], loaded.output["MASK"]
        if mask is not None:
            area = (mask[0], mask[1]) if is_link(mask) else self.add_node(
                LoadImageMask, image=mask, channel="red"
            ).output["MASK"]
        elif region is not None:
            if len(region) != 4:
                raise ValueError("region is [x, y, width, height] in pixels")
            x, y, width, height = region
            size = self.add_node(GetImageSize, image=pixels).output
            empty = self.add_node(SolidMask, value=0.0, width=size["width"], height=size["height"])
            box = self.add_node(SolidMask, value=1.0, width=width, height=height)
            area = self.add_node(
                MaskComposite,
                destination=empty.output["MASK"],
                source=box.output["MASK"],
                operation="add",
                x=x,
                y=y,
            ).output["MASK"]
        elif alpha is not None:
            area = alpha
        else:
            raise ValueError("give a mask or a region: an IMAGE output has no transparent pixels")

        unet, clip, vae = _sdxl(self, model)
        positive, negative = _text(self, clip, prompt), _text(self, clip, negative_prompt)
        settings = {
            "seed": _seed(seed),
            "steps": steps or _SDXL_STEPS,
            "cfg": _SDXL_CFG if cfg is None else cfg,
        }
        if strength >= 1.0:
            latent = self.add_node(
                VAEEncodeForInpaint, pixels=pixels, vae=vae, mask=area, grow_mask_by=grow_mask
            ).output["LATENT"]
            sampled = _sample(self, unet, positive, negative, latent, **settings)
        else:
            grown = area
            if grow_mask:
                grown = self.add_node(
                    GrowMask, mask=area, expand=grow_mask, tapered_corners=True
                ).output["MASK"]
            encoded = self.add_node(VAEEncode, pixels=pixels, vae=vae).output["LATENT"]
            latent = self.add_node(SetLatentNoiseMask, samples=encoded, mask=grown).output["LATENT"]
            sampled = _sample(self, unet, positive, negative, latent, denoise=strength, **settings)
        painted = self.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]
        # Pasting back through the mask keeps every pixel outside the area exact.
        final = self.add_node(
            ImageCompositeMasked, destination=pixels, source=painted, mask=area
        ).output["IMAGE"]
        if save:
            self.save_image(final, filename_prefix)
        return {"images": final}


class Outpaint(AgentPipeline):
    NAME = "outpaint"
    CAPABILITY = "outpaint"
    DESCRIPTION = (
        "Extends an image beyond its borders and paints the new area after a prompt, "
        "keeping the original pixels. Output: images (IMAGE)."
    )
    PARAMETERS = {
        "image": "An image file the user attached, or an IMAGE output.",
        "prompt": "What the whole extended image shows, in English.",
        "model": _MODEL,
        "left": "Pixels to add on the left; rounded up to a multiple of 8.",
        "top": "Pixels to add on top; rounded up to a multiple of 8.",
        "right": "Pixels to add on the right; rounded up to a multiple of 8.",
        "bottom": "Pixels to add at the bottom; rounded up to a multiple of 8.",
        "feathering": "Pixels over which the new area blends into the old.",
        "negative_prompt": "What the image must not show.",
        "steps": _STEPS,
        "cfg": _CFG,
        "seed": _SEED,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file names.",
    }
    FILES = {"image": "image"}
    COVERS = ("ImagePadForOutpaint",)

    def build(
        self,
        image: str | Link,
        prompt: str,
        model: ModelChoice,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
        feathering: int = 40,
        negative_prompt: str = SDXL_NEGATIVE,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        save: bool = True,
        filename_prefix: str = "image",
    ):
        _require(model, ("sdxl",), self.NAME)
        sides = [_multiple(side, 8, up=True) if side > 0 else side for side in (left, top, right, bottom)]
        if not any(sides):
            raise ValueError("give at least one side to extend, in pixels")
        unet, clip, vae = _sdxl(self, model)
        padded = self.add_node(
            ImagePadForOutpaint,
            image=self.image(image),
            left=sides[0],
            top=sides[1],
            right=sides[2],
            bottom=sides[3],
            feathering=feathering,
        ).output
        latent = self.add_node(
            VAEEncodeForInpaint, pixels=padded["IMAGE"], vae=vae, mask=padded["MASK"], grow_mask_by=16
        ).output["LATENT"]
        sampled = _sample(
            self,
            unet,
            _text(self, clip, prompt),
            _text(self, clip, negative_prompt),
            latent,
            seed=_seed(seed),
            steps=steps or _SDXL_STEPS,
            cfg=_SDXL_CFG if cfg is None else cfg,
        )
        painted = self.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]
        final = self.add_node(
            ImageCompositeMasked, destination=padded["IMAGE"], source=painted, mask=padded["MASK"]
        ).output["IMAGE"]
        if save:
            self.save_image(final, filename_prefix)
        return {"images": final}


def _frames(duration: float, fps: int, spec: _VideoSpec) -> int:
    blocks = max(1, round((duration * fps - spec.extra) / spec.block))
    length = spec.block * blocks + spec.extra
    if length > spec.max_frames:
        longest = (spec.max_frames - spec.extra) / fps
        raise ValueError(
            f"the model makes at most {spec.max_frames} frames ({longest:g} s at {fps} fps); "
            "for a longer video make several clips and join them with concatenate_videos"
        )
    return length


def _video_size(
    spec: _VideoSpec, orientation: str, width: int | None, height: int | None
) -> tuple[int, int]:
    size_width, size_height = spec.sizes[orientation]
    if width is not None:
        size_width = _multiple(width, spec.multiple)
    if height is not None:
        size_height = _multiple(height, spec.multiple)
    return size_width, size_height


def _wan_frames(
    pipeline: AgentPipeline,
    model: ModelChoice,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int | None,
    cfg: float | None,
    seed: int | None,
    start: Link | None = None,
) -> Link:
    unet = pipeline.add_node(UNETLoader, unet_name=model.file("unet"), weight_dtype="default")
    patched = pipeline.add_node(ModelSamplingSD3, model=unet.output["MODEL"], shift=_WAN_SHIFT)
    clip = pipeline.add_node(
        CLIPLoader, clip_name=model.file("text_encoder"), type="wan", device="default"
    ).output["CLIP"]
    vae = pipeline.add_node(VAELoader, vae_name=model.file("vae")).output["VAE"]
    start_image = {} if start is None else {"start_image": start}
    latent = pipeline.add_node(
        Wan22ImageToVideoLatent,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=1,
        **start_image,
    ).output["LATENT"]
    sampled = _sample(
        pipeline,
        patched.output["MODEL"],
        _text(pipeline, clip, prompt),
        _text(pipeline, clip, negative_prompt),
        latent,
        seed=_seed(seed),
        steps=steps or _WAN_STEPS,
        cfg=_WAN_CFG if cfg is None else cfg,
        sampler="uni_pc",
        scheduler="simple",
    )
    return pipeline.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]


def _wan14b_expert(pipeline: AgentPipeline, model: ModelChoice, expert: str, fast: bool) -> Link:
    unet = pipeline.add_node(UNETLoader, unet_name=model.file(expert), weight_dtype="default").output["MODEL"]
    if fast:
        unet = _with_lora(pipeline, unet, model.file(f"fast_{expert}"))
    return pipeline.add_node(ModelSamplingSD3, model=unet, shift=_WAN_14B_SHIFT).output["MODEL"]


def _wan14b_frames(
    pipeline: AgentPipeline,
    model: ModelChoice,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int | None,
    cfg: float | None,
    seed: int | None,
    fast: bool,
    start: Link | None = None,
    end: Link | None = None,
) -> Link:
    fast = fast and "fast_high_noise" in model.files
    steps = steps or (_WAN_14B_FAST_STEPS if fast else _WAN_14B_STEPS)
    cfg = (1.0 if fast else _WAN_14B_CFG) if cfg is None else cfg
    high = _wan14b_expert(pipeline, model, "high_noise", fast)
    low = _wan14b_expert(pipeline, model, "low_noise", fast)
    clip = pipeline.add_node(
        CLIPLoader, clip_name=model.file("text_encoder"), type="wan", device="default"
    ).output["CLIP"]
    vae = pipeline.add_node(VAELoader, vae_name=model.file("vae")).output["VAE"]
    positive, negative = _text(pipeline, clip, prompt), _text(pipeline, clip, negative_prompt)
    size = {"width": width, "height": height, "length": length, "batch_size": 1}
    if start is None:
        latent = pipeline.add_node(EmptyHunyuanLatentVideo, **size).output["LATENT"]
    else:
        if end is None:
            node_type, images = WanImageToVideo, {"start_image": start}
        else:
            node_type, images = WanFirstLastFrameToVideo, {"start_image": start, "end_image": end}
        conditioned = pipeline.add_node(
            node_type, positive=positive, negative=negative, vae=vae, **size, **images
        ).output
        positive, negative, latent = conditioned["positive"], conditioned["negative"], conditioned["latent"]
    split = steps // 2
    shared = {
        "noise_seed": _seed(seed),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": positive,
        "negative": negative,
    }
    noisy = pipeline.add_node(
        KSamplerAdvanced,
        model=high,
        add_noise="enable",
        latent_image=latent,
        start_at_step=0,
        end_at_step=split,
        return_with_leftover_noise="enable",
        **shared,
    ).output["LATENT"]
    sampled = pipeline.add_node(
        KSamplerAdvanced,
        model=low,
        add_noise="disable",
        latent_image=noisy,
        start_at_step=split,
        end_at_step=steps,
        return_with_leftover_noise="disable",
        **shared,
    ).output["LATENT"]
    return pipeline.add_node(VAEDecode, samples=sampled, vae=vae).output["IMAGE"]


def _minimax_clip(
    pipeline: AgentPipeline,
    model: ModelChoice,
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int | None,
    seed: int | None,
    fast: bool,
    with_sound: bool,
    start: Link | None = None,
    end: Link | None = None,
) -> tuple[Link, Link | None]:
    fast = fast and "fast" in model.files
    unet = pipeline.add_node(UNETLoader, unet_name=model.file("unet"), weight_dtype="default").output["MODEL"]
    if fast:
        unet = _with_lora(pipeline, unet, model.file("fast"))
    clip = pipeline.add_node(
        CLIPLoader, clip_name=model.file("text_encoder"), type="minimax", device="default"
    ).output["CLIP"]
    video_vae = pipeline.add_node(VAELoader, vae_name=model.file("video_vae")).output["VAE"]
    keyframes = {name: image for name, image in (("first_frame", start), ("last_frame", end)) if image is not None}
    conditioned = pipeline.add_node(
        MiniMaxH3ImageToVideo,
        clip=clip,
        vae=video_vae,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        **keyframes,
    ).output
    # One latent holds the video and its sound; each VAE decodes its own part.
    sampled = pipeline.add_node(
        SamplerCustomAdvanced,
        noise=pipeline.add_node(RandomNoise, noise_seed=_seed(seed)).output["NOISE"],
        guider=pipeline.add_node(BasicGuider, model=unet, conditioning=conditioned["positive"]).output["GUIDER"],
        sampler=pipeline.add_node(KSamplerSelect, sampler_name="res_multistep").output["SAMPLER"],
        sigmas=pipeline.add_node(
            BasicScheduler,
            model=unet,
            scheduler="simple",
            steps=steps or (_MINIMAX_FAST_STEPS if fast else _MINIMAX_STEPS),
            denoise=1.0,
        ).output["SIGMAS"],
        latent_image=conditioned["LATENT"],
    ).output["output"]
    frames = pipeline.add_node(VAEDecode, samples=sampled, vae=video_vae).output["IMAGE"]
    if not with_sound:
        return frames, None
    audio_vae = pipeline.add_node(VAELoader, vae_name=model.file("audio_vae")).output["VAE"]
    sound = pipeline.add_node(VAEDecodeAudio, samples=sampled, vae=audio_vae).output["AUDIO"]
    return frames, sound


def _clip(
    pipeline: AgentPipeline,
    frames: Link,
    length: int,
    fps: int,
    audio: str | Link | None,
    sound: Link | None = None,
) -> Link:
    soundtrack = {}
    if audio is not None:
        # The clip's length is known here, so the soundtrack is cut to it directly.
        soundtrack["audio"] = pipeline.add_node(
            TrimAudioDuration, audio=pipeline.audio(audio), start_index=0.0, duration=length / fps
        ).output["AUDIO"]
    elif sound is not None:
        soundtrack["audio"] = sound
    return pipeline.add_node(CreateVideo, images=frames, fps=float(fps), **soundtrack).output["VIDEO"]


def _video(
    pipeline: AgentPipeline,
    model: ModelChoice,
    *,
    prompt: str,
    negative_prompt: str,
    orientation: str,
    duration_seconds: float,
    fps: int | None,
    width: int | None,
    height: int | None,
    steps: int | None,
    cfg: float | None,
    seed: int | None,
    fast: bool,
    audio: str | Link | None,
    start_image: str | Link | None = None,
    end_image: str | Link | None = None,
) -> tuple[Link, Link]:
    spec = _VIDEO_SPECS[model.family]
    fps = fps or spec.fps
    length = _frames(duration_seconds, fps, spec)
    size_width, size_height = _video_size(spec, orientation, width, height)
    if end_image is not None and model.family == "wan22_5b":
        raise ValueError(
            f"the model {model.id!r} cannot end on an image; use a Wan 2.2 14B or MiniMax H3 model"
        )
    start = None if start_image is None else _fit(pipeline, start_image, size_width, size_height)
    end = None if end_image is None else _fit(pipeline, end_image, size_width, size_height)
    settings = {
        "prompt": prompt,
        "width": size_width,
        "height": size_height,
        "length": length,
        "steps": steps,
        "seed": seed,
    }
    sound = None
    if model.family == "minimax_h3":
        # A soundtrack of the user replaces the sound the model makes.
        frames, sound = _minimax_clip(
            pipeline, model, fast=fast, with_sound=audio is None, start=start, end=end, **settings
        )
    elif model.family == "wan22_5b":
        frames = _wan_frames(pipeline, model, negative_prompt=negative_prompt, cfg=cfg, start=start, **settings)
    else:
        frames = _wan14b_frames(
            pipeline, model, negative_prompt=negative_prompt, cfg=cfg, fast=fast, start=start, end=end, **settings
        )
    return _clip(pipeline, frames, length, fps, audio, sound), frames


_VIDEO_PARAMETERS = {
    "model": _MODEL,
    "negative_prompt": "What the video must not show; the default is the model's own. Wan models only.",
    "orientation": "landscape 16:9, portrait 9:16 or square.",
    "duration_seconds": "Length of the clip in seconds: up to 5 with Wan, 5 to 15 with MiniMax H3.",
    "fps": "Frames per second; empty uses the model's own: 24, or 16 for Wan 2.2 14B.",
    "width": "Exact width instead of the orientation; rounded to a multiple of 16 or 32.",
    "height": "Exact height instead of the orientation; rounded to a multiple of 16 or 32.",
    "steps": _STEPS,
    "cfg": f"{_CFG} Wan models only.",
    "seed": _SEED,
    "fast": _FAST,
    "audio": (
        "Optional soundtrack: an audio file the user attached or an AUDIO output, cut to "
        "the clip. With MiniMax H3 it replaces the sound the model makes."
    ),
    "save": _SAVE,
    "format": "Container of the saved video: mp4 (H.264) or webm (AV1).",
    "filename_prefix": "Prefix of the saved file name.",
}
_VIDEO_PROMPT = (
    "in English: subject, motion, setting, camera. For a model with sound, also what "
    "is heard: speech (quoted, with who says it), sound effects and music."
)


class TextToVideo(AgentPipeline):
    NAME = "text_to_video"
    CAPABILITY = "text_to_video"
    DESCRIPTION = (
        "Generates a video clip from a prompt and saves it as a video file. A model with "
        "sound gives the clip the sound the prompt describes. Outputs: video (VIDEO) and "
        "frames (IMAGE)."
    )
    PARAMETERS = {
        "prompt": f"What happens in the clip, {_VIDEO_PROMPT}",
        **_VIDEO_PARAMETERS,
    }
    FILES = {"audio": "audio"}
    COVERS = ("Wan22ImageToVideoLatent", "ModelSamplingSD3", "EmptyHunyuanLatentVideo", "MiniMaxH3ImageToVideo")

    def build(
        self,
        prompt: str,
        model: ModelChoice,
        negative_prompt: str = WAN_NEGATIVE,
        orientation: Orientation = "landscape",
        duration_seconds: float = 5.0,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        fast: bool = True,
        audio: str | Link | None = None,
        save: bool = True,
        format: VideoFormat = "mp4",
        filename_prefix: str = "video",
    ):
        _require(model, _TEXT_TO_VIDEO, self.NAME)
        video, frames = _video(
            self,
            model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            orientation=orientation,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            fast=fast,
            audio=audio,
        )
        if save:
            self.save_video(video, filename_prefix, format)
        return {"video": video, "frames": frames}


class ImageToVideo(AgentPipeline):
    NAME = "image_to_video"
    CAPABILITY = "image_to_video"
    DESCRIPTION = (
        "Animates an image: generates a video clip that starts from it, following a "
        "prompt, and saves it; some models can also end on a second image. Pick the "
        "orientation closest to the image's shape. Outputs: video (VIDEO) and frames (IMAGE)."
    )
    PARAMETERS = {
        "image": "The start image: a file the user attached, or an IMAGE output.",
        "prompt": f"What happens in the clip, starting from the image, {_VIDEO_PROMPT}",
        "end_image": "Optional image the clip ends on (Wan 2.2 14B and MiniMax H3): a file or an IMAGE output.",
        **_VIDEO_PARAMETERS,
    }
    FILES = {"image": "image", "end_image": "image", "audio": "audio"}
    COVERS = (
        "Wan22ImageToVideoLatent",
        "ModelSamplingSD3",
        "WanImageToVideo",
        "WanFirstLastFrameToVideo",
        "MiniMaxH3ImageToVideo",
    )

    def build(
        self,
        image: str | Link,
        prompt: str,
        model: ModelChoice,
        end_image: str | Link | None = None,
        negative_prompt: str = WAN_NEGATIVE,
        orientation: Orientation = "landscape",
        duration_seconds: float = 5.0,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        fast: bool = True,
        audio: str | Link | None = None,
        save: bool = True,
        format: VideoFormat = "mp4",
        filename_prefix: str = "video",
    ):
        _require(model, _IMAGE_TO_VIDEO, self.NAME)
        video, frames = _video(
            self,
            model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            orientation=orientation,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            fast=fast,
            audio=audio,
            start_image=image,
            end_image=end_image,
        )
        if save:
            self.save_video(video, filename_prefix, format)
        return {"video": video, "frames": frames}


def _s2v_parts(duration: float) -> tuple[int, int]:
    """The frames of each part of a sound-to-video clip, and how many parts it has."""
    frames = duration * _S2V_FPS
    if frames < _S2V_PART:
        return 4 * max(1, round((frames - 1) / 4)) + 1, 1
    return _S2V_PART, round(frames / _S2V_PART)


class SoundToVideo(AgentPipeline):
    NAME = "sound_to_video"
    CAPABILITY = "sound_to_video"
    DESCRIPTION = (
        "Animates a photo in sync with an audio: the person or character in it speaks or "
        "sings the audio, with lips, face and body following it, and the video keeps that "
        "audio. Use it to make a photo talk or sing. Outputs: video (VIDEO) and frames (IMAGE)."
    )
    PARAMETERS = {
        "image": "The photo to animate: a file the user attached, or an IMAGE output.",
        "audio": "The speech or song: an audio file the user attached, or an AUDIO output.",
        "prompt": "What the person does while the audio plays, in English: expression, gestures, setting, camera.",
        "model": _MODEL,
        "duration_seconds": (
            f"Length of the video from the start of the audio, up to {_S2V_MAX_SECONDS:g} "
            "seconds; set it to the audio's length to use all of it."
        ),
        "orientation": "landscape, portrait or square: the one closest to the photo's shape.",
        "width": "Exact width instead of the orientation; rounded to a multiple of 16.",
        "height": "Exact height instead of the orientation; rounded to a multiple of 16.",
        "negative_prompt": "What the video must not show; the default is the model's own.",
        "steps": _STEPS,
        "cfg": _CFG,
        "seed": _SEED,
        "fast": _FAST,
        "save": _SAVE,
        "format": "Container of the saved video: mp4 (H.264) or webm (AV1).",
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"image": "image", "audio": "audio"}
    COVERS = ("WanSoundImageToVideo", "WanSoundImageToVideoExtend", "AudioEncoderLoader", "AudioEncoderEncode")

    def build(
        self,
        image: str | Link,
        audio: str | Link,
        prompt: str,
        model: ModelChoice,
        duration_seconds: float = 5.0,
        orientation: Orientation = "square",
        width: int | None = None,
        height: int | None = None,
        negative_prompt: str = WAN_NEGATIVE,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        fast: bool = True,
        save: bool = True,
        format: VideoFormat = "mp4",
        filename_prefix: str = "video",
    ):
        _require(model, ("wan22_s2v",), self.NAME)
        if not 0 < duration_seconds <= _S2V_MAX_SECONDS:
            raise ValueError(
                f"the video lasts from 0 to {_S2V_MAX_SECONDS:g} seconds; for a longer audio "
                "make several videos from parts of it and join them with concatenate_videos"
            )
        part, parts = _s2v_parts(duration_seconds)
        size_width, size_height = _S2V_SIZES[orientation]
        if width is not None:
            size_width = _multiple(width, 16)
        if height is not None:
            size_height = _multiple(height, 16)
        fast = fast and "fast" in model.files
        unet = self.add_node(UNETLoader, unet_name=model.file("unet"), weight_dtype="default").output["MODEL"]
        if fast:
            unet = _with_lora(self, unet, model.file("fast"))
        patched = self.add_node(ModelSamplingSD3, model=unet, shift=_S2V_SHIFT).output["MODEL"]
        clip = self.add_node(
            CLIPLoader, clip_name=model.file("text_encoder"), type="wan", device="default"
        ).output["CLIP"]
        vae = self.add_node(VAELoader, vae_name=model.file("vae")).output["VAE"]
        sound = self.audio(audio)
        encoder = self.add_node(AudioEncoderLoader, audio_encoder_name=model.file("audio_encoder"))
        encoded = self.add_node(
            AudioEncoderEncode, audio_encoder=encoder.output["AUDIO_ENCODER"], audio=sound
        ).output["AUDIO_ENCODER_OUTPUT"]
        shared = {
            "positive": _text(self, clip, prompt),
            "negative": _text(self, clip, negative_prompt),
            "vae": vae,
            "length": part,
            "audio_encoder_output": encoded,
            "ref_image": _fit(self, image, size_width, size_height),
        }
        settings = {
            "steps": steps or (_S2V_FAST_STEPS if fast else _S2V_STEPS),
            "cfg": (1.0 if fast else _S2V_CFG) if cfg is None else cfg,
            "sampler": "uni_pc",
            "scheduler": "simple",
        }
        seed = _seed(seed)
        first = self.add_node(
            WanSoundImageToVideo, width=size_width, height=size_height, batch_size=1, **shared
        ).output
        latent = _sample(self, patched, first["positive"], first["negative"], first["latent"], seed=seed, **settings)
        for index in range(1, parts):
            # Each part continues the video so far, following the next stretch of the audio.
            extended = self.add_node(WanSoundImageToVideoExtend, video_latent=latent, **shared).output
            more = _sample(
                self,
                patched,
                extended["positive"],
                extended["negative"],
                extended["latent"],
                seed=seed + index,
                **settings,
            )
            latent = self.add_node(LatentConcat, samples1=latent, samples2=more, dim="t").output["LATENT"]
        # As in ComfyUI's template: the VAE overbakes the first frame, so its latent is
        # decoded twice and the extra frames are dropped.
        head = self.add_node(LatentCut, samples=latent, dim="t", index=0, amount=1).output["LATENT"]
        doubled = self.add_node(LatentConcat, samples1=head, samples2=latent, dim="t").output["LATENT"]
        decoded = self.add_node(VAEDecode, samples=doubled, vae=vae).output["IMAGE"]
        frames = self.add_node(ImageFromBatch, image=decoded, batch_index=parts, length=4096).output["IMAGE"]
        # Each latent frame decodes to 4 frames and the doubled first one to 5; the
        # template drops one frame per part. The soundtrack is cut to what is left.
        length = 4 * parts * ((part - 1) // 4 + 1) + 1 - parts
        video = _clip(self, frames, length, _S2V_FPS, sound)
        if save:
            self.save_video(video, filename_prefix, format)
        return {"video": video, "frames": frames}


class TextToAudio(AgentPipeline):
    NAME = "text_to_audio"
    CAPABILITY = "text_to_audio"
    DESCRIPTION = (
        "Generates music from a description of its style, a song with sung lyrics or an "
        "instrumental, and saves it as an audio file. Output: audio (AUDIO)."
    )
    PARAMETERS = {
        "style": (
            "The music, in English: genre, mood, instruments, voices and production, as "
            "tags or a short description. Like 'acoustic pop, warm female vocals, guitar, upbeat'."
        ),
        "model": _MODEL,
        "lyrics": (
            "The words to sing, in the song's language, with section tags on their own "
            "lines like [verse], [chorus] and [bridge]. Empty for an instrumental."
        ),
        "duration_seconds": f"Length of the music in seconds, up to {_ACE_MAX_SECONDS:g}.",
        "language": "Language of the lyrics, as a code like en, pt, es, fr, de, it, ja, ko or zh.",
        "bpm": "Tempo in beats per minute, fitting the style.",
        "key": "Key and mode, like 'C major', 'F# minor' or 'Eb major'.",
        "time_signature": "Beats per bar.",
        "steps": _STEPS,
        "seed": _SEED,
        "save": _SAVE,
        "format": "Format of the saved audio.",
        "filename_prefix": "Prefix of the saved file name.",
    }
    COVERS = ("TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio", "VAEDecodeAudio")

    def build(
        self,
        style: str,
        model: ModelChoice,
        lyrics: str = "",
        duration_seconds: float = 60.0,
        language: str = "en",
        bpm: int = 120,
        key: str = "C major",
        time_signature: TimeSignature = "4",
        steps: int | None = None,
        seed: int | None = None,
        save: bool = True,
        format: AudioFormat = "mp3",
        filename_prefix: str = "music",
    ):
        _require(model, ("ace_step_15",), self.NAME)
        if not 1 <= duration_seconds <= _ACE_MAX_SECONDS:
            raise ValueError(f"the music lasts from 1 to {_ACE_MAX_SECONDS:g} seconds")
        seed = _seed(seed)
        unet = self.add_node(UNETLoader, unet_name=model.file("unet"), weight_dtype="default").output["MODEL"]
        patched = self.add_node(ModelSamplingAuraFlow, model=unet, shift=_ACE_SHIFT).output["MODEL"]
        clip = self.add_node(
            DualCLIPLoader,
            clip_name1=model.file("text_encoder"),
            clip_name2=model.file("lm"),
            type="ace",
            device="default",
        ).output["CLIP"]
        positive = self.add_node(
            TextEncodeAceStepAudio1_5,
            clip=clip,
            tags=style,
            lyrics=lyrics.strip() or _INSTRUMENTAL,
            seed=seed,
            bpm=bpm,
            duration=duration_seconds,
            timesignature=time_signature,
            language=language,
            keyscale=key,
            generate_audio_codes=True,
            cfg_scale=2.0,
            temperature=0.85,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
        ).output["CONDITIONING"]
        negative = self.add_node(ConditioningZeroOut, conditioning=positive).output["CONDITIONING"]
        latent = self.add_node(
            EmptyAceStep1_5LatentAudio, seconds=duration_seconds, batch_size=1
        ).output["LATENT"]
        sampled = _sample(
            self,
            patched,
            positive,
            negative,
            latent,
            seed=seed,
            steps=steps or _ACE_STEPS,
            cfg=1.0,
            sampler="euler",
            scheduler="simple",
        )
        vae = self.add_node(VAELoader, vae_name=model.file("vae")).output["VAE"]
        music = self.add_node(VAEDecodeAudio, samples=sampled, vae=vae).output["AUDIO"]
        if save:
            self.save_audio(music, filename_prefix, format)
        return {"audio": music}


class UpscaleImageWithModel(AgentPipeline):
    NAME = "upscale_image_with_model"
    CAPABILITY = "upscale_with_model"
    DESCRIPTION = (
        "Enlarges images with an upscale model, which restores sharp detail: much better "
        "than upscale_image for photos and art. It takes a batch too, like the frames of "
        "a video from load_video. Output: image (IMAGE)."
    )
    PARAMETERS = {
        "image": "An image file the user attached, or an IMAGE output: one image or a batch of frames.",
        "model": _MODEL,
        "scale": "How many times larger; empty uses the model's own factor, others resize its result.",
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file names.",
    }
    FILES = {"image": "image"}
    COVERS = ("UpscaleModelLoader", "ImageUpscaleWithModel")

    def build(
        self,
        image: str | Link,
        model: ModelChoice,
        scale: float | None = None,
        save: bool = True,
        filename_prefix: str = "image",
    ):
        _require(model, ("upscaler",), self.NAME)
        loader = self.add_node(UpscaleModelLoader, model_name=model.file("upscaler"))
        upscaled = self.add_node(
            ImageUpscaleWithModel, upscale_model=loader.output["UPSCALE_MODEL"], image=self.image(image)
        ).output["IMAGE"]
        factor = upscale_factor(model.file("upscaler"))
        if scale is not None and scale != factor:
            upscaled = self.add_node(
                ImageScaleBy, image=upscaled, upscale_method="lanczos", scale_by=scale / factor
            ).output["IMAGE"]
        if save:
            self.save_image(upscaled, filename_prefix)
        return {"image": upscaled}
