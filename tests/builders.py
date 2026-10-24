"""Workflows built with the lib directly, for tests that need one in place."""

from comfy_ui_api_workflow import Workflow
from comfy_ui_api_workflow.nodes.image import SaveImage
from comfy_ui_api_workflow.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow.nodes.model.sampling import KSampler

from conftest import SDXL


def text_to_image(
    workflow: Workflow, prompt: str = "a lighthouse", checkpoint: str = SDXL
) -> None:
    """Adds CheckpointLoaderSimple_0 … SaveImage_6, a plain SDXL text to image chain."""
    loader = workflow.add_node(CheckpointLoaderSimple, ckpt_name=checkpoint)
    positive = workflow.add_node(CLIPTextEncode, text=prompt, clip=loader.output["CLIP"])
    negative = workflow.add_node(CLIPTextEncode, text="blurry", clip=loader.output["CLIP"])
    latent = workflow.add_node(EmptyLatentImage, width=1024, height=1024, batch_size=1)
    sampler = workflow.add_node(
        KSampler,
        model=loader.output["MODEL"],
        positive=positive.output["CONDITIONING"],
        negative=negative.output["CONDITIONING"],
        latent_image=latent.output["LATENT"],
        sampler_name="dpmpp_2m",
        scheduler="karras",
        steps=25,
        cfg=7.0,
        seed=1,
    )
    image = workflow.add_node(VAEDecode, samples=sampler.output["LATENT"], vae=loader.output["VAE"])
    workflow.add_node(SaveImage, images=image.output["IMAGE"], filename_prefix="out")
