import pytest
from comfy_ui_api_workflow import Workflow, errors

from agent_comfy_ui_api_workflow._impl.families import find_models
from agent_comfy_ui_api_workflow._impl.pipelines import GENERATION_PIPELINES
from agent_comfy_ui_api_workflow._impl.pipelines.generation import (
    ImageToImage,
    ImageToVideo,
    Inpaint,
    Outpaint,
    SoundToVideo,
    TextToAudio,
    TextToImage,
    TextToVideo,
    UpscaleImageWithModel,
)
from conftest import (
    ACE_ENCODER,
    ACE_LM,
    REALESRGAN,
    SDXL,
    WAN21_VAE,
    WAN_14B_T2V,
    WAN_14B_T2V_FAST,
    WAN_MODEL,
    WAV2VEC2,
)
from test_pipelines_media import classes, node_of, reloads


@pytest.fixture
def models(inventory):
    return {model.family: model for model in find_models(inventory)}


@pytest.fixture
def server(server_inventory):
    return {model.family: model for model in find_models(server_inventory)}


def nodes_of(workflow: Workflow, class_type: str) -> list[dict]:
    return [
        node["inputs"]
        for node in workflow.get_comfy_ui_workflow_format().values()
        if node["class_type"] == class_type
    ]


def source_class(workflow: Workflow, link) -> str:
    return workflow.get_comfy_ui_workflow_format()[link[0]]["class_type"]


@pytest.mark.parametrize("pipeline", GENERATION_PIPELINES, ids=lambda p: p.NAME)
def test_every_parameter_is_documented(pipeline):
    names = [parameter.name for parameter in pipeline.parameters()]

    assert pipeline.CAPABILITY
    assert "model" in names
    assert sorted(pipeline.PARAMETERS) == sorted(names)


def test_text_to_image_uses_the_sdxl_size_of_the_aspect_ratio(models):
    workflow = Workflow()

    pipeline = workflow.add_pipeline(
        TextToImage, prompt="a koi pond", model=models["sdxl"], aspect_ratio="16:9", seed=3
    )

    assert node_of(workflow, "CheckpointLoaderSimple") == {"ckpt_name": SDXL}
    assert node_of(workflow, "EmptyLatentImage") == {"width": 1344, "height": 768, "batch_size": 1}
    sampler = node_of(workflow, "KSampler")
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (3, 25, 7.0)
    assert (sampler["sampler_name"], sampler["scheduler"]) == ("dpmpp_2m", "karras")
    assert "SaveImage" in classes(workflow)
    assert set(pipeline.output.outputs) == {"images"}
    reloads(workflow)


def test_text_to_image_takes_an_exact_size_rounded_to_8(models):
    workflow = Workflow()

    workflow.add_pipeline(TextToImage, prompt="x", model=models["sdxl"], width=1001, height=700)

    assert node_of(workflow, "EmptyLatentImage")["width"] == 1000
    assert node_of(workflow, "EmptyLatentImage")["height"] == 704


def test_text_to_image_upscale_adds_a_second_pass(models):
    workflow = Workflow()

    workflow.add_pipeline(TextToImage, prompt="x", model=models["sdxl"], upscale=1.5, seed=1)

    assert classes(workflow).count("KSampler") == 2
    assert node_of(workflow, "LatentUpscaleBy")["scale_by"] == 1.5


def test_a_random_seed_is_chosen_when_none_is_given(models):
    workflow = Workflow()

    workflow.add_pipeline(TextToImage, prompt="x", model=models["sdxl"])

    assert isinstance(node_of(workflow, "KSampler")["seed"], int)


def test_image_to_image_scales_to_a_megapixel_and_denoises_partly(models):
    workflow = Workflow()

    workflow.add_pipeline(ImageToImage, image="me.png", prompt="watercolor", model=models["sdxl"], strength=0.55)

    assert node_of(workflow, "LoadImage") == {"image": "me.png"}
    assert node_of(workflow, "ImageScaleToTotalPixels")["megapixels"] == 1.0
    assert node_of(workflow, "KSampler")["denoise"] == 0.55
    reloads(workflow)


def test_inpaint_a_region_replaces_and_pastes_back(models):
    workflow = Workflow()

    workflow.add_pipeline(
        Inpaint, image="me.png", prompt="a red hat", model=models["sdxl"], region=[10, 20, 300, 200]
    )

    names = classes(workflow)
    assert names.count("SolidMask") == 2
    assert node_of(workflow, "MaskComposite")["x"] == 10
    assert "VAEEncodeForInpaint" in names
    assert "ImageCompositeMasked" in names
    reloads(workflow)


def test_inpaint_with_a_strength_below_one_keeps_the_content(models):
    workflow = Workflow()

    workflow.add_pipeline(
        Inpaint, image="me.png", prompt="older", model=models["sdxl"], mask="mask.png", strength=0.5
    )

    assert node_of(workflow, "LoadImageMask") == {"image": "mask.png", "channel": "red"}
    assert "SetLatentNoiseMask" in classes(workflow)
    assert node_of(workflow, "KSampler")["denoise"] == 0.5


def test_inpaint_without_mask_or_region_uses_the_transparent_pixels(models):
    workflow = Workflow()

    workflow.add_pipeline(Inpaint, image="cutout.png", prompt="a beach", model=models["sdxl"])

    mask_of_image = [
        node["inputs"]["mask"]
        for node in workflow.get_comfy_ui_workflow_format().values()
        if node["class_type"] == "VAEEncodeForInpaint"
    ][0]
    assert mask_of_image[1] == 1


def test_inpaint_takes_a_mask_or_a_region_but_not_both(models):
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(
            Inpaint, image="me.png", prompt="x", model=models["sdxl"], mask="m.png", region=[0, 0, 8, 8]
        )


def test_outpaint_pads_and_rounds_to_8(models):
    workflow = Workflow()

    workflow.add_pipeline(Outpaint, image="me.png", prompt="beach", model=models["sdxl"], right=250)

    padded = node_of(workflow, "ImagePadForOutpaint")
    assert (padded["left"], padded["right"]) == (0, 256)
    assert node_of(workflow, "VAEEncodeForInpaint")["grow_mask_by"] == 16
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(Outpaint, image="me.png", prompt="beach", model=models["sdxl"])


def test_text_to_video_makes_wan_frames_from_seconds(models):
    workflow = Workflow()

    pipeline = workflow.add_pipeline(
        TextToVideo, prompt="a lion running", model=models["wan22_5b"], duration_seconds=3.0, seed=5
    )

    assert node_of(workflow, "UNETLoader") == {"unet_name": WAN_MODEL, "weight_dtype": "default"}
    assert node_of(workflow, "ModelSamplingSD3")["shift"] == 8.0
    assert node_of(workflow, "CLIPLoader")["type"] == "wan"
    latent = node_of(workflow, "Wan22ImageToVideoLatent")
    assert (latent["width"], latent["height"], latent["length"]) == (1280, 704, 73)
    sampler = node_of(workflow, "KSampler")
    assert (sampler["sampler_name"], sampler["scheduler"], sampler["cfg"]) == ("uni_pc", "simple", 5.0)
    assert node_of(workflow, "CreateVideo")["fps"] == 24.0
    assert node_of(workflow, "SaveVideo")["format"] == "mp4"
    assert set(pipeline.output.outputs) == {"video", "frames"}
    reloads(workflow)


def test_text_to_video_refuses_more_than_the_model_can_make(models):
    with pytest.raises(errors.pipeline.PipelineError) as error:
        Workflow().add_pipeline(TextToVideo, prompt="x", model=models["wan22_5b"], duration_seconds=12.0)

    assert "concatenate_videos" in str(error.value)


def test_text_to_video_cuts_a_soundtrack_to_the_clip(models):
    workflow = Workflow()

    workflow.add_pipeline(
        TextToVideo, prompt="x", model=models["wan22_5b"], duration_seconds=2.0, audio="song.mp3", orientation="portrait"
    )

    assert node_of(workflow, "TrimAudioDuration")["duration"] == pytest.approx(49 / 24)
    assert node_of(workflow, "Wan22ImageToVideoLatent")["width"] == 704


def test_image_to_video_starts_from_the_image_at_the_video_size(models):
    workflow = Workflow()

    workflow.add_pipeline(ImageToVideo, image="me.png", prompt="she smiles", model=models["wan22_5b"])

    scaled = node_of(workflow, "ImageScale")
    assert (scaled["width"], scaled["height"], scaled["crop"]) == (1280, 704, "center")
    assert node_of(workflow, "Wan22ImageToVideoLatent")["start_image"][0] in workflow.nodes


def test_a_model_of_another_family_is_refused(models):
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(TextToImage, prompt="x", model=models["wan22_5b"])


def test_wan_14b_samples_with_both_experts_and_the_fast_loras(server):
    workflow = Workflow()

    workflow.add_pipeline(TextToVideo, prompt="a fox in snow", model=server["wan22_t2v_14b"], duration_seconds=3.0, seed=5)

    assert [unet["unet_name"] for unet in nodes_of(workflow, "UNETLoader")] == [
        WAN_14B_T2V,
        WAN_14B_T2V.replace("high_noise", "low_noise"),
    ]
    assert nodes_of(workflow, "LoraLoaderModelOnly")[0]["lora_name"] == WAN_14B_T2V_FAST
    assert {patch["shift"] for patch in nodes_of(workflow, "ModelSamplingSD3")} == {5.0}
    samplers = nodes_of(workflow, "KSamplerAdvanced")
    assert [(s["add_noise"], s["start_at_step"], s["end_at_step"]) for s in samplers] == [
        ("enable", 0, 2),
        ("disable", 2, 4),
    ]
    assert {(s["steps"], s["cfg"], s["sampler_name"], s["scheduler"], s["noise_seed"]) for s in samplers} == {
        (4, 1.0, "euler", "simple", 5)
    }
    latent = node_of(workflow, "EmptyHunyuanLatentVideo")
    assert (latent["width"], latent["height"], latent["length"]) == (1280, 720, 49)
    assert node_of(workflow, "VAELoader") == {"vae_name": WAN21_VAE}
    assert node_of(workflow, "CreateVideo")["fps"] == 16.0
    reloads(workflow)


def test_wan_14b_without_fast_uses_the_original_settings(server):
    workflow = Workflow()

    workflow.add_pipeline(TextToVideo, prompt="x", model=server["wan22_t2v_14b"], fast=False)

    assert "LoraLoaderModelOnly" not in classes(workflow)
    samplers = nodes_of(workflow, "KSamplerAdvanced")
    assert [(s["steps"], s["cfg"], s["end_at_step"]) for s in samplers] == [(20, 3.5, 10), (20, 3.5, 20)]
    assert node_of(workflow, "EmptyHunyuanLatentVideo")["length"] == 81


def test_wan_14b_image_to_video_can_end_on_an_image(server):
    ends = Workflow()
    starts = Workflow()

    ends.add_pipeline(ImageToVideo, image="a.png", end_image="b.png", prompt="x", model=server["wan22_i2v_14b"])
    starts.add_pipeline(ImageToVideo, image="a.png", prompt="x", model=server["wan22_i2v_14b"])

    both = node_of(ends, "WanFirstLastFrameToVideo")
    assert source_class(ends, both["start_image"]) == source_class(ends, both["end_image"]) == "ImageScale"
    assert "WanImageToVideo" not in classes(ends)
    assert node_of(starts, "WanImageToVideo")["length"] == 81
    reloads(ends)


def test_the_wan_5b_cannot_end_on_an_image(models):
    with pytest.raises(errors.pipeline.PipelineError) as error:
        Workflow().add_pipeline(ImageToVideo, image="a.png", end_image="b.png", prompt="x", model=models["wan22_5b"])

    assert "14B" in str(error.value)


def test_minimax_h3_makes_the_video_with_its_sound(server):
    workflow = Workflow()

    workflow.add_pipeline(TextToVideo, prompt="rain on a tin roof", model=server["minimax_h3"], seed=2)

    conditioned = node_of(workflow, "MiniMaxH3ImageToVideo")
    assert (conditioned["width"], conditioned["height"], conditioned["length"]) == (1344, 768, 124)
    assert node_of(workflow, "CLIPLoader")["type"] == "minimax"
    assert node_of(workflow, "KSamplerSelect") == {"sampler_name": "res_multistep"}
    assert node_of(workflow, "BasicScheduler")["steps"] == 8
    assert node_of(workflow, "RandomNoise") == {"noise_seed": 2}
    video = node_of(workflow, "CreateVideo")
    assert source_class(workflow, video["audio"]) == "VAEDecodeAudio"
    assert video["fps"] == 24.0
    reloads(workflow)


def test_minimax_h3_clips_follow_its_frame_grid(server):
    for seconds, frames in ((2.0, 56), (10.0, 243), (15.0, 362)):
        workflow = Workflow()
        workflow.add_pipeline(TextToVideo, prompt="x", model=server["minimax_h3"], duration_seconds=seconds)
        assert node_of(workflow, "MiniMaxH3ImageToVideo")["length"] == frames
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(TextToVideo, prompt="x", model=server["minimax_h3"], duration_seconds=20.0)


def test_a_soundtrack_replaces_the_sound_of_minimax_h3(server):
    workflow = Workflow()

    workflow.add_pipeline(TextToVideo, prompt="x", model=server["minimax_h3"], audio="song.mp3")

    assert source_class(workflow, node_of(workflow, "CreateVideo")["audio"]) == "TrimAudioDuration"
    assert "VAEDecodeAudio" not in classes(workflow)


def test_minimax_h3_starts_and_ends_on_images(server):
    workflow = Workflow()

    workflow.add_pipeline(ImageToVideo, image="a.png", end_image="b.png", prompt="x", model=server["minimax_h3"])

    conditioned = node_of(workflow, "MiniMaxH3ImageToVideo")
    assert source_class(workflow, conditioned["first_frame"]) == "ImageScale"
    assert source_class(workflow, conditioned["last_frame"]) == "ImageScale"


def test_sound_to_video_follows_the_audio_in_parts(server):
    workflow = Workflow()

    workflow.add_pipeline(
        SoundToVideo, image="me.png", audio="talk.wav", prompt="she talks", model=server["wan22_s2v"], duration_seconds=12.0, seed=3
    )

    names = classes(workflow)
    assert node_of(workflow, "WanSoundImageToVideo")["length"] == 77
    assert names.count("WanSoundImageToVideoExtend") == 1
    assert [sampler["seed"] for sampler in nodes_of(workflow, "KSampler")] == [3, 4]
    assert node_of(workflow, "ImageFromBatch")["batch_index"] == 2
    assert node_of(workflow, "AudioEncoderLoader") == {"audio_encoder_name": WAV2VEC2}
    assert node_of(workflow, "LoraLoaderModelOnly")["lora_name"] == WAN_14B_T2V_FAST
    assert node_of(workflow, "ModelSamplingSD3")["shift"] == 8.0
    # Measured on the worker: two parts of 77 frames come out as 159 frames.
    assert node_of(workflow, "TrimAudioDuration")["duration"] == pytest.approx(159 / 16)
    reloads(workflow)


def test_a_short_sound_to_video_is_one_part(server):
    workflow = Workflow()

    workflow.add_pipeline(
        SoundToVideo, image="me.png", audio="talk.wav", prompt="x", model=server["wan22_s2v"], duration_seconds=2.0
    )

    assert node_of(workflow, "WanSoundImageToVideo")["length"] == 33
    assert "WanSoundImageToVideoExtend" not in classes(workflow)
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(
            SoundToVideo, image="me.png", audio="talk.wav", prompt="x", model=server["wan22_s2v"], duration_seconds=45.0
        )


def test_text_to_audio_follows_the_ace_step_template(server):
    workflow = Workflow()

    workflow.add_pipeline(
        TextToAudio,
        style="bossa nova, nylon guitar, soft male vocals",
        model=server["ace_step_15"],
        lyrics="[verse]\nO sol se pôs no mar",
        language="pt",
        duration_seconds=30.0,
        bpm=100,
        key="A minor",
        seed=4,
    )

    encoder = node_of(workflow, "TextEncodeAceStepAudio1.5")
    assert (encoder["lyrics"], encoder["language"], encoder["keyscale"]) == ("[verse]\nO sol se pôs no mar", "pt", "A minor")
    assert (encoder["bpm"], encoder["duration"], encoder["seed"], encoder["timesignature"]) == (100, 30.0, 4, "4")
    assert node_of(workflow, "EmptyAceStep1.5LatentAudio") == {"seconds": 30.0, "batch_size": 1}
    clip = node_of(workflow, "DualCLIPLoader")
    assert (clip["clip_name1"], clip["clip_name2"], clip["type"]) == (ACE_ENCODER, ACE_LM, "ace")
    sampler = node_of(workflow, "KSampler")
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (8, 1.0, "euler", "simple")
    assert node_of(workflow, "ModelSamplingAuraFlow")["shift"] == 3.0
    assert "SaveAudioAdvanced" in classes(workflow)
    reloads(workflow)


def test_text_to_audio_without_lyrics_is_an_instrumental(server):
    workflow = Workflow()

    workflow.add_pipeline(TextToAudio, style="ambient pads", model=server["ace_step_15"])

    assert node_of(workflow, "TextEncodeAceStepAudio1.5")["lyrics"] == "[instrumental]"
    with pytest.raises(errors.pipeline.PipelineError):
        Workflow().add_pipeline(TextToAudio, style="x", model=server["ace_step_15"], duration_seconds=900.0)


def test_upscale_with_model_resizes_only_to_another_scale(server):
    resized = Workflow()
    native = Workflow()

    resized.add_pipeline(UpscaleImageWithModel, image="small.png", model=server["upscaler"], scale=2.0)
    native.add_pipeline(UpscaleImageWithModel, image="small.png", model=server["upscaler"])

    assert node_of(resized, "UpscaleModelLoader") == {"model_name": REALESRGAN}
    assert node_of(resized, "ImageScaleBy")["scale_by"] == 0.5
    assert "ImageScaleBy" not in classes(native)
    reloads(resized)
