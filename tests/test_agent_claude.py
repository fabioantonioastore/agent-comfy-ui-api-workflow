"""The agent on the real Claude API: run with --anthropic (paid tokens)."""

import pytest

from agent_comfy_ui_api_workflow import WorkflowAgent
from builders import text_to_image
from comfy_ui_api_workflow import Workflow
from conftest import SDXL
from media import MP4, png, wav

pytestmark = pytest.mark.anthropic


def classes(workflow: dict) -> list[str]:
    return [node["class_type"] for node in workflow.values()]


async def test_asks_for_the_photo_it_needs(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send("crie uma imagem de um elefante com a minha cara")

    assert reply.status == "needs_input", reply.message
    assert "image" in reply.question.needs
    assert reply.workflow == {}


async def test_trims_a_video_with_the_pipeline(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send(
        "corte este vídeo do segundo 2 ao segundo 5", files={"viagem.mp4": MP4}
    )

    assert reply.status == "completed", reply.message
    (sliced,) = [node["inputs"] for node in reply.workflow.values() if node["class_type"] == "Video Slice"]
    assert (sliced["start_time"], sliced["duration"]) == (2.0, 3.0)
    assert sorted(classes(reply.workflow)) == ["LoadVideo", "SaveVideo", "Video Slice"]
    assert reply.input_files == ("viagem.mp4",)


async def test_builds_an_image_with_the_text_to_image_pipeline(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send("crie uma imagem widescreen de um gato astronauta flutuando no espaço")

    assert reply.status == "completed", reply.message
    assert reply.validation.ok, reply.validation.text()
    assert sorted(classes(reply.workflow)) == sorted(
        ["CheckpointLoaderSimple", "CLIPTextEncode", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"]
    )
    (loader,) = [node["inputs"] for node in reply.workflow.values() if node["class_type"] == "CheckpointLoaderSimple"]
    assert loader["ckpt_name"] == SDXL
    (latent,) = [node["inputs"] for node in reply.workflow.values() if node["class_type"] == "EmptyLatentImage"]
    assert (latent["width"], latent["height"]) == (1344, 768)


async def test_makes_a_short_video_from_text(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send("faça um vídeo de 3 segundos de um leão correndo na savana")

    assert reply.status == "completed", reply.message
    (latent,) = [node["inputs"] for node in reply.workflow.values() if node["class_type"] == "Wan22ImageToVideoLatent"]
    assert latent["length"] == 73
    assert "SaveVideo" in classes(reply.workflow)


async def test_animates_an_attached_image(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send(
        "anime esta imagem: o vermelho deve pulsar devagar, ficando mais claro e mais escuro",
        files={"fundo.png": png(1280, 720)},
    )

    assert reply.status == "completed", reply.message
    (latent,) = [node["inputs"] for node in reply.workflow.values() if node["class_type"] == "Wan22ImageToVideoLatent"]
    assert "start_image" in latent
    assert reply.input_files == ("fundo.png",)


async def test_asks_when_the_attached_image_does_not_match_the_request(inventory):
    agent = WorkflowAgent(inventory)

    # The picture is a plain red rectangle: there is no beach to animate.
    reply = await agent.send(
        "anime esta foto da praia, quero ver as ondas se mexendo", files={"praia.png": png(1280, 720)}
    )

    assert reply.status == "needs_input", reply.message
    assert reply.workflow == {}


async def test_says_so_when_the_server_cannot_make_music(inventory):
    agent = WorkflowAgent(inventory)

    reply = await agent.send("crie uma música de piano calma de 30 segundos")

    assert reply.status in ("needs_input", "completed"), reply.message
    assert reply.workflow == {}


async def test_edits_only_what_was_asked(inventory):
    built = Workflow()
    text_to_image(built)
    agent = WorkflowAgent.from_workflow(built.get_comfy_ui_workflow_format(), inventory)

    reply = await agent.send("mude o prompt para um farol ao entardecer e use 30 passos")

    assert reply.status == "completed", reply.message
    assert set(reply.changes.changed) == {"CLIPTextEncode_1", "KSampler_4"}
    assert not reply.changes.added and not reply.changes.removed
    assert reply.workflow["KSampler_4"]["inputs"]["steps"] == 30
    assert "lighthouse" in reply.workflow["CLIPTextEncode_1"]["inputs"]["text"].lower()


# ---------------------------------------------------------------- the server since 2026-10-23


def inputs_of(workflow: dict, class_type: str) -> dict:
    (node,) = [node["inputs"] for node in workflow.values() if node["class_type"] == class_type]
    return node


async def test_makes_music_with_the_audio_model(server_inventory):
    agent = WorkflowAgent(server_inventory)

    reply = await agent.send("crie uma música de piano calma de 30 segundos")

    assert reply.status == "completed", reply.message
    encoder = inputs_of(reply.workflow, "TextEncodeAceStepAudio1.5")
    assert encoder["duration"] == 30.0
    assert encoder["lyrics"] == "[instrumental]"
    assert "SaveAudioAdvanced" in classes(reply.workflow)


async def test_picks_the_model_with_sound_for_a_video_with_sound(server_inventory):
    agent = WorkflowAgent(server_inventory)

    reply = await agent.send(
        "faça um vídeo de 6 segundos de uma tempestade no mar, com o som das ondas e dos trovões"
    )

    assert reply.status == "completed", reply.message
    assert inputs_of(reply.workflow, "MiniMaxH3ImageToVideo")["length"] == 141
    assert "VAEDecodeAudio" in classes(reply.workflow)


async def test_makes_a_photo_talk_with_the_attached_audio(server_inventory):
    # Without vision the plain test image cannot look wrong to the model.
    agent = WorkflowAgent(server_inventory, vision=False)

    reply = await agent.send(
        "faça a pessoa desta foto falar este áudio de 4 segundos",
        files={"rosto.png": png(640, 640), "fala.wav": wav(4.0)},
    )

    assert reply.status == "completed", reply.message
    assert "WanSoundImageToVideo" in classes(reply.workflow)
    assert set(reply.input_files) == {"rosto.png", "fala.wav"}


async def test_upscales_with_the_upscale_model(server_inventory):
    agent = WorkflowAgent(server_inventory)

    reply = await agent.send(
        "aumente esta imagem 4 vezes, com o máximo de nitidez", files={"logo.png": png(256, 192)}
    )

    assert reply.status == "completed", reply.message
    assert "ImageUpscaleWithModel" in classes(reply.workflow)
