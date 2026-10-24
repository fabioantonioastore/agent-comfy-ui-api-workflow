"""The agent's workflows on the RunPod worker: run with --anthropic --runpod (paid).

The agent builds each workflow with the real Claude API, then the worker runs it.
"""

import asyncio

import pytest
from comfy_ui_api_workflow import Workflow

from agent_comfy_ui_api_workflow import JobResult, RunPodClient, ServerInventory, WorkflowAgent
from agent_comfy_ui_api_workflow._impl.runpod import OutputFile
from agent_comfy_ui_api_workflow._impl.session import api_json
from builders import text_to_image
from media import png, tone
from test_runpod_pipelines import folder, loudness, make_video, picture, probe, run, seconds, stream  # noqa: F401

pytestmark = [pytest.mark.anthropic, pytest.mark.runpod]


@pytest.fixture(scope="module")
def live_inventory() -> ServerInventory:
    async def discover():
        async with RunPodClient.from_env() as client:
            return await client.discover_inventory()

    return asyncio.run(discover())


def outputs(result: JobResult, agent: WorkflowAgent) -> list[OutputFile]:
    # The worker also returns some of the files it was sent; those are not results.
    return [file for file in result.files if file.filename not in agent.files]


async def test_the_image_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send("crie uma imagem quadrada de 512x512 de um farol ao entardecer, com 10 passos")

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    assert [picture(file.data).size for file in outputs(result, agent)] == [(512, 512)]


async def test_the_music_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send("crie uma música instrumental de violão de 10 segundos")

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (song,) = outputs(result, agent)
    extension = "." + song.filename.rsplit(".", 1)[1]
    assert seconds(probe(song.data, extension)) == pytest.approx(10.0, abs=0.5)


async def test_the_video_cut_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send(
        "corte este vídeo do segundo 1 ao segundo 2", files={"viagem.mp4": make_video(3.0)}
    )

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (clip,) = outputs(result, agent)
    info = probe(clip.data)
    assert seconds(info) == pytest.approx(1.0, abs=0.2)
    assert stream(info, "audio") is not None


async def test_the_upscale_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send("aumente esta imagem 2 vezes", files={"logo.png": png(64, 48)})

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    assert [picture(file.data).size for file in outputs(result, agent)] == [(128, 96)]


async def test_the_animation_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send(
        "anime esta foto: as cores pulsam devagar. Quero um vídeo de 1 segundo em 320x192",
        files={"foto.png": png(320, 192)},
    )

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (clip,) = outputs(result, agent)
    info = probe(clip.data)
    video = stream(info, "video")
    assert (video["width"], video["height"]) == (320, 192)
    assert seconds(info) == pytest.approx(1.0, abs=0.15)


async def test_the_talking_photo_the_agent_builds_runs(live_inventory, folder):
    # Without vision: the test photo is a flat color, not a person.
    agent = WorkflowAgent(live_inventory, vision=False)

    reply = await agent.send(
        "faça a pessoa desta foto falar este áudio, que dura 2 segundos, num vídeo quadrado de 320x320",
        files={"rosto.png": png(320, 320), "fala.wav": tone(2.0, rate=16000)},
    )

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (clip,) = outputs(result, agent)
    info = probe(clip.data)
    video = stream(info, "video")
    assert (video["width"], video["height"]) == (320, 320)
    assert stream(info, "audio") is not None


async def test_a_change_asked_in_a_second_turn_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)
    first = await agent.send("crie uma imagem de 512x512 de um farol, com 6 passos")
    assert first.status == "completed", first.message

    reply = await agent.send("agora faça a mesma imagem em 768x512")

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    assert [picture(file.data).size for file in outputs(result, agent)] == [(768, 512)]


async def test_an_exported_workflow_the_agent_edits_runs(live_inventory, folder):
    workflow = Workflow()
    text_to_image(workflow, prompt="a lighthouse at dusk")
    agent = WorkflowAgent.from_workflow(api_json(workflow), live_inventory)

    reply = await agent.send("mude o tamanho para 512x384 e use 8 passos")

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    assert [picture(file.data).size for file in outputs(result, agent)] == [(512, 384)]


async def test_the_join_the_agent_builds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send(
        "junte estes dois vídeos: primeiro o parte1 e depois o parte2",
        files={"parte1.mp4": make_video(1.0), "parte2.mp4": make_video(1.0)},
    )

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (joined,) = outputs(result, agent)
    assert seconds(probe(joined.data)) == pytest.approx(2.0, abs=0.2)


async def test_the_soundtrack_the_agent_adds_runs(live_inventory, folder):
    agent = WorkflowAgent(live_inventory)

    reply = await agent.send(
        "coloque esta música como trilha sonora deste vídeo",
        files={"passeio.mp4": make_video(2.0, tone=False), "musica.wav": tone(5.0)},
    )

    assert reply.status == "completed", reply.message
    result = await run(reply.workflow, folder, agent.files)
    (scored,) = outputs(result, agent)
    assert float(stream(probe(scored.data), "audio")["duration"]) == pytest.approx(2.0, abs=0.15)
    assert loudness(scored.data) > -30
