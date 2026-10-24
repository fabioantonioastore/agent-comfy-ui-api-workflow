"""The RunPod client on the real endpoint: run with --runpod (paid, little GPU time)."""

import asyncio
import os

import httpx
import pytest
from comfy_ui_api_workflow import Workflow

from agent_comfy_ui_api_workflow import RunPodClient
from agent_comfy_ui_api_workflow._impl.env import load_settings
from agent_comfy_ui_api_workflow._impl.errors import JobTimeoutError, RunPodError
from agent_comfy_ui_api_workflow._impl.pipelines.generation import TextToImage
from agent_comfy_ui_api_workflow._impl.pipelines.media import ConcatenateVideos, UpscaleImage
from agent_comfy_ui_api_workflow._impl.runpod import SETTINGS, endpoint_url
from agent_comfy_ui_api_workflow._impl.session import api_json
from media import png
from test_runpod_pipelines import family, make_video, models, picture  # noqa: F401

pytestmark = pytest.mark.runpod


def upscale(name: str) -> dict:
    """A job that doubles the image `name`, saved with its stem as the prefix."""
    workflow = Workflow()
    workflow.add_pipeline(UpscaleImage, image=name, scale=2.0, filename_prefix=name.removesuffix(".png"))
    return api_json(workflow)


async def job_status(job_id: str) -> str:
    settings = load_settings(SETTINGS)
    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{endpoint_url(settings['RUNPOD_ENDPOINT_ID'])}/status/{job_id}",
            headers={"Authorization": f"Bearer {settings['RUNPOD_API_KEY']}"},
        )
    return response.json()["status"]


async def test_jobs_run_at_the_same_time_on_one_client():
    sizes = {"a.png": (64, 48), "b.png": (32, 32), "c.png": (40, 20)}

    async with RunPodClient.from_env() as client:
        results = await asyncio.gather(*(client.run(upscale(name), {name: png(*size)}) for name, size in sizes.items()))

    for (name, (width, height)), result in zip(sizes.items(), results):
        assert result.ok, result.error
        (image,) = [file for file in result.files if file.filename.startswith(name.removesuffix(".png") + "_")]
        assert picture(image.data).size == (2 * width, 2 * height)


async def test_a_node_that_fails_fails_the_job():
    # The worker calls this job COMPLETED, since it returns the videos it loaded.
    workflow = Workflow()
    workflow.add_pipeline(ConcatenateVideos, videos=["small.mp4", "big.mp4"], filename_prefix="mixed")
    files = {"small.mp4": make_video(1.0, size="160x96"), "big.mp4": make_video(1.0, size="320x192")}

    async with RunPodClient.from_env() as client:
        result = await client.run(api_json(workflow), files)

    assert not result.ok
    assert "incompatible frame dimensions" in result.error
    assert not [file for file in result.files if file.filename.startswith("mixed")]


async def test_a_file_the_job_was_not_sent_fails_with_its_name():
    async with RunPodClient.from_env() as client:
        result = await client.run(upscale("never_sent.png"))

    assert not result.ok
    assert "never_sent.png" in result.error


async def test_files_over_the_request_limit_are_refused_with_the_reason():
    # The files go base64 encoded in the request, which RunPod takes up to 10 MiB.
    async with RunPodClient.from_env() as client:
        with pytest.raises(RunPodError) as error:
            await client.run(upscale("big.png"), {"big.png": os.urandom(8 * 1024 * 1024)})

    assert "HTTP 400" in str(error.value)
    assert "10MiB" in str(error.value)


async def test_a_job_that_outlasts_its_timeout_is_cancelled_on_the_server(models):
    workflow = Workflow()
    workflow.add_pipeline(TextToImage, prompt="a dense forest", model=family(models, "sdxl"), batch_size=4, steps=40, seed=1)

    async with RunPodClient.from_env(poll_interval=0.5) as client:
        with pytest.raises(JobTimeoutError) as error:
            await client.run(api_json(workflow), timeout=2.0)

    assert "was cancelled" in str(error.value)
    for _ in range(10):
        status = await job_status(error.value.job_id)
        if status == "CANCELLED":
            break
        await asyncio.sleep(1)
    assert status == "CANCELLED"
