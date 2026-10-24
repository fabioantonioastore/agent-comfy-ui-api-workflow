"""Every pipeline on the RunPod worker: run with --runpod (paid GPU time).

The files each job returns land in runpod_outputs/<test name>/, next to the workflow.
"""

import asyncio
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from comfy_ui_api_workflow import Workflow
from PIL import Image, ImageDraw

from agent_comfy_ui_api_workflow import JobResult, RunPodClient
from agent_comfy_ui_api_workflow._impl.families import ModelChoice, find_models
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
from agent_comfy_ui_api_workflow._impl.pipelines.media import (
    AddAudioToVideo,
    CombineAudio,
    ConcatenateVideos,
    EditAudio,
    FramesToVideo,
    LoadVideoParts,
    TrimVideo,
    UpscaleImage,
    VideoToAnimation,
)
from agent_comfy_ui_api_workflow._impl.session import api_json
from media import png, tone, wav

pytestmark = pytest.mark.runpod

OUTPUTS = Path(__file__).resolve().parent.parent / "runpod_outputs"
GPU_TIMEOUT = 1800.0
GRAY = (128, 128, 128)


@pytest.fixture(scope="session")
def models() -> dict[str, ModelChoice]:
    async def discover():
        async with RunPodClient.from_env() as client:
            return await client.discover_inventory()

    return {model.family: model for model in find_models(asyncio.run(discover()))}


def family(models: dict[str, ModelChoice], name: str) -> ModelChoice:
    if name not in models:
        pytest.skip(f"the server has no {name} model")
    return models[name]


@pytest.fixture
def folder(request) -> Path:
    path = OUTPUTS / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


async def run(
    workflow: Workflow | dict[str, Any], folder: Path, files: dict[str, bytes] | None = None
) -> JobResult:
    payload = api_json(workflow) if isinstance(workflow, Workflow) else workflow
    (folder / "workflow.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    async with RunPodClient.from_env() as client:
        result = await client.run(payload, files, timeout=GPU_TIMEOUT)
    for file in result.files:
        (folder / file.filename).write_bytes(file.data)
    timing = {"execution_ms": result.execution_ms, "delay_ms": result.delay_ms}
    (folder / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    assert result.ok, f"{result.status}: {result.error}"
    return result


def saved(result: JobResult, prefix: str) -> list[bytes]:
    # The worker also returns previews of some inputs; the prefix picks the saved files.
    return [file.data for file in result.files if file.filename.startswith(f"{prefix}_")]


def one(result: JobResult, prefix: str) -> bytes:
    found = saved(result, prefix)
    assert len(found) == 1, [file.filename for file in result.files]
    return found[0]


def loudness(data: bytes, suffix: str = ".mp4") -> float:
    """The mean volume of the audio, in dB; -inf for digital silence."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as file:
        file.write(data)
        file.flush()
        output = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", file.name, "-vn", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=True,
        )
    return float(re.search(r"mean_volume: (\S+) dB", output.stderr).group(1))


def square_mask(size: int = 512, box: tuple[int, int, int, int] = (192, 192, 320, 320)) -> bytes:
    """A black image with a white square: the area an inpaint repaints."""
    image = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def with_hole(size: int = 512, box: tuple[int, int, int, int] = (192, 192, 320, 320)) -> bytes:
    """A gray image with a transparent square: the area an inpaint repaints without a mask."""
    image = Image.new("RGBA", (size, size), (*GRAY, 255))
    ImageDraw.Draw(image).rectangle(box, fill=(0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def picture(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def probe(data: bytes, suffix: str = ".mp4") -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix) as file:
        file.write(data)
        file.flush()
        output = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", file.name],
            capture_output=True,
            check=True,
        )
    return json.loads(output.stdout)


def stream(info: dict, kind: str) -> dict | None:
    return next((item for item in info["streams"] if item["codec_type"] == kind), None)


def seconds(info: dict) -> float:
    return float(info["format"]["duration"])


def make_video(duration: float = 2.0, fps: int = 12, size: str = "160x96", tone: bool = True) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "input.mp4"
        command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={fps}"]
        if tone:
            command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(target)]
        subprocess.run(command, check=True)
        return target.read_bytes()


# ---------------------------------------------------------------- generation


async def test_text_to_image(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToImage,
        prompt="a red apple on a wooden table, studio photo",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        steps=10,
        seed=7,
        filename_prefix="apple",
    )

    result = await run(workflow, folder)

    assert picture(one(result, "apple")).size == (512, 512)


async def test_text_to_image_second_pass_upscales(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToImage,
        prompt="a lighthouse at dusk",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        steps=8,
        seed=7,
        upscale=1.5,
        filename_prefix="hires",
    )

    result = await run(workflow, folder)

    assert picture(one(result, "hires")).size == (768, 768)


async def test_image_to_image_works_at_one_megapixel(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        ImageToImage,
        image="source.png",
        prompt="an abstract watercolor painting",
        model=family(models, "sdxl"),
        strength=0.5,
        steps=8,
        seed=7,
        filename_prefix="restyled",
    )

    result = await run(workflow, folder, {"source.png": png(256, 256)})

    width, height = picture(one(result, "restyled")).size
    assert width == height
    assert abs(width * height - 1024 * 1024) < 0.1 * 1024 * 1024


async def test_inpaint_keeps_the_pixels_outside_the_region(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        Inpaint,
        image="canvas.png",
        prompt="a shiny red apple on a gray table, studio photo",
        model=family(models, "sdxl"),
        region=[192, 192, 128, 128],
        steps=10,
        seed=7,
        filename_prefix="inpainted",
    )

    result = await run(workflow, folder, {"canvas.png": png(512, 512, GRAY)})

    image = picture(one(result, "inpainted")).convert("RGB")
    assert image.size == (512, 512)
    assert all(abs(a - b) <= 2 for a, b in zip(image.getpixel((10, 10)), GRAY))
    assert image.getpixel((256, 256)) != GRAY


async def test_outpaint_extends_the_canvas(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        Outpaint,
        image="beach.png",
        prompt="a wide beach with palm trees",
        model=family(models, "sdxl"),
        right=128,
        steps=10,
        seed=7,
        filename_prefix="wider",
    )

    result = await run(workflow, folder, {"beach.png": png(512, 512, (230, 200, 140))})

    assert picture(one(result, "wider")).size == (640, 512)


async def test_text_to_video(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="a paper boat drifting on a calm lake",
        model=family(models, "wan22_5b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        steps=6,
        seed=1,
        filename_prefix="boat",
    )

    result = await run(workflow, folder)

    video = stream(probe(one(result, "boat")), "video")
    assert (video["width"], video["height"]) == (320, 192)
    assert int(video["nb_read_frames"]) == 25
    assert video["r_frame_rate"] == "24/1"


async def test_image_to_video_with_a_soundtrack(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        ImageToVideo,
        image="start.png",
        prompt="the colors slowly pulse",
        model=family(models, "wan22_5b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        steps=6,
        seed=1,
        audio="tone.wav",
        filename_prefix="pulse",
    )

    result = await run(workflow, folder, {"start.png": png(320, 192), "tone.wav": wav(3.0)})

    info = probe(one(result, "pulse"))
    assert int(stream(info, "video")["nb_read_frames"]) == 25
    assert stream(info, "audio") is not None
    assert seconds(info) == pytest.approx(25 / 24, abs=0.15)


async def test_text_to_video_with_wan_14b(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="a paper boat drifting on a calm lake",
        model=family(models, "wan22_t2v_14b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        seed=1,
        filename_prefix="boat14b",
    )

    result = await run(workflow, folder)

    video = stream(probe(one(result, "boat14b")), "video")
    assert (video["width"], video["height"]) == (320, 192)
    assert int(video["nb_read_frames"]) == 17
    assert video["r_frame_rate"] == "16/1"


async def test_image_to_video_with_wan_14b_ends_on_an_image(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        ImageToVideo,
        image="start.png",
        end_image="end.png",
        prompt="the color slowly changes",
        model=family(models, "wan22_i2v_14b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        seed=1,
        filename_prefix="fade",
    )

    result = await run(workflow, folder, {"start.png": png(320, 192, (200, 40, 40)), "end.png": png(320, 192, (40, 40, 200))})

    assert int(stream(probe(one(result, "fade")), "video")["nb_read_frames"]) == 17


async def test_text_to_video_with_sound_from_minimax_h3(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="rain falls on a tin roof; the steady sound of the rain and distant thunder",
        model=family(models, "minimax_h3"),
        width=384,
        height=224,
        duration_seconds=1.0,
        seed=1,
        filename_prefix="rain",
    )

    result = await run(workflow, folder)

    info = probe(one(result, "rain"))
    video = stream(info, "video")
    assert (video["width"], video["height"]) == (384, 224)
    assert int(video["nb_read_frames"]) == 22
    assert video["r_frame_rate"] == "24/1"
    assert stream(info, "audio") is not None


async def test_image_to_video_with_minimax_h3_between_two_images(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        ImageToVideo,
        image="start.png",
        end_image="end.png",
        prompt="the color slowly changes while a soft chime rings",
        model=family(models, "minimax_h3"),
        width=384,
        height=224,
        duration_seconds=1.0,
        seed=1,
        filename_prefix="chime",
    )

    result = await run(workflow, folder, {"start.png": png(384, 224, (200, 40, 40)), "end.png": png(384, 224, (40, 40, 200))})

    info = probe(one(result, "chime"))
    assert int(stream(info, "video")["nb_read_frames"]) == 22
    assert stream(info, "audio") is not None


async def test_sound_to_video_in_two_parts_keeps_the_audio(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        SoundToVideo,
        image="face.png",
        audio="talk.wav",
        prompt="a person talks to the camera",
        model=family(models, "wan22_s2v"),
        width=320,
        height=320,
        duration_seconds=9.6,
        seed=1,
        filename_prefix="talk",
    )

    result = await run(workflow, folder, {"face.png": png(320, 320), "talk.wav": wav(12.0, rate=16000)})

    info = probe(one(result, "talk"))
    video = stream(info, "video")
    assert (video["width"], video["height"]) == (320, 320)
    assert video["r_frame_rate"] == "16/1"
    assert int(video["nb_read_frames"]) == 159
    assert float(stream(info, "audio")["duration"]) == pytest.approx(159 / 16, abs=0.1)


async def test_text_to_audio_makes_music(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToAudio,
        style="calm acoustic guitar instrumental",
        model=family(models, "ace_step_15"),
        duration_seconds=10.0,
        bpm=90,
        seed=1,
        filename_prefix="song",
    )

    result = await run(workflow, folder)

    assert seconds(probe(one(result, "song"), ".mp3")) == pytest.approx(10.0, abs=0.3)


async def test_upscale_image_with_model(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        UpscaleImageWithModel, image="small.png", model=family(models, "upscaler"), scale=2.0, filename_prefix="sharper"
    )

    result = await run(workflow, folder, {"small.png": png(64, 48)})

    assert picture(one(result, "sharper")).size == (128, 96)


# ---------------------------------------------------------------- media


async def test_upscale_image(folder):
    workflow = Workflow()
    workflow.add_pipeline(UpscaleImage, image="small.png", scale=2.0, filename_prefix="bigger")

    result = await run(workflow, folder, {"small.png": png(64, 48)})

    assert picture(one(result, "bigger")).size == (128, 96)


async def test_trim_video_keeps_the_sound(folder):
    workflow = Workflow()
    workflow.add_pipeline(TrimVideo, video="input.mp4", start_seconds=1.0, duration_seconds=1.0, filename_prefix="cut")

    result = await run(workflow, folder, {"input.mp4": make_video(3.0)})

    info = probe(one(result, "cut"))
    assert seconds(info) == pytest.approx(1.0, abs=0.15)
    assert stream(info, "audio") is not None


async def test_frames_round_trip_keeps_timing_and_sound(folder):
    workflow = Workflow()
    parts = workflow.add_pipeline(LoadVideoParts, video="input.mp4").output
    workflow.add_pipeline(
        FramesToVideo, frames=parts["frames"], fps=parts["fps"], audio=parts["audio"], filename_prefix="rebuilt"
    )

    result = await run(workflow, folder, {"input.mp4": make_video(2.0)})

    info = probe(one(result, "rebuilt"))
    assert int(stream(info, "video")["nb_read_frames"]) == 24
    assert stream(info, "audio") is not None


async def test_add_audio_to_video_cuts_the_sound(folder):
    workflow = Workflow()
    workflow.add_pipeline(AddAudioToVideo, video="silent.mp4", audio="long.wav", filename_prefix="scored")

    result = await run(workflow, folder, {"silent.mp4": make_video(2.0, tone=False), "long.wav": wav(5.0)})

    info = probe(one(result, "scored"))
    assert float(stream(info, "audio")["duration"]) == pytest.approx(2.0, abs=0.15)


async def test_concatenate_videos(folder):
    workflow = Workflow()
    workflow.add_pipeline(ConcatenateVideos, videos=["a.mp4", "b.mp4"], filename_prefix="joined")

    result = await run(workflow, folder, {"a.mp4": make_video(1.0), "b.mp4": make_video(1.0)})

    assert seconds(probe(one(result, "joined"))) == pytest.approx(2.0, abs=0.2)


async def test_video_to_animation(folder):
    workflow = Workflow()
    workflow.add_pipeline(VideoToAnimation, video="input.mp4", filename_prefix="loop")

    result = await run(workflow, folder, {"input.mp4": make_video(1.0)})

    assert picture(one(result, "loop")).n_frames == 12


async def test_edit_audio_comes_back_as_a_file(folder):
    # Whether the worker returns audio saved on its own decides how audio is delivered.
    workflow = Workflow()
    workflow.add_pipeline(EditAudio, audio="talk.wav", start_seconds=0.5, duration_seconds=1.0, volume_db=-6, filename_prefix="clip")

    result = await run(workflow, folder, {"talk.wav": wav(3.0, rate=16000)})

    assert seconds(probe(one(result, "clip"), ".mp3")) == pytest.approx(1.0, abs=0.15)


async def test_combine_audio_in_sequence(folder):
    workflow = Workflow()
    workflow.add_pipeline(CombineAudio, audios=["a.wav", "b.wav"], format="flac", filename_prefix="both")

    result = await run(workflow, folder, {"a.wav": wav(1.0), "b.wav": wav(2.0)})

    assert seconds(probe(one(result, "both"), ".flac")) == pytest.approx(3.0, abs=0.15)


async def test_combine_audio_mixed_lasts_as_long_as_the_first(folder):
    workflow = Workflow()
    workflow.add_pipeline(CombineAudio, audios=["short.wav", "long.wav"], mode="mix", format="flac", filename_prefix="cut")
    workflow.add_pipeline(CombineAudio, audios=["long.wav", "short.wav"], mode="mix", format="flac", filename_prefix="kept")

    result = await run(workflow, folder, {"short.wav": wav(1.0), "long.wav": wav(2.0)})

    assert seconds(probe(one(result, "cut"), ".flac")) == pytest.approx(1.0, abs=0.15)
    assert seconds(probe(one(result, "kept"), ".flac")) == pytest.approx(2.0, abs=0.15)


async def test_concatenate_three_videos(folder):
    workflow = Workflow()
    workflow.add_pipeline(ConcatenateVideos, videos=["a.mp4", "b.mp4", "c.mp4"], filename_prefix="three")

    clips = {name: make_video(1.0) for name in ("a.mp4", "b.mp4", "c.mp4")}
    result = await run(workflow, folder, clips)

    assert seconds(probe(one(result, "three"))) == pytest.approx(3.0, abs=0.25)


async def test_video_to_animation_as_png(folder):
    workflow = Workflow()
    workflow.add_pipeline(VideoToAnimation, video="input.mp4", format="png", filename_prefix="apng")

    result = await run(workflow, folder, {"input.mp4": make_video(1.0)})

    assert picture(one(result, "apng")).n_frames == 12


# ---------------------------------------------------------------- more generation cases


async def test_text_to_image_makes_a_batch(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToImage,
        prompt="a red apple on a wooden table",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        batch_size=2,
        steps=8,
        seed=7,
        filename_prefix="pair",
    )

    result = await run(workflow, folder)

    images = saved(result, "pair")
    assert [picture(data).size for data in images] == [(512, 512), (512, 512)]


async def test_inpaint_with_a_mask_changes_only_the_masked_area(models, folder):
    # A partial strength changes what is there, so the image needs content to change.
    workflow = Workflow()
    still = workflow.add_pipeline(
        TextToImage,
        prompt="a vase of red roses on a wooden table",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        steps=10,
        seed=7,
        filename_prefix="original",
    ).output
    workflow.add_pipeline(
        Inpaint,
        image=still["images"],
        mask="mask.png",
        prompt="a vase of bright blue tulips on a wooden table",
        model=family(models, "sdxl"),
        strength=0.6,
        steps=10,
        seed=7,
        filename_prefix="masked",
    )

    result = await run(workflow, folder, {"mask.png": square_mask()})

    original = picture(one(result, "original")).convert("RGB")
    masked = picture(one(result, "masked")).convert("RGB")
    for point in ((20, 20), (500, 40), (60, 480)):
        assert all(abs(a - b) <= 2 for a, b in zip(masked.getpixel(point), original.getpixel(point)))
    box = (200, 200, 312, 312)
    changes = [
        sum(abs(a - b) for a, b in zip(new, old))
        for new, old in zip(masked.crop(box).get_flattened_data(), original.crop(box).get_flattened_data())
    ]
    assert sum(changes) / len(changes) > 10


async def test_outpaint_on_every_side(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        Outpaint,
        image="room.png",
        prompt="a cozy living room",
        model=family(models, "sdxl"),
        left=64,
        top=64,
        right=64,
        bottom=64,
        steps=10,
        seed=7,
        filename_prefix="roomier",
    )

    result = await run(workflow, folder, {"room.png": png(512, 512, (180, 150, 120))})

    image = picture(one(result, "roomier")).convert("RGB")
    assert image.size == (640, 640)
    assert all(abs(a - b) <= 2 for a, b in zip(image.getpixel((320, 320)), (180, 150, 120)))


async def test_wan_14b_without_the_fast_loras(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="a paper boat drifting on a calm lake",
        model=family(models, "wan22_t2v_14b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        fast=False,
        steps=6,
        seed=1,
        filename_prefix="slow",
    )

    result = await run(workflow, folder)

    assert int(stream(probe(one(result, "slow")), "video")["nb_read_frames"]) == 17


async def test_minimax_h3_animates_a_start_image(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        ImageToVideo,
        image="start.png",
        prompt="the red light slowly dims while a low hum fades out",
        model=family(models, "minimax_h3"),
        width=384,
        height=224,
        duration_seconds=1.0,
        seed=1,
        filename_prefix="dim",
    )

    result = await run(workflow, folder, {"start.png": png(384, 224, (200, 40, 40))})

    info = probe(one(result, "dim"))
    assert int(stream(info, "video")["nb_read_frames"]) == 22
    assert stream(info, "audio")["channels"] == 2


async def test_a_soundtrack_replaces_the_sound_of_minimax_h3(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="a dog barks loudly in a park",
        model=family(models, "minimax_h3"),
        width=384,
        height=224,
        duration_seconds=1.0,
        seed=6,
        audio="quiet.wav",
        filename_prefix="quiet",
    )

    result = await run(workflow, folder, {"quiet.wav": wav(2.0)})

    data = one(result, "quiet")
    assert stream(probe(data), "audio") is not None
    # The user's track is silence; the model's barking would be loud.
    assert loudness(data) < -60


async def test_a_song_with_lyrics(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToAudio,
        style="acoustic folk, warm male vocals, guitar",
        lyrics="[verse]\nCaminhando pela estrada\nCom o sol no coração\n[chorus]\nVou cantar esta canção",
        language="pt",
        model=family(models, "ace_step_15"),
        duration_seconds=15.0,
        bpm=100,
        key="G major",
        seed=5,
        filename_prefix="folk",
    )

    result = await run(workflow, folder)

    data = one(result, "folk")
    info = probe(data, ".mp3")
    assert seconds(info) == pytest.approx(15.0, abs=0.3)
    assert stream(info, "audio")["channels"] == 2
    assert loudness(data, ".mp3") > -40


async def test_upscale_image_with_model_at_its_own_factor(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(UpscaleImageWithModel, image="small.png", model=family(models, "upscaler"), filename_prefix="x4")

    result = await run(workflow, folder, {"small.png": png(64, 48)})

    assert picture(one(result, "x4")).size == (256, 192)


# ---------------------------------------------------------------- chains: one pipeline's output feeds another


async def test_a_generated_image_is_animated(models, folder):
    workflow = Workflow()
    still = workflow.add_pipeline(
        TextToImage,
        prompt="a sailboat on a lake, oil painting",
        model=family(models, "sdxl"),
        width=512,
        height=320,
        steps=10,
        seed=3,
        save=False,
    ).output
    workflow.add_pipeline(
        ImageToVideo,
        image=still["images"],
        prompt="the boat sails slowly across the lake",
        model=family(models, "wan22_i2v_14b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        seed=3,
        filename_prefix="sail",
    )

    result = await run(workflow, folder)

    video = stream(probe(one(result, "sail")), "video")
    assert (video["width"], video["height"], int(video["nb_read_frames"])) == (320, 192, 17)
    assert not [file.filename for file in result.files if file.filename.endswith(".png")]


async def test_generated_music_drives_a_talking_photo(models, folder):
    workflow = Workflow()
    music = workflow.add_pipeline(
        TextToAudio,
        style="cheerful pop song, female vocals",
        lyrics="[verse]\nLa la la, sing along with me",
        model=family(models, "ace_step_15"),
        duration_seconds=4.0,
        seed=2,
        save=False,
    ).output
    workflow.add_pipeline(
        SoundToVideo,
        image="face.png",
        audio=music["audio"],
        prompt="a person sings happily",
        model=family(models, "wan22_s2v"),
        width=320,
        height=320,
        duration_seconds=2.0,
        seed=2,
        filename_prefix="sing",
    )

    result = await run(workflow, folder, {"face.png": png(320, 320)})

    info = probe(one(result, "sing"))
    assert int(stream(info, "video")["nb_read_frames"]) == 36
    assert float(stream(info, "audio")["duration"]) == pytest.approx(36 / 16, abs=0.1)
    assert loudness(one(result, "sing")) > -50


async def test_a_video_upscaled_with_the_model_keeps_its_timing_and_sound(models, folder):
    workflow = Workflow()
    parts = workflow.add_pipeline(LoadVideoParts, video="input.mp4").output
    bigger = workflow.add_pipeline(
        UpscaleImageWithModel, image=parts["frames"], model=family(models, "upscaler"), scale=2.0, save=False
    ).output
    workflow.add_pipeline(
        FramesToVideo, frames=bigger["image"], fps=parts["fps"], audio=parts["audio"], filename_prefix="sharp"
    )

    result = await run(workflow, folder, {"input.mp4": make_video(2.0)})

    info = probe(one(result, "sharp"))
    video = stream(info, "video")
    assert (video["width"], video["height"], int(video["nb_read_frames"])) == (320, 192, 24)
    assert stream(info, "audio") is not None


async def test_a_music_video(models, folder):
    workflow = Workflow()
    music = workflow.add_pipeline(
        TextToAudio,
        style="epic orchestral fanfare",
        model=family(models, "ace_step_15"),
        duration_seconds=3.0,
        seed=4,
        save=False,
    ).output
    workflow.add_pipeline(
        TextToVideo,
        prompt="fireworks over a city at night",
        model=family(models, "wan22_5b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        steps=6,
        seed=4,
        audio=music["audio"],
        filename_prefix="fireworks",
    )

    result = await run(workflow, folder)

    data = one(result, "fireworks")
    assert float(stream(probe(data), "audio")["duration"]) == pytest.approx(25 / 24, abs=0.1)
    assert loudness(data) > -50


async def test_a_generated_image_is_restyled(models, folder):
    sdxl = family(models, "sdxl")
    workflow = Workflow()
    photo = workflow.add_pipeline(
        TextToImage,
        prompt="a red barn in a green field, photo",
        model=sdxl,
        width=512,
        height=384,
        steps=8,
        seed=3,
        filename_prefix="photo",
    ).output
    workflow.add_pipeline(
        ImageToImage,
        image=photo["images"],
        prompt="a red barn in a green field, watercolor painting",
        model=sdxl,
        strength=0.6,
        steps=8,
        seed=3,
        filename_prefix="painting",
    )

    result = await run(workflow, folder)

    assert picture(one(result, "photo")).size == (512, 384)
    width, height = picture(one(result, "painting")).size
    assert width / height == pytest.approx(512 / 384, rel=0.02)
    assert abs(width * height - 1024 * 1024) < 0.1 * 1024 * 1024


async def test_generated_images_become_a_slideshow(models, folder):
    workflow = Workflow()
    stills = workflow.add_pipeline(
        TextToImage,
        prompt="a watercolor of a cat",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        batch_size=3,
        steps=4,
        seed=1,
        save=False,
    ).output
    workflow.add_pipeline(FramesToVideo, frames=stills["images"], fps=1.0, filename_prefix="slides")

    result = await run(workflow, folder)

    info = probe(one(result, "slides"))
    assert int(stream(info, "video")["nb_read_frames"]) == 3
    assert seconds(info) == pytest.approx(3.0, abs=0.1)
    assert stream(info, "audio") is None


async def test_two_generated_clips_are_joined(models, folder):
    # The way to a video longer than the model makes: clips joined one after the other.
    wan = family(models, "wan22_5b")
    workflow = Workflow()
    clips = [
        workflow.add_pipeline(
            TextToVideo,
            prompt=prompt,
            model=wan,
            width=320,
            height=192,
            duration_seconds=1.0,
            steps=6,
            seed=1,
            save=False,
        ).output["video"]
        for prompt in ("a paper boat drifting on a calm lake", "the paper boat reaches a sandy shore")
    ]
    workflow.add_pipeline(ConcatenateVideos, videos=clips, filename_prefix="story")

    result = await run(workflow, folder)

    video = stream(probe(one(result, "story")), "video")
    assert (video["width"], video["height"], int(video["nb_read_frames"])) == (320, 192, 50)


async def test_a_generated_clip_is_cut_into_an_animation(models, folder):
    workflow = Workflow()
    clip = workflow.add_pipeline(
        TextToVideo,
        prompt="a candle flame flickering in the dark",
        model=family(models, "wan22_5b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        steps=6,
        seed=2,
        save=False,
    ).output
    cut = workflow.add_pipeline(TrimVideo, video=clip["video"], start_seconds=0.5, save=False).output
    workflow.add_pipeline(VideoToAnimation, video=cut["video"], format="png", filename_prefix="flame")

    result = await run(workflow, folder)

    animation = picture(one(result, "flame"))
    assert (animation.size, animation.n_frames) == ((320, 192), 13)
    assert animation.info["duration"] == pytest.approx(1000 / 24, abs=1)


async def test_a_generated_portrait_talks(models, folder):
    workflow = Workflow()
    portrait = workflow.add_pipeline(
        TextToImage,
        prompt="a portrait photo of a smiling woman looking at the camera",
        model=family(models, "sdxl"),
        width=512,
        height=512,
        steps=10,
        seed=4,
        save=False,
    ).output
    workflow.add_pipeline(
        SoundToVideo,
        image=portrait["images"],
        audio="talk.wav",
        prompt="the woman talks to the camera",
        model=family(models, "wan22_s2v"),
        width=320,
        height=320,
        duration_seconds=2.0,
        seed=4,
        filename_prefix="speech",
    )

    result = await run(workflow, folder, {"talk.wav": tone(2.0, rate=16000)})

    info = probe(one(result, "speech"))
    video = stream(info, "video")
    assert (video["width"], video["height"], int(video["nb_read_frames"])) == (320, 320, 36)
    # A shorter audio is not padded: the video ends in silence.
    assert float(stream(info, "audio")["duration"]) == pytest.approx(2.0, abs=0.1)


async def test_generated_music_is_added_to_a_video(models, folder):
    workflow = Workflow()
    music = workflow.add_pipeline(
        TextToAudio,
        style="upbeat electronic dance music",
        model=family(models, "ace_step_15"),
        duration_seconds=4.0,
        seed=1,
        save=False,
    ).output
    workflow.add_pipeline(AddAudioToVideo, video="silent.mp4", audio=music["audio"], filename_prefix="scored")

    result = await run(workflow, folder, {"silent.mp4": make_video(2.0, tone=False)})

    data = one(result, "scored")
    assert float(stream(probe(data), "audio")["duration"]) == pytest.approx(2.0, abs=0.15)
    assert loudness(data) > -50


async def test_generated_music_is_mixed_under_a_voice(models, folder):
    workflow = Workflow()
    music = workflow.add_pipeline(
        TextToAudio,
        style="soft ambient piano",
        model=family(models, "ace_step_15"),
        duration_seconds=5.0,
        seed=1,
        save=False,
    ).output
    workflow.add_pipeline(
        CombineAudio, audios=["voice.wav", music["audio"]], mode="mix", format="flac", filename_prefix="podcast"
    )

    result = await run(workflow, folder, {"voice.wav": tone(2.0, rate=16000)})

    info = probe(one(result, "podcast"), ".flac")
    assert seconds(info) == pytest.approx(2.0, abs=0.05)
    assert stream(info, "audio")["channels"] == 2


# ---------------------------------------------------------------- more media cases: formats, ranges, links


async def test_trim_video_until_the_end(folder):
    workflow = Workflow()
    workflow.add_pipeline(TrimVideo, video="input.mp4", start_seconds=1.0, filename_prefix="rest")

    result = await run(workflow, folder, {"input.mp4": make_video(3.0)})

    assert seconds(probe(one(result, "rest"))) == pytest.approx(2.0, abs=0.15)


async def test_a_video_saved_as_webm_is_av1_with_opus_sound(folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TrimVideo, video="input.mp4", start_seconds=0.5, duration_seconds=1.0, format="webm", filename_prefix="clip"
    )

    result = await run(workflow, folder, {"input.mp4": make_video(2.0)})

    info = probe(one(result, "clip"), ".webm")
    assert "webm" in info["format"]["format_name"]
    assert stream(info, "video")["codec_name"] == "av1"
    assert stream(info, "audio")["codec_name"] == "opus"
    assert seconds(info) == pytest.approx(1.0, abs=0.15)


async def test_cuts_of_a_video_are_joined_in_a_new_order(folder):
    workflow = Workflow()
    start = workflow.add_pipeline(TrimVideo, video="input.mp4", duration_seconds=1.0, save=False).output
    end = workflow.add_pipeline(TrimVideo, video="input.mp4", start_seconds=2.0, save=False).output
    workflow.add_pipeline(ConcatenateVideos, videos=[end["video"], start["video"]], filename_prefix="swapped")

    result = await run(workflow, folder, {"input.mp4": make_video(3.0)})

    info = probe(one(result, "swapped"))
    assert seconds(info) == pytest.approx(2.0, abs=0.2)
    assert stream(info, "audio") is not None


async def test_video_to_animation_of_frames_at_a_given_rate(folder):
    workflow = Workflow()
    parts = workflow.add_pipeline(LoadVideoParts, video="input.mp4").output
    workflow.add_pipeline(VideoToAnimation, frames=parts["frames"], fps=6.0, format="png", filename_prefix="slow")

    result = await run(workflow, folder, {"input.mp4": make_video(1.0)})

    animation = picture(one(result, "slow"))
    assert animation.n_frames == 12
    assert animation.info["duration"] == pytest.approx(1000 / 6, abs=1)


async def test_edit_audio_counts_a_negative_start_from_the_end(folder):
    workflow = Workflow()
    workflow.add_pipeline(EditAudio, audio="talk.wav", start_seconds=-1.0, format="flac", filename_prefix="tail")

    result = await run(workflow, folder, {"talk.wav": tone(3.0)})

    assert seconds(probe(one(result, "tail"), ".flac")) == pytest.approx(1.0, abs=0.05)


async def test_edit_audio_changes_the_volume_in_decibels(folder):
    workflow = Workflow()
    workflow.add_pipeline(EditAudio, audio="tone.wav", format="flac", filename_prefix="same")
    workflow.add_pipeline(EditAudio, audio="tone.wav", volume_db=-6, format="flac", filename_prefix="lower")

    result = await run(workflow, folder, {"tone.wav": tone(2.0)})

    change = loudness(one(result, "lower"), ".flac") - loudness(one(result, "same"), ".flac")
    assert change == pytest.approx(-6.0, abs=0.5)


async def test_audio_saved_as_opus(folder):
    workflow = Workflow()
    workflow.add_pipeline(EditAudio, audio="talk.wav", format="opus", filename_prefix="voice")

    result = await run(workflow, folder, {"talk.wav": tone(2.0, rate=16000)})

    info = probe(one(result, "voice"), ".opus")
    assert stream(info, "audio")["codec_name"] == "opus"
    assert seconds(info) == pytest.approx(2.0, abs=0.05)


async def test_combine_audio_of_different_rates_and_channels(folder):
    workflow = Workflow()
    workflow.add_pipeline(CombineAudio, audios=["voice.wav", "music.wav"], format="flac", filename_prefix="sequence")
    workflow.add_pipeline(
        CombineAudio, audios=["voice.wav", "music.wav"], mode="mix", format="flac", filename_prefix="mix"
    )

    files = {"voice.wav": tone(1.0, rate=8000), "music.wav": tone(2.0, rate=44100, channels=2)}
    result = await run(workflow, folder, files)

    sequence = probe(one(result, "sequence"), ".flac")
    assert seconds(sequence) == pytest.approx(3.0, abs=0.05)
    assert stream(sequence, "audio")["channels"] == 2
    assert seconds(probe(one(result, "mix"), ".flac")) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------- more generation cases: sizes, rates, slow paths


async def test_text_to_image_at_the_size_of_its_aspect_ratio(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToImage,
        prompt="a mountain range at sunrise, panorama",
        model=family(models, "sdxl"),
        aspect_ratio="21:9",
        steps=4,
        seed=7,
        filename_prefix="panorama",
    )

    result = await run(workflow, folder)

    assert picture(one(result, "panorama")).size == (1536, 640)


async def test_inpaint_repaints_the_transparent_pixels(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        Inpaint,
        image="hole.png",
        prompt="a shiny red apple on a gray table, studio photo",
        model=family(models, "sdxl"),
        steps=10,
        seed=7,
        filename_prefix="filled",
    )

    result = await run(workflow, folder, {"hole.png": with_hole()})

    image = picture(one(result, "filled")).convert("RGB")
    assert all(abs(a - b) <= 2 for a, b in zip(image.getpixel((10, 10)), GRAY))
    # Under its transparency the hole is black; the repaint fills it.
    inside = image.crop((200, 200, 312, 312)).get_flattened_data()
    assert sum(sum(pixel) for pixel in inside) / len(inside) > 60


async def test_text_to_video_at_another_frame_rate(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="a paper boat drifting on a calm lake",
        model=family(models, "wan22_5b"),
        width=320,
        height=192,
        duration_seconds=1.0,
        fps=12,
        steps=6,
        seed=1,
        filename_prefix="boat12",
    )

    result = await run(workflow, folder)

    video = stream(probe(one(result, "boat12")), "video")
    assert int(video["nb_read_frames"]) == 13
    assert video["r_frame_rate"] == "12/1"


async def test_minimax_h3_without_the_turbo_lora(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToVideo,
        prompt="waves crash on a rocky shore; the roar of the surf",
        model=family(models, "minimax_h3"),
        width=384,
        height=224,
        duration_seconds=1.0,
        fast=False,
        steps=4,
        seed=1,
        filename_prefix="surf",
    )

    result = await run(workflow, folder)

    info = probe(one(result, "surf"))
    assert int(stream(info, "video")["nb_read_frames"]) == 22
    assert stream(info, "audio") is not None


async def test_sound_to_video_without_the_fast_lora(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        SoundToVideo,
        image="face.png",
        audio="talk.wav",
        prompt="a person talks to the camera",
        model=family(models, "wan22_s2v"),
        width=320,
        height=320,
        duration_seconds=1.0,
        fast=False,
        steps=4,
        seed=1,
        filename_prefix="slow",
    )

    result = await run(workflow, folder, {"face.png": png(320, 320), "talk.wav": tone(2.0, rate=16000)})

    info = probe(one(result, "slow"))
    assert int(stream(info, "video")["nb_read_frames"]) == 20
    assert float(stream(info, "audio")["duration"]) == pytest.approx(20 / 16, abs=0.1)


async def test_a_waltz_saved_as_flac(models, folder):
    workflow = Workflow()
    workflow.add_pipeline(
        TextToAudio,
        style="a gentle orchestral waltz, strings and piano",
        model=family(models, "ace_step_15"),
        duration_seconds=5.0,
        bpm=90,
        time_signature="3",
        format="flac",
        seed=1,
        filename_prefix="waltz",
    )

    result = await run(workflow, folder)

    data = one(result, "waltz")
    info = probe(data, ".flac")
    assert stream(info, "audio")["codec_name"] == "flac"
    assert seconds(info) == pytest.approx(5.0, abs=0.3)
    assert loudness(data, ".flac") > -40
