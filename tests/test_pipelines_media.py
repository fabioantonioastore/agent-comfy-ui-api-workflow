import pytest
from comfy_ui_api_workflow import Workflow, errors
from comfy_ui_api_workflow.nodes.image import EmptyImage

from agent_comfy_ui_api_workflow._impl.pipelines import MEDIA_PIPELINES
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


def classes(workflow: Workflow) -> list[str]:
    return [node["class_type"] for node in workflow.get_comfy_ui_workflow_format().values()]


def node_of(workflow: Workflow, class_type: str) -> dict:
    (node,) = [
        node
        for node in workflow.get_comfy_ui_workflow_format().values()
        if node["class_type"] == class_type
    ]
    return node["inputs"]


def reloads(workflow: Workflow) -> None:
    Workflow().load_workflow(workflow.get_comfy_ui_workflow_format())


@pytest.mark.parametrize("pipeline", MEDIA_PIPELINES, ids=lambda p: p.NAME)
def test_every_parameter_is_documented(pipeline):
    names = [parameter.name for parameter in pipeline.parameters()]

    assert pipeline.NAME and pipeline.DESCRIPTION
    assert sorted(pipeline.PARAMETERS) == sorted(names)
    assert set(pipeline.FILES) <= set(names)


def test_pipeline_names_are_unique():
    names = [pipeline.NAME for pipeline in MEDIA_PIPELINES]

    assert len(names) == len(set(names))


def test_load_video_splits_a_user_video():
    workflow = Workflow()

    parts = workflow.add_pipeline(LoadVideoParts, video="clip.mp4")

    assert classes(workflow) == ["LoadVideo", "GetVideoComponents"]
    assert node_of(workflow, "LoadVideo") == {"file": "clip.mp4"}
    assert set(parts.output.outputs) == {"frames", "audio", "fps", "video"}


def test_frames_to_video_keeps_fps_and_sound_and_saves_mp4():
    workflow = Workflow()
    parts = workflow.add_pipeline(LoadVideoParts, video="clip.mp4")

    workflow.add_pipeline(
        FramesToVideo,
        frames=parts.output["frames"],
        fps=parts.output["fps"],
        audio=parts.output["audio"],
        filename_prefix="edited",
    )

    created = node_of(workflow, "CreateVideo")
    assert tuple(created["fps"]) == parts.output["fps"]
    assert tuple(created["audio"]) == parts.output["audio"]
    saved = node_of(workflow, "SaveVideo")
    assert (saved["format"], saved["format.codec"], saved["filename_prefix"]) == (
        "mp4",
        "h264",
        "edited",
    )
    reloads(workflow)


def test_frames_to_video_can_skip_saving():
    workflow = Workflow()
    frames = workflow.add_node(EmptyImage, width=64, height=64, batch_size=8).output["IMAGE"]

    workflow.add_pipeline(FramesToVideo, frames=frames, fps=8.0, save=False)

    assert "SaveVideo" not in classes(workflow)


def test_trim_video_slices_by_time():
    workflow = Workflow()

    workflow.add_pipeline(TrimVideo, video="clip.mp4", start_seconds=2.0, duration_seconds=3.5)

    sliced = node_of(workflow, "Video Slice")
    assert (sliced["start_time"], sliced["duration"]) == (2.0, 3.5)
    assert "SaveVideo" in classes(workflow)


def test_add_audio_to_video_cuts_the_sound_to_the_video_length():
    workflow = Workflow()

    workflow.add_pipeline(AddAudioToVideo, video="clip.mp4", audio="song.mp3")

    assert node_of(workflow, "LoadAudio") == {"audio": "song.mp3"}
    assert node_of(workflow, "ComfyMathExpression")["expression"] == "a / b"
    trimmed = node_of(workflow, "TrimAudioDuration")
    math_id = next(
        node_id
        for node_id, node in workflow.get_comfy_ui_workflow_format().items()
        if node["class_type"] == "ComfyMathExpression"
    )
    assert trimmed["duration"][0] == math_id
    assert node_of(workflow, "CreateVideo")["audio"][0] in workflow.nodes
    reloads(workflow)


def test_concatenate_videos_joins_files_and_links_in_order():
    workflow = Workflow()
    first = workflow.add_pipeline(LoadVideoParts, video="a.mp4").output["video"]

    workflow.add_pipeline(ConcatenateVideos, videos=[first, "b.mp4"])

    joined = node_of(workflow, "ConcatenateVideo")
    assert tuple(joined["videos.video0"]) == first
    assert joined["videos.video1"][0] != first[0]
    reloads(workflow)


def test_concatenate_videos_needs_two_videos():
    with pytest.raises(errors.pipeline.PipelineError) as error:
        Workflow().add_pipeline(ConcatenateVideos, videos=["a.mp4"])

    assert "at least 2 videos" in str(error.value)


def test_video_to_animation_saves_webp_at_the_video_fps():
    workflow = Workflow()

    workflow.add_pipeline(VideoToAnimation, video="clip.mp4")

    saved = node_of(workflow, "SaveAnimatedWEBP")
    assert saved["fps"][0] in workflow.nodes
    assert saved["lossless"] is False


def test_video_to_animation_takes_frames_or_a_video_but_not_both():
    workflow = Workflow()
    frames = workflow.add_node(EmptyImage, width=64, height=64, batch_size=4).output["IMAGE"]

    workflow.add_pipeline(VideoToAnimation, frames=frames, fps=4.0, format="png")
    assert node_of(workflow, "SaveAnimatedPNG")["fps"] == 4.0

    with pytest.raises(errors.pipeline.PipelineError):
        workflow.add_pipeline(VideoToAnimation, frames=frames, video="clip.mp4")
    with pytest.raises(errors.pipeline.PipelineError):
        workflow.add_pipeline(VideoToAnimation)


def test_edit_audio_trims_and_changes_the_volume():
    workflow = Workflow()

    workflow.add_pipeline(
        EditAudio, audio="talk.wav", start_seconds=1.0, duration_seconds=4.0, volume_db=-6
    )

    assert classes(workflow) == [
        "LoadAudio",
        "TrimAudioDuration",
        "AudioAdjustVolume",
        "SaveAudioAdvanced",
    ]
    assert node_of(workflow, "TrimAudioDuration")["duration"] == 4.0
    assert node_of(workflow, "SaveAudioAdvanced")["format"] == "mp3"


def test_edit_audio_without_changes_only_converts():
    workflow = Workflow()

    workflow.add_pipeline(EditAudio, audio="talk.wav", format="flac")

    assert classes(workflow) == ["LoadAudio", "SaveAudioAdvanced"]


def test_edit_audio_from_a_start_keeps_the_rest():
    workflow = Workflow()

    workflow.add_pipeline(EditAudio, audio="talk.wav", start_seconds=10.0)

    assert node_of(workflow, "TrimAudioDuration")["duration"] >= 3600


def test_combine_audio_chains_the_tracks():
    workflow = Workflow()

    workflow.add_pipeline(CombineAudio, audios=["a.wav", "b.wav", "c.wav"], mode="mix")

    assert classes(workflow).count("AudioMerge") == 2
    assert node_of(workflow, "SaveAudioAdvanced")
    with pytest.raises(errors.pipeline.PipelineError):
        workflow.add_pipeline(CombineAudio, audios=["a.wav"])


def test_upscale_image_scales_a_user_image():
    workflow = Workflow()

    workflow.add_pipeline(UpscaleImage, image="me.png", scale=2.5)

    scaled = node_of(workflow, "ImageScaleBy")
    assert (scaled["scale_by"], scaled["upscale_method"]) == (2.5, "lanczos")
    assert "SaveImage" in classes(workflow)
    reloads(workflow)
