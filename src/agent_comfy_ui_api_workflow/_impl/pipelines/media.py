"""Pipelines that edit images, videos and audio without a model."""

from typing import Literal

from comfy_ui_api_workflow.nodes.audio import (
    AudioAdjustVolume,
    AudioConcat,
    AudioMerge,
    TrimAudioDuration,
)
from comfy_ui_api_workflow.nodes.image import GetImageSize, SaveAnimatedPNG, SaveAnimatedWEBP
from comfy_ui_api_workflow.nodes.image.upscaling import ImageScaleBy
from comfy_ui_api_workflow.nodes.utilities import ComfyMathExpression
from comfy_ui_api_workflow.nodes.video import (
    ConcatenateVideo,
    CreateVideo,
    GetVideoComponents,
    VideoSlice,
)

from .base import AgentPipeline, AudioFormat, Link, VideoFormat

# TrimAudioDuration has no "until the end"; a day is longer than any input.
_UNTIL_THE_END = 86400.0
_SAVE = "Add the save node. Set false when another step uses the output."
_VIDEO_FORMAT = "Container of the saved video: mp4 (H.264) or webm (AV1)."
_AUDIO_FORMAT = "Format of the saved audio."


class LoadVideoParts(AgentPipeline):
    NAME = "load_video"
    DESCRIPTION = (
        "Opens a video (a file the user attached, or a VIDEO output) and splits it into "
        "its frames, its soundtrack and its frame rate, so image nodes can edit the "
        "frames. Rebuild the video with frames_to_video, passing the fps and audio "
        "outputs to keep the timing and the sound. Outputs: frames (IMAGE), audio "
        "(AUDIO), fps (FLOAT), video (VIDEO)."
    )
    PARAMETERS = {
        "video": 'Name of a video file the user attached, or a VIDEO output like "LoadVideo_1.VIDEO".',
    }
    FILES = {"video": "video"}
    COVERS = ("LoadVideo", "GetVideoComponents")

    def build(self, video: str | Link):
        source = self.video(video)
        parts = self.add_node(GetVideoComponents, video=source)
        return {
            "frames": parts.output["images"],
            "audio": parts.output["audio"],
            "fps": parts.output["fps"],
            "video": source,
        }


class FramesToVideo(AgentPipeline):
    NAME = "frames_to_video"
    DESCRIPTION = (
        "Turns a batch of frames (IMAGE) into a video, with an optional soundtrack, and "
        "saves it. Use it after editing the frames of a video or to make a video from "
        "images. Output: video (VIDEO)."
    )
    PARAMETERS = {
        "frames": 'The frames, an IMAGE output like "load_video frames" or "VAEDecode_5.IMAGE".',
        "fps": "Frames per second: a number, or the fps output of load_video to keep the original timing.",
        "audio": "Optional soundtrack: an audio file the user attached or an AUDIO output.",
        "format": _VIDEO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"audio": "audio"}
    COVERS = ("CreateVideo",)

    def build(
        self,
        frames: Link,
        fps: float | Link = 24.0,
        audio: str | Link | None = None,
        format: VideoFormat = "mp4",
        save: bool = True,
        filename_prefix: str = "video",
    ):
        sound = {} if audio is None else {"audio": self.audio(audio)}
        video = self.add_node(CreateVideo, images=frames, fps=fps, **sound)
        if save:
            self.save_video(video.output["VIDEO"], filename_prefix, format)
        return {"video": video.output["VIDEO"]}


class TrimVideo(AgentPipeline):
    NAME = "trim_video"
    DESCRIPTION = (
        "Cuts a video to a time range, keeping its sound, and saves it. Output: video (VIDEO)."
    )
    PARAMETERS = {
        "video": "A video file the user attached, or a VIDEO output.",
        "start_seconds": "Where the cut starts, in seconds from the beginning.",
        "duration_seconds": "How long the cut lasts, in seconds; 0 keeps everything after the start.",
        "format": _VIDEO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"video": "video"}
    COVERS = ("VideoSlice",)

    def build(
        self,
        video: str | Link,
        start_seconds: float = 0.0,
        duration_seconds: float = 0.0,
        format: VideoFormat = "mp4",
        save: bool = True,
        filename_prefix: str = "video",
    ):
        sliced = self.add_node(
            VideoSlice,
            video=self.video(video),
            start_time=start_seconds,
            duration=duration_seconds,
        )
        if save:
            self.save_video(sliced.output["VIDEO"], filename_prefix, format)
        return {"video": sliced.output["VIDEO"]}


class AddAudioToVideo(AgentPipeline):
    NAME = "add_audio_to_video"
    DESCRIPTION = (
        "Puts a soundtrack on a video, replacing the sound it had, cut to the video's "
        "length, and saves it. Output: video (VIDEO)."
    )
    PARAMETERS = {
        "video": "A video file the user attached, or a VIDEO output.",
        "audio": "An audio file the user attached, or an AUDIO output.",
        "format": _VIDEO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"video": "video", "audio": "audio"}

    def build(
        self,
        video: str | Link,
        audio: str | Link,
        format: VideoFormat = "mp4",
        save: bool = True,
        filename_prefix: str = "video",
    ):
        parts = self.add_node(GetVideoComponents, video=self.video(video))
        size = self.add_node(GetImageSize, image=parts.output["images"])
        seconds = self.add_node(
            ComfyMathExpression,
            values_a=size.output["batch_size"],
            values_b=parts.output["fps"],
            expression="a / b",
        )
        sound = self.add_node(
            TrimAudioDuration,
            audio=self.audio(audio),
            start_index=0.0,
            duration=seconds.output["FLOAT"],
        )
        video_out = self.add_node(
            CreateVideo,
            images=parts.output["images"],
            fps=parts.output["fps"],
            audio=sound.output["AUDIO"],
        )
        if save:
            self.save_video(video_out.output["VIDEO"], filename_prefix, format)
        return {"video": video_out.output["VIDEO"]}


class ConcatenateVideos(AgentPipeline):
    NAME = "concatenate_videos"
    DESCRIPTION = (
        "Joins videos one after the other, in the given order, and saves the result. "
        "The videos need the same size and frame rate, like clips made with the same "
        "model and settings: the server refuses others. Output: video (VIDEO)."
    )
    PARAMETERS = {
        "videos": "From 2 to 100 videos, in order: attached video files or VIDEO outputs.",
        "format": _VIDEO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"videos": "video"}
    COVERS = ("ConcatenateVideo",)

    def build(
        self,
        videos: list[str | Link],
        format: VideoFormat = "mp4",
        save: bool = True,
        filename_prefix: str = "video",
    ):
        if not 2 <= len(videos) <= 100:
            raise ValueError(f"it takes at least 2 videos and at most 100, got {len(videos)}")
        segments = {f"videos_video{index}": self.video(video) for index, video in enumerate(videos)}
        joined = self.add_node(ConcatenateVideo, **segments)
        if save:
            self.save_video(joined.output["VIDEO"], filename_prefix, format)
        return {"video": joined.output["VIDEO"]}


class VideoToAnimation(AgentPipeline):
    NAME = "video_to_animation"
    DESCRIPTION = (
        "Saves a video, or a batch of frames, as an animated WebP or PNG image (for a "
        "GIF-like result). Give either video or frames."
    )
    PARAMETERS = {
        "video": "A video file the user attached, or a VIDEO output. Leave empty when giving frames.",
        "frames": "An IMAGE output with the frames. Leave empty when giving a video.",
        "fps": "Frames per second; empty keeps the video's own rate (12 for frames).",
        "format": "webp (smaller files) or png.",
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"video": "video"}
    COVERS = ("SaveAnimatedWEBP", "SaveAnimatedPNG")

    def build(
        self,
        video: str | Link | None = None,
        frames: Link | None = None,
        fps: float | Link | None = None,
        format: Literal["webp", "png"] = "webp",
        filename_prefix: str = "animation",
    ):
        if (video is None) == (frames is None):
            raise ValueError("give either video or frames")
        if video is not None:
            parts = self.add_node(GetVideoComponents, video=self.video(video))
            images, rate = parts.output["images"], parts.output["fps"]
        else:
            images, rate = frames, 12.0
        if fps is not None:
            rate = fps
        if format == "webp":
            self.add_node(
                SaveAnimatedWEBP,
                images=images,
                fps=rate,
                lossless=False,
                quality=90,
                method="default",
                filename_prefix=filename_prefix,
            )
        else:
            self.add_node(SaveAnimatedPNG, images=images, fps=rate, filename_prefix=filename_prefix)


class EditAudio(AgentPipeline):
    NAME = "edit_audio"
    DESCRIPTION = (
        "Cuts an audio to a time range and/or changes its volume, then saves it. With no "
        "change it just converts the format. Output: audio (AUDIO)."
    )
    PARAMETERS = {
        "audio": "An audio file the user attached, or an AUDIO output.",
        "start_seconds": "Where the cut starts, in seconds; negative counts from the end.",
        "duration_seconds": "How long the cut lasts, in seconds; 0 keeps everything after the start.",
        "volume_db": "Volume change in decibels: 0 keeps it, +6 doubles, -6 halves.",
        "format": _AUDIO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"audio": "audio"}
    COVERS = ("TrimAudioDuration", "AudioAdjustVolume", "SaveAudioAdvanced")

    def build(
        self,
        audio: str | Link,
        start_seconds: float = 0.0,
        duration_seconds: float = 0.0,
        volume_db: int = 0,
        format: AudioFormat = "mp3",
        save: bool = True,
        filename_prefix: str = "audio",
    ):
        sound = self.audio(audio)
        if start_seconds or duration_seconds:
            sound = self.add_node(
                TrimAudioDuration,
                audio=sound,
                start_index=start_seconds,
                duration=duration_seconds or _UNTIL_THE_END,
            ).output["AUDIO"]
        if volume_db:
            sound = self.add_node(AudioAdjustVolume, audio=sound, volume=volume_db).output["AUDIO"]
        if save:
            self.save_audio(sound, filename_prefix, format)
        return {"audio": sound}


class CombineAudio(AgentPipeline):
    NAME = "combine_audio"
    DESCRIPTION = (
        "Combines audios: in sequence, one after the other, or mixed on top of each "
        "other (for music under a voice). A mix lasts as long as the first audio: the "
        "others are cut to it or padded with silence. Saves the result. Output: audio (AUDIO)."
    )
    PARAMETERS = {
        "audios": (
            "Two or more audio files the user attached or AUDIO outputs, in order. To mix, put "
            "first the one whose length the result keeps: the voice to cut the music to it, "
            "the music to keep all of it."
        ),
        "mode": "sequence plays them one after the other; mix plays them at the same time.",
        "format": _AUDIO_FORMAT,
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"audios": "audio"}
    COVERS = ("AudioConcat", "AudioMerge")

    def build(
        self,
        audios: list[str | Link],
        mode: Literal["sequence", "mix"] = "sequence",
        format: AudioFormat = "mp3",
        save: bool = True,
        filename_prefix: str = "audio",
    ):
        if len(audios) < 2:
            raise ValueError(f"it takes at least 2 audios, got {len(audios)}")
        sounds = [self.audio(audio) for audio in audios]
        combined = sounds[0]
        for sound in sounds[1:]:
            if mode == "sequence":
                node = self.add_node(AudioConcat, audio1=combined, audio2=sound, direction="after")
            else:
                node = self.add_node(AudioMerge, audio1=combined, audio2=sound, merge_method="add")
            combined = node.output["AUDIO"]
        if save:
            self.save_audio(combined, filename_prefix, format)
        return {"audio": combined}


class UpscaleImage(AgentPipeline):
    NAME = "upscale_image"
    DESCRIPTION = (
        "Enlarges an image by a factor with a resampling filter (no model), and saves "
        "it. When upscale_image_with_model is offered, prefer it: it is much sharper. "
        "Output: image (IMAGE)."
    )
    PARAMETERS = {
        "image": "An image file the user attached, or an IMAGE output.",
        "scale": "How many times larger, from 0.01 to 8.",
        "method": "Resampling filter; lanczos is the sharpest.",
        "save": _SAVE,
        "filename_prefix": "Prefix of the saved file name.",
    }
    FILES = {"image": "image"}
    COVERS = ("ImageScaleBy",)

    def build(
        self,
        image: str | Link,
        scale: float = 2.0,
        method: Literal["lanczos", "bicubic", "bilinear", "area", "nearest-exact"] = "lanczos",
        save: bool = True,
        filename_prefix: str = "image",
    ):
        scaled = self.add_node(
            ImageScaleBy, image=self.image(image), upscale_method=method, scale_by=scale
        )
        if save:
            self.save_image(scaled.output["IMAGE"], filename_prefix)
        return {"image": scaled.output["IMAGE"]}
