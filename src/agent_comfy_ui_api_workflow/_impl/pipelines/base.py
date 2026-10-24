import inspect
from typing import Any, ClassVar, Literal

from comfy_ui_api_workflow import Pipeline
from comfy_ui_api_workflow.nodes.audio import LoadAudio, SaveAudioAdvanced
from comfy_ui_api_workflow.nodes.image import LoadImage, SaveImage
from comfy_ui_api_workflow.nodes.video import LoadVideo, SaveVideo

from ..families import ModelChoice

Link = tuple[str, int]
MediaKind = Literal["image", "video", "audio"]
VideoFormat = Literal["mp4", "webm"]
AudioFormat = Literal["mp3", "flac", "opus"]

_VIDEO_CODECS: dict[str, str] = {"mp4": "h264", "webm": "av1"}


def is_link(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


class AgentPipeline(Pipeline):
    """A pipeline the agent adds with one tool call, described for the model."""

    # The tool name, what the pipeline does and when to use it, and a line for each
    # parameter of build: they become the tool's description and schema.
    NAME: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    PARAMETERS: ClassVar[dict[str, str]] = {}
    # Parameters that take the name of a file the user attached, or a link.
    FILES: ClassVar[dict[str, MediaKind]] = {}
    # Node classes the pipeline adds, so a hint can point add_node at it.
    COVERS: ClassVar[tuple[str, ...]] = ()
    # What a generation pipeline does with its model parameter; it is offered only when
    # a model on the server can do it. None for pipelines that need no model.
    CAPABILITY: ClassVar[str | None] = None

    @classmethod
    def parameters(cls) -> list[inspect.Parameter]:
        return [
            parameter
            for parameter in inspect.signature(cls.build).parameters.values()
            if parameter.name != "self"
        ]

    @classmethod
    def model_parameter(cls) -> str | None:
        return next(
            (parameter.name for parameter in cls.parameters() if parameter.annotation is ModelChoice),
            None,
        )

    def image(self, value: str | Link) -> Link:
        if is_link(value):
            return (value[0], value[1])
        return self.add_node(LoadImage, image=value).output["IMAGE"]

    def video(self, value: str | Link) -> Link:
        if is_link(value):
            return (value[0], value[1])
        return self.add_node(LoadVideo, file=value).output["VIDEO"]

    def audio(self, value: str | Link) -> Link:
        if is_link(value):
            return (value[0], value[1])
        return self.add_node(LoadAudio, audio=value).output["AUDIO"]

    def save_image(self, images: Link, prefix: str) -> None:
        self.add_node(SaveImage, images=images, filename_prefix=prefix)

    def save_video(self, video: Link, prefix: str, format: VideoFormat = "mp4") -> None:
        self.add_node(
            SaveVideo,
            video=video,
            filename_prefix=prefix,
            format=format,
            format_codec=_VIDEO_CODECS[format],
        )

    def save_audio(self, audio: Link, prefix: str, format: AudioFormat = "mp3") -> None:
        self.add_node(SaveAudioAdvanced, audio=audio, filename_prefix=prefix, format=format)
