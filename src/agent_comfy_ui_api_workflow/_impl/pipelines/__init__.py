from collections.abc import Iterable

from ..families import ModelChoice
from .base import AgentPipeline, Link, MediaKind, is_link
from .generation import (
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
from .media import (
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

__all__ = [
    "GENERATION_PIPELINES",
    "MEDIA_PIPELINES",
    "AgentPipeline",
    "Link",
    "MediaKind",
    "available_pipelines",
    "is_link",
]

# Pipelines that generate with a model; each is offered when a model can do its job.
GENERATION_PIPELINES: tuple[type[AgentPipeline], ...] = (
    TextToImage,
    ImageToImage,
    Inpaint,
    Outpaint,
    TextToVideo,
    ImageToVideo,
    SoundToVideo,
    TextToAudio,
    UpscaleImageWithModel,
)

# Pipelines that need no model, so they work on any server.
MEDIA_PIPELINES: tuple[type[AgentPipeline], ...] = (
    UpscaleImage,
    LoadVideoParts,
    FramesToVideo,
    TrimVideo,
    AddAudioToVideo,
    ConcatenateVideos,
    VideoToAnimation,
    EditAudio,
    CombineAudio,
)


def available_pipelines(models: Iterable[ModelChoice]) -> tuple[type[AgentPipeline], ...]:
    capabilities = {capability for model in models for capability in model.capabilities}
    generation = tuple(
        pipeline for pipeline in GENERATION_PIPELINES if pipeline.CAPABILITY in capabilities
    )
    return generation + MEDIA_PIPELINES
