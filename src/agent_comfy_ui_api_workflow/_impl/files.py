import base64
import io
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from .errors import InputFileError

FileKind = Literal["image", "video", "audio"]
_EXTENSIONS: dict[FileKind, tuple[str, ...]] = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"),
    "video": (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"),
    "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac"),
}
_KINDS: dict[str, FileKind] = {
    extension: kind for kind, extensions in _EXTENSIONS.items() for extension in extensions
}
# Larger images are scaled down by the API anyway; sending less keeps requests small.
MAX_VISION_SIDE = 1568
# The loader inputs that name a file the user attached, and the kind of that file.
FILE_INPUTS: dict[tuple[str, str], FileKind] = {
    ("LoadImage", "image"): "image",
    ("LoadImageMask", "image"): "image",
    ("LoadVideo", "file"): "video",
    ("LoadAudio", "audio"): "audio",
}


@dataclass(frozen=True)
class InputFile:
    name: str
    data: bytes = field(repr=False)
    kind: FileKind
    width: int | None = None
    height: int | None = None

    def describe(self) -> str:
        if self.width is None:
            return f"{self.name} ({self.kind})"
        return f"{self.name} ({self.kind}, {self.width}x{self.height})"


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _safe_name(name: str) -> str:
    # ComfyUI keeps the name in its input folder, so only the base name, in ASCII.
    base = PurePath(name.replace("\\", "/")).name
    stem, dot, extension = base.rpartition(".")
    if not dot:
        stem, extension = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", _ascii(stem)).strip("_") or "file"
    extension = re.sub(r"[^A-Za-z0-9]+", "", _ascii(extension))
    return f"{stem}.{extension}" if extension else stem


def _kind(name: str) -> FileKind:
    kind = _KINDS.get(PurePath(name).suffix.lower())
    if kind is None:
        raise InputFileError(name, "it is not an image, video or audio file (by its extension)")
    return kind


def _image_size(name: str, data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except (UnidentifiedImageError, OSError) as error:
        raise InputFileError(name, "it is not a valid image") from error


def _numbered(name: str, number: int) -> str:
    stem, dot, extension = name.rpartition(".")
    return f"{stem}_{number}.{extension}" if dot else f"{name}_{number}"


class InputFiles:
    """The files the user attached, by the name the workflow gives them."""

    def __init__(self) -> None:
        self.__files: dict[str, InputFile] = {}

    def add(self, name: str, data: bytes) -> InputFile:
        if not data:
            raise InputFileError(name, "it is empty")
        safe = _safe_name(name)
        kind = _kind(safe)
        width = height = None
        if kind == "image":
            width, height = _image_size(name, data)
        candidate, number = safe, 1
        while candidate in self.__files:
            if self.__files[candidate].data == data:
                return self.__files[candidate]
            number += 1
            candidate = _numbered(safe, number)
        added = InputFile(candidate, data, kind, width, height)
        self.__files[candidate] = added
        return added

    def get(self, name: str) -> InputFile | None:
        return self.__files.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.__files

    def __iter__(self) -> Iterator[InputFile]:
        return iter(self.__files.values())

    def __len__(self) -> int:
        return len(self.__files)

    def names(self, kind: FileKind | None = None) -> list[str]:
        return [file.name for file in self if kind is None or file.kind == kind]

    def payload(self, names: Iterable[str]) -> dict[str, bytes]:
        payload: dict[str, bytes] = {}
        for name in names:
            file = self.__files.get(name)
            if file is None:
                raise InputFileError(name, "it was not attached")
            payload[name] = file.data
        return payload


def vision_block(file: InputFile) -> dict[str, Any]:
    """The image as a content block Claude can see, no larger than MAX_VISION_SIDE."""
    with Image.open(io.BytesIO(file.data)) as image:
        image.load()
        picture = image.copy()
    picture.thumbnail((MAX_VISION_SIDE, MAX_VISION_SIDE))
    buffer = io.BytesIO()
    if picture.mode in ("RGBA", "LA", "PA") or "transparency" in picture.info:
        picture.save(buffer, "PNG")
        media_type = "image/png"
    else:
        picture.convert("RGB").save(buffer, "JPEG", quality=85)
        media_type = "image/jpeg"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(buffer.getvalue()).decode(),
        },
    }
