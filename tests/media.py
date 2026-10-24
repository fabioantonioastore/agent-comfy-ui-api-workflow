"""Small media files for the tests, made in memory."""

import io
import math
import struct
import wave

from PIL import Image


def png(width: int = 64, height: int = 48, color=(200, 30, 30), alpha: bool = False) -> bytes:
    mode = "RGBA" if alpha else "RGB"
    fill = (*color, 128) if alpha else color
    buffer = io.BytesIO()
    Image.new(mode, (width, height), fill).save(buffer, "PNG")
    return buffer.getvalue()


def jpeg(width: int = 64, height: int = 48, color=(30, 30, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "JPEG")
    return buffer.getvalue()


def wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


def tone(seconds: float = 1.0, rate: int = 44100, channels: int = 1, frequency: float = 440.0) -> bytes:
    """A sine wave at half the full scale: audio whose volume can be measured."""
    samples = (
        struct.pack("<h", round(16383 * math.sin(2 * math.pi * frequency * index / rate))) * channels
        for index in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(b"".join(samples))
    return buffer.getvalue()


# Only the extension matters to the registry, so the box header is enough.
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32
