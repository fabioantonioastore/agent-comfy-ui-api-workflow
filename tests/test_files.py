import base64
import io

import pytest
from PIL import Image

from agent_comfy_ui_api_workflow._impl.errors import InputFileError
from agent_comfy_ui_api_workflow._impl.files import InputFiles, vision_block
from media import MP4, jpeg, png, wav


def test_an_image_keeps_its_name_kind_and_size():
    files = InputFiles()

    added = files.add("me.png", png(64, 48))

    assert (added.name, added.kind, added.width, added.height) == ("me.png", "image", 64, 48)
    assert files.get("me.png") is added
    assert "me.png" in files
    assert added.describe() == "me.png (image, 64x48)"


def test_the_kind_comes_from_the_extension():
    files = InputFiles()

    assert files.add("clip.MP4", MP4).kind == "video"
    assert files.add("song.wav", wav()).kind == "audio"
    assert files.add("photo.jpeg", jpeg()).kind == "image"
    assert files.names("audio") == ["song.wav"]


def test_names_lose_folders_and_unsafe_characters():
    files = InputFiles()

    added = files.add("../fotos/minha foto de São Paulo (1).png", png())

    assert added.name == "minha_foto_de_Sao_Paulo_1.png"


def test_the_same_name_with_other_data_gets_a_new_name():
    files = InputFiles()
    files.add("x.png", png(10, 10))

    second = files.add("x.png", png(20, 20))

    assert second.name == "x_2.png"
    assert len(files) == 2


def test_the_same_file_twice_is_kept_once():
    files = InputFiles()
    data = png()

    first = files.add("x.png", data)

    assert files.add("x.png", data) is first
    assert len(files) == 1


@pytest.mark.parametrize(
    ("name", "data", "reason"),
    [
        ("notes.txt", b"hello", "not an image, video or audio file"),
        ("empty.png", b"", "it is empty"),
        ("broken.png", b"not a png", "not a valid image"),
    ],
)
def test_unusable_files_are_rejected(name, data, reason):
    with pytest.raises(InputFileError) as error:
        InputFiles().add(name, data)

    assert reason in str(error.value)


def test_payload_has_the_requested_files():
    files = InputFiles()
    data = png()
    files.add("a.png", data)
    files.add("b.wav", wav())

    assert files.payload(["a.png"]) == {"a.png": data}
    with pytest.raises(InputFileError):
        files.payload(["missing.png"])


def test_vision_block_downscales_large_images():
    files = InputFiles()
    added = files.add("wide.png", png(3000, 1000))

    block = vision_block(added)

    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"
    image = Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))
    assert max(image.size) == 1568


def test_vision_block_keeps_transparency_as_png():
    added = InputFiles().add("mask.png", png(32, 32, alpha=True))

    assert vision_block(added)["source"]["media_type"] == "image/png"
