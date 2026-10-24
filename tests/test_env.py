import pytest

from agent_comfy_ui_api_workflow._impl.env import load_settings, read_env_file
from agent_comfy_ui_api_workflow._impl.errors import SettingsError


def test_reads_equals_and_colon_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "A=1\nB: two\nexport C = 'three'\n# D=4\nE=\"x:y=z\"", encoding="utf-8"
    )

    assert read_env_file(env) == {"A": "1", "B": "two", "C": "three", "E": "x:y=z"}


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_env_file(tmp_path / ".env") == {}


def test_the_environment_wins_over_the_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=file\nB=file\n", encoding="utf-8")

    settings = load_settings(["A", "B"], env, environ={"A": "environment"})

    assert settings == {"A": "environment", "B": "file"}


def test_missing_settings_are_named(tmp_path):
    with pytest.raises(SettingsError) as error:
        load_settings(["X", "Y"], tmp_path / ".env", environ={"Y": "1"})

    assert error.value.missing == ["X"]
    assert "X" in str(error.value)
