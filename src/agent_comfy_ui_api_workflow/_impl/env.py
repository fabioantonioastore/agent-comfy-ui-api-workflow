import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .errors import SettingsError

_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*[:=]\s*(.*)")


def read_env_file(path: str | Path) -> dict[str, str]:
    """Reads NAME=value and NAME: value lines. A missing file reads as empty."""
    path = Path(path)
    if not path.exists():
        return {}
    settings: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match:
            settings[match.group(1)] = match.group(2).strip().strip("'\"")
    return settings


def load_settings(
    names: Iterable[str],
    env_file: str | Path = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The named settings, taken from the environment first and then from the .env file."""
    environ = os.environ if environ is None else environ
    from_file = read_env_file(env_file)
    settings: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = environ.get(name) or from_file.get(name)
        if value:
            settings[name] = value
        else:
            missing.append(name)
    if missing:
        raise SettingsError(missing, str(env_file))
    return settings
