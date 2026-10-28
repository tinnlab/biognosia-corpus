"""Read the shell-style configuration files in config/ from Python."""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_BRACED = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")
_PLAIN = re.compile(r"\$(\w+)")


def _expand(value: str, variables: dict[str, str]) -> str:
    def lookup(name: str) -> str:
        return os.environ.get(name) or variables.get(name, "")

    def braced(match: re.Match) -> str:
        found = lookup(match.group(1))
        return found if found else (match.group(2) or "")

    value = _BRACED.sub(braced, value)
    return _PLAIN.sub(lambda m: lookup(m.group(1)), value)


def load_config(config_file: str) -> dict[str, str]:
    """Parse KEY="value" lines and expand $VAR, ${VAR} and ${VAR:-default}."""
    variables = {"PROJECT_ROOT": str(PROJECT_ROOT)}
    config: dict[str, str] = {}
    with open(config_file) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            value = _expand(value, variables)
            variables[key] = value
            config[key] = value
    return config
