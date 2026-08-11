"""Minimal indentation-based YAML-subset loader.

We depend only on numpy/pandas/requests/matplotlib/pytest (per project
constraints), so PyYAML is not available. ``config.yaml`` only ever needs a
small subset of YAML: two-space indentation, ``key: value`` pairs, nested
mappings, ``#`` comments, and scalar values (int/float/quoted-string/plain
string). This loader implements exactly that subset -- nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    """Parse a YAML scalar token into a Python int/float/str."""
    text = raw.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _strip_comment(line: str) -> str:
    """Remove a trailing ``# ...`` comment, respecting quoted strings."""
    in_quote: str | None = None
    for i, ch in enumerate(line):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in ("'", '"'):
            in_quote = ch
        elif ch == "#":
            return line[:i]
    return line


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a nested dict of scalars from a small YAML subset.

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        Nested dictionary mirroring the file's mapping structure, with
        scalar leaves parsed as int/float/str.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    root: dict[str, Any] = {}
    # Stack of (indent_level, dict_at_that_level)
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_val = line.strip()
        if ":" not in key_val:
            raise ValueError(f"Cannot parse config line: {raw_line!r}")
        key, _, value = key_val.partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            new_dict: dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _parse_scalar(value)

    return root


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def default_config() -> dict[str, Any]:
    """Load the project's default ``config.yaml`` from the repo root."""
    return load_config(_DEFAULT_CONFIG_PATH)
