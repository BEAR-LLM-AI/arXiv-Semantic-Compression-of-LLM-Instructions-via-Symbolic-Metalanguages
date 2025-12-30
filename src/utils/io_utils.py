"""I/O utilities for the MetaGlyph pipeline."""

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, creating if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path | str) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Save data as JSON to file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_text(path: Path | str) -> str:
    """Load text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(text: str, path: Path | str) -> None:
    """Save text to file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def list_files(directory: Path | str, pattern: str = "*") -> list[Path]:
    """List files in directory matching pattern."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))
