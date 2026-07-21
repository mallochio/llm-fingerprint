"""Storage utilities for Fingerprint files and reference libraries."""

import json
import os
from pathlib import Path
from fingerprint.types import Fingerprint


def save_fingerprint(fp: Fingerprint, file_path: str | Path) -> None:
    """Saves a Fingerprint instance to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(fp.model_dump(), f, indent=2, ensure_ascii=False)


def load_fingerprint(file_path: str | Path) -> Fingerprint:
    """Loads a Fingerprint instance from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Fingerprint.model_validate(data)


def load_library(lib_path: str | Path) -> list[Fingerprint]:
    """Loads a list of reference Fingerprints from a directory or JSON file."""
    lib_path = Path(lib_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"Reference library path not found: {lib_path}")

    fps: list[Fingerprint] = []

    if lib_path.is_file():
        if lib_path.suffix.lower() == ".json":
            with open(lib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    fps.append(Fingerprint.model_validate(item))
            elif isinstance(data, dict):
                fps.append(Fingerprint.model_validate(data))
    elif lib_path.is_dir():
        for path in sorted(lib_path.rglob("*.json")):
            try:
                fps.append(load_fingerprint(path))
            except Exception:
                pass

    return fps
