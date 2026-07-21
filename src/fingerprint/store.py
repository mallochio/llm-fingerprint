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
    path_obj = Path(lib_path)

    if not path_obj.exists():
        # Fallback to bundled package data directory
        data_dir = Path(__file__).parent / "data" / "refs"
        candidate = data_dir / str(lib_path)
        if candidate.exists():
            path_obj = candidate
        elif (data_dir / "api-v1").exists():
            path_obj = data_dir / "api-v1"
        elif data_dir.exists():
            path_obj = data_dir
        else:
            raise FileNotFoundError(f"Reference library path not found: {lib_path}")

    fps: list[Fingerprint] = []

    if path_obj.is_file():
        if path_obj.suffix.lower() == ".json":
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    fps.append(Fingerprint.model_validate(item))
            elif isinstance(data, dict):
                fps.append(Fingerprint.model_validate(data))
    elif path_obj.is_dir():
        for path in sorted(path_obj.rglob("*.json")):
            try:
                fps.append(load_fingerprint(path))
            except Exception:
                pass

    return fps
