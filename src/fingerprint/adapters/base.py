"""Base adapter definition and interface."""

from typing import Protocol, Any
from fingerprint.types import AdapterResult


class BaseAdapter(Protocol):
    name: str
    environment: str

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        ...
