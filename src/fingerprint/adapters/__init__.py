"""Adapter registry and factory functions."""

import os
from typing import Any
import yaml

from fingerprint.types import ModelAdapter
from fingerprint.adapters.cli_subprocess import CLIAdapter
from fingerprint.adapters.openai_compat import OpenAICompatAdapter
from fingerprint.adapters.mock import MockAdapter


BUILTIN_RECIPES = {
    "cursor_auto": {
        "type": "cli",
        "command": ["agent", "-p", "--output-format", "text", "--mode", "ask"],
        "prompt_mode": "argv",
        "prompt_template": (
            "Reply with exactly one word. No punctuation. No markdown. No preamble.\n"
            "Do not use tools, shell, or read files. Do not explain.\n"
            "Question: {prompt}"
        ),
        "cwd": "/tmp/fingerprint-empty",
        "timeout_s": 120.0,
        "environment": "cursor-cli",
    },
    "devin_auto": {
        "type": "cli",
        "command": ["devin", "-p", "--"],
        "prompt_mode": "argv",
        "prompt_template": (
            "/ask Reply with exactly one word. No tools. No code. No preamble.\n"
            "{prompt}"
        ),
        "cwd": "/tmp/fingerprint-empty",
        "timeout_s": 180.0,
        "environment": "devin-cli",
    },
}


def load_adapter(
    adapter_id: str,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ModelAdapter:
    """Loads a ModelAdapter instance by ID, optional YAML config file, or builtin recipes."""
    overrides = overrides or {}

    cfg: dict[str, Any] = {}

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f) or {}
            adapters_sec = raw_yaml.get("adapters", {})
            if adapter_id in adapters_sec:
                cfg = adapters_sec[adapter_id]

    if not cfg:
        if adapter_id in BUILTIN_RECIPES:
            cfg = dict(BUILTIN_RECIPES[adapter_id])
        elif adapter_id in ("openai", "openai-api"):
            cfg = {
                "type": "openai",
                "environment": "openai-api",
                "model": overrides.get("model", "gpt-4o"),
                "base_url": overrides.get("base_url"),
                "api_key": overrides.get("api_key"),
            }
        elif adapter_id.startswith("mock"):
            profile = "gpt-4o"
            if "claude" in adapter_id:
                profile = "claude-3-5-sonnet"
            elif "devin" in adapter_id or "router" in adapter_id:
                profile = "devin-auto-mock"
            cfg = {
                "type": "mock",
                "target_profile": profile,
                "environment": "mock-cli",
            }
        else:
            # Fallback to CLI adapter assuming command is adapter_id
            cfg = {
                "type": "cli",
                "command": [adapter_id],
                "environment": f"{adapter_id}-cli",
            }

    # Apply overrides
    cfg.update(overrides)
    adapter_type = cfg.get("type", "cli")

    if adapter_type == "cli":
        return CLIAdapter(
            name=adapter_id,
            command=cfg.get("command", [adapter_id]),
            environment=cfg.get("environment", "cli"),
            prompt_mode=cfg.get("prompt_mode", "argv"),
            prompt_template=cfg.get("prompt_template", "{prompt}"),
            cwd=cfg.get("cwd", "/tmp/fingerprint-empty"),
            timeout_s=float(cfg.get("timeout_s", 120.0)),
            env=cfg.get("env"),
            extra_args=cfg.get("extra_args"),
        )
    elif adapter_type in ("openai", "openai_compat"):
        return OpenAICompatAdapter(
            name=adapter_id,
            model=cfg.get("model", "gpt-4o"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            environment=cfg.get("environment", "openai-api"),
            system_prompt=cfg.get("system_prompt"),
            max_completion_tokens=cfg.get("max_completion_tokens"),
            reasoning_effort=cfg.get("reasoning_effort"),
            timeout_s=float(cfg.get("timeout_s", 30.0)),
        )
    elif adapter_type == "mock":
        return MockAdapter(
            name=adapter_id,
            target_profile=cfg.get("target_profile", "gpt-4o"),
            environment=cfg.get("environment", "mock-cli"),
            failure_rate=float(cfg.get("failure_rate", 0.0)),
        )
    else:
        raise ValueError(f"Unknown adapter type '{adapter_type}' for adapter '{adapter_id}'")
