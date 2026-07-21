"""OpenAI-compatible HTTP API Adapter."""

import os
import time
from typing import Any
import httpx

from fingerprint.types import AdapterResult


class OpenAICompatAdapter:
    """Adapter interacting with OpenAI-compatible HTTP chat completions endpoints."""

    def __init__(
        self,
        name: str = "openai",
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        environment: str = "openai-api",
        system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ):
        self.name = name
        self.model = model
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.environment = environment
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 32,
        }

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 200:
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
                raw_text = content.strip()
                return AdapterResult(
                    raw_text=raw_text,
                    extracted_token=None,
                    valid=len(raw_text) > 0,
                    invalid_reason=None if len(raw_text) > 0 else "Empty response",
                    latency_ms=round(elapsed_ms, 2),
                    exit_code=0,
                    stdout=response.text,
                    stderr="",
                    meta={
                        "adapter": self.name,
                        "environment": self.environment,
                        "model": self.model,
                        "base_url": self.base_url,
                        "usage": data.get("usage", {}),
                    },
                )
            else:
                return AdapterResult(
                    raw_text="",
                    extracted_token=None,
                    valid=False,
                    invalid_reason=f"HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=round(elapsed_ms, 2),
                    exit_code=response.status_code,
                    stdout="",
                    stderr=response.text,
                    meta={
                        "adapter": self.name,
                        "environment": self.environment,
                        "model": self.model,
                    },
                )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AdapterResult(
                raw_text="",
                extracted_token=None,
                valid=False,
                invalid_reason=str(e),
                latency_ms=round(elapsed_ms, 2),
                exit_code=-1,
                stdout="",
                stderr="",
                meta={
                    "adapter": self.name,
                    "environment": self.environment,
                    "model": self.model,
                },
            )
