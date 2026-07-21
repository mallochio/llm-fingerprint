"""Generic CLI Subprocess Adapter for headless LLM agents and CLI tools."""

import os
import time
import subprocess
import tempfile
from typing import Any
from fingerprint.types import AdapterResult


class CLIAdapter:
    """Adapter executing a headless CLI command via subprocess."""

    def __init__(
        self,
        name: str,
        command: list[str],
        environment: str = "cli",
        prompt_mode: str = "argv",  # argv | stdin | file
        prompt_template: str = "{prompt}",
        cwd: str | None = None,
        timeout_s: float = 120.0,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ):
        self.name = name
        self.environment = environment
        self.command = list(command)
        self.prompt_mode = prompt_mode
        self.prompt_template = prompt_template
        self.cwd = cwd or tempfile.gettempdir()
        self.timeout_s = timeout_s
        self.env_vars = env or {}
        self.extra_args = extra_args or []

    def _prepare_env(self) -> dict[str, str]:
        env_copy = os.environ.copy()
        for k, v in self.env_vars.items():
            # Expand environment variable references like ${VAR}
            expanded = os.path.expandvars(v)
            env_copy[k] = expanded
        return env_copy

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        wrapped_prompt = self.prompt_template.format(prompt=prompt)
        full_command = list(self.command) + list(self.extra_args)

        stdin_input: str | None = None
        temp_file_path: str | None = None

        if self.prompt_mode == "argv":
            full_command.append(wrapped_prompt)
        elif self.prompt_mode == "stdin":
            stdin_input = wrapped_prompt
        elif self.prompt_mode == "file":
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
                f.write(wrapped_prompt)
                temp_file_path = f.name
            full_command.append(temp_file_path)
        else:
            raise ValueError(f"Unsupported prompt_mode: {self.prompt_mode}")

        start_time = time.perf_counter()
        try:
            if not os.path.exists(self.cwd):
                os.makedirs(self.cwd, exist_ok=True)

            res = subprocess.run(
                full_command,
                input=stdin_input,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=self.timeout_s,
                env=self._prepare_env(),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            raw_text = res.stdout.strip() if res.returncode == 0 else ""
            valid = res.returncode == 0 and len(raw_text) > 0

            return AdapterResult(
                raw_text=raw_text,
                extracted_token=None,  # Handled downstream by normalizer
                valid=valid,
                invalid_reason=None if valid else f"Exit code {res.returncode}",
                latency_ms=round(elapsed_ms, 2),
                exit_code=res.returncode,
                stdout=res.stdout,
                stderr=res.stderr,
                meta={
                    "adapter": self.name,
                    "environment": self.environment,
                    "command": full_command,
                    "cwd": self.cwd,
                },
            )
        except subprocess.TimeoutExpired as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AdapterResult(
                raw_text="",
                extracted_token=None,
                valid=False,
                invalid_reason=f"Timeout after {self.timeout_s}s",
                latency_ms=round(elapsed_ms, 2),
                exit_code=-1,
                stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
                stderr=e.stderr or "" if isinstance(e.stderr, str) else "",
                meta={"adapter": self.name, "environment": self.environment},
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
                meta={"adapter": self.name, "environment": self.environment},
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
