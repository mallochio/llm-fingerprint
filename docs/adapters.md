# Adapters Guide & Specifications

`llm-fingerprint` uses modular adapters to query diverse LLM targets—including OpenAI-compatible HTTP APIs, generic CLI subprocesses, and custom test mocks.

---

## 1. Adapter Protocol Specification

All adapters implement the `ModelAdapter` protocol:

```python
class AdapterResult(TypedDict):
    raw_text: str
    extracted_token: str | None
    valid: bool
    invalid_reason: str | None
    latency_ms: float
    exit_code: int | None
    stdout: str
    stderr: str
    meta: dict

class ModelAdapter(Protocol):
    name: str
    environment: str
    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult: ...
```

---

## 2. OpenAI-Compatible Adapter (`openai`)

Queries HTTP REST endpoints conforming to OpenAI's `/v1/chat/completions` API specification.

### Parameters
- `model`: Model identifier string (e.g. `gpt-4o`, `anthropic/claude-3.5-sonnet`).
- `base_url`: Target endpoint URL (e.g. `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`).
- `api_key`: Secret API token.
- `temperature`: Sampling temperature (default 1.0 per paper recommendations).

---

## 3. Generic Subprocess CLI Adapter (`cli`)

Spawns a local executable or CLI command for each probe.

### Prompt Passing Modes
- `argv`: Appends the wrapped prompt as the final positional command-line argument.
- `stdin`: Pipes the wrapped prompt to standard input (`stdin`).
- `file`: Writes the prompt to a temporary file and passes the file path as the final argument.

### Environment Variable Expansion
Environment values in configuration support standard bash-style expansion (`${ENV_VAR}`).
