# LLM Single-Token Fingerprinting Toolkit (`llm-fingerprint`)

[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`llm-fingerprint` is an open-source toolkit that **fingerprints, verifies, and audits Large Language Models (LLMs)** from single-token output distributions across black-box HTTP APIs and headless agent CLIs (e.g., Cursor, Devin).

---

## 1. Mission & Scientific Basis

This toolkit implements the behavioral fingerprinting methodology proposed by **Tomáš Bruckner** in:
> **One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions** (arXiv:2607.10252, Jul 2026).
> - **Software Artifact**: [Zenodo DOI: 10.5281/zenodo.21278793](https://doi.org/10.5281/zenodo.21278793)
> - **Dataset Artifact**: [Zenodo DOI: 10.5281/zenodo.21278557](https://doi.org/10.5281/zenodo.21278557)

### Core Capabilities
1. **Verify**: Prove whether an API endpoint or provider is serving the exact model it claims by computing base-2 Jensen–Shannon Divergence ($D_{JS}(E, X) < \tau$).
2. **Identify**: Find the top-$k$ nearest reference model families for unlabelled or opaque backends.
3. **Audit Routers & Auto Modes**: Benchmark "Auto" routing modes (e.g. Cursor Auto, Devin Auto) as **mixtures over sessions**, measuring session-to-session distribution variance and model proportions.
4. **Concurrent High-Throughput Probing**: Parallelized thread-pool collection with persistent HTTP client connection reuse for fast execution.
5. **Universal Adapters**: Out-of-the-box support for OpenAI-compatible HTTP endpoints, generic CLI subprocesses, and offline synthetic test mocks.

---

## 2. Installation

Install via `uv` or `pip`:

```bash
git clone https://github.com/mallochio/llm-fingerprint.git
cd llm-fingerprint

# Create virtual environment and install in editable mode
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 3. Quickstart & Usage

### Probe a Single Prompt
Inspect raw response and canonical normalized token from a mock or CLI adapter:

```bash
fingerprint probe once --adapter cursor_auto --prompt "Name a random color."
```

### Build Reference Fingerprint Library
Generate golden reference fingerprints for known API models:

```bash
export OPENAI_API_KEY=sk-...
fingerprint ref build --adapter openai --base-url https://openrouter.ai/api/v1 \
  --models-file models.txt --out refs/api-v1
```

### Fast Endpoint Quickcheck
Run a fast 5-session quickcheck against a reference library:

```bash
fingerprint quickcheck --adapter cursor_auto --sessions 5 --lib refs/api-v1
```

### Audit Router / Auto Mode Mixtures
Run a full multi-session audit on Cursor Auto or Devin Auto:

```bash
# Set up isolated empty workspace
mkdir -p /tmp/fingerprint-empty && git -C /tmp/fingerprint-empty init -q

# Audit Cursor Auto router mixture across 20 sessions
fingerprint audit --adapter cursor_auto --sessions 20 --n-per-cell 15 \
  --battery batteries/v1 --lib refs/api-v1 --report html --out runs/cursor-auto

# Audit claimed model endpoint
fingerprint audit --adapter openai --model gpt-4o --claim gpt-4o \
  --lib refs/api-v1 --tau 0.05 --out runs/gpt4o-audit
```

---

## 4. Architecture & Adapter Protocol

```text
CLI / YAML Config
       │
       ▼
Orchestrator Collector (concurrent thread pool, battery, cache)
       │
       ▼
Adapters (OpenAI-compatible HTTP, Generic Subprocess CLI, Mock)
       │
       ▼
Normalizer (NFC, casefold, punctuation strip, lexicons, validity)
       │
       ▼
Empirical Cell Distributions & Jensen-Shannon Divergence (JSD)
       │
       ▼
Verification / Identification / Router Mixture Reports
```

### Adapter Interface (`src/fingerprint/types.py`)

```python
class AdapterResult(BaseModel):
    raw_text: str
    extracted_token: str | None = None
    valid: bool
    invalid_reason: str | None = None
    latency_ms: float
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

class ModelAdapter(Protocol):
    name: str
    environment: str

    def complete(self, prompt: str, *, temperature: float = 1.0) -> AdapterResult:
        ...
```

---

## 5. Threat Model & Ethics

- **In Scope**: Detecting silent backend swaps, reseller model degradation, router Auto mode model mixtures, and honest misconfigurations.
- **Environment Rule**: **API fingerprints $\neq$ Agent CLI fingerprints.** Every run is tagged with its `environment` (`openai-api`, `cursor-cli`, `devin-cli`, etc.). Comparing across environments is flagged unless explicitly forced.
- **Safety & Isolation**: Run CLI fingerprinting inside an isolated, empty directory (e.g. `/tmp/fingerprint-empty`). Never run unattended agent CLIs in sensitive production codebases.

---

## 6. Citation

If you use this toolkit in your research or production auditing, please cite:

```bibtex
@article{bruckner2026onetoken,
  title={One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions},
  author={Bruckner, Tom{\'{a}}{\v{s}}},
  journal={arXiv preprint arXiv:2607.10252},
  year={2026}
}
```

---

## 7. License

Distributed under the [MIT License](LICENSE).
