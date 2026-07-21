# LLM Single-Token Fingerprinting Toolkit (`llm-fingerprint`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`llm-fingerprint` is an open-source toolkit that **fingerprints, verifies, and audits Large Language Models (LLMs)** from single-token output distributions across black-box APIs and headless agent CLIs (e.g. Cursor, Devin).

---

## 1. Mission & Scientific Basis

This toolkit implements the behavioral fingerprinting methodology proposed by **Tomáš Bruckner** in:
> **One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions** (arXiv:2607.10252, Jul 2026).
> - **Software Artifact**: [Zenodo DOI: 10.5281/zenodo.21278793](https://doi.org/10.5281/zenodo.21278793)
> - **Dataset Artifact**: [Zenodo DOI: 10.5281/zenodo.21278557](https://doi.org/10.5281/zenodo.21278557)

### Core Capabilities
1. **Verify**: Prove whether an API endpoint or provider is serving the exact model it claims (by checking if Jensen–Shannon divergence $D(E, X) < \tau$).
2. **Identify**: Find the nearest reference model family when querying opaque or unlabelled backends (1-NN / top-$k$).
3. **Audit Routers & Auto Modes**: Benchmark "Auto" routing modes (e.g. Cursor Auto, Devin Auto) as **mixtures over sessions**, measuring session-to-session distribution variance and model proportions.
4. **Universal CLI & API Adapters**: Query any headless CLI tool or OpenAI-compatible HTTP endpoint.

---

## 2. Installation

Install via `uv` or `pip`:

```bash
git clone https://github.com/your-org/llm-fingerprint.git
cd llm-fingerprint

# Create virtual environment and install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 3. Quickstart & Target UX

### Probe a Single Prompt Once
Inspect raw response and canonical normalized token from a mock or CLI adapter:

```bash
fingerprint probe once --adapter cursor_auto --prompt "Name a random color."
```

### Build Reference Fingerprint Library
Generate golden reference fingerprints for known API models (e.g. via OpenRouter / OpenAI):

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
export CURSOR_API_KEY=...
fingerprint audit --adapter cursor_auto --sessions 20 --n-per-cell 15 \
  --battery batteries/v1 --lib refs/api-v1 --report html --out runs/cursor-auto

# Audit claimed model endpoint
fingerprint audit --adapter openai --model vendor/gpt-4o --claim gpt-4o \
  --lib refs/api-v1 --tau 0.05 --out runs/gpt4o-audit
```

---

## 4. Architecture & Adapter Protocol

```
CLI / YAML Config
       │
       ▼
Orchestrator Collector (battery, N samples, resume cache)
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

### Adapter Interface
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
