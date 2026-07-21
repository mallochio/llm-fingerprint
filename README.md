# 🔬 LLM Single-Token Fingerprinting Toolkit (`llm-fingerprint`)

[![Unofficial Implementation](https://img.shields.io/badge/Status-Unofficial%20Implementation-orange.svg)](#disclaimer)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2607.10252-b31b1b.svg)](https://arxiv.org/abs/2607.10252)
[![Zenodo Software](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21278793-blue)](https://doi.org/10.5281/zenodo.21278793)
[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21278557-green)](https://doi.org/10.5281/zenodo.21278557)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Detect silent model downgrades, identify opaque API endpoints, and audit agent router mixtures using single-token output distributions.**

> [!NOTE]
> **Disclaimer & Attribution**: `llm-fingerprint` is an **independent, unofficial open-source toolkit and reproduction** based on the methodology proposed by **Tomáš Bruckner** in *"One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions"* (arXiv:2607.10252). This project is maintained by the open-source community and is not officially affiliated with or endorsed by the original paper authors.

`llm-fingerprint` is a high-performance Python toolkit and CLI designed to fingerprint, verify, and audit Large Language Models (LLMs) across black-box HTTP APIs and headless agent CLIs (such as Cursor, Devin, Aider, and custom subagents).

---

## 🎯 Why Single-Token Fingerprinting?

### The Problem
Commercial API providers, proxy resellers, and AI coding agents often dynamically route requests or silently downgrade underlying models to cut inference costs. Traditional LLM benchmarks (e.g. MMLU, HumanEval) are **slow, expensive** (costing thousands of output tokens), prone to prompt contamination, and easily spoofed by wrappers.

### The Scientific Solution
When LLMs are asked simple, unconstrained questions (e.g., *"Name a random number between 1 and 100"* or *"Flip a coin"*), they reveal **intrinsic, statistically immutable behavioral signatures**. 

For instance:
- **GPT-4o** exhibits a sharp preference for `7` and `42`.
- **Claude 3.5 Sonnet** exhibits distinct peaks at `42` and `17`.
- **Gemini Flash** displays a different characteristic empirical distribution altogether.

By sampling just **1 output token per prompt** across a structured prompt battery and measuring base-2 **Jensen–Shannon Divergence ($D_{JS}$)**, `llm-fingerprint` uniquely identifies and verifies LLMs at near-zero latency and cost.

---

## 🚀 Key Features

- 🛡️ **Model Swapping Verification**: Prove mathematically whether an API endpoint or reseller is serving the exact model claimed ($D_{JS} < \tau$).
- 🔀 **Router & Auto Mode Auditing**: Estimate model mixture proportions in dynamic "Auto" router modes (e.g. Cursor Auto, Devin Auto) over multi-session samples.
- 🕵️ **Opaque Model Identification**: Perform nearest-neighbor ($k$-NN) matching to classify unlabelled API endpoints against golden reference fingerprints.
- ⚡ **Concurrent High-Throughput Engine**: Parallelized sampling (`ThreadPoolExecutor`) with persistent connection pool reuse (`httpx.Client`).
- 🔌 **Universal Adapter Support**: Plug-and-play support for OpenAI-compatible HTTP APIs, generic headless CLI subprocesses, and offline synthetic test mocks.
- 🌐 **Multilingual Canonicalization**: Pre-built normalizers handling Unicode NFC normalization, Arabic-Indic digits, Chinese numerals (一-九十九), binary coin mappings, and canonical color lexicons.

---

## 📊 System Architecture & Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PROMPT BATTERY (v1)                             │
│     Tasks: [num100-random, num10-random, color-random, coin-flip, ...]     │
│     Languages: [English (en), Russian (ru), Chinese (zh), Arabic (ar)]      │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONCURRENT PROBE ENGINE (Collector)                    │
│     - ThreadPoolExecutor parallel dispatch                                  │
│     - Persistent HTTP connection pooling / Subprocess executor              │
│     - Resume cache (cache_*.json)                                           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CANONICAL NORMALIZER SYSTEM                         │
│     - Text cleaning (NFC, casefold, punctuation removal)                    │
│     - Digit translation (Arabic-Indic ٧ -> 7, Chinese 四十二 -> 42)           │
│     - Validity Taxonomy: [valid, invalid, refusal, empty]                 │
│     - Cross-lingual Color Lexicon mapping                                   │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      JENSEN-SHANNON DIVERGENCE (JSD)                        │
│     - Calculates bounded JSD metric: D_JS(P, Q) in [0, 1]                   │
│     - Minimum valid sample threshold filtering (n_valid >= min_n)           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION & REPORT GENERATION                        │
│     - Verdict: PASSED / FAILED verification against claim                   │
│     - Nearest-Neighbor (k-NN) Reference Library match                       │
│     - Router Mixture Share Estimation (%)                                   │
│     - Markdown & HTML Audit Reports                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

Install via `uv` (recommended) or `pip`:

```bash
# Clone repository
git clone https://github.com/mallochio/llm-fingerprint.git
cd llm-fingerprint

# Create virtual environment and install in editable mode with dev dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify installation:

```bash
fingerprint --help
```

---

## ⚡ 1-Minute Quickstart

### 1. Probe an Adapter Once
Inspect raw responses and canonical normalized tokens:

```bash
fingerprint probe once --adapter mock --prompt "Name a random color."
```

*Example Output:*
```text
┌───────────────────┬──────────────┐
│ Field             │ Value        │
├───────────────────┼──────────────┤
│ Raw Text          │ 'blue'       │
│ Normalized Token  │ 'blue'       │
│ Answer Class      │ valid        │
│ Valid             │ True         │
│ Latency (ms)      │ 5.2          │
└───────────────────┴──────────────┘
```

### 2. Build Reference Fingerprint Library
Generate reference fingerprints for target model APIs:

```bash
export OPENAI_API_KEY=sk-...
fingerprint ref build --adapter openai --model gpt-4o --out refs/api-v1
```

### 3. Fast Quickcheck
Run a fast 5-session sanity check against a reference library:

```bash
fingerprint quickcheck --adapter cursor_auto --sessions 5 --lib refs/api-v1
```

### 4. Full Router Audit & HTML Report
Audit dynamic router mixture proportions across 10 independent sessions:

```bash
fingerprint audit --adapter cursor_auto --sessions 10 --n-per-cell 15 \
  --battery batteries/v1 --lib refs/api-v1 --report html --out runs/cursor-auto-audit
```

---

## 🐍 Python SDK API Usage

You can use `llm-fingerprint` as a standalone Python library in your research or auditing tools:

```python
from pathlib import Path
from fingerprint.adapters import OpenAICompatAdapter, MockAdapter
from fingerprint.collect import collect
from fingerprint.distance import distance
from fingerprint.verify import verify, identify, mixture_report

# 1. Initialize Adapters
adapter_a = MockAdapter(target_profile="gpt-4o")
adapter_b = MockAdapter(target_profile="claude-3-5-sonnet")

# 2. Collect Empirical Fingerprints concurrently
fp_a = collect(adapter_a, battery_path="batteries/v1", n_per_cell=20, max_workers=8)
fp_b = collect(adapter_b, battery_path="batteries/v1", n_per_cell=20, max_workers=8)

# 3. Compute Jensen-Shannon Divergence
dist_result = distance(fp_a, fp_b, min_n=10)
print(f"JSD Distance between GPT-4o and Claude 3.5 Sonnet: {dist_result.distance:.4f}")

# 4. Verify Model Claim
verify_result = verify(fp_a, fp_b, tau=0.05)
print(f"Verification Verdict: {'PASSED' if verify_result.verified else 'FAILED'}")

# 5. Router Mixture Audit over sessions
mix_result = mixture_report(sessions=[fp_a, fp_b], library=[fp_a, fp_b])
print(f"Estimated Mixture Shares: {mix_result.estimated_mixture}")
```

---

## 🛠️ CLI Command Reference

| Command | Subcommand | Description |
| :--- | :--- | :--- |
| `fingerprint` | `probe once` | Probes an endpoint once with a prompt and prints raw + normalized output. |
| `fingerprint` | `ref build` | Builds golden reference fingerprints for known model endpoints. |
| `fingerprint` | `quickcheck` | Fast sanity check of an endpoint/router against a reference library. |
| `fingerprint` | `audit` | Runs a multi-session audit, generating JSD distance metrics, verification, & mixture reports. |
| `fingerprint` | `report` | Renders an HTML or Markdown report from a saved `fingerprint.json` file. |

---

## ⚙️ Configuration File (`fingerprint.example.yaml`)

Define reusable recipe configurations for custom CLI agents and OpenAI-compatible API providers:

```yaml
adapters:
  cursor_auto:
    type: cli
    command: ["agent", "-p", "--output-format", "text", "--mode", "ask"]
    prompt_mode: argv          # argv | stdin | file
    prompt_template: |
      Reply with exactly one word. No punctuation. No markdown. No preamble.
      Do not use tools, shell, or read files. Do not explain.
      Question: {prompt}
    cwd: /tmp/fingerprint-empty
    timeout_s: 120
    environment: cursor-cli

  openrouter_gpt4o:
    type: openai
    environment: openai-api
    model: openai/gpt-4o
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    timeout_s: 30
```

---

## 🛡️ Threat Model & Safety Guidelines

- **Environment Rule**: **API fingerprints $\neq$ Agent CLI fingerprints.** Every fingerprint is tagged with an explicit `environment` tag (`openai-api`, `cursor-cli`, `devin-cli`, etc.). Comparing fingerprints across different environments is flagged unless forced.
- **Agent Isolation**: When fingerprinting headless coding agents (e.g. Cursor, Devin), **always run probes inside an isolated empty directory** (e.g. `/tmp/fingerprint-empty`). Never execute unattended CLI agent probes in sensitive codebase repositories.

---

## 📖 Scientific Citation & References

This implementation is based on the research presented in:

> **Tomáš Bruckner**  
> *One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions*  
> arXiv:2607.10252, July 2026.

If you use `llm-fingerprint` in your research or auditing projects, please cite:

```bibtex
@article{bruckner2026onetoken,
  title={One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output Distributions},
  author={Bruckner, Tom{\'{a}}{\v{s}}},
  journal={arXiv preprint arXiv:2607.10252},
  year={2026}
}
```

### Artifacts & Datasets
- 📜 **arXiv Paper**: [arXiv:2607.10252](https://arxiv.org/abs/2607.10252)
- 💾 **Software Artifact**: [Zenodo DOI: 10.5281/zenodo.21278793](https://doi.org/10.5281/zenodo.21278793)
- 📊 **Dataset Artifact**: [Zenodo DOI: 10.5281/zenodo.21278557](https://doi.org/10.5281/zenodo.21278557)

---

## 📄 License

Distributed under the [MIT License](LICENSE).
