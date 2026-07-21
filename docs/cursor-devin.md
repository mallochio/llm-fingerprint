# Cursor & Devin Headless Fingerprinting Guide

This document details how to configure `llm-fingerprint` to run black-box fingerprinting and router mixture auditing against headless instances of **Cursor CLI** (`agent`) and **Devin CLI** (`devin`).

---

## 1. Cursor CLI Headless Pattern

### Overview
Cursor provides a headless command-line interface (`agent`). When running in Auto routing mode, Cursor dynamically selects underlying models without disclosing the specific model choice per turn. `llm-fingerprint` treats Cursor Auto as a **mixture model over independent sessions**.

### Flag & Invocation Pattern
```bash
agent -p --output-format text --mode ask --workspace /tmp/fingerprint-empty "<wrapped_prompt>"
```

- `-p`: Headless print/non-interactive prompt mode.
- `--output-format text`: Output plain text to stdout.
- `--mode ask`: Ask mode without file editing side-effects.
- `--workspace`: Points to an empty, isolated repository.

### Fingerprint Adapter Configuration
```yaml
adapters:
  cursor_auto:
    type: cli
    command: ["agent", "-p", "--output-format", "text", "--mode", "ask"]
    prompt_mode: argv
    prompt_template: |
      Reply with exactly one word. No punctuation. No markdown. No preamble.
      Do not use tools, shell, or read files. Do not explain.
      Question: {prompt}
    cwd: /tmp/fingerprint-empty
    timeout_s: 120
    environment: cursor-cli
    env:
      CURSOR_API_KEY: ${CURSOR_API_KEY}
```

---

## 2. Devin CLI Headless Pattern

### Overview
Devin provides a CLI interface (`devin`) for non-interactive task invocation. For single-token fingerprinting, probes are wrapped with `/ask` to prevent Devin from attempting tool calls or multi-file edits.

### Flag & Invocation Pattern
```bash
devin -p -- "/ask Reply with exactly one word. No tools. No code. No preamble. <prompt>"
```

- `-p`: Non-interactive headless execution mode.
- `--`: Delimits command-line options from prompt input.
- `/ask`: Forces Devin into Q&A mode.

### Fingerprint Adapter Configuration
```yaml
adapters:
  devin_auto:
    type: cli
    command: ["devin", "-p", "--"]
    prompt_mode: argv
    prompt_template: |
      /ask Reply with exactly one word. No tools. No code. No preamble.
      {prompt}
    cwd: /tmp/fingerprint-empty
    timeout_s: 180
    environment: devin-cli
    env:
      DEVIN_API_KEY: ${DEVIN_API_KEY}
```

---

## 3. Best Practices & Safety

1. **Empty Workspace**: Always point `cwd` to an empty directory (e.g. `/tmp/fingerprint-empty` with `git init`).
2. **Session Isolation**: Each probe or audit session should spawn a fresh subprocess to prevent conversational context leakage.
3. **Environment Tagging**: Fingerprints collected from `cursor-cli` or `devin-cli` are tagged accordingly and must only be compared against reference fingerprints from compatible environments or explicitly marked baselines.
