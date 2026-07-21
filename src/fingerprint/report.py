"""Report generation utilities (Markdown, HTML, JSON)."""

import json
from typing import Any
from fingerprint.types import (
    Fingerprint,
    VerifyResult,
    Neighbor,
    MixtureResult,
)


def generate_markdown_report(
    fp: Fingerprint,
    verify_res: VerifyResult | None = None,
    neighbors: list[Neighbor] | None = None,
    mixture_res: MixtureResult | None = None,
) -> str:
    """Generates a Markdown report from fingerprint evaluation results."""
    lines = []
    lines.append("# LLM Fingerprint Audit Report\n")
    lines.append(f"- **Timestamp**: `{fp.created_at}`")
    lines.append(f"- **Adapter**: `{fp.adapter}`")
    lines.append(f"- **Environment**: `{fp.environment}`")
    lines.append(f"- **Claimed Model**: `{fp.claimed_model or 'None specified'}`")
    lines.append(f"- **Temperature**: `{fp.temperature}`")
    lines.append(f"- **Battery ID / Hash**: `{fp.battery_id}` (`{fp.battery_hash}`)\n")

    # Cell statistics summary
    total_cells = len(fp.cells)
    total_valid = sum(c.n_valid for c in fp.cells.values())
    total_invalid = sum(c.n_invalid for c in fp.cells.values())
    total_refusal = sum(c.n_refusal for c in fp.cells.values())
    total_empty = sum(c.n_empty for c in fp.cells.values())
    total_samples = total_valid + total_invalid + total_refusal + total_empty

    valid_pct = (total_valid / total_samples * 100.0) if total_samples > 0 else 0.0

    lines.append("## Collection Summary")
    lines.append(f"- **Total Cells**: {total_cells}")
    lines.append(f"- **Total Probes**: {total_samples}")
    lines.append(f"- **Valid Answers**: {total_valid} ({valid_pct:.1f}%)")
    lines.append(f"- **Invalid / Off-format**: {total_invalid}")
    lines.append(f"- **Refusals**: {total_refusal}")
    lines.append(f"- **Empty Responses**: {total_empty}\n")

    if verify_res:
        lines.append("## Verification Result")
        status = "PASSED" if verify_res.verified else "FAILED"
        lines.append(f"**Verdict**: `{status}`")
        lines.append(f"- **Claimed Model**: `{verify_res.claimed_model}`")
        lines.append(f"- **Observed Distance (JSD)**: `{verify_res.distance:.4f}`")
        lines.append(f"- **Threshold (τ)**: `{verify_res.threshold:.4f}`")
        lines.append(f"- **Environment Match**: `{verify_res.environment_match}`")
        lines.append(f"- **Eligible Cells (n >= 10)**: {verify_res.eligible_cells}\n")

    if neighbors:
        lines.append("## Top Nearest Model Neighbors")
        lines.append("| Rank | Reference Model | Environment | JSD Distance | Eligible Cells |")
        lines.append("| :--- | :-------------- | :---------- | :----------- | :------------- |")
        for i, n in enumerate(neighbors, 1):
            lines.append(f"| {i} | `{n.model_id}` | `{n.environment}` | {n.distance:.4f} | {n.eligible_cells} |")
        lines.append("")

    if mixture_res:
        lines.append("## Router / Auto Mixture Audit")
        lines.append(f"- **Sessions Audited**: {mixture_res.num_sessions}")
        lines.append(f"- **Pairwise Session Mean JSD**: `{mixture_res.session_pairwise_mean_jsd:.4f}`\n")
        lines.append("### Estimated Model Mixture Proportions")
        lines.append("| Model Family / Reference | Estimated Share |")
        lines.append("| :----------------------- | :-------------- |")
        for model, share in mixture_res.estimated_mixture.items():
            lines.append(f"| `{model}` | {share * 100.0:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def generate_html_report(
    fp: Fingerprint,
    verify_res: VerifyResult | None = None,
    neighbors: list[Neighbor] | None = None,
    mixture_res: MixtureResult | None = None,
) -> str:
    """Generates an HTML report from fingerprint evaluation results."""
    md_content = generate_markdown_report(fp, verify_res, neighbors, mixture_res)

    # Basic styled HTML wrapper
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Fingerprint Audit Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f8f9fa;
        }}
        .card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }}
        h1, h2, h3 {{ color: #0f172a; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        th {{ background-color: #f1f5f9; font-weight: 600; }}
        code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        .passed {{ color: #16a34a; font-weight: bold; }}
        .failed {{ color: #dc2626; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="card">
        <pre style="white-space: pre-wrap; font-family: inherit;">{md_content}</pre>
    </div>
</body>
</html>"""
    return html_template
