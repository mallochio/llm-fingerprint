"""Command-Line Interface for LLM Single-Token Fingerprinting Toolkit."""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from fingerprint.types import Fingerprint
from fingerprint.adapters import load_adapter
from fingerprint.collect import collect
from fingerprint.normalize import normalize_answer
from fingerprint.distance import distance
from fingerprint.verify import verify, identify, mixture_report
from fingerprint.store import save_fingerprint, load_fingerprint, load_library
from fingerprint.report import generate_markdown_report, generate_html_report

app = typer.Typer(
    name="fingerprint",
    help="LLM Single-Token Fingerprinting Toolkit (OSS)",
    no_args_is_help=True,
)
probe_app = typer.Typer(help="Probe endpoints or single prompts")
ref_app = typer.Typer(help="Manage reference fingerprint libraries")

app.add_typer(probe_app, name="probe")
app.add_typer(ref_app, name="ref")

console = Console()


@probe_app.command("once")
def probe_once(
    adapter: str = typer.Option("mock", "--adapter", "-a", help="Adapter ID or name"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file path"),
    prompt: str = typer.Option("Name a random color.", "--prompt", "-p", help="Prompt string"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature"),
):
    """Probes an adapter once with a prompt and displays raw + normalized output."""
    model_adapter = load_adapter(adapter, config_path=str(config) if config else None)
    console.print(f"[bold blue]Probing adapter:[/bold blue] {model_adapter.name} (env: {model_adapter.environment})")

    res = model_adapter.complete(prompt, temperature=temperature)
    norm = normalize_answer(res["raw_text"])

    table = Table(title="Probe Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Raw Text", repr(res["raw_text"]))
    table.add_row("Normalized Token", repr(norm.normalized))
    table.add_row("Answer Class", norm.answer_class)
    table.add_row("Valid", str(res["valid"]))
    table.add_row("Latency (ms)", f"{res['latency_ms']:.1f}")
    table.add_row("Exit Code", str(res["exit_code"]))

    console.print(table)


@ref_app.command("build")
def ref_build(
    adapter: str = typer.Option("openai", "--adapter", "-a", help="Adapter type (e.g. openai)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file path"),
    models_file: Optional[Path] = typer.Option(None, "--models-file", "-m", help="File listing model IDs"),
    model: Optional[str] = typer.Option(None, "--model", help="Single model ID"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API Key"),
    battery: Path = typer.Option("batteries/v1", "--battery", "-b", help="Path to battery directory or JSON"),
    n_per_cell: int = typer.Option(20, "--n-per-cell", "-n", help="Number of samples per task cell"),
    out: Path = typer.Option("refs/api-v1", "--out", "-o", help="Output directory for reference fingerprints"),
):
    """Builds reference fingerprint library for known models."""
    models_to_build: list[str] = []
    if model:
        models_to_build.append(model)
    if models_file and models_file.exists():
        with open(models_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models_to_build.append(line)

    if not models_to_build:
        models_to_build = ["gpt-4o", "claude-3-5-sonnet"]
        console.print("[yellow]No models specified, defaulting to mock reference build for demo models.[/yellow]")

    out.mkdir(parents=True, exist_ok=True)

    for m_id in models_to_build:
        console.print(f"[bold green]Building reference fingerprint for model:[/bold green] {m_id}")
        overrides = {}
        if m_id:
            overrides["model"] = m_id
            overrides["target_profile"] = m_id
        if base_url:
            overrides["base_url"] = base_url
        if api_key:
            overrides["api_key"] = api_key

        model_adapter = load_adapter(adapter, config_path=str(config) if config else None, overrides=overrides)

        fp = collect(
            adapter=model_adapter,
            battery_path=battery,
            n_per_cell=n_per_cell,
            claimed_model=m_id,
        )

        safe_filename = m_id.replace("/", "_").replace(":", "_") + ".json"
        save_path = out / safe_filename
        save_fingerprint(fp, save_path)
        console.print(f"[bold green]Saved reference fingerprint to:[/bold green] {save_path}")


@app.command("quickcheck")
def quickcheck(
    adapter: str = typer.Option("cursor_auto", "--adapter", "-a", help="Adapter ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file path"),
    sessions: int = typer.Option(5, "--sessions", "-s", help="Number of test sessions"),
    battery: Path = typer.Option("batteries/v1", "--battery", "-b", help="Battery path"),
    lib: Path = typer.Option("refs/api-v1", "--lib", "-l", help="Reference library directory"),
    min_n: int = typer.Option(1, "--min-n", help="Minimum samples per cell"),
):
    """Fast sanity check of an unknown endpoint/router against a reference library."""
    console.print(f"[bold blue]Running Quickcheck across {sessions} session(s)...[/bold blue]")

    library = load_library(lib)
    if not library:
        console.print(f"[bold red]Error:[/bold red] No reference fingerprints found in {lib}")
        raise typer.Exit(code=1)

    collected_sessions: list[Fingerprint] = []

    for s_idx in range(sessions):
        model_adapter = load_adapter(adapter, config_path=str(config) if config else None)
        fp = collect(model_adapter, battery_path=battery, n_per_cell=5)
        collected_sessions.append(fp)

    mix_res = mixture_report(collected_sessions, library, min_n=min_n)

    console.print("\n[bold green]Quickcheck Summary:[/bold green]")
    console.print(f"Sessions Audited: {mix_res.num_sessions}")
    console.print(f"Pairwise Session Mean JSD: {mix_res.session_pairwise_mean_jsd:.4f}")

    table = Table(title="Estimated Model Mixture")
    table.add_column("Reference Model", style="cyan")
    table.add_column("Estimated Share", style="magenta")
    for model, share in mix_res.estimated_mixture.items():
        table.add_row(model, f"{share * 100.0:.1f}%")
    console.print(table)


@app.command("audit")
def audit(
    adapter: str = typer.Option("cursor_auto", "--adapter", "-a", help="Adapter ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file path"),
    sessions: int = typer.Option(10, "--sessions", "-s", help="Number of independent sessions"),
    n_per_cell: int = typer.Option(15, "--n-per-cell", "-n", help="Samples per cell per session"),
    battery: Path = typer.Option("batteries/v1", "--battery", "-b", help="Battery path"),
    lib: Path = typer.Option("refs/api-v1", "--lib", "-l", help="Reference library directory"),
    claim: Optional[str] = typer.Option(None, "--claim", help="Claimed model ID to verify"),
    tau: float = typer.Option(0.05, "--tau", help="Verification threshold JSD"),
    report_format: str = typer.Option("html", "--report", "-r", help="Report format (html|md|json)"),
    out: Path = typer.Option("runs/latest", "--out", "-o", help="Output directory or file path"),
):
    """Runs a complete fingerprint audit and generates verification/mixture reports."""
    console.print(f"[bold blue]Starting Audit: {sessions} session(s), {n_per_cell} samples/cell[/bold blue]")

    library = load_library(lib)
    if not library:
        console.print(f"[bold red]Error:[/bold red] No reference fingerprints found in {lib}")
        raise typer.Exit(code=1)

    collected_sessions: list[Fingerprint] = []

    for s_idx in range(sessions):
        console.print(f"Collecting session {s_idx + 1}/{sessions}...")
        model_adapter = load_adapter(adapter, config_path=str(config) if config else None)
        fp = collect(model_adapter, battery_path=battery, n_per_cell=n_per_cell, claimed_model=claim)
        collected_sessions.append(fp)

    primary_fp = collected_sessions[0]
    mix_res = mixture_report(collected_sessions, library)

    # Verification against claim if provided
    verify_res = None
    if claim:
        ref_claim_fp = next((f for f in library if (f.claimed_model or f.adapter) == claim), None)
        if ref_claim_fp:
            verify_res = verify(primary_fp, ref_claim_fp, tau=tau)

    neighbors = identify(primary_fp, library, k=5)

    # Export report
    out.mkdir(parents=True, exist_ok=True)
    save_fingerprint(primary_fp, out / "fingerprint.json")

    md_report = generate_markdown_report(primary_fp, verify_res, neighbors, mix_res)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    html_report = generate_html_report(primary_fp, verify_res, neighbors, mix_res)
    with open(out / "report.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    console.print(f"\n[bold green]Audit complete! Results saved to:[/bold green] {out}")
    console.print(f"Report HTML: {out / 'report.html'}")


@app.command("report")
def report_cmd(
    run_dir: Path = typer.Argument(..., help="Path to run directory or fingerprint.json"),
    out_format: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown|html)"),
):
    """Renders a report from a saved fingerprint JSON file or run directory."""
    if run_dir.is_dir():
        fp_path = run_dir / "fingerprint.json"
    else:
        fp_path = run_dir

    fp = load_fingerprint(fp_path)
    md_rep = generate_markdown_report(fp)

    if out_format.lower() == "html":
        console.print(generate_html_report(fp))
    else:
        console.print(md_rep)


def main():
    app()


if __name__ == "__main__":
    main()
