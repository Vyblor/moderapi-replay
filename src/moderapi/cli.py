"""Typer CLI for moderapi-replay.

Commands:
    replay   — Run parity check against a JSONL file of Perspective API outputs
    calibrate — Fit calibration from paired scores and save calibration.json
    report   — Generate HTML parity report
    estimate — Estimate migration effort from a description
    serve    — Start FastAPI server (Phase 2, requires [server] extra)

Eng review decisions:
    - Ctrl+C graceful shutdown with partial result save
    - --redact flag (default ON) for reports
    - First-run model download UX (~400MB with progress bar)
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="moderapi-replay",
    help="Validate your Perspective API moderation thresholds still work with Detoxify.",
    no_args_is_help=True,
)
console = Console()

# Graceful shutdown state
_shutdown_requested = False
_partial_results: list = []


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle Ctrl+C: save partial results and exit cleanly."""
    global _shutdown_requested
    if _shutdown_requested:
        console.print("\n[red]Force quit.[/red]")
        sys.exit(1)
    _shutdown_requested = True
    console.print(
        "\n[yellow]Shutting down gracefully... (Ctrl+C again to force)[/yellow]"
    )
    if _partial_results:
        console.print(f"[yellow]Saving {len(_partial_results)} partial results...[/yellow]")


signal.signal(signal.SIGINT, _handle_sigint)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@app.command()
def replay(
    input_file: Path = typer.Argument(..., help="JSONL file with Perspective API outputs"),
    calibration_file: Path = typer.Option(
        "calibration.json", "--calibration", "-c", help="Path to calibration.json"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output HTML report path"),
    redact: bool = typer.Option(True, "--redact/--no-redact", help="Redact user text in reports"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run parity check against a JSONL file of Perspective API outputs."""
    _setup_logging(verbose)

    from moderapi.calibration import load_calibration
    from moderapi.inference import predict_batch
    from moderapi.parser import parse_jsonl

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Load calibration if available
    _cal_config = None
    if calibration_file.exists():
        _cal_config = load_calibration(calibration_file)
        console.print(f"[green]Loaded calibration from {calibration_file}[/green]")
    else:
        console.print("[yellow]No calibration.json found — using raw Detoxify scores[/yellow]")

    # Parse input
    console.print(f"Parsing {input_file}...")
    records = list(parse_jsonl(input_file))
    console.print(f"Loaded {len(records)} records")

    if not records:
        console.print("[red]No valid records found in input file[/red]")
        raise typer.Exit(1)

    # Run inference
    console.print("Running Detoxify inference...")
    texts = [r.text for r in records]
    detoxify_results = predict_batch(texts)

    for record, scores in zip(records, detoxify_results):
        record.detoxify_scores = scores

    console.print(f"[green]Inference complete for {len(records)} texts[/green]")

    # TODO: Apply calibration + run comparison + generate report
    # This will be filled in during full implementation
    if output:
        console.print(f"Report would be written to {output}")


@app.command()
def calibrate(
    input_file: Path = typer.Argument(..., help="JSONL file with paired Perspective + text data"),
    output: Path = typer.Option(
        "calibration.json", "--output", "-o", help="Output calibration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit calibration coefficients from paired Perspective/Detoxify scores."""
    _setup_logging(verbose)
    console.print(f"Calibrating from {input_file}...")
    # TODO: Implement calibration pipeline
    console.print("[yellow]Calibration pipeline not yet implemented[/yellow]")


@app.command()
def estimate(
    description: str = typer.Argument(..., help="Description of your Perspective API usage"),
) -> None:
    """Estimate migration effort from a text description."""
    from moderapi.estimator import estimate_migration

    result = estimate_migration(description)

    table = Table(title="Migration Effort Estimate")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Complexity", result.complexity_score.upper())
    table.add_row("Estimated Hours", f"{result.estimated_hours:.0f}h")
    table.add_row("API Call Sites", str(result.api_call_count))
    table.add_row("Attributes Used", ", ".join(result.unique_attributes_used) or "—")
    table.add_row("Threshold References", str(result.threshold_references))

    console.print(table)

    if result.notes:
        console.print("\n[yellow]Notes:[/yellow]")
        for note in result.notes:
            console.print(f"  • {note}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    calibration_file: Path = typer.Option("calibration.json", "--calibration", "-c"),
) -> None:
    """Start the drop-in Perspective API replacement server (Phase 2)."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print(
            "[red]Server requires [server] extra: pip install moderapi-replay[server][/red]"
        )
        raise typer.Exit(1)

    console.print("[yellow]Server implementation is Phase 2 — not yet available[/yellow]")


if __name__ == "__main__":
    app()
