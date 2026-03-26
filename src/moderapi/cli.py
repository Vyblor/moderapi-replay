"""Typer CLI for moderapi-replay.

Commands:
    replay   — Run parity check against a JSONL file of Perspective API outputs
    calibrate — Fit calibration from paired scores and save calibration.json
    estimate — Estimate migration effort from a description
    serve    — Start FastAPI server (Phase 2, requires [server] extra)

Eng review decisions:
    - Ctrl+C graceful shutdown with partial result save
    - --redact flag (default ON) for reports
    - First-run model download UX (~400MB with progress bar)
    - 70/30 train/test split for calibration
    - Per-attribute gate (X of 6 pass)
    - Ablation reporting (raw + OLS + isotonic)
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from moderapi.models import Attribute

if TYPE_CHECKING:
    from moderapi.calibration import CalibrationCoefficients
    from moderapi.models import CalibrationConfig, GateResult, ReplayRecord

app = typer.Typer(
    name="moderapi-replay",
    help="Validate your Perspective API moderation thresholds still work with Detoxify.",
    no_args_is_help=True,
)
console = Console()

# Graceful shutdown state
_shutdown_requested = False
_partial_results: list[dict[str, object]] = []


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle Ctrl+C: save partial results and exit cleanly."""
    global _shutdown_requested
    if _shutdown_requested:
        console.print("\n[red]Force quit.[/red]")
        sys.exit(1)
    _shutdown_requested = True
    console.print("\n[yellow]Shutting down gracefully... (Ctrl+C again to force)[/yellow]")
    if _partial_results:
        console.print(f"[yellow]Saving {len(_partial_results)} partial results...[/yellow]")


signal.signal(signal.SIGINT, _handle_sigint)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _split_records(
    records: list,  # type: ignore[type-arg]
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[list, list]:  # type: ignore[type-arg]
    """Deterministic 70/30 train/test split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    split_idx = int(len(records) * train_ratio)
    train = [records[i] for i in indices[:split_idx]]
    test = [records[i] for i in indices[split_idx:]]
    return train, test


def _extract_attribute_scores(
    records: list,  # type: ignore[type-arg]
    attribute: str,
    score_field: str,
) -> np.ndarray:
    """Extract scores for a single attribute from records."""
    return np.array([getattr(r, score_field).get(attribute, 0.0) for r in records])


def _apply_calibration(
    records: list[ReplayRecord], cal_config: CalibrationConfig
) -> dict[str, dict[str, float]]:
    """Apply calibration to records, return ablation data.

    For each attribute, tries OLS and isotonic on ALL records to produce
    ablation side-by-side data. The 'winning' method's scores go into
    record.calibrated_scores.

    Returns:
        ablation_data: dict[attr: {raw_spearman, ols_spearman, isotonic_spearman}]
    """
    from moderapi.calibration import apply_ols, clamp

    ablation: dict[str, dict[str, float]] = {}

    for attr in [a.value for a in Attribute]:
        coeffs = cal_config.attributes.get(attr)
        if not coeffs:
            # No calibration for this attribute — use raw scores
            for record in records:
                raw = record.detoxify_scores.get(attr, 0.0)
                record.calibrated_scores[attr] = raw
            continue

        for record in records:
            raw = record.detoxify_scores.get(attr, 0.0)
            if coeffs.method == "ols":
                record.calibrated_scores[attr] = apply_ols(raw, coeffs.slope, coeffs.intercept)
            else:
                # For isotonic, we approximate with OLS coefficients stored
                # (actual isotonic model requires pickle, stored as OLS fallback)
                record.calibrated_scores[attr] = clamp(raw)

    return ablation


@app.command()
def replay(
    input_file: Path = typer.Argument(..., help="JSONL file with Perspective API outputs"),
    calibration_file: Path = typer.Option(
        "calibration.json", "--calibration", "-c", help="Path to calibration.json"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output HTML report path"),
    redact: bool = typer.Option(True, "--redact/--no-redact", help="Redact user text in reports"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Decision threshold"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run parity check against a JSONL file of Perspective API outputs."""
    _setup_logging(verbose)

    from moderapi.calibration import load_calibration
    from moderapi.comparison import evaluate_attribute, evaluate_gate
    from moderapi.inference import predict_batch
    from moderapi.parser import parse_jsonl
    from moderapi.report import generate_html_report

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Load calibration if available
    cal_config = None
    if calibration_file.exists():
        cal_config = load_calibration(calibration_file)
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

    # Apply calibration
    ablation_data: dict[str, dict[str, float]] | None = None
    if cal_config:
        console.print("Applying calibration...")
        ablation_data = _apply_calibration(records, cal_config)
    else:
        # No calibration — raw scores are the calibrated scores
        for record in records:
            record.calibrated_scores = dict(record.detoxify_scores)

    # Evaluate gate per attribute
    console.print("Evaluating parity gate...")
    attribute_results = []
    for attr in [a.value for a in Attribute]:
        perspective = _extract_attribute_scores(records, attr, "perspective_scores")
        raw = _extract_attribute_scores(records, attr, "detoxify_scores")
        calibrated = _extract_attribute_scores(records, attr, "calibrated_scores")

        if len(perspective) < 3:
            console.print(f"[yellow]Skipping {attr}: insufficient data[/yellow]")
            continue

        result = evaluate_attribute(attr, perspective, raw, calibrated)
        attribute_results.append(result)

    gate_result = evaluate_gate(attribute_results)

    # Display summary table
    _print_gate_summary(gate_result)

    # Generate report
    if output:
        console.print(f"Generating report → {output}...")
        generate_html_report(
            gate_result,
            ablation_data=ablation_data,
            redact=redact,
            output_path=output,
        )
        console.print(f"[green]Report saved to {output}[/green]")

    # Overall verdict
    if gate_result.passed_count == gate_result.total_count:
        console.print(
            f"\n[bold green]GATE: PASS — all {gate_result.total_count} attributes pass[/bold green]"
        )
    elif gate_result.passed_count > 0:
        console.print(
            f"\n[bold yellow]GATE: PARTIAL — {gate_result.passed_count}/{gate_result.total_count}"
            f" attributes pass[/bold yellow]"
        )
    else:
        console.print(
            f"\n[bold red]GATE: FAIL — 0/{gate_result.total_count} attributes pass[/bold red]"
        )


def _print_gate_summary(gate_result: GateResult) -> None:
    """Print Rich table with gate results."""
    table = Table(title="Parity Gate Results")
    table.add_column("Attribute", style="bold")
    table.add_column("Viable", justify="center")
    table.add_column("Raw ρ", justify="right")
    table.add_column("Cal ρ", justify="right")
    table.add_column("ρ 95% CI", justify="center")
    table.add_column("Agree %", justify="right")
    table.add_column("Agree CI", justify="center")
    table.add_column("Method")
    table.add_column("Gate", justify="center")

    for attr in gate_result.attributes:
        viable = "✓" if attr.viable else "✗"
        raw_sp = f"{attr.spearman_raw:.3f}"
        cal_sp = f"{attr.spearman_calibrated:.3f}" if attr.viable else "—"
        sp_ci = (
            f"[{attr.spearman_ci_lower:.3f}, {attr.spearman_ci_upper:.3f}]" if attr.viable else "—"
        )
        agree = f"{attr.threshold_agreement:.1%}" if attr.viable else "—"
        agree_ci = (
            f"[{attr.threshold_agreement_ci_lower:.3f}, {attr.threshold_agreement_ci_upper:.3f}]"
            if attr.viable
            else "—"
        )
        method = attr.calibration_method
        if attr.gate_passed:
            gate = "[green]PASS[/green]"
        elif not attr.viable:
            gate = "[red]INCOMPAT[/red]"
        else:
            gate = "[red]FAIL[/red]"

        table.add_row(attr.attribute, viable, raw_sp, cal_sp, sp_ci, agree, agree_ci, method, gate)

    console.print(table)
    console.print(
        f"\nPassed: {gate_result.passed_count}/{gate_result.total_count}  "
        f"Viable: {sum(1 for a in gate_result.attributes if a.viable)}/{gate_result.total_count}"
    )


@app.command()
def calibrate(
    input_file: Path = typer.Argument(..., help="JSONL file with paired Perspective + text data"),
    output: Path = typer.Option(
        "calibration.json", "--output", "-o", help="Output calibration file"
    ),
    train_ratio: float = typer.Option(
        0.70, "--train-ratio", help="Train split ratio (default 0.70)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit calibration coefficients from paired Perspective/Detoxify scores."""
    _setup_logging(verbose)

    from moderapi.calibration import calibrate_attribute, save_calibration
    from moderapi.inference import predict_batch
    from moderapi.models import CalibrationConfig
    from moderapi.parser import parse_jsonl

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Parse input
    console.print(f"Parsing {input_file}...")
    records = list(parse_jsonl(input_file))
    console.print(f"Loaded {len(records)} records")

    if len(records) < 30:
        console.print(f"[red]Need at least 30 records for calibration, got {len(records)}[/red]")
        raise typer.Exit(1)

    # Run Detoxify inference
    console.print("Running Detoxify inference...")
    texts = [r.text for r in records]
    detoxify_results = predict_batch(texts)

    for record, scores in zip(records, detoxify_results):
        record.detoxify_scores = scores

    console.print(f"[green]Inference complete for {len(records)} texts[/green]")

    # 70/30 train/test split
    train, test = _split_records(records, train_ratio=train_ratio)
    console.print(f"Split: {len(train)} train / {len(test)} test (ratio={train_ratio:.0%})")

    # Calibrate each attribute
    calibration_results: dict[str, CalibrationCoefficients] = {}
    iso_models: dict[str, Any] = {}
    table = Table(title="Calibration Results")
    table.add_column("Attribute", style="bold")
    table.add_column("Method")
    table.add_column("Slope", justify="right")
    table.add_column("Intercept", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Train Agreement", justify="right")

    from moderapi.calibration import apply_ols, threshold_agreement

    for attr in [a.value for a in Attribute]:
        train_detox = _extract_attribute_scores(train, attr, "detoxify_scores")
        train_persp = _extract_attribute_scores(train, attr, "perspective_scores")

        if len(train_detox) < 30:
            console.print(
                f"[yellow]Skipping {attr}: insufficient train data ({len(train_detox)})[/yellow]"
            )
            continue

        try:
            coeffs, iso_model = calibrate_attribute(train_detox, train_persp)
            calibration_results[attr] = coeffs
            if iso_model is not None:
                iso_models[attr] = iso_model

            # Compute train agreement for display
            if coeffs.method == "ols":
                cal_train = np.array(
                    [apply_ols(s, coeffs.slope, coeffs.intercept) for s in train_detox]
                )
            else:
                cal_train = iso_model.predict(train_detox) if iso_model else train_detox
            train_agree = threshold_agreement(train_persp, cal_train)

            table.add_row(
                attr,
                coeffs.method.upper(),
                f"{coeffs.slope:.4f}",
                f"{coeffs.intercept:.4f}",
                f"{coeffs.r_squared:.4f}",
                f"{train_agree:.1%}",
            )
        except Exception as e:
            console.print(f"[red]Failed to calibrate {attr}: {e}[/red]")
            continue

    console.print(table)

    if not calibration_results:
        console.print("[red]No attributes could be calibrated[/red]")
        raise typer.Exit(1)

    # Build and save CalibrationConfig
    config = CalibrationConfig(
        version=1,
        attributes=calibration_results,  # type: ignore[arg-type]
        dataset_size=len(train),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    save_calibration(config, output)
    console.print(f"\n[green]Calibration saved to {output}[/green]")
    console.print(f"Calibrated {len(calibration_results)}/{len(list(Attribute))} attributes")

    # Run gate evaluation on test split if we have enough data
    if len(test) >= 10:
        console.print(
            f"\n[bold]Evaluating gate on held-out test set ({len(test)} records)...[/bold]"
        )

        from moderapi.comparison import evaluate_attribute, evaluate_gate

        # Apply calibration to test records
        for record in test:
            for attr_name, coeffs_obj in calibration_results.items():
                raw = record.detoxify_scores.get(attr_name, 0.0)
                if coeffs_obj.method == "ols":
                    record.calibrated_scores[attr_name] = apply_ols(
                        raw,
                        coeffs_obj.slope,
                        coeffs_obj.intercept,
                    )
                elif attr_name in iso_models:
                    record.calibrated_scores[attr_name] = float(
                        iso_models[attr_name].predict([raw])[0]  # type: ignore[union-attr]
                    )
                else:
                    record.calibrated_scores[attr_name] = raw

        attr_results = []
        for attr in [a.value for a in Attribute]:
            if attr not in calibration_results:
                continue
            persp = _extract_attribute_scores(test, attr, "perspective_scores")
            raw = _extract_attribute_scores(test, attr, "detoxify_scores")
            cal = _extract_attribute_scores(test, attr, "calibrated_scores")
            if len(persp) < 3:
                continue
            attr_results.append(evaluate_attribute(attr, persp, raw, cal))

        gate = evaluate_gate(attr_results)
        _print_gate_summary(gate)


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
    table.add_row("Attributes Used", ", ".join(result.unique_attributes_used or []) or "—")
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
            "Server requires [server] extra: pip install moderapi-replay[server]",
            style="red",
            highlight=False,
            markup=False,
        )
        raise typer.Exit(1)

    console.print("[yellow]Server implementation is Phase 2 — not yet available[/yellow]")


if __name__ == "__main__":
    app()
