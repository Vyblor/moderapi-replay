"""HTML report generation with ablation comparison.

Eng review decisions:
- Ablation: raw, OLS, isotonic scores side-by-side per attribute
- --redact flag (default ON) strips user text from reports
- Disk space pre-check before writing
"""

from __future__ import annotations

import html
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from moderapi.exceptions import DiskSpaceError, ReportError
from moderapi.models import GateResult

logger = logging.getLogger(__name__)


def _check_disk_space(path: Path, min_bytes: int = 5 * 1024 * 1024) -> None:
    """Ensure at least min_bytes free on the target filesystem."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < min_bytes:
        raise DiskSpaceError(f"Only {free // 1024}KB free on {parent}, need {min_bytes // 1024}KB")


def generate_html_report(
    gate_result: GateResult,
    ablation_data: dict[str, dict[str, float]] | None = None,
    redact: bool = True,
    output_path: Path | None = None,
) -> str:
    """Generate an HTML report summarizing the gate results.

    Args:
        gate_result: Overall gate result with per-attribute details.
        ablation_data: Optional dict of {attribute: {raw, ols, isotonic}} scores.
        redact: If True, strip user text from the report.
        output_path: If provided, write the report to this file.

    Returns:
        HTML string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for attr in gate_result.attributes:
        if attr.gate_passed:
            status = "✅ PASS"
        elif not attr.viable:
            status = "⚠️ INCOMPATIBLE"
        else:
            status = "❌ FAIL"
        ci_sp = (
            f"[{attr.spearman_ci_lower:.3f}, {attr.spearman_ci_upper:.3f}]" if attr.viable else "—"
        )
        ci_ta = (
            f"[{attr.threshold_agreement_ci_lower:.3f}, {attr.threshold_agreement_ci_upper:.3f}]"
            if attr.viable
            else "—"
        )

        ablation_cells = ""
        if ablation_data and attr.attribute in ablation_data:
            ab = ablation_data[attr.attribute]
            ablation_cells = f"""
            <td>{ab.get("raw_spearman", 0):.3f}</td>
            <td>{ab.get("ols_spearman", 0):.3f}</td>
            <td>{ab.get("isotonic_spearman", 0):.3f}</td>"""

        rows.append(f"""
        <tr class="{"pass" if attr.gate_passed else "fail"}">
            <td><strong>{html.escape(attr.attribute)}</strong></td>
            <td>{attr.spearman_raw:.3f}</td>
            <td>{attr.spearman_calibrated:.3f}</td>
            <td>{ci_sp}</td>
            <td>{attr.threshold_agreement:.1%}</td>
            <td>{ci_ta}</td>
            <td>{html.escape(attr.calibration_method)}</td>
            <td>{status}</td>
            {ablation_cells}
        </tr>""")

    ablation_headers = ""
    if ablation_data:
        ablation_headers = """
            <th>Raw Spearman</th>
            <th>OLS Spearman</th>
            <th>Isotonic Spearman</th>"""

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ModerAPI Replay — Parity Report</title>
    <style>
        body {{
            font-family: -apple-system, system-ui, sans-serif;
            max-width: 1200px; margin: 2rem auto;
            padding: 0 1rem; color: #1a1a1a;
        }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr.pass {{ background: #f0fff0; }}
        tr.fail {{ background: #fff0f0; }}
        .summary {{ background: #f8f8f8; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
        .gate-pass {{ color: #2d7d2d; font-weight: bold; }}
        .gate-fail {{ color: #d32f2f; font-weight: bold; }}
        footer {{ margin-top: 2rem; color: #666; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <h1>ModerAPI Replay — Parity Report</h1>
    <p>Generated: {now}</p>

    <div class="summary">
        <p>Gate result: <span class="{"gate-pass" if gate_result.overall_viable else "gate-fail"}">
            {gate_result.passed_count}/{gate_result.total_count} attributes passed
        </span></p>
        <p>Gate criteria: Spearman &ge; 0.85 AND threshold agreement &ge; 90% at T=0.8</p>
    </div>

    <h2>Per-Attribute Results</h2>
    <table>
        <thead>
            <tr>
                <th>Attribute</th>
                <th>Raw Spearman</th>
                <th>Calibrated Spearman</th>
                <th>Spearman 95% CI</th>
                <th>Threshold Agree.</th>
                <th>Agree. 95% CI</th>
                <th>Cal. Method</th>
                <th>Status</th>
                {ablation_headers}
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>

    {"<p><em>User text redacted from this report.</em></p>" if redact else ""}

    <footer>
        <p>ModerAPI Replay v0.1.0 — <a href="https://github.com/Vyblor/moderapi-replay">GitHub</a></p>
    </footer>
</body>
</html>"""

    if output_path:
        _check_disk_space(output_path)
        try:
            output_path.write_text(report_html)
            logger.info("Report written to %s", output_path)
        except OSError as e:
            raise ReportError(f"Cannot write report to {output_path}: {e}") from e

    return report_html
