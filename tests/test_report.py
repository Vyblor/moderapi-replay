"""Tests for HTML report generation."""

from pathlib import Path

from moderapi.models import AttributeGateResult, GateResult
from moderapi.report import generate_html_report


def test_generate_html_report():
    gate = GateResult(
        attributes=[
            AttributeGateResult(
                attribute="TOXICITY",
                viable=True,
                spearman_raw=0.9,
                spearman_calibrated=0.92,
                spearman_ci_lower=0.88,
                spearman_ci_upper=0.95,
                threshold_agreement=0.94,
                threshold_agreement_ci_lower=0.90,
                threshold_agreement_ci_upper=0.97,
                calibration_method="ols",
                gate_passed=True,
            ),
        ],
        passed_count=1,
        total_count=1,
        overall_viable=True,
    )
    html = generate_html_report(gate)
    assert "ModerAPI Replay" in html
    assert "TOXICITY" in html
    assert "PASS" in html


def test_report_with_redaction():
    gate = GateResult(attributes=[], passed_count=0, total_count=0, overall_viable=False)
    html = generate_html_report(gate, redact=True)
    assert "redacted" in html.lower()


def test_report_write_to_file(tmp_path: Path):
    gate = GateResult(attributes=[], passed_count=0, total_count=0, overall_viable=False)
    output = tmp_path / "report.html"
    generate_html_report(gate, output_path=output)
    assert output.exists()
    assert "ModerAPI" in output.read_text()
