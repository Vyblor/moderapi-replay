"""Tests for Pydantic models."""

from moderapi.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Attribute,
    AttributeScore,
    CalibrationCoefficients,
    CalibrationConfig,
    GateResult,
    SummaryScore,
    TextEntry,
)


def test_attribute_enum():
    assert len(Attribute) == 6
    assert Attribute.TOXICITY.value == "TOXICITY"


def test_analyze_request():
    req = AnalyzeRequest(
        comment=TextEntry(text="hello world"),
        requestedAttributes={"TOXICITY": {"scoreType": "PROBABILITY"}},
    )
    assert req.comment.text == "hello world"


def test_analyze_response():
    resp = AnalyzeResponse(
        attributeScores={"TOXICITY": AttributeScore(summaryScore=SummaryScore(value=0.85))}
    )
    assert resp.attributeScores["TOXICITY"].summaryScore.value == 0.85


def test_calibration_config():
    config = CalibrationConfig(
        version=1,
        attributes={
            "TOXICITY": CalibrationCoefficients(slope=1.02, intercept=-0.03, r_squared=0.91),
        },
        dataset_size=1000,
        generated_at="2026-03-26T00:00:00Z",
    )
    assert config.attributes["TOXICITY"].slope == 1.02


def test_gate_result_per_attribute():
    result = GateResult(attributes=[], passed_count=4, total_count=6, overall_viable=True)
    assert result.overall_viable
    assert result.passed_count == 4
