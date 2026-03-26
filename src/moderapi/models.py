"""Pydantic models matching the Perspective API request/response format.

Perspective API format reference:
    POST /v1alpha1/comments:analyze
    6 attributes: TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT
    Each attribute returns a summaryScore with value 0.0-1.0.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Attribute(str, Enum):
    """The 6 Perspective API toxicity attributes."""

    TOXICITY = "TOXICITY"
    SEVERE_TOXICITY = "SEVERE_TOXICITY"
    IDENTITY_ATTACK = "IDENTITY_ATTACK"
    INSULT = "INSULT"
    PROFANITY = "PROFANITY"
    THREAT = "THREAT"


# --- Perspective API Request Format ---


class TextEntry(BaseModel):
    text: str


class RequestedAttribute(BaseModel):
    scoreType: str = "PROBABILITY"
    scoreThreshold: float = 0.0


class AnalyzeRequest(BaseModel):
    """Perspective API /v1alpha1/comments:analyze request body."""

    comment: TextEntry
    requestedAttributes: dict[str, RequestedAttribute] = Field(default_factory=dict)
    languages: list[str] = Field(default_factory=list)
    doNotStore: bool = True


# --- Perspective API Response Format ---


class SummaryScore(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    type: str = "PROBABILITY"


class AttributeScore(BaseModel):
    summaryScore: SummaryScore
    spanScores: list[Any] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Perspective API /v1alpha1/comments:analyze response body."""

    attributeScores: dict[str, AttributeScore]
    languages: list[str] = Field(default_factory=list)
    detectedLanguages: list[str] = Field(default_factory=list)


# --- Replay/Comparison Models ---


class ReplayRecord(BaseModel):
    """A single text with both Perspective and Detoxify scores."""

    text: str
    perspective_scores: dict[str, float]  # attribute -> score 0.0-1.0
    detoxify_scores: dict[str, float]  # attribute -> raw score 0.0-1.0
    calibrated_scores: dict[str, float] = Field(default_factory=dict)


class AttributeGateResult(BaseModel):
    """Gate result for a single attribute."""

    attribute: str
    viable: bool  # pre-calibration semantic viability (raw Spearman >= 0.5)
    spearman_raw: float  # before calibration
    spearman_calibrated: float  # after calibration
    spearman_ci_lower: float = 0.0  # bootstrap 95% CI
    spearman_ci_upper: float = 0.0
    threshold_agreement: float  # at T=0.8, percentage agreement
    threshold_agreement_ci_lower: float = 0.0
    threshold_agreement_ci_upper: float = 0.0
    calibration_method: str = "none"  # "ols", "isotonic", or "none"
    gate_passed: bool = False


class GateResult(BaseModel):
    """Overall hard gate result across all attributes."""

    attributes: list[AttributeGateResult]
    passed_count: int = 0
    total_count: int = 6
    overall_viable: bool = False  # True if enough attributes pass


class CalibrationCoefficients(BaseModel):
    """Per-attribute calibration coefficients."""

    slope: float = 1.0
    intercept: float = 0.0
    r_squared: float = 0.0
    method: str = "ols"  # "ols" or "isotonic"


class CalibrationConfig(BaseModel):
    """Full calibration.json schema."""

    version: int = 1
    attributes: dict[str, CalibrationCoefficients] = Field(default_factory=dict)
    dataset_size: int = 0
    generated_at: str = ""
