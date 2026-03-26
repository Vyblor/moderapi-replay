"""Migration effort estimator.

Analyzes a codebase or configuration to estimate the effort
of migrating from Perspective API to ModerAPI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MigrationEstimate:
    """Migration complexity assessment."""

    api_call_count: int = 0
    unique_attributes_used: list[str] | None = None
    threshold_references: int = 0
    custom_thresholds: list[float] | None = None
    complexity_score: str = "low"  # low, medium, high
    estimated_hours: float = 0.0
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.unique_attributes_used is None:
            self.unique_attributes_used = []
        if self.custom_thresholds is None:
            self.custom_thresholds = []
        if self.notes is None:
            self.notes = []


def estimate_migration(description: str) -> MigrationEstimate:
    """Produce a rough migration estimate from a text description.

    This is a heuristic estimator — it parses keywords and patterns
    to produce a ballpark. Not a substitute for actual code analysis.
    """
    desc_lower = description.lower()

    # Count API-related keywords
    api_indicators = ["perspectiveapi", "perspective api", "comments:analyze", "v1alpha1"]
    api_count = sum(desc_lower.count(kw) for kw in api_indicators)

    # Detect attributes
    from moderapi.models import Attribute

    attrs_used = [a.value for a in Attribute if a.value.lower() in desc_lower]

    # Detect thresholds
    threshold_refs = desc_lower.count("threshold") + desc_lower.count("score >")
    custom_thresholds: list[float] = []

    # Complexity heuristic
    if api_count <= 2 and len(attrs_used) <= 2:
        complexity = "low"
        hours = 2.0
    elif api_count <= 5 and len(attrs_used) <= 4:
        complexity = "medium"
        hours = 8.0
    else:
        complexity = "high"
        hours = 24.0

    notes = []
    if len(attrs_used) > 4:
        notes.append("Uses many attributes — verify all pass the parity gate")
    if threshold_refs > 3:
        notes.append("Multiple threshold references — calibration accuracy is critical")

    return MigrationEstimate(
        api_call_count=max(1, api_count),
        unique_attributes_used=attrs_used,
        threshold_references=threshold_refs,
        custom_thresholds=custom_thresholds,
        complexity_score=complexity,
        estimated_hours=hours,
        notes=notes,
    )
