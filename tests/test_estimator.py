"""Tests for migration estimator."""

from moderapi.estimator import estimate_migration


def test_estimate_simple():
    result = estimate_migration("We use Perspective API for TOXICITY checking")
    assert result.complexity_score in ("low", "medium", "high")
    assert result.estimated_hours > 0


def test_estimate_complex():
    result = estimate_migration(
        "We use Perspective API v1alpha1 comments:analyze for TOXICITY INSULT PROFANITY "
        "THREAT IDENTITY_ATTACK with threshold > 0.8 and threshold > 0.6 and threshold > 0.7 "
        "and threshold for SEVERE_TOXICITY"
    )
    assert result.complexity_score == "high"
    assert len(result.unique_attributes_used) >= 4
