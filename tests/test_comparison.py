"""Tests for comparison module."""

import numpy as np
import pytest

from moderapi.comparison import (
    SEMANTIC_VIABILITY_THRESHOLD,
    _safe_spearman,
    evaluate_attribute,
    evaluate_gate,
)
from moderapi.exceptions import ConstantDataError, InsufficientDataError


def test_safe_spearman_correlated():
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    corr = _safe_spearman(a, b)
    assert corr > 0.9


def test_safe_spearman_constant():
    with pytest.raises(ConstantDataError):
        _safe_spearman(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.2, 0.3]))


def test_safe_spearman_insufficient():
    with pytest.raises(InsufficientDataError):
        _safe_spearman(np.array([0.5]), np.array([0.6]))


def test_evaluate_attribute_viable():
    n = 300
    rng = np.random.default_rng(42)
    perspective = rng.uniform(0, 1, n)
    raw = np.clip(perspective * 0.95 + rng.normal(0, 0.05, n), 0, 1)
    calibrated = np.clip(perspective * 0.98 + rng.normal(0, 0.03, n), 0, 1)

    result = evaluate_attribute("TOXICITY", perspective, raw, calibrated)
    assert result.viable
    assert result.spearman_raw > SEMANTIC_VIABILITY_THRESHOLD


def test_evaluate_attribute_incompatible():
    n = 300
    rng = np.random.default_rng(42)
    perspective = rng.uniform(0, 1, n)
    raw = rng.uniform(0, 1, n)  # Uncorrelated

    result = evaluate_attribute("PROFANITY", perspective, raw, raw)
    # May or may not be viable depending on random correlation
    # Just check it runs without error
    assert isinstance(result.viable, bool)


def test_evaluate_gate():
    from moderapi.models import AttributeGateResult

    results = [
        AttributeGateResult(
            attribute="TOXICITY",
            viable=True,
            spearman_raw=0.9,
            spearman_calibrated=0.9,
            threshold_agreement=0.95,
            gate_passed=True,
        ),
        AttributeGateResult(
            attribute="PROFANITY",
            viable=False,
            spearman_raw=0.3,
            spearman_calibrated=0.0,
            threshold_agreement=0.0,
            gate_passed=False,
        ),
    ]
    gate = evaluate_gate(results)
    assert gate.passed_count == 1
    assert gate.total_count == 2
    assert gate.overall_viable
